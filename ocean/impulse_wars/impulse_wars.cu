// Impulse Wars CUDA encoder: map grid CNN, projectile branch, one-hot type
// slots, concat with the continuous obs, projection. The map grid's packed
// cell codes are unpacked into one-hot channels by conv1's im2col and run
// through 3 convs with ReLU. Projectile slots are records for a shared
// Linear -> GELU -> Linear that is max pooled over the non-empty records and
// added to the projection, like the OSRS entity encoder. Wall, weapon and
// ray hit type slots are one-hot encoded into the concat. Included by
// src/ocean.cu.

#include "impulse_wars_obs.h"

static_assert(OBS_SIZE >= IW_BASE_OBS_SIZE + IW_ENEMY_DRONE_SIZE && (OBS_SIZE - IW_BASE_OBS_SIZE) % IW_ENEMY_DRONE_SIZE == 0, "impulse_wars encoder obs layout does not match OBS_SIZE");
static_assert(IW_GRID_CODE_MAX < 256, "impulse_wars map grid codes must be exact in bf16");
static_assert(IW_C3_OUT == 3, "impulse_wars CNN output should keep a 3x3 layout");

// obs index of each continuous scalar, in concat order
struct IwScalarIdx {
    int idx[IW_NUM_SCALARS];
};

static constexpr IwScalarIdx iw_make_scalar_idx() {
    IwScalarIdx t = {};
    int n = 0;
    for (int i = 0; i < OBS_SIZE; i++) {
        if (iw_is_scalar(i)) {
            t.idx[n++] = i;
        }
    }
    return t;
}

static constexpr IwScalarIdx IW_SCALAR_IDX_HOST = iw_make_scalar_idx();
static_assert(IW_SCALAR_IDX_HOST.idx[IW_NUM_SCALARS - 1] == OBS_SIZE - 1, "impulse_wars scalar count does not match IW_NUM_SCALARS");
__constant__ IwScalarIdx IW_SCALAR_IDX = iw_make_scalar_idx();

// conv1 im2col straight from the obs cell codes, skipping the dense one-hot
// planes: (B * IW_C1_SPATIAL, IW_C1_COL_W)
__global__ void iw_c1_im2col_obs(
    precision_t *__restrict__ col, const precision_t *__restrict__ obs, int B, int obs_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_C1_SPATIAL * IW_C1_COL_W) {
        return;
    }
    int row = idx / IW_C1_COL_W;
    int c = idx % IW_C1_COL_W;
    int b = row / IW_C1_SPATIAL;
    int cell = iw_c1_cell(row % IW_C1_SPATIAL, c);
    float v = cell < 0 ? 0.0f : to_float(obs[(int64_t)b * obs_size + cell]);
    col[idx] = from_float(iw_c1_col_value(v, cell, c));
}

// im2col of an unpadded square conv over (B, ic, in, in):
// (B * out * out, ic * k * k)
__global__ void iw_im2col(
    precision_t *__restrict__ col, const precision_t *__restrict__ input, int B, int ic, int in, int out, int k, int stride
) {
    int colW = ic * k * k;
    int spatial = out * out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * spatial * colW) {
        return;
    }
    int row = idx / colW;
    int b = row / spatial;
    col[idx] = input[(int64_t)b * ic * in * in + iw_im2col_src(row % spatial, idx % colW, in, out, k, stride)];
}

// gradient of the (B, ic, in, in) input of an unpadded square conv from the
// gradient of its im2col
__global__ void iw_col2im(
    precision_t *__restrict__ grad_input, const precision_t *__restrict__ col, int B, int ic, int in, int out, int k, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * ic * in * in) {
        return;
    }
    int iw = idx % in;
    int ih = (idx / in) % in;
    int c = (idx / (in * in)) % ic;
    int b = idx / (ic * in * in);
    int colW = ic * k * k;
    float sum = 0.0f;
    for (int kh = 0; kh < k; kh++) {
        int ohs = ih - kh;
        if (ohs < 0 || ohs % stride != 0 || ohs / stride >= out) {
            continue;
        }
        for (int kw = 0; kw < k; kw++) {
            int ows = iw - kw;
            if (ows < 0 || ows % stride != 0 || ows / stride >= out) {
                continue;
            }
            int64_t r = (int64_t)b * out * out + (ohs / stride) * out + ows / stride;
            sum += to_float(col[r * colW + (c * k + kh) * k + kw]);
        }
    }
    grad_input[idx] = from_float(sum);
}

// (B * spatial, oc) matmul rows -> (B, oc, spatial), with optional ReLU
__global__ void iw_rows_to_nchw(
    precision_t *__restrict__ dst, const precision_t *__restrict__ src, int B, int oc, int spatial, int relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * oc * spatial) {
        return;
    }
    int b = idx / (oc * spatial);
    int c = (idx / spatial) % oc;
    int s = idx % spatial;
    float v = to_float(src[((int64_t)b * spatial + s) * oc + c]);
    if (relu) {
        v = fmaxf(v, 0.0f);
    }
    dst[idx] = from_float(v);
}

// (B, oc, spatial) -> (B * spatial, oc) matmul rows
__global__ void iw_nchw_to_rows(
    precision_t *__restrict__ dst, const precision_t *__restrict__ src, int B, int oc, int spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * oc * spatial) {
        return;
    }
    int b = idx / (oc * spatial);
    int c = (idx / spatial) % oc;
    int s = idx % spatial;
    dst[((int64_t)b * spatial + s) * oc + c] = src[idx];
}

// zeroes the gradient where the ReLU output was 0
__global__ void iw_relu_backward(precision_t *__restrict__ grad, const precision_t *__restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (to_float(out[idx]) <= 0.0f) {
        grad[idx] = from_float(0.0f);
    }
}

// copies the CNN block of the concat gradient, masked by the conv3 ReLU
__global__ void iw_conv_grad_from_concat(
    precision_t *__restrict__ conv_grad, const precision_t *__restrict__ grad_concat, const precision_t *__restrict__ conv_out, int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_CONV_FLAT) {
        return;
    }
    int b = idx / IW_CONV_FLAT;
    int c = idx % IW_CONV_FLAT;
    conv_grad[idx] = to_float(conv_out[idx]) > 0.0f ? grad_concat[(int64_t)b * IW_CONCAT + c] : from_float(0.0f);
}

__global__ void iw_concat_kernel(
    precision_t *__restrict__ concat, const precision_t *__restrict__ obs, const precision_t *__restrict__ conv_out, int B, int obs_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_CONCAT) {
        return;
    }
    int c = idx % IW_CONCAT;
    int b = idx / IW_CONCAT;
    const precision_t *o = obs + (int64_t)b * obs_size;
    precision_t v;
    if (c < IW_CONV_FLAT) {
        v = conv_out[(int64_t)b * IW_CONV_FLAT + c];
    } else if (c < IW_CONV_FLAT + IW_ONEHOT_COLS) {
        iwOneHotCol col = iw_onehot_col(c - IW_CONV_FLAT);
        v = from_float(iw_onehot_value(to_float(o[col.obsIdx]), col));
    } else {
        v = o[IW_SCALAR_IDX.idx[c - IW_CONV_FLAT - IW_ONEHOT_COLS]];
    }
    concat[idx] = v;
}

// projectile records from the obs: (B * IW_NUM_PROJECTILES, IW_PROJ_FEATURES),
// and whether each record is a projectile
__global__ void iw_proj_gather(
    precision_t *__restrict__ rec, int *__restrict__ active, const precision_t *__restrict__ obs, int B, int obs_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_NUM_PROJECTILES * IW_PROJ_FEATURES) {
        return;
    }
    int f = idx % IW_PROJ_FEATURES;
    int row = idx / IW_PROJ_FEATURES;
    int b = row / IW_NUM_PROJECTILES;
    const precision_t *slot = obs + (int64_t)b * obs_size + iw_proj_slot_idx(row % IW_NUM_PROJECTILES);
    rec[idx] = from_float(iw_proj_feature(to_float(slot[iw_proj_field(f)]), f));
    if (f == 0) {
        active[row] = iw_proj_active(to_float(slot[1]));
    }
}

__global__ void iw_gelu_kernel(precision_t *__restrict__ out, const precision_t *__restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out[idx] = from_float(iw_gelu(to_float(in[idx])));
}

// multiplies the gradient of GELU's output by GELU's derivative at its input
__global__ void iw_gelu_backward(precision_t *__restrict__ grad, const precision_t *__restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    grad[idx] = from_float(to_float(grad[idx]) * iw_gelu_grad(to_float(in[idx])));
}

// adds the max over the active projectile records of each record output
// (B * IW_NUM_PROJECTILES, H) to out (B, H); a sample with no projectiles
// adds 0. argmax is the pooled record, -1 if none, or null for rollout.
__global__ void iw_proj_pool(
    precision_t *__restrict__ out, int *__restrict__ argmax, const precision_t *__restrict__ rec_out, const int *__restrict__ active, int B, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    int b = idx / H;
    int h = idx % H;
    float best = 0.0f;
    int bestRecord = -1;
    for (int r = 0; r < IW_NUM_PROJECTILES; r++) {
        int row = b * IW_NUM_PROJECTILES + r;
        if (!active[row]) {
            continue;
        }
        float v = to_float(rec_out[(int64_t)row * H + h]);
        if (bestRecord < 0 || v > best) {
            best = v;
            bestRecord = r;
        }
    }
    out[idx] = from_float(to_float(out[idx]) + best);
    if (argmax) {
        argmax[idx] = bestRecord;
    }
}

// routes the encoder output gradient (B, H) to the pooled record outputs:
// (B * IW_NUM_PROJECTILES, H), 0 for records that weren't pooled
__global__ void iw_proj_pool_backward(
    precision_t *__restrict__ grad_rec_out, const precision_t *__restrict__ grad, const int *__restrict__ argmax, int B, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_NUM_PROJECTILES * H) {
        return;
    }
    int h = idx % H;
    int row = idx / H;
    int b = row / IW_NUM_PROJECTILES;
    int r = row % IW_NUM_PROJECTILES;
    grad_rec_out[idx] = argmax[b * H + h] == r ? grad[b * H + h] : from_float(0.0f);
}

struct IwEncoderWeights {
    Prec conv1_w, conv2_w, conv3_w, proj_w, proj_l1_w, proj_l2_w;
    int obs_size, hidden;
};

struct IwEncoderActivations {
    Prec concat, out;
    // per conv: im2col, matmul rows, NCHW ReLU output, output gradient
    Prec col1, mm1, conv1_out, conv1_grad;
    Prec col2, mm2, conv2_out, conv2_grad;
    Prec col3, mm3, conv3_out, conv3_grad;
    // projectile branch: records, first layer pre and post GELU, per record
    // output; backward reuses rec_out and rec_h1 for their gradients
    Prec rec, rec_z1, rec_h1, rec_out;
    Int rec_active, rec_argmax;
    Prec conv1_wgrad, conv2_wgrad, conv3_wgrad, proj_wgrad, proj_l1_wgrad, proj_l2_wgrad;
};

// matmul of a conv's im2col with its weights, then ReLU into NCHW
static void iw_conv_mm(
    precision_t *col, Prec *w, precision_t *mm, precision_t *out, int B, int oc, int spatial, int colW, cudaStream_t stream
) {
    Prec colT = {.data = col, .shape = {B * spatial, colW}};
    Prec mmT = {.data = mm, .shape = {B * spatial, oc}};
    puf_mm(&colT, w, &mmT, stream);
    iw_rows_to_nchw<<<grid_size(B * oc * spatial), BLOCK_SIZE, 0, stream>>>(out, mm, B, oc, spatial, 1);
}

// weight gradient of a conv from its (ReLU masked) NCHW output gradient;
// col still holds the forward im2col and mm is reused as scratch
static void iw_conv_wgrad(
    precision_t *grad_out, precision_t *wgrad, precision_t *col, precision_t *mm, int B, int oc, int spatial, int colW, cudaStream_t stream
) {
    iw_nchw_to_rows<<<grid_size(B * oc * spatial), BLOCK_SIZE, 0, stream>>>(mm, grad_out, B, oc, spatial);
    Prec mmT = {.data = mm, .shape = {B * spatial, oc}};
    Prec colT = {.data = col, .shape = {B * spatial, colW}};
    Prec wgT = {.data = wgrad, .shape = {oc, colW}};
    puf_mm_tn(&mmT, &colT, &wgT, stream);
}

// input gradient of a conv into grad_in (B, ic, in, in), overwriting col;
// call after iw_conv_wgrad, which leaves the output gradient rows in mm
static void iw_conv_input_grad(
    precision_t *grad_in, Prec *w, precision_t *col, precision_t *mm, int B, int ic, int in, int out, int k, int stride, int oc, cudaStream_t stream
) {
    int spatial = out * out;
    int colW = ic * k * k;
    Prec mmT = {.data = mm, .shape = {B * spatial, oc}};
    Prec colT = {.data = col, .shape = {B * spatial, colW}};
    puf_mm_nn(&mmT, w, &colT, stream);
    iw_col2im<<<grid_size(B * ic * in * in), BLOCK_SIZE, 0, stream>>>(grad_in, col, B, ic, in, out, k, stride);
}

// projectile branch forward: adds the pooled records to a->out; argmax is
// null for rollout
static void iw_proj_forward(IwEncoderWeights *ew, IwEncoderActivations *a, Prec input, int *argmax, int B, cudaStream_t stream) {
    int rows = B * IW_NUM_PROJECTILES;
    iw_proj_gather<<<grid_size(rows * IW_PROJ_FEATURES), BLOCK_SIZE, 0, stream>>>(a->rec.data, a->rec_active.data, input.data, B, ew->obs_size);
    puf_mm(&a->rec, &ew->proj_l1_w, &a->rec_z1, stream);
    iw_gelu_kernel<<<grid_size(rows * IW_PROJ_HIDDEN), BLOCK_SIZE, 0, stream>>>(a->rec_h1.data, a->rec_z1.data, rows * IW_PROJ_HIDDEN);
    puf_mm(&a->rec_h1, &ew->proj_l2_w, &a->rec_out, stream);
    iw_proj_pool<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(a->out.data, argmax, a->rec_out.data, a->rec_active.data, B, ew->hidden);
}

static Prec iw_encoder_forward(
    void *w, void *activations, Prec input, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = input.shape[0];

    iw_c1_im2col_obs<<<grid_size(B * IW_C1_SPATIAL * IW_C1_COL_W), BLOCK_SIZE, 0, stream>>>(
        a->col1.data, input.data, B, ew->obs_size
    );
    iw_conv_mm(a->col1.data, &ew->conv1_w, a->mm1.data, a->conv1_out.data, B, IW_C1_OC, IW_C1_SPATIAL, IW_C1_COL_W, stream);
    iw_im2col<<<grid_size(B * IW_C2_SPATIAL * IW_C2_COL_W), BLOCK_SIZE, 0, stream>>>(
        a->col2.data, a->conv1_out.data, B, IW_C2_IC, IW_C2_IN, IW_C2_OUT, IW_C2_K, IW_C2_S
    );
    iw_conv_mm(a->col2.data, &ew->conv2_w, a->mm2.data, a->conv2_out.data, B, IW_C2_OC, IW_C2_SPATIAL, IW_C2_COL_W, stream);
    iw_im2col<<<grid_size(B * IW_C3_SPATIAL * IW_C3_COL_W), BLOCK_SIZE, 0, stream>>>(
        a->col3.data, a->conv2_out.data, B, IW_C3_IC, IW_C3_IN, IW_C3_OUT, IW_C3_K, IW_C3_S
    );
    iw_conv_mm(a->col3.data, &ew->conv3_w, a->mm3.data, a->conv3_out.data, B, IW_C3_OC, IW_C3_SPATIAL, IW_C3_COL_W, stream);

    iw_concat_kernel<<<grid_size(B * IW_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, input.data, a->conv3_out.data, B, ew->obs_size
    );
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);

    iw_proj_forward(ew, a, input, a->rec_argmax.data, B, stream);
    return a->out;
}

static void iw_encoder_backward(
    void *w, void *activations, Prec grad, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = grad.shape[0];

    // projectile branch: rec_out holds the record output gradient and
    // rec_h1, once its weight gradient is done, the first layer's
    iw_proj_pool_backward<<<grid_size(B * IW_NUM_PROJECTILES * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->rec_out.data, grad.data, a->rec_argmax.data, B, ew->hidden
    );
    puf_mm_tn(&a->rec_out, &a->rec_h1, &a->proj_l2_wgrad, stream);
    puf_mm_nn(&a->rec_out, &ew->proj_l2_w, &a->rec_h1, stream);
    iw_gelu_backward<<<grid_size(B * IW_NUM_PROJECTILES * IW_PROJ_HIDDEN), BLOCK_SIZE, 0, stream>>>(
        a->rec_h1.data, a->rec_z1.data, B * IW_NUM_PROJECTILES * IW_PROJ_HIDDEN
    );
    puf_mm_tn(&a->rec_h1, &a->rec, &a->proj_l1_wgrad, stream);

    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);
    Prec grad_concat = {.data = a->concat.data, .shape = {B, IW_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    // CNN, back to front; conv1's input isn't learned so it gets no input grad
    iw_conv_grad_from_concat<<<grid_size(B * IW_CONV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->conv3_grad.data, grad_concat.data, a->conv3_out.data, B
    );
    iw_conv_wgrad(a->conv3_grad.data, a->conv3_wgrad.data, a->col3.data, a->mm3.data, B, IW_C3_OC, IW_C3_SPATIAL, IW_C3_COL_W, stream);
    iw_conv_input_grad(a->conv2_grad.data, &ew->conv3_w, a->col3.data, a->mm3.data, B, IW_C3_IC, IW_C3_IN, IW_C3_OUT, IW_C3_K, IW_C3_S, IW_C3_OC, stream);
    iw_relu_backward<<<grid_size(B * IW_C2_OC * IW_C2_SPATIAL), BLOCK_SIZE, 0, stream>>>(
        a->conv2_grad.data, a->conv2_out.data, B * IW_C2_OC * IW_C2_SPATIAL
    );
    iw_conv_wgrad(a->conv2_grad.data, a->conv2_wgrad.data, a->col2.data, a->mm2.data, B, IW_C2_OC, IW_C2_SPATIAL, IW_C2_COL_W, stream);
    iw_conv_input_grad(a->conv1_grad.data, &ew->conv2_w, a->col2.data, a->mm2.data, B, IW_C2_IC, IW_C2_IN, IW_C2_OUT, IW_C2_K, IW_C2_S, IW_C2_OC, stream);
    iw_relu_backward<<<grid_size(B * IW_C1_OC * IW_C1_SPATIAL), BLOCK_SIZE, 0, stream>>>(
        a->conv1_grad.data, a->conv1_out.data, B * IW_C1_OC * IW_C1_SPATIAL
    );
    iw_conv_wgrad(a->conv1_grad.data, a->conv1_wgrad.data, a->col1.data, a->mm1.data, B, IW_C1_OC, IW_C1_SPATIAL, IW_C1_COL_W, stream);
}

static void iw_encoder_init_weights(void *w, ulong *seed, cudaStream_t stream) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    puf_kaiming_init(&ew->conv1_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->conv2_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->conv3_w, 1.0f, (*seed)++, stream);
    Prec pw = {.data = ew->proj_w.data, .shape = {ew->hidden, IW_CONCAT}};
    puf_kaiming_init(&pw, 1.0f, (*seed)++, stream);
    // as asteroids' entity encoder: gain sqrt(2) into the nonlinearity
    puf_kaiming_init(&ew->proj_l1_w, sqrtf(2.0f), (*seed)++, stream);
    puf_kaiming_init(&ew->proj_l2_w, 1.0f, (*seed)++, stream);
}

static void iw_encoder_reg_params(void *w, Allocator *alloc) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    ew->conv1_w = {.shape = {IW_C1_OC, IW_C1_COL_W}};
    ew->conv2_w = {.shape = {IW_C2_OC, IW_C2_COL_W}};
    ew->conv3_w = {.shape = {IW_C3_OC, IW_C3_COL_W}};
    ew->proj_w = {.shape = {ew->hidden, IW_CONCAT}};
    ew->proj_l1_w = {.shape = {IW_PROJ_HIDDEN, IW_PROJ_FEATURES}};
    ew->proj_l2_w = {.shape = {ew->hidden, IW_PROJ_HIDDEN}};
    // the CPU net (impulse_wars_net.h) loads weights in this order
    alloc_register(alloc, &ew->conv1_w);
    alloc_register(alloc, &ew->conv2_w);
    alloc_register(alloc, &ew->conv3_w);
    alloc_register(alloc, &ew->proj_w);
    alloc_register(alloc, &ew->proj_l1_w);
    alloc_register(alloc, &ew->proj_l2_w);
}

// projectile branch forward buffers, used by both training and rollout
static void iw_reg_proj_acts(IwEncoderActivations *a, Allocator *alloc, int B, int hidden) {
    int rows = B * IW_NUM_PROJECTILES;
    a->rec = {.shape = {rows, IW_PROJ_FEATURES}};
    a->rec_z1 = {.shape = {rows, IW_PROJ_HIDDEN}};
    a->rec_h1 = {.shape = {rows, IW_PROJ_HIDDEN}};
    a->rec_out = {.shape = {rows, hidden}};
    a->rec_active = {.shape = {rows}};
    alloc_register(alloc, &a->rec);
    alloc_register(alloc, &a->rec_z1);
    alloc_register(alloc, &a->rec_h1);
    alloc_register(alloc, &a->rec_out);
    alloc_register(alloc, &a->rec_active);
}

// forward CNN buffers, used by both training and rollout
static void iw_reg_conv_acts(IwEncoderActivations *a, Allocator *alloc, int B) {
    a->col1 = {.shape = {B * IW_C1_SPATIAL, IW_C1_COL_W}};
    a->mm1 = {.shape = {B * IW_C1_SPATIAL, IW_C1_OC}};
    a->conv1_out = {.shape = {B * IW_C1_OC * IW_C1_SPATIAL}};
    a->col2 = {.shape = {B * IW_C2_SPATIAL, IW_C2_COL_W}};
    a->mm2 = {.shape = {B * IW_C2_SPATIAL, IW_C2_OC}};
    a->conv2_out = {.shape = {B * IW_C2_OC * IW_C2_SPATIAL}};
    a->col3 = {.shape = {B * IW_C3_SPATIAL, IW_C3_COL_W}};
    a->mm3 = {.shape = {B * IW_C3_SPATIAL, IW_C3_OC}};
    a->conv3_out = {.shape = {B * IW_CONV_FLAT}};
    alloc_register(alloc, &a->col1);
    alloc_register(alloc, &a->mm1);
    alloc_register(alloc, &a->conv1_out);
    alloc_register(alloc, &a->col2);
    alloc_register(alloc, &a->mm2);
    alloc_register(alloc, &a->conv2_out);
    alloc_register(alloc, &a->col3);
    alloc_register(alloc, &a->mm3);
    alloc_register(alloc, &a->conv3_out);
}

static void iw_encoder_reg_train(
    void *w, void *activations, Allocator *acts, Allocator *grads, int B_TT
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    *a = {};
    a->concat = {.shape = {B_TT, IW_CONCAT}};
    a->out = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts, &a->concat);
    alloc_register(acts, &a->out);
    iw_reg_conv_acts(a, acts, B_TT);
    iw_reg_proj_acts(a, acts, B_TT, ew->hidden);
    a->rec_argmax = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts, &a->rec_argmax);
    a->conv1_grad = {.shape = {B_TT * IW_C1_OC * IW_C1_SPATIAL}};
    a->conv2_grad = {.shape = {B_TT * IW_C2_OC * IW_C2_SPATIAL}};
    a->conv3_grad = {.shape = {B_TT * IW_CONV_FLAT}};
    alloc_register(acts, &a->conv1_grad);
    alloc_register(acts, &a->conv2_grad);
    alloc_register(acts, &a->conv3_grad);
    a->conv1_wgrad = {.shape = {IW_C1_OC, IW_C1_COL_W}};
    a->conv2_wgrad = {.shape = {IW_C2_OC, IW_C2_COL_W}};
    a->conv3_wgrad = {.shape = {IW_C3_OC, IW_C3_COL_W}};
    a->proj_wgrad = {.shape = {ew->hidden, IW_CONCAT}};
    a->proj_l1_wgrad = {.shape = {IW_PROJ_HIDDEN, IW_PROJ_FEATURES}};
    a->proj_l2_wgrad = {.shape = {ew->hidden, IW_PROJ_HIDDEN}};
    // same order as the params in iw_encoder_reg_params
    alloc_register(grads, &a->conv1_wgrad);
    alloc_register(grads, &a->conv2_wgrad);
    alloc_register(grads, &a->conv3_wgrad);
    alloc_register(grads, &a->proj_wgrad);
    alloc_register(grads, &a->proj_l1_wgrad);
    alloc_register(grads, &a->proj_l2_wgrad);
}

static void iw_encoder_reg_rollout(
    void *w, void *activations, Allocator *alloc, int B
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    *a = {};
    a->concat = {.shape = {B, IW_CONCAT}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->concat);
    alloc_register(alloc, &a->out);
    iw_reg_conv_acts(a, alloc, B);
    iw_reg_proj_acts(a, alloc, B, ew->hidden);
}

static void *iw_encoder_create_weights(void *self) {
    Encoder *e = (Encoder *)self;
    IwEncoderWeights *ew =
        (IwEncoderWeights *)calloc(1, sizeof(IwEncoderWeights));
    ew->obs_size = e->in_dim;
    ew->hidden = e->out_dim;
    return ew;
}

static void create_impulse_wars_encoder(Encoder *enc) {
    *enc = Encoder{
        .forward = iw_encoder_forward,
        .backward = iw_encoder_backward,
        .init_weights = iw_encoder_init_weights,
        .reg_params = iw_encoder_reg_params,
        .reg_train = iw_encoder_reg_train,
        .reg_rollout = iw_encoder_reg_rollout,
        .create_weights = iw_encoder_create_weights,
        .in_dim = enc->in_dim,
        .out_dim = enc->out_dim,
        .activation_size = sizeof(IwEncoderActivations),
    };
}
