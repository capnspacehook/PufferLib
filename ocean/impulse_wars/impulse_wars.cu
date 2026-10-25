// Impulse Wars CUDA encoder: map grid CNN, own projectile, enemy projectile
// and floating wall branches, one-hot type slots, concat with the continuous
// obs and the enemy projectile count, projection.
// The map grid's packed cell codes are unpacked into one-hot channels by
// conv1's im2col and run through 3 convs with ReLU, then concatenated and
// projected through a GELU. Conv activations stay NHWC, as the matmul rows,
// and are kept before ReLU: each consumer applies it when reading. Own and
// enemy projectile and floating wall slots are records for a shared Linear -> GELU -> Linear
// per branch that is max pooled over the non-empty records and added to the
// projection, like the OSRS entity encoder; the second Linear is fused with
// the pool so the (records, hidden) outputs are never stored. Wall, weapon
// and ray hit type slots are one-hot encoded into the concat. Included by
// src/ocean.cu.

#include "impulse_wars_obs.h"

static_assert(OBS_SIZE >= IW_BASE_OBS_SIZE + IW_ENEMY_DRONE_SIZE && (OBS_SIZE - IW_BASE_OBS_SIZE) % IW_ENEMY_DRONE_SIZE == 0, "impulse_wars encoder obs layout does not match OBS_SIZE");
static_assert(IW_GRID_CODE_MAX < 256, "impulse_wars map grid codes must be exact in bf16");
static_assert(IW_C3_OUT == 3, "impulse_wars CNN output should keep a 3x3 layout");

// samples per block of the record branches' second layer weight gradient
#define IW_REC_WGRAD_CHUNK 64
// hidden units per block of the record branches' second layer weight gradient
#define IW_REC_WGRAD_UNITS 16
// threads per sample of the record branches' pool backward, each
// accumulating every IW_REC_BWD_GROUPS-th hidden unit
#define IW_REC_BWD_GROUPS 16

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

// im2col of an unpadded square conv over the NHWC (B, in, in, ic) output of
// the previous conv before its ReLU, applying the ReLU:
// (B * out * out, k * k * ic)
__global__ void iw_im2col(
    precision_t *__restrict__ col, const precision_t *__restrict__ z, int B, int ic, int in, int out, int k, int stride
) {
    int colW = k * k * ic;
    int spatial = out * out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * spatial * colW) {
        return;
    }
    int row = idx / colW;
    int b = row / spatial;
    float v = to_float(z[(int64_t)b * in * in * ic + iw_im2col_src(row % spatial, idx % colW, ic, in, out, k, stride)]);
    col[idx] = from_float(fmaxf(v, 0.0f));
}

// gradient at the NHWC (B, in, in, ic) pre-ReLU input z of an unpadded
// square conv from the gradient of its im2col
__global__ void iw_col2im(
    precision_t *__restrict__ grad_z, const precision_t *__restrict__ col, const precision_t *__restrict__ z, int B, int ic, int in, int out, int k, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * in * in * ic) {
        return;
    }
    if (to_float(z[idx]) <= 0.0f) {
        grad_z[idx] = from_float(0.0f);
        return;
    }
    int c = idx % ic;
    int pix = idx / ic;
    int iw = pix % in;
    int ih = (pix / in) % in;
    int b = pix / (in * in);
    int colW = k * k * ic;
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
            sum += to_float(col[r * colW + (kh * k + kw) * ic + c]);
        }
    }
    grad_z[idx] = from_float(sum);
}

// copies the CNN block of the concat gradient, masked by the conv3 ReLU
__global__ void iw_conv_grad_from_concat(
    precision_t *__restrict__ conv_grad, const precision_t *__restrict__ grad_concat, const precision_t *__restrict__ z, int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_CONV_FLAT) {
        return;
    }
    int b = idx / IW_CONV_FLAT;
    int c = idx % IW_CONV_FLAT;
    conv_grad[idx] = to_float(z[idx]) > 0.0f ? grad_concat[(int64_t)b * IW_CONCAT + c] : from_float(0.0f);
}

__global__ void iw_concat_kernel(
    precision_t *__restrict__ concat, const precision_t *__restrict__ obs, const precision_t *__restrict__ conv_z, int B, int obs_size
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
        v = from_float(fmaxf(to_float(conv_z[(int64_t)b * IW_CONV_FLAT + c]), 0.0f));
    } else if (c < IW_CONV_FLAT + IW_ONEHOT_COLS) {
        iwOneHotCol col = iw_onehot_col(c - IW_CONV_FLAT);
        v = from_float(iw_onehot_value(to_float(o[col.obsIdx]), col));
    } else if (c < IW_SCALAR_COLS_END) {
        v = o[IW_SCALAR_IDX.idx[c - IW_CONV_FLAT - IW_ONEHOT_COLS]];
    } else if (c < IW_CONCAT_USED) {
        int n = 0;
        for (int r = 0; r < IW_NUM_ENEMY_PROJECTILES; r++) {
            n += iw_proj_active(to_float(o[iw_proj_slot_idx(IW_NUM_OWN_PROJECTILES + r)]));
        }
        v = from_float((float)n / IW_NUM_ENEMY_PROJECTILES);
    } else {
        v = from_float(0.0f);
    }
    concat[idx] = v;
}

// projectile records from the R projectile slots starting at slot FIRST:
// (B * R, IW_PROJ_FEATURES), and whether each record is a projectile
template <int R, int FIRST>
__global__ void iw_proj_gather(
    precision_t *__restrict__ rec, int *__restrict__ active, const precision_t *__restrict__ obs, int B, int obs_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * R * IW_PROJ_FEATURES) {
        return;
    }
    int f = idx % IW_PROJ_FEATURES;
    int row = idx / IW_PROJ_FEATURES;
    int b = row / R;
    const precision_t *slot = obs + (int64_t)b * obs_size + iw_proj_slot_idx(FIRST + row % R);
    rec[idx] = from_float(iw_proj_feature(to_float(slot[iw_proj_field(f)]), f));
    if (f == 0) {
        active[row] = iw_proj_active(to_float(slot[0]));
    }
}

// floating wall records from the obs: (B * IW_NUM_FLOATING_WALLS,
// IW_FW_FEATURES), and whether each record is a floating wall
__global__ void iw_fw_gather(
    precision_t *__restrict__ rec, int *__restrict__ active, const precision_t *__restrict__ obs, int B, int obs_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_NUM_FLOATING_WALLS * IW_FW_FEATURES) {
        return;
    }
    int f = idx % IW_FW_FEATURES;
    int row = idx / IW_FW_FEATURES;
    int b = row / IW_NUM_FLOATING_WALLS;
    const precision_t *slot = obs + (int64_t)b * obs_size + iw_fw_slot_idx(row % IW_NUM_FLOATING_WALLS);
    rec[idx] = from_float(iw_fw_feature(to_float(slot[iw_fw_field(f)]), f));
    if (f == 0) {
        active[row] = iw_fw_active(to_float(slot[0]));
    }
}

__global__ void iw_gelu_kernel(precision_t *__restrict__ out, const precision_t *__restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out[idx] = from_float(iw_gelu(to_float(in[idx])));
}

// as GELU's backward, but overwrites GELU's input with the gradient at it
// so grad, which the record branches also read, is left untouched
__global__ void iw_gelu_backward_into_input(precision_t *__restrict__ in, const precision_t *__restrict__ grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    in[idx] = from_float(to_float(grad[idx]) * iw_gelu_grad(to_float(in[idx])));
}

// record branch second layer fused with the pool: adds the max over the
// active records r of h1[b, r] . w2[h] to out[b, h]; a sample with no active
// records adds 0. argmax is the pooled record, -1 if none, or null for
// rollout. One block per sample.
template <int R, int HR>
__global__ void iw_rec_pool(
    precision_t *__restrict__ out, int *__restrict__ argmax, const precision_t *__restrict__ h1, const precision_t *__restrict__ w2, const int *__restrict__ active, int H
) {
    __shared__ float sh1[R * HR];
    __shared__ int sact[R];
    int b = blockIdx.x;
    for (int i = threadIdx.x; i < R * HR; i += blockDim.x) {
        sh1[i] = to_float(h1[(int64_t)b * R * HR + i]);
    }
    for (int r = threadIdx.x; r < R; r += blockDim.x) {
        sact[r] = active[b * R + r];
    }
    __syncthreads();
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float w[HR];
        for (int k = 0; k < HR; k++) {
            w[k] = to_float(w2[h * HR + k]);
        }
        float best = 0.0f;
        int bestRecord = -1;
        for (int r = 0; r < R; r++) {
            if (!sact[r]) {
                continue;
            }
            float v = 0.0f;
            for (int k = 0; k < HR; k++) {
                v += sh1[r * HR + k] * w[k];
            }
            if (bestRecord < 0 || v > best) {
                best = v;
                bestRecord = r;
            }
        }
        int64_t o = (int64_t)b * H + h;
        out[o] = from_float(to_float(out[o]) + best);
        if (argmax) {
            argmax[o] = bestRecord;
        }
    }
}

// routes the encoder output gradient (B, H) through the pool and second
// layer to the first layer's pre-GELU outputs z1 (B * R, HR), overwriting
// z1 with the gradient at it. One block of IW_REC_BWD_GROUPS * HR threads
// per sample: thread (g, k) sums over every IW_REC_BWD_GROUPS-th hidden
// unit into per record partials, which are then summed over g.
template <int R, int HR>
__global__ void iw_rec_pool_backward(
    precision_t *__restrict__ z1, const precision_t *__restrict__ grad, const int *__restrict__ argmax, const precision_t *__restrict__ w2, int H
) {
    __shared__ float partial[IW_REC_BWD_GROUPS][R * HR];
    int b = blockIdx.x;
    int g = threadIdx.x / HR;
    int k = threadIdx.x % HR;
    for (int i = threadIdx.x; i < IW_REC_BWD_GROUPS * R * HR; i += blockDim.x) {
        partial[i / (R * HR)][i % (R * HR)] = 0.0f;
    }
    __syncthreads();
    // each (g, k) slot is only touched by its own thread
    for (int h = g; h < H; h += IW_REC_BWD_GROUPS) {
        int r = argmax[(int64_t)b * H + h];
        if (r >= 0) {
            partial[g][r * HR + k] += to_float(grad[(int64_t)b * H + h]) * to_float(w2[h * HR + k]);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < R * HR; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < IW_REC_BWD_GROUPS; j++) {
            sum += partial[j][i];
        }
        int64_t zi = (int64_t)b * R * HR + i;
        z1[zi] = from_float(sum * iw_gelu_grad(to_float(z1[zi])));
    }
}

// second layer weight gradient partials of a record branch,
// dw2[h, k] = sum over b of grad[b, h] * h1[b, argmax[b, h], k], for
// IW_REC_WGRAD_UNITS hidden units (blockIdx.x) and IW_REC_WGRAD_CHUNK samples
// (blockIdx.y) per block: (gridDim.y, H * HR) into partials
template <int R, int HR>
__global__ void iw_rec_wgrad_partial(
    float *__restrict__ partials, const precision_t *__restrict__ grad, const int *__restrict__ argmax, const precision_t *__restrict__ h1, int B, int H
) {
    int h = blockIdx.x * IW_REC_WGRAD_UNITS + threadIdx.x / HR;
    int k = threadIdx.x % HR;
    if (h >= H) {
        return;
    }
    int b0 = blockIdx.y * IW_REC_WGRAD_CHUNK;
    int b1 = min(b0 + IW_REC_WGRAD_CHUNK, B);
    float sum = 0.0f;
    for (int b = b0; b < b1; b++) {
        int r = argmax[(int64_t)b * H + h];
        if (r >= 0) {
            sum += to_float(grad[(int64_t)b * H + h]) * to_float(h1[((int64_t)b * R + r) * HR + k]);
        }
    }
    partials[(int64_t)blockIdx.y * H * HR + h * HR + k] = sum;
}

// sums the (chunks, n) weight gradient partials into wgrad (n)
__global__ void iw_rec_wgrad_reduce(precision_t *__restrict__ wgrad, const float *__restrict__ partials, int chunks, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float sum = 0.0f;
    for (int c = 0; c < chunks; c++) {
        sum += partials[(int64_t)c * n + idx];
    }
    wgrad[idx] = from_float(sum);
}

struct IwEncoderWeights {
    Prec conv1_w, conv2_w, conv3_w, proj_w, own_proj_l1_w, own_proj_l2_w, enemy_proj_l1_w, enemy_proj_l2_w, fw_l1_w, fw_l2_w;
    int obs_size, hidden;
};

struct IwEncoderActivations {
    // proj_z is the projection before GELU; backward reuses it for its
    // gradient
    Prec concat, proj_z, out;
    // per conv: im2col, NHWC output before ReLU, gradient at that output;
    // backward reuses the im2col for its gradient
    Prec col1, conv1_z, conv1_grad;
    Prec col2, conv2_z, conv2_grad;
    Prec col3, conv3_z, conv3_grad;
    // own projectile branch: records, first layer pre and post GELU;
    // backward reuses own_z1 for its gradient
    Prec own_rec, own_z1, own_h1;
    Int own_active, own_argmax;
    // enemy projectile and floating wall branches, as the own projectile
    // branch
    Prec enemy_rec, enemy_z1, enemy_h1;
    Int enemy_active, enemy_argmax;
    Prec fw_rec, fw_z1, fw_h1;
    Int fw_active, fw_argmax;
    // second layer weight gradient partials, shared by the branches
    Float rec_wgrad_partials;
    Prec conv1_wgrad, conv2_wgrad, conv3_wgrad, proj_wgrad, own_proj_l1_wgrad, own_proj_l2_wgrad, enemy_proj_l1_wgrad,
        enemy_proj_l2_wgrad, fw_l1_wgrad, fw_l2_wgrad;
};

static_assert(IW_PROJ_HIDDEN == IW_FW_HIDDEN, "impulse_wars record branches share their weight gradient partials");

// matmul of a conv's im2col with its weights: NHWC (B * spatial, oc) output
static void iw_conv_mm(precision_t *col, Prec *w, precision_t *z, int B, int oc, int spatial, int colW, cudaStream_t stream) {
    Prec colT = {.data = col, .shape = {B * spatial, colW}};
    Prec zT = {.data = z, .shape = {B * spatial, oc}};
    puf_mm(&colT, w, &zT, stream);
}

// weight gradient of a conv from the NHWC gradient at its output, then the
// gradient of its im2col into col
static void iw_conv_backward(
    precision_t *grad_out, Prec *w, precision_t *wgrad, precision_t *col, int B, int oc, int spatial, int colW, cudaStream_t stream
) {
    Prec gT = {.data = grad_out, .shape = {B * spatial, oc}};
    Prec colT = {.data = col, .shape = {B * spatial, colW}};
    Prec wgT = {.data = wgrad, .shape = {oc, colW}};
    puf_mm_tn(&gT, &colT, &wgT, stream);
    puf_mm_nn(&gT, w, &colT, stream);
}

// record branch forward from gathered records: adds the pooled records
// (B * R rows) to out; argmax is null for rollout
template <int R, int HR>
static void iw_rec_forward(
    Prec *rec, Prec *l1_w, Prec *z1, Prec *h1, Prec *l2_w, const int *active, precision_t *out, int *argmax, int B, int H, cudaStream_t stream
) {
    int n1 = B * R * HR;
    puf_mm(rec, l1_w, z1, stream);
    iw_gelu_kernel<<<grid_size(n1), BLOCK_SIZE, 0, stream>>>(h1->data, z1->data, n1);
    iw_rec_pool<R, HR><<<B, BLOCK_SIZE, 0, stream>>>(out, argmax, h1->data, l2_w->data, active, H);
}

// record branch backward: the second layer's weight gradient, then z1 is
// overwritten with the gradient at it for the first layer's
template <int R, int HR>
static void iw_rec_backward(
    Prec grad, Prec *rec, Prec *l1_w_grad, Prec *z1, Prec *h1, Prec *l2_w, Prec *l2_w_grad, Float *partials, const int *argmax, int B, int H, cudaStream_t stream
) {
    int chunks = (B + IW_REC_WGRAD_CHUNK - 1) / IW_REC_WGRAD_CHUNK;
    dim3 wgradGrid((H + IW_REC_WGRAD_UNITS - 1) / IW_REC_WGRAD_UNITS, chunks);
    iw_rec_wgrad_partial<R, HR><<<wgradGrid, IW_REC_WGRAD_UNITS * HR, 0, stream>>>(
        partials->data, grad.data, argmax, h1->data, B, H
    );
    iw_rec_wgrad_reduce<<<grid_size(H * HR), BLOCK_SIZE, 0, stream>>>(l2_w_grad->data, partials->data, chunks, H * HR);
    iw_rec_pool_backward<R, HR><<<B, IW_REC_BWD_GROUPS * HR, 0, stream>>>(z1->data, grad.data, argmax, l2_w->data, H);
    puf_mm_tn(z1, rec, l1_w_grad, stream);
}

// own projectile, enemy projectile and floating wall branches forward: adds
// the pooled records to a->out; the argmaxes aren't allocated for rollout,
// so stay null
static void iw_branches_forward(IwEncoderWeights *ew, IwEncoderActivations *a, Prec input, int B, cudaStream_t stream) {
    iw_proj_gather<IW_NUM_OWN_PROJECTILES, 0><<<grid_size(B * IW_NUM_OWN_PROJECTILES * IW_PROJ_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->own_rec.data, a->own_active.data, input.data, B, ew->obs_size
    );
    iw_rec_forward<IW_NUM_OWN_PROJECTILES, IW_PROJ_HIDDEN>(
        &a->own_rec, &ew->own_proj_l1_w, &a->own_z1, &a->own_h1, &ew->own_proj_l2_w, a->own_active.data, a->out.data,
        a->own_argmax.data, B, ew->hidden, stream
    );
    iw_proj_gather<IW_NUM_ENEMY_PROJECTILES, IW_NUM_OWN_PROJECTILES>
        <<<grid_size(B * IW_NUM_ENEMY_PROJECTILES * IW_PROJ_FEATURES), BLOCK_SIZE, 0, stream>>>(
            a->enemy_rec.data, a->enemy_active.data, input.data, B, ew->obs_size
        );
    iw_rec_forward<IW_NUM_ENEMY_PROJECTILES, IW_PROJ_HIDDEN>(
        &a->enemy_rec, &ew->enemy_proj_l1_w, &a->enemy_z1, &a->enemy_h1, &ew->enemy_proj_l2_w, a->enemy_active.data,
        a->out.data, a->enemy_argmax.data, B, ew->hidden, stream
    );
    iw_fw_gather<<<grid_size(B * IW_NUM_FLOATING_WALLS * IW_FW_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->fw_rec.data, a->fw_active.data, input.data, B, ew->obs_size
    );
    iw_rec_forward<IW_NUM_FLOATING_WALLS, IW_FW_HIDDEN>(
        &a->fw_rec, &ew->fw_l1_w, &a->fw_z1, &a->fw_h1, &ew->fw_l2_w, a->fw_active.data, a->out.data,
        a->fw_argmax.data, B, ew->hidden, stream
    );
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
    iw_conv_mm(a->col1.data, &ew->conv1_w, a->conv1_z.data, B, IW_C1_OC, IW_C1_SPATIAL, IW_C1_COL_W, stream);
    iw_im2col<<<grid_size(B * IW_C2_SPATIAL * IW_C2_COL_W), BLOCK_SIZE, 0, stream>>>(
        a->col2.data, a->conv1_z.data, B, IW_C2_IC, IW_C2_IN, IW_C2_OUT, IW_C2_K, IW_C2_S
    );
    iw_conv_mm(a->col2.data, &ew->conv2_w, a->conv2_z.data, B, IW_C2_OC, IW_C2_SPATIAL, IW_C2_COL_W, stream);
    iw_im2col<<<grid_size(B * IW_C3_SPATIAL * IW_C3_COL_W), BLOCK_SIZE, 0, stream>>>(
        a->col3.data, a->conv2_z.data, B, IW_C3_IC, IW_C3_IN, IW_C3_OUT, IW_C3_K, IW_C3_S
    );
    iw_conv_mm(a->col3.data, &ew->conv3_w, a->conv3_z.data, B, IW_C3_OC, IW_C3_SPATIAL, IW_C3_COL_W, stream);

    iw_concat_kernel<<<grid_size(B * IW_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, input.data, a->conv3_z.data, B, ew->obs_size
    );
    puf_mm(&a->concat, &ew->proj_w, &a->proj_z, stream);
    iw_gelu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(a->out.data, a->proj_z.data, B * ew->hidden);

    iw_branches_forward(ew, a, input, B, stream);
    return a->out;
}

static void iw_encoder_backward(
    void *w, void *activations, Prec grad, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = grad.shape[0];

    iw_rec_backward<IW_NUM_OWN_PROJECTILES, IW_PROJ_HIDDEN>(
        grad, &a->own_rec, &a->own_proj_l1_wgrad, &a->own_z1, &a->own_h1, &ew->own_proj_l2_w, &a->own_proj_l2_wgrad,
        &a->rec_wgrad_partials, a->own_argmax.data, B, ew->hidden, stream
    );
    iw_rec_backward<IW_NUM_ENEMY_PROJECTILES, IW_PROJ_HIDDEN>(
        grad, &a->enemy_rec, &a->enemy_proj_l1_wgrad, &a->enemy_z1, &a->enemy_h1, &ew->enemy_proj_l2_w,
        &a->enemy_proj_l2_wgrad, &a->rec_wgrad_partials, a->enemy_argmax.data, B, ew->hidden, stream
    );
    iw_rec_backward<IW_NUM_FLOATING_WALLS, IW_FW_HIDDEN>(
        grad, &a->fw_rec, &a->fw_l1_wgrad, &a->fw_z1, &a->fw_h1, &ew->fw_l2_w, &a->fw_l2_wgrad, &a->rec_wgrad_partials,
        a->fw_argmax.data, B, ew->hidden, stream
    );

    // the record branches are added after GELU, so only the projection's
    // gradient goes through it
    iw_gelu_backward_into_input<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->proj_z.data, grad.data, B * ew->hidden
    );
    Prec grad_proj = {.data = a->proj_z.data, .shape = {B, ew->hidden}};
    puf_mm_tn(&grad_proj, &a->concat, &a->proj_wgrad, stream);
    Prec grad_concat = {.data = a->concat.data, .shape = {B, IW_CONCAT}};
    puf_mm_nn(&grad_proj, &ew->proj_w, &grad_concat, stream);

    // CNN, back to front; conv1's input isn't learned so it gets no input grad
    iw_conv_grad_from_concat<<<grid_size(B * IW_CONV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->conv3_grad.data, grad_concat.data, a->conv3_z.data, B
    );
    iw_conv_backward(a->conv3_grad.data, &ew->conv3_w, a->conv3_wgrad.data, a->col3.data, B, IW_C3_OC, IW_C3_SPATIAL, IW_C3_COL_W, stream);
    iw_col2im<<<grid_size(B * IW_C2_SPATIAL * IW_C2_OC), BLOCK_SIZE, 0, stream>>>(
        a->conv2_grad.data, a->col3.data, a->conv2_z.data, B, IW_C3_IC, IW_C3_IN, IW_C3_OUT, IW_C3_K, IW_C3_S
    );
    iw_conv_backward(a->conv2_grad.data, &ew->conv2_w, a->conv2_wgrad.data, a->col2.data, B, IW_C2_OC, IW_C2_SPATIAL, IW_C2_COL_W, stream);
    iw_col2im<<<grid_size(B * IW_C1_SPATIAL * IW_C1_OC), BLOCK_SIZE, 0, stream>>>(
        a->conv1_grad.data, a->col2.data, a->conv1_z.data, B, IW_C2_IC, IW_C2_IN, IW_C2_OUT, IW_C2_K, IW_C2_S
    );
    Prec g1 = {.data = a->conv1_grad.data, .shape = {B * IW_C1_SPATIAL, IW_C1_OC}};
    Prec col1 = {.data = a->col1.data, .shape = {B * IW_C1_SPATIAL, IW_C1_COL_W}};
    puf_mm_tn(&g1, &col1, &a->conv1_wgrad, stream);
}

static void iw_encoder_init_weights(void *w, ulong *seed, cudaStream_t stream) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    puf_kaiming_init(&ew->conv1_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->conv2_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->conv3_w, 1.0f, (*seed)++, stream);
    // gains are scaled by sqrt(padded / used width) so the fan in is the
    // used width, as if unpadded
    Prec pw = {.data = ew->proj_w.data, .shape = {ew->hidden, IW_CONCAT}};
    puf_kaiming_init(&pw, sqrtf((float)IW_CONCAT / IW_CONCAT_USED), (*seed)++, stream);
    // as asteroids' entity encoder: gain sqrt(2) into the nonlinearity
    puf_kaiming_init(&ew->own_proj_l1_w, sqrtf(2.0f * IW_PROJ_FEATURES / IW_PROJ_USED_FEATURES), (*seed)++, stream);
    puf_kaiming_init(&ew->own_proj_l2_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->enemy_proj_l1_w, sqrtf(2.0f * IW_PROJ_FEATURES / IW_PROJ_USED_FEATURES), (*seed)++, stream);
    puf_kaiming_init(&ew->enemy_proj_l2_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->fw_l1_w, sqrtf(2.0f * IW_FW_FEATURES / IW_FW_USED_FEATURES), (*seed)++, stream);
    puf_kaiming_init(&ew->fw_l2_w, 1.0f, (*seed)++, stream);
}

static void iw_encoder_reg_params(void *w, Allocator *alloc) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    // conv weights are (OC, K, K, IC), see IW_C1_COL_W
    ew->conv1_w = {.shape = {IW_C1_OC, IW_C1_COL_W}};
    ew->conv2_w = {.shape = {IW_C2_OC, IW_C2_COL_W}};
    ew->conv3_w = {.shape = {IW_C3_OC, IW_C3_COL_W}};
    ew->proj_w = {.shape = {ew->hidden, IW_CONCAT}};
    ew->own_proj_l1_w = {.shape = {IW_PROJ_HIDDEN, IW_PROJ_FEATURES}};
    ew->own_proj_l2_w = {.shape = {ew->hidden, IW_PROJ_HIDDEN}};
    ew->enemy_proj_l1_w = {.shape = {IW_PROJ_HIDDEN, IW_PROJ_FEATURES}};
    ew->enemy_proj_l2_w = {.shape = {ew->hidden, IW_PROJ_HIDDEN}};
    ew->fw_l1_w = {.shape = {IW_FW_HIDDEN, IW_FW_FEATURES}};
    ew->fw_l2_w = {.shape = {ew->hidden, IW_FW_HIDDEN}};
    // the CPU net (impulse_wars_net.h) loads weights in this order
    alloc_register(alloc, &ew->conv1_w);
    alloc_register(alloc, &ew->conv2_w);
    alloc_register(alloc, &ew->conv3_w);
    alloc_register(alloc, &ew->proj_w);
    alloc_register(alloc, &ew->own_proj_l1_w);
    alloc_register(alloc, &ew->own_proj_l2_w);
    alloc_register(alloc, &ew->enemy_proj_l1_w);
    alloc_register(alloc, &ew->enemy_proj_l2_w);
    alloc_register(alloc, &ew->fw_l1_w);
    alloc_register(alloc, &ew->fw_l2_w);
}

// own projectile, enemy projectile and floating wall branch forward buffers,
// used by both training and rollout
static void iw_reg_branch_acts(IwEncoderActivations *a, Allocator *alloc, int B) {
    int ownRows = B * IW_NUM_OWN_PROJECTILES;
    a->own_rec = {.shape = {ownRows, IW_PROJ_FEATURES}};
    a->own_z1 = {.shape = {ownRows, IW_PROJ_HIDDEN}};
    a->own_h1 = {.shape = {ownRows, IW_PROJ_HIDDEN}};
    a->own_active = {.shape = {ownRows}};
    alloc_register(alloc, &a->own_rec);
    alloc_register(alloc, &a->own_z1);
    alloc_register(alloc, &a->own_h1);
    alloc_register(alloc, &a->own_active);
    int enemyRows = B * IW_NUM_ENEMY_PROJECTILES;
    a->enemy_rec = {.shape = {enemyRows, IW_PROJ_FEATURES}};
    a->enemy_z1 = {.shape = {enemyRows, IW_PROJ_HIDDEN}};
    a->enemy_h1 = {.shape = {enemyRows, IW_PROJ_HIDDEN}};
    a->enemy_active = {.shape = {enemyRows}};
    alloc_register(alloc, &a->enemy_rec);
    alloc_register(alloc, &a->enemy_z1);
    alloc_register(alloc, &a->enemy_h1);
    alloc_register(alloc, &a->enemy_active);
    int fwRows = B * IW_NUM_FLOATING_WALLS;
    a->fw_rec = {.shape = {fwRows, IW_FW_FEATURES}};
    a->fw_z1 = {.shape = {fwRows, IW_FW_HIDDEN}};
    a->fw_h1 = {.shape = {fwRows, IW_FW_HIDDEN}};
    a->fw_active = {.shape = {fwRows}};
    alloc_register(alloc, &a->fw_rec);
    alloc_register(alloc, &a->fw_z1);
    alloc_register(alloc, &a->fw_h1);
    alloc_register(alloc, &a->fw_active);
}

// forward CNN buffers, used by both training and rollout
static void iw_reg_conv_acts(IwEncoderActivations *a, Allocator *alloc, int B) {
    a->col1 = {.shape = {B * IW_C1_SPATIAL, IW_C1_COL_W}};
    a->conv1_z = {.shape = {B * IW_C1_SPATIAL, IW_C1_OC}};
    a->col2 = {.shape = {B * IW_C2_SPATIAL, IW_C2_COL_W}};
    a->conv2_z = {.shape = {B * IW_C2_SPATIAL, IW_C2_OC}};
    a->col3 = {.shape = {B * IW_C3_SPATIAL, IW_C3_COL_W}};
    a->conv3_z = {.shape = {B * IW_C3_SPATIAL, IW_C3_OC}};
    alloc_register(alloc, &a->col1);
    alloc_register(alloc, &a->conv1_z);
    alloc_register(alloc, &a->col2);
    alloc_register(alloc, &a->conv2_z);
    alloc_register(alloc, &a->col3);
    alloc_register(alloc, &a->conv3_z);
}

static void iw_encoder_reg_train(
    void *w, void *activations, Allocator *acts, Allocator *grads, int B_TT
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    *a = {};
    a->concat = {.shape = {B_TT, IW_CONCAT}};
    a->proj_z = {.shape = {B_TT, ew->hidden}};
    a->out = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts, &a->concat);
    alloc_register(acts, &a->proj_z);
    alloc_register(acts, &a->out);
    iw_reg_conv_acts(a, acts, B_TT);
    iw_reg_branch_acts(a, acts, B_TT);
    a->own_argmax = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts, &a->own_argmax);
    a->enemy_argmax = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts, &a->enemy_argmax);
    a->fw_argmax = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts, &a->fw_argmax);
    int chunks = (B_TT + IW_REC_WGRAD_CHUNK - 1) / IW_REC_WGRAD_CHUNK;
    a->rec_wgrad_partials = {.shape = {chunks, ew->hidden * IW_PROJ_HIDDEN}};
    alloc_register(acts, &a->rec_wgrad_partials);
    a->conv1_grad = {.shape = {B_TT * IW_C1_SPATIAL * IW_C1_OC}};
    a->conv2_grad = {.shape = {B_TT * IW_C2_SPATIAL * IW_C2_OC}};
    a->conv3_grad = {.shape = {B_TT * IW_CONV_FLAT}};
    alloc_register(acts, &a->conv1_grad);
    alloc_register(acts, &a->conv2_grad);
    alloc_register(acts, &a->conv3_grad);
    a->conv1_wgrad = {.shape = {IW_C1_OC, IW_C1_COL_W}};
    a->conv2_wgrad = {.shape = {IW_C2_OC, IW_C2_COL_W}};
    a->conv3_wgrad = {.shape = {IW_C3_OC, IW_C3_COL_W}};
    a->proj_wgrad = {.shape = {ew->hidden, IW_CONCAT}};
    a->own_proj_l1_wgrad = {.shape = {IW_PROJ_HIDDEN, IW_PROJ_FEATURES}};
    a->own_proj_l2_wgrad = {.shape = {ew->hidden, IW_PROJ_HIDDEN}};
    a->enemy_proj_l1_wgrad = {.shape = {IW_PROJ_HIDDEN, IW_PROJ_FEATURES}};
    a->enemy_proj_l2_wgrad = {.shape = {ew->hidden, IW_PROJ_HIDDEN}};
    a->fw_l1_wgrad = {.shape = {IW_FW_HIDDEN, IW_FW_FEATURES}};
    a->fw_l2_wgrad = {.shape = {ew->hidden, IW_FW_HIDDEN}};
    // same order as the params in iw_encoder_reg_params
    alloc_register(grads, &a->conv1_wgrad);
    alloc_register(grads, &a->conv2_wgrad);
    alloc_register(grads, &a->conv3_wgrad);
    alloc_register(grads, &a->proj_wgrad);
    alloc_register(grads, &a->own_proj_l1_wgrad);
    alloc_register(grads, &a->own_proj_l2_wgrad);
    alloc_register(grads, &a->enemy_proj_l1_wgrad);
    alloc_register(grads, &a->enemy_proj_l2_wgrad);
    alloc_register(grads, &a->fw_l1_wgrad);
    alloc_register(grads, &a->fw_l2_wgrad);
}

static void iw_encoder_reg_rollout(
    void *w, void *activations, Allocator *alloc, int B
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    *a = {};
    a->concat = {.shape = {B, IW_CONCAT}};
    a->proj_z = {.shape = {B, ew->hidden}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->concat);
    alloc_register(alloc, &a->proj_z);
    alloc_register(alloc, &a->out);
    iw_reg_conv_acts(a, alloc, B);
    iw_reg_branch_acts(a, alloc, B);
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
