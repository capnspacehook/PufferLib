// Impulse Wars CUDA encoder: map grid CNN, wall type, weapon type and ray
// hit type embeddings, concat with the continuous obs, projection. The map
// grid's packed cell codes are unpacked into one-hot channels by conv1's
// im2col and run through 3 convs with ReLU. Wall types of near walls and
// floating walls share one table; weapon types of pickups, projectiles,
// enemy drones and the agent's drone share another; hit types of the enemy
// and agent rays share a third. Included by src/ocean.cu.

#include "impulse_wars_obs.h"

static_assert(OBS_SIZE >= IW_BASE_OBS_SIZE + IW_ENEMY_DRONE_SIZE && (OBS_SIZE - IW_BASE_OBS_SIZE) % IW_ENEMY_DRONE_SIZE == 0, "impulse_wars encoder obs layout does not match OBS_SIZE");
static_assert(IW_EMB_N <= 1024, "impulse_wars embed tables too big for shared memory");
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
        if (!iw_is_type_slot(i)) {
            t.idx[n++] = i;
        }
    }
    return t;
}

static constexpr IwScalarIdx IW_SCALAR_IDX_HOST = iw_make_scalar_idx();
static_assert(IW_SCALAR_IDX_HOST.idx[IW_NUM_SCALARS - 1] == OBS_SIZE - 1, "impulse_wars type slot count does not match IW_NUM_SCALARS");
__constant__ IwScalarIdx IW_SCALAR_IDX = iw_make_scalar_idx();

// 2^24 fixed-point: integer atomics are associative (nmmo3 embed scatter).
#define IW_FXP 16777216.0f

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

// maps embedding concat column c to its embedding table entry
__device__ __forceinline__ int iw_emb_entry(const precision_t *obs, int c) {
    if (c < IW_WALL_COLS) {
        int slot = c / IW_WALL_DIM, d = c % IW_WALL_DIM;
        int row = iw_type_idx(to_float(obs[iw_wall_slot_idx(slot)]), IW_WALL_VOCAB);
        return row * IW_WALL_DIM + d;
    }
    c -= IW_WALL_COLS;
    if (c < IW_WEAPON_COLS) {
        int slot = c / IW_WEAPON_DIM, d = c % IW_WEAPON_DIM;
        int row = iw_type_idx(to_float(obs[iw_weapon_slot_idx(slot)]), IW_WEAPON_VOCAB);
        return IW_WALL_EMB_N + row * IW_WEAPON_DIM + d;
    }
    c -= IW_WEAPON_COLS;
    int slot = c / IW_RAY_DIM, d = c % IW_RAY_DIM;
    int row = iw_type_idx(to_float(obs[iw_ray_slot_idx(slot)]), IW_RAY_VOCAB);
    return IW_WALL_EMB_N + IW_WEAPON_EMB_N + row * IW_RAY_DIM + d;
}

// reads embedding table entry e from the table it belongs to
__device__ __forceinline__ precision_t iw_emb_value(
    const precision_t *wall_emb, const precision_t *weapon_emb, const precision_t *ray_emb, int e
) {
    if (e < IW_WALL_EMB_N) {
        return wall_emb[e];
    }
    e -= IW_WALL_EMB_N;
    if (e < IW_WEAPON_EMB_N) {
        return weapon_emb[e];
    }
    return ray_emb[e - IW_WEAPON_EMB_N];
}

__global__ void iw_concat_kernel(
    precision_t *__restrict__ concat, const precision_t *__restrict__ obs, const precision_t *__restrict__ conv_out, const precision_t *__restrict__ wall_emb, const precision_t *__restrict__ weapon_emb, const precision_t *__restrict__ ray_emb, int B, int obs_size
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
    } else if (c < IW_CONV_FLAT + IW_EMB_COLS) {
        int e = iw_emb_entry(o, c - IW_CONV_FLAT);
        v = iw_emb_value(wall_emb, weapon_emb, ray_emb, e);
    } else {
        v = o[IW_SCALAR_IDX.idx[c - IW_CONV_FLAT - IW_EMB_COLS]];
    }
    concat[idx] = v;
}

__global__ void iw_fxp_to_precision_kernel(
    precision_t *__restrict__ wall_dst, precision_t *__restrict__ weapon_dst, precision_t *__restrict__ ray_dst, const long long *__restrict__ src
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= IW_EMB_N) {
        return;
    }
    precision_t v = from_float((float)((double)src[idx] * (1.0 / (double)IW_FXP)));
    if (idx < IW_WALL_EMB_N) {
        wall_dst[idx] = v;
    } else if (idx < IW_WALL_EMB_N + IW_WEAPON_EMB_N) {
        weapon_dst[idx - IW_WALL_EMB_N] = v;
    } else {
        ray_dst[idx - IW_WALL_EMB_N - IW_WEAPON_EMB_N] = v;
    }
}

// Per-block int64 histogram over all tables, then one global integer atomic
// per table entry. Integer add is associative so scatter order does not
// change bits.
__global__ void iw_embed_backward_kernel(
    long long *__restrict__ emb_wgrad_i,
    const precision_t *__restrict__ grad_concat,
    const precision_t *__restrict__ obs,
    int B,
    int obs_size
) {
    __shared__ long long acc[IW_EMB_N];
    for (int i = threadIdx.x; i < IW_EMB_N; i += blockDim.x) {
        acc[i] = 0;
    }
    __syncthreads();
    int total = B * IW_EMB_COLS;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        int c = idx % IW_EMB_COLS;
        int b = idx / IW_EMB_COLS;
        long long g = __float2ll_rn(to_float(grad_concat[(int64_t)b * IW_CONCAT + IW_CONV_FLAT + c]) * IW_FXP);
        if (g == 0) {
            continue;
        }
        int e = iw_emb_entry(obs + (int64_t)b * obs_size, c);
        atomicAdd((unsigned long long *)&acc[e], (unsigned long long)g);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < IW_EMB_N; i += blockDim.x) {
        if (acc[i] != 0) {
            atomicAdd((unsigned long long *)&emb_wgrad_i[i], (unsigned long long)acc[i]);
        }
    }
}

struct IwEncoderWeights {
    Prec wall_emb, weapon_emb, ray_emb, conv1_w, conv2_w, conv3_w, proj_w;
    int obs_size, hidden;
};

struct IwEncoderActivations {
    Prec concat, out, saved_obs;
    // per conv: im2col, matmul rows, NCHW ReLU output, output gradient
    Prec col1, mm1, conv1_out, conv1_grad;
    Prec col2, mm2, conv2_out, conv2_grad;
    Prec col3, mm3, conv3_out, conv3_grad;
    Prec wall_emb_wgrad, weapon_emb_wgrad, ray_emb_wgrad;
    Prec conv1_wgrad, conv2_wgrad, conv3_wgrad, proj_wgrad;
    Long emb_wgrad_i;
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

static Prec iw_encoder_forward(
    void *w, void *activations, Prec input, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = input.shape[0];
    if (a->saved_obs.data) {
        puf_copy(&a->saved_obs, &input, stream);
    }

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
        a->concat.data, input.data, a->conv3_out.data, ew->wall_emb.data, ew->weapon_emb.data, ew->ray_emb.data, B, ew->obs_size
    );

    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    return a->out;
}

static void iw_encoder_backward(
    void *w, void *activations, Prec grad, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = grad.shape[0];

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

    cudaMemsetAsync(a->emb_wgrad_i.data, 0, IW_EMB_N * sizeof(long), stream);
    int blocks = (B * IW_EMB_COLS) / BLOCK_SIZE;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 2048) {
        blocks = 2048;
    }
    iw_embed_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        (long long *)a->emb_wgrad_i.data, grad_concat.data, a->saved_obs.data, B, ew->obs_size
    );
    iw_fxp_to_precision_kernel<<<grid_size(IW_EMB_N), BLOCK_SIZE, 0, stream>>>(
        a->wall_emb_wgrad.data, a->weapon_emb_wgrad.data, a->ray_emb_wgrad.data, (long long *)a->emb_wgrad_i.data
    );
}

static void iw_encoder_init_weights(void *w, ulong *seed, cudaStream_t stream) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    puf_normal_init(&ew->wall_emb, 1.0f, (*seed)++, stream);
    puf_normal_init(&ew->weapon_emb, 1.0f, (*seed)++, stream);
    puf_normal_init(&ew->ray_emb, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->conv1_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->conv2_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->conv3_w, 1.0f, (*seed)++, stream);
    Prec pw = {.data = ew->proj_w.data, .shape = {ew->hidden, IW_CONCAT}};
    puf_kaiming_init(&pw, 1.0f, (*seed)++, stream);
}

static void iw_encoder_reg_params(void *w, Allocator *alloc) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    ew->wall_emb = {.shape = {IW_WALL_VOCAB, IW_WALL_DIM}};
    ew->weapon_emb = {.shape = {IW_WEAPON_VOCAB, IW_WEAPON_DIM}};
    ew->ray_emb = {.shape = {IW_RAY_VOCAB, IW_RAY_DIM}};
    ew->conv1_w = {.shape = {IW_C1_OC, IW_C1_COL_W}};
    ew->conv2_w = {.shape = {IW_C2_OC, IW_C2_COL_W}};
    ew->conv3_w = {.shape = {IW_C3_OC, IW_C3_COL_W}};
    ew->proj_w = {.shape = {ew->hidden, IW_CONCAT}};
    // the CPU net (impulse_wars_net.h) loads weights in this order
    alloc_register(alloc, &ew->wall_emb);
    alloc_register(alloc, &ew->weapon_emb);
    alloc_register(alloc, &ew->ray_emb);
    alloc_register(alloc, &ew->conv1_w);
    alloc_register(alloc, &ew->conv2_w);
    alloc_register(alloc, &ew->conv3_w);
    alloc_register(alloc, &ew->proj_w);
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
    a->saved_obs = {.shape = {B_TT, ew->obs_size}};
    alloc_register(acts, &a->concat);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_obs);
    iw_reg_conv_acts(a, acts, B_TT);
    a->conv1_grad = {.shape = {B_TT * IW_C1_OC * IW_C1_SPATIAL}};
    a->conv2_grad = {.shape = {B_TT * IW_C2_OC * IW_C2_SPATIAL}};
    a->conv3_grad = {.shape = {B_TT * IW_CONV_FLAT}};
    alloc_register(acts, &a->conv1_grad);
    alloc_register(acts, &a->conv2_grad);
    alloc_register(acts, &a->conv3_grad);
    a->wall_emb_wgrad = {.shape = {IW_WALL_VOCAB, IW_WALL_DIM}};
    a->weapon_emb_wgrad = {.shape = {IW_WEAPON_VOCAB, IW_WEAPON_DIM}};
    a->ray_emb_wgrad = {.shape = {IW_RAY_VOCAB, IW_RAY_DIM}};
    a->conv1_wgrad = {.shape = {IW_C1_OC, IW_C1_COL_W}};
    a->conv2_wgrad = {.shape = {IW_C2_OC, IW_C2_COL_W}};
    a->conv3_wgrad = {.shape = {IW_C3_OC, IW_C3_COL_W}};
    a->proj_wgrad = {.shape = {ew->hidden, IW_CONCAT}};
    a->emb_wgrad_i = {.shape = {IW_EMB_N}};
    // same order as the params in iw_encoder_reg_params
    alloc_register(grads, &a->wall_emb_wgrad);
    alloc_register(grads, &a->weapon_emb_wgrad);
    alloc_register(grads, &a->ray_emb_wgrad);
    alloc_register(grads, &a->conv1_wgrad);
    alloc_register(grads, &a->conv2_wgrad);
    alloc_register(grads, &a->conv3_wgrad);
    alloc_register(grads, &a->proj_wgrad);
    alloc_register(acts, &a->emb_wgrad_i);
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
