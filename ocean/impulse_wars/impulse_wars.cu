// Impulse Wars CUDA encoder: map grid CNN, one-hot type slots, concat with
// the continuous obs, projection.
// The map grid's packed cell codes are unpacked into one-hot channels by
// conv1's im2col and run through 3 convs with ReLU, then concatenated and
// projected through a GELU. Conv activations stay NHWC, as the matmul rows,
// and are kept before ReLU: each consumer applies it when reading. Wall,
// pickup, weapon, utility, ray hit and projectile type slots are one-hot
// encoded into the concat; every other obs value (including the floating
// wall and projectile slots' fields) goes in as a scalar. Included by
// src/ocean.cu.

#include "impulse_wars_obs.h"

static_assert(OBS_SIZE >= IW_BASE_OBS_SIZE + IW_ENEMY_DRONE_SIZE && (OBS_SIZE - IW_BASE_OBS_SIZE) % IW_ENEMY_DRONE_SIZE == 0, "impulse_wars encoder obs layout does not match OBS_SIZE");
static_assert(IW_GRID_CODE_MAX < 256, "impulse_wars map grid codes must be exact in bf16");
static_assert(IW_C3_OUT == 3, "impulse_wars CNN output should keep a 3x3 layout");

// most blocks of the rollout conv1, whose warps loop over output
// positions, and output channels per lane
#define IW_C1_DIRECT_MAX_BLOCKS 1024
#define IW_C1_DIRECT_OC 8
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

// a pair of adjacent elements, and IW_VEC elements in 16 bytes, for wider
// loads and stores
struct alignas(2 * sizeof(precision_t)) IwPair {
    precision_t v[2];
};
#define IW_VEC ((int)(16 / sizeof(precision_t)))
struct alignas(16) IwVec {
    precision_t v[IW_VEC];
};

static_assert(IW_C1_IC % 2 == 0, "impulse_wars conv1 im2col writes column pairs within a tap");

// conv1 im2col straight from the obs cell codes, skipping the dense one-hot
// planes: (B * IW_C1_SPATIAL, IW_C1_COL_W). Each thread writes the columns of
// one tap, the channels of the cell it reads, in pairs
__global__ void iw_c1_im2col_obs(
    precision_t *__restrict__ col, const precision_t *__restrict__ obs, int B, int obs_size
) {
    constexpr int TAPS = IW_C1_K * IW_C1_K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_C1_SPATIAL * TAPS) {
        return;
    }
    int row = idx / TAPS;
    int c0 = idx % TAPS * IW_C1_IC;
    int cell = iw_c1_cell(row % IW_C1_SPATIAL, c0);
    float v = cell < 0 ? 0.0f : to_float(obs[(int64_t)(row / IW_C1_SPATIAL) * obs_size + cell]);
    IwPair *dst = (IwPair *)(col + (int64_t)row * IW_C1_COL_W + c0);
#pragma unroll
    for (int i = 0; i < IW_C1_IC / 2; i++) {
        IwPair p;
        p.v[0] = from_float(iw_c1_col_value(v, cell, c0 + 2 * i));
        p.v[1] = from_float(iw_c1_col_value(v, cell, c0 + 2 * i + 1));
        dst[i] = p;
    }
}

static_assert(IW_C1_OC % IW_C1_DIRECT_OC == 0 && 32 % (IW_C1_OC / IW_C1_DIRECT_OC) == 0 && IW_C1_DIRECT_OC % IW_VEC == 0,
    "impulse_wars rollout conv1 lanes take whole vectors of output channels");

// rollout conv1 straight from the obs cell codes, without the im2col: NHWC
// (B * IW_C1_SPATIAL, IW_C1_OC). Each lane takes IW_C1_DIRECT_OC output
// channels of an output position, and warps loop over the positions. Each
// output is the sum of the weights its cells' one-hot channels pick, in
// im2col column order, which matches the im2col matmul unless fp32 rounds
// the sum
__global__ void iw_c1_direct(
    precision_t *__restrict__ z, const precision_t *__restrict__ obs, const precision_t *__restrict__ w, int B, int obs_size
) {
    constexpr int LANES = IW_C1_OC / IW_C1_DIRECT_OC;
    __shared__ __align__(16) float sw[IW_C1_COL_W][IW_C1_OC];
    for (int i = threadIdx.x; i < IW_C1_OC * IW_C1_COL_W; i += blockDim.x) {
        sw[i % IW_C1_COL_W][i / IW_C1_COL_W] = to_float(w[i]);
    }
    __syncthreads();
    int oc0 = threadIdx.x % LANES * IW_C1_DIRECT_OC;
    int stride = gridDim.x * blockDim.x / LANES;
    for (int row = (blockIdx.x * blockDim.x + threadIdx.x) / LANES; row < B * IW_C1_SPATIAL; row += stride) {
        int s = row % IW_C1_SPATIAL;
        const precision_t *o = obs + (int64_t)(row / IW_C1_SPATIAL) * obs_size;
        float sum[IW_C1_DIRECT_OC] = {};
#pragma unroll
        for (int tap = 0; tap < IW_C1_K * IW_C1_K; tap++) {
            int cell = iw_c1_cell(s, tap * IW_C1_IC);
            if (cell < 0) {
                continue;
            }
            float v = to_float(o[cell]);
#pragma unroll
            for (int ch = 0; ch < IW_C1_IC; ch++) {
                if (iw_grid_channel(v, ch) != 0.0f) {
                    const float *wr = &sw[tap * IW_C1_IC + ch][oc0];
#pragma unroll
                    for (int j = 0; j < IW_C1_DIRECT_OC; j++) {
                        sum[j] += wr[j];
                    }
                }
            }
        }
#pragma unroll
        for (int q = 0; q < IW_C1_DIRECT_OC / IW_VEC; q++) {
            IwVec x;
#pragma unroll
            for (int j = 0; j < IW_VEC; j++) {
                x.v[j] = from_float(sum[q * IW_VEC + j]);
            }
            *(IwVec *)(z + (int64_t)row * IW_C1_OC + oc0 + q * IW_VEC) = x;
        }
    }
}

// im2col of an unpadded square conv over the NHWC (B, IN, IN, IC) output of
// the previous conv before its ReLU, applying the ReLU:
// (B * OUT * OUT, K * K * IC). Each thread copies IW_VEC channels
template <int IC, int IN, int OUT, int K, int S>
__global__ void iw_im2col(precision_t *__restrict__ col, const precision_t *__restrict__ z, int B) {
    static_assert(IC % IW_VEC == 0, "impulse_wars im2col copies whole vectors of channels");
    constexpr int colW = K * K * IC;
    constexpr int spatial = OUT * OUT;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * spatial * colW / IW_VEC) {
        return;
    }
    int e = idx * IW_VEC;
    int row = e / colW;
    int b = row / spatial;
    IwVec x = *(const IwVec *)(z + (int64_t)b * IN * IN * IC + iw_im2col_src(row % spatial, e % colW, IC, IN, OUT, K, S));
#pragma unroll
    for (int j = 0; j < IW_VEC; j++) {
        x.v[j] = from_float(fmaxf(to_float(x.v[j]), 0.0f));
    }
    ((IwVec *)col)[idx] = x;
}

// gradient at the NHWC (B, IN, IN, IC) pre-ReLU input z of an unpadded
// square conv from the gradient of its im2col. Each thread sums IW_VEC
// channels
template <int IC, int IN, int OUT, int K, int S>
__global__ void iw_col2im(
    precision_t *__restrict__ grad_z, const precision_t *__restrict__ col, const precision_t *__restrict__ z, int B
) {
    static_assert(IC % IW_VEC == 0, "impulse_wars col2im sums whole vectors of channels");
    constexpr int colW = K * K * IC;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IN * IN * IC / IW_VEC) {
        return;
    }
    int e = idx * IW_VEC;
    int c = e % IC;
    int pix = e / IC;
    int iw = pix % IN;
    int ih = (pix / IN) % IN;
    int b = pix / (IN * IN);
    float sum[IW_VEC] = {};
#pragma unroll
    for (int kh = 0; kh < K; kh++) {
        int ohs = ih - kh;
        if (ohs < 0 || ohs % S != 0 || ohs / S >= OUT) {
            continue;
        }
#pragma unroll
        for (int kw = 0; kw < K; kw++) {
            int ows = iw - kw;
            if (ows < 0 || ows % S != 0 || ows / S >= OUT) {
                continue;
            }
            int64_t r = (int64_t)b * OUT * OUT + (ohs / S) * OUT + ows / S;
            IwVec g = *(const IwVec *)(col + r * colW + (kh * K + kw) * IC + c);
#pragma unroll
            for (int j = 0; j < IW_VEC; j++) {
                sum[j] += to_float(g.v[j]);
            }
        }
    }
    IwVec zv = ((const IwVec *)z)[idx];
    IwVec out;
#pragma unroll
    for (int j = 0; j < IW_VEC; j++) {
        out.v[j] = to_float(zv.v[j]) <= 0.0f ? from_float(0.0f) : from_float(sum[j]);
    }
    ((IwVec *)grad_z)[idx] = out;
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
    } else {
        v = from_float(0.0f);
    }
    concat[idx] = v;
}

__global__ void iw_gelu_kernel(precision_t *__restrict__ out, const precision_t *__restrict__ in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out[idx] = from_float(iw_gelu(to_float(in[idx])));
}

// overwrites z, the input of a GELU, with the gradient at it given the
// gradient at the GELU's output
__global__ void iw_gelu_backward_kernel(precision_t *__restrict__ z, const precision_t *__restrict__ grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    z[idx] = from_float(to_float(grad[idx]) * iw_gelu_grad(to_float(z[idx])));
}

struct IwEncoderWeights {
    Prec conv1_w, conv2_w, conv3_w, proj_w;
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
    Prec conv1_wgrad, conv2_wgrad, conv3_wgrad, proj_wgrad;
};

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

static Prec iw_encoder_forward(
    void *w, void *activations, Prec input, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = input.shape[0];

    // rollout has no conv1 im2col: conv1 straight from the obs
    if (a->col1.data) {
        iw_c1_im2col_obs<<<grid_size(B * IW_C1_SPATIAL * IW_C1_K * IW_C1_K), BLOCK_SIZE, 0, stream>>>(
            a->col1.data, input.data, B, ew->obs_size
        );
        iw_conv_mm(a->col1.data, &ew->conv1_w, a->conv1_z.data, B, IW_C1_OC, IW_C1_SPATIAL, IW_C1_COL_W, stream);
    } else {
        int lanes = B * IW_C1_SPATIAL * (IW_C1_OC / IW_C1_DIRECT_OC);
        int blocks = min((lanes + BLOCK_SIZE - 1) / BLOCK_SIZE, IW_C1_DIRECT_MAX_BLOCKS);
        iw_c1_direct<<<blocks, BLOCK_SIZE, 0, stream>>>(a->conv1_z.data, input.data, ew->conv1_w.data, B, ew->obs_size);
    }
    iw_im2col<IW_C2_IC, IW_C2_IN, IW_C2_OUT, IW_C2_K, IW_C2_S><<<grid_size(B * IW_C2_SPATIAL * IW_C2_COL_W / IW_VEC), BLOCK_SIZE, 0, stream>>>(
        a->col2.data, a->conv1_z.data, B
    );
    iw_conv_mm(a->col2.data, &ew->conv2_w, a->conv2_z.data, B, IW_C2_OC, IW_C2_SPATIAL, IW_C2_COL_W, stream);
    iw_im2col<IW_C3_IC, IW_C3_IN, IW_C3_OUT, IW_C3_K, IW_C3_S><<<grid_size(B * IW_C3_SPATIAL * IW_C3_COL_W / IW_VEC), BLOCK_SIZE, 0, stream>>>(
        a->col3.data, a->conv2_z.data, B
    );
    iw_conv_mm(a->col3.data, &ew->conv3_w, a->conv3_z.data, B, IW_C3_OC, IW_C3_SPATIAL, IW_C3_COL_W, stream);

    iw_concat_kernel<<<grid_size(B * IW_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, input.data, a->conv3_z.data, B, ew->obs_size
    );
    puf_mm(&a->concat, &ew->proj_w, &a->proj_z, stream);
    int n = B * ew->hidden;
    iw_gelu_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->out.data, a->proj_z.data, n);
    return a->out;
}

static void iw_encoder_backward(
    void *w, void *activations, Prec grad, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = grad.shape[0];

    int n = B * ew->hidden;
    iw_gelu_backward_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->proj_z.data, grad.data, n);
    Prec grad_proj = {.data = a->proj_z.data, .shape = {B, ew->hidden}};
    puf_mm_tn(&grad_proj, &a->concat, &a->proj_wgrad, stream);
    Prec grad_concat = {.data = a->concat.data, .shape = {B, IW_CONCAT}};
    puf_mm_nn(&grad_proj, &ew->proj_w, &grad_concat, stream);

    // CNN, back to front; conv1's input isn't learned so it gets no input grad
    iw_conv_grad_from_concat<<<grid_size(B * IW_CONV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->conv3_grad.data, grad_concat.data, a->conv3_z.data, B
    );
    iw_conv_backward(a->conv3_grad.data, &ew->conv3_w, a->conv3_wgrad.data, a->col3.data, B, IW_C3_OC, IW_C3_SPATIAL, IW_C3_COL_W, stream);
    iw_col2im<IW_C3_IC, IW_C3_IN, IW_C3_OUT, IW_C3_K, IW_C3_S><<<grid_size(B * IW_C2_SPATIAL * IW_C2_OC / IW_VEC), BLOCK_SIZE, 0, stream>>>(
        a->conv2_grad.data, a->col3.data, a->conv2_z.data, B
    );
    iw_conv_backward(a->conv2_grad.data, &ew->conv2_w, a->conv2_wgrad.data, a->col2.data, B, IW_C2_OC, IW_C2_SPATIAL, IW_C2_COL_W, stream);
    iw_col2im<IW_C2_IC, IW_C2_IN, IW_C2_OUT, IW_C2_K, IW_C2_S><<<grid_size(B * IW_C1_SPATIAL * IW_C1_OC / IW_VEC), BLOCK_SIZE, 0, stream>>>(
        a->conv1_grad.data, a->col2.data, a->conv1_z.data, B
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
}

static void iw_encoder_reg_params(void *w, Allocator *alloc) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    // conv weights are (OC, K, K, IC), see IW_C1_COL_W
    ew->conv1_w = {.shape = {IW_C1_OC, IW_C1_COL_W}};
    ew->conv2_w = {.shape = {IW_C2_OC, IW_C2_COL_W}};
    ew->conv3_w = {.shape = {IW_C3_OC, IW_C3_COL_W}};
    ew->proj_w = {.shape = {ew->hidden, IW_CONCAT}};
    // the CPU net (impulse_wars_net.h) loads weights in this order
    alloc_register(alloc, &ew->conv1_w);
    alloc_register(alloc, &ew->conv2_w);
    alloc_register(alloc, &ew->conv3_w);
    alloc_register(alloc, &ew->proj_w);
}

// forward CNN buffers; rollout has no conv1 im2col
static void iw_reg_conv_acts(IwEncoderActivations *a, Allocator *alloc, int B, bool train) {
    if (train) {
        a->col1 = {.shape = {B * IW_C1_SPATIAL, IW_C1_COL_W}};
        alloc_register(alloc, &a->col1);
    }
    a->conv1_z = {.shape = {B * IW_C1_SPATIAL, IW_C1_OC}};
    a->col2 = {.shape = {B * IW_C2_SPATIAL, IW_C2_COL_W}};
    a->conv2_z = {.shape = {B * IW_C2_SPATIAL, IW_C2_OC}};
    a->col3 = {.shape = {B * IW_C3_SPATIAL, IW_C3_COL_W}};
    a->conv3_z = {.shape = {B * IW_C3_SPATIAL, IW_C3_OC}};
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
    iw_reg_conv_acts(a, acts, B_TT, true);
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
    // same order as the params in iw_encoder_reg_params
    alloc_register(grads, &a->conv1_wgrad);
    alloc_register(grads, &a->conv2_wgrad);
    alloc_register(grads, &a->conv3_wgrad);
    alloc_register(grads, &a->proj_wgrad);
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
    iw_reg_conv_acts(a, alloc, B, false);
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
