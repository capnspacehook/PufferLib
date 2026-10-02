// Super Mario Land CUDA encoder: tile-class embedding, two strided convs,
// concat with the scalar tail, projection. Included by src/ocean.cu.
//
// Obs is the 16x20 game area of integer tile classes (0..SML_TILE_CLASS_MAX)
// followed by 48 floats (Mario state, then 10 enemies x class/x/y), which are
// copied into the concat as-is.
//
//   ids [B,16,20] -embed-> [B,8,16,20] -conv 4x4 s2-> [B,32,7,9]
//   -conv 3x3 s2-> [B,64,3,4] -> [768 | 48] -proj-> [B,hidden]
//
// Every conv is ReLU'd. Kernel sizes are picked so the unpadded strided convs
// read every input row and column (a 3x3 s2 conv1 would drop the bottom row
// and right column).

static constexpr int SE_H = 16, SE_W = 20;
static constexpr int SE_CELLS = SE_H * SE_W; // 320
static constexpr int SE_NUM_SCALARS = 48;
static constexpr int SE_VOCAB = 45; // classes 0..44
static constexpr int SE_EMB = 4;
static constexpr int SE_EMBED_N = SE_VOCAB * SE_EMB;   // 360
static constexpr int SE_EMB_PLANE = SE_EMB * SE_CELLS; // 2560
static constexpr int SE_C1_K = 4, SE_C1_S = 2, SE_C1_OC = 32;
static constexpr int SE_C1_OH = (SE_H - SE_C1_K) / SE_C1_S + 1; // 7
static constexpr int SE_C1_OW = (SE_W - SE_C1_K) / SE_C1_S + 1; // 9
static constexpr int SE_C1_SPATIAL = SE_C1_OH * SE_C1_OW;       // 63
static constexpr int SE_C1_COL_W = SE_EMB * SE_C1_K * SE_C1_K;  // 128
static constexpr int SE_C1_PLANE = SE_C1_OC * SE_C1_SPATIAL;    // 2016
static constexpr int SE_C2_K = 3, SE_C2_S = 2, SE_C2_OC = 64;
static constexpr int SE_C2_OH = (SE_C1_OH - SE_C2_K) / SE_C2_S + 1; // 3
static constexpr int SE_C2_OW = (SE_C1_OW - SE_C2_K) / SE_C2_S + 1; // 4
static constexpr int SE_C2_SPATIAL = SE_C2_OH * SE_C2_OW;           // 12
static constexpr int SE_C2_COL_W = SE_C1_OC * SE_C2_K * SE_C2_K;    // 288
static constexpr int SE_CONV_FLAT = SE_C2_OC * SE_C2_SPATIAL;       // 768
static constexpr int SE_CONCAT = SE_CONV_FLAT + SE_NUM_SCALARS;     // 816
static_assert(SE_CELLS == SML_GAME_AREA_SIZE, "sml encoder game area mismatch");
static_assert(SE_VOCAB == SML_TILE_CLASS_MAX + 1, "sml encoder vocab mismatch");
static_assert(SE_CELLS + SE_NUM_SCALARS == OBS_SIZE, "sml encoder expects 16x20 tile classes + 48 scalar observations");
static_assert((SE_H - SE_C1_K) % SE_C1_S == 0 && (SE_W - SE_C1_K) % SE_C1_S == 0, "sml conv1 must cover the whole game area");
static_assert((SE_C1_OH - SE_C2_K) % SE_C2_S == 0 && (SE_C1_OW - SE_C2_K) % SE_C2_S == 0, "sml conv2 must cover conv1 output");
// 2^24 fixed-point: integer atomics are associative (nmmo3 embed scatter).
#define SE_FXP 16777216.0f

// out[b, e, cell] = embed_w[id(b, cell), e]; NCHW so conv1 im2col reads it.
__global__ void sml_embed_kernel(
    precision_t *__restrict__ out, const precision_t *__restrict__ obs, const precision_t *__restrict__ embed_w, int B, int obs_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * SE_EMB_PLANE) {
        return;
    }
    int cell = idx % SE_CELLS;
    int e = (idx / SE_CELLS) % SE_EMB;
    int b = idx / SE_EMB_PLANE;
    int id = (int)to_float(obs[(int64_t)b * obs_size + cell]);
    out[idx] = embed_w[id * SE_EMB + e];
}

// col[(b, oh, ow), (ic, kh, kw)] = in[b, ic, oh*S + kh, ow*S + kw]
__global__ void sml_im2col_kernel(
    precision_t *__restrict__ col, const precision_t *__restrict__ in, int B, int C, int H, int W, int K, int S, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_w = C * K * K;
    if (idx >= B * OH * OW * col_w) {
        return;
    }
    int c = idx % col_w;
    int row = idx / col_w;
    int ow = row % OW;
    int oh = (row / OW) % OH;
    int b = row / (OH * OW);
    int ic = c / (K * K);
    int kh = (c / K) % K;
    int kw = c % K;
    col[idx] = in[((int64_t)(b * C + ic) * H + oh * S + kh) * W + ow * S + kw];
}

// Adjoint of sml_im2col_kernel. Gathers per input element instead of
// scattering, so no atomics and the sum order is fixed.
__global__ void sml_col2im_kernel(
    precision_t *__restrict__ grad_in, const precision_t *__restrict__ col, int B, int C, int H, int W, int K, int S, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H * W) {
        return;
    }
    int iw = idx % W;
    int ih = (idx / W) % H;
    int ic = (idx / (H * W)) % C;
    int b = idx / (C * H * W);
    int col_w = C * K * K;
    float sum = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        int dh = ih - kh;
        if (dh < 0 || dh % S != 0 || dh / S >= OH) {
            continue;
        }
        int oh = dh / S;
        for (int kw = 0; kw < K; kw++) {
            int dw = iw - kw;
            if (dw < 0 || dw % S != 0 || dw / S >= OW) {
                continue;
            }
            int ow = dw / S;
            sum += to_float(col[((int64_t)(b * OH + oh) * OW + ow) * col_w + (ic * K + kh) * K + kw]);
        }
    }
    grad_in[idx] = from_float(sum);
}

__global__ void sml_rows_to_nchw_relu_kernel(
    precision_t *__restrict__ dst, const precision_t *__restrict__ src, int B, int OC, int spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * OC * spatial) {
        return;
    }
    int b = idx / (OC * spatial);
    int oc = (idx / spatial) % OC;
    int s = idx % spatial;
    dst[idx] = from_float(fmaxf(0.0f, to_float(src[((int64_t)b * spatial + s) * OC + oc])));
}

__global__ void sml_nchw_to_rows_kernel(
    precision_t *__restrict__ dst, const precision_t *__restrict__ src, int B, int OC, int spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * OC * spatial) {
        return;
    }
    int b = idx / (OC * spatial);
    int oc = (idx / spatial) % OC;
    int s = idx % spatial;
    dst[((int64_t)b * spatial + s) * OC + oc] = src[idx];
}

__global__ void sml_concat_kernel(
    precision_t *__restrict__ concat, const precision_t *__restrict__ conv_flat, const precision_t *__restrict__ obs, int B, int obs_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * SE_CONCAT) {
        return;
    }
    int c = idx % SE_CONCAT;
    int b = idx / SE_CONCAT;
    concat[idx] = c < SE_CONV_FLAT
                      ? conv_flat[(int64_t)b * SE_CONV_FLAT + c]
                      : obs[(int64_t)b * obs_size + SE_CELLS + (c - SE_CONV_FLAT)];
}

// Conv slice of grad_concat, masked by conv2's ReLU.
__global__ void sml_concat_backward_kernel(
    precision_t *__restrict__ conv_grad, const precision_t *__restrict__ grad_concat, const precision_t *__restrict__ conv_out, int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * SE_CONV_FLAT) {
        return;
    }
    int c = idx % SE_CONV_FLAT;
    int b = idx / SE_CONV_FLAT;
    conv_grad[idx] = to_float(conv_out[idx]) > 0.0f
                         ? grad_concat[(int64_t)b * SE_CONCAT + c]
                         : from_float(0.0f);
}

__global__ void sml_relu_kernel(precision_t *__restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx])));
}

__global__ void sml_relu_backward_kernel(
    precision_t *__restrict__ grad, const precision_t *__restrict__ out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (to_float(out[idx]) <= 0.0f) {
        grad[idx] = from_float(0.0f);
    }
}

// Per-block int64 histogram over the 45x8 table, then one global integer
// atomic per table entry. Integer add is associative so scatter order does
// not change bits. e varies fastest so a warp's atomics spread over 8 table
// entries even when its cells share a class (mostly empty tiles).
__global__ void sml_embed_backward_kernel(
    long long *__restrict__ embed_wgrad_i, const precision_t *__restrict__ grad_emb, const precision_t *__restrict__ obs, int B, int obs_size
) {
    __shared__ long long acc[SE_EMBED_N];
    for (int i = threadIdx.x; i < SE_EMBED_N; i += blockDim.x) {
        acc[i] = 0;
    }
    __syncthreads();
    int total = B * SE_EMB_PLANE;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        int e = idx % SE_EMB;
        int cell = (idx / SE_EMB) % SE_CELLS;
        int b = idx / SE_EMB_PLANE;
        long long g = __float2ll_rn(to_float(grad_emb[(int64_t)b * SE_EMB_PLANE + e * SE_CELLS + cell]) * SE_FXP);
        if (g == 0) {
            continue;
        }
        int id = (int)to_float(obs[(int64_t)b * obs_size + cell]);
        atomicAdd((unsigned long long *)&acc[id * SE_EMB + e], (unsigned long long)g);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < SE_EMBED_N; i += blockDim.x) {
        if (acc[i] != 0) {
            atomicAdd((unsigned long long *)&embed_wgrad_i[i], (unsigned long long)acc[i]);
        }
    }
}

__global__ void sml_fxp_to_precision_kernel(
    precision_t *__restrict__ dst, const long long *__restrict__ src, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)((double)src[idx] * (1.0 / (double)SE_FXP)));
    }
}

// im2col -> GEMM -> NCHW + ReLU. col_buf and mm_buf keep the forward values
// for the backward pass.
static void sml_conv_forward(
    precision_t *in, precision_t *col_buf, precision_t *mm_buf, Prec *w, precision_t *out, int B, int C, int H, int W, int K, int S, int OC, int OH, int OW, cudaStream_t stream
) {
    int rows = B * OH * OW, col_w = C * K * K;
    sml_im2col_kernel<<<grid_size(rows * col_w), BLOCK_SIZE, 0, stream>>>(
        col_buf, in, B, C, H, W, K, S, OH, OW
    );
    Prec col = {.data = col_buf, .shape = {rows, col_w}};
    Prec mm = {.data = mm_buf, .shape = {rows, OC}};
    puf_mm(&col, w, &mm, stream);
    sml_rows_to_nchw_relu_kernel<<<grid_size(B * OC * OH * OW), BLOCK_SIZE, 0, stream>>>(
        out, mm_buf, B, OC, OH * OW
    );
}

// grad_out is NCHW and already ReLU-masked. col_buf must still hold the
// forward im2col; it is overwritten with the column gradient.
static void sml_conv_backward(
    precision_t *grad_out, precision_t *col_buf, precision_t *mm_buf, Prec *w, Prec *wgrad, precision_t *grad_in, int B, int C, int H, int W, int K, int S, int OC, int OH, int OW, cudaStream_t stream
) {
    int rows = B * OH * OW, col_w = C * K * K;
    sml_nchw_to_rows_kernel<<<grid_size(B * OC * OH * OW), BLOCK_SIZE, 0, stream>>>(
        mm_buf, grad_out, B, OC, OH * OW
    );
    Prec mm = {.data = mm_buf, .shape = {rows, OC}};
    Prec col = {.data = col_buf, .shape = {rows, col_w}};
    puf_mm_tn(&mm, &col, wgrad, stream);
    puf_mm_nn(&mm, w, &col, stream);
    sml_col2im_kernel<<<grid_size(B * C * H * W), BLOCK_SIZE, 0, stream>>>(
        grad_in, col_buf, B, C, H, W, K, S, OH, OW
    );
}

struct SmlEncoderWeights {
    Prec embed_w, conv1_w, conv2_w, proj_w;
    int obs_size, hidden;
};

struct SmlEncoderActivations {
    Prec embed_out, col1, mm1, conv1_out, conv1_grad;
    Prec col2, mm2, conv2_out, conv2_grad;
    Prec concat, out, saved_obs;
    Prec embed_wgrad, conv1_wgrad, conv2_wgrad, proj_wgrad;
    Long embed_wgrad_i;
};

static Prec sml_encoder_forward(
    void *w, void *activations, Prec input, cudaStream_t stream
) {
    SmlEncoderWeights *ew = (SmlEncoderWeights *)w;
    SmlEncoderActivations *a = (SmlEncoderActivations *)activations;
    int B = input.shape[0];
    if (a->saved_obs.data) {
        puf_copy(&a->saved_obs, &input, stream);
    }

    sml_embed_kernel<<<grid_size(B * SE_EMB_PLANE), BLOCK_SIZE, 0, stream>>>(
        a->embed_out.data, input.data, ew->embed_w.data, B, ew->obs_size
    );
    sml_conv_forward(a->embed_out.data, a->col1.data, a->mm1.data, &ew->conv1_w, a->conv1_out.data, B, SE_EMB, SE_H, SE_W, SE_C1_K, SE_C1_S, SE_C1_OC, SE_C1_OH, SE_C1_OW, stream);
    sml_conv_forward(a->conv1_out.data, a->col2.data, a->mm2.data, &ew->conv2_w, a->conv2_out.data, B, SE_C1_OC, SE_C1_OH, SE_C1_OW, SE_C2_K, SE_C2_S, SE_C2_OC, SE_C2_OH, SE_C2_OW, stream);
    sml_concat_kernel<<<grid_size(B * SE_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->conv2_out.data, input.data, B, ew->obs_size
    );

    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    sml_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, B * ew->hidden
    );
    return a->out;
}

static void sml_encoder_backward(
    void *w, void *activations, Prec grad, cudaStream_t stream
) {
    SmlEncoderWeights *ew = (SmlEncoderWeights *)w;
    SmlEncoderActivations *a = (SmlEncoderActivations *)activations;
    int B = grad.shape[0];

    sml_relu_backward_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        grad.data, a->out.data, B * ew->hidden
    );
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);
    Prec grad_concat = {.data = a->concat.data, .shape = {B, SE_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);
    sml_concat_backward_kernel<<<grid_size(B * SE_CONV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->conv2_grad.data, grad_concat.data, a->conv2_out.data, B
    );

    sml_conv_backward(a->conv2_grad.data, a->col2.data, a->mm2.data, &ew->conv2_w, &a->conv2_wgrad, a->conv1_grad.data, B, SE_C1_OC, SE_C1_OH, SE_C1_OW, SE_C2_K, SE_C2_S, SE_C2_OC, SE_C2_OH, SE_C2_OW, stream);
    sml_relu_backward_kernel<<<grid_size(B * SE_C1_PLANE), BLOCK_SIZE, 0, stream>>>(
        a->conv1_grad.data, a->conv1_out.data, B * SE_C1_PLANE
    );
    // embed_out is dead after conv1's im2col; reuse it for the embed grad.
    sml_conv_backward(a->conv1_grad.data, a->col1.data, a->mm1.data, &ew->conv1_w, &a->conv1_wgrad, a->embed_out.data, B, SE_EMB, SE_H, SE_W, SE_C1_K, SE_C1_S, SE_C1_OC, SE_C1_OH, SE_C1_OW, stream);

    cudaMemsetAsync(a->embed_wgrad_i.data, 0, SE_EMBED_N * sizeof(long), stream);
    int blocks = (B * SE_EMB_PLANE) / BLOCK_SIZE;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 2048) {
        blocks = 2048;
    }
    sml_embed_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        (long long *)a->embed_wgrad_i.data, a->embed_out.data, a->saved_obs.data, B, ew->obs_size
    );
    sml_fxp_to_precision_kernel<<<grid_size(SE_EMBED_N), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, (long long *)a->embed_wgrad_i.data, SE_EMBED_N
    );
}

static void sml_encoder_init_weights(void *w, ulong *seed, cudaStream_t stream) {
    SmlEncoderWeights *ew = (SmlEncoderWeights *)w;
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->conv1_w, sqrtf(2.0f), (*seed)++, stream);
    puf_kaiming_init(&ew->conv2_w, sqrtf(2.0f), (*seed)++, stream);
    puf_kaiming_init(&ew->proj_w, sqrtf(2.0f), (*seed)++, stream);
}

// Grads must be registered in the same order and shapes as params.
static void sml_encoder_reg_params(void *w, Allocator *alloc) {
    SmlEncoderWeights *ew = (SmlEncoderWeights *)w;
    ew->embed_w = {.shape = {SE_VOCAB, SE_EMB}};
    ew->conv1_w = {.shape = {SE_C1_OC, SE_C1_COL_W}};
    ew->conv2_w = {.shape = {SE_C2_OC, SE_C2_COL_W}};
    ew->proj_w = {.shape = {ew->hidden, SE_CONCAT}};
    alloc_register(alloc, &ew->embed_w);
    alloc_register(alloc, &ew->conv1_w);
    alloc_register(alloc, &ew->conv2_w);
    alloc_register(alloc, &ew->proj_w);
}

static void sml_encoder_reg_train(
    void *w, void *activations, Allocator *acts, Allocator *grads, int B_TT
) {
    SmlEncoderWeights *ew = (SmlEncoderWeights *)w;
    SmlEncoderActivations *a = (SmlEncoderActivations *)activations;
    *a = {};
    a->embed_out = {.shape = {B_TT * SE_EMB_PLANE}};
    a->col1 = {.shape = {B_TT * SE_C1_SPATIAL, SE_C1_COL_W}};
    a->mm1 = {.shape = {B_TT * SE_C1_SPATIAL, SE_C1_OC}};
    a->conv1_out = {.shape = {B_TT * SE_C1_PLANE}};
    a->conv1_grad = {.shape = {B_TT * SE_C1_PLANE}};
    a->col2 = {.shape = {B_TT * SE_C2_SPATIAL, SE_C2_COL_W}};
    a->mm2 = {.shape = {B_TT * SE_C2_SPATIAL, SE_C2_OC}};
    a->conv2_out = {.shape = {B_TT * SE_CONV_FLAT}};
    a->conv2_grad = {.shape = {B_TT * SE_CONV_FLAT}};
    a->concat = {.shape = {B_TT, SE_CONCAT}};
    a->out = {.shape = {B_TT, ew->hidden}};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}};
    alloc_register(acts, &a->embed_out);
    alloc_register(acts, &a->col1);
    alloc_register(acts, &a->mm1);
    alloc_register(acts, &a->conv1_out);
    alloc_register(acts, &a->conv1_grad);
    alloc_register(acts, &a->col2);
    alloc_register(acts, &a->mm2);
    alloc_register(acts, &a->conv2_out);
    alloc_register(acts, &a->conv2_grad);
    alloc_register(acts, &a->concat);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_obs);
    a->embed_wgrad = {.shape = {SE_VOCAB, SE_EMB}};
    a->embed_wgrad_i = {.shape = {SE_VOCAB, SE_EMB}};
    a->conv1_wgrad = {.shape = {SE_C1_OC, SE_C1_COL_W}};
    a->conv2_wgrad = {.shape = {SE_C2_OC, SE_C2_COL_W}};
    a->proj_wgrad = {.shape = {ew->hidden, SE_CONCAT}};
    alloc_register(grads, &a->embed_wgrad);
    alloc_register(acts, &a->embed_wgrad_i);
    alloc_register(grads, &a->conv1_wgrad);
    alloc_register(grads, &a->conv2_wgrad);
    alloc_register(grads, &a->proj_wgrad);
}

static void sml_encoder_reg_rollout(
    void *w, void *activations, Allocator *alloc, int B
) {
    SmlEncoderWeights *ew = (SmlEncoderWeights *)w;
    SmlEncoderActivations *a = (SmlEncoderActivations *)activations;
    *a = {};
    a->embed_out = {.shape = {B * SE_EMB_PLANE}};
    a->col1 = {.shape = {B * SE_C1_SPATIAL, SE_C1_COL_W}};
    a->mm1 = {.shape = {B * SE_C1_SPATIAL, SE_C1_OC}};
    a->conv1_out = {.shape = {B * SE_C1_PLANE}};
    a->col2 = {.shape = {B * SE_C2_SPATIAL, SE_C2_COL_W}};
    a->mm2 = {.shape = {B * SE_C2_SPATIAL, SE_C2_OC}};
    a->conv2_out = {.shape = {B * SE_CONV_FLAT}};
    a->concat = {.shape = {B, SE_CONCAT}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->embed_out);
    alloc_register(alloc, &a->col1);
    alloc_register(alloc, &a->mm1);
    alloc_register(alloc, &a->conv1_out);
    alloc_register(alloc, &a->col2);
    alloc_register(alloc, &a->mm2);
    alloc_register(alloc, &a->conv2_out);
    alloc_register(alloc, &a->concat);
    alloc_register(alloc, &a->out);
}

static void *sml_encoder_create_weights(void *self) {
    Encoder *e = (Encoder *)self;
    SmlEncoderWeights *ew = (SmlEncoderWeights *)calloc(1, sizeof(SmlEncoderWeights));
    ew->obs_size = e->in_dim;
    ew->hidden = e->out_dim;
    return ew;
}

static void create_sml_encoder(Encoder *enc) {
    fprintf(stderr, "super_mario_land: using embed+conv encoder (obs=%d hidden=%d concat=%d)\n", enc->in_dim, enc->out_dim, SE_CONCAT);
    *enc = Encoder{
        .forward = sml_encoder_forward,
        .backward = sml_encoder_backward,
        .init_weights = sml_encoder_init_weights,
        .reg_params = sml_encoder_reg_params,
        .reg_train = sml_encoder_reg_train,
        .reg_rollout = sml_encoder_reg_rollout,
        .create_weights = sml_encoder_create_weights,
        .in_dim = enc->in_dim,
        .out_dim = enc->out_dim,
        .activation_size = sizeof(SmlEncoderActivations),
    };
}
