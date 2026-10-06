// Impulse Wars CUDA encoder: wall type and weapon type embeddings, concat with
// the continuous obs, projection. Wall types in the map grid, near walls and
// floating walls share one table; weapon types of pickups, projectiles, enemy
// drones and the agent's drone share another. Included by src/ocean.cu.

#include "impulse_wars_obs.h"

static_assert(OBS_SIZE >= IW_BASE_OBS_SIZE + IW_ENEMY_DRONE_SIZE && (OBS_SIZE - IW_BASE_OBS_SIZE) % IW_ENEMY_DRONE_SIZE == 0, "impulse_wars encoder obs layout does not match OBS_SIZE");
static_assert(IW_EMB_N <= 1024, "impulse_wars embed tables too big for shared memory");

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

// maps embedding concat column c to its embedding table entry
__device__ __forceinline__ int iw_emb_entry(const precision_t *obs, int c) {
    if (c < IW_WALL_COLS) {
        int slot = c / IW_WALL_DIM, d = c % IW_WALL_DIM;
        int row = iw_type_idx(to_float(obs[iw_wall_slot_idx(slot)]), IW_WALL_VOCAB);
        return row * IW_WALL_DIM + d;
    }
    c -= IW_WALL_COLS;
    int slot = c / IW_WEAPON_DIM, d = c % IW_WEAPON_DIM;
    int row = iw_type_idx(to_float(obs[iw_weapon_slot_idx(slot)]), IW_WEAPON_VOCAB);
    return IW_WALL_EMB_N + row * IW_WEAPON_DIM + d;
}

__global__ void iw_concat_kernel(
    precision_t *__restrict__ concat, const precision_t *__restrict__ obs, const precision_t *__restrict__ wall_emb, const precision_t *__restrict__ weapon_emb, int B, int obs_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_CONCAT) {
        return;
    }
    int c = idx % IW_CONCAT;
    int b = idx / IW_CONCAT;
    const precision_t *o = obs + (int64_t)b * obs_size;
    precision_t v;
    if (c < IW_EMB_COLS) {
        int e = iw_emb_entry(o, c);
        v = e < IW_WALL_EMB_N ? wall_emb[e] : weapon_emb[e - IW_WALL_EMB_N];
    } else {
        v = o[IW_SCALAR_IDX.idx[c - IW_EMB_COLS]];
    }
    concat[idx] = v;
}

__global__ void iw_fxp_to_precision_kernel(
    precision_t *__restrict__ wall_dst, precision_t *__restrict__ weapon_dst, const long long *__restrict__ src
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= IW_EMB_N) {
        return;
    }
    precision_t v = from_float((float)((double)src[idx] * (1.0 / (double)IW_FXP)));
    if (idx < IW_WALL_EMB_N) {
        wall_dst[idx] = v;
    } else {
        weapon_dst[idx - IW_WALL_EMB_N] = v;
    }
}

// Per-block int64 histogram over both tables, then one global integer atomic
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
        long long g = __float2ll_rn(to_float(grad_concat[(int64_t)b * IW_CONCAT + c]) * IW_FXP);
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
    Prec wall_emb, weapon_emb, proj_w;
    int obs_size, hidden;
};

struct IwEncoderActivations {
    Prec concat, out, saved_obs;
    Prec wall_emb_wgrad, weapon_emb_wgrad, proj_wgrad;
    Long emb_wgrad_i;
};

static Prec iw_encoder_forward(
    void *w, void *activations, Prec input, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = input.shape[0];
    if (a->saved_obs.data) {
        puf_copy(&a->saved_obs, &input, stream);
    }

    iw_concat_kernel<<<grid_size(B * IW_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, input.data, ew->wall_emb.data, ew->weapon_emb.data, B, ew->obs_size
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
        a->wall_emb_wgrad.data, a->weapon_emb_wgrad.data, (long long *)a->emb_wgrad_i.data
    );
}

static void iw_encoder_init_weights(void *w, ulong *seed, cudaStream_t stream) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    puf_normal_init(&ew->wall_emb, 1.0f, (*seed)++, stream);
    puf_normal_init(&ew->weapon_emb, 1.0f, (*seed)++, stream);
    Prec pw = {.data = ew->proj_w.data, .shape = {ew->hidden, IW_CONCAT}};
    puf_kaiming_init(&pw, 1.0f, (*seed)++, stream);
}

static void iw_encoder_reg_params(void *w, Allocator *alloc) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    ew->wall_emb = {.shape = {IW_WALL_VOCAB, IW_WALL_DIM}};
    ew->weapon_emb = {.shape = {IW_WEAPON_VOCAB, IW_WEAPON_DIM}};
    ew->proj_w = {.shape = {ew->hidden, IW_CONCAT}};
    alloc_register(alloc, &ew->wall_emb);
    alloc_register(alloc, &ew->weapon_emb);
    alloc_register(alloc, &ew->proj_w);
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
    a->wall_emb_wgrad = {.shape = {IW_WALL_VOCAB, IW_WALL_DIM}};
    a->weapon_emb_wgrad = {.shape = {IW_WEAPON_VOCAB, IW_WEAPON_DIM}};
    a->proj_wgrad = {.shape = {ew->hidden, IW_CONCAT}};
    a->emb_wgrad_i = {.shape = {IW_EMB_N}};
    alloc_register(grads, &a->wall_emb_wgrad);
    alloc_register(grads, &a->weapon_emb_wgrad);
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
