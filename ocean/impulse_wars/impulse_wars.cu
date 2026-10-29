// Impulse Wars CUDA encoder: map grid CNN, own projectile, enemy projectile
// and floating wall branches, one-hot type slots, concat with the continuous
// obs and the enemy projectile count, projection.
// The map grid's packed cell codes are unpacked into one-hot channels by
// conv1's im2col and run through 3 convs with ReLU, then concatenated and
// projected through a GELU. Conv activations stay NHWC, as the matmul rows,
// and are kept before ReLU: each consumer applies it when reading. Own and
// enemy projectile and floating wall slots are records for a shared Linear -> GELU -> Linear
// per branch that is max pooled over the non-empty records and added to the
// projection, like the OSRS entity encoder; the second Linears of all three
// branches are fused with their pools and the projection's GELU in one
// kernel, so the (records, hidden) outputs are never stored. Wall, weapon
// and ray hit type slots are one-hot encoded into the concat. Included by
// src/ocean.cu.

#include "impulse_wars_obs.h"

static_assert(OBS_SIZE >= IW_BASE_OBS_SIZE + IW_ENEMY_DRONE_SIZE && (OBS_SIZE - IW_BASE_OBS_SIZE) % IW_ENEMY_DRONE_SIZE == 0, "impulse_wars encoder obs layout does not match OBS_SIZE");
static_assert(IW_GRID_CODE_MAX < 256, "impulse_wars map grid codes must be exact in bf16");
static_assert(IW_C3_OUT == 3, "impulse_wars CNN output should keep a 3x3 layout");

// samples per block of the record branches' second layer weight gradient,
// hidden units per block and units k per thread
#define IW_REC_WGRAD_CHUNK 64
#define IW_REC_WGRAD_UNITS 64
#define IW_REC_WGRAD_K 4
// samples per step of the record branches' second layer weight gradient
#define IW_REC_WGRAD_UNROLL 4
// samples per block of the record branches' pool backward, and the largest
// hidden size it takes
#define IW_REC_BWD_SAMPLES 16
#define IW_REC_BWD_MAX_H 2048
// most blocks of the rollout conv1, whose warps loop over output
// positions, and output channels per lane
#define IW_C1_DIRECT_MAX_BLOCKS 1024
#define IW_C1_DIRECT_OC 8
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

// the record branches, in the order their pools add to the encoder output
// and their argmaxes are packed: own projectile, enemy projectile, floating
// wall. Their records are numbered across the branches in this order
#define IW_NUM_REC_BRANCHES 3
#define IW_REC_HIDDEN IW_PROJ_HIDDEN
#define IW_NUM_RECORDS (IW_NUM_PROJECTILES + IW_NUM_FLOATING_WALLS)

__host__ __device__ constexpr int iw_rec_count(int br) {
    return br == 0 ? IW_NUM_OWN_PROJECTILES : br == 1 ? IW_NUM_ENEMY_PROJECTILES : IW_NUM_FLOATING_WALLS;
}

// number of branch br's first record across the branches
__host__ __device__ constexpr int iw_rec_first(int br) {
    return br == 0 ? 0 : br == 1 ? IW_NUM_OWN_PROJECTILES : IW_NUM_PROJECTILES;
}

__host__ __device__ constexpr int iw_rec_branch(int t) {
    return t < IW_NUM_OWN_PROJECTILES ? 0 : t < IW_NUM_PROJECTILES ? 1 : 2;
}

// branch br's pooled record in a packed argmax, -1 if none
__device__ inline int iw_rec_unpack(int packed, int br) {
    return ((packed >> (8 * br)) & 0xff) - 1;
}

// per branch: first layer output before GELU (B * R, HR), after GELU, record
// active flags (B * R), second layer weights (H, HR) and their gradient
struct IwRecPtrs {
    precision_t *z1[IW_NUM_REC_BRANCHES];
    const precision_t *h1[IW_NUM_REC_BRANCHES];
    const int *active[IW_NUM_REC_BRANCHES];
    const precision_t *w2[IW_NUM_REC_BRANCHES];
    precision_t *w2_grad[IW_NUM_REC_BRANCHES];
};

// element br of a per branch array of a kernel parameter, by selects:
// indexing a parameter's array at run time makes nvcc copy the parameter to
// local memory
template <typename T>
__device__ inline T iw_rec_sel(T const (&a)[IW_NUM_REC_BRANCHES], int br) {
    return br == 0 ? a[0] : br == 1 ? a[1] : a[2];
}

// h1 of record t (across the branches) of sample b at unit k
__device__ inline float iw_rec_h1(const IwRecPtrs &p, int b, int t, int k) {
    int br = iw_rec_branch(t);
    int r = t - iw_rec_first(br);
    return to_float(iw_rec_sel(p.h1, br)[((int64_t)b * iw_rec_count(br) + r) * IW_REC_HIDDEN + k]);
}

// the projection's GELU and the record branches' second layers fused with
// their pools: out[b, h] is GELU(proj_z[b, h]) plus, per branch in order, the
// max over its active records r of h1[b, r] . w2[h], 0 if none, rounded to
// precision_t after each add as separate kernels would. argmax packs each
// branch's pooled record + 1 (0 if none) into byte br, or is null for
// rollout. A thread per hidden unit; each block takes S samples so the
// second layer weights are read once per S samples, fewer for small batches
// to keep enough blocks
template <int S>
__global__ void iw_rec_pool(
    precision_t *__restrict__ out, int *__restrict__ argmax, const precision_t *__restrict__ proj_z, IwRecPtrs p, int B, int H
) {
    constexpr int HR = IW_REC_HIDDEN;
    __shared__ float sh1[S][IW_NUM_RECORDS * HR];
    __shared__ int sact[S][IW_NUM_RECORDS];
    int b0 = blockIdx.y * S;
    for (int i = threadIdx.x; i < S * IW_NUM_RECORDS * HR; i += blockDim.x) {
        int s = i / (IW_NUM_RECORDS * HR);
        int j = i % (IW_NUM_RECORDS * HR);
        sh1[s][j] = b0 + s < B ? iw_rec_h1(p, b0 + s, j / HR, j % HR) : 0.0f;
    }
    for (int i = threadIdx.x; i < S * IW_NUM_RECORDS; i += blockDim.x) {
        int s = i / IW_NUM_RECORDS;
        int t = i % IW_NUM_RECORDS;
        int br = iw_rec_branch(t);
        int b = b0 + s;
        sact[s][t] = b < B ? __ldg(&iw_rec_sel(p.active, br)[b * iw_rec_count(br) + t - iw_rec_first(br)]) : 0;
    }
    __syncthreads();
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    // each unit's weights are contiguous: whole vectors
    float w[IW_NUM_REC_BRANCHES][HR];
#pragma unroll
    for (int br = 0; br < IW_NUM_REC_BRANCHES; br++) {
#pragma unroll
        for (int q = 0; q < HR / IW_VEC; q++) {
            IwVec x = *(const IwVec *)(iw_rec_sel(p.w2, br) + h * HR + q * IW_VEC);
#pragma unroll
            for (int j = 0; j < IW_VEC; j++) {
                w[br][q * IW_VEC + j] = to_float(x.v[j]);
            }
        }
    }
    for (int s = 0; s < S && b0 + s < B; s++) {
        int64_t o = (int64_t)(b0 + s) * H + h;
        float x = to_float(from_float(iw_gelu(to_float(__ldg(&proj_z[o])))));
        int packed = 0;
#pragma unroll
        for (int br = 0; br < IW_NUM_REC_BRANCHES; br++) {
            float best = 0.0f;
            int bestRecord = -1;
#pragma unroll
            for (int r = 0; r < iw_rec_count(br); r++) {
                int t = iw_rec_first(br) + r;
                if (!sact[s][t]) {
                    continue;
                }
                float v = 0.0f;
#pragma unroll
                for (int k = 0; k < HR; k++) {
                    v += sh1[s][t * HR + k] * w[br][k];
                }
                if (bestRecord < 0 || v > best) {
                    best = v;
                    bestRecord = r;
                }
            }
            x = to_float(from_float(x + best));
            packed |= (bestRecord + 1) << (8 * br);
        }
        out[o] = from_float(x);
        if (argmax) {
            argmax[o] = packed;
        }
    }
}

// most records of a branch
#define IW_REC_MAX_COUNT \
    (IW_NUM_OWN_PROJECTILES > IW_NUM_ENEMY_PROJECTILES ? (IW_NUM_OWN_PROJECTILES > IW_NUM_FLOATING_WALLS ? IW_NUM_OWN_PROJECTILES : IW_NUM_FLOATING_WALLS) \
        : (IW_NUM_ENEMY_PROJECTILES > IW_NUM_FLOATING_WALLS ? IW_NUM_ENEMY_PROJECTILES : IW_NUM_FLOATING_WALLS))

// row length of the record pool backward's staged sample: a group's hidden
// units, padded to whole vectors plus 4 so groups 8 apart share banks
// rather than all of them
__host__ __device__ constexpr int iw_rec_bwd_row(int H) {
    return (H / IW_REC_BWD_GROUPS + 3) / 4 * 4 + 4;
}

// 32 bit words of a record pool backward mask over a group's hidden units
__host__ __device__ constexpr int iw_rec_bwd_mask_words(int H) {
    return (H / IW_REC_BWD_GROUPS + 31) / 32;
}

// routes the encoder output gradient (B, H) through one branch's pool
// (blockIdx.y) and second layer to its first layer pre-GELU outputs z1,
// overwriting z1 with the gradient at it; the own projectile branch's blocks
// also overwrite proj_z with the gradient at it through the projection's
// GELU. A block of IW_REC_BWD_GROUPS * HR threads takes IW_REC_BWD_SAMPLES
// samples: thread (g, k) sums every IW_REC_BWD_GROUPS-th hidden unit from g
// into per record partials, which are then summed over g.
//
// Per sample, the pooled records and gradient are staged in shared memory
// by group, then warp ballots turn the records into a mask per (group,
// record) of the group's hidden units that pool it. Each thread then makes a
// pass per record that any of its warp's hidden units pool, adding the
// masked units in order with predicated FMAs, rather than dispatching every
// unit to its record's slot: the same adds in the same order per slot.
//
// J is H / IW_REC_BWD_GROUPS, the hidden units per thread, whose second
// layer weights it keeps in registers for all the samples, or 0 to read them
// from memory for any H up to IW_REC_BWD_MAX_H. The next sample's records
// and gradient are loaded while the current one is summed.
template <int J>
__global__ void __launch_bounds__(IW_REC_BWD_GROUPS * IW_REC_HIDDEN) iw_rec_pool_backward(
    precision_t *__restrict__ proj_z, const precision_t *__restrict__ grad, const int *__restrict__ argmax, IwRecPtrs p, int B, int H
) {
    constexpr int HR = IW_REC_HIDDEN;
    constexpr int G = IW_REC_BWD_GROUPS;
    constexpr int T = G * HR;
    static_assert(T / 32 * 2 == G, "impulse_wars record pool backward builds the masks of 2 groups per warp");
    static_assert(J % 4 == 0, "impulse_wars record pool backward reads 4 hidden units at a time");
    // hidden units each thread stages per sample, and mask words
    constexpr int NS = J ? (J * G + T - 1) / T : IW_REC_BWD_MAX_H / T;
    constexpr int MW = J ? (J + 31) / 32 : iw_rec_bwd_mask_words(IW_REC_BWD_MAX_H);
    __shared__ float partial[G][IW_REC_MAX_COUNT * HR];
    extern __shared__ __align__(16) unsigned char iw_rec_bwd_dyn[];
    const int row = iw_rec_bwd_row(H);
    const int mw = J ? MW : iw_rec_bwd_mask_words(H);
    float *sg = (float *)iw_rec_bwd_dyn;
    unsigned *smask = (unsigned *)(sg + G * row);
    signed char *sr = (signed char *)(smask + G * IW_REC_MAX_COUNT * mw);
    const int br = blockIdx.y;
    const int R = iw_rec_count(br);
    const int b0 = blockIdx.x * IW_REC_BWD_SAMPLES;
    const int nS = min(IW_REC_BWD_SAMPLES, B - b0);
    const int jn = J ? J : H / G;
    const int g = threadIdx.x / HR;
    const int k = threadIdx.x % HR;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const precision_t *w2 = iw_rec_sel(p.w2, br);
    precision_t *z1 = iw_rec_sel(p.z1, br);

    float w[J ? J : 1];
    if (J) {
#pragma unroll
        for (int j = 0; j < J; j++) {
            w[j] = to_float(__ldg(&w2[(g + G * j) * HR + k]));
        }
    }

    // the next sample's pooled records, gradient, proj_z (own projectile
    // branch) and, for the threads that sum a slot, z1
    int nr[NS];
    float ng[NS], nz[NS], nz1 = 0.0f;
    auto load = [&](int b) {
#pragma unroll
        for (int q = 0; q < NS; q++) {
            int h = threadIdx.x + q * T;
            if (h < H) {
                int64_t o = (int64_t)b * H + h;
                nr[q] = iw_rec_unpack(__ldg(&argmax[o]), br);
                ng[q] = to_float(__ldg(&grad[o]));
                nz[q] = br == 0 ? to_float(proj_z[o]) : 0.0f;
            }
        }
        if (threadIdx.x < R * HR) {
            nz1 = to_float(z1[(int64_t)b * R * HR + threadIdx.x]);
        }
    };
    load(b0);
    for (int s = 0; s < nS; s++) {
        const int b = b0 + s;
        // the previous sample is summed
        __syncthreads();
#pragma unroll
        for (int q = 0; q < NS; q++) {
            int h = threadIdx.x + q * T;
            if (h < H) {
                sg[(h % G) * row + h / G] = ng[q];
                sr[(h % G) * row + h / G] = (signed char)nr[q];
                if (br == 0) {
                    proj_z[(int64_t)b * H + h] = from_float(ng[q] * iw_gelu_grad(nz[q]));
                }
            }
        }
        float z1v = nz1;
        __syncthreads();
        if (s + 1 < nS) {
            load(b + 1);
        }
        // masks: warp w takes groups 2w and 2w + 1, 32 hidden units a ballot
#pragma unroll
        for (int gg = 2 * warp; gg < 2 * warp + 2; gg++) {
            for (int c = 0; c < mw; c++) {
                int j = c * 32 + lane;
                int r = j < jn ? sr[gg * row + j] : -1;
                for (int q = 0; q < R; q++) {
                    unsigned bal = __ballot_sync(0xffffffffu, r == q);
                    if (lane == 0) {
                        smask[(gg * IW_REC_MAX_COUNT + q) * mw + c] = bal;
                    }
                }
            }
        }
        __syncthreads();
        const float *gRow = sg + g * row;
        for (int q = 0; q < R; q++) {
            const unsigned *mq = smask + (g * IW_REC_MAX_COUNT + q) * mw;
            unsigned m[MW];
            unsigned any = 0;
#pragma unroll
            for (int c = 0; c < MW; c++) {
                m[c] = c < mw ? mq[c] : 0u;
                any |= m[c];
            }
            float acc = 0.0f;
            // uniform per warp, so the pass's loads and FMAs aren't divergent.
            // The adds are explicit FMAs: the contracted a += g * w they
            // replace, which a compiler could otherwise turn into a select
            // between a rounded product and -0 added to acc
            if (__any_sync(0xffffffffu, any != 0)) {
                if (J) {
#pragma unroll
                    for (int j = 0; j < J; j += 4) {
                        float4 gv = *(const float4 *)(gRow + j);
                        if ((m[j / 32] >> (j % 32)) & 1u) {
                            acc = fmaf(gv.x, w[j], acc);
                        }
                        if ((m[(j + 1) / 32] >> ((j + 1) % 32)) & 1u) {
                            acc = fmaf(gv.y, w[j + 1], acc);
                        }
                        if ((m[(j + 2) / 32] >> ((j + 2) % 32)) & 1u) {
                            acc = fmaf(gv.z, w[j + 2], acc);
                        }
                        if ((m[(j + 3) / 32] >> ((j + 3) % 32)) & 1u) {
                            acc = fmaf(gv.w, w[j + 3], acc);
                        }
                    }
                } else {
                    for (int j = 0; j < jn; j++) {
                        if ((m[j / 32] >> (j % 32)) & 1u) {
                            acc = fmaf(gRow[j], to_float(__ldg(&w2[(g + G * j) * HR + k])), acc);
                        }
                    }
                }
            }
            partial[g][q * HR + k] = acc;
        }
        __syncthreads();
        if (threadIdx.x < R * HR) {
            float sum = 0.0f;
            for (int i = 0; i < G; i++) {
                sum += partial[i][threadIdx.x];
            }
            z1[(int64_t)b * R * HR + threadIdx.x] = from_float(sum * iw_gelu_grad(z1v));
        }
    }
}

// dynamic shared memory of iw_rec_pool_backward: a sample's gradient by
// group, the masks and the pooled records by group
static int iw_rec_bwd_smem(int H) {
    return IW_REC_BWD_GROUPS * iw_rec_bwd_row(H) * (int)(sizeof(float) + sizeof(signed char))
        + IW_REC_BWD_GROUPS * IW_REC_MAX_COUNT * iw_rec_bwd_mask_words(H) * (int)sizeof(unsigned);
}

// IW_REC_WGRAD_K adjacent units of a record's h1
struct alignas(IW_REC_WGRAD_K * sizeof(precision_t)) IwRecUnits {
    precision_t v[IW_REC_WGRAD_K];
};

// second layer weight gradient partials of every branch,
// dw2[h, k] = sum over b of grad[b, h] * h1[b, argmax[b, h], k], over
// IW_REC_WGRAD_CHUNK samples per blockIdx.y; thread (j, h) takes hidden unit
// h of IW_REC_WGRAD_UNITS per block and units k of group j of
// IW_REC_WGRAD_K: (gridDim.y, branches, H * HR) into partials
__global__ void iw_rec_wgrad_partial(
    float *__restrict__ partials, const precision_t *__restrict__ grad, const int *__restrict__ argmax, IwRecPtrs p, int B, int H
) {
    constexpr int HR = IW_REC_HIDDEN;
    int h = blockIdx.x * IW_REC_WGRAD_UNITS + threadIdx.x % IW_REC_WGRAD_UNITS;
    int k0 = threadIdx.x / IW_REC_WGRAD_UNITS * IW_REC_WGRAD_K;
    if (h >= H) {
        return;
    }
    int b0 = blockIdx.y * IW_REC_WGRAD_CHUNK;
    int b1 = min(b0 + IW_REC_WGRAD_CHUNK, B);
    float acc[IW_NUM_REC_BRANCHES][IW_REC_WGRAD_K] = {};
    // IW_REC_WGRAD_UNROLL samples at a time so their loads are all in flight
    // together; empty branches load record 0 but don't add it
    int b = b0;
    for (; b + IW_REC_WGRAD_UNROLL <= b1; b += IW_REC_WGRAD_UNROLL) {
        int r[IW_REC_WGRAD_UNROLL][IW_NUM_REC_BRANCHES];
        float gv[IW_REC_WGRAD_UNROLL];
        IwRecUnits x[IW_REC_WGRAD_UNROLL][IW_NUM_REC_BRANCHES];
#pragma unroll
        for (int s = 0; s < IW_REC_WGRAD_UNROLL; s++) {
            int64_t o = (int64_t)(b + s) * H + h;
            int packed = __ldg(&argmax[o]);
            gv[s] = to_float(__ldg(&grad[o]));
#pragma unroll
            for (int br = 0; br < IW_NUM_REC_BRANCHES; br++) {
                r[s][br] = iw_rec_unpack(packed, br);
                int rec = max(r[s][br], 0);
                x[s][br] = *(const IwRecUnits *)(iw_rec_sel(p.h1, br) + ((int64_t)(b + s) * iw_rec_count(br) + rec) * HR + k0);
            }
        }
#pragma unroll
        for (int s = 0; s < IW_REC_WGRAD_UNROLL; s++) {
#pragma unroll
            for (int br = 0; br < IW_NUM_REC_BRANCHES; br++) {
                if (r[s][br] >= 0) {
#pragma unroll
                    for (int j = 0; j < IW_REC_WGRAD_K; j++) {
                        acc[br][j] += gv[s] * to_float(x[s][br].v[j]);
                    }
                }
            }
        }
    }
    for (; b < b1; b++) {
        int64_t o = (int64_t)b * H + h;
        int packed = __ldg(&argmax[o]);
        float gv = to_float(__ldg(&grad[o]));
#pragma unroll
        for (int br = 0; br < IW_NUM_REC_BRANCHES; br++) {
            int rec = iw_rec_unpack(packed, br);
            if (rec >= 0) {
                IwRecUnits x = *(const IwRecUnits *)(iw_rec_sel(p.h1, br) + ((int64_t)b * iw_rec_count(br) + rec) * HR + k0);
#pragma unroll
                for (int j = 0; j < IW_REC_WGRAD_K; j++) {
                    acc[br][j] += gv * to_float(x.v[j]);
                }
            }
        }
    }
#pragma unroll
    for (int br = 0; br < IW_NUM_REC_BRANCHES; br++) {
#pragma unroll
        for (int j = 0; j < IW_REC_WGRAD_K; j++) {
            partials[((int64_t)blockIdx.y * IW_NUM_REC_BRANCHES + br) * H * HR + h * HR + k0 + j] = acc[br][j];
        }
    }
}

// sums the (chunks, branches, n) weight gradient partials into each
// branch's second layer weight gradient (n)
__global__ void iw_rec_wgrad_reduce(IwRecPtrs p, const float *__restrict__ partials, int chunks, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= IW_NUM_REC_BRANCHES * n) {
        return;
    }
    float sum = 0.0f;
    for (int c = 0; c < chunks; c++) {
        sum += partials[(int64_t)c * IW_NUM_REC_BRANCHES * n + idx];
    }
    iw_rec_sel(p.w2_grad, idx / n)[idx % n] = from_float(sum);
}

// per branch: records, first layer output before and after GELU, record
// active flags and first layer weights (HR, features)
struct IwRecLayer1Ptrs {
    precision_t *rec[IW_NUM_REC_BRANCHES];
    precision_t *z1[IW_NUM_REC_BRANCHES];
    precision_t *h1[IW_NUM_REC_BRANCHES];
    int *active[IW_NUM_REC_BRANCHES];
    const precision_t *l1_w[IW_NUM_REC_BRANCHES];
};

// unit k of record t of sample b of a branch with F features from NF slot
// fields: writes features k, k + HR, ... of the record and, for unit 0, its
// active flag as the gathers do, then z1 and h1 from the record rounded to
// precision_t as the matmul reads it, summing the features in order. w is
// the branch's first layer weights as (features, HR)
template <int F, int NF, bool PROJ>
__device__ void iw_rec_layer1_one(
    const IwRecLayer1Ptrs &q, const float *w, const precision_t *slot, int br, int64_t row, int k
) {
    constexpr int HR = IW_REC_HIDDEN;
    float fields[NF];
#pragma unroll
    for (int i = 0; i < NF; i++) {
        fields[i] = to_float(slot[i]);
    }
    float v = 0.0f;
#pragma unroll
    for (int i = 0; i < F; i++) {
        precision_t x = PROJ ? from_float(iw_proj_feature(fields[iw_proj_field(i)], i)) : from_float(iw_fw_feature(fields[iw_fw_field(i)], i));
        if (i % HR == k) {
            iw_rec_sel(q.rec, br)[row * F + i] = x;
        }
        v += to_float(x) * w[i * HR + k];
    }
    if (k == 0) {
        iw_rec_sel(q.active, br)[row] = PROJ ? iw_proj_active(fields[0]) : iw_fw_active(fields[0]);
    }
    precision_t zv = from_float(v);
    iw_rec_sel(q.z1, br)[row * HR + k] = zv;
    iw_rec_sel(q.h1, br)[row * HR + k] = from_float(iw_gelu(to_float(zv)));
}

static_assert(IW_FW_FEATURES <= IW_PROJ_FEATURES, "impulse_wars rollout record layer stages the weights at the projectile width");

// rollout record branches up to their first layers' GELU straight from the
// obs, without the gathers' and first layers' kernels: a thread per unit of
// each record. z1 can round differently from the first layer matmul's
__global__ void iw_rec_layer1_direct(IwRecLayer1Ptrs q, const precision_t *__restrict__ obs, int B, int obs_size) {
    constexpr int HR = IW_REC_HIDDEN;
    // (features, HR) per branch so a warp's units read adjacent banks
    __shared__ float sw[IW_NUM_REC_BRANCHES][IW_PROJ_FEATURES * HR];
    for (int i = threadIdx.x; i < IW_NUM_REC_BRANCHES * IW_PROJ_FEATURES * HR; i += blockDim.x) {
        int br = i / (IW_PROJ_FEATURES * HR);
        int j = i % (IW_PROJ_FEATURES * HR);
        int f = j / HR;
        int nf = br < 2 ? IW_PROJ_FEATURES : IW_FW_FEATURES;
        sw[br][j] = f < nf ? to_float(iw_rec_sel(q.l1_w, br)[(j % HR) * nf + f]) : 0.0f;
    }
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * IW_NUM_RECORDS * HR) {
        return;
    }
    int k = idx % HR;
    int b = idx / HR / IW_NUM_RECORDS;
    int t = idx / HR % IW_NUM_RECORDS;
    int br = iw_rec_branch(t);
    int64_t row = (int64_t)b * iw_rec_count(br) + t - iw_rec_first(br);
    const precision_t *o = obs + (int64_t)b * obs_size;
    // own then enemy projectile records are the projectile slots in order
    if (br < 2) {
        iw_rec_layer1_one<IW_PROJ_FEATURES, IW_PROJECTILE_SIZE, true>(q, sw[br], o + iw_proj_slot_idx(t), br, row, k);
    } else {
        iw_rec_layer1_one<IW_FW_FEATURES, IW_FLOATING_WALL_SIZE, false>(q, sw[br], o + iw_fw_slot_idx(t - iw_rec_first(br)), br, row, k);
    }
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
    Int own_active;
    // enemy projectile and floating wall branches, as the own projectile
    // branch
    Prec enemy_rec, enemy_z1, enemy_h1;
    Int enemy_active;
    Prec fw_rec, fw_z1, fw_h1;
    Int fw_active;
    // every branch's pooled record per (sample, hidden unit), packed as
    // iw_rec_pool writes them
    Int rec_argmax;
    // second layer weight gradient partials of every branch
    Float rec_wgrad_partials;
    Prec conv1_wgrad, conv2_wgrad, conv3_wgrad, proj_wgrad, own_proj_l1_wgrad, own_proj_l2_wgrad, enemy_proj_l1_wgrad,
        enemy_proj_l2_wgrad, fw_l1_wgrad, fw_l2_wgrad;
};

static_assert(IW_PROJ_HIDDEN == IW_FW_HIDDEN, "impulse_wars record branches share their pool and weight gradient kernels");
static_assert(IW_NUM_OWN_PROJECTILES < 255 && IW_NUM_ENEMY_PROJECTILES < 255 && IW_NUM_FLOATING_WALLS < 255,
    "impulse_wars pooled records must fit a byte of the packed argmax");
static_assert(IW_REC_HIDDEN % IW_VEC == 0, "impulse_wars record pool loads whole vectors of second layer weights");
static_assert(IW_REC_HIDDEN % IW_REC_WGRAD_K == 0, "impulse_wars record weight gradient threads take whole groups of units");


// the record branches' buffers, in branch order; the second layer weight
// gradients are null for rollout
static IwRecPtrs iw_rec_ptrs(IwEncoderWeights *ew, IwEncoderActivations *a) {
    IwRecPtrs p = {
        .z1 = {a->own_z1.data, a->enemy_z1.data, a->fw_z1.data},
        .h1 = {a->own_h1.data, a->enemy_h1.data, a->fw_h1.data},
        .active = {a->own_active.data, a->enemy_active.data, a->fw_active.data},
        .w2 = {ew->own_proj_l2_w.data, ew->enemy_proj_l2_w.data, ew->fw_l2_w.data},
        .w2_grad = {a->own_proj_l2_wgrad.data, a->enemy_proj_l2_wgrad.data, a->fw_l2_wgrad.data},
    };
    return p;
}

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

// record branch first layer and GELU from gathered records (B * R rows)
template <int R, int HR>
static void iw_rec_layer1(Prec *rec, Prec *l1_w, Prec *z1, Prec *h1, int B, cudaStream_t stream) {
    int n1 = B * R * HR;
    puf_mm(rec, l1_w, z1, stream);
    iw_gelu_kernel<<<grid_size(n1), BLOCK_SIZE, 0, stream>>>(h1->data, z1->data, n1);
}

// own projectile, enemy projectile and floating wall branches to their
// first layers' GELU for training: gathers, matmuls and GELUs
static void iw_branches_layer1(IwEncoderWeights *ew, IwEncoderActivations *a, Prec input, int B, cudaStream_t stream) {
    iw_proj_gather<IW_NUM_OWN_PROJECTILES, 0><<<grid_size(B * IW_NUM_OWN_PROJECTILES * IW_PROJ_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->own_rec.data, a->own_active.data, input.data, B, ew->obs_size
    );
    iw_rec_layer1<IW_NUM_OWN_PROJECTILES, IW_PROJ_HIDDEN>(&a->own_rec, &ew->own_proj_l1_w, &a->own_z1, &a->own_h1, B, stream);
    iw_proj_gather<IW_NUM_ENEMY_PROJECTILES, IW_NUM_OWN_PROJECTILES>
        <<<grid_size(B * IW_NUM_ENEMY_PROJECTILES * IW_PROJ_FEATURES), BLOCK_SIZE, 0, stream>>>(
            a->enemy_rec.data, a->enemy_active.data, input.data, B, ew->obs_size
        );
    iw_rec_layer1<IW_NUM_ENEMY_PROJECTILES, IW_PROJ_HIDDEN>(&a->enemy_rec, &ew->enemy_proj_l1_w, &a->enemy_z1, &a->enemy_h1, B, stream);
    iw_fw_gather<<<grid_size(B * IW_NUM_FLOATING_WALLS * IW_FW_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->fw_rec.data, a->fw_active.data, input.data, B, ew->obs_size
    );
    iw_rec_layer1<IW_NUM_FLOATING_WALLS, IW_FW_HIDDEN>(&a->fw_rec, &ew->fw_l1_w, &a->fw_z1, &a->fw_h1, B, stream);
}

// own projectile, enemy projectile and floating wall branches forward to
// their first layers' GELU, as for training or in one kernel for rollout,
// then the projection's GELU and the branches' pools into a->out; the
// argmax isn't allocated for rollout, so stays null
static void iw_branches_forward(IwEncoderWeights *ew, IwEncoderActivations *a, Prec input, int B, cudaStream_t stream) {
    if (a->rec_argmax.data) {
        iw_branches_layer1(ew, a, input, B, stream);
    } else {
        IwRecLayer1Ptrs q = {
            .rec = {a->own_rec.data, a->enemy_rec.data, a->fw_rec.data},
            .z1 = {a->own_z1.data, a->enemy_z1.data, a->fw_z1.data},
            .h1 = {a->own_h1.data, a->enemy_h1.data, a->fw_h1.data},
            .active = {a->own_active.data, a->enemy_active.data, a->fw_active.data},
            .l1_w = {ew->own_proj_l1_w.data, ew->enemy_proj_l1_w.data, ew->fw_l1_w.data},
        };
        iw_rec_layer1_direct<<<grid_size(B * IW_NUM_RECORDS * IW_REC_HIDDEN), BLOCK_SIZE, 0, stream>>>(q, input.data, B, ew->obs_size);
    }

    int H = ew->hidden;
    IwRecPtrs p = iw_rec_ptrs(ew, a);
    int hBlocks = (H + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // samples per block as fastest measured: 1 for B 16 and 496
    if (B >= 2048) {
        iw_rec_pool<8><<<dim3(hBlocks, (B + 7) / 8), BLOCK_SIZE, 0, stream>>>(a->out.data, a->rec_argmax.data, a->proj_z.data, p, B, H);
    } else {
        iw_rec_pool<1><<<dim3(hBlocks, B), BLOCK_SIZE, 0, stream>>>(a->out.data, a->rec_argmax.data, a->proj_z.data, p, B, H);
    }
}

// record branches backward, also through the projection's GELU: the second
// layers' weight gradients, then z1 and proj_z are overwritten with the
// gradients at them, and the first layers' weight gradients
static void iw_branches_backward(IwEncoderWeights *ew, IwEncoderActivations *a, Prec grad, int B, cudaStream_t stream) {
    int H = ew->hidden;
    IwRecPtrs p = iw_rec_ptrs(ew, a);
    int chunks = (B + IW_REC_WGRAD_CHUNK - 1) / IW_REC_WGRAD_CHUNK;
    dim3 wgradGrid((H + IW_REC_WGRAD_UNITS - 1) / IW_REC_WGRAD_UNITS, chunks);
    iw_rec_wgrad_partial<<<wgradGrid, IW_REC_WGRAD_UNITS * IW_REC_HIDDEN / IW_REC_WGRAD_K, 0, stream>>>(
        a->rec_wgrad_partials.data, grad.data, a->rec_argmax.data, p, B, H
    );
    int n = H * IW_REC_HIDDEN;
    iw_rec_wgrad_reduce<<<grid_size(IW_NUM_REC_BRANCHES * n), BLOCK_SIZE, 0, stream>>>(p, a->rec_wgrad_partials.data, chunks, n);
    dim3 bwdGrid((B + IW_REC_BWD_SAMPLES - 1) / IW_REC_BWD_SAMPLES, IW_NUM_REC_BRANCHES);
    int bwdThreads = IW_REC_BWD_GROUPS * IW_REC_HIDDEN;
    int smem = iw_rec_bwd_smem(H);
    precision_t *proj_z = a->proj_z.data;
    const int *argmax = a->rec_argmax.data;
    // the hidden sizes the sweeps use keep the weights in registers
    switch (H) {
    case 128:
        iw_rec_pool_backward<128 / IW_REC_BWD_GROUPS><<<bwdGrid, bwdThreads, smem, stream>>>(proj_z, grad.data, argmax, p, B, H);
        break;
    case 256:
        iw_rec_pool_backward<256 / IW_REC_BWD_GROUPS><<<bwdGrid, bwdThreads, smem, stream>>>(proj_z, grad.data, argmax, p, B, H);
        break;
    case 512:
        iw_rec_pool_backward<512 / IW_REC_BWD_GROUPS><<<bwdGrid, bwdThreads, smem, stream>>>(proj_z, grad.data, argmax, p, B, H);
        break;
    case 1024:
        iw_rec_pool_backward<1024 / IW_REC_BWD_GROUPS><<<bwdGrid, bwdThreads, smem, stream>>>(proj_z, grad.data, argmax, p, B, H);
        break;
    default:
        assert(H <= IW_REC_BWD_MAX_H && H % IW_REC_BWD_GROUPS == 0 && "impulse_wars record pool backward: unsupported hidden size");
        iw_rec_pool_backward<0><<<bwdGrid, bwdThreads, smem, stream>>>(proj_z, grad.data, argmax, p, B, H);
    }
    puf_mm_tn(&a->own_z1, &a->own_rec, &a->own_proj_l1_wgrad, stream);
    puf_mm_tn(&a->enemy_z1, &a->enemy_rec, &a->enemy_proj_l1_wgrad, stream);
    puf_mm_tn(&a->fw_z1, &a->fw_rec, &a->fw_l1_wgrad, stream);
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

    // also applies the projection's GELU
    iw_branches_forward(ew, a, input, B, stream);
    return a->out;
}

static void iw_encoder_backward(
    void *w, void *activations, Prec grad, cudaStream_t stream
) {
    IwEncoderWeights *ew = (IwEncoderWeights *)w;
    IwEncoderActivations *a = (IwEncoderActivations *)activations;
    int B = grad.shape[0];

    // the record branches are added after GELU, so only the projection's
    // gradient goes through it; the branches' pool backward applies it
    iw_branches_backward(ew, a, grad, B, stream);
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
    iw_reg_branch_acts(a, acts, B_TT);
    a->rec_argmax = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts, &a->rec_argmax);
    int chunks = (B_TT + IW_REC_WGRAD_CHUNK - 1) / IW_REC_WGRAD_CHUNK;
    a->rec_wgrad_partials = {.shape = {chunks, IW_NUM_REC_BRANCHES * ew->hidden * IW_REC_HIDDEN}};
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
    iw_reg_conv_acts(a, alloc, B, false);
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
