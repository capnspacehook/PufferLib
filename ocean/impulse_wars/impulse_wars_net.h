// CPU Impulse Wars encoder: map grid CNN, one-hot wall/weapon/ray types +
// projection plus the max pooled projectile and floating wall branches, then
// MinGRU + decoder.
// Weight order matches CUDA create_impulse_wars_encoder + algo
// decoder/mingru:
//   conv1 (32 x 90), conv2 (64 x 288), conv3 (64 x 576), each (OC, K, K, IC)
//   as the CUDA encoder keeps conv activations NHWC; init reorders them
//   to the (OC, IC, K, K) the CPU conv wants,
//   proj (hidden x IW_CONCAT),
//   projectile l1 (IW_PROJ_HIDDEN x IW_PROJ_FEATURES),
//   projectile l2 (hidden x IW_PROJ_HIDDEN),
//   floating wall l1 (IW_FW_HIDDEN x IW_FW_FEATURES),
//   floating wall l2 (hidden x IW_FW_HIDDEN),
//   decoder ((atn_sum+1) x hidden), decoder logstd (NUM_ATNS) if continuous,
//   mingru layers (3*hidden x hidden each).
// CONTINUOUS_ACTIONS in impulse_wars.h selects gaussian mean or multidiscrete
// sampling, matching the generic PufferNet.
// Include puffercpu.c first (Linear / MinGRU).

#include "impulse_wars_obs.h"

typedef struct ImpulseWarsNet {
    int num_agents;
    // one-hot map grid planes, (num_agents, IW_C1_IC, IW_C1_IN, IW_C1_IN)
    float *grid;
    Conv2D *conv1;
    Conv2D *conv2;
    Conv2D *conv3;
    int scalar_idx[IW_NUM_SCALARS];
    float *concat;
    Linear *proj;
    // projectile records, (num_agents * IW_NUM_PROJECTILES, IW_PROJ_FEATURES)
    float *rec;
    int *rec_active;
    Linear *proj_l1;
    Linear *proj_l2;
    // floating wall records, (num_agents * IW_NUM_FLOATING_WALLS, IW_FW_FEATURES)
    float *fw_rec;
    int *fw_rec_active;
    Linear *fw_l1;
    Linear *fw_l2;
    Linear *decoder;
    float *log_std;
    MinGRU *mingru;
    Multidiscrete *multidiscrete;
} ImpulseWarsNet;

static inline int impulse_wars_atn_sum(void) {
    const int act_sizes[] = ACT_SIZES;
    int sum = 0;
    for (int i = 0; i < NUM_ATNS; i++) {
        sum += act_sizes[i];
    }
    return sum;
}

static inline int puf_iw_align8(int n) {
    return (n + 7) & ~7;
}

static inline int impulse_wars_weight_count(int hidden, int layers) {
    int n = 0;
    n = puf_iw_align8(n + IW_C1_W_N);
    n = puf_iw_align8(n + IW_C2_W_N);
    n = puf_iw_align8(n + IW_C3_W_N);
    n = puf_iw_align8(n + hidden * IW_CONCAT);
    n = puf_iw_align8(n + IW_PROJ_L1_N);
    n = puf_iw_align8(n + hidden * IW_PROJ_HIDDEN);
    n = puf_iw_align8(n + IW_FW_L1_N);
    n = puf_iw_align8(n + hidden * IW_FW_HIDDEN);
    n = puf_iw_align8(n + (impulse_wars_atn_sum() + 1) * hidden);
#if CONTINUOUS_ACTIONS
    n = puf_iw_align8(n + NUM_ATNS);
#endif
    for (int l = 0; l < layers; l++) {
        n = puf_iw_align8(n + 3 * hidden * hidden);
    }
    return n;
}

// copy of conv weights reordered from the CUDA encoder's (OC, K, K, IC) to
// the CPU conv's (OC, IC, K, K); a copy as the loaded weights may be shared
// between nets
static inline float *iw_conv_weights_to_nchw(const float *w, int oc, int ic, int k) {
    float *out = (float *)malloc((size_t)oc * ic * k * k * sizeof(float));
    for (int o = 0; o < oc; o++) {
        for (int c = 0; c < ic; c++) {
            for (int kk = 0; kk < k * k; kk++) {
                out[(o * ic + c) * k * k + kk] = w[(o * k * k + kk) * ic + c];
            }
        }
    }
    return out;
}

static inline ImpulseWarsNet *init_impulse_wars_net(Weights *weights, int num_agents, int hidden, int layers) {
    // impulse_wars_obs.h mirrors settings.h, ensure they agree
    assert(IW_MAP_CELLS == MAP_OBS_SIZE);
    assert(IW_GRID_SIZE == MAP_OBS_ROWS && IW_GRID_SIZE == MAP_OBS_COLUMNS);
    assert(IW_GRID_FLOATING_SHIFT == GRID_FLOATING_WALL_SHIFT);
    assert(IW_GRID_PICKUP_BIT == GRID_PICKUP_BIT && IW_GRID_ENEMY_BIT == GRID_ENEMY_BIT);
    assert(IW_GRID_STATIC_IDS == GRID_OUT_OF_BOUNDS + 1);
    assert(IW_GRID_CODE_MAX == (GRID_ENEMY_BIT | GRID_PICKUP_BIT | (0x3 << GRID_FLOATING_WALL_SHIFT) | IW_GRID_STATIC_MASK));
    // make_conv2d doesn't align, so the conv weights must keep the 8 float alignment
    assert(IW_C1_W_N % 8 == 0 && IW_C2_W_N % 8 == 0 && IW_C3_W_N % 8 == 0);
    assert(IW_NEAR_WALL_OFFSET == DISCRETE_OBS_SIZE + NEAR_WALL_INFO_OBS_OFFSET);
    assert(IW_NUM_FLOATING_WALLS == NUM_FLOATING_WALL_OBS);
    assert(IW_FLOATING_WALL_SIZE == FLOATING_WALL_INFO_OBS_SIZE);
    assert(IW_FLOATING_WALL_OFFSET == DISCRETE_OBS_SIZE + FLOATING_WALL_INFO_OBS_OFFSET);
    assert(IW_PICKUP_OFFSET == DISCRETE_OBS_SIZE + WEAPON_PICKUP_INFO_OBS_OFFSET);
    assert(IW_PROJECTILE_OFFSET == DISCRETE_OBS_SIZE + PROJECTILE_INFO_OBS_OFFSET);
    assert(IW_ENEMY_DRONE_OFFSET == DISCRETE_OBS_SIZE + ENEMY_DRONE_OBS_OFFSET);
    assert(IW_ENEMY_DRONE_SIZE == ENEMY_DRONE_OBS_SIZE);
    assert(IW_ENEMY_RAY_IDX == ENEMY_DRONE_RAY_OBS_IDX);
    assert(IW_DRONE_SIZE == DRONE_OBS_SIZE);
    assert(IW_DRONE_RAY_IDX == DRONE_RAY_OBS_IDX);
    assert(IW_NUM_DRONE_RAYS == NUM_DRONE_RAY_OBS);
    assert(IW_RAY_SIZE == RAY_OBS_SIZE);
    assert(IW_WEAPON_SIZE == WEAPON_OBS_SIZE);
    assert(IW_UTILITY_SIZE == UTILITY_OBS_SIZE);
    assert(IW_RAY_VOCAB == NUM_RAY_HIT_TYPES);
    assert(IW_BASE_OBS_SIZE == DISCRETE_OBS_SIZE + _CONTINUOUS_OBS_SIZE);
    assert(IW_WEAPON_VOCAB == NUM_WEAPONS && AMMO_CRATE_OBS_TYPE == NUM_WEAPONS);
    assert(IW_UTILITY_VOCAB == NUM_UTILITIES);
    assert(IW_PICKUP_VOCAB == UTILITY_PICKUP_OBS_TYPE(NUM_UTILITIES - 1) + 1);

    ImpulseWarsNet *net = (ImpulseWarsNet *)calloc(1, sizeof(ImpulseWarsNet));
    net->num_agents = num_agents;
    net->grid = (float *)calloc((size_t)num_agents * IW_C1_IC * IW_C1_IN * IW_C1_IN, sizeof(float));
    net->conv1 = make_conv2d(weights, num_agents, IW_C1_IN, IW_C1_IN, IW_C1_IC, IW_C1_OC, IW_C1_K, IW_C1_S);
    net->conv2 = make_conv2d(weights, num_agents, IW_C2_IN, IW_C2_IN, IW_C2_IC, IW_C2_OC, IW_C2_K, IW_C2_S);
    net->conv3 = make_conv2d(weights, num_agents, IW_C3_IN, IW_C3_IN, IW_C3_IC, IW_C3_OC, IW_C3_K, IW_C3_S);
    net->conv1->weights = iw_conv_weights_to_nchw(net->conv1->weights, IW_C1_OC, IW_C1_IC, IW_C1_K);
    net->conv2->weights = iw_conv_weights_to_nchw(net->conv2->weights, IW_C2_OC, IW_C2_IC, IW_C2_K);
    net->conv3->weights = iw_conv_weights_to_nchw(net->conv3->weights, IW_C3_OC, IW_C3_IC, IW_C3_K);
    net->concat = (float *)calloc((size_t)num_agents * IW_CONCAT, sizeof(float));
    net->proj = make_linear(weights, num_agents, IW_CONCAT, hidden);
    const int recs = num_agents * IW_NUM_PROJECTILES;
    net->rec = (float *)calloc((size_t)recs * IW_PROJ_FEATURES, sizeof(float));
    net->rec_active = (int *)calloc((size_t)recs, sizeof(int));
    net->proj_l1 = make_linear(weights, recs, IW_PROJ_FEATURES, IW_PROJ_HIDDEN);
    net->proj_l2 = make_linear(weights, recs, IW_PROJ_HIDDEN, hidden);
    const int fwRecs = num_agents * IW_NUM_FLOATING_WALLS;
    net->fw_rec = (float *)calloc((size_t)fwRecs * IW_FW_FEATURES, sizeof(float));
    net->fw_rec_active = (int *)calloc((size_t)fwRecs, sizeof(int));
    net->fw_l1 = make_linear(weights, fwRecs, IW_FW_FEATURES, IW_FW_HIDDEN);
    net->fw_l2 = make_linear(weights, fwRecs, IW_FW_HIDDEN, hidden);
    net->decoder = make_linear(weights, num_agents, hidden, impulse_wars_atn_sum() + 1);
#if CONTINUOUS_ACTIONS
    net->log_std = get_weights_aligned(weights, NUM_ATNS);
#endif
    net->mingru = make_mingru(weights, num_agents, hidden, layers);
#if !CONTINUOUS_ACTIONS
    int act_sizes[] = ACT_SIZES;
    net->multidiscrete = make_multidiscrete(num_agents, act_sizes, NUM_ATNS);
#endif

    int n = 0;
    for (int i = 0; i < OBS_SIZE; i++) {
        if (iw_is_scalar(i)) {
            net->scalar_idx[n++] = i;
        }
    }
    assert(n == IW_NUM_SCALARS);
    return net;
}

static inline void iw_relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = fmaxf(x[i], 0.0f);
    }
}

// unpacks the map grid cell codes into padded one-hot planes and runs the
// CNN, leaving (num_agents, IW_CONV_FLAT) in conv3->output
static inline void impulse_wars_grid_cnn(ImpulseWarsNet *net, const float *observations) {
    const int plane = IW_C1_IN * IW_C1_IN;
    for (int b = 0; b < net->num_agents; b++) {
        const float *obs = observations + b * OBS_SIZE;
        float *grid = net->grid + b * IW_C1_IC * plane;
        for (int row = 0; row < IW_GRID_SIZE; row++) {
            for (int col = 0; col < IW_GRID_SIZE; col++) {
                const float v = obs[row * IW_GRID_SIZE + col];
                const int p = (row + IW_GRID_PAD) * IW_C1_IN + col + IW_GRID_PAD;
                for (int ch = 0; ch < IW_C1_IC; ch++) {
                    grid[ch * plane + p] = iw_grid_channel(v, ch);
                }
            }
        }
    }
    const int B = net->num_agents;
    conv2d(net->conv1, net->grid);
    iw_relu_inplace(net->conv1->output, B * IW_C1_OC * IW_C1_SPATIAL);
    conv2d(net->conv2, net->conv1->output);
    iw_relu_inplace(net->conv2->output, B * IW_C2_OC * IW_C2_SPATIAL);
    conv2d(net->conv3, net->conv2->output);
    iw_relu_inplace(net->conv3->output, B * IW_CONV_FLAT);
}

// record branch: shared Linear -> GELU -> Linear over every record, max
// pooled over the active records and added to out; a sample with no active
// records adds 0
static inline void impulse_wars_rec_branch(
    Linear *l1, Linear *l2, const float *rec, const int *active, float *out, int B, int R
) {
    linear(l1, rec);
    for (int i = 0; i < B * R * l1->output_dim; i++) {
        l1->output[i] = iw_gelu(l1->output[i]);
    }
    linear(l2, l1->output);

    const int hidden = l2->output_dim;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < hidden; h++) {
            float best = 0.0f;
            int bestRecord = -1;
            for (int r = 0; r < R; r++) {
                const int row = b * R + r;
                if (!active[row]) {
                    continue;
                }
                const float v = l2->output[row * hidden + h];
                if (bestRecord < 0 || v > best) {
                    best = v;
                    bestRecord = r;
                }
            }
            out[b * hidden + h] += best;
        }
    }
}

// projectile branch, added to proj->output
static inline void impulse_wars_proj_branch(ImpulseWarsNet *net, const float *observations) {
    const int recs = net->num_agents * IW_NUM_PROJECTILES;
    for (int row = 0; row < recs; row++) {
        const float *slot = observations + (row / IW_NUM_PROJECTILES) * OBS_SIZE + iw_proj_slot_idx(row % IW_NUM_PROJECTILES);
        for (int f = 0; f < IW_PROJ_FEATURES; f++) {
            net->rec[row * IW_PROJ_FEATURES + f] = iw_proj_feature(slot[iw_proj_field(f)], f);
        }
        net->rec_active[row] = iw_proj_active(slot[1]);
    }
    impulse_wars_rec_branch(net->proj_l1, net->proj_l2, net->rec, net->rec_active, net->proj->output, net->num_agents, IW_NUM_PROJECTILES);
}

// floating wall branch, added to proj->output
static inline void impulse_wars_fw_branch(ImpulseWarsNet *net, const float *observations) {
    const int recs = net->num_agents * IW_NUM_FLOATING_WALLS;
    for (int row = 0; row < recs; row++) {
        const float *slot = observations + (row / IW_NUM_FLOATING_WALLS) * OBS_SIZE + iw_fw_slot_idx(row % IW_NUM_FLOATING_WALLS);
        for (int f = 0; f < IW_FW_FEATURES; f++) {
            net->fw_rec[row * IW_FW_FEATURES + f] = iw_fw_feature(slot[iw_fw_field(f)], f);
        }
        net->fw_rec_active[row] = iw_fw_active(slot[0]);
    }
    impulse_wars_rec_branch(net->fw_l1, net->fw_l2, net->fw_rec, net->fw_rec_active, net->proj->output, net->num_agents, IW_NUM_FLOATING_WALLS);
}

static inline void impulse_wars_encode(ImpulseWarsNet *net, const float *observations) {
    impulse_wars_grid_cnn(net, observations);
    for (int b = 0; b < net->num_agents; b++) {
        const float *obs = observations + b * OBS_SIZE;
        float *concat = net->concat + b * IW_CONCAT;
        // conv3's NCHW output, flattened NHWC as the CUDA encoder does
        const float *conv = net->conv3->output + b * IW_CONV_FLAT;
        for (int sp = 0; sp < IW_C3_SPATIAL; sp++) {
            for (int c = 0; c < IW_C3_OC; c++) {
                concat[sp * IW_C3_OC + c] = conv[c * IW_C3_SPATIAL + sp];
            }
        }
        concat += IW_CONV_FLAT;
        for (int c = 0; c < IW_ONEHOT_COLS; c++) {
            const iwOneHotCol col = iw_onehot_col(c);
            concat[c] = iw_onehot_value(obs[col.obsIdx], col);
        }
        concat += IW_ONEHOT_COLS;
        // the padding after the scalars stays 0 from calloc
        for (int j = 0; j < IW_NUM_SCALARS; j++) {
            concat[j] = obs[net->scalar_idx[j]];
        }
    }
    linear(net->proj, net->concat);
    for (int i = 0; i < net->num_agents * net->proj->output_dim; i++) {
        net->proj->output[i] = iw_gelu(net->proj->output[i]);
    }
    impulse_wars_proj_branch(net, observations);
    impulse_wars_fw_branch(net, observations);
}

static inline void forward_impulse_wars(ImpulseWarsNet *net, const float *observations, float *terminals, float *actions) {
    mingru_zero_term(net->mingru, terminals);
    impulse_wars_encode(net, observations);
    mingru(net->mingru, net->proj->output);
    linear(net->decoder, net->mingru->output);
#if CONTINUOUS_ACTIONS
    _gaussian_mean(net->decoder->output, actions, net->num_agents, NUM_ATNS);
#else
    multidiscrete(net->multidiscrete, net->decoder->output, actions, 0, NULL);
#endif
}
