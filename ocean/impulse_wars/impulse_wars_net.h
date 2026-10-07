// CPU Impulse Wars encoder: wall/weapon type embeddings + projection, then
// MinGRU + decoder. Weight order matches CUDA create_impulse_wars_encoder +
// algo decoder/mingru:
//   wall embed (4 x 4), weapon embed (11 x 8), proj (hidden x IW_CONCAT),
//   decoder ((atn_sum+1) x hidden), decoder logstd (NUM_ATNS) if continuous,
//   mingru layers (3*hidden x hidden each).
// CONTINUOUS_ACTIONS in impulse_wars.h selects gaussian mean or multidiscrete
// sampling, matching the generic PufferNet.
// Include puffercpu.c first (Linear / MinGRU).

#include "impulse_wars_obs.h"

typedef struct ImpulseWarsNet {
    int num_agents;
    float *wall_emb;
    float *weapon_emb;
    int scalar_idx[IW_NUM_SCALARS];
    float *concat;
    Linear *proj;
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
    n = puf_iw_align8(n + IW_WALL_EMB_N);
    n = puf_iw_align8(n + IW_WEAPON_EMB_N);
    n = puf_iw_align8(n + hidden * IW_CONCAT);
    n = puf_iw_align8(n + (impulse_wars_atn_sum() + 1) * hidden);
#if CONTINUOUS_ACTIONS
    n = puf_iw_align8(n + NUM_ATNS);
#endif
    for (int l = 0; l < layers; l++) {
        n = puf_iw_align8(n + 3 * hidden * hidden);
    }
    return n;
}

static inline ImpulseWarsNet *init_impulse_wars_net(Weights *weights, int num_agents, int hidden, int layers) {
    // impulse_wars_obs.h mirrors settings.h, ensure they agree
    assert(IW_MAP_CELLS == MAP_OBS_SIZE);
    assert(IW_NEAR_WALL_OFFSET == DISCRETE_OBS_SIZE + NEAR_WALL_INFO_OBS_OFFSET);
    assert(IW_FLOATING_WALL_OFFSET == DISCRETE_OBS_SIZE + FLOATING_WALL_INFO_OBS_OFFSET);
    assert(IW_PICKUP_OFFSET == DISCRETE_OBS_SIZE + WEAPON_PICKUP_INFO_OBS_OFFSET);
    assert(IW_PROJECTILE_OFFSET == DISCRETE_OBS_SIZE + PROJECTILE_INFO_OBS_OFFSET);
    assert(IW_ENEMY_DRONE_OFFSET == DISCRETE_OBS_SIZE + ENEMY_DRONE_OBS_OFFSET);
    assert(IW_ENEMY_DRONE_SIZE == ENEMY_DRONE_OBS_SIZE);
    assert(IW_BASE_OBS_SIZE == DISCRETE_OBS_SIZE + _CONTINUOUS_OBS_SIZE);
    assert(IW_WEAPON_VOCAB == NUM_WEAPONS + 1);

    ImpulseWarsNet *net = (ImpulseWarsNet *)calloc(1, sizeof(ImpulseWarsNet));
    net->num_agents = num_agents;
    net->wall_emb = get_weights_aligned(weights, IW_WALL_EMB_N);
    net->weapon_emb = get_weights_aligned(weights, IW_WEAPON_EMB_N);
    net->concat = (float *)calloc((size_t)num_agents * IW_CONCAT, sizeof(float));
    net->proj = make_linear(weights, num_agents, IW_CONCAT, hidden);
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
        if (!iw_is_type_slot(i)) {
            net->scalar_idx[n++] = i;
        }
    }
    assert(n == IW_NUM_SCALARS);
    return net;
}

static inline void impulse_wars_encode(ImpulseWarsNet *net, const float *observations) {
    for (int b = 0; b < net->num_agents; b++) {
        const float *obs = observations + b * OBS_SIZE;
        float *concat = net->concat + b * IW_CONCAT;
        for (int j = 0; j < IW_NUM_WALL_SLOTS; j++) {
            int row = iw_type_idx(obs[iw_wall_slot_idx(j)], IW_WALL_VOCAB);
            memcpy(concat + j * IW_WALL_DIM, net->wall_emb + row * IW_WALL_DIM, IW_WALL_DIM * sizeof(float));
        }
        concat += IW_WALL_COLS;
        for (int j = 0; j < IW_NUM_WEAPON_SLOTS; j++) {
            int row = iw_type_idx(obs[iw_weapon_slot_idx(j)], IW_WEAPON_VOCAB);
            memcpy(concat + j * IW_WEAPON_DIM, net->weapon_emb + row * IW_WEAPON_DIM, IW_WEAPON_DIM * sizeof(float));
        }
        concat += IW_WEAPON_COLS;
        for (int j = 0; j < IW_NUM_SCALARS; j++) {
            concat[j] = obs[net->scalar_idx[j]];
        }
    }
    linear(net->proj, net->concat);
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
