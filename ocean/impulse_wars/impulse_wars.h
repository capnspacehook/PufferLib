#ifndef IMPULSE_WARS_OBS_T
#define IMPULSE_WARS_OBS_T
typedef float obs_t;
#endif

// uncomment to use continuous actions
#define NUM_ATNS 7
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1}
#define CONTINUOUS_ACTIONS 1

// actions:
// 9: move, noop + 8 directions
// 17: aim, noop + 16 directions
// 2: shoot or not
// 2: brake or not
// 2: burst or not
// #define NUM_ATNS 5
// #define ACT_SIZES {9, 17, 2, 2, 2}
// #define CONTINUOUS_ACTIONS 0

#define OBS_SIZE 395 // for 2 drones (players)

#ifdef __cplusplus
// Game/Box2D/collections-c are C (void* implicit conv); pufferl.cu is C++17.
#define puf_init puf_init_cxx_decl
#define puf_reset puf_reset_cxx_decl
#define puf_step puf_step_cxx_decl
#define puf_render puf_render_cxx_decl
#define puf_close puf_close_cxx_decl
#define puf_log puf_log_cxx_decl
#define CC_ARRAY_NO_IMPL
#include "types.h"
#undef puf_init
#undef puf_reset
#undef puf_step
#undef puf_render
#undef puf_close
#undef puf_log

extern "C" {
void puf_init(Env *env, Dict *kwargs);
void puf_reset(Env *env);
void puf_step(Env *env);
void puf_render(Env *env);
void puf_close(Env *env);
void puf_log(Log *log, Dict *out);
}

#else

#include "env.h"
#include <pthread.h>

#define PUF_STEPS_PER_SEC _EVAL_FRAME_RATE
#ifdef PUFFERCPU_EVAL_MAIN
#define PUF_EVAL_SHOULD_FORWARD
#endif

int b2InternalAssert(const char *condition, const char *fileName, int lineNumber) {
    fprintf(stderr, "box2d assert %s at %s:%d\n", condition, fileName, lineNumber);
    ON_ERROR;
}

void puf_init(Env *env, Dict *kwargs) {
    uint8_t num_drones = dict_get(kwargs, "num_drones");
    uint8_t num_agents = dict_get(kwargs, "num_agents");
    int8_t map_idx = dict_get(kwargs, "map_idx");
    bool enable_teams = dict_get(kwargs, "enable_teams");
    bool sitting_duck = dict_get(kwargs, "sitting_duck");
    bool is_training = dict_get(kwargs, "is_training");
    initEnv(env, num_drones, num_agents, map_idx, env->rng, enable_teams, sitting_duck, is_training, (bool)CONTINUOUS_ACTIONS);

    static pthread_mutex_t maps_mu = PTHREAD_MUTEX_INITIALIZER;
    static bool maps_ready = false;
    pthread_mutex_lock(&maps_mu);
    if (!maps_ready) {
        initMaps(env);
        maps_ready = true;
    }
    pthread_mutex_unlock(&maps_mu);

    setRewards(
        env,
        dict_get(kwargs, "reward_win"),
        dict_get(kwargs, "reward_self_kill"),
        dict_get(kwargs, "reward_enemy_death"),
        dict_get(kwargs, "reward_enemy_kill"),
        0.0f, // teammate death punishment
        0.0f, // teammate kill punishment
        dict_get(kwargs, "reward_death"),
        dict_get(kwargs, "reward_energy_emptied"),
        dict_get(kwargs, "reward_weapon_pickup"),
        dict_get(kwargs, "reward_shield_break"),
        dict_get(kwargs, "reward_shot_hit_coef"),
        dict_get(kwargs, "reward_explosion_hit_coef")
    );

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

#define LOG_DRONE_STATS(log, out, idx, idxStr)                                                    \
    dict_set(out, "drone_" idxStr "_returns", log->stats[idx].returns);                           \
    dict_set(out, "drone_" idxStr "_distance_traveled", log->stats[idx].distanceTraveled);        \
    dict_set(out, "drone_" idxStr "_abs_distance_traveled", log->stats[idx].absDistanceTraveled); \
    dict_set(out, "drone_" idxStr "_brake_time", log->stats[idx].brakeTime);                      \
    dict_set(out, "drone_" idxStr "_total_bursts", log->stats[idx].totalBursts);                  \
    dict_set(out, "drone_" idxStr "_bursts_hit", log->stats[idx].burstsHit);                      \
    dict_set(out, "drone_" idxStr "_energy_emptied", log->stats[idx].energyEmptied);              \
    dict_set(out, "drone_" idxStr "_shields_broken", log->stats[idx].shieldsBroken);              \
    dict_set(out, "drone_" idxStr "_own_shield_broken", log->stats[idx].ownShieldBroken);         \
    dict_set(out, "drone_" idxStr "_self_kills", log->stats[idx].selfKills);                      \
    dict_set(out, "drone_" idxStr "_kills", log->stats[idx].kills);                               \
    dict_set(out, "drone_" idxStr "_unknown_kills", log->stats[idx].unknownKills);                \
    dict_set(out, "drone_" idxStr "_wins", log->stats[idx].wins);                                 \
    dict_set(out, "drone_" idxStr "_total_shots_fired", log->stats[idx].totalShotsFired);         \
    dict_set(out, "drone_" idxStr "_total_shots_hit", log->stats[idx].totalShotsHit);             \
    dict_set(out, "drone_" idxStr "_total_shots_taken", log->stats[idx].totalShotsTaken);         \
    dict_set(out, "drone_" idxStr "_total_own_shots_taken", log->stats[idx].totalOwnShotsTaken);  \
    dict_set(out, "drone_" idxStr "_total_picked_up", log->stats[idx].totalWeaponsPickedUp);      \
    dict_set(out, "drone_" idxStr "_total_shot_distances", log->stats[idx].totalShotDistances)

void puf_log(Log *log, Dict *out) {
    dict_set(out, "episode_length", log->length);
    dict_set(out, "ties", log->ties);
    dict_set(out, "perf", log->stats[0].wins);
    dict_set(out, "score", log->stats[0].wins);
    dict_set(out, "n", log->n);

    LOG_DRONE_STATS(log, out, 0, "0");
    LOG_DRONE_STATS(log, out, 1, "1");
}

#endif
