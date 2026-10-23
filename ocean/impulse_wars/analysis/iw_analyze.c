// Behavior analysis of a learned policy against the scripted bots or
// another learned policy, in training mode. Against bots the policy plays
// drone 0 and a bot personality drone 1, like the bot ladder. Against a
// policy (--opponent) both drones are agents, like self-play training, and
// every seed is played twice with the policies swapping drones. Writes one
// JSON line per game with the env's episode stats (droneStats and
// droneAnalysisStats) plus per step behavior stats polled from the env
// state, for compare.py. "p" is always the analyzed policy and "b" its
// opponent; "seat" is the drone the policy played.
//
// Games are paired: game i against every opponent uses the same map and
// seed, and the seeds only depend on --seed, so runs of different
// checkpoints or env builds can be compared game by game. The starting
// utility pickups, one nearest each drone, cycle over every pair of
// utilities across games, so each utility is played as much as the others.
//
// Observation studies, for the analyzed policy only:
//   --ablate groups    the policy acts on observations with these groups
//                      zeroed (or with --ablate-mode swap, copied from
//                      another game running in the same batch, which keeps
//                      them in distribution); compare against an unablated
//                      run with compare.py
//   --shadow groups    may be repeated, up to MAX_SHADOWS; a copy of the
//                      policy with its own recurrent state sees the ablated
//                      observations while the policy acts on the full ones,
//                      and the KL of its action heads and its value change
//                      are recorded per game
//   --weights-out f    writes per input feature projection weight norms and
//                      input spreads measured over the games to a TSV, for
//                      weights.py
//   Groups are comma separated names from obsGroups (--list-groups) or
//   obs index ranges like 440-445.
//
// Combo tournament (--combos, needs --opponent): every loadout of a pickup
// weapon (no standard gun) and a utility against every other one and
// itself, --episodes games per matchup.
// Pickups keep spawning, as in combo training episodes, unless
// --no-pickups removes them, so drones keep their loadout all game. Maps
// never lack pickups in training, so that's out of distribution. Weapons
// still run out of ammo then, so --loadout-pickups instead turns any weapon
// or utility picked up into the drone's loadout one with full ammo,
// keeping drones on their loadout all game; pickups still show their
// spawned type in the obs until picked up. --crates plays every game as a
// combo training episode, where all pickups are ammo crates, which is
// in distribution for a policy trained with combo episodes.
//
// Build: ocean/impulse_wars/analysis/build.sh
// Usage: iw_analyze --policy ckpt.bin --hidden H --layers L [--episodes N]
//                   [--threads T] [--batch B] [--seed S] [--maps 1,2,3]
//                   [--out games.jsonl] [opponent ...]
//        iw_analyze --policy ckpt.bin --hidden H --layers L
//                   --opponent ckpt.bin --opponent-hidden H --opponent-layers L
//                   [--opponent-name name] [--combos] [--no-pickups | --loadout-pickups | --crates] [options]
//   --episodes is per opponent (per matchup with --combos); against a
//   policy it's rounded up to an even number so every seed is played from
//   both drones. Bot opponents are bot personality names and default to
//   all of them. --batch games run in lockstep per thread so the network
//   weights are reused across the batch.
#include <pthread.h>
#include <stdatomic.h>

#include "puffercpu.c"
#include "impulse_wars.h"

// puffercpu's linear reads every weight row once per batch row; this reads
// each weight row once for the whole batch. The net header and the MinGRU
// below use it
static void iw_linear(Linear *layer, const float *input) {
    const int B = layer->batch_size;
    const int I = layer->input_dim;
    const int O = layer->output_dim;
    for (int o = 0; o < O; o++) {
        const float *w = layer->weights + (size_t)o * I;
        for (int b = 0; b < B; b++) {
            const float *x = input + (size_t)b * I;
            float sum = 0.0f;
            for (int i = 0; i < I; i++) {
                sum += x[i] * w[i];
            }
            layer->output[(size_t)b * O + o] = sum;
        }
    }
}
#define linear(layer, input) iw_linear(layer, input)

// puffercpu's conv2d recomputes input addresses per multiply and is ~10x
// slower than the rest of the net; this gathers each sample's patches into
// rows (im2col) and takes contiguous dot products with the weight rows
static void iw_conv2d(Conv2D *layer, const float *input) {
    const int IC = layer->in_channels;
    const int K = layer->kernel_size;
    const int S = layer->stride;
    const int IH = layer->in_height;
    const int IW = layer->in_width;
    const int OH = (IH - K) / S + 1;
    const int OW = (IW - K) / S + 1;
    const int OC = layer->out_channels;
    const int cols = IC * K * K;
    float *patches = malloc((size_t)OH * OW * cols * sizeof(float));
    for (int b = 0; b < layer->batch_size; b++) {
        const float *in = input + (size_t)b * IC * IH * IW;
        for (int y = 0; y < OH; y++) {
            for (int x = 0; x < OW; x++) {
                float *row = patches + (size_t)(y * OW + x) * cols;
                for (int c = 0; c < IC; c++) {
                    for (int ky = 0; ky < K; ky++) {
                        const float *src = in + ((size_t)c * IH + y * S + ky) * IW + x * S;
                        for (int kx = 0; kx < K; kx++) {
                            *row++ = src[kx];
                        }
                    }
                }
            }
        }
        float *out = layer->output + (size_t)b * OC * OH * OW;
        for (int o = 0; o < OC; o++) {
            const float *w = layer->weights + (size_t)o * cols;
            for (int sp = 0; sp < OH * OW; sp++) {
                const float *row = patches + (size_t)sp * cols;
                float sum = 0.0f;
                for (int i = 0; i < cols; i++) {
                    sum += row[i] * w[i];
                }
                out[o * OH * OW + sp] = sum;
            }
        }
    }
    free(patches);
}
#define conv2d(layer, input) iw_conv2d(layer, input)

#include "impulse_wars_net.h"

// puffercpu's mingru with the batched linear
static void iw_mingru(MinGRU *layer, const float *input) {
    const int B = layer->batch_size;
    const int H = layer->hidden_size;
    const float *x = input;
    for (int l = 0; l < layer->num_layers; l++) {
        float *state = layer->state + l * B * H;
        iw_linear(layer->proj[l], x);
        const float *combined = layer->proj[l]->output;
        for (int b = 0; b < B; b++) {
            const float *cb = combined + b * 3 * H;
            float *sb = state + b * H;
            const float *xb = x + b * H;
            float *ob = layer->output + b * H;
            for (int h = 0; h < H; h++) {
                const float gate = _sigmoid(cb[H + h]);
                const float hidden = cb[h];
                const float hTilde = hidden >= 0.0f ? hidden + 0.5f : _sigmoid(hidden);
                const float out = sb[h] + gate * (hTilde - sb[h]);
                const float hw = _sigmoid(cb[2 * H + h]);
                ob[h] = hw * out + (1.0f - hw) * xb[h];
                sb[h] = out;
            }
        }
        x = layer->output;
    }
}

#define NW _NUM_WEAPONS
#define MAX_OPPS NUM_BOT_POLICIES
#define AIM_DIRS (AIM_COARSE_SECTORS * AIM_FINE_STEPS)
#define MAX_SHADOWS 16
// steps before a kill another weapon must have been fired in to count as
// part of a combo
#define COMBO_WINDOW 30
// steps before a kill another weapon must have hit the enemy in to count
// as setting it up
#define SETUP_WINDOW 20
// combo loadouts are a pickup weapon and a utility: every weapon after the
// standard one except utility projectiles, with every utility; counted in
// main
#define COMBO_FIRST (STANDARD_WEAPON + 1)
#define MAX_COMBOS (NW * _NUM_UTILITIES)
#define MAX_MATCHUPS (MAX_COMBOS * (MAX_COMBOS + 1) / 2)

static const char *policyNames[NUM_BOT_POLICIES] = {
    [BOT_SITTING_DUCK] = "sitting_duck",
    [BOT_SCRIPTED] = "scripted",
    [BOT_EVADER] = "evader",
    [BOT_CAMPER] = "camper",
    [BOT_RUSHER] = "rusher",
    [BOT_TRAPPER] = "trapper",
    [BOT_TURTLE] = "turtle",
    [BOT_FUNDAMENTALS] = "fundamentals",
};

static const char *analysisWeaponNames[NW] = {
    "unset", "standard", "machinegun", "sniper", "shotgun", "imploder",
    "accelerator", "flak", "blackhole", "nuke", "grapple", "mine",
};

// per step behavior stats polled from the env, per drone
typedef struct behaviorStats {
    double held[NW];
    double fireDist[NW];
    double fireLos[NW];
    double alive;
    double trigger;
    double triggerLos;
    double triggerNoLos;
    double brake;
    double burstCharge;
    double moveNoop;
    double speed;
    double speedLos;
    double speedNoLos;
    double energy;
    double charging;
    double chargeHeldFull;
    double losSteps;
    double noLosSteps;
    double sepSum;
    double sepSteps;
    double closeSteps;
    double farSteps;
    double nearDeath3;
    // situation while both are alive (STANDOFF_LOS, STANDOFF_CLOSE: no LoS
    // within STANDOFF_DIST, STANDOFF_FAR: no LoS further away) by own lives
    // minus the enemy's (behind, even, ahead); approach is speed towards
    // the enemy
    double sitSteps[9];
    double sitApproach[9];
    double sitSpeed[9];
    double sitTrigger[9];
    // standoffs: STANDOFF_MIN_STEPS or more steps of STANDOFF_CLOSE, the
    // steps in them, ones this drone ended by moving towards the enemy
    // harder when LoS came back, and kills and deaths within
    // STANDOFF_AFTER steps of the drone peeking or being peeked
    double standoffs;
    double standoffSteps;
    double peeks;
    double peekKills;
    double peekDeaths;
    double peekedKills;
    double peekedDeaths;
    double nearDeath6;
    double nearFloatingDeath6;
    double firstLosStep;
    double firstPickupStep;
    double fireSteps;
    double fireAimErr;
    double fireAim10;
    double fireNoLos;
    double fireNoLosAim10;

    // weapons and pickups
    // steps holding only the standard gun
    double standardOnly;
    // weapon in hand changed without a pickup: ran out of ammo
    double autoSwitches;
    // where pickups went: top up of the held weapon, into hand replacing the
    // standard gun, or replacing a pickup weapon
    double pickupTopUp;
    double pickupToHand;
    double pickupReplace;
    double dropped[NW];
    // ammo crates: the ammo fraction of the drone's pickup weapon just
    // before picking one up, summed, and crates picked up with no pickup
    // weapon; ammoFrac over ammoSteps is the same mean over live steps
    // with a pickup weapon, for reference
    double crateAmmoFrac;
    double crateAmmoN;
    double crateNoWeapons;
    double ammoFrac;
    double ammoSteps;
    // weapon kills with another weapon fired in the COMBO_WINDOW steps
    // before, by killing weapon (rows) and the other weapon
    double multiWeaponKills;
    double comboKills[NW * NW];
    // weapon kills where another weapon hit the enemy, or had its grapple
    // attached to it, in the SETUP_WINDOW steps before, by killing weapon
    // (rows) and setup weapon; black hole pulls aren't hits
    double setupKills[NW * NW];
    double setupKillsAny;

    // utilities, by utility type: steps holding it and with ammo to use it
    double utilityHeld[_NUM_UTILITIES];
    double utilityReady[_NUM_UTILITIES];
    // steps with the grapple attached and the button up or reeling, and the
    // drone's speed summed over the reeling steps
    double grappleTethered;
    double grappleReeling;
    double grappleReelSpeed;
} behaviorStats;

typedef struct shadowStats {
    double kl[NUM_ATNS];
    double dv;
    double absDv;
    double n;
} shadowStats;

typedef struct gameResult {
    uint8_t opp;
    // drone the analyzed policy played
    uint8_t seat;
    uint8_t map;
    uint32_t seed;
    // combo tournament: starting weapons and utilities of the policy then
    // the opponent
    uint8_t loadout[2];
    uint8_t utilityLoadout[2];
    // utilities of the starting utility pickups nearest each drone,
    // cycled over seeds; 0 without starting utility pickups
    uint8_t startUtilities[2];
    float length;
    droneStats stats[2];
    droneAnalysisStats analysis[2];
    behaviorStats behavior[2];
    // stats are ordered analyzed policy then opponent
    // policy action histograms while alive
    uint32_t moveHist[9];
    uint32_t aimHist[AIM_DIRS];
    uint32_t aimFireHist[AIM_DIRS];
    shadowStats shadow[MAX_SHADOWS];
} gameResult;

typedef struct policyBatch {
    ImpulseWarsNet *net;
    int size;
    float *obs;
    float *actions;
    float *rewards;
    float *terminals;
    unsigned int *rng;
} policyBatch;

// weapon state of a drone at the start of a step
typedef struct slotState {
    enum weaponType weapon;
    int8_t ammo;
} slotState;

// a game slot of a worker's batch
typedef struct gameSlot {
    iwEnv *e;
    gameResult *g;
    droneStats prev[2];
    slotState prevSlots[2];
    float prevKills[2][NW];
    float prevUtilityAmmo[2][_NUM_UTILITIES];
    float prevUtilityHits[2][_NUM_UTILITIES];
    int32_t lastFired[2][NW];
    int32_t lastHit[2][NW];
    uint32_t step;
    // current run of STANDOFF_CLOSE steps, and the drone that peeked to
    // end the last standoff and when
    uint32_t blockedRun;
    int8_t peeker;
    int32_t peekStep;
    bool active;
} gameSlot;

#define STANDOFF_DIST 25.0f
#define STANDOFF_MIN_STEPS 30
#define STANDOFF_AFTER 30
enum { STANDOFF_LOS, STANDOFF_CLOSE, STANDOFF_FAR };

// obs index ranges of the drone blocks
#define E_OBS IW_ENEMY_DRONE_OFFSET
#define D_OBS IW_DRONE_OFFSET
#define M_OBS (IW_DRONE_OFFSET + IW_DRONE_SIZE)

// named obs groups, each up to 2 index ranges {start, len}; the first group
// covering an index is its group in the weights output
typedef struct obsGroup {
    const char *name;
    uint16_t ranges[2][2];
} obsGroup;

static const obsGroup obsGroups[] = {
    {"grid", {{0, IW_MAP_CELLS}}},
    {"near_walls", {{IW_NEAR_WALL_OFFSET, IW_NUM_NEAR_WALLS * IW_NEAR_WALL_SIZE}}},
    {"floating_walls", {{IW_FLOATING_WALL_OFFSET, IW_NUM_FLOATING_WALLS * IW_FLOATING_WALL_SIZE}}},
    {"pickups", {{IW_PICKUP_OFFSET, IW_NUM_PICKUPS * IW_PICKUP_SIZE}}},
    {"projectiles", {{IW_PROJECTILE_OFFSET, IW_NUM_PROJECTILES * IW_PROJECTILE_SIZE}}},
    {"enemy_weapon", {{E_OBS + 0, 5}}},
    // the utility's type, ammo, regen and active count, and the mine's
    // state; the grapple's state between them is its own group
    {"enemy_utility", {{E_OBS + 5, 4}, {E_OBS + 19, 4}}},
    {"enemy_grapple", {{E_OBS + 9, 10}}},
    {"enemy_pos", {{E_OBS + 23, 3}}},
    {"enemy_vel", {{E_OBS + 26, 2}}},
    {"enemy_dir", {{E_OBS + 28, 2}}},
    {"enemy_aim", {{E_OBS + 30, 2}}},
    {"enemy_shield", {{E_OBS + 32, 2}}},
    {"enemy_energy", {{E_OBS + 34, 7}}},
    {"hit_flags", {{E_OBS + 41, 2}, {D_OBS + 42, 3}}},
    {"enemy_lives", {{E_OBS + 43, 3}}},
    {"aim_to_enemy", {{E_OBS + 46, 2}}},
    {"enemy_ray", {{E_OBS + 48, 2}}},
    {"own_weapon", {{D_OBS + 0, 5}}},
    {"own_utility", {{D_OBS + 5, 4}, {D_OBS + 19, 4}}},
    {"own_grapple", {{D_OBS + 9, 10}}},
    {"own_buttons", {{D_OBS + 23, 2}}},
    {"own_pos", {{D_OBS + 25, 2}}},
    {"own_vel", {{D_OBS + 27, 2}}},
    {"own_aim", {{D_OBS + 29, 2}}},
    {"aim_vel", {{D_OBS + 31, 2}}},
    {"own_shield", {{D_OBS + 33, 2}}},
    {"own_energy", {{D_OBS + 35, 7}}},
    {"own_lives", {{D_OBS + 45, 3}}},
    {"own_aim_ray", {{D_OBS + 48, 2}}},
    {"own_vel_ray", {{D_OBS + 50, 2}}},
    {"own_drift_ray", {{D_OBS + 52, 2}}},
    {"misc", {{M_OBS, 3}}},
    // finer groups, not used for the weights output
    {"own_weapon_type", {{D_OBS + 0, 1}}},
    {"own_weapon_state", {{D_OBS + 1, 4}}},
    {"own_charge", {{D_OBS + 3, 2}}},
    {"own_utility_type", {{D_OBS + 5, 1}}},
    {"own_utility_state", {{D_OBS + 6, 17}}},
    {"own_mine", {{D_OBS + 19, 4}}},
    {"own_mine_pos", {{D_OBS + 21, 2}}},
    {"enemy_weapon_type", {{E_OBS + 0, 1}}},
    {"enemy_weapon_state", {{E_OBS + 1, 4}}},
    {"enemy_charge", {{E_OBS + 3, 2}}},
    {"enemy_utility_type", {{E_OBS + 5, 1}}},
    {"enemy_utility_state", {{E_OBS + 6, 17}}},
    {"enemy_mine", {{E_OBS + 19, 4}}},
    {"enemy_mine_pos", {{E_OBS + 21, 2}}},
    {"enemy_loadout", {{E_OBS + 0, 23}}},
    {"own_loadout", {{D_OBS + 0, 25}}},
    {"steps_left", {{M_OBS + 1, 1}}},
    {"sudden_death", {{M_OBS + 2, 1}}},
};
#define NUM_OBS_GROUPS (sizeof(obsGroups) / sizeof(obsGroups[0]))
// groups before this one partition the obs for the weights output
#define NUM_PRIMARY_GROUPS 33

typedef struct obsMask {
    char name[96];
    bool idx[OBS_SIZE];
} obsMask;

enum ablateMode {
    ABLATE_ZERO,
    ABLATE_SWAP,
};

static enum botPolicy opps[MAX_OPPS];
static uint8_t numOpps = 0;
static uint32_t gamesPerOpp = 1400;
static uint64_t baseSeed = 7;
static uint8_t tourMaps[NUM_MAPS];
static uint8_t numMaps = 0;
static int batchSize = 8;
static Weights *policyWeights = NULL;
static int policyHidden = 0;
static int policyLayers = 0;
// learned opponent, NULL to play the bots
static Weights *oppWeights = NULL;
static int oppHidden = 0;
static int oppLayers = 0;
static char oppName[64] = "policy";
static bool combos = false;
static bool noPickups = false;
static bool loadoutPickups = false;
static bool crates = false;
// weapon and utility of each combo loadout
static uint8_t comboLoadouts[MAX_COMBOS][2];
static uint16_t numCombos = 0;
static uint16_t matchups[MAX_MATCHUPS][2];
static uint32_t numMatchups = 0;
static obsMask ablateMask;
static bool ablating = false;
static enum ablateMode ablateMode = ABLATE_ZERO;
static obsMask shadowMasks[MAX_SHADOWS];
static int numShadows = 0;
static const char *weightsOut = NULL;
static char obsNames[OBS_SIZE][48];
static int8_t obsGroupOf[OBS_SIZE];
static int scalarIdx[IW_NUM_SCALARS];
// input statistics for the weights output, over steps the policy is alive
static double concatSum[IW_CONCAT];
static double concatSq[IW_CONCAT];
static double concatN;
static double projRecSum[IW_PROJ_FEATURES];
static double projRecSq[IW_PROJ_FEATURES];
static double projRecN;
static double fwRecSum[IW_FW_FEATURES];
static double fwRecSq[IW_FW_FEATURES];
static double fwRecN;
static pthread_mutex_t statsMu = PTHREAD_MUTEX_INITIALIZER;
static gameResult *results;
static uint32_t totalGames;
static atomic_uint nextJob;
static atomic_uint jobsDone;
static pthread_mutex_t envMu = PTHREAD_MUTEX_INITIALIZER;
static bool mapsReady = false;

static uint32_t hashSeed(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    uint32_t r = (uint32_t)x;
    return r == 0 ? 1 : r;
}

static void buildObsNames(void) {
    static const char *weaponFields[IW_WEAPON_SIZE] = {"type", "ammo", "cooldown", "charge", "hold"};
    static const char *utilityFields[4] = {"type", "ammo", "regen", "active"};
    static const char *mineFields[MINE_OBS_SIZE] = {"in_flight", "welded", "x", "y"};
    static const char *grappleFields[GRAPPLE_OBS_SIZE] = {"on", "x", "y", "tension", "reeling", "hp", "att_none", "att_wall", "att_floating", "att_drone"};
    static const char *rayFields[2] = {"type", "dist"};
    static const char *enemyFields[] = {"rel_x", "rel_y", "dist", "vel_x", "vel_y", "dir_x", "dir_y", "aim_x", "aim_y",
                                        "shield_hp", "shield_dur", "energy", "depleted", "refill_wait", "braking", "burst_cd",
                                        "charging_burst", "burst_charge"};
    static const char *enemyTail[] = {"took_shot", "hit_shot", "lives", "alive", "respawn", "aim_to_enemy_sin", "aim_to_enemy_cos"};
    static const char *ownFields[] = {"trigger_held", "utility_held", "pos_x", "pos_y", "vel_x", "vel_y", "aim_x", "aim_y",
                                      "aim_vel_sin", "aim_vel_cos", "shield_hp", "shield_dur", "energy", "depleted", "refill_wait",
                                      "braking", "burst_cd", "charging_burst", "burst_charge"};
    static const char *ownTail[] = {"hit_shot", "took_shot", "own_shot_taken", "lives", "alive", "respawn"};
    static const char *ownRays[3] = {"aim_ray", "vel_ray", "drift_ray"};
    static const char *fwFields[IW_FLOATING_WALL_SIZE] = {"type", "x", "y", "sin", "cos", "vel_x", "vel_y"};
    static const char *projFields[IW_PROJECTILE_SIZE] = {"owner", "type", "x", "y", "vel_x", "vel_y"};
    static const char *misc[3] = {"not_easy_ep", "steps_left", "sudden_death_steps"};

    for (int i = 0; i < OBS_SIZE; i++) {
        snprintf(obsNames[i], sizeof(obsNames[i]), "obs%d", i);
    }
    for (int i = 0; i < IW_MAP_CELLS; i++) {
        snprintf(obsNames[i], sizeof(obsNames[i]), "grid_%d_%d", i / IW_GRID_SIZE, i % IW_GRID_SIZE);
    }
    for (int s = 0; s < IW_NUM_NEAR_WALLS; s++) {
        static const char *f[3] = {"type", "x", "y"};
        for (int k = 0; k < IW_NEAR_WALL_SIZE; k++) {
            snprintf(obsNames[IW_NEAR_WALL_OFFSET + s * IW_NEAR_WALL_SIZE + k], 48, "near_wall%d_%s", s, f[k]);
        }
    }
    for (int s = 0; s < IW_NUM_FLOATING_WALLS; s++) {
        for (int k = 0; k < IW_FLOATING_WALL_SIZE; k++) {
            snprintf(obsNames[IW_FLOATING_WALL_OFFSET + s * IW_FLOATING_WALL_SIZE + k], 48, "fw%d_%s", s, fwFields[k]);
        }
    }
    for (int s = 0; s < IW_NUM_PICKUPS; s++) {
        static const char *f[3] = {"type", "x", "y"};
        for (int k = 0; k < IW_PICKUP_SIZE; k++) {
            snprintf(obsNames[IW_PICKUP_OFFSET + s * IW_PICKUP_SIZE + k], 48, "pickup%d_%s", s, f[k]);
        }
    }
    for (int s = 0; s < IW_NUM_PROJECTILES; s++) {
        for (int k = 0; k < IW_PROJECTILE_SIZE; k++) {
            snprintf(obsNames[IW_PROJECTILE_OFFSET + s * IW_PROJECTILE_SIZE + k], 48, "proj%d_%s", s, projFields[k]);
        }
    }
    int i = E_OBS;
    for (int k = 0; k < IW_WEAPON_SIZE; k++) {
        snprintf(obsNames[i++], 48, "enemy_weapon_%s", weaponFields[k]);
    }
    for (int k = 0; k < 4; k++) {
        snprintf(obsNames[i++], 48, "enemy_utility_%s", utilityFields[k]);
    }
    for (int k = 0; k < GRAPPLE_OBS_SIZE; k++) {
        snprintf(obsNames[i++], 48, "enemy_grapple_%s", grappleFields[k]);
    }
    for (int k = 0; k < MINE_OBS_SIZE; k++) {
        snprintf(obsNames[i++], 48, "enemy_mine_%s", mineFields[k]);
    }
    for (size_t k = 0; k < sizeof(enemyFields) / sizeof(enemyFields[0]); k++) {
        snprintf(obsNames[i++], 48, "enemy_%s", enemyFields[k]);
    }
    for (size_t k = 0; k < sizeof(enemyTail) / sizeof(enemyTail[0]); k++) {
        snprintf(obsNames[i++], 48, "enemy_%s", enemyTail[k]);
    }
    assert(i == E_OBS + IW_ENEMY_RAY_IDX);
    for (int k = 0; k < 2; k++) {
        snprintf(obsNames[i++], 48, "enemy_ray_%s", rayFields[k]);
    }
    assert(i == D_OBS);
    for (int k = 0; k < IW_WEAPON_SIZE; k++) {
        snprintf(obsNames[i++], 48, "own_weapon_%s", weaponFields[k]);
    }
    for (int k = 0; k < 4; k++) {
        snprintf(obsNames[i++], 48, "own_utility_%s", utilityFields[k]);
    }
    for (int k = 0; k < GRAPPLE_OBS_SIZE; k++) {
        snprintf(obsNames[i++], 48, "own_grapple_%s", grappleFields[k]);
    }
    for (int k = 0; k < MINE_OBS_SIZE; k++) {
        snprintf(obsNames[i++], 48, "own_mine_%s", mineFields[k]);
    }
    for (size_t k = 0; k < sizeof(ownFields) / sizeof(ownFields[0]); k++) {
        snprintf(obsNames[i++], 48, "own_%s", ownFields[k]);
    }
    for (size_t k = 0; k < sizeof(ownTail) / sizeof(ownTail[0]); k++) {
        snprintf(obsNames[i++], 48, "own_%s", ownTail[k]);
    }
    assert(i == D_OBS + IW_DRONE_RAY_IDX);
    for (int r = 0; r < 3; r++) {
        for (int k = 0; k < 2; k++) {
            snprintf(obsNames[i++], 48, "own_%s_%s", ownRays[r], rayFields[k]);
        }
    }
    for (int k = 0; k < 3; k++) {
        snprintf(obsNames[i++], 48, "%s", misc[k]);
    }
    assert(i == OBS_SIZE);

    memset(obsGroupOf, -1, sizeof(obsGroupOf));
    for (int gi = NUM_PRIMARY_GROUPS - 1; gi >= 0; gi--) {
        for (int r = 0; r < 2; r++) {
            for (int k = 0; k < obsGroups[gi].ranges[r][1]; k++) {
                obsGroupOf[obsGroups[gi].ranges[r][0] + k] = (int8_t)gi;
            }
        }
    }
    int n = 0;
    for (int j = 0; j < OBS_SIZE; j++) {
        assert(obsGroupOf[j] >= 0);
        if (iw_is_scalar(j)) {
            scalarIdx[n++] = j;
        }
    }
    assert(n == IW_NUM_SCALARS);
}

static bool parseMask(const char *spec, obsMask *mask) {
    memset(mask, 0x0, sizeof(*mask));
    snprintf(mask->name, sizeof(mask->name), "%s", spec);
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", spec);
    for (char *tok = strtok(buf, ","); tok != NULL; tok = strtok(NULL, ",")) {
        int lo, hi;
        if (sscanf(tok, "%d-%d", &lo, &hi) == 2 && lo >= 0 && hi < OBS_SIZE && lo <= hi) {
            for (int i = lo; i <= hi; i++) {
                mask->idx[i] = true;
            }
            continue;
        }
        bool found = false;
        for (size_t gi = 0; gi < NUM_OBS_GROUPS; gi++) {
            if (strcmp(obsGroups[gi].name, tok) != 0) {
                continue;
            }
            for (int r = 0; r < 2; r++) {
                for (int k = 0; k < obsGroups[gi].ranges[r][1]; k++) {
                    mask->idx[obsGroups[gi].ranges[r][0] + k] = true;
                }
            }
            found = true;
        }
        if (!found) {
            fprintf(stderr, "unknown obs group %s\n", tok);
            return false;
        }
    }
    return true;
}

// ablates the masked obs of every game in the batch; swap copies them from
// the game half a batch away, src holds the unablated obs
static void applyMask(float *obs, const float *src, const int B, const obsMask *mask, const enum ablateMode mode) {
    for (int b = 0; b < B; b++) {
        const float *other = src + ((b + B / 2) % B) * OBS_SIZE;
        float *o = obs + b * OBS_SIZE;
        for (int i = 0; i < OBS_SIZE; i++) {
            if (mask->idx[i]) {
                o[i] = mode == ABLATE_SWAP && B > 1 ? other[i] : 0.0f;
            }
        }
    }
}

static void netForward(policyBatch *p) {
    ImpulseWarsNet *net = p->net;
    mingru_zero_term(net->mingru, p->terminals);
    impulse_wars_encode(net, p->obs);
    iw_mingru(net->mingru, net->proj->output);
    iw_linear(net->decoder, net->mingru->output);
}

// sample actions for the whole batch with each game's own rng
static void sampleActions(policyBatch *p) {
    const int actSizes[] = ACT_SIZES;
    const int atnSum = impulse_wars_atn_sum();
    for (int b = 0; b < p->size; b++) {
        // logits for every action head, followed by the value
        const float *logits = p->net->decoder->output + b * (atnSum + 1);
        for (int a = 0; a < NUM_ATNS; a++) {
            const int n = actSizes[a];
            int best = 0;
            for (int i = 1; i < n; i++) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += expf(logits[i] - logits[best]);
            }
            const float target = ((float)rand_r(&p->rng[b]) / (float)RAND_MAX) * sum;
            float acc = 0.0f;
            int choice = best;
            for (int i = 0; i < n; i++) {
                acc += expf(logits[i] - logits[best]);
                if (target < acc) {
                    choice = i;
                    break;
                }
            }
            p->actions[b * NUM_ATNS + a] = (float)choice;
            logits += n;
        }
    }
}

static void logSoftmax(const float *x, const int n, float *out) {
    float m = x[0];
    for (int i = 1; i < n; i++) {
        m = fmaxf(m, x[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += expf(x[i] - m);
    }
    const float lse = m + logf(sum);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] - lse;
    }
}

// KL(policy || shadow) per action head and the value change of shadow
// game b, added to its stats
static void shadowCompare(const policyBatch *p, const policyBatch *sh, const int b, shadowStats *s) {
    const int actSizes[] = ACT_SIZES;
    const int atnSum = impulse_wars_atn_sum();
    const float *lp = p->net->decoder->output + b * (atnSum + 1);
    const float *lq = sh->net->decoder->output + b * (atnSum + 1);
    float pl[AIM_DIRS], ql[AIM_DIRS];
    for (int a = 0; a < NUM_ATNS; a++) {
        const int n = actSizes[a];
        logSoftmax(lp, n, pl);
        logSoftmax(lq, n, ql);
        double kl = 0.0;
        for (int i = 0; i < n; i++) {
            kl += expf(pl[i]) * (pl[i] - ql[i]);
        }
        s->kl[a] += kl;
        lp += n;
        lq += n;
    }
    const float dv = *lq - *lp;
    s->dv += dv;
    s->absDv += fabsf(dv);
    s->n++;
}

static void bindAgent(iwEnv *e, const uint8_t agent, policyBatch *p, const int b) {
    e->agents[agent].observations = p->obs + b * OBS_SIZE;
    e->agents[agent].actions = p->actions + b * NUM_ATNS;
    e->agents[agent].rewards = p->rewards + b;
    e->agents[agent].terminals = p->terminals + b;
}

// the policy plays drone seat; the opponent policy, if any, plays the other
static void initAnalyzeEnv(iwEnv *e, policyBatch *p, policyBatch *opp, const int b, const uint8_t seat) {
    initEnv(e, 2, opp != NULL ? 2 : 1, -1, 1, false, true, false, BOT_SCRIPTED, 0.0f, 0.0f);
    bindAgent(e, seat, p, b);
    if (opp != NULL) {
        bindAgent(e, 1 - seat, opp, b);
    }
    setRewards(e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    e->easyEpProb = 0.0f;
    // every game is a combo episode, so pickups are ammo crates; the
    // harness then replaces the env's random loadouts with the matchup's
    if (crates) {
        e->comboTrainingProb = 1.0f;
    }
}

static bool lineOfSight(iwEnv *e, const droneEntity *a, const droneEntity *b) {
    const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
    return !b2World_CastRayClosest(e->worldID, a->pos, b2Sub(b->pos, a->pos), filter).hit;
}

// distances from the drone's center to the closest static and floating
// death walls
static void deathWallDistances(iwEnv *e, const droneEntity *d, float *staticDist, float *floatingDist) {
    *staticDist = FLT_MAX;
    *floatingDist = FLT_MAX;
    if (d->mapCellIdx >= 0) {
        const nearWall *walls = e->map->nearestWalls + (MAX_NEAREST_WALLS * d->mapCellIdx);
        for (uint8_t i = 0; i < MAX_NEAREST_WALLS; i++) {
            if (walls[i].type == DEATH_WALL_ENTITY) {
                *staticDist = min(*staticDist, staticWallDistance(walls[i].pos, d->pos));
            }
        }
    }
    for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        const wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        if (wall->type == DEATH_WALL_ENTITY) {
            *floatingDist = min(*floatingDist, floatingWallDistance(wall, d->pos));
        }
    }
}

static slotState readSlots(const droneEntity *d) {
    return (slotState){
        .weapon = d->weapon.weaponInfo->type,
        .ammo = d->weapon.ammo,
    };
}

static void snapshot(gameSlot *slot, const droneAnalysisStats *analysis) {
    iwEnv *e = slot->e;
    memcpy(slot->prev, e->stats, sizeof(slot->prev));
    for (uint8_t i = 0; i < 2; i++) {
        slot->prevSlots[i] = readSlots(safe_array_get_at(e->drones, i));
        memcpy(slot->prevKills[i], analysis[i].killsByWeapon, sizeof(slot->prevKills[i]));
        memcpy(slot->prevUtilityAmmo[i], analysis[i].utilityAmmoSpent, sizeof(slot->prevUtilityAmmo[i]));
        memcpy(slot->prevUtilityHits[i], analysis[i].utilityHits, sizeof(slot->prevUtilityHits[i]));
    }
}

// sets the types of the starting utility pickups, which are the first
// pickups, created one nearest each drone in drone order
static void setStartUtilities(iwEnv *e, const uint8_t *utilities) {
    for (uint8_t i = 0; i < e->numDrones && i < cc_array_size(e->pickups); i++) {
        weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
        if (pickup->kind != UTILITY_PICKUP) {
            return;
        }
        e->spawnedUtilityPickups[pickup->utility]--;
        pickup->utility = utilities[i];
        e->spawnedUtilityPickups[pickup->utility]++;
    }
}

static void startGame(gameSlot *slot, policyBatch *p, policyBatch *opp, const int b, gameResult *g) {
    iwEnv *e = slot->e;
    // box2d is only deterministic for identical creation order and id
    // pool state, so every game gets a fresh world
    pthread_mutex_lock(&envMu);
    puf_close(e);
    memset(e, 0x0, sizeof(*e));
    initAnalyzeEnv(e, p, opp, b, g->seat);
    pthread_mutex_unlock(&envMu);

    e->pinnedMapIdx = g->map;
    e->rng = g->seed;
    e->botPersonality[1] = opps[g->opp];
    e->botFeatures[1] = botPresets[opps[g->opp]].feats;
    memset(&e->log, 0x0, sizeof(e->log));
    puf_reset(e);
    setStartUtilities(e, g->startUtilities);
    if (combos) {
        for (uint8_t i = 0; i < 2; i++) {
            droneSetLoadout(safe_array_get_at(e->drones, i == 0 ? g->seat : 1 - g->seat), g->loadout[i], g->utilityLoadout[i]);
        }
    }
    if (noPickups) {
        for (size_t i = 0; i < cc_array_size(e->pickups); i++) {
            weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
            if (pickup->respawnWait == 0.0f) {
                disableWeaponPickup(e, pickup);
            }
            // counts down from here so it never respawns
            pickup->respawnWait = FLT_MAX;
        }
    }
    computeObs(e);
    // a fresh recurrent state for every game
    p->terminals[b] = 1.0f;
    p->rng[b] = g->seed;
    if (opp != NULL) {
        opp->terminals[b] = 1.0f;
        opp->rng[b] = hashSeed(g->seed);
    }

    slot->g = g;
    slot->step = 0;
    slot->blockedRun = 0;
    slot->peeker = -1;
    slot->peekStep = -1000;
    slot->active = true;
    snapshot(slot, e->analysisStats);
    for (uint8_t i = 0; i < 2; i++) {
        for (uint8_t w = 0; w < NW; w++) {
            slot->lastFired[i][w] = -1000;
            slot->lastHit[i][w] = -1000;
        }
        g->behavior[i].firstLosStep = -1;
        g->behavior[i].firstPickupStep = -1;
    }
}

static inline behaviorStats *droneBehavior(gameSlot *slot, const uint8_t drone) {
    return &slot->g->behavior[drone == slot->g->seat ? 0 : 1];
}

// records the weapons each drone fired this step; returns shots fired
static float recordFired(gameSlot *slot, const uint8_t i, const droneStats *cur) {
    float fired = 0.0f;
    for (uint8_t w = 0; w < NW; w++) {
        const float shots = cur->shotsFired[w] - slot->prev[i].shotsFired[w];
        if (shots <= 0.0f) {
            continue;
        }
        fired += shots;
        slot->lastFired[i][w] = slot->step;
    }
    return fired;
}

// records the weapons each drone hit the enemy with this step
static void recordHits(gameSlot *slot, const uint8_t i, const droneStats *cur, const droneEntity *d) {
    for (uint8_t w = 0; w < NW; w++) {
        if (cur->shotsHit[w] > slot->prev[i].shotsHit[w]) {
            slot->lastHit[i][w] = slot->step;
        }
    }
    if (d != NULL && d->grapple != NULL && d->grapple->welded && d->grapple->grappledEnt->type == DRONE_ENTITY) {
        slot->lastHit[i][GRAPPLE_WEAPON] = slot->step;
    }
}

// records the utilities each drone fired and hit with this step, as the
// weapon their projectiles are, for combo and setup kills; utility
// projectiles aren't counted as shots
static void recordUtilityUse(gameSlot *slot, const droneAnalysisStats *analysis) {
    for (uint8_t i = 0; i < 2; i++) {
        for (uint8_t u = UNSET_UTILITY + 1; u < _NUM_UTILITIES; u++) {
            const enum weaponType w = utilityInfos[u]->projectileInfo->type;
            if (analysis[i].utilityAmmoSpent[u] > slot->prevUtilityAmmo[i][u]) {
                slot->lastFired[i][w] = slot->step;
            }
            if (analysis[i].utilityHits[u] > slot->prevUtilityHits[i][u]) {
                slot->lastHit[i][w] = slot->step;
            }
        }
    }
}

// attributes this step's weapon kills to combos and setups
static void recordKills(gameSlot *slot, const droneAnalysisStats *analysis) {
    for (uint8_t i = 0; i < 2; i++) {
        behaviorStats *s = droneBehavior(slot, i);
        for (uint8_t w = 0; w < NW; w++) {
            const float kills = analysis[i].killsByWeapon[w] - slot->prevKills[i][w];
            if (kills <= 0.0f) {
                continue;
            }
            bool multi = false;
            for (uint8_t o = 0; o < NW; o++) {
                if (o != w && (int32_t)slot->step - slot->lastFired[i][o] <= COMBO_WINDOW) {
                    s->comboKills[w * NW + o] += kills;
                    multi = true;
                }
            }
            s->multiWeaponKills += multi * kills;
            bool setup = false;
            for (uint8_t o = 0; o < NW; o++) {
                if (o != w && (int32_t)slot->step - slot->lastHit[i][o] <= SETUP_WINDOW) {
                    s->setupKills[w * NW + o] += kills;
                    setup = true;
                }
            }
            s->setupKillsAny += setup * kills;
        }
    }
}

// weapon changes of a live drone over the last step
static void observeSlots(gameSlot *slot, const uint8_t i, const droneEntity *d) {
    iwEnv *e = slot->e;
    behaviorStats *s = droneBehavior(slot, i);
    const slotState *prev = &slot->prevSlots[i];
    const slotState cur = readSlots(d);
    const droneStats *stats = &e->stats[i];

    s->held[cur.weapon]++;
    s->standardOnly += cur.weapon == STANDARD_WEAPON;

    const enum utilityType utility = d->utility.utilityInfo->type;
    s->utilityHeld[utility]++;
    s->utilityReady[utility] += d->utility.ammo > 0;
    if (d->grapple != NULL && d->grapple->welded) {
        if (d->utilityHeld) {
            s->grappleReeling++;
            s->grappleReelSpeed += b2Length(d->velocity);
        } else {
            s->grappleTethered++;
        }
    }

    const bool hadPickupWeapon = prev->weapon > STANDARD_WEAPON;
    float ammoFrac = 0.0f;
    if (hadPickupWeapon) {
        ammoFrac = (float)prev->ammo / (float)weaponInfos[prev->weapon]->ammo;
        s->ammoFrac += ammoFrac;
        s->ammoSteps++;
    }
    if (stats->cratesPickedUp > slot->prev[i].cratesPickedUp) {
        if (hadPickupWeapon) {
            s->crateAmmoFrac += ammoFrac;
            s->crateAmmoN++;
        } else {
            s->crateNoWeapons++;
        }
    }

    const float picked = stats->totalWeaponsPickedUp - slot->prev[i].totalWeaponsPickedUp;
    if (picked > 0.0f) {
        enum weaponType pickup = UNSET_WEAPON;
        for (uint8_t w = 0; w < NW; w++) {
            if (stats->weaponsPickedUp[w] > slot->prev[i].weaponsPickedUp[w]) {
                pickup = w;
            }
        }
        if (pickup == prev->weapon) {
            s->pickupTopUp++;
        } else if (prev->weapon == STANDARD_WEAPON) {
            s->pickupToHand++;
        } else {
            s->pickupReplace++;
            s->dropped[prev->weapon]++;
        }
    } else if (cur.weapon != prev->weapon) {
        s->autoSwitches++;
    }
}

// --loadout-pickups: replaces a weapon outside the drone's loadout, which
// only comes from pickups, with its loadout weapon at full ammo; returns
// true if the weapon was replaced
static bool enforceLoadout(gameSlot *slot, const uint8_t i, droneEntity *d) {
    const uint8_t side = i == slot->g->seat ? 0 : 1;
    const uint8_t loadout = slot->g->loadout[side];
    const uint8_t utilityLoadout = slot->g->utilityLoadout[side];
    bool replaced = false;
    weaponSlot *weapon = &d->weapon;
    const enum weaponType w = weapon->weaponInfo->type;
    if (w != STANDARD_WEAPON && w != loadout) {
        weapon->weaponInfo = weaponInfos[loadout];
        resetWeapon(weapon);
        weapon->ammo = weapon->weaponInfo->ammo;
        replaced = true;
    }
    if (d->utility.utilityInfo->type != utilityLoadout) {
        droneChangeUtility(slot->e, d, utilityLoadout);
        replaced = true;
    }
    return replaced;
}

// speed of drone i towards the other drone
static float approachSpeed(droneEntity *const drones[2], const uint8_t i) {
    const b2Vec2 dir = b2Normalize(b2Sub(drones[1 - i]->pos, drones[i]->pos));
    return b2Dot(drones[i]->velocity, dir);
}

// per drone situation stats, and standoffs: long stretches close to each
// other with a wall between, who breaks them and what happens after
static void observeStandoff(gameSlot *slot, droneEntity *const drones[2], const bool bothAlive, const bool los, const float sep) {
    gameResult *g = slot->g;
    iwEnv *e = slot->e;
    const bool close = bothAlive && !los && sep < STANDOFF_DIST;

    // kills after a peek; a kill also ends a standoff, so check before
    // the run is reset
    if (slot->peeker >= 0 && (int32_t)slot->step - slot->peekStep <= STANDOFF_AFTER) {
        for (uint8_t i = 0; i < 2; i++) {
            if (e->stats[i].kills > slot->prev[i].kills) {
                behaviorStats *peeker = &g->behavior[slot->peeker == g->seat ? 0 : 1];
                behaviorStats *peeked = &g->behavior[slot->peeker == g->seat ? 1 : 0];
                if (i == slot->peeker) {
                    peeker->peekKills++;
                    peeked->peekedDeaths++;
                } else {
                    peeker->peekDeaths++;
                    peeked->peekedKills++;
                }
                slot->peeker = -1;
            }
        }
    }

    if (close) {
        slot->blockedRun++;
    } else {
        if (slot->blockedRun >= STANDOFF_MIN_STEPS) {
            for (uint8_t i = 0; i < 2; i++) {
                behaviorStats *s = &g->behavior[i == g->seat ? 0 : 1];
                s->standoffs++;
                s->standoffSteps += slot->blockedRun;
            }
            if (bothAlive && los) {
                const uint8_t p = approachSpeed(drones, 0) >= approachSpeed(drones, 1) ? 0 : 1;
                g->behavior[p == g->seat ? 0 : 1].peeks++;
                slot->peeker = p;
                slot->peekStep = slot->step;
            }
        }
        slot->blockedRun = 0;
    }

    if (!bothAlive) {
        return;
    }
    const uint8_t sit = los ? STANDOFF_LOS : (close ? STANDOFF_CLOSE : STANDOFF_FAR);
    for (uint8_t i = 0; i < 2; i++) {
        behaviorStats *s = &g->behavior[i == g->seat ? 0 : 1];
        const int lead = (int)drones[i]->livesLeft - (int)drones[1 - i]->livesLeft;
        const uint8_t k = sit * 3 + (lead < 0 ? 0 : (lead == 0 ? 1 : 2));
        s->sitSteps[k]++;
        s->sitApproach[k] += approachSpeed(drones, i);
        s->sitSpeed[k] += b2Length(drones[i]->velocity);
        s->sitTrigger[k] += e->cachedActions[i].trigger;
    }
}

static void observeStep(gameSlot *slot) {
    iwEnv *e = slot->e;
    gameResult *g = slot->g;
    droneEntity *drones[2] = {safe_array_get_at(e->drones, 0), safe_array_get_at(e->drones, 1)};
    const bool bothAlive = !drones[0]->dead && !drones[1]->dead;
    const bool los = bothAlive && lineOfSight(e, drones[0], drones[1]);
    const float sep = bothAlive ? b2Distance(drones[0]->pos, drones[1]->pos) : 0.0f;
    observeStandoff(slot, drones, bothAlive, los, sep);

    for (uint8_t i = 0; i < 2; i++) {
        const droneEntity *d = drones[i];
        const droneEntity *enemy = drones[1 - i];
        behaviorStats *s = &g->behavior[i == g->seat ? 0 : 1];
        const droneStats *cur = &e->stats[i];
        const droneStats *prev = &slot->prev[i];

        if (s->firstPickupStep < 0 && cur->totalWeaponsPickedUp > prev->totalWeaponsPickedUp) {
            s->firstPickupStep = slot->step;
        }
        const float fired = recordFired(slot, i, cur);
        recordHits(slot, i, cur, d);
        if (d->dead) {
            continue;
        }

        const agentActions *a = &e->cachedActions[i];
        const float speed = b2Length(d->velocity);
        observeSlots(slot, i, d);
        s->alive++;
        s->speed += speed;
        s->energy += d->energyLeft;
        s->trigger += a->trigger;
        s->brake += a->brake;
        s->burstCharge += a->chargingBurst;
        s->moveNoop += b2Length(a->move) < 0.01f;
        const weaponSlot *active = &d->weapon;
        if (active->weaponInfo->maxCharge != 0.0f && d->triggerHeld) {
            s->charging++;
            s->chargeHeldFull += active->charge >= active->weaponInfo->maxCharge;
        }
        if (bothAlive) {
            s->sepSum += sep;
            s->sepSteps++;
            s->closeSteps += sep < 15.0f;
            s->farSteps += sep > 40.0f;
            if (los) {
                s->losSteps++;
                s->triggerLos += a->trigger;
                s->speedLos += speed;
                if (s->firstLosStep < 0) {
                    s->firstLosStep = slot->step;
                }
            } else {
                s->noLosSteps++;
                s->triggerNoLos += a->trigger;
                s->speedNoLos += speed;
            }
        }
        float staticDist, floatingDist;
        deathWallDistances(e, d, &staticDist, &floatingDist);
        const float deathDist = min(staticDist, floatingDist) - DRONE_RADIUS;
        s->nearDeath3 += deathDist < 3.0f;
        s->nearDeath6 += deathDist < 6.0f;
        s->nearFloatingDeath6 += floatingDist - DRONE_RADIUS < 6.0f;

        if (bothAlive) {
            for (uint8_t w = 0; w < NW; w++) {
                const float shots = cur->shotsFired[w] - prev->shotsFired[w];
                if (shots > 0.0f) {
                    s->fireDist[w] += sep * shots;
                    s->fireLos[w] += los * shots;
                }
            }
        }
        if (fired > 0.0f && bothAlive) {
            const float err = fabsf(signedAngle(d->lastAim, b2Sub(enemy->pos, d->pos)));
            const bool aimed10 = err < 10.0f * PI / 180.0f;
            s->fireSteps++;
            s->fireAimErr += err;
            s->fireAim10 += aimed10;
            if (!los) {
                s->fireNoLos++;
                s->fireNoLosAim10 += aimed10;
            }
        }
    }
    recordUtilityUse(slot, e->analysisStats);
    recordKills(slot, e->analysisStats);
    if (loadoutPickups) {
        bool replaced = false;
        for (uint8_t i = 0; i < 2; i++) {
            if (!drones[i]->dead) {
                replaced |= enforceLoadout(slot, i, drones[i]);
            }
        }
        if (replaced) {
            computeObs(e);
        }
    }
    snapshot(slot, e->analysisStats);
}

// one agent step of a game; returns true when the game ended
static bool stepGame(gameSlot *slot, policyBatch *p, const int b) {
    iwEnv *e = slot->e;
    gameResult *g = slot->g;
    const float *act = p->actions + b * NUM_ATNS;
    if (!((droneEntity *)safe_array_get_at(e->drones, g->seat))->dead) {
        const uint16_t aimDir = (uint16_t)act[1] * AIM_FINE_STEPS + (uint16_t)act[2];
        g->moveHist[(uint8_t)act[0]]++;
        g->aimHist[aimDir]++;
        if (act[3] > 0.0f) {
            g->aimFireHist[aimDir]++;
        }
    }
    slot->step++;
    puf_step(e);
    if (e->log.n != 0.0f) {
        // the env was reset, the final step's shots and kills are in the
        // episode log and the last analysis stats
        for (uint8_t i = 0; i < 2; i++) {
            recordFired(slot, i, &e->log.stats[i]);
            recordHits(slot, i, &e->log.stats[i], NULL);
        }
        recordUtilityUse(slot, e->lastAnalysisStats);
        recordKills(slot, e->lastAnalysisStats);
        g->length = e->log.length;
        for (uint8_t i = 0; i < 2; i++) {
            const uint8_t drone = i == 0 ? g->seat : 1 - g->seat;
            g->stats[i] = e->log.stats[drone];
            g->analysis[i] = e->lastAnalysisStats[drone];
        }
        return true;
    }
    observeStep(slot);
    return false;
}

static bool nextGame(gameResult **out) {
    const uint32_t job = atomic_fetch_add(&nextJob, 1);
    if (job >= totalGames) {
        return false;
    }
    uint32_t gameIdx = job % gamesPerOpp;
    gameResult *g = &results[job];
    memset(g, 0x0, sizeof(*g));
    // against a policy every seed is played from both drones
    if (oppWeights != NULL) {
        g->seat = gameIdx % 2;
        gameIdx /= 2;
    }
    if (combos) {
        const uint32_t m = job / gamesPerOpp;
        g->opp = 0;
        g->map = tourMaps[(m * 7 + gameIdx) % numMaps];
        g->seed = hashSeed(baseSeed * 1000003ull + m * 1009ull + gameIdx);
        for (uint8_t i = 0; i < 2; i++) {
            g->loadout[i] = comboLoadouts[matchups[m][i]][0];
            g->utilityLoadout[i] = comboLoadouts[matchups[m][i]][1];
        }
    } else {
        g->opp = job / gamesPerOpp;
        g->map = tourMaps[gameIdx % numMaps];
        g->seed = hashSeed(baseSeed * 1000003ull + gameIdx);
    }
    // cycle the starting utility pickups over every pair of utilities,
    // shifted every round of maps so each map gets every pair too; drone
    // order, so a policy playing both drones of a seed plays both sides
    const uint32_t numUtilities = _NUM_UTILITIES - 1;
    const uint32_t pair = (gameIdx + gameIdx / numMaps) % (numUtilities * numUtilities);
    g->startUtilities[0] = UNSET_UTILITY + 1 + pair / numUtilities;
    g->startUtilities[1] = UNSET_UTILITY + 1 + pair % numUtilities;
    *out = g;
    return true;
}

static policyBatch *createPolicyBatch(const Weights *weights, const int B, const int hidden, const int layers) {
    policyBatch *p = calloc(1, sizeof(policyBatch));
    p->size = B;
    p->obs = calloc(B * OBS_SIZE, sizeof(float));
    p->actions = calloc(B * NUM_ATNS, sizeof(float));
    p->rewards = calloc(B, sizeof(float));
    p->terminals = calloc(B, sizeof(float));
    p->rng = calloc(B, sizeof(unsigned int));
    // layers point into the shared weight data, only the read index is
    // per net
    Weights *w = malloc(sizeof(Weights));
    *w = *weights;
    w->idx = 0;
    p->net = init_impulse_wars_net(w, B, hidden, layers);
    return p;
}

// per worker input statistics for the weights output
typedef struct inputStats {
    double concatSum[IW_CONCAT];
    double concatSq[IW_CONCAT];
    double concatN;
    double projSum[IW_PROJ_FEATURES];
    double projSq[IW_PROJ_FEATURES];
    double projN;
    double fwSum[IW_FW_FEATURES];
    double fwSq[IW_FW_FEATURES];
    double fwN;
} inputStats;

static void accumulateInputs(inputStats *st, const ImpulseWarsNet *net, const int b) {
    const float *c = net->concat + b * IW_CONCAT;
    for (int j = 0; j < IW_CONCAT; j++) {
        st->concatSum[j] += c[j];
        st->concatSq[j] += (double)c[j] * c[j];
    }
    st->concatN++;
    for (int r = 0; r < IW_NUM_PROJECTILES; r++) {
        const int row = b * IW_NUM_PROJECTILES + r;
        if (!net->rec_active[row]) {
            continue;
        }
        for (int f = 0; f < IW_PROJ_FEATURES; f++) {
            const float v = net->rec[row * IW_PROJ_FEATURES + f];
            st->projSum[f] += v;
            st->projSq[f] += (double)v * v;
        }
        st->projN++;
    }
    for (int r = 0; r < IW_NUM_FLOATING_WALLS; r++) {
        const int row = b * IW_NUM_FLOATING_WALLS + r;
        if (!net->fw_rec_active[row]) {
            continue;
        }
        for (int f = 0; f < IW_FW_FEATURES; f++) {
            const float v = net->fw_rec[row * IW_FW_FEATURES + f];
            st->fwSum[f] += v;
            st->fwSq[f] += (double)v * v;
        }
        st->fwN++;
    }
}

static void mergeInputs(const inputStats *st) {
    pthread_mutex_lock(&statsMu);
    for (int j = 0; j < IW_CONCAT; j++) {
        concatSum[j] += st->concatSum[j];
        concatSq[j] += st->concatSq[j];
    }
    concatN += st->concatN;
    for (int f = 0; f < IW_PROJ_FEATURES; f++) {
        projRecSum[f] += st->projSum[f];
        projRecSq[f] += st->projSq[f];
    }
    projRecN += st->projN;
    for (int f = 0; f < IW_FW_FEATURES; f++) {
        fwRecSum[f] += st->fwSum[f];
        fwRecSq[f] += st->fwSq[f];
    }
    fwRecN += st->fwN;
    pthread_mutex_unlock(&statsMu);
}

static void *worker(void *arg) {
    MAYBE_UNUSED(arg);
    const int B = batchSize;
    policyBatch *p = createPolicyBatch(policyWeights, B, policyHidden, policyLayers);
    policyBatch *opp = NULL;
    if (oppWeights != NULL) {
        opp = createPolicyBatch(oppWeights, B, oppHidden, oppLayers);
    }
    policyBatch *shadows[MAX_SHADOWS];
    for (int k = 0; k < numShadows; k++) {
        shadows[k] = createPolicyBatch(policyWeights, B, policyHidden, policyLayers);
    }
    float *cleanObs = calloc(B * OBS_SIZE, sizeof(float));
    inputStats *inputs = weightsOut != NULL ? calloc(1, sizeof(inputStats)) : NULL;

    gameSlot *slots = calloc(B, sizeof(gameSlot));
    pthread_mutex_lock(&envMu);
    for (int b = 0; b < B; b++) {
        slots[b].e = fastCalloc(1, sizeof(iwEnv));
        initAnalyzeEnv(slots[b].e, p, opp, b, 0);
    }
    if (!mapsReady) {
        initMaps(slots[0].e);
        mapsReady = true;
    }
    pthread_mutex_unlock(&envMu);

    int active = 0;
    for (int b = 0; b < B; b++) {
        gameResult *g;
        if (nextGame(&g)) {
            startGame(&slots[b], p, opp, b, g);
            active++;
        }
    }
    while (active > 0) {
        memcpy(cleanObs, p->obs, B * OBS_SIZE * sizeof(float));
        for (int k = 0; k < numShadows; k++) {
            memcpy(shadows[k]->obs, cleanObs, B * OBS_SIZE * sizeof(float));
            memcpy(shadows[k]->terminals, p->terminals, B * sizeof(float));
            applyMask(shadows[k]->obs, cleanObs, B, &shadowMasks[k], ablateMode);
            netForward(shadows[k]);
        }
        if (ablating) {
            applyMask(p->obs, cleanObs, B, &ablateMask, ablateMode);
        }
        netForward(p);
        sampleActions(p);
        if (opp != NULL) {
            netForward(opp);
            sampleActions(opp);
        }
        for (int b = 0; b < B; b++) {
            gameSlot *slot = &slots[b];
            if (!slot->active || ((droneEntity *)safe_array_get_at(slot->e->drones, slot->g->seat))->dead) {
                continue;
            }
            for (int k = 0; k < numShadows; k++) {
                shadowCompare(p, shadows[k], b, &slot->g->shadow[k]);
            }
            if (inputs != NULL) {
                accumulateInputs(inputs, p->net, b);
            }
        }
        for (int b = 0; b < B; b++) {
            gameSlot *slot = &slots[b];
            if (!slot->active || !stepGame(slot, p, b)) {
                continue;
            }
            slot->active = false;
            active--;
            atomic_fetch_add(&jobsDone, 1);
            gameResult *g;
            if (nextGame(&g)) {
                startGame(slot, p, opp, b, g);
                active++;
            }
        }
    }
    if (inputs != NULL) {
        mergeInputs(inputs);
        free(inputs);
    }

    pthread_mutex_lock(&envMu);
    for (int b = 0; b < B; b++) {
        puf_close(slots[b].e);
        fastFree(slots[b].e);
    }
    pthread_mutex_unlock(&envMu);
    return NULL;
}

static void writeArray(FILE *f, const char *key, const float *v, const int n) {
    fprintf(f, "\"%s\":[", key);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s%g", i ? "," : "", v[i]);
    }
    fprintf(f, "],");
}

static void writeArrayD(FILE *f, const char *key, const double *v, const int n) {
    fprintf(f, "\"%s\":[", key);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s%g", i ? "," : "", v[i]);
    }
    fprintf(f, "],");
}

static void writeArrayU(FILE *f, const char *key, const uint32_t *v, const int n) {
    fprintf(f, "\"%s\":[", key);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s%u", i ? "," : "", v[i]);
    }
    fprintf(f, "]");
}

#define WRITE_F(name) fprintf(f, "\"" #name "\":%g,", (double)s->name)
#define WRITE_B(name) fprintf(f, "\"" #name "\":%g,", b->name)
#define WRITE_BA(name, n) writeArrayD(f, #name, b->name, n)
#define WRITE_A(name, n) writeArray(f, #name, (const float *)a->name, n)

static void writeDrone(FILE *f, const gameResult *g, const uint8_t i) {
    const droneStats *s = &g->stats[i];
    const droneAnalysisStats *a = &g->analysis[i];
    const behaviorStats *b = &g->behavior[i];
    const droneStats *o = &g->stats[1 - i];
    fprintf(f, "{");
    WRITE_F(score);
    WRITE_F(kills);
    WRITE_F(selfKills);
    WRITE_F(unknownKills);
    WRITE_F(assists);
    WRITE_F(cratesPickedUp);
    WRITE_F(cratesRefilled);
    fprintf(f, "\"deaths\":%g,", (double)(o->kills + s->selfKills + s->unknownKills));
    WRITE_F(totalBursts);
    WRITE_F(burstsHit);
    WRITE_F(energyEmptied);
    WRITE_F(shieldsBroken);
    WRITE_F(ownShieldBroken);
    writeArray(f, "shotsFired", s->shotsFired, NW);
    writeArray(f, "utilitiesPickedUp", s->utilitiesPickedUp, _NUM_UTILITIES);
    writeArray(f, "shotsHit", s->shotsHit, NW);
    writeArray(f, "weaponsPickedUp", s->weaponsPickedUp, NW);
    WRITE_A(killsByWeapon, NW);
    WRITE_A(selfKillsByWeapon, NW);
    WRITE_A(deathsBySource, NUM_PHYSICS_TYPES);
    WRITE_A(selfKillsBySource, NUM_PHYSICS_TYPES);
    fprintf(f, "\"floatingWallDeaths\":%g,\"floatingWallSelfKills\":%g,", a->floatingWallDeaths, a->floatingWallSelfKills);
    WRITE_A(assistsByWeapon, NW);
    WRITE_A(diedHoldingActive, NW);
    WRITE_A(pickupsFrom, NW * NW);
    WRITE_A(boostedShotsFired, NW);
    WRITE_A(boostedShotsHit, NW);
    WRITE_A(boostedKills, NW);
    WRITE_A(boostedSelfKills, NW);
    WRITE_A(boostedSpeedBefore, NW);
    WRITE_A(utilityPresses, _NUM_UTILITIES);
    WRITE_A(utilityPressesUseful, _NUM_UTILITIES);
    WRITE_A(utilityAmmoSpent, _NUM_UTILITIES);
    WRITE_A(utilityHits, _NUM_UTILITIES);
    WRITE_A(grappleAttaches, 3);
    fprintf(f, "\"grappleAttachedDeaths\":%g,\"minesDetonated\":%g,", a->grappleAttachedDeaths, a->minesDetonated);
    WRITE_A(boostedSpeedAfter, NW);
    WRITE_BA(held, NW);
    WRITE_BA(fireDist, NW);
    WRITE_BA(fireLos, NW);
    WRITE_BA(dropped, NW);
    WRITE_BA(comboKills, NW * NW);
    WRITE_BA(setupKills, NW * NW);
    WRITE_BA(utilityHeld, _NUM_UTILITIES);
    WRITE_BA(utilityReady, _NUM_UTILITIES);
    WRITE_B(alive);
    WRITE_B(trigger);
    WRITE_B(triggerLos);
    WRITE_B(triggerNoLos);
    WRITE_B(brake);
    WRITE_B(burstCharge);
    WRITE_B(moveNoop);
    WRITE_B(speed);
    WRITE_B(speedLos);
    WRITE_B(speedNoLos);
    WRITE_B(energy);
    WRITE_B(charging);
    WRITE_B(chargeHeldFull);
    WRITE_B(losSteps);
    WRITE_B(noLosSteps);
    WRITE_B(sepSum);
    WRITE_B(sepSteps);
    WRITE_B(closeSteps);
    WRITE_B(farSteps);
    WRITE_B(nearDeath3);
    WRITE_BA(sitSteps, 9);
    WRITE_BA(sitApproach, 9);
    WRITE_BA(sitSpeed, 9);
    WRITE_BA(sitTrigger, 9);
    WRITE_B(standoffs);
    WRITE_B(standoffSteps);
    WRITE_B(peeks);
    WRITE_B(peekKills);
    WRITE_B(peekDeaths);
    WRITE_B(peekedKills);
    WRITE_B(peekedDeaths);
    WRITE_B(nearDeath6);
    WRITE_B(nearFloatingDeath6);
    WRITE_B(firstLosStep);
    WRITE_B(firstPickupStep);
    WRITE_B(fireSteps);
    WRITE_B(fireAimErr);
    WRITE_B(fireAim10);
    WRITE_B(fireNoLos);
    WRITE_B(fireNoLosAim10);
    WRITE_B(standardOnly);
    WRITE_B(autoSwitches);
    WRITE_B(pickupTopUp);
    WRITE_B(pickupToHand);
    WRITE_B(pickupReplace);
    WRITE_B(crateAmmoFrac);
    WRITE_B(crateAmmoN);
    WRITE_B(crateNoWeapons);
    WRITE_B(ammoFrac);
    WRITE_B(ammoSteps);
    WRITE_B(setupKillsAny);
    WRITE_B(grappleTethered);
    WRITE_B(grappleReeling);
    WRITE_B(grappleReelSpeed);
    fprintf(f, "\"multiWeaponKills\":%g}", b->multiWeaponKills);
}

// name of a pickup type obs value: a weapon, an ammo crate or a utility
static const char *pickupObsName(const int type) {
    if (type == AMMO_CRATE_OBS_TYPE) {
        return "ammo_crate";
    }
    if (type > AMMO_CRATE_OBS_TYPE) {
        static const char *utilityPickups[] = {"utility_grapple", "utility_mines"};
        return utilityPickups[type - AMMO_CRATE_OBS_TYPE - 1];
    }
    return analysisWeaponNames[type];
}

// one-hot class name of a type slot value
static const char *className(const int vocab, const int cls) {
    static const char *walls[] = {"none", "standard", "bouncy", "death"};
    static const char *rays[] = {"none", "wall", "bouncy", "death", "fw", "fw_bouncy", "fw_death", "drone"};
    static const char *utilities[] = {"none", "grapple", "mines"};
    if (vocab == IW_PICKUP_VOCAB || vocab == IW_WEAPON_VOCAB) {
        return pickupObsName(cls);
    }
    if (vocab == IW_UTILITY_VOCAB) {
        return utilities[cls];
    }
    if (vocab == IW_WALL_VOCAB) {
        return walls[cls];
    }
    return rays[cls];
}

static double colNorm(const float *w, const int rows, const int cols, const int c) {
    double sq = 0.0;
    for (int h = 0; h < rows; h++) {
        sq += (double)w[h * cols + c] * w[h * cols + c];
    }
    return sqrt(sq);
}

static void writeFeature(FILE *f, const char *branch, const char *group, const char *name, const double sum, const double sq, const double n, const double norm) {
    const double mean = n > 0 ? sum / n : 0.0;
    const double var = n > 0 ? fmax(sq / n - mean * mean, 0.0) : 0.0;
    fprintf(f, "%s\t%s\t%s\t%g\t%g\t%g\t%g\n", branch, group, name, mean, sqrt(var), norm, norm * sqrt(var));
}

// per input feature L2 norm of its first layer weight column, with the
// feature's mean and std over the policy's live steps; norm * std is how
// much the feature moves the first layer's preactivations
static bool writeWeights(const char *path) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "can't write %s\n", path);
        return false;
    }
    const int H = policyHidden;
    int off = 0;
    off = puf_iw_align8(off + IW_C1_W_N);
    off = puf_iw_align8(off + IW_C2_W_N);
    off = puf_iw_align8(off + IW_C3_W_N);
    const float *proj = policyWeights->data + off;
    off = puf_iw_align8(off + H * IW_CONCAT);
    const float *projL1 = policyWeights->data + off;
    off = puf_iw_align8(off + IW_PROJ_L1_N);
    off = puf_iw_align8(off + H * IW_PROJ_HIDDEN);
    const float *fwL1 = policyWeights->data + off;

    fprintf(f, "branch\tgroup\tfeature\tmean\tstd\twnorm\teffect\n");
    char name[96];
    for (int c = 0; c < IW_CONV_FLAT; c++) {
        snprintf(name, sizeof(name), "cnn%d", c);
        writeFeature(f, "concat", "grid", name, concatSum[c], concatSq[c], concatN, colNorm(proj, H, IW_CONCAT, c));
    }
    for (int k = 0; k < IW_ONEHOT_COLS; k++) {
        const int c = IW_CONV_FLAT + k;
        const iwOneHotCol col = iw_onehot_col(k);
        snprintf(name, sizeof(name), "%s=%s", obsNames[col.obsIdx], className(col.vocab, col.cls + 1));
        writeFeature(f, "concat", obsGroups[obsGroupOf[col.obsIdx]].name, name, concatSum[c], concatSq[c], concatN, colNorm(proj, H, IW_CONCAT, c));
    }
    for (int j = 0; j < IW_NUM_SCALARS; j++) {
        const int c = IW_CONV_FLAT + IW_ONEHOT_COLS + j;
        const int i = scalarIdx[j];
        writeFeature(f, "concat", obsGroups[obsGroupOf[i]].name, obsNames[i], concatSum[c], concatSq[c], concatN, colNorm(proj, H, IW_CONCAT, c));
    }
    static const char *projScalars[] = {"own", "enemy", "x", "y", "vel_x", "vel_y"};
    for (int k = 0; k < IW_PROJ_USED_FEATURES; k++) {
        if (k < IW_WEAPON_VOCAB - 1) {
            snprintf(name, sizeof(name), "type=%s", analysisWeaponNames[k + 1]);
        } else {
            snprintf(name, sizeof(name), "%s", projScalars[k - (IW_WEAPON_VOCAB - 1)]);
        }
        writeFeature(f, "projectile", "projectiles", name, projRecSum[k], projRecSq[k], projRecN, colNorm(projL1, IW_PROJ_HIDDEN, IW_PROJ_FEATURES, k));
    }
    static const char *fwScalars[] = {"x", "y", "sin", "cos", "vel_x", "vel_y"};
    for (int k = 0; k < IW_FW_USED_FEATURES; k++) {
        if (k < IW_WALL_VOCAB - 1) {
            snprintf(name, sizeof(name), "type=%s", className(IW_WALL_VOCAB, k + 1));
        } else {
            snprintf(name, sizeof(name), "%s", fwScalars[k - (IW_WALL_VOCAB - 1)]);
        }
        writeFeature(f, "floating_wall", "floating_walls", name, fwRecSum[k], fwRecSq[k], fwRecN, colNorm(fwL1, IW_FW_HIDDEN, IW_FW_FEATURES, k));
    }
    fclose(f);
    return true;
}

static void parseMaps(const char *arg) {
    numMaps = 0;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", arg);
    for (char *tok = strtok(buf, ","); tok != NULL; tok = strtok(NULL, ",")) {
        tourMaps[numMaps++] = (uint8_t)atoi(tok);
    }
}

static bool checkWeights(const char *name, const Weights *weights, const int hidden, const int layers) {
    const int need = impulse_wars_weight_count(hidden, layers);
    const int have = weights->size - 7;
    if (hidden <= 0 || layers <= 0 || have != need) {
        fprintf(stderr, "%s has %d weights, hidden %d layers %d needs %d\n", name, have, hidden, layers, need);
        return false;
    }
    return true;
}

static bool parseOpp(const char *name) {
    for (uint8_t p = BOT_SCRIPTED; p < NUM_BOT_POLICIES; p++) {
        if (strcmp(policyNames[p], name) == 0) {
            opps[numOpps++] = (enum botPolicy)p;
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv) {
    uint32_t threads = 12;
    const char *out = "games.jsonl";
    bool episodesSet = false;
    buildObsNames();
    for (uint8_t i = 1; i < NUM_MAPS; i++) {
        tourMaps[numMaps++] = i;
    }
    for (int i = 1; i < argc; i++) {
        const bool hasValue = i + 1 < argc;
        if (strcmp(argv[i], "--episodes") == 0 && hasValue) {
            gamesPerOpp = (uint32_t)atoi(argv[++i]);
            episodesSet = true;
        } else if (strcmp(argv[i], "--threads") == 0 && hasValue) {
            threads = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && hasValue) {
            batchSize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && hasValue) {
            baseSeed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--maps") == 0 && hasValue) {
            parseMaps(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && hasValue) {
            out = argv[++i];
        } else if (strcmp(argv[i], "--policy") == 0 && hasValue) {
            policyWeights = load_weights(argv[++i]);
            if (policyWeights == NULL) {
                fprintf(stderr, "can't read policy %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--hidden") == 0 && hasValue) {
            policyHidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && hasValue) {
            policyLayers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--opponent") == 0 && hasValue) {
            oppWeights = load_weights(argv[++i]);
            if (oppWeights == NULL) {
                fprintf(stderr, "can't read opponent %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--opponent-hidden") == 0 && hasValue) {
            oppHidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--opponent-layers") == 0 && hasValue) {
            oppLayers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--opponent-name") == 0 && hasValue) {
            snprintf(oppName, sizeof(oppName), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--combos") == 0) {
            combos = true;
        } else if (strcmp(argv[i], "--no-pickups") == 0) {
            noPickups = true;
        } else if (strcmp(argv[i], "--loadout-pickups") == 0) {
            loadoutPickups = true;
        } else if (strcmp(argv[i], "--crates") == 0) {
            crates = true;
        } else if (strcmp(argv[i], "--ablate") == 0 && hasValue) {
            if (!parseMask(argv[++i], &ablateMask)) {
                return 1;
            }
            ablating = true;
        } else if (strcmp(argv[i], "--ablate-mode") == 0 && hasValue) {
            i++;
            if (strcmp(argv[i], "zero") == 0) {
                ablateMode = ABLATE_ZERO;
            } else if (strcmp(argv[i], "swap") == 0) {
                ablateMode = ABLATE_SWAP;
            } else {
                fprintf(stderr, "--ablate-mode is zero or swap\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--shadow") == 0 && hasValue) {
            if (numShadows == MAX_SHADOWS) {
                fprintf(stderr, "at most %d --shadow groups\n", MAX_SHADOWS);
                return 1;
            }
            if (!parseMask(argv[++i], &shadowMasks[numShadows++])) {
                return 1;
            }
        } else if (strcmp(argv[i], "--weights-out") == 0 && hasValue) {
            weightsOut = argv[++i];
        } else if (strcmp(argv[i], "--list-groups") == 0) {
            for (size_t gi = 0; gi < NUM_OBS_GROUPS; gi++) {
                printf("%-22s", obsGroups[gi].name);
                for (int r = 0; r < 2 && obsGroups[gi].ranges[r][1] != 0; r++) {
                    const int lo = obsGroups[gi].ranges[r][0];
                    const int hi = lo + obsGroups[gi].ranges[r][1] - 1;
                    printf(" %d-%d (%s..%s)", lo, hi, obsNames[lo], obsNames[hi]);
                }
                printf("\n");
            }
            return 0;
        } else if (numOpps == MAX_OPPS || !parseOpp(argv[i])) {
            fprintf(stderr, "bad argument %s\n", argv[i]);
            return 1;
        }
    }
    if (policyWeights == NULL) {
        fprintf(stderr, "--policy is required\n");
        return 1;
    }
    if (!checkWeights("policy", policyWeights, policyHidden, policyLayers)) {
        return 1;
    }
    if ((loadoutPickups || crates) && !combos) {
        fprintf(stderr, "--loadout-pickups and --crates need --combos\n");
        return 1;
    }
    if (noPickups + loadoutPickups + crates > 1) {
        fprintf(stderr, "--no-pickups, --loadout-pickups and --crates are exclusive\n");
        return 1;
    }
    if (combos && oppWeights == NULL) {
        fprintf(stderr, "--combos needs --opponent\n");
        return 1;
    }
    if (oppWeights != NULL) {
        if (numOpps != 0) {
            fprintf(stderr, "bot opponents can't be combined with --opponent\n");
            return 1;
        }
        if (oppHidden == 0 && oppLayers == 0) {
            oppHidden = policyHidden;
            oppLayers = policyLayers;
        }
        if (!checkWeights("opponent", oppWeights, oppHidden, oppLayers)) {
            return 1;
        }
        // the opponent index is unused, it names the opponent policy
        opps[numOpps++] = BOT_SCRIPTED;
        if (combos && !episodesSet) {
            gamesPerOpp = 16;
        }
        gamesPerOpp += gamesPerOpp % 2;
    }
    if (numOpps == 0) {
        for (uint8_t p = BOT_SCRIPTED; p < NUM_BOT_POLICIES; p++) {
            opps[numOpps++] = (enum botPolicy)p;
        }
    }
    batchSize = max(batchSize, 1);
    threads = min(max(threads, 1), 64);

    totalGames = numOpps * gamesPerOpp;
    if (combos) {
        for (uint8_t w = COMBO_FIRST; w < NW; w++) {
            if (weaponIsUtility(w)) {
                continue;
            }
            for (uint8_t u = UNSET_UTILITY + 1; u < _NUM_UTILITIES; u++) {
                comboLoadouts[numCombos][0] = w;
                comboLoadouts[numCombos][1] = u;
                numCombos++;
            }
        }
        for (uint16_t a = 0; a < numCombos; a++) {
            for (uint16_t b = a; b < numCombos; b++) {
                matchups[numMatchups][0] = a;
                matchups[numMatchups][1] = b;
                numMatchups++;
            }
        }
        totalGames = numMatchups * gamesPerOpp;
    }
    results = calloc(totalGames, sizeof(gameResult));
    if (results == NULL) {
        fprintf(stderr, "can't allocate %u games\n", totalGames);
        return 1;
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    pthread_t tids[64];
    for (uint32_t i = 0; i < threads; i++) {
        pthread_create(&tids[i], NULL, worker, NULL);
    }
    while (atomic_load(&jobsDone) < totalGames) {
        struct timespec ts = {.tv_nsec = 500000000};
        nanosleep(&ts, NULL);
        fprintf(stderr, "\r%u / %u games", atomic_load(&jobsDone), totalGames);
    }
    for (uint32_t i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    fprintf(stderr, "\n%u games in %.1fs\n", totalGames, (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);

    FILE *f = fopen(out, "w");
    if (f == NULL) {
        fprintf(stderr, "can't write %s\n", out);
        return 1;
    }
    for (uint32_t j = 0; j < totalGames; j++) {
        const gameResult *g = &results[j];
        const char *opp = oppWeights != NULL ? oppName : policyNames[opps[g->opp]];
        fprintf(f, "{\"opp\":\"%s\",\"seat\":%u,\"map\":%u,\"seed\":%u,\"length\":%g,", opp, g->seat, g->map, g->seed, g->length);
        // starting utility pickups by drone, the policy's first
        fprintf(f, "\"startUtilities\":[%u,%u],", g->startUtilities[g->seat], g->startUtilities[1 - g->seat]);
        if (ablating) {
            fprintf(f, "\"ablate\":\"%s\",\"ablateMode\":\"%s\",", ablateMask.name, ablateMode == ABLATE_SWAP ? "swap" : "zero");
        }
        if (combos) {
            fprintf(f, "\"loadout\":[%u,%u],\"utilityLoadout\":[%u,%u],", g->loadout[0], g->loadout[1], g->utilityLoadout[0], g->utilityLoadout[1]);
        }
        if (numShadows > 0) {
            fprintf(f, "\"shadow\":[");
            for (int k = 0; k < numShadows; k++) {
                const shadowStats *s = &g->shadow[k];
                fprintf(f, "%s{\"name\":\"%s\",\"n\":%g,\"dv\":%g,\"absDv\":%g,", k ? "," : "", shadowMasks[k].name, s->n, s->dv, s->absDv);
                fprintf(f, "\"kl\":[");
                for (int a = 0; a < NUM_ATNS; a++) {
                    fprintf(f, "%s%g", a ? "," : "", s->kl[a]);
                }
                fprintf(f, "]}");
            }
            fprintf(f, "],");
        }
        fprintf(f, "\"p\":");
        writeDrone(f, g, 0);
        fprintf(f, ",\"b\":");
        writeDrone(f, g, 1);
        fprintf(f, ",");
        writeArrayU(f, "moveHist", g->moveHist, 9);
        fprintf(f, ",");
        writeArrayU(f, "aimHist", g->aimHist, AIM_DIRS);
        fprintf(f, ",");
        writeArrayU(f, "aimFireHist", g->aimFireHist, AIM_DIRS);
        fprintf(f, "}\n");
    }
    fclose(f);
    if (weightsOut != NULL && !writeWeights(weightsOut)) {
        return 1;
    }
    free(results);
    destroyMaps();
    return 0;
}
