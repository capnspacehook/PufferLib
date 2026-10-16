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
// checkpoints or env builds can be compared game by game.
//
// Build: ocean/impulse_wars/analysis/build.sh
// Usage: iw_analyze --policy ckpt.bin --hidden H --layers L [--episodes N]
//                   [--threads T] [--batch B] [--seed S] [--maps 1,2,3]
//                   [--out games.jsonl] [opponent ...]
//        iw_analyze --policy ckpt.bin --hidden H --layers L
//                   --opponent ckpt.bin --opponent-hidden H --opponent-layers L
//                   [--opponent-name name] [options]
//   --episodes is per opponent; against a policy it's rounded up to an
//   even number so every seed is played from both drones. Bot opponents
//   are bot personality names and default to all of them. --batch games
//   run in lockstep per thread so the network weights are reused across
//   the batch.
#include <pthread.h>
#include <stdatomic.h>

#include "puffercpu.c"
#include "impulse_wars.h"
#include "impulse_wars_net.h"

#define NW _NUM_WEAPONS
#define MAX_OPPS NUM_BOT_POLICIES
#define AIM_DIRS (AIM_COARSE_SECTORS * AIM_FINE_STEPS)

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
    double nearDeath6;
    double nearFloatingDeath6;
    double firstLosStep;
    double firstPickupStep;
    double fireSteps;
    double fireAimErr;
    double fireAim10;
    double fireNoLos;
    double fireNoLosAim10;
} behaviorStats;

typedef struct gameResult {
    uint8_t opp;
    // drone the analyzed policy played
    uint8_t seat;
    uint8_t map;
    uint32_t seed;
    float length;
    droneStats stats[2];
    droneAnalysisStats analysis[2];
    behaviorStats behavior[2];
    // stats are ordered analyzed policy then opponent
    // policy action histograms while alive
    uint32_t moveHist[9];
    uint32_t aimHist[AIM_DIRS];
    uint32_t aimFireHist[AIM_DIRS];
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

// a game slot of a worker's batch
typedef struct gameSlot {
    iwEnv *e;
    gameResult *g;
    droneStats prev[2];
    enum weaponType prevWeapon[2];
    uint32_t step;
    bool active;
} gameSlot;

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
static gameResult *results;
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

// forward the whole batch and sample actions with each game's own rng
static void policyForward(policyBatch *p) {
    ImpulseWarsNet *net = p->net;
    mingru_zero_term(net->mingru, p->terminals);
    impulse_wars_encode(net, p->obs);
    mingru(net->mingru, net->proj->output);
    linear(net->decoder, net->mingru->output);
    const int actSizes[] = ACT_SIZES;
    const int atnSum = impulse_wars_atn_sum();
    for (int b = 0; b < p->size; b++) {
        // logits for every action head, followed by the value
        const float *logits = net->decoder->output + b * (atnSum + 1);
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
    // a fresh recurrent state for every game
    p->terminals[b] = 1.0f;
    p->rng[b] = g->seed;
    if (opp != NULL) {
        opp->terminals[b] = 1.0f;
        opp->rng[b] = hashSeed(g->seed);
    }

    slot->g = g;
    slot->step = 0;
    slot->active = true;
    memcpy(slot->prev, e->stats, sizeof(slot->prev));
    for (uint8_t i = 0; i < 2; i++) {
        const droneEntity *d = safe_array_get_at(e->drones, i);
        slot->prevWeapon[i] = d->weaponInfo->type;
        g->behavior[i].firstLosStep = -1;
        g->behavior[i].firstPickupStep = -1;
    }
}

static void observeStep(gameSlot *slot) {
    iwEnv *e = slot->e;
    gameResult *g = slot->g;
    droneEntity *drones[2] = {safe_array_get_at(e->drones, 0), safe_array_get_at(e->drones, 1)};
    const bool bothAlive = !drones[0]->dead && !drones[1]->dead;
    const bool los = bothAlive && lineOfSight(e, drones[0], drones[1]);
    const float sep = bothAlive ? b2Distance(drones[0]->pos, drones[1]->pos) : 0.0f;

    for (uint8_t i = 0; i < 2; i++) {
        const droneEntity *d = drones[i];
        const droneEntity *enemy = drones[1 - i];
        behaviorStats *s = &g->behavior[i == g->seat ? 0 : 1];
        const droneStats *cur = &e->stats[i];
        const droneStats *prev = &slot->prev[i];

        if (s->firstPickupStep < 0 && cur->totalWeaponsPickedUp > prev->totalWeaponsPickedUp) {
            s->firstPickupStep = slot->step;
        }
        if (d->dead) {
            continue;
        }

        const enum weaponType weapon = d->weaponInfo->type;
        const agentActions *a = &e->cachedActions[i];
        const float speed = b2Length(d->velocity);
        s->alive++;
        s->held[weapon]++;
        s->speed += speed;
        s->energy += d->energyLeft;
        s->trigger += a->chargingWeapon;
        s->brake += a->brake;
        s->burstCharge += a->chargingBurst;
        s->moveNoop += b2Length(a->move) < 0.01f;
        if (d->weaponInfo->charge != 0.0f && d->chargingWeapon) {
            s->charging++;
            s->chargeHeldFull += d->weaponCharge >= d->weaponInfo->charge;
        }
        if (bothAlive) {
            s->sepSum += sep;
            s->sepSteps++;
            s->closeSteps += sep < 15.0f;
            s->farSteps += sep > 40.0f;
            if (los) {
                s->losSteps++;
                s->triggerLos += a->chargingWeapon;
                s->speedLos += speed;
                if (s->firstLosStep < 0) {
                    s->firstLosStep = slot->step;
                }
            } else {
                s->noLosSteps++;
                s->triggerNoLos += a->chargingWeapon;
                s->speedNoLos += speed;
            }
        }
        float staticDist, floatingDist;
        deathWallDistances(e, d, &staticDist, &floatingDist);
        const float deathDist = min(staticDist, floatingDist) - DRONE_RADIUS;
        s->nearDeath3 += deathDist < 3.0f;
        s->nearDeath6 += deathDist < 6.0f;
        s->nearFloatingDeath6 += floatingDist - DRONE_RADIUS < 6.0f;

        float fired = 0.0f;
        for (uint8_t w = 0; w < NW; w++) {
            const float shots = cur->shotsFired[w] - prev->shotsFired[w];
            if (shots > 0.0f && bothAlive) {
                fired += shots;
                s->fireDist[w] += sep * shots;
                s->fireLos[w] += los * shots;
            }
        }
        if (fired > 0.0f) {
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
        slot->prevWeapon[i] = weapon;
    }
    memcpy(slot->prev, e->stats, sizeof(slot->prev));
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
    puf_step(e);
    if (e->log.n != 0.0f) {
        g->length = e->log.length;
        for (uint8_t i = 0; i < 2; i++) {
            const uint8_t drone = i == 0 ? g->seat : 1 - g->seat;
            g->stats[i] = e->log.stats[drone];
            g->analysis[i] = e->lastAnalysisStats[drone];
        }
        return true;
    }
    slot->step++;
    observeStep(slot);
    return false;
}

static bool nextGame(gameResult **out) {
    const uint32_t job = atomic_fetch_add(&nextJob, 1);
    if (job >= numOpps * gamesPerOpp) {
        return false;
    }
    uint32_t gameIdx = job % gamesPerOpp;
    gameResult *g = &results[job];
    memset(g, 0x0, sizeof(*g));
    g->opp = job / gamesPerOpp;
    // against a policy every seed is played from both drones
    if (oppWeights != NULL) {
        g->seat = gameIdx % 2;
        gameIdx /= 2;
    }
    g->map = tourMaps[gameIdx % numMaps];
    g->seed = hashSeed(baseSeed * 1000003ull + gameIdx);
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

static void *worker(void *arg) {
    MAYBE_UNUSED(arg);
    const int B = batchSize;
    policyBatch *p = createPolicyBatch(policyWeights, B, policyHidden, policyLayers);
    policyBatch *opp = NULL;
    if (oppWeights != NULL) {
        opp = createPolicyBatch(oppWeights, B, oppHidden, oppLayers);
    }

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
        policyForward(p);
        if (opp != NULL) {
            policyForward(opp);
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
    fprintf(f, "\"deaths\":%g,", (double)(o->kills + s->selfKills + s->unknownKills));
    WRITE_F(totalBursts);
    WRITE_F(burstsHit);
    WRITE_F(energyEmptied);
    WRITE_F(shieldsBroken);
    WRITE_F(ownShieldBroken);
    writeArray(f, "shotsFired", s->shotsFired, NW);
    writeArray(f, "shotsHit", s->shotsHit, NW);
    writeArray(f, "weaponsPickedUp", s->weaponsPickedUp, NW);
    WRITE_A(killsByWeapon, NW);
    WRITE_A(selfKillsByWeapon, NW);
    WRITE_A(deathsBySource, NUM_PHYSICS_TYPES);
    WRITE_A(selfKillsBySource, NUM_PHYSICS_TYPES);
    fprintf(f, "\"floatingWallDeaths\":%g,\"floatingWallSelfKills\":%g,", a->floatingWallDeaths, a->floatingWallSelfKills);
    WRITE_A(diedHolding, NW);
    WRITE_A(pickupsFrom, NW * NW);
    WRITE_A(boostedShotsFired, NW);
    WRITE_A(boostedShotsHit, NW);
    WRITE_A(boostedKills, NW);
    WRITE_A(boostedSelfKills, NW);
    WRITE_A(boostedSpeedBefore, NW);
    WRITE_A(boostedSpeedAfter, NW);
    writeArrayD(f, "held", b->held, NW);
    writeArrayD(f, "fireDist", b->fireDist, NW);
    writeArrayD(f, "fireLos", b->fireLos, NW);
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
    WRITE_B(nearDeath6);
    WRITE_B(nearFloatingDeath6);
    WRITE_B(firstLosStep);
    WRITE_B(firstPickupStep);
    WRITE_B(fireSteps);
    WRITE_B(fireAimErr);
    WRITE_B(fireAim10);
    WRITE_B(fireNoLos);
    fprintf(f, "\"fireNoLosAim10\":%g}", b->fireNoLosAim10);
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
    for (uint8_t i = 1; i < NUM_MAPS; i++) {
        tourMaps[numMaps++] = i;
    }
    for (int i = 1; i < argc; i++) {
        const bool hasValue = i + 1 < argc;
        if (strcmp(argv[i], "--episodes") == 0 && hasValue) {
            gamesPerOpp = (uint32_t)atoi(argv[++i]);
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
        gamesPerOpp += gamesPerOpp % 2;
    }
    if (numOpps == 0) {
        for (uint8_t p = BOT_SCRIPTED; p < NUM_BOT_POLICIES; p++) {
            opps[numOpps++] = (enum botPolicy)p;
        }
    }
    batchSize = max(batchSize, 1);
    threads = min(max(threads, 1), 64);

    const uint32_t total = numOpps * gamesPerOpp;
    results = calloc(total, sizeof(gameResult));
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    pthread_t tids[64];
    for (uint32_t i = 0; i < threads; i++) {
        pthread_create(&tids[i], NULL, worker, NULL);
    }
    while (atomic_load(&jobsDone) < total) {
        struct timespec ts = {.tv_nsec = 500000000};
        nanosleep(&ts, NULL);
        fprintf(stderr, "\r%u / %u games", atomic_load(&jobsDone), total);
    }
    for (uint32_t i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    fprintf(stderr, "\n%u games in %.1fs\n", total, (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);

    FILE *f = fopen(out, "w");
    if (f == NULL) {
        fprintf(stderr, "can't write %s\n", out);
        return 1;
    }
    for (uint32_t j = 0; j < total; j++) {
        const gameResult *g = &results[j];
        const char *opp = oppWeights != NULL ? oppName : policyNames[opps[g->opp]];
        fprintf(f, "{\"opp\":\"%s\",\"seat\":%u,\"map\":%u,\"seed\":%u,\"length\":%g,\"p\":", opp, g->seat, g->map, g->seed, g->length);
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
    free(results);
    destroyMaps();
    return 0;
}
