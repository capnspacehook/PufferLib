// Scripted bot tournament: plays variants of the scripted bot against each
// other with paired seeds. Every seed is played twice with the variants
// swapping drones, so spawn positions, map and floating walls are identical
// for both sides. Maps are pinned round robin over the non empty maps.
//
// Build: ./ocean/impulse_wars/build_tournament.sh
// Usage: ./iw_tournament [--episodes N] [--threads T] [--seed S] [--rr] [--eval]
//                        [--maps 1,2,3] name=featureMask ...
//   Without --rr the first variant is the reference and every other variant
//   plays it; with --rr every pair of variants plays.
//   --episodes is per pair (each seed is 2 episodes).
//   Feature masks are enum botFeature bits: 0x1 (ammo reserve) is the
//   standard bot, the other bits enable experimental behaviors.
//   A variant can instead be a personality (see enum botPolicy) with its
//   default features or an explicit mask: name=evader, name=rusher:0x1d.
//   Personalities: scripted evader camper rusher trapper turtle fundamentals.
//
// --eval plays in eval mode instead of training mode: eval frame rate,
//   diagonal spawns and sudden death, which training mode disables for
//   games against scripted bots. Headless episodes are capped at
//   TRAINING_MAX_STEPS frames, well before the round would normally end,
//   so sudden death starts EVAL_SUDDEN_DEATH_SECONDS before the cap.
//
// Learned opponent: --policy checkpoint.bin --hidden H --layers L [--argmax]
//   plays the policy on drone 0, as in training and the bot ladder, against
//   every variant on drone 1, one game per seed. Every variant faces the
//   policy on identical seeds, so variants are compared by their paired
//   difference to the first variant. Actions are sampled with a per game
//   seeded rng so runs are reproducible; --argmax picks the greedy action.
#include <pthread.h>
#include <stdatomic.h>

#include "puffercpu.c"
#include "impulse_wars.h"
#include "impulse_wars_net.h"

#define MAX_VARIANTS 32
#define MAX_PAIRS (MAX_VARIANTS * MAX_VARIANTS)

typedef struct variant {
    char name[64];
    enum botPolicy policy;
    uint32_t features;
} variant;

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

// parses "featureMask" or "personality[:featureMask]"
static bool parseVariant(variant *v, const char *spec) {
    v->policy = BOT_SCRIPTED;
    const char *colon = strchr(spec, ':');
    const size_t nameLen = colon != NULL ? (size_t)(colon - spec) : strlen(spec);
    for (uint8_t p = BOT_SCRIPTED; p < NUM_BOT_POLICIES; p++) {
        if (strlen(policyNames[p]) == nameLen && strncmp(policyNames[p], spec, nameLen) == 0) {
            v->policy = (enum botPolicy)p;
            v->features = botPresets[p].feats;
            if (colon != NULL) {
                v->features = (uint32_t)strtoul(colon + 1, NULL, 0);
            }
            return true;
        }
    }
    char *end;
    v->features = (uint32_t)strtoul(spec, &end, 0);
    return end != spec && *end == '\0';
}

typedef struct sideStats {
    double kills;
    double selfKills;
    double unknownKills;
    double deaths;
    double shotsFired;
    double shotsHit;
    double pickups;
    double energyEmptied;
    double bursts;
    double burstsHit;
    double distance;
    // mean speed while alive, and mean distance between the drones while
    // both are alive
    double speed;
    double separation;
    double events[NUM_BOT_EVENTS];
} sideStats;

typedef struct gameStats {
    double speed[2];
    double separation;
} gameStats;

// result of one seed: two games, variant A on drone 0 then on drone 1
typedef struct seedResult {
    float scoreA[2];
    uint8_t map;
    float length[2];
    sideStats a;
    sideStats b;
} seedResult;

typedef struct pairing {
    uint8_t a;
    uint8_t b;
} pairing;

static variant variants[MAX_VARIANTS];
static uint8_t numVariants = 0;
static pairing pairs[MAX_PAIRS];
static uint16_t numPairs = 0;
static uint32_t seedsPerPair = 0;
static uint64_t baseSeed = 1;
static uint8_t tourMaps[NUM_MAPS];
static uint8_t numMaps = 0;
static seedResult *results;
static atomic_uint nextJob;
static atomic_uint jobsDone;
static pthread_mutex_t mapsMu = PTHREAD_MUTEX_INITIALIZER;
static bool mapsReady = false;
static bool evalMode = false;
#define EVAL_SUDDEN_DEATH_SECONDS 25

// learned opponent
static Weights *policyWeights = NULL;
static int policyHidden = 0;
static int policyLayers = 0;
static bool policyArgmax = false;

typedef struct policyRunner {
    ImpulseWarsNet *net;
    float obs[OBS_SIZE];
    float actions[NUM_ATNS];
    float reward;
    float terminal;
    unsigned int rng;
} policyRunner;

// forward_impulse_wars with a thread local rng for sampling
static void policyForward(policyRunner *p) {
    ImpulseWarsNet *net = p->net;
    mingru_zero_term(net->mingru, &p->terminal);
    impulse_wars_encode(net, p->obs);
    mingru(net->mingru, net->proj->output);
    linear(net->decoder, net->mingru->output);
    const int actSizes[] = ACT_SIZES;
    // the logits of every action head come first, the value is last
    const float *logits = net->decoder->output;
    for (int a = 0; a < NUM_ATNS; a++) {
        const int n = actSizes[a];
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        int choice = best;
        if (!policyArgmax) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += expf(logits[i] - logits[best]);
            }
            const float target = ((float)rand_r(&p->rng) / (float)RAND_MAX) * sum;
            float acc = 0.0f;
            for (int i = 0; i < n; i++) {
                acc += expf(logits[i] - logits[best]);
                if (target < acc) {
                    choice = i;
                    break;
                }
            }
        }
        p->actions[a] = (float)choice;
        logits += n;
    }
}

static uint32_t hashSeed(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    uint32_t r = (uint32_t)x;
    return r == 0 ? 1 : r;
}

static void addStats(sideStats *s, const droneStats *d) {
    s->kills += d->kills;
    s->selfKills += d->selfKills;
    s->unknownKills += d->unknownKills;
    s->shotsFired += d->totalShotsFired;
    s->shotsHit += d->totalShotsHit;
    s->pickups += d->totalWeaponsPickedUp;
    s->energyEmptied += d->energyEmptied;
    s->bursts += d->totalBursts;
    s->burstsHit += d->burstsHit;
    s->distance += d->distanceTraveled;
}

static void initTournamentEnv(iwEnv *e, policyRunner *p) {
    initEnv(e, p != NULL ? 1 : 0, -1, 1, false, !evalMode, false, BOT_SCRIPTED, 0.0f, 0.0f);
    if (p != NULL) {
        e->agents[0].observations = p->obs;
        e->agents[0].actions = p->actions;
        e->agents[0].rewards = &p->reward;
        e->agents[0].terminals = &p->terminal;
    }
    setRewards(e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    e->easyEpProb = 0.0f;
    if (evalMode) {
        e->totalSteps = TRAINING_MAX_STEPS - (EVAL_SUDDEN_DEATH_SECONDS * e->frameRate);
    }
}

static void playGame(iwEnv *e, policyRunner *p, const uint32_t rngSeed, const uint8_t mapIdx, const variant *v0, const variant *v1, gameStats *gs) {
    // box2d is only deterministic for identical creation order and id
    // pool state, so every game gets a fresh world
    pthread_mutex_lock(&mapsMu);
    puf_close(e);
    memset(e, 0x0, sizeof(*e));
    initTournamentEnv(e, p);
    pthread_mutex_unlock(&mapsMu);

    e->pinnedMapIdx = mapIdx;
    e->rng = rngSeed;
    if (v0 != NULL) {
        e->botPersonality[0] = v0->policy;
        e->botFeatures[0] = v0->features;
    }
    e->botPersonality[1] = v1->policy;
    e->botFeatures[1] = v1->features;
    memset(&e->log, 0x0, sizeof(e->log));
    memset(e->botEvents, 0x0, sizeof(e->botEvents));
    puf_reset(e);
    if (p != NULL) {
        // a fresh recurrent state for every game
        p->terminal = 1.0f;
        p->rng = rngSeed;
    }
    double speedSum[2] = {0}, speedSteps[2] = {0}, sepSum = 0, sepSteps = 0;
    while (e->log.n == 0.0f) {
        if (p != NULL) {
            policyForward(p);
        }
        puf_step(e);
        const droneEntity *d0 = safe_array_get_at(e->drones, 0);
        const droneEntity *d1 = safe_array_get_at(e->drones, 1);
        const droneEntity *ds[2] = {d0, d1};
        for (uint8_t i = 0; i < 2; i++) {
            if (!ds[i]->dead) {
                speedSum[i] += b2Length(ds[i]->velocity);
                speedSteps[i]++;
            }
        }
        if (!d0->dead && !d1->dead) {
            sepSum += b2Distance(d0->pos, d1->pos);
            sepSteps++;
        }
    }
    for (uint8_t i = 0; i < 2; i++) {
        gs->speed[i] = speedSteps[i] > 0 ? speedSum[i] / speedSteps[i] : 0.0;
    }
    gs->separation = sepSteps > 0 ? sepSum / sepSteps : 0.0;
}

static void *worker(void *arg) {
    MAYBE_UNUSED(arg);
    iwEnv *e = fastCalloc(1, sizeof(iwEnv));
    policyRunner *p = NULL;
    if (policyWeights != NULL) {
        p = calloc(1, sizeof(policyRunner));
        // layers point into the shared weight data, only the read index
        // is per net
        Weights *w = malloc(sizeof(Weights));
        *w = *policyWeights;
        w->idx = 0;
        p->net = init_impulse_wars_net(w, 1, policyHidden, policyLayers);
    }
    // box2d world creation and destruction touch global state
    pthread_mutex_lock(&mapsMu);
    initTournamentEnv(e, p);
    if (!mapsReady) {
        initMaps(e);
        mapsReady = true;
    }
    pthread_mutex_unlock(&mapsMu);

    const uint32_t totalJobs = numPairs * seedsPerPair;
    while (true) {
        const uint32_t job = atomic_fetch_add(&nextJob, 1);
        if (job >= totalJobs) {
            break;
        }
        const uint16_t pairIdx = job / seedsPerPair;
        const uint32_t seedIdx = job % seedsPerPair;
        const variant *va = &variants[pairs[pairIdx].a];
        const variant *vb = &variants[pairs[pairIdx].b];
        // the same seeds are used for every pair so variants are compared
        // on identical games
        const uint32_t rngSeed = hashSeed(baseSeed * 1000003ull + seedIdx);
        const uint8_t mapIdx = tourMaps[seedIdx % numMaps];

        seedResult *res = &results[job];
        memset(res, 0x0, sizeof(*res));
        res->map = mapIdx;
        // against a policy the variant always plays drone 1, once per seed
        const uint8_t numSides = p != NULL ? 1 : 2;
        for (uint8_t side = 0; side < numSides; side++) {
            uint8_t aIdx = side;
            uint8_t bIdx = 1 - side;
            gameStats gs = {0};
            if (p != NULL) {
                aIdx = 1;
                bIdx = 0;
                playGame(e, p, rngSeed, mapIdx, NULL, va, &gs);
            } else if (side == 0) {
                playGame(e, p, rngSeed, mapIdx, va, vb, &gs);
            } else {
                playGame(e, p, rngSeed, mapIdx, vb, va, &gs);
            }
            res->a.speed += gs.speed[aIdx];
            res->b.speed += gs.speed[bIdx];
            res->a.separation += gs.separation;
            res->b.separation += gs.separation;
            res->scoreA[side] = e->log.stats[aIdx].score;
            res->length[side] = e->log.length;
            addStats(&res->a, &e->log.stats[aIdx]);
            addStats(&res->b, &e->log.stats[bIdx]);
            for (uint8_t k = 0; k < NUM_BOT_EVENTS; k++) {
                res->a.events[k] += e->botEvents[aIdx][k];
                res->b.events[k] += e->botEvents[bIdx][k];
            }
            res->a.deaths += e->log.stats[bIdx].kills + e->log.stats[aIdx].selfKills + e->log.stats[bIdx].unknownKills;
            res->b.deaths += e->log.stats[aIdx].kills + e->log.stats[bIdx].selfKills + e->log.stats[aIdx].unknownKills;
        }
        atomic_fetch_add(&jobsDone, 1);
    }

    pthread_mutex_lock(&mapsMu);
    puf_close(e);
    pthread_mutex_unlock(&mapsMu);
    fastFree(e);
    return NULL;
}

#define ADD_SIDE(dst, src)                   \
    dst.kills += src.kills;                  \
    dst.selfKills += src.selfKills;          \
    dst.unknownKills += src.unknownKills;    \
    dst.deaths += src.deaths;                \
    dst.shotsFired += src.shotsFired;        \
    dst.shotsHit += src.shotsHit;            \
    dst.pickups += src.pickups;              \
    dst.energyEmptied += src.energyEmptied;  \
    dst.bursts += src.bursts;                \
    dst.burstsHit += src.burstsHit;          \
    dst.distance += src.distance;            \
    dst.speed += src.speed;                  \
    dst.separation += src.separation;        \
    for (uint8_t k = 0; k < NUM_BOT_EVENTS; k++) dst.events[k] += src.events[k]

static void printSide(const char *name, const sideStats *s, const double games) {
    printf("    %-24s kills %.2f  self %.2f  unk %.2f  shots %.0f  acc %.3f  pickups %.2f  empty %.2f  bursts %.1f (hit %.2f)  dist %.0f  speed %.1f  sep %.1f\n",
           name, s->kills / games, s->selfKills / games, s->unknownKills / games, s->shotsFired / games,
           s->shotsFired > 0 ? s->shotsHit / s->shotsFired : 0.0, s->pickups / games, s->energyEmptied / games,
           s->bursts / games, s->bursts > 0 ? s->burstsHit / s->bursts : 0.0, s->distance / games, s->speed / games,
           s->separation / games);
    static const char *eventNames[NUM_BOT_EVENTS] = {
        "threat", "threat_shot", "threat_burst", "sniper_dodge", "sniper_urgent", "burst_start",
        "boost", "kill_geo", "pickup_skip", "ammo_reserved", "clear_steer",
        "escape_fire", "long_shot", "spare_ammo_move", "dyn_path", "dyn_path_fail",
        "flee", "hold_spot", "spot_move", "recoil_boost", "mine_placed", "counter_shot", "sd_avoid",
        "swap", "swap_cycle", "charge_cancel", "pickup_swap", "mine_detonate", "safe_geo", "kite", "shotgun_boost",
    };
    printf("    %-24s", "");
    for (uint8_t k = 0; k < NUM_BOT_EVENTS; k++) {
        if (s->events[k] > 0) {
            printf(" %s %.1f", eventNames[k], s->events[k] / games);
        }
    }
    printf("\n");
}

// scores of each variant against the policy, and the paired difference of
// each variant to the first one on identical seeds
static void reportVsPolicy(void) {
    const seedResult *ref = &results[0];
    for (uint16_t p = 0; p < numPairs; p++) {
        const variant *v = &variants[pairs[p].a];
        const seedResult *res = &results[p * seedsPerPair];
        double sum = 0.0, sumSq = 0.0, diff = 0.0, diffSq = 0.0, wins = 0, ties = 0, length = 0;
        double mapScore[NUM_MAPS] = {0}, mapGames[NUM_MAPS] = {0};
        sideStats sv = {0}, sp = {0};
        for (uint32_t s = 0; s < seedsPerPair; s++) {
            const seedResult *r = &res[s];
            const double score = r->scoreA[0];
            sum += score;
            sumSq += score * score;
            const double d = score - ref[s].scoreA[0];
            diff += d;
            diffSq += d * d;
            wins += score == 1.0;
            ties += score == 0.5;
            length += r->length[0];
            mapScore[r->map] += score;
            mapGames[r->map] += 1;
            ADD_SIDE(sv, r->a);
            ADD_SIDE(sp, r->b);
        }
        const double n = seedsPerPair;
        const double mean = sum / n;
        const double ci = 1.96 * sqrt(fmax(sumSq / n - mean * mean, 0.0) / n);
        const double diffMean = diff / n;
        const double diffCi = 1.96 * sqrt(fmax(diffSq / n - diffMean * diffMean, 0.0) / n);
        printf("%s vs policy: score %.4f +- %.4f  (W %.3f T %.3f L %.3f, len %.0f)", v->name, mean, ci, wins / n, ties / n,
               (n - wins - ties) / n, length / n);
        if (p != 0) {
            printf("  vs %s: %+.4f +- %.4f", variants[pairs[0].a].name, diffMean, diffCi);
        }
        printf("\n    per map:");
        for (uint8_t m = 0; m < NUM_MAPS; m++) {
            if (mapGames[m] > 0) {
                printf(" %u:%.3f", m, mapScore[m] / mapGames[m]);
            }
        }
        printf("\n");
        printSide(v->name, &sv, n);
        printSide("policy", &sp, n);
        fflush(stdout);
    }
}

static void parseMaps(const char *arg) {
    numMaps = 0;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", arg);
    for (char *tok = strtok(buf, ","); tok != NULL; tok = strtok(NULL, ",")) {
        tourMaps[numMaps++] = (uint8_t)atoi(tok);
    }
}

int main(int argc, char **argv) {
    uint32_t episodes = 10000;
    uint32_t threads = 12;
    bool roundRobin = false;
    for (uint8_t i = 1; i < NUM_MAPS; i++) {
        tourMaps[numMaps++] = i;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
            episodes = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            baseSeed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--maps") == 0 && i + 1 < argc) {
            parseMaps(argv[++i]);
        } else if (strcmp(argv[i], "--eval") == 0) {
            evalMode = true;
        } else if (strcmp(argv[i], "--rr") == 0) {
            roundRobin = true;
        } else if (strcmp(argv[i], "--policy") == 0 && i + 1 < argc) {
            policyWeights = load_weights(argv[++i]);
            if (policyWeights == NULL) {
                fprintf(stderr, "can't read policy %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            policyHidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            policyLayers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--argmax") == 0) {
            policyArgmax = true;
        } else {
            char *eq = strchr(argv[i], '=');
            if (eq == NULL || numVariants == MAX_VARIANTS) {
                fprintf(stderr, "bad argument %s\n", argv[i]);
                return 1;
            }
            variant *v = &variants[numVariants++];
            snprintf(v->name, sizeof(v->name), "%.*s", (int)(eq - argv[i]), argv[i]);
            if (!parseVariant(v, eq + 1)) {
                fprintf(stderr, "bad variant %s\n", argv[i]);
                return 1;
            }
        }
    }
    const bool vsPolicy = policyWeights != NULL;
    if (vsPolicy) {
        const int need = impulse_wars_weight_count(policyHidden, policyLayers);
        const int have = policyWeights->size - 7;
        if (policyHidden <= 0 || policyLayers <= 0 || have > need || need - have > 7) {
            fprintf(stderr, "policy has %d weights, hidden %d layers %d needs %d\n", have, policyHidden, policyLayers, need);
            return 1;
        }
    }
    if (numVariants < (vsPolicy ? 1 : 2)) {
        fprintf(stderr, "need at least %d variants\n", vsPolicy ? 1 : 2);
        return 1;
    }

    if (vsPolicy) {
        for (uint8_t a = 0; a < numVariants; a++) {
            pairs[numPairs++] = (pairing){a, a};
        }
    } else if (roundRobin) {
        for (uint8_t a = 0; a < numVariants; a++) {
            for (uint8_t b = a + 1; b < numVariants; b++) {
                pairs[numPairs++] = (pairing){a, b};
            }
        }
    } else {
        for (uint8_t b = 1; b < numVariants; b++) {
            pairs[numPairs++] = (pairing){b, 0};
        }
    }
    seedsPerPair = vsPolicy ? episodes : (episodes + 1) / 2;
    const uint32_t totalJobs = numPairs * seedsPerPair;
    results = calloc(totalJobs, sizeof(seedResult));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    pthread_t tids[64];
    threads = min(threads, 64);
    for (uint32_t i = 0; i < threads; i++) {
        pthread_create(&tids[i], NULL, worker, NULL);
    }
    while (atomic_load(&jobsDone) < totalJobs) {
        struct timespec ts = {.tv_nsec = 200000000};
        nanosleep(&ts, NULL);
        fprintf(stderr, "\r%u / %u seeds", atomic_load(&jobsDone), totalJobs);
    }
    for (uint32_t i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    const double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    fprintf(stderr, "\n%u episodes in %.1fs\n", totalJobs * (vsPolicy ? 1 : 2), elapsed);
    if (vsPolicy) {
        reportVsPolicy();
        free(results);
        destroyMaps();
        return 0;
    }

    // aggregate score per variant for the round robin table
    double variantScore[MAX_VARIANTS] = {0};
    double variantGames[MAX_VARIANTS] = {0};

    for (uint16_t p = 0; p < numPairs; p++) {
        const variant *va = &variants[pairs[p].a];
        const variant *vb = &variants[pairs[p].b];
        const seedResult *res = &results[p * seedsPerPair];
        double sum = 0.0, sumSq = 0.0, wins = 0, ties = 0, losses = 0, length = 0;
        double mapScore[NUM_MAPS] = {0}, mapGames[NUM_MAPS] = {0};
        sideStats sa = {0}, sb = {0};
        for (uint32_t s = 0; s < seedsPerPair; s++) {
            const seedResult *r = &res[s];
            const double paired = (r->scoreA[0] + r->scoreA[1]) / 2.0;
            sum += paired;
            sumSq += paired * paired;
            for (uint8_t side = 0; side < 2; side++) {
                wins += r->scoreA[side] == 1.0f;
                ties += r->scoreA[side] == 0.5f;
                losses += r->scoreA[side] == 0.0f;
                length += r->length[side];
                mapScore[r->map] += r->scoreA[side];
                mapGames[r->map] += 1;
            }
            ADD_SIDE(sa, r->a);
            ADD_SIDE(sb, r->b);
        }
        const double n = seedsPerPair;
        const double mean = sum / n;
        const double var = fmax(sumSq / n - mean * mean, 0.0);
        const double ci = 1.96 * sqrt(var / n);
        const double games = n * 2;
        printf("%s vs %s: score %.4f +- %.4f  (W %.3f T %.3f L %.3f, len %.0f)\n", va->name, vb->name, mean, ci,
               wins / games, ties / games, losses / games, length / games);
        printf("    per map:");
        for (uint8_t m = 0; m < NUM_MAPS; m++) {
            if (mapGames[m] > 0) {
                printf(" %u:%.3f", m, mapScore[m] / mapGames[m]);
            }
        }
        printf("\n");
        printSide(va->name, &sa, games);
        printSide(vb->name, &sb, games);
        fflush(stdout);

        variantScore[pairs[p].a] += sum * 2;
        variantGames[pairs[p].a] += games;
        variantScore[pairs[p].b] += games - sum * 2;
        variantGames[pairs[p].b] += games;
    }

    if (roundRobin) {
        printf("\nround robin standings:\n");
        uint8_t order[MAX_VARIANTS];
        for (uint8_t i = 0; i < numVariants; i++) {
            order[i] = i;
        }
        for (uint8_t i = 0; i < numVariants; i++) {
            for (uint8_t j = i + 1; j < numVariants; j++) {
                if (variantScore[order[j]] / variantGames[order[j]] > variantScore[order[i]] / variantGames[order[i]]) {
                    const uint8_t t = order[i];
                    order[i] = order[j];
                    order[j] = t;
                }
            }
        }
        for (uint8_t i = 0; i < numVariants; i++) {
            const uint8_t v = order[i];
            printf("  %2u. %-24s %-12s 0x%03x  %.4f\n", i + 1, variants[v].name, policyNames[variants[v].policy], variants[v].features,
                   variantScore[v] / variantGames[v]);
        }
    }

    free(results);
    destroyMaps();
    return 0;
}
