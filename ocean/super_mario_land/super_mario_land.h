#ifndef SML_ENV_H
#define SML_ENV_H

#include "sml_core.h"
#include "sml_game_area.h"
#include "sml_state.h"

typedef float obs_t;
#include "pufferenv.h"
#include "raylib.h"

#define ACT_SIZES {6}
#define NUM_ATNS 1
#define OBS_SIZE 368

#define PUF_STEPS_PER_SEC 60

struct LevelLog {
    float progress;
    float cleared;
};

#define SML_LEVELS_COUNT 10

const uint16_t LEVEL_1_1_END = 2602;
const uint16_t LEVEL_1_2_END = 2442;
const uint16_t LEVEL_1_3_END = 2588;
const uint16_t LEVEL_2_1_END = 2762;
const uint16_t LEVEL_2_2_END = 2442;
const uint16_t LEVEL_3_1_END = 3882;
const uint16_t LEVEL_3_2_END = 2762;
const uint16_t LEVEL_3_3_END = 2588;
const uint16_t LEVEL_4_1_END = 3882;
const uint16_t LEVEL_4_2_END = 3402;
const uint32_t GAME_END = LEVEL_1_1_END + LEVEL_1_2_END + LEVEL_1_3_END + LEVEL_2_1_END + LEVEL_2_2_END + LEVEL_3_1_END + LEVEL_3_2_END + LEVEL_3_3_END + LEVEL_4_1_END + LEVEL_4_2_END;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;

    float progress;
    float coins;
    float mushrooms;
    float superballs;
    float timesHit;
    float stars;
    float hearts;
    float deaths;
    float levelsCleared;

    struct LevelLog levels[SML_LEVELS_COUNT];

    float n;
};

typedef enum {
    SMALL,
    MUSHROOM,
    SUPERBALL,
} Powerup;

struct SmlProfile {
    const uint8_t nextLevelIdx;
    const char *path;
    uint8_t *data;
};

#define SML_PROFILES_COUNT 45
static struct SmlProfile profiles[SML_PROFILES_COUNT] = {
    {4, "./ocean/super_mario_land/profiles/1-1-0_e.profile.bin", NULL},
    {4, "./ocean/super_mario_land/profiles/1-1-1_e.profile.bin", NULL},
    {4, "./ocean/super_mario_land/profiles/1-1-2_e.profile.bin", NULL},
    {4, "./ocean/super_mario_land/profiles/1-1-3_e.profile.bin", NULL},
    {8, "./ocean/super_mario_land/profiles/1-2-0_e.profile.bin", NULL},
    {8, "./ocean/super_mario_land/profiles/1-2-1_e.profile.bin", NULL},
    {8, "./ocean/super_mario_land/profiles/1-2-2_e.profile.bin", NULL},
    {8, "./ocean/super_mario_land/profiles/1-2-3_e.profile.bin", NULL},
    {12, "./ocean/super_mario_land/profiles/1-3-0_e.profile.bin", NULL},
    {12, "./ocean/super_mario_land/profiles/1-3-1_e.profile.bin", NULL},
    {12, "./ocean/super_mario_land/profiles/1-3-2_e.profile.bin", NULL},
    {12, "./ocean/super_mario_land/profiles/1-3-3_e.profile.bin", NULL},
    {16, "./ocean/super_mario_land/profiles/2-1-0_e.profile.bin", NULL},
    {16, "./ocean/super_mario_land/profiles/2-1-1_e.profile.bin", NULL},
    {16, "./ocean/super_mario_land/profiles/2-1-2_e.profile.bin", NULL},
    {16, "./ocean/super_mario_land/profiles/2-1-3_e.profile.bin", NULL},
    {20, "./ocean/super_mario_land/profiles/2-2-0_e.profile.bin", NULL},
    {20, "./ocean/super_mario_land/profiles/2-2-1_e.profile.bin", NULL},
    {20, "./ocean/super_mario_land/profiles/2-2-2_e.profile.bin", NULL},
    {20, "./ocean/super_mario_land/profiles/2-2-3_e.profile.bin", NULL},
    {26, "./ocean/super_mario_land/profiles/3-1-0_e.profile.bin", NULL},
    {26, "./ocean/super_mario_land/profiles/3-1-1_e.profile.bin", NULL},
    {26, "./ocean/super_mario_land/profiles/3-1-2_e.profile.bin", NULL},
    {26, "./ocean/super_mario_land/profiles/3-1-3_e.profile.bin", NULL},
    {26, "./ocean/super_mario_land/profiles/3-1-4_e.profile.bin", NULL},
    {26, "./ocean/super_mario_land/profiles/3-1-5_e.profile.bin", NULL},
    {30, "./ocean/super_mario_land/profiles/3-2-0_e.profile.bin", NULL},
    {30, "./ocean/super_mario_land/profiles/3-2-1_e.profile.bin", NULL},
    {30, "./ocean/super_mario_land/profiles/3-2-2_e.profile.bin", NULL},
    {30, "./ocean/super_mario_land/profiles/3-2-3_e.profile.bin", NULL},
    {34, "./ocean/super_mario_land/profiles/3-3-0_e.profile.bin", NULL},
    {34, "./ocean/super_mario_land/profiles/3-3-1_e.profile.bin", NULL},
    {34, "./ocean/super_mario_land/profiles/3-3-2_e.profile.bin", NULL},
    {34, "./ocean/super_mario_land/profiles/3-3-3_e.profile.bin", NULL},
    {40, "./ocean/super_mario_land/profiles/4-1-0_e.profile.bin", NULL},
    {40, "./ocean/super_mario_land/profiles/4-1-1_e.profile.bin", NULL},
    {40, "./ocean/super_mario_land/profiles/4-1-2_e.profile.bin", NULL},
    {40, "./ocean/super_mario_land/profiles/4-1-3_e.profile.bin", NULL},
    {40, "./ocean/super_mario_land/profiles/4-1-4_e.profile.bin", NULL},
    {40, "./ocean/super_mario_land/profiles/4-1-5_e.profile.bin", NULL},
    {0, "./ocean/super_mario_land/profiles/4-2-0_e.profile.bin", NULL},
    {0, "./ocean/super_mario_land/profiles/4-2-1_e.profile.bin", NULL},
    {0, "./ocean/super_mario_land/profiles/4-2-2_e.profile.bin", NULL},
    {0, "./ocean/super_mario_land/profiles/4-2-3_e.profile.bin", NULL},
    {0, "./ocean/super_mario_land/profiles/4-2-4_e.profile.bin", NULL},
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    unsigned int num_agents;
    unsigned int rng;

    float progressReward;
    float coinReward;
    float powerupReward;
    float hitPunishment;
    float heartReward;
    float levelClearReward;
    float deathPunishment;
    float gameOverPunishment;

    uint8_t *rom;
    uint8_t profileIdx;
    SmlState *prevState;
    SmlState *state;
    bool isEval;

    float episodeReward;
    uint32_t steps;

    uint16_t initialPos;
    uint16_t furthestPos;
    uint32_t progress;
    uint16_t coins;
    uint8_t mushrooms;
    uint8_t superballs;
    uint8_t timesHit;
    uint8_t stars;
    uint8_t hearts;
    uint8_t deaths;
    uint8_t levelsCleared;

    struct LevelLog levels[SML_LEVELS_COUNT];

    Texture2D texture;
    uint8_t *pixels;
};
typedef Env SmlEnv;

#define SML_LEVEL_HASH(world, level) ((world << 4) | level)

#define SML_LEVEL_IDX(state) (smlLevelIdx(state->worldLevel >> 4, state->worldLevel & 0x0F))

uint8_t smlLevelIdx(uint8_t world, uint8_t level) {
    switch (SML_LEVEL_HASH(world, level)) {
    case SML_LEVEL_HASH(1, 1):
        return 0;
    case SML_LEVEL_HASH(1, 2):
        return 1;
    case SML_LEVEL_HASH(1, 3):
        return 2;
    case SML_LEVEL_HASH(2, 1):
        return 3;
    case SML_LEVEL_HASH(2, 2):
        return 4;
    case SML_LEVEL_HASH(3, 1):
        return 5;
    case SML_LEVEL_HASH(3, 2):
        return 6;
    case SML_LEVEL_HASH(3, 3):
        return 7;
    case SML_LEVEL_HASH(4, 1):
        return 8;
    case SML_LEVEL_HASH(4, 2):
        return 9;
    default:
        printf("Invalid level %d %d\n", world, level);
        exit(1);
    }
}

void addLog(SmlEnv *env) {
    Log *log = &env->log;

    const uint32_t totalProgress = env->progress + (env->furthestPos - env->initialPos);

    log->perf += (float)totalProgress / (float)GAME_END;
    log->score += totalProgress;
    log->episode_return += env->episodeReward;
    log->episode_length += env->steps;

    log->progress += totalProgress;
    log->coins += env->coins;
    log->mushrooms += env->mushrooms;
    log->superballs += env->superballs;
    log->timesHit += env->timesHit;
    log->stars += env->stars;
    log->hearts += env->hearts;
    log->deaths += env->deaths;
    log->levelsCleared += env->levelsCleared;

    const uint8_t levelIdx = SML_LEVEL_IDX(env->state);
    log->levels[levelIdx].progress += env->furthestPos - env->initialPos;

    for (uint8_t i = 0; i < SML_LEVELS_COUNT; i++) {
        log->levels[i].progress += env->levels[i].progress;
        log->levels[i].cleared += env->levels[i].cleared;
    }

    log->n += 1.0f;
}

void puf_log(Log *log, Dict *out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);

    dict_set(out, "progress", log->progress);
    dict_set(out, "coins", log->coins);
    dict_set(out, "mushrooms", log->mushrooms);
    dict_set(out, "superballs", log->superballs);
    dict_set(out, "timesHit", log->timesHit);
    dict_set(out, "stars", log->stars);
    dict_set(out, "hearts", log->hearts);
    dict_set(out, "deaths", log->deaths);
    dict_set(out, "levelsCleared", log->levelsCleared);

    dict_set(out, "1-1_progress", log->levels[0].progress);
    dict_set(out, "1-1_cleared", log->levels[0].cleared);
    dict_set(out, "1-2_progress", log->levels[1].progress);
    dict_set(out, "1-2_cleared", log->levels[1].cleared);
    dict_set(out, "1-3_progress", log->levels[2].progress);
    dict_set(out, "1-3_cleared", log->levels[2].cleared);
    dict_set(out, "2-1_progress", log->levels[3].progress);
    dict_set(out, "2-1_cleared", log->levels[3].cleared);
    dict_set(out, "2-2_progress", log->levels[4].progress);
    dict_set(out, "2-2_cleared", log->levels[4].cleared);
    dict_set(out, "3-1_progress", log->levels[5].progress);
    dict_set(out, "3-1_cleared", log->levels[5].cleared);
    dict_set(out, "3-2_progress", log->levels[6].progress);
    dict_set(out, "3-2_cleared", log->levels[6].cleared);
    dict_set(out, "3-3_progress", log->levels[7].progress);
    dict_set(out, "3-3_cleared", log->levels[7].cleared);
    dict_set(out, "4-1_progress", log->levels[8].progress);
    dict_set(out, "4-1_cleared", log->levels[8].cleared);
    dict_set(out, "4-2_progress", log->levels[9].progress);
    dict_set(out, "4-2_cleared", log->levels[9].cleared);

    dict_set(out, "n", log->n);
}

uint16_t bcdToDec(uint16_t v) {
    return (v >> 4) * 10 + (v & 0xf);
}

uint16_t decToBcd(uint16_t v) {
    return ((v / 10) << 4) | (v % 10);
}

static inline int randInt(unsigned int *state, const int min, const int max) {
    return min + rand_r(state) % (max - min + 1);
}

/* Map ROM object types to the semantic classes used by smlGameArea.
   Use the groups' class fields so the two observations share class IDs.
   This identifies the object's main type, not every tile in its animation.
   Empty, unknown and sprite-dependent types return 0 (unclassified).
   In particular, 0D retains its previous sprite, so its ID alone is not
   enough to distinguish the class. No smlGameAreaInit call is required. */
static inline uint8_t smlEnemyGameAreaClass(uint8_t id) {
    switch (id) {
    case 0x00:
    case 0x11:
        return smlGoomba.tileClass;
    case 0x02:
    case 0x55:
        return smlPiranhaPlant.tileClass;
    case 0x04:
    case 0x12:
        return smlKoopa.tileClass;
    case 0x05:
        return smlShell.tileClass;
    case 0x07:
    case 0x13:
    case 0x14:
        return smlWorld1Blocks.tileClass; /* lift-block tile 230 */
    case 0x08:
        return smlBigSphinx.tileClass;
    case 0x09:
        return smlSpittingPlant.tileClass;
    case 0x0A:
    case 0x0B:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
        return smlMovingBlock.tileClass;
    case 0x0C:
        return smlCrushBlocks.tileClass;
    case 0x0E:
    case 0x20:
        return smlMoth.tileClass; /* also the world-3 jumping spider */
    case 0x0F:
    case 0x15:
    case 0x19:
    case 0x1C:
    case 0x2F:
    case 0x30:
    case 0x3D:
    case 0x3E:
    case 0x40:
    case 0x41:
    case 0x43:
    case 0x44:
        return smlDeadMoth.tileClass; /* common dead-enemy class */
    case 0x10:
        return smlBoneFish.tileClass;
    case 0x16:
    case 0x17:
    case 0x18:
        return smlRobot.tileClass;
    case 0x1D:
    case 0x24:
        return smlSeahorse.tileClass;
    case 0x1E:
        return smlBigSphinxFire.tileClass;
    case 0x21:
    case 0x26:
    case 0x27:
    case 0x46:
    case 0x4F:
        return smlExplosion.tileClass;
    case 0x23:
    case 0x54:
        return smlFireball.tileClass;
    case 0x25:
    case 0x53:
        return smlFallingSpider.tileClass;
    case 0x28:
    case 0x29:
        return smlMushroom.tileClass;
    case 0x2A:
    case 0x2B:
        return smlHeart.tileClass;
    case 0x2C:
    case 0x34:
        return smlStar.tileClass;
    case 0x2D:
    case 0x2E:
        return smlFlower.tileClass;
    case 0x31:
        return smlFistRock.tileClass;
    case 0x32:
        return smlBigFistRock.tileClass;
    case 0x35:
        return smlFallingStalactites.tileClass;
    case 0x36:
    case 0x37:
    case 0x4C:
    case 0x4D:
    case 0x4E:
        return smlFallingBlock.tileClass;
    case 0x3C:
        return smlFlyingRock.tileClass;
    case 0x3F:
        return smlSphinx.tileClass; /* also the world-4 fire worm */
    case 0x42:
        return smlFlyingMoth.tileClass;
    case 0x45:
        return smlFlyingMothArrow.tileClass;
    case 0x47:
    case 0x52:
        return smlBouncingBoulder.tileClass;
    case 0x49:
        return smlBillLauncher.tileClass;
    case 0x4B:
        return smlBulletBill.tileClass;
    case 0x51:
        return smlSpittingPlantSeed.tileClass;
    case 0x56:
    case 0x57:
        return smlZombie.tileClass;
    default:
        return 0;
    }
}

uint16_t levelPos(const SmlState *s) {
    int phase = s->scrollX & 15;
    if ((s->scrollX & 8) && s->columnToggle == 0) {
        phase -= 16;
    }

    return s->levelProgress * 16 + phase + s->mario.x + 9;
}

uint8_t getActions(uint8_t agentAction) {
    switch (agentAction) {
    default:
        return 0;
    case 1:
        return SML_A;
    case 2:
        return SML_LEFT | SML_B;
    case 3:
        return SML_RIGHT | SML_B;
    case 4:
        return SML_LEFT | SML_B | SML_A;
    case 5:
        return SML_RIGHT | SML_B | SML_A;
    }
}

void puf_init(SmlEnv *env, Dict *kwargs) {
    env->num_agents = 1;

    env->progressReward = dict_get(kwargs, "progress_reward");
    env->coinReward = dict_get(kwargs, "coin_reward");
    env->powerupReward = dict_get(kwargs, "powerup_reward");
    env->hitPunishment = dict_get(kwargs, "hit_reward");
    env->heartReward = dict_get(kwargs, "heart_reward");
    env->levelClearReward = dict_get(kwargs, "level_clear_reward");
    env->deathPunishment = dict_get(kwargs, "death_reward");
    env->gameOverPunishment = dict_get(kwargs, "game_over_reward");

    env->isEval = (bool)dict_get(kwargs, "PUFFER_IS_EVAL");

    uint8_t *rom = (uint8_t *)calloc(1, SML_ROM_SIZE);
    if (smlLoadRom("./ocean/super_mario_land/sml_patched.gb", rom) != 0) {
        printf("rom load failed\n");
        exit(1);
    }
    env->rom = rom;

    env->state = (SmlState *)calloc(1, sizeof(SmlState));
    env->prevState = (SmlState *)calloc(1, sizeof(SmlState));

    static pthread_mutex_t sml_mutex = PTHREAD_MUTEX_INITIALIZER;
    static bool initialized = false;
    pthread_mutex_lock(&sml_mutex);
    if (!initialized) {
        initialized = true;

        for (uint8_t i = 0; i < SML_PROFILES_COUNT; i++) {
            profiles[i].data = (uint8_t *)calloc(1, SML_PROFILE_SIZE);
            if (smlInitProfile(profiles[i].path, profiles[i].data) != 0) {
                printf("profile load %d failed\n", i);
                exit(1);
            }
        }

        smlGameAreaInit();
    }
    pthread_mutex_unlock(&sml_mutex);
}

void puf_close(SmlEnv *env) {
    free(env->rom);
    free(env->state);
    free(env->prevState);

    if (env->pixels != NULL) {
        free(env->pixels);
        UnloadTexture(env->texture);
        CloseWindow();
    }
}

void puf_render(SmlEnv *env) {
    if (!IsWindowReady()) {
        InitWindow(SML_WIDTH * 4, SML_HEIGHT * 4, "Super Mario Land C port");
        SetTargetFPS(60);
        const Image img = GenImageColor(SML_WIDTH, SML_HEIGHT, WHITE);
        env->texture = LoadTextureFromImage(img);
        UnloadImage(img);
        env->pixels = (uint8_t *)calloc(1, SML_WIDTH * SML_HEIGHT * 4);
    }

    smlRenderFrame(env->state, env->pixels);
    UpdateTexture(env->texture, env->pixels);
    BeginDrawing();
    DrawTextureEx(env->texture, (Vector2){0, 0}, 0, 4, WHITE);
    EndDrawing();
    puf_web_vsync();
}

void setObs(SmlEnv *env) {
    const SmlState *state = env->state;

    float *obs = env->agents[0].observations;
    memset(obs, 0x0, OBS_SIZE * sizeof(float));

    smlGameArea(state, obs);

    uint8_t direction = 0;
    if (state->mario.walkDirection == 0x20) {
        direction = 1;
    } else if (state->mario.walkDirection == 0x10) {
        direction = 2;
    } else if (state->mario.walkDirection == 1) {
        direction = 3;
    }

    uint8_t powerup = 0;
    if (state->superStatus == 2) {
        powerup = 1;
    }
    if (state->superballMario) {
        powerup = 2;
    }

    const uint16_t timeLeft =
        (bcdToDec(state->timerHundreds) * 100) + bcdToDec(state->timerDigits);

    uint16_t offset = SML_GAME_AREA_SIZE;

    obs[offset++] = (float)state->mario.x / 255.0f;
    obs[offset++] = (float)state->mario.y / 255.0f;

    obs[offset + state->mario.jumpState] = 1.0f;
    offset += 3;
    obs[offset + direction] = 1.0f;
    offset += 4;

    obs[offset++] = (float)state->mario.gait / 6.0f;
    obs[offset++] = (float)state->mario.speed / 48.0f;

    // TODO: level progress percentage
    obs[offset + powerup] = 1.0f;
    offset += 3;
    obs[offset++] = (float)state->invincibilityTimer / 240.0f;
    obs[offset++] = (float)timeLeft / 400.0f;
    obs[offset++] = (float)bcdToDec(state->lives) / 99.0f;
    obs[offset++] = (float)bcdToDec(state->coins) / 99.0f;

    for (uint8_t i = 0; i < 10; i++) {
        const SmlEnemy entity = state->enemies[i];
        if (entity.id == 0xff) {
            offset += 3;
            continue;
        }

        obs[offset++] =
            (float)smlEnemyGameAreaClass(entity.id) / SML_TILE_CLASS_MAX;
        obs[offset++] = (float)entity.x / 255.0f;
        obs[offset++] = (float)entity.y / 255.0f;
    }
}

Powerup getPowerup(const SmlState *s) {
    if (s->superballMario) {
        return SUPERBALL;
    }
    if (s->superStatus == 1 || s->superStatus == 2) {
        return MUSHROOM;
    }
    return SMALL;
}

SmlStepStatus stepStatus(const SmlState *s, const SmlStepStatus status) {
    if (status == SML_STEP_UNSUPPORTED) {
        return SML_STEP_UNSUPPORTED;
    }

    switch (s->gameState) {
    case 0x1:
    case 0x3:
    case 0x4:
    case 0x3B:
    case 0x3C:
        if (s->lives == 0) {
            return SML_STEP_GAME_OVER;
        }
        return SML_STEP_DEAD;
    case 0x39:
    case 0x3A:
        return SML_STEP_GAME_OVER;
    case 0x7:
    case 0x08:
    case 0x12:
    case 0x1C:
        return SML_STEP_LEVEL_CLEAR;
    default:
        return SML_STEP_READY;
    }
}

bool isResetStatus(const SmlStepStatus status) {
    switch (status) {
    case SML_STEP_GAME_OVER:
    case SML_STEP_UNSUPPORTED:
        return true;
    default:
        return false;
    }
}

void setReward(SmlEnv *env, const uint16_t curPos, const SmlStepStatus status, bool died) {
    float reward = 0.0f;

    if (status == SML_STEP_GAME_OVER) {
        died = true;
    }

    if (curPos > env->furthestPos) {
        reward += (curPos - env->furthestPos) * env->progressReward;
    }

    const uint8_t coins = bcdToDec(env->state->coins);
    const uint8_t prevCoins = bcdToDec(env->prevState->coins);
    if (coins != prevCoins) {
        uint8_t coinsGathered = 0;
        if (coins < prevCoins) {
            coinsGathered = coins + 100 - prevCoins;
        } else {
            coinsGathered = coins - prevCoins;
        }

        reward += coinsGathered * env->coinReward;
        env->coins += coinsGathered;
    }

    const Powerup curPowerup = getPowerup(env->state);
    const Powerup prevPowerup = getPowerup(env->prevState);
    if (curPowerup > prevPowerup) {
        reward += env->powerupReward;

        if (prevPowerup == SMALL) {
            env->mushrooms++;
        } else if (curPowerup == SUPERBALL) {
            env->superballs++;
        }
    } else if (!died && prevPowerup != SMALL && curPowerup == SMALL) {
        reward += env->hitPunishment;
        env->timesHit++;
    }
    if (env->state->invincibilityTimer > env->prevState->invincibilityTimer) {
        reward += env->powerupReward;
        env->stars++;
    }

    if (env->state->lives > env->prevState->lives) {
        reward += env->heartReward;
        env->hearts++;
    } else if (died) {
        reward += env->deathPunishment;
        env->deaths++;
    }

    if (status == SML_STEP_LEVEL_CLEAR) {
        reward += env->levelClearReward;
        env->levelsCleared++;
    } else if (status == SML_STEP_GAME_OVER) {
        reward += env->gameOverPunishment;
    }

    env->agents[0].rewards[0] = reward;
    env->episodeReward += reward;
}

void setTimer(SmlEnv *env, const uint16_t timer) {
    env->state->timerHundreds = timer / 100;
    const uint8_t timerTens = timer - (env->state->timerHundreds * 100);
    env->state->timerDigits = decToBcd(timerTens);
    env->state->timerTicks = 0x28;
}

void puf_reset(SmlEnv *env) {
    if (env->isEval) {
        env->profileIdx = 0;
    } else {
        env->profileIdx = randInt(&env->rng, 0, SML_PROFILES_COUNT - 1);
    }
    const uint8_t *profile = profiles[env->profileIdx].data;
    smlInitProfileBytes(env->state, env->rom, profile);

    if (env->isEval) {
        env->state->lives = 2;
        env->state->coins = 0;
        env->state->superStatus = 0;
        env->state->superballMario = 0;
        env->state->invincibilityTimer = 0;

        setTimer(env, 400);
    } else  {
        setTimer(env, randInt(&env->rng, 100, 400));

        env->state->lives = decToBcd(randInt(&env->rng, 0, 4));
        env->state->coins = decToBcd(randInt(&env->rng, 0, 99));
        if (randInt(&env->rng, 0, 3) == 0) {
            switch (randInt(&env->rng, 0, 1)) {
            case 0:
                env->state->superStatus = 2;
                env->state->mario.pose |= 0x10;
                break;
            case 1:
                env->state->superStatus = 2;
                env->state->mario.pose |= 0x10;
                env->state->superballMario = 2;
                break;
            }
        }
        if (randInt(&env->rng, 0, 3) == 0) {
            env->state->invincibilityTimer = randInt(&env->rng, 40, 240);
        }
    }

    memcpy(env->prevState, env->state, sizeof(SmlState));

    env->episodeReward = 0.0f;
    env->steps = 0;

    env->initialPos = levelPos(env->state);
    env->furthestPos = env->initialPos;
    env->progress = 0;
    env->coins = 0;
    env->mushrooms = 0;
    env->superballs = 0;
    env->timesHit = 0;
    env->stars = 0;
    env->hearts = 0;
    env->deaths = 0;
    env->levelsCleared = 0;

    memset(env->levels, 0x0, sizeof(env->levels));

    setObs(env);
}

void puf_step(SmlEnv *env) {
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0.0f;
    memcpy(env->prevState, env->state, sizeof(SmlState));

    const uint8_t agentAction = (uint8_t)env->agents[0].actions[0];
    const uint8_t actions = getActions(agentAction);

    bool died = false;
    bool needsReset = false;
    SmlStepStatus status = SML_STEP_UNSUPPORTED;
    for (uint8_t i = 0; i < 4; i++) {
        status = smlStepFrame(env->state, actions);
        smlRunPipe(env->state, actions);
        status = stepStatus(env->state, status);
        env->steps++;

        if (status == SML_STEP_DEAD) {
            died = true;
            while (!smlAgentCanAct(env->state)) {
                status = smlStepFrame(env->state, actions);
                status = stepStatus(env->state, status);

                if (isResetStatus(status)) {
                    needsReset = true;
                    break;
                }
            }

            const uint8_t timerTens = bcdToDec(env->state->timerDigits);
            const uint16_t timer = (env->state->timerHundreds * 100) + timerTens;
            if (timer > 20) {
                setTimer(env, timer - 20);
            }

            break;
        }

        if (isResetStatus(status)) {
            needsReset = true;
            break;
        } else if (status == SML_STEP_LEVEL_CLEAR) {
            break;
        }
    }

    const uint16_t curPos = levelPos(env->state);
    setReward(env, curPos, status, died);

    if (curPos > env->furthestPos) {
        env->furthestPos = curPos;
    }

    if (status == SML_STEP_LEVEL_CLEAR) {
        const uint8_t levelIdx = SML_LEVEL_IDX(env->state);
        env->levels[levelIdx].progress += env->furthestPos - env->initialPos;
        env->levels[levelIdx].cleared += 1;

        memcpy(env->prevState, env->state, sizeof(SmlState));
        env->profileIdx = profiles[env->profileIdx].nextLevelIdx;
        smlInitProfileBytes(env->state, env->rom, profiles[env->profileIdx].data);

        env->state->lives = env->prevState->lives;
        env->state->coins = env->prevState->coins;
        env->state->superStatus = env->prevState->superStatus;
        if (env->state->superStatus == 2) {
            env->state->mario.pose |= 0x10;
        }
        env->state->superballMario = env->prevState->superballMario;
        env->state->invincibilityTimer = env->prevState->invincibilityTimer;

        env->progress += env->furthestPos - env->initialPos;
        env->initialPos = levelPos(env->state);
        env->furthestPos = env->initialPos;
    }

    if (needsReset) {
        env->agents[0].terminals[0] = 1;

        addLog(env);
        puf_reset(env);
        return;
    }

    setObs(env);
}

#endif
