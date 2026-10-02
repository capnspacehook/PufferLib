#ifndef SML_GAME_AREA_H
#define SML_GAME_AREA_H
#include "sml_state.h"

/* The game-area observation of super_mario_land_rl (game_area.py): PyBoy's
   game_area() for Super Mario Land, the 20x16 tiles below the HUD with the
   sprites drawn over them, each tile mapped to a small class by a per-world
   table (0 empty, 1 Mario, up to 44). PyBoy reads the scroll the LCD used on
   each row; at an update boundary that is hScrollX (FFA4), which the LYC 15
   handler loads into SCX below the HUD, and SCX itself is 0. */

enum {
    SML_GAME_AREA_WIDTH = 20,
    SML_GAME_AREA_HEIGHT = 16,
    SML_GAME_AREA_SIZE = SML_GAME_AREA_WIDTH * SML_GAME_AREA_HEIGHT,
    SML_TILE_CLASS_MAX = 44
};

/* Builds the per-world class tables. Call once before smlGameArea, and
   before starting threads that call it. */
void smlGameAreaInit(void);
/* Fills out row by row (out[y * SML_GAME_AREA_WIDTH + x]), top row first. */
void smlGameArea(const SmlState *s, float *out);

/* PyBoy tile identifiers: 0-255 are the tiles at 8000-8FFF (all sprites, and
   background tiles 80-FF under LCDC bit 4 = 0); 256-383 are background tiles
   00-7F, which then come from 9000-97FF. */
enum { SML_TILE_IDS = 384 };

typedef struct {
    uint16_t first, last;
} SmlTileRange;

typedef struct {
    uint8_t tileClass;
    uint8_t count;
    const SmlTileRange *ranges;
} SmlTileGroup;

#define SML_T(t) {(t), (t)}
#define SML_GROUP(name, tileClass, ...)                                        \
    static const SmlTileRange name##Ranges[] = {__VA_ARGS__};                  \
    static const SmlTileGroup name = {                                         \
        (tileClass), sizeof(name##Ranges) / sizeof(SmlTileRange), name##Ranges \
    }

/* The groups of game_area.py, in its order and with its classes. */
SML_GROUP(smlMario, 1, {0, 10}, {15, 26}, {31, 42}, {48, 58}, {66, 70});
SML_GROUP(smlMarioFireball, 2, SML_T(96), SML_T(110), SML_T(122));
SML_GROUP(smlCoin, 3, SML_T(244));
SML_GROUP(smlMushroom, 4, SML_T(131));
SML_GROUP(smlFlower, 5, SML_T(224), SML_T(229));
SML_GROUP(smlStar, 6, SML_T(134));
SML_GROUP(smlHeart, 7, SML_T(132));

#define SML_PIPE_TILES {368, 380}
#define SML_WORLD_4_EXTRA_PIPE_TILES {363, 366} /* normal blocks elsewhere */
#define SML_COMMON_BLOCK_TILES \
    {142, 143}, {230, 236}, {301, 304}, SML_T(340), {356, 359}, {381, 383}
#define SML_WORLD_1_JUMP_THROUGH_TILES {360, 362}
#define SML_WORLD_2_JUMP_THROUGH_TILES {352, 353}, SML_T(355)
SML_GROUP(smlWorld123Pipes, 8, SML_PIPE_TILES);
SML_GROUP(smlWorld4Pipes, 8, SML_PIPE_TILES, SML_WORLD_4_EXTRA_PIPE_TILES);
/* 319 is scenery in the other worlds. */
SML_GROUP(smlWorld1Blocks, 9, SML_COMMON_BLOCK_TILES, SML_WORLD_2_JUMP_THROUGH_TILES, SML_T(319), SML_WORLD_4_EXTRA_PIPE_TILES);
SML_GROUP(smlWorld2Blocks, 9, SML_COMMON_BLOCK_TILES, SML_WORLD_1_JUMP_THROUGH_TILES, SML_WORLD_4_EXTRA_PIPE_TILES);
SML_GROUP(smlWorld3Blocks, 9, SML_COMMON_BLOCK_TILES, SML_WORLD_1_JUMP_THROUGH_TILES, SML_WORLD_2_JUMP_THROUGH_TILES, SML_WORLD_4_EXTRA_PIPE_TILES);
SML_GROUP(smlWorld4Blocks, 9, SML_COMMON_BLOCK_TILES, SML_WORLD_1_JUMP_THROUGH_TILES, SML_WORLD_2_JUMP_THROUGH_TILES);
SML_GROUP(smlWorld1JumpThrough, 10, SML_WORLD_1_JUMP_THROUGH_TILES);
SML_GROUP(smlWorld2JumpThrough, 10, SML_WORLD_2_JUMP_THROUGH_TILES);
SML_GROUP(smlMovingBlock, 11, SML_T(239));
SML_GROUP(smlCrushBlocks, 12, {221, 222});
SML_GROUP(smlFallingStalactites, 13, SML_T(223));
SML_GROUP(smlFallingBlock, 14, SML_T(238));
SML_GROUP(smlPushableBlocks, 15, SML_T(128), SML_T(130), SML_T(354)); /* 354 is invisible in 2-2 */
SML_GROUP(smlQuestionBlock, 16, SML_T(129));
SML_GROUP(smlSpike, 17, SML_T(237));
SML_GROUP(smlLever, 18, SML_T(225)); /* the lever at a boss's gate */

SML_GROUP(smlGoomba, 19, SML_T(144));
SML_GROUP(smlKoopa, 20, {150, 153});
SML_GROUP(smlShell, 21, {154, 155});
SML_GROUP(smlExplosion, 22, {157, 158});
SML_GROUP(smlPiranhaPlant, 23, {146, 149});
SML_GROUP(smlBillLauncher, 24, {135, 136});
SML_GROUP(smlBulletBill, 25, SML_T(249));
SML_GROUP(smlFireball, 26, SML_T(226));
SML_GROUP(smlSpittingPlantSeed, 27, SML_T(227));
SML_GROUP(smlFlyingMothArrow, 28, SML_T(172), SML_T(188));

/* Enemies of one world, sharing tiles with those of another. */
#define SML_SHARED_ENEMY_1 {160, 163}, {176, 179}
#define SML_SHARED_ENEMY_2 {164, 167}, {180, 183}
#define SML_SHARED_ENEMY_3 {192, 193}, {208, 209}
#define SML_SHARED_ENEMY_4 {196, 199}, {212, 215}
SML_GROUP(smlMoth, 29, SML_SHARED_ENEMY_1);
SML_GROUP(smlFlyingMoth, 30, {192, 195}, {208, 211});
SML_GROUP(smlSphinx, 31, SML_SHARED_ENEMY_2);
SML_GROUP(smlBoneFish, 32, SML_SHARED_ENEMY_3);
SML_GROUP(smlSeahorse, 33, SML_SHARED_ENEMY_2);
SML_GROUP(smlRobot, 34, SML_SHARED_ENEMY_4);
SML_GROUP(smlFistRock, 35, SML_SHARED_ENEMY_2);
SML_GROUP(smlFlyingRock, 36, SML_T(171), SML_T(187));
SML_GROUP(smlFallingSpider, 37, SML_SHARED_ENEMY_4);
SML_GROUP(smlJumpingSpider, 29, SML_SHARED_ENEMY_1);
SML_GROUP(smlZombie, 38, SML_SHARED_ENEMY_1, {168, 169});
SML_GROUP(smlFireWorm, 31, SML_SHARED_ENEMY_2);
SML_GROUP(smlSpittingPlant, 39, SML_SHARED_ENEMY_3);
SML_GROUP(smlBouncingBoulder, 40, {194, 195}, {210, 211});

#define SML_DEAD_SHARED_ENEMY_1 {168, 169}
#define SML_DEAD_SHARED_ENEMY_2 {184, 185}
#define SML_DEAD_SHARED_ENEMY_3 {216, 217}
SML_GROUP(smlDeadMoth, 41, SML_DEAD_SHARED_ENEMY_1);
SML_GROUP(smlDeadFlyingMoth, 41, {200, 201});
SML_GROUP(smlDeadSphinx, 41, SML_DEAD_SHARED_ENEMY_2);
SML_GROUP(smlDeadRobot, 41, SML_DEAD_SHARED_ENEMY_3);
SML_GROUP(smlDeadFlyingRock, 41, SML_T(173));
SML_GROUP(smlDeadFistRock, 41, SML_DEAD_SHARED_ENEMY_2);
SML_GROUP(smlDeadFallingSpider, 41, SML_DEAD_SHARED_ENEMY_3);
SML_GROUP(smlDeadJumpingSpider, 41, SML_DEAD_SHARED_ENEMY_1);
SML_GROUP(smlDeadFireWorm, 41, SML_DEAD_SHARED_ENEMY_2);

SML_GROUP(smlBigSphinx, 42, SML_T(171), SML_T(187), {198, 199}, {202, 206}, {214, 215}, {218, 220});
SML_GROUP(smlBigSphinxFire, 43, {196, 197}, {212, 213});
SML_GROUP(smlBigFistRock, 44, {188, 189}, {204, 205}, {174, 175}, {190, 191}, {206, 207});

#define SML_BASE_GROUPS                                                         \
    &smlMario, &smlMarioFireball, &smlCoin, &smlMushroom, &smlFlower, &smlStar, \
        &smlHeart, &smlMovingBlock, &smlCrushBlocks, &smlFallingStalactites,    \
        &smlFallingBlock, &smlPushableBlocks, &smlQuestionBlock, &smlSpike,     \
        &smlLever, &smlGoomba, &smlKoopa, &smlShell, &smlExplosion,             \
        &smlPiranhaPlant, &smlBillLauncher, &smlBulletBill, &smlFireball

/* Each world's groups; a later group overrides an earlier one's tiles. */
static const SmlTileGroup *const smlWorld1Groups[] = {SML_BASE_GROUPS, &smlWorld123Pipes, &smlWorld1Blocks, &smlWorld1JumpThrough, &smlMoth, &smlDeadMoth, &smlFlyingMoth, &smlDeadFlyingMoth, &smlFlyingMothArrow, &smlSphinx, &smlDeadSphinx, &smlBigSphinx, &smlBigSphinxFire};
static const SmlTileGroup *const smlWorld2Groups[] = {
    SML_BASE_GROUPS, &smlWorld123Pipes, &smlWorld2Blocks, &smlWorld2JumpThrough, &smlBoneFish, &smlSeahorse, &smlRobot, &smlDeadRobot
};
static const SmlTileGroup *const smlWorld3Groups[] = {
    SML_BASE_GROUPS, &smlWorld123Pipes, &smlWorld3Blocks, &smlFlyingRock, &smlDeadFlyingRock, &smlFistRock, &smlDeadFistRock, &smlBouncingBoulder, &smlFallingSpider, &smlDeadFallingSpider, &smlJumpingSpider, &smlDeadJumpingSpider, &smlBigFistRock
};
static const SmlTileGroup *const smlWorld4Groups[] = {
    SML_BASE_GROUPS, &smlWorld4Pipes, &smlWorld4Blocks, &smlZombie, &smlSpittingPlant, &smlSpittingPlantSeed, &smlFireWorm, &smlDeadFireWorm
};
#undef SML_BASE_GROUPS
#undef SML_GROUP
#undef SML_T

static uint8_t smlTileClasses[4][SML_TILE_IDS];

static void smlBuildTileClasses(uint8_t classes[SML_TILE_IDS], const SmlTileGroup *const *groups, size_t count) {
    for (size_t g = 0; g < count; g++) {
        for (int r = 0; r < groups[g]->count; r++) {
            for (int t = groups[g]->ranges[r].first; t <= groups[g]->ranges[r].last;
                 t++) {
                classes[t] = groups[g]->tileClass;
            }
        }
    }
}

void smlGameAreaInit(void) {
    smlBuildTileClasses(smlTileClasses[0], smlWorld1Groups, sizeof(smlWorld1Groups) / sizeof(smlWorld1Groups[0]));
    smlBuildTileClasses(smlTileClasses[1], smlWorld2Groups, sizeof(smlWorld2Groups) / sizeof(smlWorld2Groups[0]));
    smlBuildTileClasses(smlTileClasses[2], smlWorld3Groups, sizeof(smlWorld3Groups) / sizeof(smlWorld3Groups[0]));
    smlBuildTileClasses(smlTileClasses[3], smlWorld4Groups, sizeof(smlWorld4Groups) / sizeof(smlWorld4Groups[0]));
}

/* Python's floor division by 8, for sprite coordinates left of or above the
   screen. */
static inline int smlFloorDiv8(int v) {
    return v >= 0 ? v / 8 : -((7 - v) / 8);
}

void smlGameArea(const SmlState *s, float *out) {
    int world = s->worldLevel >> 4;
    const uint8_t *classes =
        smlTileClasses[world >= 1 && world <= 4 ? world - 1 : 0];

    /* Rows 2-17 of the screen, all below the HUD split. */
    const uint8_t *map = (s->lcdc & 0x08) ? s->windowMap : s->bgMap;
    int signedTiles = !(s->lcdc & 0x10);
    int column0 = s->scrollX / 8;
    int row0 = 2 + (s->scrollYEnabled ? s->scrollY : 0) / 8;
    for (int y = 0; y < SML_GAME_AREA_HEIGHT; y++) {
        const uint8_t *mapRow = map + ((row0 + y) % 32) * 32;
        for (int x = 0; x < SML_GAME_AREA_WIDTH; x++) {
            uint8_t tile = mapRow[(column0 + x) % 32];
            const uint8_t cls =
                classes[signedTiles && tile < 0x80 ? tile + 256 : tile];
            out[y * SML_GAME_AREA_WIDTH + x] = cls;
        }
    }

    /* Sprites in OAM order, each at the tile its top-left pixel is in. */
    int height = (s->lcdc & 0x04) ? 16 : 8;
    for (int i = 0; i < 40; i++) {
        int sy = s->oam[i].y - 16, sx = s->oam[i].x - 8;
        if (sy <= -height || sy >= SML_HEIGHT || sx <= -8 || sx >= SML_WIDTH) {
            continue;
        }
        int x = smlFloorDiv8(sx), y = smlFloorDiv8(sy) - 2;
        if (x >= 0 && x < SML_GAME_AREA_WIDTH && y >= 0 &&
            y < SML_GAME_AREA_HEIGHT) {
            out[y * SML_GAME_AREA_WIDTH + x] =
                classes[s->oam[i].tile];
        }
    }

    /* _drawMario: while Mario is invincible (a star, or shrinking and
       blinking after a hit), his sprites can be hidden, so mark the 2x2 tiles
       at OAM entry 3, or below his entity's y when that sprite is off the
       bottom. game_area.py lets a column of -1 wrap to the right edge; this
       drops it. */
    if (s->invincibilityTimer || s->superStatus == 3 || s->superStatus == 4) {
        const SmlOamEntry *head = &s->oam[3];
        int x1 = smlFloorDiv8(head->x - 8);
        if (x1 > SML_GAME_AREA_WIDTH - 1) {
            x1 = SML_GAME_AREA_WIDTH - 1;
        }
        int x2 = (head->flags & 0x20) ? x1 - 1 : x1 + 1;
        if (x2 > SML_GAME_AREA_WIDTH - 1) {
            x2 = SML_GAME_AREA_WIDTH - 1;
        }
        int y1 = smlFloorDiv8(head->y - 16) - 1;
        if (y1 >= SML_GAME_AREA_HEIGHT) {
            y1 = smlFloorDiv8(s->mario.y - 22) - 1;
        }
        int ys[2] = {y1, y1 - 1}, xs[2] = {x1, x2};
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                if (ys[j] >= 0 && ys[j] < SML_GAME_AREA_HEIGHT && xs[k] >= 0 &&
                    xs[k] < SML_GAME_AREA_WIDTH) {
                    out[ys[j] * SML_GAME_AREA_WIDTH + xs[k]] = 1.0f;
                }
            }
        }
    }
}

#endif
