#ifndef SML_STATE_H
#define SML_STATE_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

enum {
    SML_WIDTH = 160,
    SML_HEIGHT = 144,
    SML_ROM_SIZE = 0x10000,
    SML_PROFILE_SIZE = 0x8000,
    SML_MEM_BASE = 0x8000 /* the Game Boy address of mem[0] */
};
enum {
    SML_A = 0x10,
    SML_B = 0x20,
    SML_SELECT = 0x40,
    SML_START = 0x80,
    SML_RIGHT = 0x01,
    SML_LEFT = 0x02,
    SML_UP = 0x04,
    SML_DOWN = 0x08
};

/* Result of one step. Terminal and handoff states stop advancing, and an
   unsupported step returns before changing the state. */
typedef enum {
    SML_STEP_READY = 0,
    SML_STEP_DEAD,
    SML_STEP_GAME_OVER,
    SML_STEP_LEVEL_CLEAR,
    SML_STEP_UNSUPPORTED
} SmlStepStatus;

/* A hardware sprite, in OAM (FE00) and in the game's OAM buffer (C000). */
typedef struct {
    uint8_t y, x, tile, flags;
} SmlOamEntry;

/* OAM buffer entries by use. */
enum {
    SML_OAM_SUPERBALL = 0, /* 0-2: projectiles */
    SML_OAM_BOUNCED_BLOCK = 11,
    SML_OAM_FLOATIES = 12, /* 12-19: four two-sprite floaties */
    SML_OAM_ENEMIES = 20   /* 20-39 */
};

/* Mario (C200) and the four brick fragments (C210-C240). Their first seven
   bytes are what the sprite programs (3:4823) draw; smlVerticalMotion moves
   all five along the jump table. The walking fields are Mario's only. */
typedef struct {
    uint8_t visibility; /* 00 drawn, 80 drawn off screen (blinking), FF gone */
    uint8_t y, x;
    uint8_t pose;     /* sprite program; Mario's upper nibble is Super Mario */
    uint8_t priority; /* 80 draws behind the background (pipes) */
    uint8_t facing;   /* 20 flips left; fragments move left while nonzero */
    uint8_t palette;
    uint8_t jumpState; /* 0 on the ground, 1 rising, 2 falling */
    uint8_t jumpIndex; /* position in the jump table at $216D */
    uint8_t jumpCut;   /* index a released A applies next update */
    uint8_t grounded;
    uint8_t walkFrames;
    uint8_t speed;
    uint8_t walkDirection; /* 10 right, 20 left, 01 reversing */
    uint8_t gait;          /* 0 standing, 2 walking, 4 running */
    uint8_t gaitPhase;
} SmlEntity;

/* An enemy, platform or powerup slot (D100-D190; slot 9 holds powerups), and
   the FFC0 buffer each is copied into while it runs. Only the bytes before
   `unused` are copied. */
typedef struct {
    uint8_t id;     /* FF: empty slot */
    uint8_t motion; /* pixels per move: x in bits 0-3, y in bits 4-7 */
    uint8_t y, x;
    uint8_t scriptIndex;
    uint8_t direction; /* bit 0 right, bit 1 down */
    uint8_t sprite;    /* index into the enemy sprite tables */
    uint8_t flags;     /* movement: turn or stop at walls and edges */
    uint8_t moveCount; /* moves left in the current script command */
    uint8_t
        moveDelay;    /* updates between moves: count in bits 4-7, period in 0-3 */
    uint8_t size;     /* hitbox: height in bits 0-3, width in 4-6; bit 7 a platform */
    uint8_t carrying; /* Mario rides this object */
    uint8_t health;   /* hits left in bits 0-5, stomp reward in 6-7 */
    uint8_t unused[3];
} SmlEnemy;

enum { SML_ENEMY_SLOTS = 10,
       SML_ENEMY_COPY = offsetof(SmlEnemy, unused) };

/* One step is one pass of the ROM's main loop followed by the VBlank handler
   that publishes it. `mem` mirrors the Game Boy's VRAM, RAM, OAM and I/O
   (0x8000-0xFFFF), so mem[0] is address 0x8000; ROM reads go through `rom`.
   The ROM is immutable and shared, so a copy keeps pointing at the same image.

   The named fields overlay `mem` at their Game Boy addresses less 0x8000
   (checked below), so `s->scrollX` and `SML_MEM(s, 0xFFA4)` are the same
   byte. Names follow the upstream disassembly's labels where it has them;
   `mXXXX` arrays are the bytes the port does not name. Code that depends on
   the memory layout (an address kept in RAM, a fill that runs past its
   buffer) still goes through SML_MEM. */
typedef struct {
    union {
        uint8_t mem[SML_PROFILE_SIZE];
        struct {
            /* VRAM */
            uint8_t tiles[0x1800];
            uint8_t bgMap[0x400];
            uint8_t windowMap[0x400];
            uint8_t mA000[0x2000]; /* cartridge RAM, unused */

            /* Work RAM */
            SmlOamEntry oamBuffer[40]; /* C000, copied to OAM in VBlank */
            uint8_t score[3];          /* C0A0, BCD, least significant first */
            uint8_t livesDelta;        /* 01 earn a life, FF lose one */
            uint8_t gameOver;
            uint8_t gameOverWindowEnabled;
            uint8_t continues;
            uint8_t mC0A7[1];
            uint8_t continueWorldLevel;
            uint8_t superballTtl;
            uint8_t scrollAnchor;  /* hScrollX when the last column was decoded */
            uint8_t levelProgress; /* columns scrolled, for enemy spawns */
            uint8_t deathStep;
            uint8_t mC0AD[1];
            uint8_t superballCooldown;
            uint8_t mC0AF[1];
            uint8_t column[16]; /* C0B0, the next decoded column */
            uint8_t mC0C0[13];  /* a long column fill can run on into these */
            uint8_t blockItem;
            uint8_t coinBlockTimer;
            uint8_t mC0CF[3];
            uint8_t levelEnd;
            uint8_t invincibilityTimer; /* C0D3 */
            uint8_t mC0D4[9];
            uint8_t deathY;
            uint8_t scrollYEnabled;
            uint8_t scrollY;
            uint8_t mC0E0[2];
            uint8_t coinsDrawn;
            uint8_t mC0E3[0x11D];
            union {
                SmlEntity entities[5]; /* C200 */
                struct {
                    SmlEntity mario;
                    SmlEntity fragments[4];
                };
            };
            uint8_t mC250[0x3B0];
            uint8_t mC600[0x200];   /* C600: animated background tiles */
            uint8_t overlay[0x240]; /* C800: warp and item records per tile,
                                       at its tilemap address + 0x3000 */
            uint8_t mCA40[0x4C0];
            uint8_t stack[0x100]; /* CF00 */
            uint8_t spriteFlags;  /* D000 */
            uint8_t mD001[1];
            uint8_t currentCommand;
            uint8_t commandArgument;
            uint8_t mD004[3];
            uint8_t bossDefeated;
            uint8_t mD008[8];
            uint8_t spawnPtrLo; /* D010, next entry of the level's enemy list */
            uint8_t spawnPtrHi;
            uint8_t mD012[1];
            uint8_t objectsDrawn;
            uint8_t backgroundAnimated;
            uint8_t mD015[0xEB];
            SmlEnemy enemies[SML_ENEMY_SLOTS]; /* D100 */
            uint8_t mD1A0[0x860];
            uint8_t timerTicks;    /* DA00, updates until the next second */
            uint8_t timerDigits;   /* BCD tens and ones */
            uint8_t timerHundreds; /* BCD */
            uint8_t floatyTtl[4];
            uint8_t floatyCoinTile[4];
            uint8_t nextFloaty; /* OAM buffer offset of the next floaty */
            uint8_t floatyIsCoin[4];
            uint8_t floatyPhase[4];
            uint8_t mDA14[1];
            uint8_t lives; /* DA15, BCD */
            uint8_t mDA16[7];
            uint8_t timerExpiring; /* 1 below 100, 2 below 50, 3 at 0, FF time up */
            uint8_t mDA1E[0x4E2];
            /* Sound driver (DF00): requests and what each plays. */
            uint8_t mDF00[0x1F];
            uint8_t mDF1F[1];
            uint8_t mDF20[0x0F];
            uint8_t mDF2F[1];
            uint8_t mDF30[0x0F];
            uint8_t mDF3F[1];
            uint8_t mDF40[0x0F];
            uint8_t mDF4F[1];
            uint8_t mDF50[0x90];
            uint8_t sfx; /* DFE0 */
            uint8_t sfxPlaying;
            uint8_t mDFE2[6];
            uint8_t music; /* DFE8 */
            uint8_t musicPlaying;
            uint8_t mDFEA[6];
            uint8_t noise; /* DFF0 */
            uint8_t noisePlaying;
            uint8_t mDFF2[6];
            uint8_t effect; /* DFF8 */
            uint8_t effectPlaying;
            uint8_t mDFFA[6];
            uint8_t mE000[0x1E00]; /* echo RAM */

            SmlOamEntry oam[40]; /* FE00 */
            uint8_t mFEA0[0x60];

            /* I/O registers */
            uint8_t mFF00[6];
            uint8_t tma; /* FF06, the sound driver's timer rate */
            uint8_t mFF07[8];
            uint8_t interruptFlags;
            uint8_t mFF10[0x30];
            uint8_t lcdc; /* FF40 */
            uint8_t stat;
            uint8_t scy, scx;
            uint8_t ly, lyc;
            uint8_t dma;
            uint8_t bgp, obp0, obp1;
            uint8_t wy, wx;
            uint8_t mFF4C[0x34];

            /* High RAM */
            uint8_t joyHeld;    /* FF80: directions in bits 4-7, A and B in 0-1 */
            uint8_t joyPressed; /* newly pressed this update */
            uint8_t mFF82[3];
            uint8_t vblankOccurred;
            /* 3:4823 sprite programs: the entity being drawn (a copy of its
               first seven bytes) and the sprite being written. */
            uint8_t drawVisibility; /* FF86 */
            uint8_t drawY, drawX;
            uint8_t drawTile; /* the program index, then each sprite's tile */
            uint8_t drawPriority, drawFacing, drawPalette;
            uint8_t spriteSlotHi, spriteSlotLo; /* OAM buffer address to write */
            union {
                uint8_t spriteCount; /* FF8F: entities left to draw */
                uint8_t boxRight;    /* collisions: the right edge of the box */
            };
            uint8_t programY, programX; /* the program's offsets */
            uint8_t spriteX, spriteY;
            uint8_t spritePalette;
            uint8_t spriteHidden;
            uint8_t entityHi, entityLo; /* the entity being drawn */
            uint8_t mFF98[1];
            uint8_t superStatus; /* FF99: 0 small, 1 growing, 2 Super, 3-4 hurt */
            uint8_t winCount;    /* hard mode */
            uint8_t enemyHealth;
            uint8_t stompChainTimer;
            uint8_t stompChain;
            uint8_t stompReward;
            uint8_t demo;
            union {
                uint8_t boxTop;        /* FFA0: collisions */
                uint8_t blockContents; /* a hit block's overlay record */
            };
            uint8_t boxBottom, boxLeft;
            uint8_t columnToggle;
            uint8_t scrollX; /* FFA4 */
            uint8_t mFFA5[1];
            uint8_t timer, timer2; /* count down every update */
            uint8_t mFFA8[1];
            union {
                uint8_t projectiles[3]; /* FFA9 */
                uint8_t superball;      /* direction bits; 0 none */
            };
            uint8_t frameCounter;               /* FFAC */
            uint8_t lookupY, lookupX;           /* LookupTile's probe, object coordinates */
            uint8_t lookupAddrLo, lookupAddrHi; /* the tilemap address it found */
            uint8_t scoreDirty;
            uint8_t gamePaused;
            uint8_t gameState;  /* FFB3 */
            uint8_t worldLevel; /* world in bits 4-7, level in 0-3 */
            uint8_t superballMario;
            uint8_t dmaRoutine[10];
            SmlEnemy enemy; /* FFC0: the slot being run */
            uint8_t mFFD0[3];
            uint8_t mFFD3[1];
            uint8_t mFFD4[10];
            uint8_t pauseTuneTimer; /* FFDE */
            uint8_t pauseUnpauseMusic;
            uint8_t savedMapColumn;
            uint8_t savedRomBank;
            uint8_t textCursorHi, textCursorLo;
            uint8_t levelIndex; /* FFE4 */
            uint8_t screenIndex;
            uint8_t columnIndex;
            uint8_t columnPtrHi, columnPtrLo;
            uint8_t mapColumn;   /* tilemap column the next column is drawn to */
            uint8_t columnPhase; /* 1 decoded, 2 drawn, 3 settled */
            uint8_t floatyX, floatyY;
            uint8_t floatyControl; /* the score or coin to show next */
            uint8_t blockState;    /* FFEE */
            uint8_t blockAddrHi, blockAddrLo;
            uint8_t blockY, blockX;
            uint8_t fragmentScroll;
            union {
                uint8_t pipe[4]; /* FFF4: the warp record of the pipe entered */
                struct {
                    uint8_t pipeRoom; /* the room's screen; 0 none */
                    uint8_t pipeReturnScreen;
                    uint8_t pipeReturnX, pipeReturnY;
                };
            };
            uint8_t pipeTarget; /* FFF8 */
            uint8_t underground;
            uint8_t coins;        /* BCD */
            uint8_t tempB, tempC; /* spilled registers B and C */
            uint8_t activeRomBank;
            uint8_t coinPending;
            uint8_t interruptEnable; /* FFFF */
        };
    };
    const uint8_t *rom;
} SmlState;

/* The byte at Game Boy address `addr` (0x8000-0xFFFF), for addresses the ROM
   computes. The port never forms one below 0x8000, and the mask keeps one
   inside `mem` if it did; SML_CHECK_ADDRESSES (the self-test and fuzzer)
   aborts instead. */
static inline size_t smlMemIndex(uint16_t addr) {
#ifdef SML_CHECK_ADDRESSES
    if (addr < SML_MEM_BASE) {
        abort();
    }
#endif
    return addr & (SML_PROFILE_SIZE - 1);
}
#define SML_MEM(s, addr) ((s)->mem[smlMemIndex((uint16_t)(addr))])

#define SML_AT(field, address) \
    static_assert(offsetof(SmlState, field) == (address) - SML_MEM_BASE, #field " must be at " #address)

SML_AT(tiles, 0x8000);
SML_AT(bgMap, 0x9800);
SML_AT(windowMap, 0x9C00);
SML_AT(oamBuffer, 0xC000);
SML_AT(score, 0xC0A0);
SML_AT(livesDelta, 0xC0A3);
SML_AT(gameOver, 0xC0A4);
SML_AT(gameOverWindowEnabled, 0xC0A5);
SML_AT(continues, 0xC0A6);
SML_AT(continueWorldLevel, 0xC0A8);
SML_AT(superballTtl, 0xC0A9);
SML_AT(scrollAnchor, 0xC0AA);
SML_AT(levelProgress, 0xC0AB);
SML_AT(deathStep, 0xC0AC);
SML_AT(superballCooldown, 0xC0AE);
SML_AT(column, 0xC0B0);
SML_AT(mC0C0, 0xC0C0);
SML_AT(blockItem, 0xC0CD);
SML_AT(coinBlockTimer, 0xC0CE);
SML_AT(levelEnd, 0xC0D2);
SML_AT(invincibilityTimer, 0xC0D3);
SML_AT(deathY, 0xC0DD);
SML_AT(scrollYEnabled, 0xC0DE);
SML_AT(scrollY, 0xC0DF);
SML_AT(coinsDrawn, 0xC0E2);
SML_AT(entities, 0xC200);
SML_AT(mario, 0xC200);
SML_AT(fragments, 0xC210);
SML_AT(mC600, 0xC600);
SML_AT(overlay, 0xC800);
SML_AT(stack, 0xCF00);
SML_AT(spriteFlags, 0xD000);
SML_AT(currentCommand, 0xD002);
SML_AT(commandArgument, 0xD003);
SML_AT(bossDefeated, 0xD007);
SML_AT(spawnPtrLo, 0xD010);
SML_AT(spawnPtrHi, 0xD011);
SML_AT(objectsDrawn, 0xD013);
SML_AT(backgroundAnimated, 0xD014);
SML_AT(enemies, 0xD100);
SML_AT(timerTicks, 0xDA00);
SML_AT(timerDigits, 0xDA01);
SML_AT(timerHundreds, 0xDA02);
SML_AT(floatyTtl, 0xDA03);
SML_AT(floatyCoinTile, 0xDA07);
SML_AT(nextFloaty, 0xDA0B);
SML_AT(floatyIsCoin, 0xDA0C);
SML_AT(floatyPhase, 0xDA10);
SML_AT(lives, 0xDA15);
SML_AT(timerExpiring, 0xDA1D);
SML_AT(mDF1F, 0xDF1F);
SML_AT(mDF2F, 0xDF2F);
SML_AT(mDF3F, 0xDF3F);
SML_AT(mDF4F, 0xDF4F);
SML_AT(sfx, 0xDFE0);
SML_AT(sfxPlaying, 0xDFE1);
SML_AT(music, 0xDFE8);
SML_AT(musicPlaying, 0xDFE9);
SML_AT(noise, 0xDFF0);
SML_AT(noisePlaying, 0xDFF1);
SML_AT(effect, 0xDFF8);
SML_AT(effectPlaying, 0xDFF9);
SML_AT(oam, 0xFE00);
SML_AT(tma, 0xFF06);
SML_AT(interruptFlags, 0xFF0F);
SML_AT(lcdc, 0xFF40);
SML_AT(stat, 0xFF41);
SML_AT(scy, 0xFF42);
SML_AT(scx, 0xFF43);
SML_AT(ly, 0xFF44);
SML_AT(lyc, 0xFF45);
SML_AT(dma, 0xFF46);
SML_AT(bgp, 0xFF47);
SML_AT(obp0, 0xFF48);
SML_AT(obp1, 0xFF49);
SML_AT(wy, 0xFF4A);
SML_AT(wx, 0xFF4B);
SML_AT(joyHeld, 0xFF80);
SML_AT(joyPressed, 0xFF81);
SML_AT(vblankOccurred, 0xFF85);
SML_AT(drawVisibility, 0xFF86);
SML_AT(drawY, 0xFF87);
SML_AT(drawX, 0xFF88);
SML_AT(drawTile, 0xFF89);
SML_AT(drawPriority, 0xFF8A);
SML_AT(drawFacing, 0xFF8B);
SML_AT(drawPalette, 0xFF8C);
SML_AT(spriteSlotHi, 0xFF8D);
SML_AT(spriteSlotLo, 0xFF8E);
SML_AT(spriteCount, 0xFF8F);
SML_AT(boxRight, 0xFF8F);
SML_AT(programY, 0xFF90);
SML_AT(programX, 0xFF91);
SML_AT(spriteX, 0xFF92);
SML_AT(spriteY, 0xFF93);
SML_AT(spritePalette, 0xFF94);
SML_AT(spriteHidden, 0xFF95);
SML_AT(entityHi, 0xFF96);
SML_AT(entityLo, 0xFF97);
SML_AT(superStatus, 0xFF99);
SML_AT(winCount, 0xFF9A);
SML_AT(enemyHealth, 0xFF9B);
SML_AT(stompChainTimer, 0xFF9C);
SML_AT(stompChain, 0xFF9D);
SML_AT(stompReward, 0xFF9E);
SML_AT(demo, 0xFF9F);
SML_AT(boxTop, 0xFFA0);
SML_AT(blockContents, 0xFFA0);
SML_AT(boxBottom, 0xFFA1);
SML_AT(boxLeft, 0xFFA2);
SML_AT(columnToggle, 0xFFA3);
SML_AT(scrollX, 0xFFA4);
SML_AT(timer, 0xFFA6);
SML_AT(timer2, 0xFFA7);
SML_AT(projectiles, 0xFFA9);
SML_AT(superball, 0xFFA9);
SML_AT(frameCounter, 0xFFAC);
SML_AT(lookupY, 0xFFAD);
SML_AT(lookupX, 0xFFAE);
SML_AT(lookupAddrLo, 0xFFAF);
SML_AT(lookupAddrHi, 0xFFB0);
SML_AT(scoreDirty, 0xFFB1);
SML_AT(gamePaused, 0xFFB2);
SML_AT(gameState, 0xFFB3);
SML_AT(worldLevel, 0xFFB4);
SML_AT(superballMario, 0xFFB5);
SML_AT(dmaRoutine, 0xFFB6);
SML_AT(enemy, 0xFFC0);
SML_AT(mFFD3, 0xFFD3);
SML_AT(pauseTuneTimer, 0xFFDE);
SML_AT(pauseUnpauseMusic, 0xFFDF);
SML_AT(savedMapColumn, 0xFFE0);
SML_AT(savedRomBank, 0xFFE1);
SML_AT(textCursorHi, 0xFFE2);
SML_AT(levelIndex, 0xFFE4);
SML_AT(screenIndex, 0xFFE5);
SML_AT(columnIndex, 0xFFE6);
SML_AT(columnPtrHi, 0xFFE7);
SML_AT(columnPtrLo, 0xFFE8);
SML_AT(mapColumn, 0xFFE9);
SML_AT(columnPhase, 0xFFEA);
SML_AT(floatyX, 0xFFEB);
SML_AT(floatyY, 0xFFEC);
SML_AT(floatyControl, 0xFFED);
SML_AT(blockState, 0xFFEE);
SML_AT(blockAddrHi, 0xFFEF);
SML_AT(blockAddrLo, 0xFFF0);
SML_AT(blockY, 0xFFF1);
SML_AT(blockX, 0xFFF2);
SML_AT(fragmentScroll, 0xFFF3);
SML_AT(pipe, 0xFFF4);
SML_AT(pipeRoom, 0xFFF4);
SML_AT(pipeReturnScreen, 0xFFF5);
SML_AT(pipeReturnX, 0xFFF6);
SML_AT(pipeReturnY, 0xFFF7);
SML_AT(pipeTarget, 0xFFF8);
SML_AT(underground, 0xFFF9);
SML_AT(coins, 0xFFFA);
SML_AT(tempB, 0xFFFB);
SML_AT(tempC, 0xFFFC);
SML_AT(activeRomBank, 0xFFFD);
SML_AT(coinPending, 0xFFFE);
SML_AT(interruptEnable, 0xFFFF);

static_assert(offsetof(SmlState, rom) == SML_PROFILE_SIZE, "mem ends at 0xFFFF");
static_assert(sizeof(SmlOamEntry) == 4, "an OAM entry is 4 bytes");
static_assert(sizeof(SmlEntity) == 16 && sizeof(SmlEnemy) == 16, "entity and enemy slots are 16 bytes");
static_assert(SML_ENEMY_COPY == 13, "an enemy slot copies 13 bytes to FFC0");
#endif
