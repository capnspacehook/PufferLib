#ifndef SML_CORE_H
#define SML_CORE_H
#include "sml_state.h"

/* A routine-by-routine C port of Super Mario Land (World) Rev A gameplay.
   Function names carry the ROM address of the routine they port; comments
   name the upstream disassembly label where it exists. One smlStepFrame is
   one main-loop pass (bank 0 $0226) plus the VBlank handler ($0060), ending
   where the ROM reaches $029C. Sound-driver state (DF00-DFFF) is written as
   the ROM does but is otherwise outside the port. RAM is read and written
   through SmlState's named fields (sml_state.h); SML_MEM is used only where
   the ROM computes an address. */

/* ROM routine and table addresses used as data. */
enum {
    R_DATA_211D = 0x211D, /* Mario and fragment entity initial values */
    R_JUMP_TABLE = 0x216D,
    R_ENEMY_SPRITES_LEFT = 0x30B4,
    R_ENEMY_SPRITES_RIGHT = 0x2FE2,
    R_ENEMY_TRANSFORMS = 0x3186,
    R_ENEMY_PROPERTIES = 0x3375,
    R_ENEMY_SCRIPTS = 0x349E,
    R_LEVEL_POINTERS = 0x4000,
    R_LEVEL_ENEMY_POINTERS = 0x401A,
};

enum {
    TILE_BLANK = 0x2C,
    TILE_HIDDEN_BLOCK = 0x5F,
    TILE_SOLID = 0x60,
    TILE_PIPE = 0x70,
    TILE_SIDE_PIPE = 0x77,
    TILE_USED_BLOCK = 0x7F,
    TILE_BREAKABLE = 0x80,
    TILE_MYSTERY = 0x81,
    TILE_BRICK = 0x82,
    TILE_BOSS_SWITCH = 0xE1,
    TILE_SPIKE = 0xED,
    TILE_FIST = 0xF2,
    TILE_COIN = 0xF4,
};

#include <stdio.h>
#include <string.h>

typedef uint8_t u8;
typedef uint16_t u16;

/* The Game Boy address of a named field, for code that keeps addresses in
   RAM as the ROM does. */
#define SML_ADDR(field) ((u16)(offsetof(SmlState, field) + SML_MEM_BASE))

int smlLoadRom(const char *path, uint8_t *rom) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    size_t n = fread(rom, 1, SML_ROM_SIZE, f);
    int extra = fgetc(f);
    fclose(f);
    return n == SML_ROM_SIZE && extra == EOF ? 0 : -2;
}

/* A profile is the 0x8000-0xFFFF image the ROM holds at an update boundary,
   which is all of `mem`. */
void smlInitProfileBytes(SmlState *s, const uint8_t *rom, const uint8_t *profile) {
    memset(s, 0, sizeof(*s));
    s->rom = rom;
    memcpy(s->mem, profile, SML_PROFILE_SIZE);
}

int smlInitProfile(const char *path, uint8_t *profile) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    size_t n = fread(profile, 1, SML_PROFILE_SIZE, f);
    int extra = fgetc(f);
    fclose(f);
    if (n != SML_PROFILE_SIZE || extra != EOF) {
        return -2;
    }
    return 0;
}

void smlSaveProfile(const SmlState *s, uint8_t profile[SML_PROFILE_SIZE]) {
    memcpy(profile, s->mem, SML_PROFILE_SIZE);
}

void smlCopyState(SmlState *dst, const SmlState *src) { *dst = *src; }

/* ------------------------------------------------------------------------
   CPU-level helpers. Reads through the address space follow the MBC bank the
   ROM keeps mirrored in hActiveRomBank. */

static u8 smlRead(const SmlState *s, u16 addr) {
    if (addr < 0x4000) {
        return s->rom[addr];
    }
    if (addr < 0x8000) {
        unsigned bank = s->activeRomBank & 3;
        return s->rom[bank * 0x4000 + (addr - 0x4000)];
    }
    return SML_MEM(s, addr);
}

static u16 smlRead16(const SmlState *s, u16 addr) {
    return (u16)(smlRead(s, addr) | smlRead(s, (u16)(addr + 1)) << 8);
}

/* SAVE_AND_SWITCH_ROM_BANK / RESTORE_ROM_BANK. */
static void smlSwitchBank(SmlState *s, u8 bank) {
    s->savedRomBank = s->activeRomBank;
    s->activeRomBank = bank;
}

static void smlRestoreBank(SmlState *s) { s->activeRomBank = s->savedRomBank; }

/* A write to DFE8, the driver's music request mailbox. */
static void smlRequestMusic(SmlState *s, uint8_t song) { s->music = song; }

static u8 smlSwap(u8 a) { return (u8)(a << 4 | a >> 4); }
static u8 smlRlca(u8 a) { return (u8)(a << 1 | a >> 7); }

/* `add`/`adc` followed by `daa`. */
static u8 smlAddBcd(u8 a, u8 b, int carryIn, int *carryOut) {
    unsigned sum = (unsigned)a + b + (unsigned)carryIn;
    int half = ((a & 15) + (b & 15) + carryIn) > 15;
    int carry = sum > 0xFF;
    u8 r = (u8)sum;
    u8 correction = 0;
    if (half || (r & 15) > 9) {
        correction |= 0x06;
    }
    if (carry || r > 0x99) {
        correction |= 0x60;
        carry = 1;
    }
    *carryOut = carry;
    return (u8)(r + correction);
}

/* `sub 1` followed by `daa`. */
static u8 smlDecrementBcd(u8 a) {
    int half = (a & 15) < 1;
    int carry = a < 1;
    u8 r = (u8)(a - 1);
    if (half) {
        r = (u8)(r - 6);
    }
    if (carry) {
        r = (u8)(r - 0x60);
    }
    return r;
}

/* ------------------------------------------------------------------------
   Bank 0 utilities. */

/* LookupTile ($0153, $3EE6): tile under (FFAD, FFAE) in object coordinates.
   Records the tilemap address in FFB0:FFAF and returns it. */
static u16 smlLookupTile(SmlState *s, u8 *tile) {
    u8 row = (u8)((u8)(s->lookupY - 0x10) >> 3);
    u8 col = (u8)((u8)(s->lookupX - 0x08) >> 3);
    u16 addr = (u16)(0x9800 + row * 32 + col);
    s->lookupAddrHi = (u8)(addr >> 8);
    s->lookupAddrLo = (u8)addr;
    *tile = SML_MEM(s, addr);
    return addr;
}

/* AddScore ($0166): add BCD DE to the six-digit score, saturating. */
static void smlAddScore(SmlState *s, u16 de) {
    if (s->demo) {
        return;
    }
    int carry;
    s->score[0] = smlAddBcd((u8)de, s->score[0], 0, &carry);
    s->score[1] = smlAddBcd((u8)(de >> 8), s->score[1], carry, &carry);
    s->score[2] = smlAddBcd(0, s->score[2], carry, &carry);
    s->scoreDirty = 1;
    if (carry) {
        s->score[0] = s->score[1] = s->score[2] = 0x99;
    }
}

/* Call_3F13: object coordinates of the tile at FFB0:FFAF into FFAD/FFAE. */
static void smlTileToObject(SmlState *s) {
    u16 addr = (u16)(s->lookupAddrHi << 8 | s->lookupAddrLo);
    u8 e = (u8)(addr >> 4);
    s->lookupY = (u8)(smlRlca(smlRlca((u8)((u8)(e - 0x84) & 0xFE))) + 8);
    s->lookupX = (u8)(((s->lookupAddrLo & 0x1F) << 3) + 8);
}

/* DisplayScore ($3F39). */
static void smlPrintScore(SmlState *s, u16 src, u16 dst) {
    s->scoreDirty = 0;
    for (int c = 3; c > 0; c--, src--) {
        u8 pair = SML_MEM(s, src);
        u8 digit = pair >> 4;
        if (digit) {
            s->scoreDirty = 1;
        }
        SML_MEM(s, dst++) = digit || s->scoreDirty ? digit : TILE_BLANK;
        digit = pair & 15;
        if (digit) {
            s->scoreDirty = 1;
        }
        SML_MEM(s, dst++) = digit || s->scoreDirty || c == 1 ? digit : TILE_BLANK;
    }
    s->scoreDirty = 0;
}

static void smlDisplayScore(SmlState *s) {
    if (!s->scoreDirty || s->coinsDrawn || s->columnPhase == 2) {
        return;
    }
    smlPrintScore(s, SML_ADDR(score) + 2, 0x9820);
}

/* DisplayTimer ($3D6A). */
static void smlPrintTimer(SmlState *s) {
    SML_MEM(s, 0x9833) = s->timerDigits & 15;
    SML_MEM(s, 0x9832) = s->timerDigits >> 4;
    SML_MEM(s, 0x9831) = s->timerHundreds & 15;
}

static void smlDisplayTimer(SmlState *s) {
    if (s->gameOver || s->gameState >= 0x12 || s->timerTicks != 0x28) {
        return;
    }
    smlPrintTimer(s);
}

/* AddCoin ($1BFF) and DisplayCoins ($1C1B). */
static void smlDisplayCoins(SmlState *s) {
    SML_MEM(s, 0x982A) = s->coins & 15;
    SML_MEM(s, 0x9829) = s->coins >> 4;
    s->coinPending = 0;
    s->coinsDrawn = 1;
}

static void smlAddCoin(SmlState *s) {
    if (s->demo) {
        return;
    }
    smlAddScore(s, 0x0100);
    int carry;
    s->coins = smlAddBcd(s->coins, 1, 0, &carry);
    if (!s->coins) {
        s->livesDelta = 1;
    }
    smlDisplayCoins(s);
}

/* UpdateLives ($1C33). */
static void smlUpdateLives(SmlState *s) {
    if (s->demo || !s->livesDelta) {
        return;
    }
    u8 lives = s->lives;
    int carry;
    if (s->livesDelta == 0xFF) {
        if (!lives) {
            s->gameState = 0x39;
            s->gameOver = 0x39;
            s->livesDelta = 0;
            return;
        }
        s->lives = smlDecrementBcd(lives);
    } else if (lives != 0x99) {
        s->sfx = 8;
        s->mFFD3[0] = 8;
        s->lives = smlAddBcd(lives, 1, 0, &carry);
    } else {
        s->livesDelta = 0;
        return;
    }
    SML_MEM(s, 0x9807) = s->lives & 15;
    SML_MEM(s, 0x9806) = s->lives >> 4;
    s->livesDelta = 0;
}

/* AnimateBackground ($2401). */
static void smlAnimateBackground(SmlState *s) {
    if (!s->backgroundAnimated || s->gameState >= 0x0D || (s->frameCounter & 7)) {
        return;
    }
    u16 src;
    if (s->frameCounter & 8) {
        src = 0xC600;
    } else {
        u8 a = (u8)((s->worldLevel & 0xF0) - 0x10);
        src = (u16)(0x3FC4 + (u8)(a >> 1 | a << 7));
    }
    for (int i = 0; i < 8; i++) {
        SML_MEM(s, 0x95D1 + i * 2) = smlRead(s, (u16)(src + i));
    }
}

/* ------------------------------------------------------------------------
   Level streaming. */

/* CheckPipeForWarp ($22A9). */
static void smlCheckPipeForWarp(SmlState *s) {
    if (s->underground) {
        return;
    }
    smlSwitchBank(s, 3);
    u16 hl = smlRead16(s, (u16)(0x651C + (u8)(s->levelIndex * 2)));
    for (;;) {
        u8 screen = smlRead(s, hl);
        if (s->screenIndex == screen) {
            if (s->columnIndex == smlRead(s, (u16)(hl + 1))) {
                for (int i = 0; i < 4; i++) {
                    s->pipe[i] = smlRead(s, (u16)(hl + 2 + i));
                }
                break;
            }
        } else if (screen == 0xFF) {
            break;
        }
        hl = (u16)(hl + 6);
    }
    smlRestoreBank(s);
}

/* CheckBlockForItem ($2321). */
static void smlCheckBlockForItem(SmlState *s) {
    smlSwitchBank(s, 3);
    u16 hl = smlRead16(s, (u16)(0x6536 + (u8)(s->levelIndex * 2)));
    for (;;) {
        u8 screen = smlRead(s, hl);
        if (s->screenIndex == screen) {
            if (s->columnIndex == smlRead(s, (u16)(hl + 1))) {
                s->blockItem = smlRead(s, (u16)(hl + 2));
                break;
            }
        } else if (screen == 0xFF) {
            break;
        }
        hl = (u16)(hl + 3);
    }
    smlRestoreBank(s);
}

/* LoadNextColumn ($21B1): decode the next column into C0B0. The cache index
   wraps within page C0, so long fills can overrun into C0C0-C0FF. */
static void smlLoadNextColumn(SmlState *s) {
    memset(s->column, TILE_BLANK, sizeof(s->column));
    u16 hl;
    if (s->columnIndex) {
        hl = (u16)(s->columnPtrHi << 8 | s->columnPtrLo);
    } else {
        hl = smlRead16(s, (u16)(R_LEVEL_POINTERS + (u8)(s->levelIndex * 2)));
        hl = (u16)(hl + (u8)(s->screenIndex * 2));
        u8 lo = smlRead(s, hl++);
        if (lo == 0xFF) {
            s->levelEnd++;
            return;
        }
        hl = (u16)(lo | smlRead(s, hl) << 8);
    }
    for (;;) {
        u8 command = smlRead(s, hl++);
        if (command == 0xFE) {
            break;
        }
        u8 e = (u8)(0xB0 + (command >> 4));
        u8 count = command & 15 ? command & 15 : 16;
        while (count) {
            u8 tile = smlRead(s, hl++);
            if (tile == 0xFD) {
                tile = smlRead(s, hl);
                while (count--) {
                    SML_MEM(s, 0xC000 | e++) = tile;
                }
                hl++;
                break;
            }
            SML_MEM(s, 0xC000 | e) = tile;
            if (tile == TILE_PIPE) {
                smlCheckPipeForWarp(s);
            } else if (tile == TILE_BREAKABLE || tile == TILE_HIDDEN_BLOCK || tile == TILE_MYSTERY) {
                smlCheckBlockForItem(s);
            }
            e++;
            count--;
        }
    }
    s->columnPtrHi = (u8)(hl >> 8);
    s->columnPtrLo = (u8)hl;
    u8 index = (u8)(s->columnIndex + 1);
    if (index == 0x14) {
        s->screenIndex++;
        index = 0;
    }
    s->columnIndex = index;
    s->scrollAnchor = s->scrollX;
    s->columnPhase = 1;
}

/* Call_2198: start a column decode every 8 scrolled pixels. */
static void smlStreamColumns(SmlState *s) {
    if (s->columnPhase) {
        /* Jump_2188 */
        s->columnPhase = 3;
        if (s->scrollAnchor != s->scrollX) {
            s->columnPhase = 0;
        }
        return;
    }
    if ((s->scrollX & 8) != s->columnToggle) {
        return;
    }
    s->columnToggle ^= 8;
    if (!s->columnToggle) {
        s->levelProgress++;
    }
    smlLoadNextColumn(s);
}

/* Call_22FD and Call_2363 record warp and block contents in the overlay. */
static void smlOverlayPipe(SmlState *s, u16 hl) {
    if (!s->pipeRoom) {
        return;
    }
    u16 overlay = (u16)(hl + 0x3000);
    SML_MEM(s, overlay) = s->pipeRoom;
    SML_MEM(s, (u16)(overlay - 0x20)) = s->pipeReturnScreen;
    SML_MEM(s, (u16)(overlay - 0x40)) = s->pipeReturnX;
    SML_MEM(s, (u16)(overlay - 0x60)) = s->pipeReturnY;
    s->pipeRoom = s->pipeReturnScreen = 0;
}

static void smlOverlayBlock(SmlState *s, u16 hl) {
    if (!s->blockItem) {
        return;
    }
    SML_MEM(s, (u16)(hl + 0x3000)) = s->blockItem;
    s->blockItem = 0;
}

/* DrawColumn ($2258). */
static void smlDrawColumn(SmlState *s) {
    if (s->columnPhase != 1) {
        return;
    }
    u8 l = s->mapColumn;
    s->mapColumn = (u8)(l + 1) == 0x60 ? 0x40 : (u8)(l + 1);
    u16 hl = (u16)(0x9800 | l);
    for (int row = 0; row < 16; row++) {
        SML_MEM(s, (u16)(hl + 0x3000)) = 0;
        u8 tile = s->column[row];
        SML_MEM(s, hl) = tile;
        if (tile == TILE_PIPE) {
            smlOverlayPipe(s, hl);
        } else if (tile == TILE_BREAKABLE || tile == TILE_HIDDEN_BLOCK || tile == TILE_MYSTERY) {
            smlOverlayBlock(s, hl);
        }
        hl = (u16)(hl + 0x20);
    }
    s->columnPhase = 2;
}

/* ------------------------------------------------------------------------
   Sprites and objects. */

/* Call_1ED4: clear projectiles, floaties and block fragments. */
static void smlClearObjects(SmlState *s) {
    memset(&s->oamBuffer[7], 0, 13 * sizeof(SmlOamEntry));
    /* Projectiles 0 and 1, and the first three bytes of 2. */
    memset(s->oamBuffer, 0, 11);
    memset(s->projectiles, 0, sizeof(s->projectiles));
    for (int i = 0; i < 4; i++) {
        s->fragments[i].visibility = 0x80;
    }
}

/* 3:4823: draw entities at HL through their sprite programs (table 3:4C37).
   The routine keeps the entity's address and the next OAM buffer slot in
   HRAM, so it walks memory by address. */
static void smlDrawEntities(SmlState *s, u16 hl) {
    for (;;) {
        s->entityHi = (u8)(hl >> 8);
        s->entityLo = (u8)hl;
        u8 visible = SML_MEM(s, hl);
        if (visible == 0 || visible == 0x80) {
            if (visible == 0x80) {
                s->spriteHidden = 0x80;
            }
            s->drawVisibility = visible;
            s->drawY = SML_MEM(s, (u16)(hl + 1));
            s->drawX = SML_MEM(s, (u16)(hl + 2));
            s->drawTile = SML_MEM(s, (u16)(hl + 3));
            s->drawPriority = SML_MEM(s, (u16)(hl + 4));
            s->drawFacing = SML_MEM(s, (u16)(hl + 5));
            s->drawPalette = SML_MEM(s, (u16)(hl + 6));
            u16 entry = smlRead16(s, (u16)(0x4C37 + smlRlca(s->drawTile)));
            u16 list = smlRead16(s, entry);
            s->programY = smlRead(s, (u16)(entry + 2));
            s->programX = smlRead(s, (u16)(entry + 3));
            u16 de = smlRead16(s, list);
            u16 p = (u16)(list + 1);
            for (;;) {
                p++;
                s->spritePalette = s->drawPalette;
                u8 a = smlRead(s, p);
                if (a == 0xFF) {
                    s->spriteHidden = 0;
                    break;
                }
                if (a == 0xFD) {
                    /* The flipped palette is overwritten when the loop
                       restarts at .jmp_4872, so FD only skips a byte. */
                    s->spritePalette = (u8)(s->drawPalette ^ 0x10);
                    continue;
                }
                if (a == 0xFE) {
                    de = (u16)(de + 2);
                    continue;
                }
                s->drawTile = a;
                u8 b = s->drawY, c = smlRead(s, de);
                unsigned t;
                u8 y;
                if (!(s->drawFacing & 0x40)) {
                    t = (unsigned)s->programY + b;
                    y = (u8)((u8)t + c + (t > 0xFF));
                } else {
                    int borrow = b < s->programY;
                    u8 r = (u8)(b - s->programY);
                    int borrow2 = r < c + borrow;
                    r = (u8)(r - c - borrow);
                    y = (u8)(r - 8 - borrow2);
                }
                s->spriteY = y;
                b = s->drawX;
                de++;
                c = smlRead(s, de);
                de++;
                u8 x;
                if (!(s->drawFacing & 0x20)) {
                    t = (unsigned)s->programX + b;
                    x = (u8)((u8)t + c + (t > 0xFF));
                } else {
                    int borrow = b < s->programX;
                    u8 r = (u8)(b - s->programX);
                    int borrow2 = r < c + borrow;
                    r = (u8)(r - c - borrow);
                    x = (u8)(r - 8 - borrow2);
                }
                s->spriteX = x;
                u16 out = (u16)(s->spriteSlotHi << 8 | s->spriteSlotLo);
                SML_MEM(s, out) = s->spriteHidden ? 0xFF : s->spriteY;
                SML_MEM(s, (u16)(out + 1)) = s->spriteX;
                SML_MEM(s, (u16)(out + 2)) = s->drawTile;
                SML_MEM(s, (u16)(out + 3)) =
                    (u8)(s->spritePalette | s->drawFacing | s->drawPriority);
                out = (u16)(out + 4);
                s->spriteSlotHi = (u8)(out >> 8);
                s->spriteSlotLo = (u8)out;
            }
        }
        hl = (u16)((s->entityHi << 8 | s->entityLo) + 0x10);
        if (--s->spriteCount == 0) {
            return;
        }
    }
}

/* Call_1736: draw Mario and the four fragment entities into OAM C00C. */
static void smlDrawMario(SmlState *s) {
    s->spriteSlotLo = 0x0C;
    s->spriteSlotHi = 0xC0;
    s->spriteCount = 5;
    smlSwitchBank(s, 3);
    smlDrawEntities(s, SML_ADDR(mario));
    smlRestoreBank(s);
}

/* ------------------------------------------------------------------------
   Enemies. */

static u16 smlEnemyProperties(u8 id) {
    return (u16)(R_ENEMY_PROPERTIES + (u16)(smlRlca(id) + id));
}

/* InitEnemy ($2CBB): E is an enemy slot or the FFC0 buffer. */
static void smlInitEnemy(SmlState *s, SmlEnemy *e) {
    u16 p = smlEnemyProperties(e->id);
    e->scriptIndex = 0;
    e->flags = smlRead(s, p);
    e->moveCount = 0;
    e->moveDelay = 0;
    e->size = smlRead(s, (u16)(p + 1));
    e->health = smlRead(s, (u16)(p + 2));
}

static void smlSlotToBuffer(SmlState *s, const SmlEnemy *e) {
    memcpy(&s->enemy, e, SML_ENEMY_COPY);
}
static void smlBufferToSlot(SmlState *s, SmlEnemy *e) {
    memcpy(e, &s->enemy, SML_ENEMY_COPY);
}

/* Call_2A01, 2A23, 2A44, 2B06: transform an enemy through table 3186. */
static u8 smlTransformEnemy(SmlState *s, SmlEnemy *e, int column) {
    u8 next = smlRead(s, (u16)(R_ENEMY_TRANSFORMS + e->id * 5 + column));
    if (!next) {
        return 0;
    }
    e->id = next;
    smlInitEnemy(s, e);
    return 0xFF;
}

/* Call_2A44: side contact. FF keeps the enemy and hurts Mario. */
static u8 smlEnemyContact(SmlState *s, SmlEnemy *e) {
    u8 next = smlRead(s, (u16)(R_ENEMY_TRANSFORMS + e->id * 5 + 2));
    if (next == 0xFF || next == 0) {
        return next;
    }
    e->id = next;
    smlInitEnemy(s, e);
    return 0;
}

/* Call_2A68: superball hit. */
static u8 smlSuperballHit(SmlState *s, SmlEnemy *e) {
    if (e->health & 0x3F) {
        e->health--;
        if (e->id == 0x32 || e->id == 0x08) {
            s->noise = 1;
        }
        return 0xFE;
    }
    return smlTransformEnemy(s, e, 3);
}

/* Call_A10: score reward from the top bits of the enemy health byte. */
static void smlStompReward(SmlState *s) {
    static const u8 rewards[4] = {0x01, 0x04, 0x08, 0x50};
    s->stompReward = rewards[s->enemyHealth >> 6];
}

/* Hitbox edges from an enemy's size byte (D1xA): height in tiles in bits 0-3,
   width in bits 4-6. The ROM counts tiles down in loops that wrap for size
   00 (corpse 0D) and cost a frame's worth of time with a few corpses; these
   are the loops' exact closed forms, as in the oracle's rom_patch.py. */
static u8 smlHitboxTop(u8 y, u8 size) { return (u8)(y + 8 - 8 * (size & 15)); }
static u8 smlHitboxRight(u8 x, u8 size) {
    return (u8)(x + ((size & 0x70) >> 1));
}

/* Call_AAF: collision between the enemy's box and boxTop, boxBottom,
   boxLeft and boxRight. */
static int smlBoxCollision(const SmlState *s, const SmlEnemy *e) {
    if (s->boxTop >= (u8)(e->y + 8)) {
        return 0;
    }
    if (s->boxBottom < smlHitboxTop(e->y, e->size)) {
        return 0;
    }
    if (s->boxRight < e->x) {
        return 0;
    }
    return s->boxLeft < smlHitboxRight(e->x, e->size);
}

/* Tile probes used by enemy movement (Call_2B84 and relatives). They return
   nonzero when the tile is passable: below 5F or F0 and above. */
static int smlEnemyProbe(SmlState *s, u8 dx, u8 dy) {
    s->lookupX = (u8)(s->scrollX + s->enemy.x + dx);
    s->lookupY = (u8)(s->enemy.y + dy);
    u8 tile;
    smlLookupTile(s, &tile);
    return tile < 0x5F || tile >= 0xF0;
}

static u8 smlEnemyWidthPixels(const SmlState *s) {
    u8 a = s->enemy.size & 0x70;
    return (u8)(a >> 1 | a << 7);
}

static u8 smlEnemyHeightOffset(const SmlState *s) {
    u8 a = smlSwap((u8)((s->enemy.size & 7) - 1));
    return (u8)(a >> 1 | a << 7);
}

/* Call_2C9F: scroll every enemy by B. */
static void smlScrollEnemies(SmlState *s, u8 b) {
    if (!b) {
        return;
    }
    s->enemy.x = (u8)(s->enemy.x - b);
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        s->enemies[i].x = (u8)(s->enemies[i].x - b);
    }
}

static int smlSideCollision(SmlState *s);
static void smlWin(SmlState *s);
static void smlExplodeAllEnemies(SmlState *s);

/* Jmp_250B: finish initializing FFC0 for enemy ID FFC0 and place it. */
static void smlPlaceEnemyBuffer(SmlState *s) {
    s->enemy.scriptIndex = s->enemy.direction = s->enemy.moveCount =
        s->enemy.moveDelay = s->enemy.carrying = 0;
    u16 p = smlEnemyProperties(s->enemy.id);
    s->enemy.size = smlRead(s, (u16)(p + 1));
    s->enemy.health = smlRead(s, (u16)(p + 2));
    if (s->enemy.health >= 0xC0) {
        smlRequestMusic(s, 0x0B);
    }
    for (int i = 0; i < SML_ENEMY_SLOTS - 1; i++) { /* slot 9 is for powerups */
        if (s->enemies[i].id == 0xFF) {
            smlBufferToSlot(s, &s->enemies[i]);
            return;
        }
    }
}

/* Call_24D6: enemy fires a projectile with ID wCommandArgument. */
static void smlLaunchProjectile(SmlState *s) {
    s->enemy.id = s->commandArgument;
    if (s->enemy.id == 0xFF) {
        return;
    }
    s->enemy.flags = smlRead(s, smlEnemyProperties(s->enemy.id));
    smlPlaceEnemyBuffer(s);
}

/* SpawnEnemies ($249B). */
static void smlSpawnEnemies(SmlState *s) {
    for (;;) {
        u16 hl = (u16)(s->spawnPtrHi << 8 | s->spawnPtrLo);
        u8 at = smlRead(s, hl);
        u8 ahead = (u8)(s->levelProgress - at);
        if (!ahead || s->levelProgress < at) {
            return;
        }
        u8 c = smlSwap(ahead);
        u8 yx = smlRead(s, (u16)(hl + 1));
        s->enemy.y = (u8)(((yx & 0x1F) << 3) + 0x10);
        s->enemy.x = (u8)((u8)(smlSwap(yx & 0xC0) + 0xD0) - c);
        /* Call_24EF */
        u8 type = smlRead(s, (u16)(hl + 2));
        if (s->winCount || !(type & 0x80)) {
            s->enemy.id = type & 0x7F;
            s->enemy.flags = smlRead(s, smlEnemyProperties(s->enemy.id));
            smlPlaceEnemyBuffer(s);
        }
        hl = (u16)(hl + 3);
        s->spawnPtrLo = (u8)hl;
        s->spawnPtrHi = (u8)(hl >> 8);
    }
}

/* Call_254D: spawn a powerup in slot 9 at the bounced block. */
static void smlSpawnPowerup(SmlState *s, u8 id) {
    SmlEnemy *e = &s->enemies[9];
    e->id = id;
    e->y = (u8)((s->enemy.y & 0xF8) + 7);
    e->x = s->enemy.x;
    smlInitEnemy(s, e);
    s->sfx = 0x0B;
}

/* Mario riding a moving enemy (FFCB) is pushed with it. */
static void smlCarryMarioX(SmlState *s, u8 b, int right) {
    u8 facing = s->mario.facing;
    s->mario.facing = right ? 0 : 0x20;
    if (!smlSideCollision(s)) {
        if (!right) {
            s->mario.x = (u8)(s->mario.x - b);
            if (s->mario.x < 0x0F) {
                s->mario.x = 0x0F;
            }
        } else {
            s->mario.x = (u8)(s->mario.x + b);
            if (s->mario.x >= 0x51) {
                int scroll = s->levelEnd < 7;
                if (!scroll) {
                    scroll = (s->scrollX & 0x0C) != 0;
                    if (!scroll) {
                        s->scrollX &= 0xFC;
                    }
                }
                if (scroll) {
                    u8 over = (u8)(s->mario.x - 0x50);
                    s->mario.x = 0x50;
                    s->scrollX = (u8)(s->scrollX + over);
                    smlScrollEnemies(s, over);
                }
            }
        }
    }
    s->mario.facing = facing;
}

/* Call_2648 .moveEnemy */
static void smlMoveEnemy(SmlState *s) {
    u8 speed = s->enemy.motion;
    if (speed & 0x0F) {
        u8 flags = s->enemy.flags;
        if (!(s->enemy.direction & 1)) {
            int moveLeft = 1;
            if (!smlEnemyProbe(s, 0, 0)) {
                u8 mode = flags & 0x0C;
                if (mode == 4) {
                    s->enemy.direction |= 1;
                    moveLeft = 0;
                } else if (mode == 0x0C) {
                    s->enemy.scriptIndex = s->enemy.moveCount = 0;
                    moveLeft = 0;
                } else if (mode != 0) {
                    moveLeft = 0;
                }
            } else if ((flags & 1) && smlEnemyProbe(s, 3, 8)) {
                s->enemy.direction |= 1;
                moveLeft = 0;
            }
            if (moveLeft) {
                u8 b = speed & 0x0F;
                s->enemy.x = (u8)(s->enemy.x - b);
                if (s->enemy.carrying) {
                    smlCarryMarioX(s, b, 0);
                }
            }
        } else {
            int moveRight = 1;
            u8 dx = (u8)(8 + smlEnemyWidthPixels(s) - 8);
            if (!smlEnemyProbe(s, dx, 0)) {
                u8 mode = flags & 0x0C;
                if (mode == 4) {
                    s->enemy.direction &= 0xFE;
                    moveRight = 0;
                } else if (mode == 0x0C) {
                    s->enemy.scriptIndex = s->enemy.moveCount = 0;
                    moveRight = 0;
                } else if (mode != 0) {
                    moveRight = 0;
                }
            } else if ((flags & 1) && smlEnemyProbe(s, (u8)(5 + smlEnemyWidthPixels(s) - 8), 8)) {
                s->enemy.direction &= 0xFE;
                moveRight = 0;
            }
            if (moveRight) {
                u8 b = speed & 0x0F;
                s->enemy.x = (u8)(s->enemy.x + b);
                if (s->enemy.carrying) {
                    smlCarryMarioX(s, b, 1);
                }
            }
        }
    }
    if (s->enemy.motion & 0xF0) {
        u8 b = s->enemy.motion >> 4;
        u8 flags = s->enemy.flags;
        if (!(s->enemy.direction & 2)) {
            int move = 1;
            if (!smlEnemyProbe(s, 4, (u8)-smlEnemyHeightOffset(s))) {
                u8 mode = flags & 0xC0;
                if (mode == 0x40) {
                    s->enemy.direction |= 2;
                    move = 0;
                } else if (mode == 0xC0) {
                    s->enemy.scriptIndex = s->enemy.moveCount = 0;
                    move = 0;
                } else if (mode != 0) {
                    move = 0;
                }
            }
            if (move) {
                s->enemy.y = (u8)(s->enemy.y - b);
                if (s->enemy.carrying) {
                    s->mario.y = (u8)(s->mario.y - b);
                }
            }
        } else {
            int move = 1;
            if (!smlEnemyProbe(s, 4, 8)) {
                u8 mode = flags & 0x30;
                if (mode == 0x10) {
                    s->enemy.direction &= 0xFD;
                    move = 0;
                } else if (mode == 0x30) {
                    s->enemy.scriptIndex = s->enemy.moveCount = 0;
                    move = 0;
                } else if (mode != 0) {
                    move = 0;
                }
            }
            if (move) {
                s->enemy.y = (u8)(s->enemy.y + b);
                if (s->enemy.carrying) {
                    s->mario.y = (u8)(s->mario.y + b);
                }
            }
        }
    }
    s->enemy.carrying = 0;
}

/* Call_2648 .runScript Fx commands. Returns 1 to continue the script, 0 to
   end this update. */
static int smlEnemySpecialCommand(SmlState *s, u16 *script, u8 command, u8 arg) {
    switch (command) {
    case 0xF8:
        s->enemy.sprite = arg;
        return 1;
    case 0xF0:
        if (arg & 0xC0) {
            if (arg & 0x80) {
                u8 b = s->enemy.direction & 0xFD;
                u8 below = s->enemy.y < s->mario.y;
                s->enemy.direction = (u8)(((below << 1) & 2) | b);
            }
            if (arg & 0x40) {
                u8 center = (u8)(s->enemy.x + (u8)((s->enemy.size & 0x70) >> 2));
                u8 left = center < s->mario.x;
                s->enemy.direction = (u8)((s->enemy.direction & 0xFE) | left);
            }
        }
        if (arg & 0x0C) {
            s->enemy.direction ^= (u8)((arg & 0x0C) >> 2);
        }
        if (arg & 0x20) {
            s->enemy.direction = (u8)((s->enemy.direction | 2) & ((arg & 2) | 0xFD));
        }
        if (arg & 0x10) {
            s->enemy.direction = (u8)((s->enemy.direction | 1) & ((arg & 1) | 0xFE));
        }
        return 1;
    case 0xF5:
        /* F5 would read rDIV; not reachable in supported levels. */
        return 1;
    case 0xF1:
        /* The buffer waits at D1A0, just past the slots, while the
           projectile is placed through it. */
        memcpy(s->mD1A0, &s->enemy, SML_ENEMY_COPY);
        smlLaunchProjectile(s);
        memcpy(&s->enemy, s->mD1A0, SML_ENEMY_COPY);
        return 1;
    case 0xF2:
        s->enemy.flags = arg;
        return 1;
    case 0xF3:
        s->enemy.id = arg;
        if (arg == 0xFF) {
            return 0;
        }
        smlInitEnemy(s, &s->enemy);
        *script = smlRead16(s, (u16)(R_ENEMY_SCRIPTS + smlRlca(s->enemy.id)));
        return 1;
    case 0xF4:
        s->enemy.moveDelay = arg;
        return 1;
    case 0xF6: {
        u8 a = (u8)(s->enemy.x - s->mario.x + 0x14);
        int carry = a < 0x20;
        if (arg != 1) {
            carry = !carry;
        }
        if (carry) {
            return 1;
        }
        s->enemy.scriptIndex = (u8)(s->enemy.scriptIndex - 2);
        return 0;
    }
    case 0xF7:
        smlExplodeAllEnemies(s);
        return 0;
    case 0xF9:
        s->effect = arg;
        return 0;
    case 0xFA:
        s->sfx = arg;
        return 0;
    case 0xFB:
        if ((u8)(s->enemy.x - s->mario.x) >= arg) {
            s->enemy.scriptIndex = 0;
        }
        return 1;
    case 0xFC:
        s->enemy.y = arg;
        s->enemy.x = 0x70;
        return 1;
    case 0xFD:
        smlRequestMusic(s, arg);
        return 0;
    default:
        return 1;
    }
}

/* Call_2648 .updateEnemy on the FFC0 buffer. */
static void smlUpdateEnemy(SmlState *s, u16 script) {
    for (;;) {
        if (s->enemy.moveCount) {
            if (s->enemy.flags & 2) {
                if (smlEnemyProbe(s, 4, 8)) {
                    s->enemy.y++;
                    return;
                }
                s->enemy.y &= 0xF8;
            }
            u8 hi = s->enemy.moveDelay >> 4, lo = s->enemy.moveDelay & 15;
            if (lo != hi) {
                s->enemy.moveDelay = (u8)(lo | smlSwap((u8)(hi + 1)));
                return;
            }
            s->enemy.moveDelay = lo;
            s->enemy.moveCount--;
            smlMoveEnemy(s);
            return;
        }
        u8 command;
        for (;;) {
            command = smlRead(s, (u16)(script + s->enemy.scriptIndex));
            s->currentCommand = command;
            if (command == 0xFF) {
                s->enemy.scriptIndex = 0;
                continue;
            }
            u8 index = s->enemy.scriptIndex;
            s->enemy.scriptIndex = (u8)(index + 1);
            if ((command & 0xF0) != 0xF0) {
                break;
            }
            s->enemy.scriptIndex = (u8)(index + 2);
            u8 arg = smlRead(s, (u16)(script + index + 1));
            s->commandArgument = arg;
            if (!smlEnemySpecialCommand(s, &script, command, arg)) {
                return;
            }
        }
        if ((command & 0xE0) == 0xE0) {
            s->enemy.moveCount = command & 15;
        } else {
            s->enemy.motion = command;
            s->enemy.moveCount = 1;
        }
    }
}

/* Call_2648: run every enemy slot's script and movement. */
static void smlUpdateEnemies(SmlState *s) {
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        SmlEnemy *e = &s->enemies[i];
        if (e->id == 0xFF) {
            continue;
        }
        smlSlotToBuffer(s, e);
        u16 script = smlRead16(s, (u16)(R_ENEMY_SCRIPTS + smlRlca(s->enemy.id)));
        smlUpdateEnemy(s, script);
        smlBufferToSlot(s, e);
    }
}

/* DrawEnemies ($2568) .drawEnemy */
static void smlDrawEnemy(SmlState *s) {
    s->spriteFlags = 0;
    u16 table =
        (s->enemy.direction & 1) ? R_ENEMY_SPRITES_RIGHT : R_ENEMY_SPRITES_LEFT;
    u16 hl = smlRead16(s, (u16)(table + (u8)smlRlca(s->enemy.sprite)));
    for (;;) {
        if (s->objectsDrawn >= 0x14) {
            return;
        }
        u8 a = smlRead(s, hl);
        while (!(a & 0x80)) {
            u8 flags = (u8)(a << 1);
            s->spriteFlags = flags & 0xEF;
            if (a & 8) {
                s->enemy.y = (u8)(s->enemy.y - 8);
            }
            if (a & 4) {
                s->enemy.y = (u8)(s->enemy.y + 8);
            }
            if (a & 2) {
                s->enemy.x = (u8)(s->enemy.x - 8);
            }
            if (a & 1) {
                s->enemy.x = (u8)(s->enemy.x + 8);
            }
            hl++;
            a = smlRead(s, hl);
        }
        if (a == 0xFF) {
            return;
        }
        SmlOamEntry *o = &s->oamBuffer[SML_OAM_ENEMIES + s->objectsDrawn];
        o->y = s->enemy.y;
        o->x = s->enemy.x;
        o->tile = a;
        o->flags = s->spriteFlags;
        hl++;
        s->objectsDrawn++;
    }
}

static void smlDrawEnemies(SmlState *s) {
    s->objectsDrawn = 0;
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        if (s->objectsDrawn >= 0x14) {
            return;
        }
        SmlEnemy *e = &s->enemies[i];
        if (e->id == 0xFF) {
            continue;
        }
        smlSlotToBuffer(s, e);
        if (s->enemy.x >= 0xE0 || s->enemy.y >= 0xC0) {
            s->enemy.id = 0xFF;
            smlBufferToSlot(s, e);
            continue;
        }
        smlDrawEnemy(s);
    }
    /* Hide the enemy sprites left over from the previous update. */
    for (int i = SML_OAM_ENEMIES + s->objectsDrawn; i < 40; i++) {
        s->oamBuffer[i].y = 0xB4;
    }
}

/* ExplodeAllEnemies ($2B2A). */
static void smlExplodeAllEnemies(SmlState *s) {
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        SmlEnemy *e = &s->enemies[i];
        if (e->id == 0xFF) {
            continue;
        }
        /* Upstream's comments name +A and +C (size and health), but the
           incs reach +9 and +B. */
        e->id = 0x27;
        e->scriptIndex = 0;
        e->moveDelay = 0;
        e->carrying = 0;
    }
    s->enemy.id = 0x27;
    s->enemy.scriptIndex = s->enemy.flags = 0;
    s->effect = 1;
}

/* InitEnemySlots ($245C). */
static void smlInitEnemySlots(SmlState *s) {
    u16 hl = smlRead16(s, (u16)(R_LEVEL_ENEMY_POINTERS + smlRlca(s->levelIndex)));
    while (smlRead(s, hl) < s->levelProgress) {
        hl = (u16)(hl + 3);
    }
    s->spawnPtrLo = (u8)hl;
    s->spawnPtrHi = (u8)(hl >> 8);
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        s->enemies[i].id = 0xFF;
    }
}

/* ------------------------------------------------------------------------
   Mario. */

static void smlKillMario(SmlState *s) {
    if (s->bossDefeated) {
        return;
    }
    s->gameState = 3;
    s->superballMario = 0;
    s->tma = 0;
    smlRequestMusic(s, 2);
    s->mario.visibility = 0x80;
    s->deathY = s->mario.y;
}

static void smlInjureMario(SmlState *s) {
    s->superStatus = 3;
    s->superballMario = 0;
    s->timer = 0x50;
    s->sfx = 6;
}

/* Call_1A6B: platform tiles that are passable from the side and below. */
static u8 smlSemiSolid(SmlState *s, u8 tile) {
    u8 world = (u8)((u8)((s->worldLevel & 0xF0) >> 4) - 1);
    u16 list = smlRead16(s, (u16)(0x1A93 + (u8)(world * 2)));
    for (;; list++) {
        u8 a = smlRead(s, list);
        if (a == 0xFD) {
            return tile;
        }
        if (a == tile) {
            return 0;
        }
    }
}

/* Jmp_1B45: Mario touches the level end. */
static void smlWin(SmlState *s) {
    u8 mask = 0xFF;
    if (s->superStatus != 2) {
        mask = 0x0F;
        s->superStatus = 0;
    }
    u8 pose = s->mario.pose & mask;
    s->mario.pose = pose;
    if ((pose & 0x0F) < 0x0A) {
        s->mario.pose = pose & 0xF0;
    }
    s->gameState = 7;
    if (!s->bossDefeated) {
        smlRequestMusic(s, 1);
        s->timer = 0xF0;
    }
    smlClearObjects(s);
    s->mario.visibility = 0;
    s->timerExpiring = 0;
    s->tma = 0;
}

/* Call_1AAD: wall probe ahead of Mario. Nonzero blocks movement. */
static int smlSideCollision(SmlState *s) {
    if (s->gameState >= 0x0E) {
        return 0;
    }
    u8 d = 7, e = 1;
    if (s->superStatus == 2 && s->mario.pose != 0x18) {
        e = 2;
    }
    for (;;) {
        s->lookupY = (u8)(s->mario.y + d);
        u8 offset = s->mario.facing ? 0xFA : 6;
        s->lookupX = (u8)(s->scrollX + (u8)(s->mario.x + offset));
        u8 tile;
        u16 hl = smlLookupTile(s, &tile);
        tile = smlSemiSolid(s, tile);
        if (tile && tile >= TILE_SOLID) {
            if (tile == TILE_COIN) {
                if (s->blockState) {
                    return s->blockState;
                }
                s->blockState = 0xC0;
                s->blockAddrHi = (u8)(hl >> 8);
                s->blockAddrLo = (u8)hl;
                s->sfx = 5;
                return 0;
            }
            if (tile == TILE_SIDE_PIPE && s->underground) {
                s->gameState = 0x0B;
                s->mario.priority = 0x80;
                s->pipeTarget = (u8)(s->mario.x + 0x18);
                s->mario.y = (u8)((s->mario.y & 0xF8) + 6);
                smlClearObjects(s);
                return 0xFF;
            }
            if (tile == TILE_FIST) {
                smlWin(s);
                return 0;
            }
            s->mario.walkFrames++;
            s->mario.gait = 2;
            return 0xFF;
        }
        d = 0xFC;
        if (!--e) {
            return 0;
        }
    }
}

/* Call_1D26 .call_1EB4: pixels per step from the gait table at $1ECE. */
static u8 smlWalkStep(SmlState *s) {
    s->mario.gaitPhase ^= 1;
    return smlRead(s, (u16)(0x1ECE + (u8)(s->mario.gait + s->mario.gaitPhase)));
}

/* Call_1D26 .jmp_1D71 */
static void smlStandStill(SmlState *s) {
    if (s->mario.jumpState) {
        return;
    }
    s->mario.pose &= 0xF0;
    s->mario.walkFrames = 1;
    s->mario.gait = 0;
}

static void smlReverseDirection(SmlState *s) {
    s->mario.walkDirection = 1;
    s->mario.speed = 8;
    if (s->mario.jumpState) {
        return;
    }
    s->mario.pose = (u8)((s->mario.pose & 0xF0) | 5);
    s->mario.walkFrames = 1;
}

static void smlMoveRight(SmlState *s) {
    if (s->mario.walkDirection == 0x20) {
        smlReverseDirection(s);
        return;
    }
    s->mario.facing = 0;
    if (smlSideCollision(s)) {
        return;
    }
    if (s->joyHeld & 0x10) {
        if (s->mario.pose == 0x18) {
            s->mario.pose = (u8)((s->mario.pose & 0xF0) | 1);
        }
        if (s->mario.speed != 6) {
            s->mario.speed++;
            s->mario.walkDirection = 0x10;
        }
    }
    int scroll = !s->underground && (s->levelEnd < 7 || (s->scrollX & 0x0C));
    if (scroll && s->mario.x > 0x50) {
        u8 b = smlWalkStep(s);
        s->scrollX = (u8)(s->scrollX + b);
        for (int i = 0; i < 8; i++) {
            s->oamBuffer[SML_OAM_FLOATIES + i].x =
                (u8)(s->oamBuffer[SML_OAM_FLOATIES + i].x - b);
        }
        smlScrollEnemies(s, b);
        for (int i = 0; i < 3; i++) {
            s->oamBuffer[SML_OAM_SUPERBALL + i].x =
                (u8)(s->oamBuffer[SML_OAM_SUPERBALL + i].x - b);
        }
        s->mario.walkFrames++;
        return;
    }
    s->mario.x = (u8)(s->mario.x + smlWalkStep(s));
    if (s->gameState != 0x0D && s->levelEnd) {
        s->scrollX &= 0xFC;
        if (s->mario.x >= 0xA0) {
            smlWin(s);
            return;
        }
    }
    s->mario.walkFrames++;
}

static void smlMoveLeft(SmlState *s) {
    if (s->mario.walkDirection == 0x10) {
        smlReverseDirection(s);
        return;
    }
    s->mario.facing = 0x20;
    if (smlSideCollision(s)) {
        return;
    }
    if (s->mario.x >= 0x0F) {
        if (s->joyHeld & 0x20) {
            if (s->mario.pose == 0x18) {
                s->mario.pose = (u8)((s->mario.pose & 0xF0) | 1);
            }
            if (s->mario.speed != 6) {
                s->mario.speed++;
                s->mario.walkDirection = 0x20;
            }
        }
        s->mario.x = (u8)(s->mario.x - smlWalkStep(s));
    }
    s->mario.walkFrames--;
}

/* Call_1D26: horizontal movement from the joypad and momentum. */
static void smlMarioMovement(SmlState *s) {
    if (s->mario.walkDirection == 1) {
        if (s->mario.speed) {
            s->mario.speed--;
            return;
        }
        s->mario.walkDirection = 0;
        smlStandStill(s);
        return;
    }
    if (s->mario.speed == 6 && !s->mario.gait) {
        s->mario.gait = 2;
    }
    u8 a = s->joyHeld;
    if ((a & 0x80) && s->superStatus == 2 && !s->mario.jumpState) {
        s->mario.pose = 0x18;
        if ((s->joyHeld & 0x30) || !s->mario.speed) {
            s->mario.speed = 0;
            return;
        }
    }
    for (;;) {
        if (a & 0x10) {
            smlMoveRight(s);
            return;
        }
        if (a & 0x20) {
            smlMoveLeft(s);
            return;
        }
        if (!s->mario.speed) {
            break;
        }
        s->mario.gait = 0;
        s->mario.speed--;
        a = s->mario.walkDirection;
    }
    s->mario.walkDirection = 0;
    if (s->mario.jumpState) {
        return;
    }
    smlStandStill(s);
}

/* Call_16F5: draw Mario, advance the walk cycle, then move. */
static void smlAnimateMario(SmlState *s) {
    smlDrawMario(s);
    if (s->mario.grounded && (s->mario.pose & 0x0F) < 0x0A) {
        u8 frames = s->mario.walkFrames;
        int advance = s->mario.gait == 0x23 ? !(frames & 1) : !(frames & 3);
        if (advance && s->mario.pose != 0x18) {
            u8 pose = (u8)(s->mario.pose + 1);
            s->mario.pose = pose;
            if ((pose & 0x0F) >= 4) {
                s->mario.pose = (u8)((pose & 0xF0) | 1);
            }
        }
    }
    smlMarioMovement(s);
}

/* Jmp_185D: land on the tile below. */
static void smlLand(SmlState *s) {
    s->mario.y = (u8)(((u8)(s->mario.y - 2) & 0xFC) | 6);
    s->mario.jumpState = s->mario.jumpIndex = s->mario.jumpCut = 0;
    s->mario.grounded = 1;
    if (s->mario.speed >= 7) {
        s->mario.speed = 6;
    }
}

/* Call_17BC .jmp_1801: keep falling. */
static void smlFall(SmlState *s) {
    if (s->mario.jumpState == 2) {
        return;
    }
    s->mario.y = (u8)(s->mario.y + 3);
    s->mario.grounded = 0;
    if (!s->mario.gait) {
        s->mario.gait = 2;
    }
}

/* Call_17BC .jmp_181E: solid, spike or coin under Mario. */
static void smlFloorTile(SmlState *s, u8 tile, u16 hl) {
    if (tile == TILE_SPIKE && !s->invincibilityTimer) {
        if (!s->superStatus) {
            smlKillMario(s);
            smlLand(s);
            return;
        }
        if (s->superStatus == 2) {
            smlInjureMario(s);
            smlLand(s);
            return;
        }
    }
    if (tile == TILE_COIN) {
        if (!s->blockState) {
            s->blockState = 0xC0;
            s->blockAddrHi = (u8)(hl >> 8);
            s->blockAddrLo = (u8)hl;
            s->sfx = 5;
        }
        smlFall(s);
        return;
    }
    smlLand(s);
}

/* Jmp_1765: Down on a pipe with a warp record enters it. */
static void smlEnterPipe(SmlState *s, u16 hl) {
    if (!(s->joyHeld & 0x80)) {
        smlLand(s);
        return;
    }
    s->lookupAddrHi = (u8)(hl >> 8);
    s->lookupAddrLo = (u8)hl;
    u16 overlay = (u16)(hl + 0x3000);
    if (!SML_MEM(s, overlay)) {
        smlLand(s);
        return;
    }
    for (int i = 0; i < 4; i++) {
        s->pipe[i] = SML_MEM(s, (u16)(overlay - 0x20 * i));
    }
    smlTileToObject(s);
    s->pipeTarget = (u8)(s->mario.y + 0x10);
    s->mario.x = (u8)(s->lookupX - s->scrollX + 8);
    s->mario.priority = 0x80;
    s->gameState = 9;
    if (!s->invincibilityTimer) {
        smlRequestMusic(s, 4);
    }
    smlClearObjects(s);
    smlLand(s);
}

/* Call_17BC: floor probe below Mario's feet. */
static void smlFloorCollision(SmlState *s) {
    if (s->mario.jumpState == 1) {
        return;
    }
    s->lookupY = (u8)(s->mario.y + 0x0B);
    s->lookupX = (u8)(s->mario.x + s->scrollX + 0xFE);
    u8 tile;
    u16 hl = smlLookupTile(s, &tile);
    if (tile == TILE_PIPE) {
        smlEnterPipe(s, hl);
        return;
    }
    if (tile == TILE_BOSS_SWITCH) {
        if (s->gameState >= 0x0E) {
            smlFloorTile(s, tile, hl);
        } else {
            smlWin(s);
        }
        return;
    }
    if (tile >= TILE_SOLID) {
        smlFloorTile(s, tile, hl);
        return;
    }
    u8 b = s->mario.gait == 4 && !s->mario.jumpState ? 8 : 4;
    s->lookupX = (u8)(s->lookupX + b);
    hl = smlLookupTile(s, &tile);
    if (tile >= TILE_SOLID) {
        smlFloorTile(s, tile, hl);
        return;
    }
    smlFall(s);
}

/* Call_490D: advance one entity along the jump table at $216D. */
static void smlVerticalMotion(SmlState *s, SmlEntity *e) {
    u8 index = e->jumpIndex;
    if (!e->jumpState) {
        return;
    }
    if (e->jumpState != 2) {
        u8 delta = smlRead(s, (u16)(R_JUMP_TABLE + index));
        if (delta != 0x7F) {
            e->y = (u8)(e->y - delta);
            e->jumpIndex = (u8)(index + 1);
            return;
        }
        index--;
        e->jumpState = 2;
    } else {
        if (index == 0xFF) {
            e->jumpState = 0;
            e->jumpIndex = 0;
            return;
        }
        if (smlRead(s, (u16)(R_JUMP_TABLE + index)) == 0x7F) {
            index--;
        }
    }
    e->y = (u8)(e->y + smlRead(s, (u16)(R_JUMP_TABLE + index)));
    e->jumpIndex = (u8)(index - 1);
}

/* 3:498B: jump, crouch-run and superball input. */
static void smlFireSuperball(SmlState *s);

static void smlMarioInput(SmlState *s) {
    if (s->gameState == 0x0D) {
        return; /* autoscroll levels are outside the 1-1 port */
    }
    u8 b = s->joyPressed;
    u8 a = s->joyHeld;
    if (a & 2) {
        if (!s->mario.jumpState) {
            s->mario.gait = s->mario.speed < 3 ? 2 : 4;
        }
    } else if (s->mario.gait == 4) {
        s->mario.gait = 2;
    }
    if (a & 1) {
        if (!s->mario.jumpState && s->mario.grounded && (b & 1)) {
            s->mario.grounded = 0;
            if (s->mario.pose != 0x18) {
                s->mario.pose = (u8)((s->mario.pose & 0xF0) | 4);
                if (s->mario.gait != 4) {
                    s->mario.gait = 2;
                    s->mario.jumpIndex = 2;
                }
                s->mario.speed = 0x30;
            }
            s->sfx = 1;
            s->mario.jumpState = 1;
        }
    } else if (s->mario.jumpState == 1) {
        /* Jmp_4966: releasing A cuts the ascent at table index 0F. */
        if (s->mario.jumpIndex < 0x0F) {
            s->mario.jumpCut = (u8)(s->mario.jumpIndex - 1);
            s->mario.jumpIndex = 0x0F;
        }
    }
    if (b & 0x80) {
        s->mario.speed = 0x20;
    }
    if (b & 2) {
        if (s->mario.speed == 6 && !s->demo) {
            s->mario.speed = 0;
        }
        smlFireSuperball(s);
    }
}

/* 3:4A0C: spawn a superball in object 0. */
static void smlFireSuperball(SmlState *s) {
    SmlOamEntry *sprite = &s->oamBuffer[SML_OAM_SUPERBALL];
    if (!s->superballMario || s->superball) {
        return;
    }
    u8 facing = s->mario.facing;
    sprite->y = (u8)(s->mario.y + 0xFE);
    sprite->x = (u8)(s->mario.x + ((facing & 0x20) ? 0xF8 : 2));
    sprite->tile = 0x60;
    sprite->flags = 0;
    s->superball = (facing & 0x20) ? 0x0A : 0x09;
    s->sfx = 2;
    s->superballCooldown = 0x0C;
    s->superballTtl = 0xFF;
}

/* 3:4AEA: block fragments fall with the scroll. */
static void smlMoveFragments(SmlState *s) {
    for (int i = 0; i < 4; i++) {
        SmlEntity *e = &s->fragments[i];
        u8 a = e->visibility;
        if (a == 0x80) {
            e->visibility = 0xFF;
        }
        if (a) {
            continue;
        }
        if (!e->jumpState) {
            e->visibility = 0x80;
            e->x = 0xFF;
            continue;
        }
        u8 scrolled = (u8)(s->scrollX - s->fragmentScroll);
        if (e->facing) {
            e->x--;
        } else {
            e->x++;
        }
        e->x = (u8)(e->x - scrolled);
    }
    s->fragmentScroll = s->scrollX;
}

/* 3:4B3C: move the bounced block sprite with Mario's head. */
static void smlMoveBouncedBlock(SmlState *s) {
    if (s->blockState != 3) {
        return;
    }
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].x = (u8)(s->blockX - s->scrollX);
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y = (u8)(s->mario.y - 0x0B);
    if (!s->mario.grounded) {
        u8 b = s->blockY;
        if ((u8)(b - 4) >= s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y) {
            s->mario.jumpState = 2;
            return;
        }
        if (b >= s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y) {
            return;
        }
    }
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y = 0;
    s->blockState = 4;
}

/* 3:4B6F: falling below the screen kills Mario. */
static void smlCheckPit(SmlState *s) {
    if (s->mario.y < 0xB4 || s->mario.y >= 0xC0) {
        return;
    }
    s->superStatus = 0;
    s->superballMario = 0;
    s->gameState = 1;
    smlRequestMusic(s, 2);
    s->timer = 0x90;
}

/* 3:4B8A and 3:4BB5: grow and shrink blinking. */
static void smlPowerTransitions(SmlState *s) {
    if (s->superStatus == 1) {
        if (s->timer) {
            if (!(s->timer & 3)) {
                s->mario.visibility = 0;
                s->mario.pose ^= 0x10;
            }
        } else {
            s->superStatus = 2;
            s->mario.visibility = 0;
            s->mario.pose |= 0x10;
        }
    }
    if (s->superStatus == 4) {
        if (s->timer) {
            if (!(s->timer & 3)) {
                s->mario.visibility ^= 0x80;
            }
        } else {
            s->superStatus = 0;
            s->mario.visibility = 0;
            s->mario.pose &= 0x0F;
        }
    } else if (s->superStatus == 3) {
        if (s->timer) {
            if (!(s->timer & 3)) {
                s->mario.pose ^= 0x10;
            }
        } else {
            s->superStatus = 4;
            s->timer = 0x40;
            s->mario.pose &= 0x0F;
        }
    }
}

/* ------------------------------------------------------------------------
   Blocks hit from below (bank 0 .call_198C and neighbours). */

static void smlStartBounce(SmlState *s, u16 hl) {
    s->blockState = 2;
    s->blockAddrHi = (u8)(hl >> 8);
    s->blockAddrLo = (u8)hl;
    s->lookupAddrHi = (u8)(hl >> 8);
    s->lookupAddrLo = (u8)hl;
}

/* .jmp_1937: bounce a used-up or coin block. */
static void smlBounceBlock(SmlState *s, u16 hl) {
    smlStartBounce(s, hl);
    smlTileToObject(s);
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y = (u8)(s->mario.y - 0x0B);
    s->blockY = s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y;
    s->blockX = s->lookupX;
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].x = (u8)(s->lookupX - s->scrollX);
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].flags = 0;
    s->floatyX = s->oamBuffer[SML_OAM_BOUNCED_BLOCK].x;
}

/* .jmp_1923 */
static void smlBounceBrick(SmlState *s, u16 hl) {
    if (s->blockState) {
        return;
    }
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].tile = 0x82;
    if (!s->sfx) {
        s->sfx = 7;
    }
    smlBounceBlock(s, hl);
}

/* .jmp_18A4: a coin comes out. */
static void smlBlockCoin(SmlState *s, u16 hl) {
    if (s->blockState) {
        return;
    }
    s->sfx = 5;
    s->floatyY = (u8)(s->mario.y - 0x10);
    s->floatyControl = 0xC0;
    s->coinPending = 0xC0;
    if (s->coinBlockTimer) {
        smlBounceBrick(s, hl);
        return;
    }
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].tile = 0x80;
    smlBounceBlock(s, hl);
}

/* .jmp_18C7: an item comes out. */
static void smlBlockItem(SmlState *s, u16 hl, u8 item) {
    s->blockContents = item;
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].tile = 0x80;
    s->sfx = 7;
    if (s->blockState) {
        return;
    }
    smlStartBounce(s, hl);
    s->blockContents = SML_MEM(s, (u16)(hl + 0x3000));
    smlTileToObject(s);
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y = (u8)(s->mario.y - 0x0B);
    s->enemy.y = s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y;
    s->blockY = s->oamBuffer[SML_OAM_BOUNCED_BLOCK].y;
    s->blockX = s->lookupX;
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].x = (u8)(s->lookupX - s->scrollX);
    s->enemy.x = s->oamBuffer[SML_OAM_BOUNCED_BLOCK].x;
    s->oamBuffer[SML_OAM_BOUNCED_BLOCK].flags = 0;
    item = s->blockContents;
    if (item == 0xF0) {
        return;
    }
    if (item == 0x28 && s->superStatus == 2) {
        item = 0x2D;
    }
    smlSpawnPowerup(s, item);
}

/* .jmp_189B: block with overlay contents. */
static void smlBlockContents(SmlState *s, u16 hl, u8 contents) {
    if (contents != 0xC0) {
        smlBlockItem(s, hl, contents);
        return;
    }
    s->coinBlockTimer = 0xFF;
    smlBlockCoin(s, hl);
}

/* .jmp_19E1: breakable brick. */
static void smlHitBrick(SmlState *s, u16 hl) {
    if (SML_MEM(s, (u16)(hl + 0x3000)) == 0xC0) {
        smlBlockCoin(s, hl);
        return;
    }
    if (s->superStatus != 2) {
        smlBounceBrick(s, hl);
        return;
    }
    if (s->blockState) {
        return;
    }
    s->blockState = 1;
    s->blockAddrHi = (u8)(hl >> 8);
    s->blockAddrLo = (u8)hl;
    for (int i = 0; i < 4; i++) {
        SmlEntity *e = &s->fragments[i];
        e->visibility = 0;
        e->y = (u8)(s->mario.y - 0x0D);
        e->x = (u8)(s->mario.x + 2);
        e->jumpState = 1;
        e->jumpIndex = 7;
    }
    /* Fragments 1 and 3 start 4 pixels left; 2 and 3 start at jump table
       index 0B instead of 07. */
    s->fragments[1].x = (u8)(s->fragments[1].x - 4);
    s->fragments[3].x = (u8)(s->fragments[3].x - 4);
    s->fragments[2].jumpIndex = 0x0B;
    s->fragments[3].jumpIndex = 0x0B;
    s->fragmentScroll = s->scrollX;
    s->effect = 2;
    smlAddScore(s, 0x0050);
    s->mario.jumpState = 2;
}

/* .jmp_1888 */
static void smlHitSolidBlock(SmlState *s, u16 hl) {
    if (s->blockState) {
        return;
    }
    u8 contents = SML_MEM(s, (u16)(hl + 0x3000));
    if (!contents) {
        smlHitBrick(s, hl);
    } else if (contents == 0xF0) {
        s->oamBuffer[SML_OAM_BOUNCED_BLOCK].tile = 0x80;
        smlBounceBlock(s, hl);
    } else {
        smlBlockContents(s, hl, contents);
    }
}

/* .call_198C: head collision while ascending. */
static void smlHeadCollision(SmlState *s) {
    if (s->mario.jumpState != 1) {
        return;
    }
    s->lookupY = (u8)(s->mario.y - 3);
    s->lookupX = (u8)(s->scrollX + s->mario.x + 2);
    u8 tile;
    u16 hl = smlLookupTile(s, &tile);
    if (tile < TILE_SOLID && tile != TILE_HIDDEN_BLOCK) {
        s->lookupX = (u8)(s->lookupX - 4);
        hl = smlLookupTile(s, &tile);
        if (tile < TILE_SOLID && tile != TILE_HIDDEN_BLOCK) {
            return;
        }
    }
    if (tile == TILE_HIDDEN_BLOCK) {
        /* .jmp_187B */
        if (s->blockState || !SML_MEM(s, (u16)(hl + 0x3000))) {
            return;
        }
        smlHitSolidBlock(s, hl);
        return;
    }
    tile = smlSemiSolid(s, tile);
    if (!tile) {
        return;
    }
    if (tile == TILE_BRICK) {
        smlHitBrick(s, hl);
    } else if (tile == TILE_COIN) {
        if (!s->blockState) {
            s->blockState = 0xC0;
            s->blockAddrHi = (u8)(hl >> 8);
            s->blockAddrLo = (u8)hl;
            s->sfx = 5;
        }
    } else if (tile == TILE_MYSTERY) {
        /* .jmp_1966 */
        if (s->blockState) {
            return;
        }
        u8 contents = SML_MEM(s, (u16)(hl + 0x3000));
        if (contents) {
            smlBlockContents(s, hl, contents);
            return;
        }
        s->sfx = 5;
        s->oamBuffer[SML_OAM_BOUNCED_BLOCK].tile = 0x81;
        s->floatyY = (u8)(s->mario.y - 0x10);
        s->floatyControl = 0xC0;
        smlBounceBlock(s, hl);
    } else if (tile == TILE_BREAKABLE) {
        smlHitSolidBlock(s, hl);
    } else {
        s->mario.jumpState = 2;
        s->sfx = 7;
    }
}

/* Call_1B86 (VBlank): settle bounced blocks, bricks and touched coins. */
static void smlSettleBlocks(SmlState *s) {
    s->coinsDrawn = 0;
    if (s->coinPending) {
        smlAddCoin(s);
    }
    u8 state = s->blockState;
    u16 de = (u16)(s->blockAddrHi << 8 | s->blockAddrLo);
    if (state == 4) {
        s->blockState = 0;
        u8 tile = s->oamBuffer[SML_OAM_BOUNCED_BLOCK].tile;
        if (tile != TILE_BRICK) {
            if (tile == TILE_MYSTERY) {
                smlAddCoin(s);
            }
            tile = TILE_USED_BLOCK;
        }
        SML_MEM(s, de) = tile;
        return;
    }
    if (state != 1 && state != 2 && state != 0xC0) {
        return;
    }
    s->blockState = state == 2 ? 3 : 0;
    SML_MEM(s, de) = TILE_BLANK;
    if (state == 0xC0) {
        smlAddCoin(s);
        return;
    }
    u16 above = (u16)(de - 0x20);
    if (SML_MEM(s, above) != TILE_COIN) {
        return;
    }
    SML_MEM(s, above) = TILE_BLANK;
    s->sfx = 5;
    s->lookupAddrHi = (u8)(above >> 8);
    s->lookupAddrLo = (u8)above;
    smlTileToObject(s);
    s->floatyX = (u8)(s->lookupX - s->scrollX);
    s->floatyY = (u8)(s->lookupY + 0x14);
    s->floatyControl = 0xC0;
    smlAddCoin(s);
}

/* ------------------------------------------------------------------------
   Collisions with enemies and powerups. */

static void smlPositionFloaty(SmlState *s) {
    s->floatyX = (u8)(s->mario.x - 4);
    s->floatyY = (u8)(s->mario.y - 0x10);
}

/* Call_84E .enemyKilled: stomp reward and chain. */
static void smlEnemyKilled(SmlState *s) {
    s->sfx = 3;
    smlPositionFloaty(s);
    s->floatyControl = s->stompReward;
    if (s->stompChainTimer) {
        u8 chain = s->stompChain;
        if (chain != 3) {
            s->stompChain = ++chain;
        }
        if (s->floatyControl != 0x50) {
            u8 a = s->floatyControl;
            for (u8 b = chain; b; b--) {
                a = (u8)(a << 1);
            }
            s->floatyControl = a;
            s->stompChainTimer = 0x32;
            return;
        }
    }
    s->stompChain = 0;
    s->stompChainTimer = 0x32;
}

static void smlPowerupCollision(SmlState *s, SmlEnemy *e) {
    u8 id = s->tempB;
    if (id == 0x34) {
        s->invincibilityTimer = 0xF0; /* F8 in the original; see rom_patch.py */
        smlRequestMusic(s, 0x0C);
    } else if (id == 0x2B) {
        s->floatyControl = 0xFF;
        s->sfx = 8;
        s->livesDelta = 1;
        smlPositionFloaty(s);
        e->id = 0xFF;
        return;
    } else if (id == 0x29 || id == 0x2E) {
        if (s->superStatus != 2) {
            s->superStatus = 1;
            s->timer = 0x50;
            s->sfx = 4;
        } else if (id == 0x2E) {
            s->superballMario = 2;
            s->sfx = 4;
        }
    } else {
        return;
    }
    s->floatyControl = 0x10;
    smlPositionFloaty(s);
    e->id = 0xFF;
}

/* Call_84E: Mario against every object slot; at most one hit per update. */
static void smlMarioObjectCollision(SmlState *s) {
    if (s->stompChainTimer) {
        s->stompChainTimer--;
    }
    for (int i = SML_ENEMY_SLOTS - 1; i >= 0; i--) {
        SmlEnemy *e = &s->enemies[i];
        if (e->id == 0xFF) {
            continue;
        }
        s->tempB = e->id;
        s->tempC = (u8)(i * sizeof(SmlEnemy)); /* the slot address's low byte */
        s->enemyHealth = e->health;
        u8 top = s->mario.y;
        if (s->superStatus == 2 && s->mario.pose != 0x18) {
            top = (u8)(top - 2);
        }
        s->boxTop = top;
        s->boxBottom = (u8)(s->mario.y + 6);
        s->boxLeft = (u8)(s->mario.x - 3);
        s->boxRight = (u8)(s->mario.x + 2);
        if (!smlBoxCollision(s, e)) {
            continue;
        }
        u8 x = e->x;
        u8 right = smlHitboxRight(x, e->size);
        if (i == 9) {
            smlPowerupCollision(s, e);
            return;
        }
        int stomp = 0;
        if (s->gameState != 0x0D && !s->invincibilityTimer) {
            u8 y = e->y;
            stomp = (u8)(s->mario.x + 6) >= x && (u8)(s->mario.x - 6) < right &&
                    s->mario.y < (u8)(y - 3);
        }
        if (stomp) {
            if (e->size & 0x80) {
                return;
            }
            smlStompReward(s);
            if (!smlTransformEnemy(s, e, 0)) {
                return;
            }
            s->mario.grounded = 0;
            s->mario.jumpIndex = 0x0D;
            s->mario.jumpState = 1;
            s->mario.pose = (u8)((s->mario.pose & 0xF0) | 4);
            smlEnemyKilled(s);
            return;
        }
        if (s->invincibilityTimer) {
            if (smlTransformEnemy(s, e, 4)) {
                smlEnemyKilled(s);
            }
            return;
        }
        if (s->superStatus >= 3) {
            return;
        }
        if (!smlEnemyContact(s, e)) {
            return;
        }
        if (s->superStatus) {
            smlInjureMario(s);
        } else {
            smlKillMario(s);
        }
        return;
    }
}

/* Call_AEA: stand on platform-like objects (bit 7 of D1xA). */
static void smlPlatformCollision(SmlState *s) {
    if (s->mario.jumpState == 1) {
        return;
    }
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        SmlEnemy *e = &s->enemies[i];
        if (e->id == 0xFF || !(e->size & 0x80)) {
            continue;
        }
        u8 top = smlHitboxTop(e->y, e->size);
        s->boxTop = top;
        if ((u8)(top - (u8)(s->mario.y + 6)) >= 7) {
            continue;
        }
        u8 x = e->x;
        u8 d = (u8)(x - s->mario.x);
        if (x >= s->mario.x && d >= 3) {
            continue;
        }
        u8 right = smlHitboxRight(x, e->size);
        if (s->mario.x >= right && (u8)(s->mario.x - right) >= 3) {
            continue;
        }
        s->mario.y = (u8)(top - 0x0A);
        smlTransformEnemy(s, e, 0);
        e->carrying = 1;
        s->mario.jumpState = s->mario.jumpIndex = s->mario.jumpCut = 0;
        s->mario.grounded = 1;
        if (s->mario.speed >= 7) {
            s->mario.speed = 6;
        }
        return;
    }
}

/* Call_A2D: a bouncing block hits the enemy standing on it. */
static void smlBlockHitsEnemy(SmlState *s) {
    if (!s->blockState || s->blockState == 0xC0) {
        return;
    }
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        SmlEnemy *e = &s->enemies[i];
        if (e->id == 0xFF || (e->size & 0x80)) {
            continue;
        }
        s->enemyHealth = e->health;
        u8 y = e->y;
        if (s->mario.y < y) {
            continue;
        }
        u8 diff = (u8)(s->mario.y - y);
        if (diff > 0x14 || (u8)(0x14 - diff) >= 7) {
            continue;
        }
        u8 x = e->x;
        u8 right = smlHitboxRight(x, e->size);
        if ((u8)(s->mario.x - 6) >= right || (u8)(s->mario.x + 6) < x) {
            continue;
        }
        smlStompReward(s);
        if (!smlTransformEnemy(s, e, 1)) {
            continue;
        }
        s->floatyX = (u8)(s->mario.x + 0xFC);
        s->floatyY = (u8)(s->mario.y - 0x10);
        s->floatyControl = s->stompReward;
    }
}

/* ------------------------------------------------------------------------
   Superball. */

/* FindNeighboringTile ($1FC9 region): probe for the superball. Returns
   nonzero (carry) when the tile is passable. */
static int smlSuperballProbe(SmlState *s, u8 x) {
    s->lookupX = (u8)(s->scrollX + x);
    u8 a;
    u16 hl = smlLookupTile(s, &a);
    if (a == TILE_COIN) {
        if (s->gameState == 0x0D) {
            a = 0x0D;
        } else if (s->blockState) {
            a = s->blockState;
        } else {
            s->blockState = 0xC0;
            s->blockAddrHi = (u8)(hl >> 8);
            s->blockAddrLo = (u8)hl;
            s->sfx = 5;
            a = 5;
        }
    }
    return a < TILE_SOLID;
}

/* Call_200A: superball against enemies. */
static void smlSuperballEnemies(SmlState *s) {
    SmlOamEntry *sprite = &s->oamBuffer[SML_OAM_SUPERBALL];
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        SmlEnemy *e = &s->enemies[i];
        if (e->id == 0xFF || (e->size & 0x80)) {
            continue;
        }
        s->enemyHealth = e->health;
        s->boxLeft = sprite->x;
        s->boxRight = (u8)(sprite->x + 4);
        s->boxTop = sprite->y;
        s->boxBottom = (u8)(sprite->y + 3);
        if (!smlBoxCollision(s, e)) {
            continue;
        }
        smlStompReward(s);
        u8 result = smlSuperballHit(s, e);
        if (!result) {
            continue;
        }
        s->floatyY = (u8)(sprite->y - 8);
        s->floatyX = sprite->x;
        if (result == 0xFF) {
            s->sfx = 3;
            s->floatyControl = s->stompReward;
        }
        sprite->x = 0;
        sprite->y = 0;
        s->superball = 0;
    }
}

/* Call_1F2D: move the superball in object 0. */
static void smlMoveSuperball(SmlState *s) {
    u8 *ball = &s->superball; /* direction bits */
    SmlOamEntry *sprite = &s->oamBuffer[SML_OAM_SUPERBALL];
    if (!*ball) {
        return;
    }
    int removed = 0;
    if (!s->superballTtl) {
        removed = 1;
    } else {
        s->superballTtl--;
        u8 x;
        if (*ball & 1) {
            x = (u8)(sprite->x + 2);
            sprite->x = x;
            if (x >= 0xA2) {
                removed = 1;
            } else {
                s->lookupY = sprite->y;
                if (!smlSuperballProbe(s, (u8)(x + 3))) {
                    *ball = (u8)((*ball & 0xFC) | 2);
                }
            }
        } else {
            x = (u8)(sprite->x - 2);
            sprite->x = x;
            if (x < 4) {
                removed = 1;
            } else {
                s->lookupY = sprite->y;
                if (!smlSuperballProbe(s, (u8)(x - 2))) {
                    *ball = (u8)((*ball & 0xFC) | 1);
                }
            }
        }
        if (!removed) {
            if (*ball & 4) {
                u8 y = (u8)(sprite->y - 2);
                sprite->y = y;
                if (y < 0x10) {
                    removed = 1;
                } else {
                    s->lookupY = (u8)(y - 1);
                    if (!smlSuperballProbe(s, sprite->x)) {
                        *ball = (u8)((*ball & 0xF3) | 8);
                    }
                }
            } else {
                u8 y = (u8)(sprite->y + 2);
                sprite->y = y;
                if (y >= 0xA8) {
                    removed = 1;
                } else {
                    s->lookupY = (u8)(y + 4);
                    if (!smlSuperballProbe(s, sprite->x)) {
                        *ball = (u8)((*ball & 0xF3) | 4);
                    }
                }
            }
        }
    }
    if (removed) {
        sprite->y = 0;
        *ball = 0;
    }
    smlSuperballEnemies(s);
}

/* Call_1F03: star timer and blinking. The original also ends the star when
   its music stops; rom_patch.py removes that read of DFE9. */
static void smlStartLevelMusic(SmlState *s);

static void smlUpdateInvincibility(SmlState *s) {
    if ((s->frameCounter & 3) || !s->invincibilityTimer) {
        return;
    }
    if (s->invincibilityTimer != 1) {
        s->invincibilityTimer--;
        s->mario.visibility ^= 0x80;
        return;
    }
    s->invincibilityTimer = 0;
    s->mario.visibility = 0;
    smlStartLevelMusic(s);
}

/* ------------------------------------------------------------------------
   Bank 2: timer and floaties. */

/* UpdateGameTimer (2:584B). */
static void smlUpdateGameTimer(SmlState *s) {
    if (s->timerExpiring == 3) {
        return;
    }
    if (--s->timerTicks) {
        return;
    }
    s->timerTicks = 0x28;
    u8 c = s->timerHundreds;
    u8 a = smlDecrementBcd(s->timerDigits);
    s->timerDigits = a;
    if (a == 0x99) {
        s->timerHundreds = (u8)(c - 1);
        return;
    }
    if (a == 0x50) {
        if (!c) {
            s->timerExpiring = 2;
            s->tma = 0x50;
        }
        return;
    }
    if (a) {
        return;
    }
    if (!c) {
        s->timerExpiring = 3;
    } else if (c == 1) {
        s->timerExpiring = 1;
        s->tma = 0x30;
    }
}

/* UpdateFloaties (2:5892). */
static void smlUpdateFloaties(SmlState *s) {
    u8 b = s->floatyControl;
    if (b) {
        u8 l = s->nextFloaty;
        u8 next = (u8)(l + 8);
        s->nextFloaty = next == 0x50 ? 0x30 : next;
        unsigned i = l == 0x30 ? 0 : l == 0x38 ? 1
                                 : l == 0x40   ? 2
                                               : 3;
        s->floatyTtl[i] = 0x20;
        s->floatyCoinTile[i] = 0xF6;
        if (b == 0xC0) {
            s->floatyIsCoin[i] = 0xC0;
        }
        /* nextFloaty is an OAM buffer offset: 30, 38, 40 or 48. */
        SmlOamEntry *left = &s->oamBuffer[l / 4], *right = left + 1;
        static const u8 values[] = {0x01, 0x02, 0x04, 0x05, 0x08, 0x10, 0x20, 0x40, 0x50, 0x80, 0xFF};
        static const u16 tiles[] = {0x5958, 0x5A58, 0x5B58, 0x5C58, 0x5D58, 0x5957, 0x5A57, 0x5B57, 0x5C57, 0x5D57, 0x5E5F};
        u16 de = 0xF6FE;
        for (unsigned k = 0; k < sizeof(values); k++) {
            if (b == values[k]) {
                de = tiles[k];
                break;
            }
        }
        left->y = s->floatyY;
        left->x = s->floatyX;
        left->tile = (u8)(de >> 8);
        right->y = s->floatyY;
        right->x = (u8)(s->floatyX + 8);
        right->tile = (u8)de;
        s->floatyControl = s->floatyY = s->floatyX = 0;
        static const u8 scoreValues[] = {0x01, 0x02, 0x04, 0x05, 0x08, 0x10, 0x20, 0x40, 0x50, 0x80};
        static const u8 scores[] = {0x01, 0x02, 0x04, 0x05, 0x08, 0x10, 0x20, 0x40, 0x50, 0x80};
        for (unsigned k = 0; k < sizeof(scoreValues); k++) {
            if (b == scoreValues[k]) {
                smlAddScore(s, (u16)(scores[k] << 8));
                break;
            }
        }
    }
    for (unsigned i = 0; i < 4; i++) {
        SmlOamEntry *left = &s->oamBuffer[SML_OAM_FLOATIES + 2 * i],
                    *right = left + 1;
        if (!left->y) {
            continue;
        }
        if (s->floatyIsCoin[i] != 0xC0) {
            u8 phase = (u8)(s->floatyPhase[i] + 1);
            s->floatyPhase[i] = phase;
            if (phase != 2) {
                continue;
            }
            s->floatyPhase[i] = 0;
        }
        left->y--;
        right->y--;
        if (left->tile >= 0xF6) {
            u8 a = (u8)(s->floatyCoinTile[i] + 1);
            s->floatyCoinTile[i] = a;
            left->tile = a;
            if (a >= 0xF9) {
                a = (u8)(a - 2);
                left->tile = a;
                if (a != 0xF7) {
                    a = (u8)(a - 2);
                    s->floatyCoinTile[i] = a;
                    left->tile = a;
                }
            }
        }
        if (--s->floatyTtl[i]) {
            continue;
        }
        s->floatyTtl[i] = 0x20;
        s->floatyCoinTile[i] = 0xF6;
        left->y = left->x = left->tile = 0;
        right->y = right->x = right->tile = 0;
        s->floatyIsCoin[i] = 0;
        s->floatyPhase[i] = 0;
    }
}

/* ------------------------------------------------------------------------
   Game states. */

/* StartLevelMusic ($07A3) with InitSound's RAM effects. */
static void smlStartLevelMusic(SmlState *s) {
    if (s->invincibilityTimer) {
        return;
    }
    s->sfxPlaying = s->musicPlaying = s->noisePlaying = s->effectPlaying = 0;
    s->mDF1F[0] = s->mDF2F[0] = s->mDF3F[0] = s->mDF4F[0] = 0;
    s->pauseUnpauseMusic = s->pauseTuneTimer = 0;
    smlRequestMusic(s, s->pipeRoom ? 4 : s->rom[0x07CE + s->levelIndex]);
}

/* Call_807: reset Mario's entities and draw the first screen. */
static void smlDrawFirstScreen(SmlState *s) {
    /* The table is 0x51 bytes: its last lands just past the entities. */
    memcpy(s->entities, s->rom + R_DATA_211D, sizeof(s->entities));
    s->mC250[0] = s->rom[R_DATA_211D + sizeof(s->entities)];
    if (s->superStatus) {
        s->mario.pose = 0x10;
    }
    s->columnIndex = s->columnPtrHi = s->columnPtrLo = 0;
    s->mapColumn = s->columnPhase = s->floatyX = 0;
    s->columnToggle = 0;
    s->scrollAnchor = 0;
    s->mapColumn = 0x40;
    int columns = s->gameState == 0x0A || s->levelIndex == 0x0C ? 0x14 : 0x1B;
    for (int i = 0; i < columns; i++) {
        smlLoadNextColumn(s);
        smlDrawColumn(s);
    }
}

/* Call_165E: clear the tile overlay. */
static void smlClearOverlay(SmlState *s) {
    memset(s->overlay, 0, sizeof(s->overlay));
}

/* GameState_01 ($06BC): wait, then reset to the checkpoint. */
static void smlGameState01(SmlState *s) {
    if (s->timer) {
        return;
    }
    for (int i = 0; i < SML_ENEMY_SLOTS; i++) {
        s->enemies[i].id = 0xFF;
    }
    s->superStatus = 0;
    s->livesDelta = 0xFF;
    s->gameState = 2;
}

/* GameState_02 ($06DC): redraw the level from the checkpoint, LCD off. */
static void smlGameState02(SmlState *s) {
    s->lcdc = 0;
    smlClearObjects(s);
    smlClearOverlay(s);
    u8 screen = s->screenIndex;
    if (s->underground) {
        s->underground = 0;
        screen = (u8)(s->pipeReturnScreen + 1);
    }
    if (screen != 3) {
        screen--;
    }
    u16 bc = 0x17D4;
    static const u8 limits[] = {7, 0x0B, 0x0F, 0x13, 0x17};
    static const u16 checkpoints[] = {0x030C, 0x0734, 0x0B5C, 0x0F84, 0x13AC};
    for (int i = 0; i < 5; i++) {
        if (screen < limits[i]) {
            bc = checkpoints[i];
            break;
        }
    }
    s->screenIndex = (u8)(bc >> 8);
    s->columnIndex = 0;
    s->levelProgress = (u8)bc;
    smlDrawFirstScreen(s);
    SML_MEM(s, 0x982B) = TILE_BLANK;
    SML_MEM(s, 0x982C) = s->worldLevel >> 4;
    SML_MEM(s, 0x982E) = s->worldLevel & 15;
    for (int i = 0; i < 9; i++) {
        SML_MEM(s, 0x9C00 + i) = s->rom[0x079A + i];
    }
    s->gameState = 0;
    s->invincibilityTimer = 0;
    s->lcdc = 0xC3;
    smlStartLevelMusic(s);
    s->interruptFlags = 0;
    s->scrollX = 0;
    s->levelEnd = 0;
    s->blockState = 0;
    s->timerExpiring = 0;
    s->tma = 0;
    s->timerDigits = 0;
    s->timerHundreds = 4;
    s->timerTicks = 0x28;
    s->mapColumn = 0x5B;
    smlInitEnemySlots(s);
}

/* GameState_03 ($0B8D): Mario's death pose. */
static void smlGameState03(SmlState *s) {
    u8 c = s->deathY, d = (u8)(c - 8);
    u8 b = (u8)(s->mario.x + 0xF8);
    const u8 sprite[16] = {d, b, 0x0F, 0x00, c, b, 0x1F, 0x00, d, (u8)(b + 8), 0x0F, 0x20, c, (u8)(b + 8), 0x1F, 0x20};
    memcpy(&s->oamBuffer[3], sprite, sizeof(sprite)); /* Mario's four sprites */
    s->gameState = 4;
    s->deathStep = 0;
    s->superStatus = 0;
    s->pipeRoom = 0;
    smlClearObjects(s);
}

/* GameState_04 ($0BD6): death fall along table $0C19. */
static void smlGameState04(SmlState *s) {
    u8 index = s->deathStep;
    s->deathStep = (u8)(index + 1);
    u8 b = s->rom[0x0C19 + index];
    if (b == 0x7F) {
        s->deathStep = index;
        b = 2;
    }
    u8 a = 0;
    for (int i = 0; i < 4; i++) {
        a = (u8)(s->oamBuffer[3 + i].y + b);
        s->oamBuffer[3 + i].y = a;
    }
    if (a < 0xB4) {
        return;
    }
    if (s->timerExpiring == 0xFF) {
        s->gameState = 0x3B;
    } else {
        s->timer = 0x90;
        s->gameState = 1;
    }
}

/* GameState_07 ($0C40): victory music, then the timer countdown. */
static void smlGameState07(SmlState *s) {
    if (s->timer) {
        smlDrawMario(s);
        return;
    }
    if (!s->bossDefeated) {
        s->timer = 0x40;
    }
    s->gameState = 5;
    s->timerExpiring = 0;
    s->tma = 0;
    if ((s->worldLevel & 0x0F) != 3) {
        return;
    }
    smlExplodeAllEnemies(s);
    if (s->worldLevel == 0x43) {
        s->gameState = 6; /* no countdown after Tatanga */
    }
}

/* Call_2491: spawn, run and draw enemies. */
static void smlEnemies(SmlState *s) {
    smlSpawnEnemies(s);
    smlUpdateEnemies(s);
    smlDrawEnemies(s);
}

/* GameState_05 ($0C73): count the timer into the score. */
static void smlGameState05(SmlState *s) {
    if ((s->worldLevel & 0x0F) == 3) {
        /* Boss levels let the explosions finish. */
        s->levelProgress = 0;
        smlEnemies(s);
    }
    if (s->timer) {
        return;
    }
    if (!(s->timerDigits | s->timerHundreds)) {
        s->gameState = 6;
        s->timer = 0x26;
        return;
    }
    s->timerTicks = 1;
    smlSwitchBank(s, 2);
    smlUpdateGameTimer(s);
    smlUpdateFloaties(s);
    smlRestoreBank(s);
    smlAddScore(s, 0x0010);
    s->timer = 1;
    s->timerExpiring = 0;
    if (!(s->timerDigits & 1)) {
        s->sfx = 0x0A;
    }
}

/* GameState_06 ($0CCB): choose the next level or the bonus game. */
static void smlGameState06(SmlState *s) {
    if (s->timer) {
        return;
    }
    s->timerExpiring = 0;
    s->tma = 0;
    if ((s->worldLevel & 0x0F) == 3) {
        /* Boss levels hand off to the gate to Daisy (1C), drawn from level
           0C (the hangar); FFFB keeps the level index for later. */
        s->gameState = 0x1C;
        s->activeRomBank = 3;
        s->tempB = s->levelIndex;
        s->levelIndex = 0x0C;
        s->screenIndex = s->columnIndex = 0;
        s->columnToggle = 0;
        s->savedMapColumn = s->mapColumn;
        s->timer = 6;
        /* 4-3 goes on to Tatanga's end (27) instead; the shooter levels are
           outside the port. */
        return;
    }
    u8 y = s->mario.y;
    if (y < 0x60 || y >= 0xA0) {
        s->activeRomBank = 2;
        s->gameState = 0x12;
    } else {
        s->gameState = 8;
    }
}

/* GameState_09 ($161B): sink into a pipe. */
static void smlGameState09(SmlState *s) {
    if (s->pipeTarget == s->mario.y) {
        s->gameState = 0x0A;
        s->underground = 0x0A;
        return;
    }
    s->mario.y++;
    smlAnimateMario(s);
}

/* GameState_0A ($162F): load the underground room. */
static void smlGameState0A(SmlState *s) {
    s->lcdc = 0;
    s->columnIndex = 0;
    smlClearObjects(s);
    smlClearOverlay(s);
    s->screenIndex = s->pipeRoom;
    smlDrawFirstScreen(s);
    smlInitEnemySlots(s);
    s->mario.y = 0x20;
    s->mario.x = 0x1D;
    s->mario.priority = 0;
    s->interruptFlags = 0;
    s->gameState = 0;
    s->scrollX = 0;
    s->lcdc = 0xC3;
}

/* GameState_0B ($166C): walk into the side pipe and return overground. */
static void smlGameState0B(SmlState *s) {
    if (!(s->frameCounter & 1)) {
        return;
    }
    if (s->pipeTarget >= s->mario.x) {
        s->mario.x++;
        s->mario.walkFrames++;
        smlAnimateMario(s);
        return;
    }
    s->screenIndex = s->pipeReturnScreen;
    s->lcdc = 0;
    s->columnIndex = 0;
    smlClearOverlay(s);
    s->pipeRoom = s->pipeReturnScreen = 0;
    u8 y = s->pipeReturnY, x = s->pipeReturnX;
    smlDrawFirstScreen(s);
    s->mario.priority = 0x80;
    s->mario.y = y;
    s->pipeTarget = (u8)(y - 0x12);
    s->mario.x = x;
    u8 b = (u8)(s->screenIndex - 4);
    s->levelProgress = (u8)(smlRlca(smlRlca(smlRlca(b))) + b + b + 0x0C);
    s->interruptFlags = 0;
    s->scrollX = 0;
    s->mapColumn = 0x5B;
    smlInitEnemySlots(s);
    smlClearObjects(s);
    s->lcdc = 0xC3;
    s->gameState = 0x0C;
    smlStartLevelMusic(s);
}

/* GameState_0C ($16DA): rise out of a pipe. */
static void smlGameState0C(SmlState *s) {
    if (!(s->frameCounter & 1)) {
        return;
    }
    if (s->pipeTarget == s->mario.y) {
        s->gameState = 0;
        s->mario.priority = 0;
        s->underground = 0;
        return;
    }
    s->mario.y--;
    smlAnimateMario(s);
}

/* GameState_39 ($1C7C): prepare game over. */
static void smlGameState39(SmlState *s) {
    for (int i = 0; i < 0x11; i++) {
        SML_MEM(s, 0x9C00 + i) = s->rom[0x1CD7 + i];
    }
    smlRequestMusic(s, 0x10);
    s->continueWorldLevel = s->worldLevel;
    u8 continues = (u8)(s->continues + (s->score[2] >> 4));
    s->continues = continues >= 0x0A ? 9 : continues;
    memset(s->oamBuffer, 0, sizeof(s->oamBuffer));
    s->timerExpiring = 0;
    s->tma = 0;
    s->wy = 0x8F;
    s->wx = 0x07;
    s->tempB = 0xFF;
    s->gameState = 0x3A;
}

/* GameState_3B ($1CF0): print TIME UP in the window. */
static void smlGameState3B(SmlState *s) {
    for (int i = 0; i < 9; i++) {
        SML_MEM(s, 0x9C00 + i) = s->rom[0x1D14 + i];
    }
    s->lcdc |= 0x20;
    s->timer = 0xA0;
    s->gameState = 0x3C;
}

/* GameState_3C ($1D1D): wait, then lose the life. */
static void smlGameState3C(SmlState *s) {
    if (!s->timer) {
        s->gameState = 1;
    }
}

/* GameState_00 ($0627): normal gameplay. */
static void smlGameState00(SmlState *s) {
    smlStreamColumns(s);
    smlMarioObjectCollision(s);
    smlSwitchBank(s, 3);
    /* 3:48FC applies a jump cut released during the previous update. */
    if (s->mario.jumpCut && s->mario.jumpIndex < 0x0F) {
        s->mario.jumpIndex = s->mario.jumpCut;
        s->mario.jumpCut = 0;
    }
    for (int i = 0; i < 5; i++) {
        smlVerticalMotion(s, &s->entities[i]);
    }
    smlMarioInput(s);
    smlMoveFragments(s);
    smlMoveBouncedBlock(s);
    smlCheckPit(s);
    smlPowerTransitions(s);
    smlRestoreBank(s);
    smlMoveSuperball(s);
    smlSpawnEnemies(s);
    smlUpdateEnemies(s);
    smlDrawEnemies(s);
    smlSwitchBank(s, 2);
    smlUpdateGameTimer(s);
    smlUpdateFloaties(s);
    smlRestoreBank(s);
    smlHeadCollision(s);
    smlAnimateMario(s);
    smlFloorCollision(s);
    smlPlatformCollision(s);
    smlBlockHitsEnemy(s);
    smlUpdateInvincibility(s);
    if (s->coinBlockTimer) {
        s->coinBlockTimer--;
    }
}

/* VBlank ($0060): publish the update. */
static void smlVBlank(SmlState *s) {
    smlDrawColumn(s);
    smlSettleBlocks(s);
    smlUpdateLives(s);
    memcpy(s->oam, s->oamBuffer, sizeof(s->oam));
    smlDisplayScore(s);
    smlDisplayTimer(s);
    smlAnimateBackground(s);
    s->frameCounter++;
    if (s->gameState == 0x3A) {
        s->lcdc |= 0x20;
    }
    s->scx = s->scy = 0;
    s->vblankOccurred = 1;
}

/* The game states the port runs, by hGameState; the rest (menus, autoscroll,
   the bonus game, cutscenes) are outside it. */
static void (*const smlGameStates[256])(SmlState *) = {
    smlGameState00,
    smlGameState01,
    smlGameState02,
    smlGameState03,
    smlGameState04,
    smlGameState05,
    smlGameState06,
    smlGameState07,
    0,
    smlGameState09,
    smlGameState0A,
    smlGameState0B,
    smlGameState0C,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    smlGameState39,
    0,
    smlGameState3B,
    smlGameState3C,
};

/* The terminal states where stepping stops: game over (3A) and the
   level-clear handoffs, the next level (08), the bonus game (12), and after
   a boss the gate to Daisy (1C); that cutscene is outside the port. */
static SmlStepStatus smlStopStatus(u8 state) {
    if (state == 0x3A) {
        return SML_STEP_GAME_OVER;
    }
    if (state == 0x08 || state == 0x12 || state == 0x1C) {
        return SML_STEP_LEVEL_CLEAR;
    }
    return SML_STEP_READY;
}

SmlStepStatus smlStepFrame(SmlState *s, uint8_t buttons) {
    u8 state = s->gameState;
    SmlStepStatus stop = smlStopStatus(state);
    if (stop != SML_STEP_READY) {
        return stop;
    }
    /* Stop before changing anything if the update would run a state the port
       lacks; the time-up kill below switches to state 03 first. */
    if (!smlGameStates[s->timerExpiring == 3 && !s->bossDefeated ? 3 : state]) {
        return SML_STEP_UNSUPPORTED;
    }
    s->vblankOccurred = 0;
    if (s->timerExpiring == 3) {
        s->timerExpiring = 0xFF;
        smlKillMario(s);
        smlDrawMario(s);
    }
    /* 3:47F2 ReadJoypad. Start and Select (pause, reset) are not ported. */
    smlSwitchBank(s, 3);
    u8 joy = (u8)(((buttons & 0x0F) << 4 | (buttons & 0x30) >> 4));
    s->joyPressed = (u8)(joy & ~s->joyHeld);
    s->joyHeld = joy;
    smlRestoreBank(s);
    if (s->timer) {
        s->timer--;
    }
    if (s->timer2) {
        s->timer2--;
    }
    smlGameStates[s->gameState](s);
    smlVBlank(s);
    return SML_STEP_READY;
}

int smlAgentCanAct(const SmlState *s) {
    return s->gameState == 0 && s->timerExpiring != 3;
}

/* Pipe transitions (09-0C) ignore the joypad and always return to 00. */
void smlRunPipe(SmlState *s, uint8_t buttons) {
    while (s->gameState >= 0x09 && s->gameState <= 0x0C) {
        smlStepFrame(s, buttons);
    }
}

SmlStepStatus smlStepAgent(SmlState *s, uint8_t buttons) {
    smlRunPipe(s, buttons);
    if (!smlAgentCanAct(s)) {
        SmlStepStatus stop = smlStopStatus(s->gameState);
        return stop != SML_STEP_READY ? stop : SML_STEP_UNSUPPORTED;
    }
    smlStepFrame(s, buttons);
    smlRunPipe(s, buttons);
    return SML_STEP_READY;
}

/* ------------------------------------------------------------------------
   Rendering: the frame the LCD draws from this boundary state. The HUD
   (lines 0-15) is unscrolled; the LCD STAT handler at LYC 15 applies
   hScrollX (and wScrollY when enabled) to the world. */

/* The four DMG shades as RGBA words, in memory order. */
static void smlShades(uint8_t palette, uint8_t out[4][4]) {
    static const uint8_t shades[4][4] = {{255, 255, 255, 255}, {153, 153, 153, 255}, {85, 85, 85, 255}, {0, 0, 0, 255}};
    for (int c = 0; c < 4; c++) {
        memcpy(out[c], shades[(palette >> (2 * c)) & 3], 4);
    }
}

/* Bit 7 - i of a byte in byte i: a tile plane byte spread to eight pixels. */
#define SML_SPREAD(b) \
    {(b) >> 7 & 1, (b) >> 6 & 1, (b) >> 5 & 1, (b) >> 4 & 1, (b) >> 3 & 1, (b) >> 2 & 1, (b) >> 1 & 1, (b) & 1}
#define SML_SPREAD4(b) \
    SML_SPREAD(b), SML_SPREAD((b) + 1), SML_SPREAD((b) + 2), SML_SPREAD((b) + 3)
#define SML_SPREAD16(b)                                         \
    SML_SPREAD4(b), SML_SPREAD4((b) + 4), SML_SPREAD4((b) + 8), \
        SML_SPREAD4((b) + 12)
#define SML_SPREAD64(b)                                              \
    SML_SPREAD16(b), SML_SPREAD16((b) + 16), SML_SPREAD16((b) + 32), \
        SML_SPREAD16((b) + 48)
static const uint8_t smlSpread[256][8] = {SML_SPREAD64(0), SML_SPREAD64(64), SML_SPREAD64(128), SML_SPREAD64(192)};

/* A tile row's eight 2-bit colors, leftmost first. Each byte of the planes
   is 0 or 1, so shifting the whole word moves no bit across bytes. */
static inline void smlTileRow(const SmlState *s, uint16_t row, uint8_t out[8]) {
    uint64_t lo, hi;
    memcpy(&lo, smlSpread[SML_MEM(s, row)], 8);
    memcpy(&hi, smlSpread[SML_MEM(s, (uint16_t)(row + 1))], 8);
    lo |= hi << 1;
    memcpy(out, &lo, 8);
}

void smlRenderFrame(const SmlState *s, uint8_t rgba[SML_WIDTH * SML_HEIGHT * 4]) {
    if (!(s->lcdc & 0x80)) {
        memset(rgba, 255, SML_WIDTH * SML_HEIGHT * 4);
        return;
    }
    const SmlOamEntry *oam = s->oam;
    uint8_t lcdc = s->lcdc;
    uint8_t bgShades[4][4], objShades[2][4][4];
    smlShades(s->bgp, bgShades);
    smlShades(s->obp0, objShades[0]);
    smlShades(s->obp1, objShades[1]);
    uint8_t worldY = s->scrollYEnabled ? s->scrollY : 0;
    int height = (lcdc & 4) ? 16 : 8;
    /* The window (pause, TIME UP, game over) keeps its own line counter,
       advanced only on lines where it is drawn. */
    int windowLine = -1, windowX = s->wx - 7;
    for (int y = 0; y < SML_HEIGHT; y++) {
        uint8_t *out = rgba + y * SML_WIDTH * 4;
        /* Background and window colors, 8 extra so a tile row can overrun. */
        uint8_t bg[SML_WIDTH + 8];
        int window = (lcdc & 0x20) && s->wy <= y && windowX < SML_WIDTH;
        if (window) {
            windowLine++;
        }
        int split = window ? (windowX < 0 ? 0 : windowX) : SML_WIDTH;
        for (int layer = 0; layer < 2; layer++) {
            int from = layer ? split : 0, to = layer ? SML_WIDTH : split;
            if (from >= to) {
                continue;
            }
            int sy, originX;
            uint16_t base;
            if (layer) {
                sy = windowLine;
                originX = -windowX; /* screen x + originX = window x */
                base = lcdc & 0x40 ? 0x9C00 : 0x9800;
            } else {
                sy = (y + (y < 16 ? 0 : worldY)) & 255;
                originX = y < 16 ? 0 : s->scrollX;
                base = lcdc & 8 ? 0x9C00 : 0x9800;
            }
            for (int x = from; x < to;) {
                int sx = (x + originX) & 255;
                /* 8 spare bytes, so a partial tile copies 8 at a time. */
                uint8_t colors[16] = {0};
                if (lcdc & 1) {
                    uint8_t tile = SML_MEM(s, (uint16_t)(base + (sy / 8) * 32 + sx / 8));
                    uint16_t addr = (uint16_t)((lcdc & 16) ? 0x8000 + tile * 16
                                                           : 0x9000 + (int8_t)tile * 16);
                    smlTileRow(s, (uint16_t)(addr + (sy & 7) * 2), colors);
                }
                int n = 8 - (sx & 7);
                if (n > to - x) {
                    n = to - x;
                }
                memcpy(bg + x, colors + (sx & 7), 8);
                x += n;
            }
        }
        for (int x = 0; x < SML_WIDTH; x++) {
            memcpy(out + x * 4, bgShades[bg[x]], 4);
        }
        if (!(lcdc & 2)) {
            continue;
        }
        /* DMG object priority: the lowest X wins a pixel, equal X the lowest
           OAM slot; transparent pixels never win. BG priority applies after. */
        int8_t owner[SML_WIDTH];
        int ownerX[SML_WIDTH];
        uint8_t ownerColor[SML_WIDTH];
        int left = SML_WIDTH, right = 0; /* columns objects touched */
        for (int i = 0, found = 0; i < 40 && found < 10; i++) {
            int oy = oam[i].y - 16;
            if (y < oy || y >= oy + height) {
                continue;
            }
            found++;
            int ox = oam[i].x - 8, yy = y - oy;
            int from = ox < 0 ? 0 : ox, to = ox + 8 > SML_WIDTH ? SML_WIDTH : ox + 8;
            if (from >= to) {
                continue;
            }
            /* Clear columns as the touched range grows. */
            if (left >= right) {
                memset(owner + from, -1, (size_t)(to - from));
                left = from;
                right = to;
            } else {
                if (from < left) {
                    memset(owner + from, -1, (size_t)(left - from));
                    left = from;
                }
                if (to > right) {
                    memset(owner + right, -1, (size_t)(to - right));
                    right = to;
                }
            }
            uint8_t flags = oam[i].flags, tile = oam[i].tile;
            if (flags & 0x40) {
                yy = height - 1 - yy;
            }
            if (height == 16) {
                tile &= 0xFE;
            }
            uint8_t colors[8];
            smlTileRow(s, (uint16_t)(0x8000 + tile * 16 + yy * 2), colors);
            for (int x = from; x < to; x++) {
                int xx = x - ox;
                uint8_t color = colors[(flags & 0x20) ? 7 - xx : xx];
                if (!color || (owner[x] >= 0 && ownerX[x] <= ox)) {
                    continue;
                }
                owner[x] = (int8_t)i;
                ownerX[x] = ox;
                ownerColor[x] = color;
            }
        }
        for (int x = left; x < right; x++) {
            if (owner[x] < 0) {
                continue;
            }
            uint8_t flags = oam[owner[x]].flags;
            if (!(flags & 0x80) || bg[x] == 0) {
                memcpy(out + x * 4, objShades[(flags & 0x10) ? 1 : 0][ownerColor[x]], 4);
            }
        }
    }
}
#endif
