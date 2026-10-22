#ifndef IMPULSE_WARS_MAP_H
#define IMPULSE_WARS_MAP_H

#include <errno.h>
#include <string.h>

#include "game.h"

// clang-format off

static const char boringLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

static mapEntry boringMap = {
    .layout = boringLayout,
    .columns = 21,
    .rows = 21,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 5,
};

static const char prototypeArenaLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','d','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','W','W','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','D','D','W','W','D','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','d','O','D','D','D','D','D','O','O','O','O','O','O','O','D',
    'D','O','w','O','O','O','O','D','D','D','D','D','O','O','O','O','w','O','O','D',
    'D','O','O','O','O','O','O','D','D','D','D','D','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','W','W','O','O','O','d','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','D',
    'D','O','O','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

static mapEntry prototypeArenaMap = {
    .layout = prototypeArenaLayout,
    .columns = 20,
    .rows = 20,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 4,
    .weaponSpawnWeights = {[MACHINEGUN_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};

static const char snipersLayout[] = {
    'B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B',
    'B','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','B',
    'B','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','D','D','B','O','B','D','D','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','D','D','D','B','O','B','D','D','D','O','O','O','O','O','B',
    'B','O','O','O','O','O','D','D','D','B','O','B','D','D','D','O','O','O','O','O','B',
    'B','O','O','O','O','O','B','B','B','B','O','B','B','B','B','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','B','B','B','B','O','B','B','B','B','O','O','O','O','O','B',
    'B','O','O','O','O','O','D','D','D','B','O','B','D','D','D','O','O','O','O','O','B',
    'B','O','O','O','O','O','D','D','D','B','O','B','D','D','D','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','D','D','B','O','B','D','D','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','B',
    'B','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','B',
    'B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B',
};

static mapEntry snipersMap = {
    .layout = snipersLayout,
    .columns = 21,
    .rows = 21,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 4,
    .weaponSpawnWeights = {[SNIPER_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};

static const char roomsLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','W','O','O','O','W','D','D','D','D','D','W','O','O','O','W','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

static mapEntry roomsMap = {
    .layout = roomsLayout,
    .columns = 21,
    .rows = 21,
    .randFloatingStandardWalls = 3,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 3,
    .hasSetFloatingWalls = false,
    .weaponPickups = 8,
    .maxSuddenDeathWalls = 5,
    .weaponSpawnWeights = {[SHOTGUN_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};

static const char xArenaLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','d','O','O','O','O','O','O','O','O','O','d','O','D',
    'D','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','W','W','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','W','W','D','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','D','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','D',
    'D','O','O','O','O','O','D','D','O','O','w','O','O','O','O','D','W','W','O','O','O','O','D',
    'D','O','O','O','O','W','W','D','D','O','O','O','O','O','D','D','W','W','O','O','w','O','D',
    'D','O','w','O','O','W','W','D','O','O','O','O','d','O','O','D','D','O','O','O','O','O','D',
    'D','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','D','D','D','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','D','W','W','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','W','W','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','D',
    'D','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

static mapEntry xArena = {
    .layout = xArenaLayout,
    .columns = 23,
    .rows = 23,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 6,
    .weaponSpawnWeights = {[NUKE_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};

static const char crossBounceLayout[] = {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','B','B','B','B','O','O','O','O','B','D','D','D','D','B','O','O','O','O','B','B','B','B','D',
    'D','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','D',
    'D','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','D',
    'D','B','O','O','B','B','O','O','O','O','O','w','d','O','O','O','O','O','B','B','O','O','B','D',
    'D','O','O','O','B','D','D','O','O','O','O','O','O','O','O','O','O','D','D','B','O','O','O','D',
    'D','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','B','O','O','B','D','O','O','O','O','O','O','O','O','D',
    'D','B','O','O','O','O','O','O','D','D','B','O','O','B','D','D','O','O','O','O','O','O','B','D',
    'D','D','O','O','O','O','O','O','B','B','B','O','O','B','B','B','O','O','O','O','O','O','D','D',
    'D','D','O','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','D','D',
    'D','D','O','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','O','O','D','D',
    'D','D','O','O','O','O','O','O','B','B','B','O','O','B','B','B','O','O','O','O','O','O','D','D',
    'D','B','O','O','O','O','O','O','D','D','B','O','O','B','D','D','O','O','O','O','O','O','B','D',
    'D','O','O','O','O','O','O','O','O','D','B','O','O','B','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','D',
    'D','O','O','O','B','D','D','O','O','O','O','O','O','O','O','O','O','D','D','B','O','O','O','D',
    'D','B','O','O','B','B','O','O','O','O','O','d','w','O','O','O','O','O','B','B','O','O','B','D',
    'D','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','D',
    'D','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','D',
    'D','B','B','B','B','O','O','O','O','B','D','D','D','D','B','O','O','O','O','B','B','B','B','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

static mapEntry crossBounce = {
    .layout = crossBounceLayout,
    .columns = 24,
    .rows = 24,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 6,
};

static const char asteriskArenaLayout[]= {
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','D','W','O','O','O','W','D','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','O','O','O','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','D','D','O','O','O','O','O','O','O','O','O','D','D','O','O','O','O','D',
    'D','O','O','O','O','W','W','D','O','O','O','O','O','O','O','D','W','W','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','W','W','D','O','O','O','O','O','O','O','D','W','W','O','O','O','O','D',
    'D','O','O','O','O','D','D','O','O','O','O','O','O','O','O','O','D','D','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','D','O','O','O','D','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','D','W','O','O','O','W','D','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D',
};

static mapEntry asteriskArena = {
    .layout = asteriskArenaLayout,
    .columns = 23,
    .rows = 23,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 7,
};

static const char foamPitLayout[] = {
    'B','B','B','W','W','W','D','D','D','B','B','D','D','D','W','W','W','B','B','B',
    'B','O','O','O','O','O','O','O','D','B','B','D','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','B','B','O','O','O','O','O','O','O','O','B',
    'W','O','O','d','O','O','O','O','O','O','O','O','O','O','O','O','d','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','d','O','O','O','O','d','O','O','O','O','O','O','D',
    'D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D',
    'B','B','B','O','O','O','O','O','O','d','d','O','O','O','O','O','O','B','B','B',
    'B','B','B','O','O','O','O','O','O','d','d','O','O','O','O','O','O','B','B','B',
    'D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D',
    'D','O','O','O','O','O','O','d','O','O','O','O','d','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','d','O','O','O','O','O','O','O','O','O','O','O','O','d','O','O','W',
    'B','O','O','O','O','O','O','O','O','B','B','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','D','B','B','D','O','O','O','O','O','O','O','B',
    'B','B','B','W','W','W','D','D','D','B','B','D','D','D','W','W','W','B','B','B',
};

static mapEntry foamPitMap = {
    .layout = foamPitLayout,
    .columns = 20,
    .rows = 20,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 5,
    .maxSuddenDeathWalls = 5,
    .weaponSpawnWeights = {[IMPLODER_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};

static const char siegeLayout[] = {
    'B','B','B','W','W','W','W','W','D','D','D','D','D','D','D','D','D','W','W','W','W','W','B','B','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','O','B',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','W','O','O','O','W','D','D','W','W','O','O','O','O','O','W','W','D','D','W','O','O','O','W','W',
    'W','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','W',
    'W','O','O','O','b','O','W','O','b','O','O','O','O','O','O','O','b','O','W','O','b','O','O','O','W',
    'W','O','b','O','O','O','W','O','O','O','O','O','O','O','O','O','O','O','W','O','O','O','b','O','W',
    'W','O','O','O','O','O','W','O','O','O','O','B','B','B','O','O','O','O','W','O','O','O','O','O','W',
    'W','W','O','O','O','W','W','O','O','O','O','O','O','O','O','O','O','O','W','W','O','O','O','W','W',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','b','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'W','W','O','O','O','W','W','O','O','O','O','O','O','O','O','O','O','O','W','W','O','O','O','W','W',
    'W','O','O','O','O','O','W','O','O','O','O','B','B','B','O','O','O','O','W','O','O','O','O','O','W',
    'W','O','b','O','O','O','W','O','O','O','O','O','O','O','O','O','O','O','W','O','O','O','b','O','W',
    'W','O','O','O','b','O','W','O','b','O','O','O','O','O','O','O','b','O','W','O','b','O','O','O','W',
    'W','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','W',
    'W','W','O','O','O','W','D','D','W','W','O','O','O','O','O','W','W','D','D','W','O','O','O','W','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'B','O','d','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','d','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','B','B','W','W','W','W','W','D','D','D','D','D','D','D','D','D','W','W','W','W','W','B','B','B',
};

static mapEntry siegeMap = {
    .layout = siegeLayout,
    .columns = 25,
    .rows = 24,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 5,
    .maxSuddenDeathWalls = 8,
    .weaponSpawnWeights = {[GRAPPLE_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};

static const char pinballLayout[] = {
    'B','B','B','B','B','B','B','B','D','D','D','D','D','B','B','B','B','B','B','B','B',
    'B','D','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','D','B',
    'B','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','B',
    'B','D','O','O','O','O','O','O','B','O','O','O','B','O','O','O','O','O','O','D','B',
    'B','O','O','O','B','O','O','O','O','O','B','O','O','O','O','O','B','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','B','O','O','O','O','O','B','O','O','O','O','O','O','B',
    'D','O','O','B','O','O','O','O','O','O','O','O','O','O','O','O','O','B','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','B','O','O','O','O','O','B','O','O','O','O','O','B','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','B','O','O','O','O','O','O','O','O','O','O','O','O','O','B','O','O','D',
    'B','O','O','O','O','O','O','B','O','O','O','O','O','B','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','B','O','O','O','O','O','B','O','O','O','O','O','B','O','O','O','B',
    'B','D','O','O','O','O','O','O','B','O','O','O','B','O','O','O','O','O','O','D','B',
    'B','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','B',
    'B','D','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','D','B',
    'B','B','B','B','B','B','B','B','D','D','D','D','D','B','B','B','B','B','B','B','B',
};

static mapEntry pinballMap = {
    .layout = pinballLayout,
    .columns = 21,
    .rows = 21,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 5,
};

static const char moatLayout[] = {
    'W','W','W','W','W','W','W','W','W','W','D','D','D','W','W','W','W','W','W','W','W','W','W',
    'W','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','D','D','D','D','B','O','O','O','B','D','D','D','D','O','O','O','O','W',
    'W','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','W',
    'W','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','W',
    'W','O','O','O','O','D','O','O','w','O','O','O','O','O','w','O','O','D','O','O','O','O','W',
    'W','O','O','O','O','B','O','O','O','O','O','O','O','O','O','O','O','B','O','O','O','O','W',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','O','D',
    'D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D',
    'W','O','O','O','O','B','O','O','O','O','O','O','O','O','O','O','O','B','O','O','O','O','W',
    'W','O','O','O','O','D','O','O','w','O','O','O','O','O','w','O','O','D','O','O','O','O','W',
    'W','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','W',
    'W','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','W',
    'W','O','O','O','O','D','D','D','D','B','O','O','O','B','D','D','D','D','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B','W',
    'W','W','W','W','W','W','W','W','W','W','D','D','D','W','W','W','W','W','W','W','W','W','W',
};

static mapEntry moatMap = {
    .layout = moatLayout,
    .columns = 23,
    .rows = 19,
    .randFloatingStandardWalls = 6,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 2,
    .hasSetFloatingWalls = true,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 4,
    .weaponSpawnWeights = {[MINE_LAUNCHER_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};

static const char hourglassLayout[] = {
    'B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B',
    'B','O','O','O','O','O','B','B','B','B','B','B','B','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','B','B','B','B','B','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','B','B','B','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','B','O','O','O','O','w','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','W',
    'W','W','D','O','O','O','O','O','O','O','O','O','O','O','O','O','D','W','W',
    'W','W','W','D','D','O','O','O','O','d','O','O','O','O','D','D','W','W','W',
    'W','W','W','W','W','D','D','O','O','O','O','O','D','D','W','W','W','W','W',
    'W','W','W','W','W','W','D','O','O','d','O','O','D','W','W','W','W','W','W',
    'W','W','W','W','W','D','D','O','O','O','O','O','D','D','W','W','W','W','W',
    'W','W','W','D','D','O','O','O','O','d','O','O','O','O','D','D','W','W','W',
    'W','W','D','O','O','O','O','O','O','O','O','O','O','O','O','O','D','W','W',
    'W','D','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','D','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','w','O','O','O','O','B','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','B','B','B','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','B','B','B','B','B','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','B','B','B','B','B','B','B','O','O','O','O','O','B',
    'B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B',
};

static mapEntry hourglassMap = {
    .layout = hourglassLayout,
    .columns = 19,
    .rows = 25,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = true,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 4,
    .weaponSpawnWeights = {[BLACK_HOLE_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};

static const char jawsLayout[] = {
    'B','W','W','W','D','W','W','W','W','D','W','W','W','D','W','W','W','D','W','W','W','D','W','W','B',
    'B','O','O','O','D','O','O','O','O','D','O','O','O','D','O','O','O','D','O','O','O','D','O','O','B',
    'B','O','O','O','D','O','O','O','O','D','O','O','O','D','O','O','O','D','O','O','O','D','O','O','B',
    'B','O','O','O','D','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','D','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','b','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','w','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','W','W','O','O','O','O','O','O','O','O','O','O','O','W','W','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','w','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','b','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','D','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','D','O','O','O','B',
    'B','O','O','D','O','O','O','D','O','O','O','D','O','O','O','D','O','O','O','O','D','O','O','O','B',
    'B','O','O','D','O','O','O','D','O','O','O','D','O','O','O','D','O','O','O','O','D','O','O','O','B',
    'B','W','W','D','W','W','W','D','W','W','W','D','W','W','W','D','W','W','W','W','D','W','W','W','B',
};

static mapEntry jawsMap = {
    .layout = jawsLayout,
    .columns = 25,
    .rows = 17,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 5,
    .maxSuddenDeathWalls = 4,
};

static const char cavernLayout[] = {
    'W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W',
    'W','D','D','D','O','O','O','O','O','O','W','W','W','W','D','O','O','O','O','O','O','W','W',
    'W','D','O','O','O','O','O','O','O','O','O','W','W','D','O','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','D','W',
    'W','O','O','O','O','B','B','O','O','O','O','O','O','O','O','O','O','O','O','O','D','D','W',
    'W','O','O','O','B','B','B','O','O','O','O','O','O','O','D','D','O','O','O','O','O','O','W',
    'W','W','O','O','O','B','O','O','O','O','O','O','O','O','D','D','D','O','O','O','O','W','W',
    'W','W','W','O','O','O','O','O','O','O','O','O','O','O','D','D','O','O','O','O','O','W','W',
    'W','W','W','W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','W','D','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','W','W','W','W','W','W','W','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','O','O','O','W','O','O','O','O','O','O','O','O','D','W','W',
    'W','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','W','W','W','W',
    'W','W','O','O','O','O','O','D','D','O','O','O','O','O','O','O','O','O','O','O','W','W','W',
    'W','W','O','O','O','O','D','D','D','O','O','O','O','O','O','O','O','B','O','O','O','W','W',
    'W','O','O','O','O','O','O','D','D','O','O','O','O','O','O','O','B','B','B','O','O','O','W',
    'W','D','D','O','O','O','O','O','O','O','O','O','O','O','O','O','B','B','O','O','O','O','W',
    'W','D','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','W',
    'W','O','O','O','O','O','O','O','O','D','W','W','O','O','O','O','O','O','O','O','O','D','W',
    'W','W','O','O','O','O','O','O','D','W','W','W','W','O','O','O','O','O','O','D','D','D','W',
    'W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W',
};

static mapEntry cavernMap = {
    .layout = cavernLayout,
    .columns = 23,
    .rows = 21,
    .randFloatingStandardWalls = 0,
    .randFloatingBouncyWalls = 0,
    .randFloatingDeathWalls = 0,
    .hasSetFloatingWalls = false,
    .weaponPickups = 6,
    .maxSuddenDeathWalls = 4,
};

static const char crossroadsLayout[] = {
    'W','W','W','W','W','W','W','W','D','D','D','D','D','D','D','D','D','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','O','O','O','O','O','O','O','O','O','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','O','O','O','O','O','O','O','O','O','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','O','O','O','O','O','O','O','O','O','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','O','O','O','O','O','O','O','O','O','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','D','D','O','O','O','O','O','O','O','O','O','D','D','W','W','W','W','W','W',
    'W','W','W','W','W','W','D','O','O','O','O','O','O','O','O','O','O','O','D','W','W','W','W','W','W',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','D','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'B','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B',
    'W','W','W','W','W','W','D','O','O','O','O','O','O','O','O','O','O','O','D','W','W','W','W','W','W',
    'W','W','W','W','W','W','D','D','O','O','O','O','O','O','O','O','O','D','D','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','O','O','O','O','O','O','O','O','O','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','O','O','O','O','O','O','O','O','O','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','O','O','O','O','O','O','O','O','O','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','O','O','O','O','O','O','O','O','O','W','W','W','W','W','W','W','W',
    'W','W','W','W','W','W','W','W','D','D','D','D','D','D','D','D','D','W','W','W','W','W','W','W','W',
};

static mapEntry crossroadsMap = {
    .layout = crossroadsLayout,
    .columns = 25,
    .rows = 21,
    .randFloatingStandardWalls = 2,
    .randFloatingBouncyWalls = 2,
    .randFloatingDeathWalls = 2,
    .hasSetFloatingWalls = true,
    .weaponPickups = 5,
    .maxSuddenDeathWalls = 5,
    .weaponSpawnWeights = {[FLAK_CANNON_WEAPON] = SPAWN_WEIGHT(MAX_PICKUP_SPAWN_WEIGHT)},
};
// clang-format on

static mapEntry *maps[] = {
    &boringMap,
    &prototypeArenaMap,
    &snipersMap,
    &roomsMap,
    &xArena,
    &crossBounce,
    &asteriskArena,
    &foamPitMap,
    &siegeMap,
    &pinballMap,
    &moatMap,
    &hourglassMap,
    &jawsMap,
    &cavernMap,
    &crossroadsMap,
};

void resetMap(iwEnv *e) {
    // if sudden death walls were placed, remove them
    if (e->suddenDeathWallsPlaced) {
        e->suddenDeathWallsPlaced = false;
        DEBUG_LOG("removing sudden death walls");
        // remove walls from the end of the array, sudden death walls
        // are added last
        for (int16_t i = cc_array_size(e->walls) - 1; i >= 0; i--) {
            wallEntity *wall = safe_array_get_at(e->walls, i);
            if (!wall->isSuddenDeath) {
                // if we reached the first non sudden death wall, we're done
                break;
            }
            cc_array_remove_last(e->walls, NULL);
            destroyWall(e, wall, true);
        }
    }

    // place floating walls with a set position if there are any
    const mapEntry *map = maps[e->mapIdx];
    if (!map->hasSetFloatingWalls) {
        return;
    }
    DEBUG_LOG("placing set floating walls");

    const uint8_t columns = map->columns;
    const uint8_t rows = map->rows;
    const char *layout = map->layout;
    uint16_t cellIdx = 0;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
            char cellType = layout[col + (row * columns)];

            enum entityType wallType;
            switch (cellType) {
            case 'w':
                wallType = STANDARD_WALL_ENTITY;
                break;
            case 'b':
                wallType = BOUNCY_WALL_ENTITY;
                break;
            case 'd':
                wallType = DEATH_WALL_ENTITY;
                break;
            default:
                cellIdx++;
                continue;
            }

            const mapCell *cell = (&e->cells[cellIdx]);
            createWall(e, cell->pos, FLOATING_WALL_THICKNESS, FLOATING_WALL_THICKNESS, cellIdx, wallType, true);
            cellIdx++;
        }
    }
}

void setupMap(iwEnv *e, const uint8_t mapIdx) {
    // reset the map if we're switching to the same map
    if (e->mapIdx == mapIdx) {
        resetMap(e);
        return;
    }

    // clear the old map
    for (size_t i = 0; i < cc_array_size(e->walls); i++) {
        wallEntity *wall = safe_array_get_at(e->walls, i);
        destroyWall(e, wall, false);
    }

    cc_array_remove_all(e->walls);
    e->numCells = 0;
    e->suddenDeathWallsPlaced = false;

    const uint8_t columns = maps[mapIdx]->columns;
    const uint8_t rows = maps[mapIdx]->rows;
    const char *layout = maps[mapIdx]->layout;

    e->mapIdx = mapIdx;
    e->map = maps[mapIdx];

    uint16_t cellIdx = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
            char cellType = layout[col + (row * columns)];
            enum entityType wallType;
            const float x = (col - ((columns - 1) * 0.5f)) * WALL_THICKNESS;
            const float y = (row - (rows - 1) * 0.5f) * WALL_THICKNESS;

            b2Vec2 pos = {.x = x, .y = y};
            mapCell *cell = &e->cells[cellIdx];
            cell->ent = NULL;
            cell->pos = pos;
            e->numCells++;

            bool floating = false;
            float thickness = WALL_THICKNESS;
            switch (cellType) {
            case 'O':
                cellIdx++;
                continue;
            case 'w':
                thickness = FLOATING_WALL_THICKNESS;
                floating = true;
            case 'W':
                wallType = STANDARD_WALL_ENTITY;
                break;
            case 'b':
                thickness = FLOATING_WALL_THICKNESS;
                floating = true;
            case 'B':
                wallType = BOUNCY_WALL_ENTITY;
                break;
            case 'd':
                thickness = FLOATING_WALL_THICKNESS;
                floating = true;
            case 'D':
                wallType = DEATH_WALL_ENTITY;
                break;
            default:
                ERRORF("unknown map layout cell %c", cellType);
            }

            entity *ent = createWall(e, pos, thickness, thickness, cellIdx, wallType, floating);
            if (!floating) {
                cell->ent = ent;
            }
            cellIdx++;
        }
    }
}

void computeMapBoundsAndQuadrants(iwEnv *e, mapEntry *map) {
    mapBounds bounds = {.min = {.x = FLT_MAX, .y = FLT_MAX}, .max = {.x = FLT_MIN, .y = FLT_MIN}};
    for (size_t i = 0; i < cc_array_size(e->walls); i++) {
        const wallEntity *wall = safe_array_get_at(e->walls, i);
        bounds.min.x = min(wall->pos.x - wall->extent.x + WALL_THICKNESS, bounds.min.x);
        bounds.min.y = min(wall->pos.y - wall->extent.y + WALL_THICKNESS, bounds.min.y);
        bounds.max.x = max(wall->pos.x + wall->extent.x - WALL_THICKNESS, bounds.max.x);
        bounds.max.y = max(wall->pos.y + wall->extent.y - WALL_THICKNESS, bounds.max.y);
    }
    map->bounds = bounds;
    map->spawnQuads[0] = (mapBounds){
        .min = (b2Vec2){
            .x = map->bounds.min.x + WALL_THICKNESS,
            .y = map->bounds.min.y + WALL_THICKNESS,
        },
        .max = (b2Vec2){
            .x = 0.0f,
            .y = 0.0f,
        }
    };
    map->spawnQuads[1] = (mapBounds){
        .min = (b2Vec2){
            .x = 0.0f,
            .y = map->bounds.min.y + WALL_THICKNESS,
        },
        .max = (b2Vec2){
            .x = map->bounds.max.x - WALL_THICKNESS,
            .y = 0.0f,
        }
    };
    map->spawnQuads[2] = (mapBounds){
        .min = (b2Vec2){
            .x = map->bounds.min.x + WALL_THICKNESS,
            .y = 0.0f,
        },
        .max = (b2Vec2){
            .x = 0.0f,
            .y = map->bounds.max.y - WALL_THICKNESS,
        }
    };
    map->spawnQuads[3] = (mapBounds){
        .min = (b2Vec2){
            .x = 0.0f,
            .y = 0.0f,
        },
        .max = (b2Vec2){
            .x = map->bounds.max.x - WALL_THICKNESS,
            .y = map->bounds.max.y - WALL_THICKNESS,
        }
    };

    // count the cells overlapping each quadrant, treating its bounds as
    // half open since a position exactly on the max edge is vanishingly rare
    const float halfWidth = ((float)map->columns * WALL_THICKNESS) / 2.0f;
    const float halfHeight = ((float)map->rows * WALL_THICKNESS) / 2.0f;
    map->quadCells = UINT16_MAX;
    for (uint8_t i = 0; i < 4; i++) {
        const mapBounds q = map->spawnQuads[i];
        const int16_t firstCol = (int16_t)floorf((q.min.x + halfWidth) / WALL_THICKNESS);
        const int16_t lastCol = (int16_t)ceilf((q.max.x + halfWidth) / WALL_THICKNESS) - 1;
        const int16_t firstRow = (int16_t)floorf((q.min.y + halfHeight) / WALL_THICKNESS);
        const int16_t lastRow = (int16_t)ceilf((q.max.y + halfHeight) / WALL_THICKNESS) - 1;
        const uint16_t cells = (lastCol - firstCol + 1) * (lastRow - firstRow + 1);
        map->quadCells = min(map->quadCells, cells);
    }
}

bool posValidDroneSpawnPoint(const iwEnv *e, const b2Vec2 pos) {
    const b2QueryFilter filter = {
        .categoryBits = DRONE_SHAPE,
        .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE,
    };
    droneEntity dummyDrone = {.pos = pos};
    const entity ent = {.type = DRONE_ENTITY, .entity = &dummyDrone};
    const enum entityType deathWallType = DEATH_WALL_ENTITY;

    if (isOverlappingCircleInLineOfSight(e, &ent, pos, DRONE_DEATH_WALL_SPAWN_DISTANCE, filter, &deathWallType)) {
        return false;
    }
    if (isOverlappingAABB(e, pos, DRONE_WALL_SPAWN_DISTANCE, filter)) {
        return false;
    }

    return true;
}

// applies a map's pickup spawn weight overrides to the default weights
void resolvePickupSpawnWeights(const uint8_t mapIdx) {
    mapEntry *map = maps[mapIdx];
    float totalWeight = 0.0f;
    for (uint8_t i = 0; i < NUM_WEAPONS; i++) {
        float weight = weaponInfos[i]->spawnWeight;
        if (map->weaponSpawnWeights[i].set) {
            if (i == UNSET_WEAPON || i == STANDARD_WEAPON) {
                ERRORF("map %d overrides the spawn weight of weapon %d which can't be spawned", mapIdx, i);
            }
            weight = map->weaponSpawnWeights[i].weight;
        }
        if (weight < 0.0f || weight > MAX_PICKUP_SPAWN_WEIGHT) {
            ERRORF("map %d weapon %d spawn weight %f is outside of [0, %f]", mapIdx, i, weight, MAX_PICKUP_SPAWN_WEIGHT);
        }
        map->weaponWeights[i] = weight;
        totalWeight += weight;
    }

    float crateWeight = AMMO_CRATE_SPAWN_WEIGHT;
    if (map->crateSpawnWeight.set) {
        crateWeight = map->crateSpawnWeight.weight;
    }
    if (crateWeight < 0.0f || crateWeight > MAX_PICKUP_SPAWN_WEIGHT) {
        ERRORF("map %d ammo crate spawn weight %f is outside of [0, %f]", mapIdx, crateWeight, MAX_PICKUP_SPAWN_WEIGHT);
    }
    map->crateWeight = crateWeight;
    totalWeight += crateWeight;

    if (map->weaponPickups != 0 && totalWeight == 0.0f) {
        ERRORF("map %d spawns pickups but every pickup spawn weight is 0", mapIdx);
    }
}

void initMaps(iwEnv *e) {
    for (uint8_t i = 0; i < NUM_MAPS; i++) {
        setupMap(e, i);
        mapEntry *map = maps[i];

        resolvePickupSpawnWeights(i);
        computeMapBoundsAndQuadrants(e, map);

        const uint32_t cellCount = (uint32_t)map->columns * map->rows;
        bool *droneSpawns = fastCalloc(cellCount, sizeof(bool));
        float *packedLayout = fastCalloc(cellCount, sizeof(float));
        nearWall *nearestWalls = fastCalloc(MAX_NEAREST_WALLS * cellCount, sizeof(nearWall));

        // Precompute the flow field to every destination cell. This used to
        // be a lazily filled per-env cache, which cost ~2MiB per env (over
        // 8GiB across a 4096 env rollout) and re-ran the BFS on every miss,
        // including for unreachable destinations where it could never cache
        // a result. The field only depends on the static wall layout, so one
        // read-only copy per map serves every env.
        uint8_t *paths = fastMalloc(cellCount * cellCount * sizeof(uint8_t));
        memset(paths, UINT8_MAX, cellCount * cellCount * sizeof(uint8_t));
        int8_t (*pathBuffer)[3] = fastCalloc(8 * cellCount, 3 * sizeof(int8_t));
        for (uint16_t dst = 0; dst < cellCount; dst++) {
            pathfindBFS(e, paths + ((size_t)dst * cellCount), pathBuffer, dst);
        }
        // cells the BFS never reached (walls, or open cells walled off from
        // the destination) are marked unreachable so callers stop retrying
        for (size_t i = 0; i < (size_t)cellCount * cellCount; i++) {
            if (paths[i] == UINT8_MAX) {
                paths[i] = 8;
            }
        }
        fastFree(pathBuffer);
        map->paths = paths;

        for (uint16_t i = 0; i < e->numCells; i++) {
            const mapCell *cell = (&e->cells[i]);

            // precompute packed map layout
            if (cell->ent != NULL) {
                packedLayout[i] = (float)(cell->ent->type + 1);
                continue;
            } else {
                // precompute valid cells for drones to spawn
                droneSpawns[i] = posValidDroneSpawnPoint(e, cell->pos);
            }

            // find nearest walls for each empty cell
            uint16_t wallIdx = 0;
            nearWall walls[map->columns * map->rows];
            memset(walls, 0x0, map->columns * map->rows * sizeof(nearWall));
            for (uint16_t j = 0; j < e->numCells; j++) {
                const mapCell *c = (&e->cells[j]);
                if (c->ent == NULL) {
                    continue;
                }

                walls[wallIdx].wallIdx = wallIdx;
                walls[wallIdx].type = c->ent->type;
                walls[wallIdx].pos = c->pos;
                walls[wallIdx].distanceSquared = b2DistanceSquared(cell->pos, c->pos);
                wallIdx++;
            }
            insertionSortWalls(walls, wallIdx);

            const uint32_t startIdx = i * MAX_NEAREST_WALLS;
            memcpy(nearestWalls + startIdx, walls, MAX_NEAREST_WALLS * sizeof(nearWall));
        }
        map->droneSpawns = droneSpawns;
        map->packedLayout = packedLayout;
        map->nearestWalls = nearestWalls;

        // clear floating walls from the map
        for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
            wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
            destroyWall(e, wall, false);
        }
        cc_array_remove_all(e->floatingWalls);
    }

    e->mapIdx = -1;
}

void destroyMaps() {
    for (uint8_t i = 0; i < NUM_MAPS; i++) {
        mapEntry *map = maps[i];
        fastFree(map->droneSpawns);
        fastFree(map->packedLayout);
        fastFree(map->nearestWalls);
        fastFree(map->paths);
    }
}

void placeRandFloatingWall(iwEnv *e, const enum entityType wallType) {
    b2Vec2 pos;
    if (!findOpenPos(e, FLOATING_WALL_SHAPE, &pos, -1)) {
        ERROR("failed to find open position for floating wall");
    }
    int16_t cellIdx = entityPosToCellIdx(e, pos);
    createWall(e, pos, FLOATING_WALL_THICKNESS, FLOATING_WALL_THICKNESS, cellIdx, wallType, true);
}

void placeRandFloatingWalls(iwEnv *e, const int mapIdx) {
    for (int i = 0; i < maps[mapIdx]->randFloatingStandardWalls; i++) {
        placeRandFloatingWall(e, STANDARD_WALL_ENTITY);
    }
    for (int i = 0; i < maps[mapIdx]->randFloatingBouncyWalls; i++) {
        placeRandFloatingWall(e, BOUNCY_WALL_ENTITY);
    }
    for (int i = 0; i < maps[mapIdx]->randFloatingDeathWalls; i++) {
        placeRandFloatingWall(e, DEATH_WALL_ENTITY);
    }
}

#endif
