// Impulse Wars observation layout for the embedding encoder, shared by the
// CUDA encoder (impulse_wars.cu) and the CPU net (impulse_wars_net.h).
// Mirrors the obs offsets in settings.h, which the CUDA build can't include
// (it pulls in box2d). impulse_wars_net.h static asserts the two agree.
//
// Wall type slots hold 0 (empty) or wall type + 1, weapon type slots hold
// 0 (no entity) or weapon type + 1. Every other obs value is a continuous
// scalar that is passed through to the projection unchanged.
#ifndef IMPULSE_WARS_OBS_H
#define IMPULSE_WARS_OBS_H

#ifdef __CUDACC__
#define IW_FN __host__ __device__ constexpr inline
#else
#define IW_FN static inline
#endif

#define IW_MAP_CELLS 121
#define IW_NUM_NEAR_WALLS 4
#define IW_NEAR_WALL_SIZE 3
#define IW_NUM_FLOATING_WALLS 4
#define IW_FLOATING_WALL_SIZE 6
#define IW_NUM_PICKUPS 3
#define IW_PICKUP_SIZE 3
#define IW_NUM_PROJECTILES 30
#define IW_PROJECTILE_SIZE 6
#define IW_ENEMY_DRONE_SIZE 28

#define IW_NEAR_WALL_OFFSET IW_MAP_CELLS
#define IW_FLOATING_WALL_OFFSET (IW_NEAR_WALL_OFFSET + IW_NUM_NEAR_WALLS * IW_NEAR_WALL_SIZE)
#define IW_PICKUP_OFFSET (IW_FLOATING_WALL_OFFSET + IW_NUM_FLOATING_WALLS * IW_FLOATING_WALL_SIZE)
#define IW_PROJECTILE_OFFSET (IW_PICKUP_OFFSET + IW_NUM_PICKUPS * IW_PICKUP_SIZE)
#define IW_ENEMY_DRONE_OFFSET (IW_PROJECTILE_OFFSET + IW_NUM_PROJECTILES * IW_PROJECTILE_SIZE)
// obs size with zero enemy drones; each enemy adds IW_ENEMY_DRONE_SIZE
#define IW_BASE_OBS_SIZE 382
#define IW_NUM_ENEMIES ((OBS_SIZE - IW_BASE_OBS_SIZE) / IW_ENEMY_DRONE_SIZE)
#define IW_DRONE_OFFSET (IW_ENEMY_DRONE_OFFSET + IW_NUM_ENEMIES * IW_ENEMY_DRONE_SIZE)

// empty + standard, bouncy, death walls
#define IW_WALL_VOCAB 4
// none + 10 weapons
#define IW_WEAPON_VOCAB 11
#define IW_WALL_DIM 4
#define IW_WEAPON_DIM 8

#define IW_NUM_WALL_SLOTS (IW_MAP_CELLS + IW_NUM_NEAR_WALLS + IW_NUM_FLOATING_WALLS)
// pickups, projectiles, enemy drones, own drone
#define IW_NUM_WEAPON_SLOTS (IW_NUM_PICKUPS + IW_NUM_PROJECTILES + IW_NUM_ENEMIES + 1)
#define IW_NUM_SCALARS (OBS_SIZE - IW_NUM_WALL_SLOTS - IW_NUM_WEAPON_SLOTS)

// concat layout: [wall slot embeds | weapon slot embeds | scalars]
#define IW_WALL_COLS (IW_NUM_WALL_SLOTS * IW_WALL_DIM)
#define IW_WEAPON_COLS (IW_NUM_WEAPON_SLOTS * IW_WEAPON_DIM)
#define IW_EMB_COLS (IW_WALL_COLS + IW_WEAPON_COLS)
#define IW_CONCAT (IW_EMB_COLS + IW_NUM_SCALARS)

#define IW_WALL_EMB_N (IW_WALL_VOCAB * IW_WALL_DIM)
#define IW_WEAPON_EMB_N (IW_WEAPON_VOCAB * IW_WEAPON_DIM)
#define IW_EMB_N (IW_WALL_EMB_N + IW_WEAPON_EMB_N)

// obs index of wall type slot j
IW_FN int iw_wall_slot_idx(int j) {
    if (j < IW_MAP_CELLS) {
        return j;
    }
    j -= IW_MAP_CELLS;
    if (j < IW_NUM_NEAR_WALLS) {
        return IW_NEAR_WALL_OFFSET + j * IW_NEAR_WALL_SIZE;
    }
    j -= IW_NUM_NEAR_WALLS;
    return IW_FLOATING_WALL_OFFSET + j * IW_FLOATING_WALL_SIZE;
}

// obs index of weapon type slot j
IW_FN int iw_weapon_slot_idx(int j) {
    if (j < IW_NUM_PICKUPS) {
        return IW_PICKUP_OFFSET + j * IW_PICKUP_SIZE;
    }
    j -= IW_NUM_PICKUPS;
    if (j < IW_NUM_PROJECTILES) {
        // first value is whether the projectile is the agent's own
        return IW_PROJECTILE_OFFSET + j * IW_PROJECTILE_SIZE + 1;
    }
    j -= IW_NUM_PROJECTILES;
    if (j < IW_NUM_ENEMIES) {
        // first value is whether the enemy is a teammate
        return IW_ENEMY_DRONE_OFFSET + j * IW_ENEMY_DRONE_SIZE + 1;
    }
    return IW_DRONE_OFFSET;
}

// whether obs index i is a wall or weapon type slot
IW_FN int iw_is_type_slot(int i) {
    if (i < IW_MAP_CELLS) {
        return 1;
    }
    if (i < IW_FLOATING_WALL_OFFSET) {
        return (i - IW_NEAR_WALL_OFFSET) % IW_NEAR_WALL_SIZE == 0;
    }
    if (i < IW_PICKUP_OFFSET) {
        return (i - IW_FLOATING_WALL_OFFSET) % IW_FLOATING_WALL_SIZE == 0;
    }
    if (i < IW_PROJECTILE_OFFSET) {
        return (i - IW_PICKUP_OFFSET) % IW_PICKUP_SIZE == 0;
    }
    if (i < IW_ENEMY_DRONE_OFFSET) {
        return (i - IW_PROJECTILE_OFFSET) % IW_PROJECTILE_SIZE == 1;
    }
    if (i < IW_DRONE_OFFSET) {
        return (i - IW_ENEMY_DRONE_OFFSET) % IW_ENEMY_DRONE_SIZE == 1;
    }
    return i == IW_DRONE_OFFSET;
}

// clamps a type value read from the obs to [0, vocab)
IW_FN int iw_type_idx(float v, int vocab) {
    int idx = (int)(v + 0.5f);
    if (idx < 0) {
        return 0;
    }
    if (idx >= vocab) {
        return vocab - 1;
    }
    return idx;
}

#endif
