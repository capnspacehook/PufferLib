// Impulse Wars observation layout for the encoder, shared by the CUDA
// encoder (impulse_wars.cu) and the CPU net (impulse_wars_net.h). Mirrors
// the obs offsets in settings.h, which the CUDA build can't include (it
// pulls in box2d). impulse_wars_net.h asserts the two agree.
//
// The map grid holds packed cell codes (see GRID_* in settings.h) that are
// unpacked into one-hot channels and fed to a 3 layer CNN. The agent's own
// projectile slots, other drones' projectile slots and floating wall slots
// are records for the own projectile, enemy projectile and floating wall
// branches: a shared MLP per branch over every slot, max pooled over the
// non-empty ones. Wall type slots hold 0 (empty) or wall
// type + 1, weapon type slots hold a weapon type (0 is UNSET_WEAPON: no
// entity or no weapon), utility type slots hold a utility type (0 is
// UNSET_UTILITY), pickup type slots hold a weapon type, one past the last
// weapon for an ammo crate, or a utility type past that for a utility, ray
// hit type slots hold a RAY_HIT_* value; these are one-hot encoded with
// type 0 as all zeros. Every other obs value is a continuous scalar that is passed
// through to the projection unchanged. The encoder adds the number of
// enemy projectile slots in use to the concat, as max pooling loses it.
#ifndef IMPULSE_WARS_OBS_H
#define IMPULSE_WARS_OBS_H

#ifdef __CUDACC__
#define IW_FN __host__ __device__ constexpr inline
// for helpers that call math functions, which aren't constexpr
#define IW_MATH_FN __host__ __device__ inline
#else
#define IW_FN static inline
#define IW_MATH_FN static inline
#endif

#define IW_MAP_CELLS 121
#define IW_NUM_NEAR_WALLS 4
#define IW_NEAR_WALL_SIZE 3
#define IW_NUM_FLOATING_WALLS 8
#define IW_FLOATING_WALL_SIZE 7
#define IW_NUM_PICKUPS 3
#define IW_PICKUP_SIZE 3
// the agent's own projectile slots, then other drones'
#define IW_NUM_OWN_PROJECTILES 6
#define IW_NUM_ENEMY_PROJECTILES 10
#define IW_NUM_PROJECTILES (IW_NUM_OWN_PROJECTILES + IW_NUM_ENEMY_PROJECTILES)
// weapon type, relative position, relative velocity, distance
#define IW_PROJECTILE_SIZE 6
#define IW_RAY_SIZE 2
// every drone block starts with its weapon: type, ammo, cooldown, charge
// and full charge hold time, then its utility: type, ammo, regen progress,
// active count, the grapple's state, the mine's state and each portal's
// state
#define IW_WEAPON_SIZE 5
#define IW_UTILITY_SIZE 30
// offsets of the weapon and utility type slots in a drone block
#define IW_DRONE_WEAPON_TYPE 0
#define IW_DRONE_UTILITY_TYPE IW_WEAPON_SIZE
// ray to the enemy, last in each enemy drone block
#define IW_ENEMY_RAY_IDX 60
#define IW_ENEMY_DRONE_SIZE (IW_ENEMY_RAY_IDX + IW_RAY_SIZE)
// aim, velocity and recoil rays, last in the agent's drone block
#define IW_DRONE_RAY_IDX 60
#define IW_NUM_DRONE_RAYS 3
#define IW_DRONE_SIZE (IW_DRONE_RAY_IDX + IW_NUM_DRONE_RAYS * IW_RAY_SIZE)

#define IW_NEAR_WALL_OFFSET IW_MAP_CELLS
#define IW_FLOATING_WALL_OFFSET (IW_NEAR_WALL_OFFSET + IW_NUM_NEAR_WALLS * IW_NEAR_WALL_SIZE)
#define IW_PICKUP_OFFSET (IW_FLOATING_WALL_OFFSET + IW_NUM_FLOATING_WALLS * IW_FLOATING_WALL_SIZE)
#define IW_PROJECTILE_OFFSET (IW_PICKUP_OFFSET + IW_NUM_PICKUPS * IW_PICKUP_SIZE)
#define IW_ENEMY_PROJECTILE_OFFSET (IW_PROJECTILE_OFFSET + IW_NUM_OWN_PROJECTILES * IW_PROJECTILE_SIZE)
#define IW_ENEMY_DRONE_OFFSET (IW_PROJECTILE_OFFSET + IW_NUM_PROJECTILES * IW_PROJECTILE_SIZE)
// obs size with zero enemy drones; each enemy adds IW_ENEMY_DRONE_SIZE
#define IW_BASE_OBS_SIZE 363
#define IW_NUM_ENEMIES ((OBS_SIZE - IW_BASE_OBS_SIZE) / IW_ENEMY_DRONE_SIZE)
#define IW_DRONE_OFFSET (IW_ENEMY_DRONE_OFFSET + IW_NUM_ENEMIES * IW_ENEMY_DRONE_SIZE)

// empty + standard, bouncy, death walls
#define IW_WALL_VOCAB 4
// none + 12 weapons, for drone weapons and projectiles
#define IW_WEAPON_VOCAB 13
// none + grapple, mines, portals
#define IW_UTILITY_VOCAB 4
// weapon vocab + ammo crate + grapple, mines, portals
#define IW_PICKUP_VOCAB (IW_WEAPON_VOCAB + IW_UTILITY_VOCAB)
// none + static standard, bouncy, death walls + floating standard,
// bouncy, death walls + drone
#define IW_RAY_VOCAB 8

// map grid CNN: cell codes are unpacked into IW_GRID_CHANNELS one-hot
// planes padded by 1 cell, then 3 unpadded convs with ReLU:
// 10x13x13 -> 3x3 s1 -> 32x11x11 -> 3x3 s2 -> 64x5x5 -> 3x3 s1 -> 64x3x3
// The CUDA encoder keeps conv activations NHWC, as its matmul rows, so im2col
// columns and conv weights are (kh, kw, channel) with the channel fastest,
// and the CNN block of the concat is the 3x3 output flattened NHWC
#define IW_GRID_SIZE 11
#define IW_GRID_PAD 1
// static wall id bits, floating wall type bits, pickup and enemy bits
#define IW_GRID_STATIC_MASK 0x7
#define IW_GRID_FLOATING_SHIFT 3
#define IW_GRID_PICKUP_BIT (1 << 5)
#define IW_GRID_ENEMY_BIT (1 << 6)
#define IW_GRID_CODE_MAX 127
// channels: static empty, standard, bouncy, death, out of bounds, floating
// standard, bouncy, death, pickup, enemy; the empty channel stands in for
// a conv bias
#define IW_GRID_STATIC_IDS 5
#define IW_GRID_CHANNELS 10

#define IW_C1_IN (IW_GRID_SIZE + 2 * IW_GRID_PAD)
#define IW_C1_IC IW_GRID_CHANNELS
#define IW_C1_OC 32
#define IW_C1_K 3
#define IW_C1_S 1
#define IW_C1_OUT ((IW_C1_IN - IW_C1_K) / IW_C1_S + 1)

#define IW_C2_IN IW_C1_OUT
#define IW_C2_IC IW_C1_OC
#define IW_C2_OC 64
#define IW_C2_K 3
#define IW_C2_S 2
#define IW_C2_OUT ((IW_C2_IN - IW_C2_K) / IW_C2_S + 1)

#define IW_C3_IN IW_C2_OUT
#define IW_C3_IC IW_C2_OC
#define IW_C3_OC 64
#define IW_C3_K 3
#define IW_C3_S 1
#define IW_C3_OUT ((IW_C3_IN - IW_C3_K) / IW_C3_S + 1)

// im2col widths and output positions per sample
#define IW_C1_COL_W (IW_C1_IC * IW_C1_K * IW_C1_K)
#define IW_C2_COL_W (IW_C2_IC * IW_C2_K * IW_C2_K)
#define IW_C3_COL_W (IW_C3_IC * IW_C3_K * IW_C3_K)
#define IW_C1_SPATIAL (IW_C1_OUT * IW_C1_OUT)
#define IW_C2_SPATIAL (IW_C2_OUT * IW_C2_OUT)
#define IW_C3_SPATIAL (IW_C3_OUT * IW_C3_OUT)
// conv weights are (OC, IC * K * K)
#define IW_C1_W_N (IW_C1_OC * IW_C1_COL_W)
#define IW_C2_W_N (IW_C2_OC * IW_C2_COL_W)
#define IW_C3_W_N (IW_C3_OC * IW_C3_COL_W)
#define IW_CONV_FLAT (IW_C3_OC * IW_C3_SPATIAL)

// near walls; the map grid goes to the CNN and floating walls to the
// floating wall branch
#define IW_NUM_WALL_SLOTS IW_NUM_NEAR_WALLS
// pickup type slots
#define IW_NUM_PICKUP_SLOTS IW_NUM_PICKUPS
// the weapon and utility of the enemy drones and own drone, one slot of
// each per drone; projectiles go to the projectile branch
#define IW_NUM_WEAPON_SLOTS (IW_NUM_ENEMIES + 1)
#define IW_NUM_UTILITY_SLOTS (IW_NUM_ENEMIES + 1)
// enemy drone rays, own drone rays
#define IW_NUM_RAY_SLOTS (IW_NUM_ENEMIES + IW_NUM_DRONE_RAYS)
#define IW_NUM_SCALARS (OBS_SIZE - IW_MAP_CELLS - IW_NUM_WALL_SLOTS - IW_NUM_PICKUP_SLOTS - IW_NUM_WEAPON_SLOTS - IW_NUM_UTILITY_SLOTS - IW_NUM_RAY_SLOTS - IW_NUM_FLOATING_WALLS * IW_FLOATING_WALL_SIZE - IW_NUM_PROJECTILES * IW_PROJECTILE_SIZE)

// concat layout: [CNN | wall one-hots | pickup one-hots | weapon one-hots |
// utility one-hots | ray one-hots | scalars | enemy projectile count |
// zero padding]
#define IW_WALL_COLS (IW_NUM_WALL_SLOTS * (IW_WALL_VOCAB - 1))
#define IW_PICKUP_COLS (IW_NUM_PICKUP_SLOTS * (IW_PICKUP_VOCAB - 1))
#define IW_WEAPON_COLS (IW_NUM_WEAPON_SLOTS * (IW_WEAPON_VOCAB - 1))
#define IW_UTILITY_COLS (IW_NUM_UTILITY_SLOTS * (IW_UTILITY_VOCAB - 1))
#define IW_RAY_COLS (IW_NUM_RAY_SLOTS * (IW_RAY_VOCAB - 1))
#define IW_ONEHOT_COLS (IW_WALL_COLS + IW_PICKUP_COLS + IW_WEAPON_COLS + IW_UTILITY_COLS + IW_RAY_COLS)
#define IW_SCALAR_COLS_END (IW_CONV_FLAT + IW_ONEHOT_COLS + IW_NUM_SCALARS)
#define IW_CONCAT_USED (IW_SCALAR_COLS_END + 1)
// the record widths below are padded the same way: zero columns up to a
// multiple of 8 keep bf16 rows 16 byte aligned, otherwise cuBLAS falls off
// its fast kernels
#define IW_CONCAT ((IW_CONCAT_USED + 7) & ~7)

// projectile branches, one for the agent's own projectiles and one for
// other drones': every projectile slot is a record of weapon type one-hot,
// relative position, relative velocity and distance, run through the
// branch's Linear -> GELU -> Linear and max pooled over the non-empty
// records into the encoder output
#define IW_PROJ_SCALARS (IW_PROJECTILE_SIZE - 1)
#define IW_PROJ_USED_FEATURES ((IW_WEAPON_VOCAB - 1) + IW_PROJ_SCALARS)
#define IW_PROJ_FEATURES ((IW_PROJ_USED_FEATURES + 7) & ~7)
#define IW_PROJ_HIDDEN 16
#define IW_PROJ_L1_N (IW_PROJ_HIDDEN * IW_PROJ_FEATURES)

// floating wall branch: as the projectile branch, every floating wall slot
// is a record of wall type one-hot, relative position, angle sin and cos
// and relative velocity
#define IW_FW_SCALARS (IW_FLOATING_WALL_SIZE - 1)
#define IW_FW_USED_FEATURES ((IW_WALL_VOCAB - 1) + IW_FW_SCALARS)
#define IW_FW_FEATURES ((IW_FW_USED_FEATURES + 7) & ~7)
#define IW_FW_HIDDEN 16
#define IW_FW_L1_N (IW_FW_HIDDEN * IW_FW_FEATURES)

// obs index of wall type slot j
IW_FN int iw_wall_slot_idx(int j) {
    return IW_NEAR_WALL_OFFSET + j * IW_NEAR_WALL_SIZE;
}

// obs index of pickup type slot j
IW_FN int iw_pickup_slot_idx(int j) {
    return IW_PICKUP_OFFSET + j * IW_PICKUP_SIZE;
}

// obs index of the type slot at offset k of drone j's block, the enemy
// drones then the own drone
IW_FN int iw_drone_slot_idx(int j, int k) {
    if (j < IW_NUM_ENEMIES) {
        return IW_ENEMY_DRONE_OFFSET + j * IW_ENEMY_DRONE_SIZE + k;
    }
    return IW_DRONE_OFFSET + k;
}

// obs index of weapon type slot j
IW_FN int iw_weapon_slot_idx(int j) {
    return iw_drone_slot_idx(j, IW_DRONE_WEAPON_TYPE);
}

// obs index of utility type slot j
IW_FN int iw_utility_slot_idx(int j) {
    return iw_drone_slot_idx(j, IW_DRONE_UTILITY_TYPE);
}

// whether offset k into a drone block is its weapon or utility type slot
IW_FN int iw_is_drone_type(int k) {
    return k == IW_DRONE_WEAPON_TYPE || k == IW_DRONE_UTILITY_TYPE;
}

// obs index of ray hit type slot j
IW_FN int iw_ray_slot_idx(int j) {
    if (j < IW_NUM_ENEMIES) {
        return IW_ENEMY_DRONE_OFFSET + j * IW_ENEMY_DRONE_SIZE + IW_ENEMY_RAY_IDX;
    }
    j -= IW_NUM_ENEMIES;
    return IW_DRONE_OFFSET + IW_DRONE_RAY_IDX + j * IW_RAY_SIZE;
}

// whether obs index i is a continuous scalar that goes into the concat,
// rather than a map grid cell, a wall, pickup, weapon, utility or ray hit
// type slot or part of a floating wall or projectile slot
IW_FN int iw_is_scalar(int i) {
    if (i < IW_MAP_CELLS) {
        return 0;
    }
    if (i < IW_FLOATING_WALL_OFFSET) {
        return (i - IW_NEAR_WALL_OFFSET) % IW_NEAR_WALL_SIZE != 0;
    }
    if (i < IW_PICKUP_OFFSET) {
        return 0;
    }
    if (i < IW_PROJECTILE_OFFSET) {
        return (i - IW_PICKUP_OFFSET) % IW_PICKUP_SIZE != 0;
    }
    if (i < IW_ENEMY_DRONE_OFFSET) {
        return 0;
    }
    if (i < IW_DRONE_OFFSET) {
        const int k = (i - IW_ENEMY_DRONE_OFFSET) % IW_ENEMY_DRONE_SIZE;
        return !iw_is_drone_type(k) && k != IW_ENEMY_RAY_IDX;
    }
    const int k = i - IW_DRONE_OFFSET;
    if (k >= IW_DRONE_RAY_IDX && k < IW_DRONE_SIZE) {
        return (k - IW_DRONE_RAY_IDX) % IW_RAY_SIZE != 0;
    }
    return !iw_is_drone_type(k);
}

// a one-hot concat column: the type slot it reads and the class it is hot
// for, type values are class + 1
typedef struct iwOneHotCol {
    int obsIdx;
    int cls;
    int vocab;
} iwOneHotCol;

// the type slot and class of one-hot column c, counted from the start of
// the one-hot block of the concat
IW_FN iwOneHotCol iw_onehot_col(int c) {
    if (c < IW_WALL_COLS) {
        iwOneHotCol col = {iw_wall_slot_idx(c / (IW_WALL_VOCAB - 1)), c % (IW_WALL_VOCAB - 1), IW_WALL_VOCAB};
        return col;
    }
    c -= IW_WALL_COLS;
    if (c < IW_PICKUP_COLS) {
        iwOneHotCol col = {iw_pickup_slot_idx(c / (IW_PICKUP_VOCAB - 1)), c % (IW_PICKUP_VOCAB - 1), IW_PICKUP_VOCAB};
        return col;
    }
    c -= IW_PICKUP_COLS;
    if (c < IW_WEAPON_COLS) {
        iwOneHotCol col = {iw_weapon_slot_idx(c / (IW_WEAPON_VOCAB - 1)), c % (IW_WEAPON_VOCAB - 1), IW_WEAPON_VOCAB};
        return col;
    }
    c -= IW_WEAPON_COLS;
    if (c < IW_UTILITY_COLS) {
        iwOneHotCol col = {iw_utility_slot_idx(c / (IW_UTILITY_VOCAB - 1)), c % (IW_UTILITY_VOCAB - 1), IW_UTILITY_VOCAB};
        return col;
    }
    c -= IW_UTILITY_COLS;
    iwOneHotCol col = {iw_ray_slot_idx(c / (IW_RAY_VOCAB - 1)), c % (IW_RAY_VOCAB - 1), IW_RAY_VOCAB};
    return col;
}

// value of one-hot channel ch of a map grid cell code read from the obs
IW_FN float iw_grid_channel(float v, int ch) {
    int code = (int)(v + 0.5f);
    if (code < 0 || code > IW_GRID_CODE_MAX) {
        code = 0;
    }
    if (ch < IW_GRID_STATIC_IDS) {
        return (code & IW_GRID_STATIC_MASK) == ch ? 1.0f : 0.0f;
    }
    if (ch < IW_GRID_STATIC_IDS + 3) {
        return ((code >> IW_GRID_FLOATING_SHIFT) & 0x3) == ch - IW_GRID_STATIC_IDS + 1 ? 1.0f : 0.0f;
    }
    if (ch == IW_GRID_STATIC_IDS + 3) {
        return (code & IW_GRID_PICKUP_BIT) != 0 ? 1.0f : 0.0f;
    }
    return (code & IW_GRID_ENEMY_BIT) != 0 ? 1.0f : 0.0f;
}

// map grid cell read by conv1 output position s at im2col column c, or -1
// for padding
IW_FN int iw_c1_cell(int s, int c) {
    const int kk = c / IW_C1_IC;
    const int row = (s / IW_C1_OUT) * IW_C1_S + kk / IW_C1_K - IW_GRID_PAD;
    const int col = (s % IW_C1_OUT) * IW_C1_S + kk % IW_C1_K - IW_GRID_PAD;
    if (row < 0 || row >= IW_GRID_SIZE || col < 0 || col >= IW_GRID_SIZE) {
        return -1;
    }
    return row * IW_GRID_SIZE + col;
}

// conv1 im2col value at column c, given the grid cell iw_c1_cell returned
// for it and that cell's obs value
IW_FN float iw_c1_col_value(float cellValue, int cell, int c) {
    if (cell < 0) {
        return 0.0f;
    }
    return iw_grid_channel(cellValue, c % IW_C1_IC);
}

// index into one sample's NHWC (in, in, ic) input of an unpadded square
// conv for output position s and im2col column c
IW_FN int iw_im2col_src(int s, int c, int ic, int in, int out, int k, int stride) {
    const int kk = c / ic;
    const int ih = (s / out) * stride + kk / k;
    const int iw = (s % out) * stride + kk % k;
    return (ih * in + iw) * ic + c % ic;
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

// value of one-hot column col given the obs value of its type slot
IW_FN float iw_onehot_value(float v, iwOneHotCol col) {
    return iw_type_idx(v, col.vocab) == col.cls + 1 ? 1.0f : 0.0f;
}

// obs index of projectile slot r
IW_FN int iw_proj_slot_idx(int r) {
    return IW_PROJECTILE_OFFSET + r * IW_PROJECTILE_SIZE;
}

// slot field read for projectile record feature f: the weapon type, or
// relative position x, y, velocity x, y and distance; padding features
// read the weapon type but are always 0
IW_FN int iw_proj_field(int f) {
    if (f < IW_WEAPON_VOCAB - 1 || f >= IW_PROJ_USED_FEATURES) {
        return 0;
    }
    return 1 + f - (IW_WEAPON_VOCAB - 1);
}

// projectile record feature f given the slot field iw_proj_field(f) reads
IW_FN float iw_proj_feature(float v, int f) {
    if (f >= IW_PROJ_USED_FEATURES) {
        return 0.0f;
    }
    if (f < IW_WEAPON_VOCAB - 1) {
        return iw_type_idx(v, IW_WEAPON_VOCAB) == f + 1 ? 1.0f : 0.0f;
    }
    return v;
}

// whether a projectile record is a projectile rather than an empty slot,
// given the weapon type value of its slot
IW_FN int iw_proj_active(float weaponType) {
    return iw_type_idx(weaponType, IW_WEAPON_VOCAB) != 0;
}

// the enemy projectile count column of the concat is the enemy projectile
// slots in use as a fraction of the slots: iw_proj_active over the weapon
// types of the slots iw_proj_slot_idx(IW_NUM_OWN_PROJECTILES + r)

// obs index of floating wall slot r
IW_FN int iw_fw_slot_idx(int r) {
    return IW_FLOATING_WALL_OFFSET + r * IW_FLOATING_WALL_SIZE;
}

// slot field read for floating wall record feature f: the wall type, or
// relative position x, y, angle sin, cos and relative velocity x, y;
// padding features read the wall type but are always 0
IW_FN int iw_fw_field(int f) {
    if (f < IW_WALL_VOCAB - 1 || f >= IW_FW_USED_FEATURES) {
        return 0;
    }
    return 1 + f - (IW_WALL_VOCAB - 1);
}

// floating wall record feature f given the slot field iw_fw_field(f) reads
IW_FN float iw_fw_feature(float v, int f) {
    if (f >= IW_FW_USED_FEATURES) {
        return 0.0f;
    }
    if (f < IW_WALL_VOCAB - 1) {
        return iw_type_idx(v, IW_WALL_VOCAB) == f + 1 ? 1.0f : 0.0f;
    }
    return v;
}

// whether a floating wall record is a wall rather than an empty slot, given
// the wall type value of its slot
IW_FN int iw_fw_active(float wallType) {
    return iw_type_idx(wallType, IW_WALL_VOCAB) != 0;
}

// tanh approximation of GELU, as the OSRS entity encoder uses
IW_MATH_FN float iw_gelu(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

IW_MATH_FN float iw_gelu_grad(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    const float t = tanhf(inner);
    const float dinner = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner;
}

#endif
