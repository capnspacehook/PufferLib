#ifndef IMPULSE_WARS_SETTINGS_H
#define IMPULSE_WARS_SETTINGS_H

#include "helpers.h"
#include "types.h"

#define INFINITE -1
static const uint8_t TWO_BIT_MASK = 0x3;

// general settings
static const uint8_t TRAINING_ACTIONS_PER_SECOND = 10;
static const uint8_t TRAINING_FRAME_RATE = 30;
static const uint8_t TRAINING_BOX2D_SUBSTEPS = 2;
static const uint16_t TRAINING_MAX_STEPS = 4500;

#define _EVAL_FRAME_RATE 120
static const uint8_t EVAL_FRAME_RATE = _EVAL_FRAME_RATE;
static const uint8_t EVAL_BOX2D_SUBSTEPS = 4;

static const uint8_t NUM_MAPS = 15;
// map used for easy curriculum episodes, the empty one
static const uint8_t EASY_EP_MAP_IDX = 0;
// shots an agent must fire in an easy episode for its hit rate to count
static const uint8_t EASY_EP_MIN_SHOTS = 10;
#define _MAX_MAP_COLUMNS 25
#define _MAX_MAP_ROWS 25
#define MAX_CELLS _MAX_MAP_COLUMNS *_MAX_MAP_ROWS + 1
#define MAX_FLOATING_WALLS 18
#define MAX_WEAPON_PICKUPS 12

#define MAX_NEAREST_WALLS 8

static const uint8_t DRONE_LIVES = 3;
static const float DRONE_RESPAWN_WAIT = 2.0f;
static const uint8_t ROUND_STEPS = 90;
static const uint8_t SUDDEN_DEATH_STEPS = 5;

static const uint8_t MAX_DRONES = _MAX_DRONES;

// the reported impulses from collisions seem to be roughly half what is
// actually applied (the bounce impulse isn't included in normalImpulse),
// and they're applied again, so it's 3x what's reported
static const float PROJECTILE_RESTITUTION = 1.0f;
static const float SHOT_HIT_IMPULSE_SCALE = 1.0f + PROJECTILE_RESTITUTION + 1.0f;

static const float HIT_DV_REF = 75.0f;

// observation constants

// map layout observations
static const uint8_t MAP_OBS_ROWS = 11;
static const uint8_t MAP_OBS_COLUMNS = 11;
static const uint16_t MAP_OBS_SIZE = MAP_OBS_ROWS * MAP_OBS_COLUMNS;

// each map grid cell is a packed code: bits 0-2 hold the static wall id
// (0 for empty, wall type + 1, or GRID_OUT_OF_BOUNDS), bits 3-4 hold the
// overlapping floating wall type + 1, bit 5 is set if an available weapon
// pickup is in the cell and bit 6 if an enemy drone overlaps it; codes
// stay below 256 so they're exact in bf16
#define GRID_OUT_OF_BOUNDS 4
#define GRID_FLOATING_WALL_SHIFT 3
#define GRID_PICKUP_BIT (1 << 5)
#define GRID_ENEMY_BIT (1 << 6)
// cells only touching an entity on their border aren't marked
static const float GRID_OVERLAP_MARGIN = 0.05f;

#define _NUM_NEAR_WALL_OBS 4
static const uint8_t NUM_NEAR_WALL_OBS = _NUM_NEAR_WALL_OBS;
static const uint8_t NEAR_WALL_INFO_OBS_SIZE = 3;
static const uint8_t NEAR_WALL_OBS_SIZE = NUM_NEAR_WALL_OBS * NEAR_WALL_INFO_OBS_SIZE;
static const uint16_t NEAR_WALL_INFO_OBS_OFFSET = 0;

#define _NUM_FLOATING_WALL_OBS 8
static const uint8_t NUM_FLOATING_WALL_OBS = _NUM_FLOATING_WALL_OBS;
static const uint8_t FLOATING_WALL_INFO_OBS_SIZE = 7;
static const uint8_t FLOATING_WALL_OBS_SIZE = NUM_FLOATING_WALL_OBS * FLOATING_WALL_INFO_OBS_SIZE;
static const uint16_t FLOATING_WALL_INFO_OBS_OFFSET = NEAR_WALL_INFO_OBS_OFFSET + NEAR_WALL_OBS_SIZE;

#define _NUM_WEAPON_PICKUP_OBS 3
static const uint8_t NUM_WEAPON_PICKUP_OBS = _NUM_WEAPON_PICKUP_OBS;
static const uint8_t WEAPON_PICKUP_INFO_OBS_SIZE = 3;
static const uint8_t WEAPON_PICKUP_OBS_SIZE = NUM_WEAPON_PICKUP_OBS * WEAPON_PICKUP_INFO_OBS_SIZE;
static const uint16_t WEAPON_PICKUP_INFO_OBS_OFFSET = FLOATING_WALL_INFO_OBS_OFFSET + FLOATING_WALL_OBS_SIZE;

static const uint8_t NUM_PROJECTILE_OBS = 30;
static const uint8_t PROJECTILE_INFO_OBS_SIZE = 6;
static const uint8_t PROJECTILE_OBS_SIZE = NUM_PROJECTILE_OBS * PROJECTILE_INFO_OBS_SIZE;
static const uint16_t PROJECTILE_INFO_OBS_OFFSET = WEAPON_PICKUP_INFO_OBS_OFFSET + WEAPON_PICKUP_OBS_SIZE;

// ray cast observations are a hit type followed by a scaled distance or
// time; hit types are RAY_HIT_NONE, wall type + RAY_HIT_WALL,
// floating wall type + RAY_HIT_FLOATING_WALL or RAY_HIT_DRONE
#define RAY_HIT_NONE 0
#define RAY_HIT_WALL 1
#define RAY_HIT_FLOATING_WALL 4
#define RAY_HIT_DRONE 7
#define NUM_RAY_HIT_TYPES 8
static const uint8_t RAY_OBS_SIZE = 2;
// aim, velocity and post recoil velocity
static const uint8_t NUM_DRONE_RAY_OBS = 3;
// velocity casts follow the damped path the drone would travel in this
// many seconds
static const float RAY_TIME_HORIZON = 2.0f;

static const uint8_t WEAPON_OBS_SIZE = 6;
static const uint8_t GRAPPLE_OBS_SIZE = 10;

static const uint16_t ENEMY_DRONE_OBS_OFFSET = PROJECTILE_INFO_OBS_OFFSET + PROJECTILE_OBS_SIZE;
// the ray to the enemy is last
static const uint8_t ENEMY_DRONE_RAY_OBS_IDX = 48;
static const uint8_t ENEMY_DRONE_OBS_SIZE = ENEMY_DRONE_RAY_OBS_IDX + RAY_OBS_SIZE;

// the rays are last
static const uint8_t DRONE_RAY_OBS_IDX = 48;
static const uint8_t DRONE_OBS_SIZE = DRONE_RAY_OBS_IDX + (NUM_DRONE_RAY_OBS * RAY_OBS_SIZE);

// easy episode flag, steps left and sudden death steps left
static const uint8_t MISC_OBS_SIZE = 3;

static const uint16_t DISCRETE_OBS_SIZE = MAP_OBS_SIZE;
static const uint16_t _CONTINUOUS_OBS_SIZE = NEAR_WALL_OBS_SIZE + FLOATING_WALL_OBS_SIZE + WEAPON_PICKUP_OBS_SIZE + PROJECTILE_OBS_SIZE + DRONE_OBS_SIZE + MISC_OBS_SIZE;

static uint16_t continuousObsSize(uint8_t numDrones) {
    return _CONTINUOUS_OBS_SIZE + ((numDrones - 1) * ENEMY_DRONE_OBS_SIZE);
}

static uint16_t obsSize(uint8_t numDrones) {
    return DISCRETE_OBS_SIZE + continuousObsSize(numDrones);
}

static const float MAX_DISTANCE = 200.0f;
static const float MAX_SPEED = 500.0f;

static const float MAX_OBS_AGENT_POS = 50.0f;
static const float MAX_OBS_AGENT_SPEED = 45.0f;
static const float MAX_OBS_RELATIVE_POS = 100.0f;
static const float MAX_OBS_DRONE_DISTANCE = 85.0f;
static const float MAX_OBS_WALL_POS = 50.0f;
static const float MAX_OBS_WALL_SPEED = 30.0f;
static const float MAX_OBS_ENTITY_SPEED = 75.0f;
static const float MAX_OBS_PROJECTILE_SPEED = 150.0f;

// action constants
static const float ACTION_NOOP_MAGNITUDE = 0.1f;

static const float discMoveToContMoveMap[2][8] = {
    {1.0f, 0.707107f, 0.0f, -0.707107f, -1.0f, -0.707107f, 0.0f, 0.707107f},
    {0.0f, 0.707107f, 1.0f, 0.707107f, 0.0f, -0.707107f, -1.0f, -0.707107f},
};

static const float AIM_FINE_STEP_ANGLE = 2.0f * PI / (AIM_COARSE_SECTORS * AIM_FINE_STEPS);

static const float MIN_SPAWN_DISTANCE = 6.0f;
static const float MIN_SD_SPAWN_DISTANCE = 3.0f;

// wall settings
static const float WALL_THICKNESS = 4.0f;
static const float FLOATING_WALL_THICKNESS = 3.0f;
static const float FLOATING_WALL_DAMPING = 0.75f;
// box2d's default sleep threshold is 0.05, tuned for a world where a typical
// body is about a meter across. Floating walls are 3 units wide and everything
// here moves at tens of units per second, so at the default they creep along
// below any visible speed forever and never sleep, staying in the broad phase
// and the solver for the whole round.
static const float FLOATING_WALL_SLEEP_THRESHOLD = 0.5f;
static const float STANDARD_WALL_RESTITUTION = 0.01f;
static const float STANDARD_WALL_FRICTION = 0.3f;
static const float BOUNCY_WALL_RESTITUTION = 1.0f;
static const float WALL_DENSITY = 4.0f;
#define FLOATING_WALL_INV_MASS INV_MASS(WALL_DENSITY * FLOATING_WALL_THICKNESS * FLOATING_WALL_THICKNESS)

// weapon pickup settings
static const float PICKUP_THICKNESS = 3.0f;
static const float PICKUP_SPAWN_DISTANCE_SQUARED = SQUARED(10.0f);
static const float PICKUP_RESPAWN_WAIT = 4.0f;
static const float SUDDEN_DEATH_PICKUP_RESPAWN_WAIT = 2.0f;

static const float PICKUP_HELD_WEAPON_WEIGHT_COEF = 0.33f;
// pickup spawn weights default to a 0-10 scale where 5 is an average
// spawn rate; only the ratios between weights matter. These are the
// defaults, maps can override any of them (see mapEntry). The max is
// above the default scale so maps can feature a weapon by giving it the
// max weight
#define MAX_PICKUP_SPAWN_WEIGHT 15.0f

// ammo crates refill the picking drone's weapons; in combo training
// episodes they're the only pickups
#define AMMO_CRATE_SPAWN_WEIGHT 5.0f
// pickup type of an ammo crate in the obs, one past the last weapon
#define AMMO_CRATE_OBS_TYPE _NUM_WEAPONS

// drone settings
static const float DRONE_WALL_SPAWN_DISTANCE = 2.0f;
static const float DRONE_DEATH_WALL_SPAWN_DISTANCE = 7.5f;
static const float DRONE_DRONE_SPAWN_DISTANCE_SQUARED = SQUARED(10.0f);

#define DRONE_RADIUS 1.0f
#define DRONE_DENSITY 1.25f
static const float DRONE_RESTITUTION = 0.3f;
static const float DRONE_FRICTION = 0.1f;
#define DRONE_MASS MASS(DRONE_DENSITY, DRONE_RADIUS)
#define DRONE_INV_MASS INV_MASS(DRONE_MASS)
static const float DRONE_MOVE_MAGNITUDE = 35.0f;
static const float DRONE_LINEAR_DAMPING = 1.0f;
static const float DRONE_MOVE_AIM_COEF = 0.1f;

static const float DRONE_SWAP_COOLDOWN = 0.35f;
// seconds after a black hole pull or grapple attachment that the enemy
// dying earns the setter an assist
static const float DRONE_ASSIST_WINDOW = 2.0f;

static const float DRONE_ENERGY_MAX = 1.0f;
static const float DRONE_BRAKE_DAMPING_COEF = 2.5f;
static const float DRONE_BRAKE_DRAIN_RATE = 0.5f;
static const float DRONE_ENERGY_REFILL_WAIT = 1.0f;
static const float DRONE_ENERGY_REFILL_EMPTY_WAIT = 3.0f;
static const float DRONE_ENERGY_REFILL_RATE = 0.03f;
static const float DRONE_ENERGY_RESPAWN_REFILL = 0.5f;
static const float DRONE_CHARGE_FULL_DRAIN_WAIT = 2.0f;
static const float DRONE_CHARGE_HOLD_DRAIN_RATE = 0.02;

static const float MAX_OBS_DRONE_CHARGE_HOLD = DRONE_CHARGE_FULL_DRAIN_WAIT * 2.5f;

static const float DRONE_BURST_BASE_COST = 0.1f;
static const float DRONE_BURST_CHARGE_RATE = 0.6f;
static const float DRONE_BURST_RADIUS_BASE = 5.0f;
static const float DRONE_BURST_RADIUS_MIN = 3.5f;
static const float DRONE_BURST_IMPACT_BASE = 140.0f;
static const float DRONE_BURST_IMPACT_MIN = 35.0f;
static const float DRONE_BURST_COOLDOWN = 0.5f;
static const float DRONE_BURST_ACTIVE_TIME = 0.15f;

static const float DRONE_SHIELD_RADIUS = DRONE_RADIUS * 1.5f;
static const float DRONE_SHIELD_HEALTH = 150.0f;
static const float DRONE_SHIELD_START_DURATION = 1.5f;
static const float DRONE_SHIELD_RESPAWN_DURATION = 3.0f;
static const float DRONE_SHIELD_EXPLOSION_REDUCTION = 0.33f;
static const float DRONE_SHIELD_FORCE_REDUCTION = 0.33f;
static const float DRONE_SHIELD_HEALTH_IMPULSE_COEF = 0.5f;
static const float DRONE_SHIELD_HEALTH_EXPLOSION_COEF = 0.8;
static const float DRONE_SHIELD_BREAK_ENERGY_COST = -0.33f;
static const float DRONE_SHIELD_BREAK_ENERGY_REFILL = 0.33f;

#define PROJECTILE_ENERGY_REFILL_COEF 0.001f
// scales the burst impulse applied to a drone's own projectiles when
// boosting them; deflecting projectiles isn't affected
#define PROJECTILE_BURST_BOOST_COEF 1.0f
static const float EXPLOSION_ENERGY_REFILL_COEF = 1.75f;

static const uint8_t DRONE_PIECE_COUNT = 5;
static const float DRONE_PIECE_MIN_DISTANCE = 1.5f;
static const float DRONE_PIECE_MAX_DISTANCE = 3.0f;
static const float DRONE_PIECE_LINEAR_DAMPING = 1.0f;
static const float DRONE_PIECE_ANGULAR_DAMPING = 1.0f;
static const float DRONE_PIECE_MIN_SPEED = 5.0f;
static const float DRONE_PIECE_MAX_SPEED = 10.0f;

// TODO: increase impulse of imploder of entity it directly hits?
// weapon projectile settings
#define STANDARD_AMMO INFINITE
#define STANDARD_PROJECTILES 1
#define STANDARD_RECOIL_MAGNITUDE 20.0f
#define STANDARD_FIRE_MAGNITUDE 17.0f
#define STANDARD_DAMPING 0.0f
#define STANDARD_CHARGE 0.0f
#define STANDARD_COOL_DOWN 0.375f
#define STANDARD_MAX_DISTANCE 80.0f
#define STANDARD_RADIUS 0.2
#define STANDARD_DENSITY 3.25f
#define STANDARD_MASS MASS(STANDARD_DENSITY, STANDARD_RADIUS)
#define STANDARD_INV_MASS INV_MASS(STANDARD_MASS)
#define STANDARD_BOUNCE 2
#define STANDARD_ENERGY_REFILL_COEF 5.0f
#define STANDARD_SPAWN_WEIGHT 0.0f

#define MACHINEGUN_AMMO 50
#define MACHINEGUN_PROJECTILES 1
#define MACHINEGUN_RECOIL_MAGNITUDE 12.8f
#define MACHINEGUN_FIRE_MAGNITUDE 30.0f
#define MACHINEGUN_DAMPING 0.1f
#define MACHINEGUN_CHARGE 0.0f
#define MACHINEGUN_COOL_DOWN 0.075f
#define MACHINEGUN_MAX_DISTANCE 225.0f
#define MACHINEGUN_RADIUS 0.15f
#define MACHINEGUN_DENSITY 3.2f
#define MACHINEGUN_MASS MASS(MACHINEGUN_DENSITY, MACHINEGUN_RADIUS)
#define MACHINEGUN_INV_MASS INV_MASS(MACHINEGUN_MASS)
#define MACHINEGUN_BOUNCE 1
#define MACHINEGUN_ENERGY_REFILL_COEF 0.5f
#define MACHINEGUN_SPAWN_WEIGHT 5.0f
#define MACHINEGUN_SWAY (5.7f * PI / 180.0f)
#define MACHINEGUN_SWAY_HEAT 120.0f
#define MACHINEGUN_HEAT_BUILDUP 11
#define MACHINEGUN_HEAT_COOL 100.0f

#define SNIPER_AMMO 3
#define SNIPER_PROJECTILES 1
#define SNIPER_RECOIL_MAGNITUDE 110.0f
#define SNIPER_FIRE_MAGNITUDE 280.0f
#define SNIPER_DAMPING 0.05f
#define SNIPER_CHARGE 1.5f
#define SNIPER_COOL_DOWN 1.5f
#define SNIPER_MAX_DISTANCE INFINITE
#define SNIPER_RADIUS 0.5f
#define SNIPER_DENSITY 2.0f
#define SNIPER_MASS MASS(SNIPER_DENSITY, SNIPER_RADIUS)
#define SNIPER_INV_MASS INV_MASS(SNIPER_MASS)
#define SNIPER_BOUNCE 0
#define SNIPER_ENERGY_REFILL_COEF 1.2f
#define SNIPER_SPAWN_WEIGHT 5.0f

#define SHOTGUN_AMMO 8
#define SHOTGUN_PROJECTILES 8
#define SHOTGUN_RECOIL_MAGNITUDE 100.0f
#define SHOTGUN_FIRE_MAGNITUDE 25.0f
#define SHOTGUN_DAMPING 0.4f
#define SHOTGUN_CHARGE 0.0f
#define SHOTGUN_COOL_DOWN 0.8f
#define SHOTGUN_MAX_DISTANCE 75.0f
#define SHOTGUN_RADIUS 0.15f
#define SHOTGUN_DENSITY 2.5f
#define SHOTGUN_MASS MASS(SHOTGUN_DENSITY, SHOTGUN_RADIUS)
#define SHOTGUN_INV_MASS INV_MASS(SHOTGUN_MASS)
#define SHOTGUN_BOUNCE 0
#define SHOTGUN_ENERGY_REFILL_COEF 1.5f
#define SHOTGUN_BURST_BOOST_COEF 0.75f
#define SHOTGUN_SPAWN_WEIGHT 5.0f
#define SHOTGUN_SPREAD (8.0f * PI / 180.0f)
#define SHOTGUN_SPREAD_JITTER (1.0f * PI / 180.0f)

#define IMPLODER_AMMO 2
#define IMPLODER_PROJECTILES 1
#define IMPLODER_RECOIL_MAGNITUDE 65.0f
#define IMPLODER_FIRE_MAGNITUDE 60.0f
#define IMPLODER_DAMPING 0.0f
#define IMPLODER_CHARGE 1.5f
#define IMPLODER_COOL_DOWN 1.5f
#define IMPLODER_MAX_DISTANCE INFINITE
#define IMPLODER_RADIUS 0.8f
#define IMPLODER_DENSITY 1.0f
#define IMPLODER_MASS MASS(IMPLODER_DENSITY, IMPLODER_RADIUS)
#define IMPLODER_INV_MASS INV_MASS(IMPLODER_MASS)
#define IMPLODER_BOUNCE 0
#define IMPLODER_SPAWN_WEIGHT 5.0f
#define IMPLODER_EXPLOSION_RADIUS 15.0f
#define IMPLODER_EXPLOSION_IMPULSE -250.0f

#define ACCELERATOR_AMMO 5
#define ACCELERATOR_PROJECTILES 1
#define ACCELERATOR_RECOIL_MAGNITUDE 50.0f
#define ACCELERATOR_FIRE_MAGNITUDE 50.0f
#define ACCELERATOR_DAMPING 0.0f
#define ACCELERATOR_CHARGE 0.0f
#define ACCELERATOR_COOL_DOWN 2.0f
#define ACCELERATOR_MAX_DISTANCE INFINITE
#define ACCELERATOR_RADIUS 0.5f
#define ACCELERATOR_DENSITY 2.0f
#define ACCELERATOR_MASS MASS(ACCELERATOR_DENSITY, ACCELERATOR_RADIUS)
#define ACCELERATOR_INV_MASS INV_MASS(ACCELERATOR_MASS)
#define ACCELERATOR_BOUNCE 3
#define ACCELERATOR_BOUNCE_SPEED_COEF 2.0f
#define ACCELERATOR_MAX_SPEED 500.f
#define ACCELERATOR_SPAWN_WEIGHT 5.0f

#define FLAK_CANNON_AMMO 12
#define FLAK_CANNON_PROJECTILES 1
#define FLAK_CANNON_RECOIL_MAGNITUDE 30.0f
#define FLAK_CANNON_FIRE_MAGNITUDE 14.0f
#define FLAK_CANNON_DAMPING 0.15f
#define FLAK_CANNON_CHARGE 0.0f
#define FLAK_CANNON_COOL_DOWN 0.4f
#define FLAK_CANNON_MAX_DISTANCE 100.0f
#define FLAK_CANNON_RADIUS 0.3f
#define FLAK_CANNON_DENSITY 1.0f
#define FLAK_CANNON_MASS MASS(FLAK_CANNON_DENSITY, FLAK_CANNON_RADIUS)
#define FLAK_CANNON_INV_MASS INV_MASS(FLAK_CANNON_MASS)
#define FLAK_CANNON_BOUNCE INFINITE
#define FLAK_CANNON_BURST_BOOST_COEF 0.8f
#define FLAK_CANNON_SAFE_DISTANCE 25.0f
#define FLAK_CANNON_PROXIMITY_RADIUS 2.0f
#define FLAK_CANNON_SPAWN_WEIGHT 5.0f
#define FLAK_CANNON_EXPLOSION_RADIUS 7.5f
#define FLAK_CANNON_EXPLOSION_IMPULSE 15.0f

#define MINE_LAUNCHER_AMMO 3
#define MINE_LAUNCHER_PROJECTILES 1
#define MINE_LAUNCHER_RECOIL_MAGNITUDE 20.0f
#define MINE_LAUNCHER_FIRE_MAGNITUDE 25.0f
#define MINE_LAUNCHER_DAMPING 0.2f
#define MINE_LAUNCHER_CHARGE 0.0f
#define MINE_LAUNCHER_COOL_DOWN 0.6f
#define MINE_LAUNCHER_MAX_DISTANCE INFINITE
#define MINE_LAUNCHER_RADIUS 0.5f
#define MINE_LAUNCHER_DENSITY 0.5f
#define MINE_LAUNCHER_MASS MASS(MINE_LAUNCHER_DENSITY, MINE_LAUNCHER_RADIUS)
#define MINE_LAUNCHER_INV_MASS INV_MASS(MINE_LAUNCHER_MASS)
#define MINE_LAUNCHER_BOUNCE INFINITE // this is to avoid mines sometimes exploding when hitting walls
#define MINE_LAUNCHER_SPAWN_WEIGHT 5.0f
#define MINE_LAUNCHER_EXPLOSION_RADIUS 15.0f
#define MINE_LAUNCHER_EXPLOSION_IMPULSE 200.0f
#define MINE_LAUNCHER_PROXIMITY_RADIUS 7.5f

#define BLACK_HOLE_AMMO 3
#define BLACK_HOLE_PROJECTILES 1
#define BLACK_HOLE_RECOIL_MAGNITUDE 75.0f
#define BLACK_HOLE_FIRE_MAGNITUDE 175.0f
#define BLACK_HOLE_DAMPING 0.0f
#define BLACK_HOLE_CHARGE 0.75f
#define BLACK_HOLE_COOL_DOWN 1.0f
#define BLACK_HOLE_MAX_DISTANCE INFINITE
#define BLACK_HOLE_RADIUS 0.5f
#define BLACK_HOLE_DENSITY 10.0f
#define BLACK_HOLE_MASS MASS(BLACK_HOLE_DENSITY, BLACK_HOLE_RADIUS)
#define BLACK_HOLE_INV_MASS INV_MASS(BLACK_HOLE_MASS)
#define BLACK_HOLE_BOUNCE 1
#define BLACK_HOLE_ENERGY_REFILL_COEF 0.25f
#define BLACK_HOLE_PULL_ENERGY_REFILL_COEF 0.3f
#define BLACK_HOLE_SPAWN_WEIGHT 5.0f
#define BLACK_HOLE_PROXIMITY_RADIUS 10.0f
#define BLACK_HOLE_PARENT_IGNORE_DISTANCE BLACK_HOLE_PROXIMITY_RADIUS * 1.5f
#define BLACK_HOLE_PULL_MAGNITUDE -350.0f
// Seconds for the visual pull field and tint to fade after projectile destruction.
#define BLACK_HOLE_DISTORTION_FADE_TIME 0.3f

#define NUKE_AMMO 1
#define NUKE_PROJECTILES 1
#define NUKE_RECOIL_MAGNITUDE 150.0f
#define NUKE_FIRE_MAGNITUDE 85.0f
#define NUKE_DAMPING 0.0f
#define NUKE_CHARGE 3.0f
#define NUKE_COOL_DOWN 3.0f
#define NUKE_MAX_DISTANCE INFINITE
#define NUKE_RADIUS 0.8f
#define NUKE_DENSITY 3.0f
#define NUKE_MASS MASS(NUKE_DENSITY, NUKE_RADIUS)
#define NUKE_INV_MASS INV_MASS(NUKE_MASS)
#define NUKE_BOUNCE 0
#define NUKE_SPAWN_WEIGHT 2.5f
#define NUKE_EXPLOSION_RADIUS 35.0f
#define NUKE_EXPLOSION_IMPULSE 500.0f

#define GRAPPLE_AMMO 3
#define GRAPPLE_PROJECTILES 1
#define GRAPPLE_RECOIL_MAGNITUDE 25.0f
#define GRAPPLE_FIRE_MAGNITUDE 55.0f
#define GRAPPLE_DAMPING 0.0f
#define GRAPPLE_CHARGE 0.0f
#define GRAPPLE_COOL_DOWN 0.5f
#define GRAPPLE_MAX_DISTANCE 150.0f
#define GRAPPLE_RADIUS 0.2f
#define GRAPPLE_DENSITY 3.5f
#define GRAPPLE_MASS MASS(GRAPPLE_DENSITY, GRAPPLE_RADIUS)
#define GRAPPLE_INV_MASS INV_MASS(GRAPPLE_MASS)
#define GRAPPLE_BOUNCE INFINITE // this is to avoid grapples getting destroyed before they can create a weld joint
#define GRAPPLE_SPAWN_WEIGHT 4.0f
#define GRAPPLE_MOTOR_SPEED 100.0f
#define GRAPPLE_MOTOR_FORCE 250.0f
#define GRAPPLE_SLACK 3.0f
#define GRAPPLE_DAMPING_RATIO 0.1f
#define GRAPPLE_MAX_STRETCH 5.0f
#define GRAPPLE_SPRING_HERTZ 3.0f
#define GRAPPLE_HEALTH 100.0f
#define GRAPPLE_BURST_REDUCTION 0.8f

static const weaponInformation unsetWeapon = {
    .type = UNSET_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = false,
    .numProjectiles = 0,
    .fireMagnitude = 0.0f,
    .recoilMagnitude = 0.0f,
    .damping = 0.0f,
    .maxCharge = 0.0f,
    .cooldown = 0.0f,
    .maxDistance = 0.0f,
    .radius = 0.0f,
    .density = 0.0f,
    .mass = 0.0f,
    .invMass = 0.0f,
    .initialSpeed = 0.0f,
    .maxBounces = 0,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = 0.0f,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = 0.0f,
};

static const weaponInformation standard = {
    .type = STANDARD_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = STANDARD_PROJECTILES,
    .fireMagnitude = STANDARD_FIRE_MAGNITUDE,
    .recoilMagnitude = STANDARD_RECOIL_MAGNITUDE,
    .damping = STANDARD_DAMPING,
    .maxCharge = STANDARD_CHARGE,
    .cooldown = STANDARD_COOL_DOWN,
    .maxDistance = STANDARD_MAX_DISTANCE,
    .radius = STANDARD_RADIUS,
    .density = STANDARD_DENSITY,
    .mass = STANDARD_MASS,
    .invMass = STANDARD_INV_MASS,
    .initialSpeed = STANDARD_FIRE_MAGNITUDE * STANDARD_INV_MASS,
    .maxBounces = STANDARD_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF * STANDARD_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = STANDARD_SPAWN_WEIGHT,
};

static const weaponInformation machineGun = {
    .type = MACHINEGUN_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = MACHINEGUN_PROJECTILES,
    .fireMagnitude = MACHINEGUN_FIRE_MAGNITUDE,
    .recoilMagnitude = MACHINEGUN_RECOIL_MAGNITUDE,
    .damping = MACHINEGUN_DAMPING,
    .maxCharge = MACHINEGUN_CHARGE,
    .cooldown = MACHINEGUN_COOL_DOWN,
    .maxDistance = MACHINEGUN_MAX_DISTANCE,
    .radius = MACHINEGUN_RADIUS,
    .density = MACHINEGUN_DENSITY,
    .mass = MACHINEGUN_MASS,
    .invMass = MACHINEGUN_INV_MASS,
    .initialSpeed = MACHINEGUN_FIRE_MAGNITUDE * MACHINEGUN_INV_MASS,
    .maxBounces = MACHINEGUN_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF * MACHINEGUN_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = MACHINEGUN_SPAWN_WEIGHT,
};

static const weaponInformation sniper = {
    .type = SNIPER_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = SNIPER_PROJECTILES,
    .fireMagnitude = SNIPER_FIRE_MAGNITUDE,
    .recoilMagnitude = SNIPER_RECOIL_MAGNITUDE,
    .damping = SNIPER_DAMPING,
    .maxCharge = SNIPER_CHARGE,
    .cooldown = SNIPER_COOL_DOWN,
    .maxDistance = SNIPER_MAX_DISTANCE,
    .radius = SNIPER_RADIUS,
    .density = SNIPER_DENSITY,
    .mass = SNIPER_MASS,
    .invMass = SNIPER_INV_MASS,
    .initialSpeed = SNIPER_FIRE_MAGNITUDE * SNIPER_INV_MASS,
    .maxBounces = SNIPER_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF * SNIPER_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = SNIPER_SPAWN_WEIGHT,
};

static const weaponInformation shotgun = {
    .type = SHOTGUN_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = SHOTGUN_PROJECTILES,
    .fireMagnitude = SHOTGUN_FIRE_MAGNITUDE,
    .recoilMagnitude = SHOTGUN_RECOIL_MAGNITUDE,
    .damping = SHOTGUN_DAMPING,
    .maxCharge = SHOTGUN_CHARGE,
    .cooldown = SHOTGUN_COOL_DOWN,
    .maxDistance = SHOTGUN_MAX_DISTANCE,
    .radius = SHOTGUN_RADIUS,
    .density = SHOTGUN_DENSITY,
    .mass = SHOTGUN_MASS,
    .invMass = SHOTGUN_INV_MASS,
    .initialSpeed = SHOTGUN_FIRE_MAGNITUDE * SHOTGUN_INV_MASS,
    .maxBounces = SHOTGUN_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF * SHOTGUN_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF * SHOTGUN_BURST_BOOST_COEF,
    .spawnWeight = SHOTGUN_SPAWN_WEIGHT,
};

static const weaponInformation imploder = {
    .type = IMPLODER_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = false,
    .numProjectiles = IMPLODER_PROJECTILES,
    .fireMagnitude = IMPLODER_FIRE_MAGNITUDE,
    .recoilMagnitude = IMPLODER_RECOIL_MAGNITUDE,
    .damping = IMPLODER_DAMPING,
    .maxCharge = IMPLODER_CHARGE,
    .cooldown = IMPLODER_COOL_DOWN,
    .maxDistance = IMPLODER_MAX_DISTANCE,
    .radius = IMPLODER_RADIUS,
    .density = IMPLODER_DENSITY,
    .mass = IMPLODER_MASS,
    .invMass = IMPLODER_INV_MASS,
    .initialSpeed = IMPLODER_FIRE_MAGNITUDE * IMPLODER_INV_MASS,
    .maxBounces = IMPLODER_BOUNCE + 1,
    .explosive = true,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = true,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = IMPLODER_SPAWN_WEIGHT,
};

static const weaponInformation accelerator = {
    .type = ACCELERATOR_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = ACCELERATOR_PROJECTILES,
    .fireMagnitude = ACCELERATOR_FIRE_MAGNITUDE,
    .recoilMagnitude = ACCELERATOR_RECOIL_MAGNITUDE,
    .damping = ACCELERATOR_DAMPING,
    .maxCharge = ACCELERATOR_CHARGE,
    .cooldown = ACCELERATOR_COOL_DOWN,
    .maxDistance = ACCELERATOR_MAX_DISTANCE,
    .radius = ACCELERATOR_RADIUS,
    .density = ACCELERATOR_DENSITY,
    .mass = ACCELERATOR_MASS,
    .invMass = ACCELERATOR_INV_MASS,
    .initialSpeed = ACCELERATOR_FIRE_MAGNITUDE * ACCELERATOR_INV_MASS,
    .maxBounces = ACCELERATOR_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = ACCELERATOR_SPAWN_WEIGHT,
};

static const weaponInformation flakCannon = {
    .type = FLAK_CANNON_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = FLAK_CANNON_PROJECTILES,
    .fireMagnitude = FLAK_CANNON_FIRE_MAGNITUDE,
    .recoilMagnitude = FLAK_CANNON_RECOIL_MAGNITUDE,
    .damping = FLAK_CANNON_DAMPING,
    .maxCharge = FLAK_CANNON_CHARGE,
    .cooldown = FLAK_CANNON_COOL_DOWN,
    .maxDistance = FLAK_CANNON_MAX_DISTANCE,
    .radius = FLAK_CANNON_RADIUS,
    .density = FLAK_CANNON_DENSITY,
    .mass = FLAK_CANNON_MASS,
    .invMass = FLAK_CANNON_INV_MASS,
    .initialSpeed = FLAK_CANNON_FIRE_MAGNITUDE * FLAK_CANNON_INV_MASS,
    .maxBounces = FLAK_CANNON_BOUNCE + 1,
    .explosive = true,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = true,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF * FLAK_CANNON_BURST_BOOST_COEF,
    .spawnWeight = FLAK_CANNON_SPAWN_WEIGHT,
};

static const weaponInformation mineLauncher = {
    .type = MINE_LAUNCHER_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = true,
    .numProjectiles = MINE_LAUNCHER_PROJECTILES,
    .fireMagnitude = MINE_LAUNCHER_FIRE_MAGNITUDE,
    .recoilMagnitude = MINE_LAUNCHER_RECOIL_MAGNITUDE,
    .damping = MINE_LAUNCHER_DAMPING,
    .maxCharge = MINE_LAUNCHER_CHARGE,
    .cooldown = MINE_LAUNCHER_COOL_DOWN,
    .maxDistance = MINE_LAUNCHER_MAX_DISTANCE,
    .radius = MINE_LAUNCHER_RADIUS,
    .density = MINE_LAUNCHER_DENSITY,
    .mass = MINE_LAUNCHER_MASS,
    .invMass = MINE_LAUNCHER_INV_MASS,
    .initialSpeed = MINE_LAUNCHER_FIRE_MAGNITUDE * MINE_LAUNCHER_INV_MASS,
    .maxBounces = MINE_LAUNCHER_BOUNCE + 1,
    .explosive = true,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = false,
    .hasSensor = true,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = MINE_LAUNCHER_SPAWN_WEIGHT,
};

static const weaponInformation blackHole = {
    .type = BLACK_HOLE_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = false,
    .numProjectiles = BLACK_HOLE_PROJECTILES,
    .fireMagnitude = BLACK_HOLE_FIRE_MAGNITUDE,
    .recoilMagnitude = BLACK_HOLE_RECOIL_MAGNITUDE,
    .damping = BLACK_HOLE_DAMPING,
    .maxCharge = BLACK_HOLE_CHARGE,
    .cooldown = BLACK_HOLE_COOL_DOWN,
    .maxDistance = BLACK_HOLE_MAX_DISTANCE,
    .radius = BLACK_HOLE_RADIUS,
    .density = BLACK_HOLE_DENSITY,
    .mass = BLACK_HOLE_MASS,
    .invMass = BLACK_HOLE_INV_MASS,
    .initialSpeed = BLACK_HOLE_FIRE_MAGNITUDE * BLACK_HOLE_INV_MASS,
    .maxBounces = BLACK_HOLE_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = true,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF * BLACK_HOLE_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = BLACK_HOLE_SPAWN_WEIGHT,
};

static const weaponInformation nuke = {
    .type = NUKE_WEAPON,
    .isPhysicsBullet = false,
    .canSleep = false,
    .numProjectiles = NUKE_PROJECTILES,
    .fireMagnitude = NUKE_FIRE_MAGNITUDE,
    .recoilMagnitude = NUKE_RECOIL_MAGNITUDE,
    .damping = NUKE_DAMPING,
    .maxCharge = NUKE_CHARGE,
    .cooldown = NUKE_COOL_DOWN,
    .maxDistance = NUKE_MAX_DISTANCE,
    .radius = NUKE_RADIUS,
    .density = NUKE_DENSITY,
    .mass = NUKE_MASS,
    .invMass = NUKE_INV_MASS,
    .initialSpeed = NUKE_FIRE_MAGNITUDE * NUKE_INV_MASS,
    .maxBounces = NUKE_BOUNCE + 1,
    .explosive = true,
    .destroyedOnDroneHit = true,
    .explodesOnDroneHit = true,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = NUKE_SPAWN_WEIGHT,
};

static const weaponInformation grapple = {
    .type = GRAPPLE_WEAPON,
    .isPhysicsBullet = true,
    .canSleep = false,
    .numProjectiles = GRAPPLE_PROJECTILES,
    .fireMagnitude = GRAPPLE_FIRE_MAGNITUDE,
    .recoilMagnitude = GRAPPLE_RECOIL_MAGNITUDE,
    .damping = GRAPPLE_DAMPING,
    .maxCharge = GRAPPLE_CHARGE,
    .cooldown = GRAPPLE_COOL_DOWN,
    .maxDistance = GRAPPLE_MAX_DISTANCE,
    .radius = GRAPPLE_RADIUS,
    .density = GRAPPLE_DENSITY,
    .mass = GRAPPLE_MASS,
    .invMass = GRAPPLE_INV_MASS,
    .initialSpeed = GRAPPLE_FIRE_MAGNITUDE * GRAPPLE_INV_MASS,
    .maxBounces = GRAPPLE_BOUNCE + 1,
    .explosive = false,
    .destroyedOnDroneHit = false,
    .explodesOnDroneHit = false,
    .hasSensor = false,
    .energyRefillCoef = PROJECTILE_ENERGY_REFILL_COEF,
    .burstBoostCoef = PROJECTILE_BURST_BOOST_COEF,
    .spawnWeight = GRAPPLE_SPAWN_WEIGHT,
};

static weaponInformation *weaponInfos[] = {
    (weaponInformation *)&unsetWeapon,
    (weaponInformation *)&standard,
    (weaponInformation *)&machineGun,
    (weaponInformation *)&sniper,
    (weaponInformation *)&shotgun,
    (weaponInformation *)&imploder,
    (weaponInformation *)&accelerator,
    (weaponInformation *)&flakCannon,
    (weaponInformation *)&mineLauncher,
    (weaponInformation *)&blackHole,
    (weaponInformation *)&nuke,
    (weaponInformation *)&grapple,
};

static const char *weaponNames[] = {
    "standard",
    "machine_gun",
    "sniper",
    "shotgun",
    "imploder",
    "accelerator",
    "flak_cannon",
    "mine_launcher",
    "black_hole",
    "tactical_nuke",
    "grapple_gun",
};

// max ammo of weapon
static int8_t weaponAmmo(const enum weaponType type) {
    switch (type) {
    case UNSET_WEAPON:
        return 0;
    case STANDARD_WEAPON:
        return STANDARD_AMMO;
    case MACHINEGUN_WEAPON:
        return MACHINEGUN_AMMO;
    case SNIPER_WEAPON:
        return SNIPER_AMMO;
    case SHOTGUN_WEAPON:
        return SHOTGUN_AMMO;
    case IMPLODER_WEAPON:
        return IMPLODER_AMMO;
    case ACCELERATOR_WEAPON:
        return ACCELERATOR_AMMO;
    case FLAK_CANNON_WEAPON:
        return FLAK_CANNON_AMMO;
    case MINE_LAUNCHER_WEAPON:
        return MINE_LAUNCHER_AMMO;
    case BLACK_HOLE_WEAPON:
        return BLACK_HOLE_AMMO;
    case NUKE_WEAPON:
        return NUKE_AMMO;
    case GRAPPLE_WEAPON:
        return GRAPPLE_AMMO;
    default:
        ERRORF("unknown weapon type %d", type);
    }
}

static b2ShapeId weaponSensor(const b2BodyId bodyID, const enum weaponType type) {
    b2ShapeDef sensorShapeDef = b2DefaultShapeDef();
    sensorShapeDef.density = 0.0f;
    sensorShapeDef.isSensor = true;
    sensorShapeDef.enableSensorEvents = true;
    b2Circle sensorCircle = {.center = b2Vec2_zero};

    switch (type) {
    case FLAK_CANNON_WEAPON:
        sensorShapeDef.filter.categoryBits = PROJECTILE_SHAPE;
        sensorShapeDef.filter.maskBits = DRONE_SHAPE;
        sensorCircle.radius = FLAK_CANNON_PROXIMITY_RADIUS;
        break;
    case MINE_LAUNCHER_WEAPON:
        sensorShapeDef.filter.categoryBits = PROJECTILE_SHAPE;
        sensorShapeDef.filter.maskBits = DRONE_SHAPE;
        sensorCircle.radius = MINE_LAUNCHER_PROXIMITY_RADIUS;
        break;
    case BLACK_HOLE_WEAPON:
        sensorShapeDef.filter.categoryBits = PROJECTILE_SHAPE;
        sensorShapeDef.filter.maskBits = FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE;
        sensorCircle.radius = BLACK_HOLE_PROXIMITY_RADIUS;
        break;
    default:
        ERRORF("unknown weapon type with sensor %d", type);
    }

    return b2CreateCircleShape(bodyID, &sensorShapeDef, &sensorCircle);
}

// amount of force to apply to projectile
static float weaponFire(unsigned int *seed, const enum weaponType type) {
    switch (type) {
    case STANDARD_WEAPON:
        return STANDARD_FIRE_MAGNITUDE;
    case MACHINEGUN_WEAPON:
        return MACHINEGUN_FIRE_MAGNITUDE;
    case SNIPER_WEAPON:
        return SNIPER_FIRE_MAGNITUDE;
    case SHOTGUN_WEAPON: {
        const int maxOffset = 3;
        const int fireOffset = randInt(seed, -maxOffset, maxOffset);
        return SHOTGUN_FIRE_MAGNITUDE + fireOffset;
    }
    case IMPLODER_WEAPON:
        return IMPLODER_FIRE_MAGNITUDE;
    case ACCELERATOR_WEAPON:
        return ACCELERATOR_FIRE_MAGNITUDE;
    case FLAK_CANNON_WEAPON:
        return FLAK_CANNON_FIRE_MAGNITUDE;
    case MINE_LAUNCHER_WEAPON:
        return MINE_LAUNCHER_FIRE_MAGNITUDE;
    case BLACK_HOLE_WEAPON:
        return BLACK_HOLE_FIRE_MAGNITUDE;
    case NUKE_WEAPON:
        return NUKE_FIRE_MAGNITUDE;
    case GRAPPLE_WEAPON:
        return GRAPPLE_FIRE_MAGNITUDE;
    default:
        ERRORF("unknown weapon type %d", type);
        return 0;
    }
}

static b2Vec2 weaponAdjustAim(unsigned int *seed, const enum weaponType type, const uint16_t heat, const b2Vec2 normAim, const uint8_t projIdx) {
    switch (type) {
    case MACHINEGUN_WEAPON: {
        const float swayCoef = tanhf(SQUARED(heat / MACHINEGUN_SWAY_HEAT));
        const float angle = randFloat(seed, -1.0f, 1.0f) * MACHINEGUN_SWAY * swayCoef;
        return b2RotateVector(b2MakeRot(angle), normAim);
    }
    case SHOTGUN_WEAPON: {
        const float step = 2.0f * SHOTGUN_SPREAD / (SHOTGUN_PROJECTILES - 1);
        const float angle = -SHOTGUN_SPREAD + projIdx * step + randFloat(seed, -SHOTGUN_SPREAD_JITTER, SHOTGUN_SPREAD_JITTER);
        return b2RotateVector(b2MakeRot(angle), normAim);
    }
    default:
        return normAim;
    }
}

// sets explosion parameters and returns true if an explosion should be created
// when a projectile is destroyed
static void weaponExplosion(const enum weaponType type, b2ExplosionDef *explosionDef) {
    switch (type) {
    case IMPLODER_WEAPON:
        explosionDef->radius = IMPLODER_EXPLOSION_RADIUS;
        explosionDef->impulsePerLength = IMPLODER_EXPLOSION_IMPULSE;
        return;
    case FLAK_CANNON_WEAPON:
        explosionDef->radius = FLAK_CANNON_EXPLOSION_RADIUS;
        // will typically be closer to 45 with projectile velocity
        // factored in
        explosionDef->impulsePerLength = FLAK_CANNON_EXPLOSION_IMPULSE;
        return;
    case MINE_LAUNCHER_WEAPON:
        explosionDef->radius = MINE_LAUNCHER_EXPLOSION_RADIUS;
        explosionDef->impulsePerLength = MINE_LAUNCHER_EXPLOSION_IMPULSE;
        return;
    case NUKE_WEAPON:
        explosionDef->radius = NUKE_EXPLOSION_RADIUS;
        explosionDef->impulsePerLength = NUKE_EXPLOSION_IMPULSE;
        return;
    default:
        ERRORF("unknown weapon type %d for projectile explosion", type);
    }
}

static void insertionSortWalls(nearWall arr[], uint8_t size) {
    for (int16_t i = 1; i < size; i++) {
        const nearWall key = arr[i];
        int16_t j = i - 1;

        while (j >= 0 && arr[j].distanceSquared > key.distanceSquared) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }

        arr[j + 1] = key;
    }
}

// insertion sort should be faster than quicksort for small arrays
static void insertionSortEntities(nearEntity arr[], uint8_t size) {
    for (int16_t i = 1; i < size; i++) {
        const nearEntity key = arr[i];
        int16_t j = i - 1;

        while (j >= 0 && arr[j].distanceSquared > key.distanceSquared) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }

        arr[j + 1] = key;
    }
}

#endif
