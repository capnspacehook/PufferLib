#ifndef IMPULSE_WARS_SCRIPTED_BOT_H
#define IMPULSE_WARS_SCRIPTED_BOT_H

#include "game.h"

// behaviors every personality uses: staying out of kill geometry and using
// the utility
#define BOT_FEATURES_BASE (BOT_FEAT_SAFE_GEOMETRY | BOT_FEAT_UTILITY)

// optional behaviors the scripted bot uses by default; see enum botFeature
#ifndef BOT_FEATURES_DEFAULT
#define BOT_FEATURES_DEFAULT (BOT_FEAT_AMMO_RESERVE | BOT_FEAT_SHOTGUN_BOOST | BOT_FEAT_SHOTGUN_RANGE | BOT_FEATURES_BASE)
#endif

static const uint8_t NUM_NEAR_WALLS = 3;

static const float FLOATING_WALL_CHECK_DISTANCE_SQUARED = SQUARED(12.0f);
static const float WALL_AVOID_DISTANCE = 4.0f;
static const float WALL_DANGER_DISTANCE = 3.0f;
static const float WALL_BURST_CHECK_DISTANCE = 5.0f;
static const float WALL_BURST_CHECK_SPEED = 20.0f;
static const float BURST_MIN_RADIUS_SQUARED = SQUARED(DRONE_BURST_RADIUS_MIN);
static const float STABILIZE_MOVE_SPEED = 5.0f;
static const float SHOTGUN_DANGER_DISTANCE = 4.0f;

void addDebugPoint(iwEnv *e, b2Vec2 pos, float size, Color color) {
#ifdef PUF_DEBUG
    debugPoint *point = fastCalloc(1, sizeof(debugPoint));
    point->pos = pos;
    point->size = size;
    point->color = color;
    cc_array_add(e->debugPoints, point);
#else
    MAYBE_UNUSED(e);
    MAYBE_UNUSED(pos);
    MAYBE_UNUSED(size);
    MAYBE_UNUSED(color);
#endif
}

typedef struct castCircleCtx {
    bool hit;
    b2ShapeId shapeID;
    b2Vec2 point;
} castCircleCtx;

float castCircleCallback(b2ShapeId shapeId, b2Vec2 point, b2Vec2 normal, float fraction, void *context) {
    // these parameters are required by the callback signature
    MAYBE_UNUSED(point);
    MAYBE_UNUSED(normal);
    if (!b2Shape_IsValid(shapeId) || (fraction == 0.0f && b2Equal(normal, b2Vec2_zero))) {
        // skip this shape if it isn't valid or this is an initial overlap
        return -1.0f;
    }

    castCircleCtx *ctx = context;
    ctx->hit = true;
    ctx->shapeID = shapeId;
    ctx->point = point;

    return 0.0f;
}

float distanceWithDamping(const iwEnv *e, const float speed, const float linearDamping, const float steps) {
    const float damping = 1.0f + (linearDamping * e->deltaTime);
    return speed * (damping / linearDamping) * (1.0f - powf(1.0f / damping, steps));
}

b2Vec2 positionWithDamping(const iwEnv *e, const droneEntity *drone, const b2Vec2 impulse, const float linearDamping, const float steps) {
    const b2Vec2 newVel = b2Add(drone->velocity, impulse);
    const float speed = b2Length(newVel);
    const float distance = distanceWithDamping(e, speed, linearDamping, steps);
    const b2Vec2 direction = b2Normalize(newVel);
    return b2MulAdd(drone->pos, distance, direction);
}

// displacement after time seconds from a constant force on a body with
// linear damping, starting at rest
static inline float forcedDistance(const float accel, const float linearDamping, const float time) {
    return (accel / linearDamping) * (time - ((1.0f - expf(-linearDamping * time)) / linearDamping));
}

// returns true if drone can fire in the given direction without hitting
// or getting too close to a death wall before it can fire again; the
// planned movement is included in the predicted position
bool recoilPathSafe(iwEnv *e, const droneEntity *drone, const float recoilMagnitude, const float shotWait, const b2Vec2 direction, const b2Vec2 move);

bool safeToFireMoving(iwEnv *e, const droneEntity *drone, const b2Vec2 direction, const b2Vec2 move) {
    // don't shoot shotgun point blank at walls, the shots will immediately
    // bounce back and send the drone flying uncontrollably
    if (drone->weapon.weaponInfo->type == SHOTGUN_WEAPON) {
        const b2Vec2 rayEnd = b2MulAdd(drone->pos, SHOTGUN_DANGER_DISTANCE, direction);
        const b2Vec2 translation = b2Sub(rayEnd, drone->pos);
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
        const b2RayResult rayRes = b2World_CastRayClosest(e->worldID, drone->pos, translation, filter);
        if (rayRes.hit) {
            return false;
        }
    }

    float shotWait;
    if (drone->weapon.ammo > 1) {
        shotWait = ((drone->weapon.weaponInfo->cooldown + drone->weapon.weaponInfo->maxCharge) / e->deltaTime) * 1.5f;
    } else {
        const weaponInformation *standardInfo = weaponInfos[STANDARD_WEAPON];
        shotWait = (standardInfo->cooldown / e->deltaTime) * 1.5f;
    }
    return recoilPathSafe(e, drone, drone->weapon.weaponInfo->recoilMagnitude, shotWait, direction, move);
}

// whether the recoil of a shot along direction, followed by moving along
// move for shotWait frames, keeps the drone clear of death walls and drones
bool recoilPathSafe(iwEnv *e, const droneEntity *drone, const float recoilMagnitude, const float shotWait, const b2Vec2 direction, const b2Vec2 move) {
    const float recoilSpeed = -recoilMagnitude * DRONE_INV_MASS;
    const b2Vec2 recoil = b2MulSV(recoilSpeed, direction);
    b2Vec2 recoilPos = positionWithDamping(e, drone, recoil, DRONE_LINEAR_DAMPING, shotWait);
    if (!b2Equal(move, b2Vec2_zero)) {
        const float moveDistance = forcedDistance(DRONE_MOVE_MAGNITUDE * DRONE_INV_MASS, DRONE_LINEAR_DAMPING, shotWait * e->deltaTime);
        recoilPos = b2MulAdd(recoilPos, moveDistance, move);
    }

    addDebugPoint(e, b2MulAdd(drone->pos, 2.0f * DRONE_RADIUS, direction), 0.5f, WHITE);

    const b2Vec2 pos = drone->pos;
    const b2Vec2 rayEnd = recoilPos;
    const b2Vec2 translation = b2Sub(rayEnd, pos);
    float radius = DRONE_RADIUS;
    if (drone->shield != NULL) {
        radius = DRONE_SHIELD_RADIUS;
    }
    const b2ShapeProxy cirProxy = b2MakeProxy(&pos, 1, radius);
    const b2QueryFilter filter = {.categoryBits = DRONE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE};

    castCircleCtx ctx = {0};
    b2World_CastShape(e->worldID, &cirProxy, translation, filter, castCircleCallback, &ctx);
    if (!ctx.hit) {
        addDebugPoint(e, recoilPos, 0.5f, LIME);
        return true;
    } else {
        const entity *ent = b2Shape_GetUserData(ctx.shapeID);
        if (entityTypeIsWall(ent->type) && ent->type != DEATH_WALL_ENTITY) {
            addDebugPoint(e, recoilPos, 0.5f, LIME);
            return true;
        }
    }

    addDebugPoint(e, recoilPos, 0.5f, MAROON);
    return false;
}

bool safeToFire(iwEnv *e, const droneEntity *drone, const b2Vec2 direction) {
    return safeToFireMoving(e, drone, direction, b2Vec2_zero);
}

// whether the recoil of firing the utility along direction is safe
bool safeToFireUtility(iwEnv *e, const droneEntity *drone, const b2Vec2 direction) {
    const weaponInformation *info = drone->utility.utilityInfo->projectileInfo;
    const float shotWait = (info->cooldown / e->deltaTime) * 1.5f;
    return recoilPathSafe(e, drone, info->recoilMagnitude, shotWait, direction, b2Vec2_zero);
}

// engagement range of the flak cannon; flak
// shells only arm after FLAK_CANNON_SAFE_DISTANCE so the default ideal range
// leaves too small a window
static const float FLAK_RANGE = FLAK_CANNON_SAFE_DISTANCE + 15.0f;

// how many seconds of fighting a weapon pickup is worth
static const float botWeaponValue[_NUM_WEAPONS] = {
    [STANDARD_WEAPON] = 0.0f,
    [MACHINEGUN_WEAPON] = 4.0f,
    [SNIPER_WEAPON] = 4.5f,
    [SHOTGUN_WEAPON] = 2.0f,
    [IMPLODER_WEAPON] = 2.0f,
    [FLAK_CANNON_WEAPON] = 2.5f,
    [MINE_LAUNCHER_WEAPON] = 1.0f,
    [BLACK_HOLE_WEAPON] = 2.5f,
    [NUKE_WEAPON] = 1.5f,
};

// pickup values of the personalities; a weapon worth 0 is never detoured
// for. Speed first personalities value weapons by recoil impulse per second
static const float evaderWeaponValue[_NUM_WEAPONS] = {
    [MACHINEGUN_WEAPON] = 4.5f,
    [SHOTGUN_WEAPON] = 4.0f,
    [FLAK_CANNON_WEAPON] = 3.5f,
    [SNIPER_WEAPON] = 3.0f,
};

static const float camperWeaponValue[_NUM_WEAPONS] = {
    [MACHINEGUN_WEAPON] = 0.5f,
    [SNIPER_WEAPON] = 5.0f,
    [SHOTGUN_WEAPON] = 0.5f,
    [IMPLODER_WEAPON] = 3.5f,
    [FLAK_CANNON_WEAPON] = 0.5f,
    [MINE_LAUNCHER_WEAPON] = 0.5f,
    [BLACK_HOLE_WEAPON] = 1.0f,
    [NUKE_WEAPON] = 3.0f,
};

static const float rusherWeaponValue[_NUM_WEAPONS] = {
    [MACHINEGUN_WEAPON] = 4.5f,
    [SNIPER_WEAPON] = 0.5f,
    [SHOTGUN_WEAPON] = 5.0f,
    [IMPLODER_WEAPON] = 1.0f,
    [FLAK_CANNON_WEAPON] = 4.0f,
    [MINE_LAUNCHER_WEAPON] = 0.5f,
    [BLACK_HOLE_WEAPON] = 1.0f,
    [NUKE_WEAPON] = 1.0f,
};

static const float trapperWeaponValue[_NUM_WEAPONS] = {
    [MACHINEGUN_WEAPON] = 1.0f,
    [SNIPER_WEAPON] = 1.0f,
    [SHOTGUN_WEAPON] = 1.0f,
    [IMPLODER_WEAPON] = 4.0f,
    [FLAK_CANNON_WEAPON] = 1.0f,
    [MINE_LAUNCHER_WEAPON] = 5.0f,
    [BLACK_HOLE_WEAPON] = 4.5f,
    [NUKE_WEAPON] = 3.0f,
};

static const float fundamentalsWeaponValue[_NUM_WEAPONS] = {0};

// how many seconds of fighting a utility pickup is worth
static const float botUtilityValue[_NUM_UTILITIES] = {
    [GRAPPLE_UTILITY] = 2.0f,
    [MINE_UTILITY] = 2.5f,
};

static const float evaderUtilityValue[_NUM_UTILITIES] = {
    [GRAPPLE_UTILITY] = 3.0f,
    [MINE_UTILITY] = 1.0f,
};

static const float camperUtilityValue[_NUM_UTILITIES] = {
    [GRAPPLE_UTILITY] = 1.0f,
    [MINE_UTILITY] = 2.0f,
};

static const float rusherUtilityValue[_NUM_UTILITIES] = {
    [GRAPPLE_UTILITY] = 3.0f,
    [MINE_UTILITY] = 0.5f,
};

static const float trapperUtilityValue[_NUM_UTILITIES] = {
    [GRAPPLE_UTILITY] = 1.0f,
    [MINE_UTILITY] = 5.0f,
};

static const float turtleUtilityValue[_NUM_UTILITIES] = {
    [GRAPPLE_UTILITY] = 0.5f,
    [MINE_UTILITY] = 3.0f,
};

static const float fundamentalsUtilityValue[_NUM_UTILITIES] = {0};

typedef struct botPreset {
    // default feature mask, see enum botFeature
    uint32_t feats;
    // how many seconds of fighting a weapon pickup is worth
    const float *weaponValue;
    // scales the ideal range of short and mid range weapons
    float rangeScale;
    // fire against the direction of movement to move faster, see recoilBoost
    bool speedFirst;
    // go for pickups worth more than the held weapon; otherwise the bot
    // keeps fighting with what it has once it has a pickup weapon
    bool replacesWeapon;
    // how many seconds of fighting a utility pickup is worth
    const float *utilityValue;
    // with BOT_FEAT_UTILITY, grapple the enemy to close in on it, and
    // grapple a wall away from a close enemy to escape it
    bool grappleCloses;
    bool grappleEscapes;
} botPreset;

static const botPreset botPresets[NUM_BOT_POLICIES] = {
    [BOT_SITTING_DUCK] = {BOT_FEATURES_DEFAULT, botWeaponValue, 1.0f, false, true, botUtilityValue, true, true},
    [BOT_SCRIPTED] = {BOT_FEATURES_DEFAULT, botWeaponValue, 1.0f, false, true, botUtilityValue, true, true},
    [BOT_EVADER] = {BOT_FEAT_PROJ_DEFENSE | BOT_FEAT_SNIPER_DODGE | BOT_FEATURES_BASE, evaderWeaponValue, 1.0f, true, true, evaderUtilityValue, false, true},
    [BOT_CAMPER] = {BOT_FEAT_AMMO_RESERVE | BOT_FEAT_SNIPER_DODGE | BOT_FEATURES_BASE, camperWeaponValue, 1.0f, false, true, camperUtilityValue, false, true},
    [BOT_RUSHER] = {BOT_FEAT_BURST | BOT_FEAT_BURST_BOOST | BOT_FEAT_KILL_GEOMETRY | BOT_FEAT_SHOTGUN_BOOST | BOT_FEAT_SHOTGUN_RANGE | BOT_FEATURES_BASE, rusherWeaponValue, 0.4f, true, false, rusherUtilityValue, true, false},
    [BOT_TRAPPER] = {BOT_FEAT_AMMO_RESERVE | BOT_FEAT_PROJ_DEFENSE | BOT_FEATURES_BASE, trapperWeaponValue, 1.0f, false, false, trapperUtilityValue, false, true},
    [BOT_TURTLE] = {BOT_FEAT_AMMO_RESERVE | BOT_FEAT_PROJ_DEFENSE | BOT_FEAT_SNIPER_DODGE | BOT_FEATURES_BASE, botWeaponValue, 1.0f, false, true, turtleUtilityValue, false, false},
    [BOT_FUNDAMENTALS] = {BOT_FEAT_AMMO_RESERVE | BOT_FEATURES_BASE, fundamentalsWeaponValue, 1.0f, false, false, fundamentalsUtilityValue, false, false},
};

// sets the bot policy and gives every drone that personality and its
// default features
void setBotPersonality(iwEnv *e, const enum botPolicy policy) {
    e->botPolicy = policy;
    for (uint8_t i = 0; i < _MAX_DRONES; i++) {
        e->botPersonality[i] = policy == BOT_SITTING_DUCK ? BOT_SCRIPTED : policy;
        e->botFeatures[i] = botPresets[policy].feats;
    }
}

// speed first personalities stop firing to speed up past this speed in the
// direction they're moving
static const float BOOST_TOP_SPEED = 30.0f;
// fire at the enemy instead of straight back when it's within this angle
// of the boost direction
static const float BOOST_AIM_COS = 0.766f;
// keep this far inside the area the next sudden death walls leave open
static const float SUDDEN_DEATH_MARGIN = DRONE_RADIUS + 2.0f;
static const float SUDDEN_DEATH_AVOID_DISTANCE = 4.0f;
// seconds before the next sudden death walls land when getting clear of
// them overrides everything else
static const float SUDDEN_DEATH_URGENT_TIME = 2.0f;
static const float EVADER_DANGER_DISTANCE = 10.0f;
// the evader plans against where the enemy will be this far ahead
static const float EVADER_PREDICT_TIME = 1.0f;
// bonus per open cell around a spot, so the evader doesn't run into
// corners and dead ends it can't get out of
static const float EVADER_OPENNESS_BONUS = 1.0f;
#define EVADER_OPENNESS_RADIUS 2
// keep heading for the current spot unless another is this much better
static const float EVADER_STICKINESS = 5.0f;
// an enemy this close and closing in at least this fast makes the evader
// leave a spot it's holding
static const float EVADER_THREAT_DISTANCE = 30.0f;
static const float EVADER_THREAT_CLOSING_SPEED = 3.0f;
static const float EVADER_THREAT_SPOT_PENALTY = 15.0f;
static const float EVADER_WALL_PENALTY = 15.0f;
// the evader takes a covered spot over an exposed one this much farther
// from the enemy
static const float EVADER_COVER_BONUS = 20.0f;
// cells the enemy would reach first are only used as a last resort
static const float EVADER_CONTESTED_PENALTY = 50.0f;
// small preference for spots that are quicker to reach
static const float EVADER_TRAVEL_COST = 0.1f;
static const float CAMPER_MIN_RANGE = 15.0f;
static const float CAMPER_SPOT_MIN_RANGE = 25.0f;
static const float CAMPER_SPOT_MAX_RANGE = 50.0f;
static const float CAMPER_SPOT_SEARCH_DISTANCE = 30.0f;
static const float CAMPER_SHOT_WAIT = 3.0f;
static const float SPOT_REACHED_DISTANCE = 3.0f;
static const float SPOT_WALL_DISTANCE = 4.0f;
static const float TRAPPER_MINE_RANGE = 30.0f;
static const float TRAPPER_RANGE = 25.0f;
static const float TRAPPER_MIN_RANGE = 10.0f;
static const float TRAPPER_MAX_RANGE = 30.0f;
static const float TRAPPER_NUKE_MIN_RANGE = 20.0f;
static const float TURTLE_COUNTER_TIME = 1.0f;
static const float TURTLE_DANGER_DISTANCE = 12.0f;

static const float PICKUP_TRAVEL_SPEED = 12.0f;
static const float PICKUP_WALL_RISK_DISTANCE = 4.0f;
static const float PICKUP_COMMIT_TIME = 3.0f;
static const float HAZARD_CHECK_DISTANCE = 15.0f;
static const float HAZARD_PREDICT_TIME = 0.5f;
static const float HAZARD_SAFE_CLEARANCE = 2.5f;
static const float MINE_HAZARD_MARGIN = 2.0f;
static const float KILL_WALL_DISTANCE = 15.0f;
static const float BURST_KILL_WALL_DISTANCE = 10.0f;
static const float BURST_ENERGY_RESERVE = 0.5f;
static const float BOOST_ENERGY_RESERVE = 0.4f;
static const float THREAT_HORIZON = 2.0f;
static const float NUKE_THREAT_HORIZON = 5.0f;
static const float THREAT_BURST_TIME = 0.35f;
static const float SNIPER_TELL_CHARGE = 0.5f;
static const float SNIPER_URGENT_CHARGE = 0.8f;
static const float DODGE_HOLD_TIME = 0.5f;
static const float SNIPER_AIM_COS = 0.966f;
// only replace a held weapon with a pickup worth this much more
static const float PICKUP_REPLACE_MARGIN = 2.0f;
// mines go off for their owner too, so they're placed and set off at least
// this far away, outside of the blast
static const float MINE_SELF_SAFE_DISTANCE = MINE_LAUNCHER_EXPLOSION_RADIUS + 2.0f;
// own mines are avoided this close, inside of which their blast is strong
static const float OWN_MINE_HAZARD_RADIUS = MINE_LAUNCHER_EXPLOSION_RADIUS - 3.0f;
// bots that close in on the enemy mine the walls next to it, as they'd move
// through a mine on the way; the rest mine the middle of the way
static const float MINE_CLOSING_PATH_FRACTION = 1.0f;
// mines are only placed with the enemy at least this far away and none of
// its projectiles this close, so its shots don't set them off next to the
// drone; with a line to the enemy they aren't fired this close to straight
// at it either, where its shots come from
static const float MINE_PLACE_MIN_ENEMY_DISTANCE = 20.0f;
static const float MINE_PLACE_PROJECTILE_CLEARANCE = 20.0f;
static const float MINE_PLACE_ENEMY_AIM_COS = 0.866f;
// the drone steers away from its own welded mine this close, outside of
// which it won't set the mine off itself
static const float OWN_MINE_AVOID_DISTANCE = MINE_LAUNCHER_PROXIMITY_RADIUS + 4.0f;
// bursts reach this far, and set off mines they hit
static const float MINE_BURST_DISTANCE = DRONE_BURST_RADIUS_BASE + DRONE_BURST_RADIUS_MIN + MINE_LAUNCHER_RADIUS + 1.0f;
// with the enemy this much further than the trigger radius from the
// drone's own mine, the drone gets out of the mine's blast
static const float MINE_ENEMY_NEAR_MARGIN = 15.0f;
// own mines are set off when one is this close to the enemy, and only if
// the drone is outside of its blast
static const float MINE_DETONATE_DISTANCE = 9.0f;
// grapple the enemy when it's this much further than the weapon's ideal
// range, and stop reeling once it's within the ideal range or this close
static const float GRAPPLE_CLOSE_MARGIN = 15.0f;
static const float GRAPPLE_CLOSE_MIN_DISTANCE = 25.0f;
static const float GRAPPLE_CLOSE_STOP_DISTANCE = 12.0f;
static const float GRAPPLE_CLOSE_TIME = 2.5f;
// grapple a wall at least GRAPPLE_ESCAPE_MIN_ANCHOR away from an enemy this
// close when low on energy, or always for the evader, and let go when near
// the wall or once the enemy is far enough
static const float GRAPPLE_ESCAPE_ENEMY_DISTANCE = 12.0f;
static const float GRAPPLE_ESCAPE_ENERGY = 0.35f;
static const float GRAPPLE_ESCAPE_MIN_ANCHOR = 15.0f;
static const float GRAPPLE_ESCAPE_DONE_ANCHOR = 6.0f;
static const float GRAPPLE_ESCAPE_DONE_ENEMY = 30.0f;
static const float GRAPPLE_ESCAPE_TIME = 2.0f;
// let go of the grapple when it would take this little time to hit a death
// wall this close at the current speed towards it
static const float GRAPPLE_DANGER_WALL_DISTANCE = 8.0f;
static const float GRAPPLE_DANGER_WALL_TIME = 0.5f;
// with BOT_FEAT_SAFE_GEOMETRY, death walls this close behind the drone as
// seen from the enemy push it sideways and away
static const float SAFE_GEOMETRY_WALL_DISTANCE = 15.0f;
static const float SAFE_GEOMETRY_COS = 0.3f;
// with BOT_FEAT_KITE, fight an enemy with a shotgun drawn from at least this
// far away
static const float KITE_RANGE = 30.0f;
// with BOT_FEAT_SHOTGUN_BOOST, shotgun shots at an enemy this close are
// boosted by a burst charged this much; the smallest burst speeds pellets
// up almost as much as a charged one for a fraction of the energy
static const float SHOTGUN_BOOST_RANGE = 35.0f;
static const float SHOTGUN_BOOST_CHARGE = 0.0f;
// with BOT_FEAT_SHOTGUN_RANGE, the shotgun is only fired this close
static const float SHOTGUN_FIRE_RANGE = 35.0f;
// with BOT_FEAT_SMART_RESERVE this fraction of a special weapon's ammo is
// saved for attacking, the rest may be spent closing distance
static const float SMART_RESERVE_FRACTION = 0.5f;
// cost of moving through a cell next to a death wall, in cells
static const float PATH_DEATH_WALL_COST = 3.0f;
// cost of pushing through a non death floating wall, in cells
static const float PATH_FLOATING_WALL_COST = 4.0f;
// clearance around floating walls when marking the cells they block
static const float PATH_FLOATING_WALL_MARGIN = DRONE_RADIUS;
// only plan around floating walls this close to the drone
static const float PATH_FLOATING_WALL_CHECK_DISTANCE = 20.0f;
// a planned route is reused for this long while the drone stays in the
// same cell heading to the same destination
static const float PATH_REPLAN_TIME = 0.5f;
// how many cells ahead along the path the drone may head straight for
static const uint8_t PATH_LOOKAHEAD = BOT_PATH_WAYPOINTS;

typedef struct hazard {
    // static walls are axis aligned boxes so their distance is cheap;
    // floating walls need a shape distance query
    const wallEntity *floatingWall;
    b2Vec2 pos;
    float radius;
    bool isStaticWall;
} hazard;

#define MAX_HAZARDS 24

typedef struct botCtx {
    iwEnv *e;
    droneEntity *drone;
    uint32_t feats;
    scriptedBotState *state;
    enum botPolicy personality;
    const botPreset *preset;

    droneEntity *enemy;
    float enemyDistSq;
    float enemyDist;
    b2Vec2 enemyDir;
    // where the enemy is predicted to be shortly
    b2Vec2 enemyPredPos;
    // area clear of the next sudden death walls, if sudden death is active
    bool suddenDeath;
    mapBounds suddenDeathSafe;
    float idealRangeSquared;

    bool hazardsFound;
    hazard hazards[MAX_HAZARDS];
    uint8_t numHazards;
} botCtx;

static inline bool botHas(const botCtx *c, const uint32_t feature) {
    return (c->feats & feature) != 0;
}

static inline void botEvent(const botCtx *c, const enum botEvent event) {
    c->e->botEvents[c->drone->idx][event]++;
}

static inline uint16_t botStep(const iwEnv *e) {
    return e->episodeLength;
}

// sudden death walls fill one ring of map cells at a time, ring 0 being the
// outer wall. Returns the ring the next walls will fill, or -1 if sudden
// death isn't active or no more walls will be placed
static int8_t nextSuddenDeathRing(const iwEnv *e) {
    // matches when stepEnv places sudden death walls
    if (e->stepsLeft != 0 || (e->isTraining && NUM_DRONES != e->numAgents)) {
        return -1;
    }
    if (e->suddenDeathWallCounter >= e->map->maxSuddenDeathWalls) {
        return -1;
    }
    return e->suddenDeathWallCounter + 1;
}

// the area that will still be open after the next sudden death walls are
// placed, shrunk by margin; returns false if sudden death isn't a threat
static bool suddenDeathSafeArea(const iwEnv *e, const float margin, mapBounds *safe) {
    const int8_t ring = nextSuddenDeathRing(e);
    if (ring == -1) {
        return false;
    }
    const int16_t cols = e->map->columns;
    const int16_t rows = e->map->rows;
    const int16_t first = min(ring + 1, min((cols - 1) / 2, (rows - 1) / 2));
    const b2Vec2 a = e->cells[cellIndex(e, first, first)].pos;
    const b2Vec2 b = e->cells[cellIndex(e, cols - 1 - first, rows - 1 - first)].pos;
    const float halfExtent = WALL_THICKNESS / 2.0f;
    safe->min = (b2Vec2){.x = fminf(a.x, b.x) - halfExtent + margin, .y = fminf(a.y, b.y) - halfExtent + margin};
    safe->max = (b2Vec2){.x = fmaxf(a.x, b.x) + halfExtent - margin, .y = fmaxf(a.y, b.y) + halfExtent - margin};
    // too small to keep the margin, aim for the middle
    if (safe->min.x > safe->max.x) {
        safe->min.x = safe->max.x = (safe->min.x + safe->max.x) / 2.0f;
    }
    if (safe->min.y > safe->max.y) {
        safe->min.y = safe->max.y = (safe->min.y + safe->max.y) / 2.0f;
    }
    return true;
}

static inline b2Vec2 clampToBounds(const mapBounds *bounds, const b2Vec2 pos) {
    return (b2Vec2){
        .x = fminf(fmaxf(pos.x, bounds->min.x), bounds->max.x),
        .y = fminf(fmaxf(pos.y, bounds->min.y), bounds->max.y),
    };
}

// true if pos won't be caught by the next sudden death walls
static bool safeFromSuddenDeath(const botCtx *c, const b2Vec2 pos) {
    if (!c->suddenDeath) {
        return true;
    }
    return b2Equal(clampToBounds(&c->suddenDeathSafe, pos), pos);
}

static inline float burstRadius(const float charge) {
    return (DRONE_BURST_RADIUS_BASE * charge) + DRONE_BURST_RADIUS_MIN;
}

static inline float droneDamping(const droneEntity *drone) {
    if (drone->braking) {
        return DRONE_LINEAR_DAMPING * DRONE_BRAKE_DAMPING_COEF;
    }
    return DRONE_LINEAR_DAMPING;
}

// fraction of a velocity covered after time seconds with linear damping
static inline float dampedTime(const float linearDamping, const float time) {
    if (linearDamping == 0.0f) {
        return time;
    }
    return (1.0f - expf(-linearDamping * time)) / linearDamping;
}

// time for a damped projectile to travel distance, or -1 if it never will
static inline float dampedTravelTime(const float distance, const float speed, const float linearDamping) {
    if (linearDamping == 0.0f) {
        return distance / speed;
    }
    const float x = distance * linearDamping / speed;
    if (x >= 0.99f) {
        return -1.0f;
    }
    return -logf(1.0f - x) / linearDamping;
}

// smallest positive t where |relPos + relVel * t| = speed * t, or -1
float interceptTime(const b2Vec2 relPos, const b2Vec2 relVel, const float speed) {
    const float a = b2Dot(relVel, relVel) - SQUARED(speed);
    const float b = 2.0f * b2Dot(relPos, relVel);
    const float c = b2Dot(relPos, relPos);
    if (fabsf(a) < 1e-6f) {
        if (b >= 0.0f) {
            return -1.0f;
        }
        return -c / b;
    }
    const float disc = SQUARED(b) - (4.0f * a * c);
    if (disc < 0.0f) {
        return -1.0f;
    }
    const float sq = sqrtf(disc);
    const float t1 = (-b - sq) / (2.0f * a);
    const float t2 = (-b + sq) / (2.0f * a);
    float t = -1.0f;
    if (t1 > 0.0f) {
        t = t1;
    }
    if (t2 > 0.0f && (t < 0.0f || t2 < t)) {
        t = t2;
    }
    return t;
}

// finds the direction to fire the current weapon so its projectile meets a
// target moving with targetVel and targetDamping. Projectile damping and the
// lateral shooter velocity projectiles inherit are included, which a closed
// form intercept can't handle, so this iterates on the impact time starting
// from the undamped solution. Returns false if the target can't be reached.
bool interceptAim(const droneEntity *drone, const b2Vec2 targetPos, const b2Vec2 targetVel, const float targetDamping, b2Vec2 *aim, float *distance) {
    const weaponInformation *weapon = drone->weapon.weaponInfo;
    const float speed = weapon->initialSpeed;
    // lateral velocity is added as an impulse scaled by density, so the
    // resulting velocity only depends on the projectile's area
    const float lateralCoef = DRONE_MOVE_AIM_COEF / (PI * SQUARED(weapon->radius));
    float droneRadius = DRONE_RADIUS;
    if (drone->shield != NULL) {
        droneRadius = DRONE_SHIELD_RADIUS;
    }
    const float spawnOffset = droneRadius + (weapon->radius * 1.5f);

    const b2Vec2 relPos = b2Sub(targetPos, drone->pos);
    float t = interceptTime(relPos, targetVel, speed);
    if (t < 0.0f) {
        t = b2Length(relPos) / speed;
    }
    b2Vec2 dir = b2Normalize(relPos);
    float dist = b2Length(relPos);
    for (uint8_t i = 0; i < 4; i++) {
        const b2Vec2 forward = b2MulSV(b2Dot(drone->velocity, dir), dir);
        const b2Vec2 lateral = b2MulSV(lateralCoef, b2Sub(drone->velocity, forward));
        const float projTime = dampedTime(weapon->damping, t);
        const b2Vec2 predicted = b2MulAdd(targetPos, dampedTime(targetDamping, t), targetVel);
        const b2Vec2 toPredicted = b2Sub(b2MulAdd(predicted, -projTime, lateral), drone->pos);
        dist = b2Length(toPredicted);
        if (dist < 1e-3f) {
            break;
        }
        dir = b2MulSV(1.0f / dist, toPredicted);
        const float newT = dampedTravelTime(fmaxf(dist - spawnOffset, 0.0f), speed, weapon->damping);
        if (newT < 0.0f) {
            return false;
        }
        t = newT;
    }

    *aim = dir;
    *distance = dist;
    return true;
}

// casts the current weapon's projectile along aim; returns true if it
// reaches distance without hitting a wall, or first hits the target drone
bool shotPathClear(const botCtx *c, const b2Vec2 aim, const float distance, const droneEntity *target) {
    const droneEntity *drone = c->drone;
    const b2Vec2 translation = b2MulSV(distance, aim);
    const b2ShapeProxy cirProxy = b2MakeProxy(&drone->pos, 1, drone->weapon.weaponInfo->radius);
    uint64_t mask = WALL_SHAPE | FLOATING_WALL_SHAPE;
    if (target != NULL) {
        mask |= DRONE_SHAPE;
    }
    const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = mask};

    castCircleCtx ctx = {0};
    b2World_CastShape(c->e->worldID, &cirProxy, translation, filter, castCircleCallback, &ctx);
    if (!ctx.hit) {
        return true;
    }
    const entity *ent = b2Shape_GetUserData(ctx.shapeID);
    if (ent == NULL || ent->type != DRONE_ENTITY || target == NULL) {
        return false;
    }
    const droneEntity *hitDrone = ent->entity;
    return hitDrone->idx == target->idx;
}

// distance from a point to the closest point of a static wall
static inline float staticWallDistance(const b2Vec2 wallPos, const b2Vec2 pos) {
    const float halfExtent = WALL_THICKNESS / 2.0f;
    const float dx = fmaxf(fabsf(pos.x - wallPos.x) - halfExtent, 0.0f);
    const float dy = fmaxf(fabsf(pos.y - wallPos.y) - halfExtent, 0.0f);
    return sqrtf(SQUARED(dx) + SQUARED(dy));
}

static float floatingWallDistance(const wallEntity *wall, const b2Vec2 pos) {
    bool isCircle = false;
    b2DistanceInput input;
    input.proxyA = b2MakeProxy(&pos, 1, 0.0f);
    input.proxyB = makeDistanceProxy(wall->ent, &isCircle);
    input.transformA = b2Transform_identity;
    input.transformB = entityTransform(wall->ent);
    input.useRadii = false;
    b2SimplexCache cache = {0};
    return b2ShapeDistance(&input, &cache, NULL, 0).distance;
}

// collects death walls and enemy mines stuck to walls near the drone
void findHazards(botCtx *c) {
    if (c->hazardsFound) {
        return;
    }
    c->hazardsFound = true;
    iwEnv *e = c->e;
    const droneEntity *drone = c->drone;

    if (drone->mapCellIdx != -1) {
        nearWall nearWalls[MAX_NEAREST_WALLS] = {0};
        findNearWalls(e, drone, nearWalls, MAX_NEAREST_WALLS);
        for (uint8_t i = 0; i < MAX_NEAREST_WALLS && c->numHazards < MAX_HAZARDS; i++) {
            if (nearWalls[i].type != DEATH_WALL_ENTITY) {
                continue;
            }
            c->hazards[c->numHazards++] = (hazard){.pos = nearWalls[i].pos, .isStaticWall = true};
        }
    }
    for (uint8_t i = 0; i < cc_array_size(e->floatingWalls) && c->numHazards < MAX_HAZARDS; i++) {
        const wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        if (wall->type != DEATH_WALL_ENTITY || b2DistanceSquared(wall->pos, drone->pos) > SQUARED(HAZARD_CHECK_DISTANCE)) {
            continue;
        }
        c->hazards[c->numHazards++] = (hazard){.floatingWall = wall, .pos = wall->pos};
    }
    for (uint8_t i = 0; i < cc_array_size(e->projectiles) && c->numHazards < MAX_HAZARDS; i++) {
        const projectileEntity *proj = safe_array_get_at(e->projectiles, i);
        // mines go off for their owner too, and the enemy can set off
        // the drone's own mines, so they're avoided further out
        if (proj->weaponInfo->type != MINE_LAUNCHER_WEAPON || !proj->welded || b2DistanceSquared(proj->pos, drone->pos) > SQUARED(HAZARD_CHECK_DISTANCE + OWN_MINE_HAZARD_RADIUS)) {
            continue;
        }
        float radius = MINE_LAUNCHER_PROXIMITY_RADIUS + MINE_HAZARD_MARGIN;
        if (proj->droneIdx == drone->idx) {
            radius = OWN_MINE_HAZARD_RADIUS;
        }
        c->hazards[c->numHazards++] = (hazard){.pos = proj->pos, .radius = radius};
    }
}

// distance from the drone's edge at pos to the closest hazard
float hazardClearance(botCtx *c, const b2Vec2 pos) {
    findHazards(c);
    float droneRadius = DRONE_RADIUS;
    if (c->drone->shield != NULL) {
        droneRadius = DRONE_SHIELD_RADIUS;
    }
    float clearance = FLT_MAX;
    for (uint8_t i = 0; i < c->numHazards; i++) {
        const hazard *h = &c->hazards[i];
        float dist;
        if (h->isStaticWall) {
            dist = staticWallDistance(h->pos, pos);
        } else if (h->floatingWall != NULL) {
            dist = floatingWallDistance(h->floatingWall, pos);
        } else {
            dist = b2Distance(h->pos, pos) - h->radius;
        }
        clearance = fminf(clearance, dist - droneRadius);
    }
    return clearance;
}

// where the drone will be after time seconds moving in direction move
// with an optional extra velocity change
b2Vec2 predictDronePos(const droneEntity *drone, const b2Vec2 move, const b2Vec2 deltaVel, const float time) {
    const float damping = droneDamping(drone);
    const b2Vec2 vel = b2Add(drone->velocity, deltaVel);
    b2Vec2 pos = b2MulAdd(drone->pos, dampedTime(damping, time), vel);
    if (!b2Equal(move, b2Vec2_zero)) {
        pos = b2MulAdd(pos, forcedDistance(DRONE_MOVE_MAGNITUDE * DRONE_INV_MASS, damping, time), move);
    }
    return pos;
}

// finds a drone's closest death wall; returns false if none is within
// maxDistance
bool droneDeathWall(iwEnv *e, const droneEntity *drone, const float maxDistance, b2Vec2 *wallDir, float *wallDistance) {
    float best = maxDistance;
    b2Vec2 bestPoint = b2Vec2_zero;
    bool found = false;
    if (drone->mapCellIdx != -1) {
        nearWall nearWalls[MAX_NEAREST_WALLS] = {0};
        findNearWalls(e, drone, nearWalls, MAX_NEAREST_WALLS);
        const float halfExtent = WALL_THICKNESS / 2.0f;
        for (uint8_t i = 0; i < MAX_NEAREST_WALLS; i++) {
            if (nearWalls[i].type != DEATH_WALL_ENTITY) {
                continue;
            }
            const float dist = staticWallDistance(nearWalls[i].pos, drone->pos);
            if (dist < best) {
                best = dist;
                bestPoint.x = fminf(fmaxf(drone->pos.x, nearWalls[i].pos.x - halfExtent), nearWalls[i].pos.x + halfExtent);
                bestPoint.y = fminf(fmaxf(drone->pos.y, nearWalls[i].pos.y - halfExtent), nearWalls[i].pos.y + halfExtent);
                found = true;
            }
        }
    }
    for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        const wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        if (wall->type != DEATH_WALL_ENTITY || b2DistanceSquared(wall->pos, drone->pos) > SQUARED(maxDistance + FLOATING_WALL_THICKNESS)) {
            continue;
        }
        const b2DistanceOutput output = closestPoint(drone->ent, wall->ent);
        if (output.distance < best) {
            best = output.distance;
            bestPoint = output.pointB;
            found = true;
        }
    }
    if (!found) {
        return false;
    }
    *wallDir = b2Normalize(b2Sub(bestPoint, drone->pos));
    *wallDistance = best;
    return true;
}

// finds the enemy's closest death wall; returns false if none is within maxDistance
bool enemyDeathWall(botCtx *c, const float maxDistance, b2Vec2 *wallDir, float *wallDistance) {
    return droneDeathWall(c->e, c->enemy, maxDistance, wallDir, wallDistance);
}

// true if pushing the enemy away from the drone sends it toward a death wall
bool killGeometryGood(botCtx *c, const float maxWallDistance) {
    b2Vec2 wallDir;
    float wallDistance;
    if (!enemyDeathWall(c, maxWallDistance, &wallDir, &wallDistance)) {
        return false;
    }
    return b2Dot(c->enemyDir, wallDir) > 0.5f;
}

bool weaponSafeForMovement(const droneEntity *drone) {
    switch (drone->weapon.weaponInfo->type) {
    case IMPLODER_WEAPON:
    case BLACK_HOLE_WEAPON:
    case NUKE_WEAPON:
        return false;
    default:
        return true;
    }
}

// special weapon ammo is saved for attacking instead of being spent on
// recoil movement while an enemy is alive
bool reservingAmmo(const botCtx *c) {
    const droneEntity *drone = c->drone;
    if (!botHas(c, BOT_FEAT_AMMO_RESERVE) || c->enemy == NULL || drone->weapon.ammo == INFINITE) {
        return false;
    }
    // ammo above the reserve may be spent closing in on an enemy that's
    // out of range
    if (botHas(c, BOT_FEAT_SMART_RESERVE) && c->enemyDistSq > c->idealRangeSquared) {
        const int8_t maxAmmo = weaponInfos[drone->weapon.weaponInfo->type]->ammo;
        if (drone->weapon.ammo > maxAmmo * SMART_RESERVE_FRACTION) {
            botEvent(c, BOT_EVENT_SPARE_AMMO_MOVE);
            return false;
        }
    }
    botEvent(c, BOT_EVENT_AMMO_RESERVED);
    return true;
}

bool recoilMoveAllowed(const botCtx *c) {
    return weaponSafeForMovement(c->drone) && !reservingAmmo(c);
}

void scriptedAgentShoot(const droneEntity *drone, agentActions *actions) {
    // release a full charge to fire it, otherwise hold the trigger
    const weaponSlot *weapon = &drone->weapon;
    actions->trigger = !(weapon->weaponInfo->maxCharge != 0.0f && weapon->charge == weapon->weaponInfo->maxCharge);
}

static bool segmentClear(const iwEnv *e, const b2Vec2 start, const b2Vec2 end, const float radius) {
    const b2QueryFilter filter = {.categoryBits = DRONE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
    const b2Vec2 translation = b2Sub(end, start);
    if (radius == 0.0f) {
        return !b2World_CastRayClosest(e->worldID, start, translation, filter).hit;
    }
    const b2ShapeProxy proxy = b2MakeProxy(&start, 1, radius);
    castCircleCtx ctx = {0};
    b2World_CastShape(e->worldID, &proxy, translation, filter, castCircleCallback, &ctx);
    return !ctx.hit;
}

// head for the furthest of the next waypoints that can be reached in a
// straight line
static bool pathDirection(const botCtx *c, const int16_t *waypoints, const uint8_t numWaypoints, b2Vec2 *moveDir) {
    if (numWaypoints == 0) {
        return false;
    }
    const iwEnv *e = c->e;
    const droneEntity *drone = c->drone;
    b2Vec2 target = e->cells[waypoints[0]].pos;
    for (uint8_t k = 1; k < numWaypoints; k++) {
        const b2Vec2 candidate = e->cells[waypoints[k]].pos;
        if (!segmentClear(e, drone->pos, candidate, DRONE_RADIUS)) {
            break;
        }
        target = candidate;
    }
    const b2Vec2 toTarget = b2Sub(target, drone->pos);
    if (b2LengthSquared(toTarget) < 1e-4f) {
        return false;
    }
    *moveDir = b2Normalize(toTarget);
    return true;
}

typedef struct pathNode {
    float cost;
    uint16_t cellIdx;
} pathNode;

static void pathHeapPush(pathNode *heap, uint16_t *size, const pathNode node) {
    uint16_t i = (*size)++;
    while (i > 0) {
        const uint16_t parent = (i - 1) / 2;
        if (heap[parent].cost <= node.cost) {
            break;
        }
        heap[i] = heap[parent];
        i = parent;
    }
    heap[i] = node;
}

static pathNode pathHeapPop(pathNode *heap, uint16_t *size) {
    const pathNode top = heap[0];
    const pathNode last = heap[--(*size)];
    uint16_t i = 0;
    while (true) {
        uint16_t child = (2 * i) + 1;
        if (child >= *size) {
            break;
        }
        if (child + 1 < *size && heap[child + 1].cost < heap[child].cost) {
            child++;
        }
        if (last.cost <= heap[child].cost) {
            break;
        }
        heap[i] = heap[child];
        i = child;
    }
    heap[i] = last;
    return top;
}

static inline float octileDistance(const int16_t col, const int16_t row, const int16_t dstCol, const int16_t dstRow) {
    const float dx = fabsf((float)(col - dstCol));
    const float dy = fabsf((float)(row - dstRow));
    return fmaxf(dx, dy) + (0.41421356f * fminf(dx, dy));
}

static inline bool circleOverlapsCell(const b2Vec2 cellPos, const b2Vec2 pos, const float radius) {
    const float halfExtent = WALL_THICKNESS / 2.0f;
    const float dx = fmaxf(fabsf(pos.x - cellPos.x) - halfExtent, 0.0f);
    const float dy = fmaxf(fabsf(pos.y - cellPos.y) - halfExtent, 0.0f);
    return SQUARED(dx) + SQUARED(dy) < SQUARED(radius);
}

static bool floatingWallNear(const iwEnv *e, const droneEntity *drone) {
    for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        const wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        if (b2DistanceSquared(wall->pos, drone->pos) <= SQUARED(PATH_FLOATING_WALL_CHECK_DISTANCE)) {
            return true;
        }
    }
    return false;
}

// plans a route over the map grid that accounts for floating walls, which
// the precomputed flow field doesn't know about, and avoids hugging death
// walls. Returns false if no route was found.
bool planPath(botCtx *c, const b2Vec2 dstPos, b2Vec2 *moveDir) {
    iwEnv *e = c->e;
    const droneEntity *drone = c->drone;
    const uint8_t columns = e->map->columns;
    const uint8_t rows = e->map->rows;
    const uint16_t numCells = columns * rows;
    const int16_t srcIdx = drone->mapCellIdx;
    const int16_t dstIdx = entityPosToCellIdx(e, dstPos);
    if (srcIdx == -1 || dstIdx == -1) {
        return false;
    }
    scriptedBotState *state = c->state;
    if (state->pathSrcIdx == srcIdx && state->pathDstIdx == dstIdx && botStep(e) < state->pathUntilStep) {
        return pathDirection(c, state->pathWaypoints, state->pathNumWaypoints, moveDir);
    }

    bool blocked[MAX_CELLS];
    float extraCost[MAX_CELLS];
    for (uint16_t i = 0; i < numCells; i++) {
        const entity *ent = e->cells[i].ent;
        blocked[i] = ent != NULL && entityTypeIsWall(ent->type);
        extraCost[i] = 0.0f;
    }
    for (uint16_t i = 0; i < numCells; i++) {
        const entity *ent = e->cells[i].ent;
        if (ent == NULL || ent->type != DEATH_WALL_ENTITY) {
            continue;
        }
        const int16_t col = i % columns;
        const int16_t row = i / columns;
        for (int16_t dr = -1; dr <= 1; dr++) {
            for (int16_t dc = -1; dc <= 1; dc++) {
                const int16_t nc = col + dc, nr = row + dr;
                if (nc >= 0 && nc < columns && nr >= 0 && nr < rows) {
                    extraCost[nc + (nr * columns)] += PATH_DEATH_WALL_COST;
                }
            }
        }
    }
    const float floatingRadius = (FLOATING_WALL_THICKNESS * 0.70710678f) + PATH_FLOATING_WALL_MARGIN;
    for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        const wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        const int16_t wallIdx = entityPosToCellIdx(e, wall->pos);
        if (wallIdx == -1) {
            continue;
        }
        const int16_t col = wallIdx % columns;
        const int16_t row = wallIdx / columns;
        for (int16_t dr = -2; dr <= 2; dr++) {
            for (int16_t dc = -2; dc <= 2; dc++) {
                const int16_t nc = col + dc, nr = row + dr;
                if (nc < 0 || nc >= columns || nr < 0 || nr >= rows) {
                    continue;
                }
                const uint16_t idx = nc + (nr * columns);
                const b2Vec2 cellPos = e->cells[idx].pos;
                if (wall->type == DEATH_WALL_ENTITY) {
                    if (circleOverlapsCell(cellPos, wall->pos, floatingRadius)) {
                        blocked[idx] = true;
                    } else if (circleOverlapsCell(cellPos, wall->pos, floatingRadius + WALL_THICKNESS)) {
                        extraCost[idx] += PATH_DEATH_WALL_COST;
                    }
                } else if (circleOverlapsCell(cellPos, wall->pos, floatingRadius)) {
                    extraCost[idx] += PATH_FLOATING_WALL_COST;
                }
            }
        }
    }
    // a floating wall may overlap the drone's own cell or the destination's,
    // both need to stay reachable
    blocked[srcIdx] = false;
    blocked[dstIdx] = false;

    float cost[MAX_CELLS];
    int16_t prev[MAX_CELLS];
    bool done[MAX_CELLS];
    for (uint16_t i = 0; i < numCells; i++) {
        cost[i] = FLT_MAX;
        prev[i] = -1;
        done[i] = false;
    }
    pathNode heap[(MAX_CELLS) * 8];
    uint16_t heapSize = 0;
    // A* with the octile distance, which never overestimates since
    // extra costs are only ever added
    const int16_t dstCol = dstIdx % columns;
    const int16_t dstRow = dstIdx / columns;
    cost[srcIdx] = 0.0f;
    pathHeapPush(heap, &heapSize, (pathNode){.cost = octileDistance(srcIdx % columns, srcIdx / columns, dstCol, dstRow), .cellIdx = srcIdx});

    static const int8_t dirCol[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    static const int8_t dirRow[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    int16_t goalIdx = -1;
    while (heapSize > 0) {
        const pathNode node = pathHeapPop(heap, &heapSize);
        const uint16_t u = node.cellIdx;
        if (done[u]) {
            continue;
        }
        done[u] = true;

        if (u == dstIdx) {
            goalIdx = u;
            break;
        }

        const int16_t col = u % columns;
        const int16_t row = u / columns;
        for (uint8_t d = 0; d < 8; d++) {
            const int16_t nc = col + dirCol[d], nr = row + dirRow[d];
            if (nc < 0 || nc >= columns || nr < 0 || nr >= rows) {
                continue;
            }
            const uint16_t v = nc + (nr * columns);
            if (blocked[v] || done[v]) {
                continue;
            }
            float step = 1.0f;
            if (d >= 4) {
                // don't cut corners between walls
                if (blocked[nc + (row * columns)] || blocked[col + (nr * columns)]) {
                    continue;
                }
                step = 1.41421356f;
            }
            const float newCost = cost[u] + step + extraCost[v];
            if (newCost < cost[v]) {
                cost[v] = newCost;
                prev[v] = u;
                pathHeapPush(heap, &heapSize, (pathNode){.cost = newCost + octileDistance(nc, nr, dstCol, dstRow), .cellIdx = v});
            }
        }
    }
    if (goalIdx == -1 || goalIdx == srcIdx) {
        return false;
    }

    // walk back to the cells right after the drone's cell
    int16_t path[MAX_CELLS];
    uint16_t pathLen = 0;
    for (int16_t i = goalIdx; i != srcIdx && i != -1; i = prev[i]) {
        path[pathLen++] = i;
    }
    // path is reversed, keep the next few cells in order
    state->pathNumWaypoints = 0;
    for (uint8_t k = 1; k <= PATH_LOOKAHEAD && k <= pathLen; k++) {
        state->pathWaypoints[state->pathNumWaypoints++] = path[pathLen - k];
    }
    state->pathSrcIdx = srcIdx;
    state->pathDstIdx = dstIdx;
    state->pathUntilStep = botStep(e) + (uint16_t)(PATH_REPLAN_TIME * e->frameRate);
    return pathDirection(c, state->pathWaypoints, state->pathNumWaypoints, moveDir);
}

bool recoilBoost(botCtx *c, agentActions *actions, const b2Vec2 moveDir);

void moveTo(botCtx *c, agentActions *actions, b2Vec2 dstPos) {
    iwEnv *e = c->e;
    const droneEntity *drone = c->drone;
    ASSERT(drone->mapCellIdx != -1);
    // never head where the next sudden death walls will be
    if (c->suddenDeath) {
        dstPos = clampToBounds(&c->suddenDeathSafe, dstPos);
    }
    int16_t dstIdx = entityPosToCellIdx(e, dstPos);
    if (dstIdx == -1) {
        return;
    }

    // fully precomputed in initMaps; 8 means "no route"
    const uint8_t direction = e->map->paths[pathOffset(e, drone->mapCellIdx, dstIdx)];
    b2Vec2 moveDir = b2Vec2_zero;
    if (direction < 8) {
        moveDir = (b2Vec2){.x = discMoveToContMoveMap[0][direction], .y = discMoveToContMoveMap[1][direction]};
    }
    // the flow field only knows about static walls, so plan a route when
    // floating walls are around
    if (direction >= 8 || floatingWallNear(e, drone)) {
        b2Vec2 planned;
        if (planPath(c, dstPos, &planned)) {
            botEvent(c, BOT_EVENT_DYNAMIC_PATH);
            moveDir = planned;
        } else {
            botEvent(c, BOT_EVENT_DYNAMIC_PATH_FAILED);
        }
    }
    if (b2Equal(moveDir, b2Vec2_zero)) {
        return;
    }
    actions->move = b2Normalize(b2Add(actions->move, moveDir));
    // a shot was already chosen this step
    if (!b2Equal(actions->aim, b2Vec2_zero)) {
        return;
    }
    if (c->preset->speedFirst) {
        recoilBoost(c, actions, actions->move);
        return;
    }

    const b2Vec2 invDirection = b2MulSV(-1.0f, actions->move);
    if (!weaponSafeForMovement(drone) || reservingAmmo(c)) {
        return;
    }
    if (!safeToFireMoving(e, drone, invDirection, actions->move)) {
        return;
    }
    actions->aim = invDirection;
    scriptedAgentShoot(drone, actions);
}

float weaponTypeIdealRangeSquared(const enum weaponType type) {
    switch (type) {
    case STANDARD_WEAPON:
        return SQUARED(20.0f);
    case MACHINEGUN_WEAPON:
        return SQUARED(30.0f);
    case SHOTGUN_WEAPON:
        return SQUARED(20.0f);
    case IMPLODER_WEAPON:
        return SQUARED(30.0f);
    case FLAK_CANNON_WEAPON:
        return SQUARED(FLAK_CANNON_SAFE_DISTANCE + 5.0f);
    case SNIPER_WEAPON:
    case BLACK_HOLE_WEAPON:
    case NUKE_WEAPON:
        return FLT_MAX;

    default:
        ERRORF("unknown weapon type %d", type);
    }
}

float weaponIdealRangeSquared(const droneEntity *drone) {
    return weaponTypeIdealRangeSquared(drone->weapon.weaponInfo->type);
}

bool shouldShootAtEnemy(iwEnv *e, const droneEntity *drone, const droneEntity *enemyDrone, const b2Vec2 enemyDroneDirection) {
    // don't shoot at shielded enemies with railguns since it will likely
    // bounce back and hit us
    if (drone->weapon.weaponInfo->type == SNIPER_WEAPON && enemyDrone->shield != NULL) {
        return false;
    }
    if (!safeToFire(e, drone, enemyDroneDirection)) {
        return false;
    }

    // cast a circle that's the size of a projectile of the current weapon
    const float enemyDroneDistance = b2Distance(enemyDrone->pos, drone->pos);
    const b2Vec2 castEnd = b2MulAdd(drone->pos, enemyDroneDistance, enemyDroneDirection);
    const b2Vec2 translation = b2Sub(castEnd, drone->pos);
    const b2ShapeProxy cirProxy = b2MakeProxy(&drone->pos, 1, drone->weapon.weaponInfo->radius);
    const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE};

    castCircleCtx ctx = {0};
    b2World_CastShape(e->worldID, &cirProxy, translation, filter, castCircleCallback, &ctx);
    if (!ctx.hit) {
        return false;
    }
    ASSERT(b2Shape_IsValid(ctx.shapeID));
    const entity *ent = b2Shape_GetUserData(ctx.shapeID);
    if (ent == NULL || ent->type != DRONE_ENTITY) {
        return false;
    }
    const droneEntity *hitDrone = ent->entity;
    if (hitDrone->idx != enemyDrone->idx) {
        return false;
    }

    return true;
}

// can the current weapon's projectile travel distance before it's destroyed
bool shotReaches(const droneEntity *drone, const float distance) {
    const float maxDistance = drone->weapon.weaponInfo->maxDistance;
    return maxDistance == INFINITE || distance <= maxDistance;
}

// intercept aiming: solves for the firing direction and checks the
// path of the actual shot rather than the line to the enemy's current
// position
bool shouldShootAtEnemyIntercept(botCtx *c, b2Vec2 *aim) {
    const droneEntity *drone = c->drone;
    const droneEntity *enemy = c->enemy;
    if (drone->weapon.weaponInfo->type == SNIPER_WEAPON && enemy->shield != NULL) {
        return false;
    }
    float distance;
    if (!interceptAim(drone, enemy->pos, enemy->velocity, droneDamping(enemy), aim, &distance)) {
        return false;
    }
    if (!safeToFire(c->e, drone, *aim)) {
        return false;
    }
    return shotPathClear(c, *aim, distance + DRONE_RADIUS, enemy);
}

void scriptedAgentBurst(const droneEntity *drone, agentActions *actions) {
    if (drone->chargingBurst) {
        return;
    } else {
        actions->chargingBurst = true;
    }
}

static inline bool canBurst(const droneEntity *drone) {
    return drone->shield == NULL && !drone->energyFullyDepleted && drone->burstCooldown == 0.0f;
}

// velocity aware wall avoidance: steer away from the closest point on the
// wall, weighting closer and faster approaching walls more, and react to
// walls further away when the drone couldn't stop in time
void handleWallProximityVelocity(botCtx *c, const b2DistanceOutput *output, agentActions *actions) {
    const droneEntity *drone = c->drone;
    const float distance = output->distance;
    b2Vec2 wallDirection = output->normal;
    if (distance == 0.0f || b2Equal(wallDirection, b2Vec2_zero)) {
        wallDirection = b2Normalize(b2Sub(output->pointB, drone->pos));
    }
    const float approachSpeed = fmaxf(b2Dot(drone->velocity, wallDirection), 0.0f);
    const float brakingStop = approachSpeed / (DRONE_LINEAR_DAMPING * DRONE_BRAKE_DAMPING_COEF);
    const float coastingStop = approachSpeed / DRONE_LINEAR_DAMPING;
    const float urgency = 1.0f / fmaxf(distance, 0.25f);

    // surviving is worth spending reserved ammo on
    const bool canFire = weaponSafeForMovement(drone);
    if ((distance <= WALL_DANGER_DISTANCE || brakingStop > distance - 0.5f) && canFire && safeToFire(c->e, drone, wallDirection)) {
        if (!recoilMoveAllowed(c)) {
            botEvent(c, BOT_EVENT_ESCAPE_FIRE);
        }
        actions->aim = b2MulAdd(actions->aim, urgency, wallDirection);
        scriptedAgentShoot(drone, actions);
    }
    if (distance <= WALL_AVOID_DISTANCE || coastingStop > distance) {
        actions->move = b2MulAdd(actions->move, -(urgency + (0.1f * approachSpeed)), wallDirection);
    }
    if (brakingStop > distance - 1.0f && distance < WALL_BURST_CHECK_DISTANCE * 2.0f) {
        actions->brake = true;
    }
}

// pick the movement direction with the most predicted hazard clearance
// if the current one is getting too close
void steerForClearance(botCtx *c, agentActions *actions) {
    findHazards(c);
    if (c->numHazards == 0) {
        return;
    }
    const b2Vec2 move = b2Equal(actions->move, b2Vec2_zero) ? b2Vec2_zero : b2Normalize(actions->move);
    const float current = hazardClearance(c, predictDronePos(c->drone, move, b2Vec2_zero, HAZARD_PREDICT_TIME));
    if (current >= HAZARD_SAFE_CLEARANCE) {
        return;
    }
    float best = current;
    b2Vec2 bestMove = move;
    for (uint8_t i = 0; i < 8; i++) {
        const b2Vec2 dir = {.x = discMoveToContMoveMap[0][i], .y = discMoveToContMoveMap[1][i]};
        const float clearance = hazardClearance(c, predictDronePos(c->drone, dir, b2Vec2_zero, HAZARD_PREDICT_TIME));
        if (clearance > best + 0.1f) {
            best = clearance;
            bestMove = dir;
        }
    }
    if (!b2Equal(bestMove, move)) {
        botEvent(c, BOT_EVENT_CLEARANCE_STEER);
    }
    actions->move = bestMove;
}

// charge burst until we're close enough to a death wall to burst off
// of it
void wallBurst(iwEnv *e, const droneEntity *drone, const float distance, agentActions *actions) {
    if (distance < DRONE_BURST_RADIUS_MIN) {
        scriptedAgentBurst(drone, actions);
        return;
    }

    float damping = DRONE_LINEAR_DAMPING;
    if (drone->braking || actions->brake) {
        damping *= DRONE_BRAKE_DAMPING_COEF;
    }
    const float travelDistance = distanceWithDamping(e, b2Length(drone->velocity), damping, e->frameSkip);
    if (travelDistance > distance) {
        scriptedAgentBurst(drone, actions);
    } else {
        actions->chargingBurst = true;
    }
}

// effective danger radius and reaction horizon of hostile projectiles worth
// reacting to; everything else is ignored
bool projectileThreat(const projectileEntity *proj, float *radius, float *horizon, float *blastRadius) {
    *horizon = THREAT_HORIZON;
    switch (proj->weaponInfo->type) {
    case IMPLODER_WEAPON:
        *radius = DRONE_RADIUS + IMPLODER_RADIUS + 2.0f;
        *blastRadius = 15.0f;
        return true;
    case NUKE_WEAPON:
        *radius = DRONE_RADIUS + NUKE_RADIUS + 3.0f;
        *blastRadius = 35.0f;
        *horizon = NUKE_THREAT_HORIZON;
        return true;
    case BLACK_HOLE_WEAPON:
        *radius = BLACK_HOLE_PROXIMITY_RADIUS;
        *blastRadius = BLACK_HOLE_PROXIMITY_RADIUS;
        return true;
    default:
        return false;
    }
}

// move perpendicular to dangerDir, toward the side with more hazard clearance
b2Vec2 dodgeDirection(botCtx *c, const b2Vec2 dangerDir, const int8_t preferredSide) {
    const b2Vec2 left = b2LeftPerp(dangerDir);
    const b2Vec2 right = b2RightPerp(dangerDir);
    const float leftClear = hazardClearance(c, predictDronePos(c->drone, left, b2Vec2_zero, HAZARD_PREDICT_TIME));
    const float rightClear = hazardClearance(c, predictDronePos(c->drone, right, b2Vec2_zero, HAZARD_PREDICT_TIME));
    int8_t side = preferredSide;
    if (side == 0 || (side > 0 && leftClear < HAZARD_SAFE_CLEARANCE && rightClear > leftClear) || (side < 0 && rightClear < HAZARD_SAFE_CLEARANCE && leftClear > rightClear)) {
        side = leftClear >= rightClear ? 1 : -1;
    }
    c->state->dodgeSide = side;
    c->state->dodgeUntilStep = botStep(c->e) + (uint16_t)(DODGE_HOLD_TIME * c->e->frameRate);
    return side > 0 ? left : right;
}

// react to whitelisted dangerous projectiles: shoot them away, sidestep,
// and burst as a last resort. Returns true if the threat took over the
// drone's actions.
bool defendAgainstProjectiles(botCtx *c, agentActions *actions) {
    iwEnv *e = c->e;
    droneEntity *drone = c->drone;

    const projectileEntity *threat = NULL;
    float threatTime = FLT_MAX;
    float threatBlast = 0.0f;
    b2Vec2 threatMiss = b2Vec2_zero;
    for (uint8_t i = 0; i < cc_array_size(e->projectiles); i++) {
        const projectileEntity *proj = safe_array_get_at(e->projectiles, i);
        if (proj->droneIdx == drone->idx || proj->needsToBeDestroyed) {
            continue;
        }
        const droneEntity *owner = safe_array_get_at(e->drones, proj->droneIdx);
        if (owner->team == drone->team) {
            continue;
        }
        float radius, horizon, blast;
        if (!projectileThreat(proj, &radius, &horizon, &blast)) {
            continue;
        }
        const b2Vec2 relPos = b2Sub(proj->pos, drone->pos);
        const b2Vec2 relVel = b2Sub(proj->velocity, drone->velocity);
        const float relSpeedSq = b2Dot(relVel, relVel);
        if (relSpeedSq < 1.0f) {
            continue;
        }
        const float t = -b2Dot(relPos, relVel) / relSpeedSq;
        if (t < 0.0f || t > horizon || t >= threatTime) {
            continue;
        }
        const b2Vec2 miss = b2MulAdd(relPos, t, relVel);
        if (b2Length(miss) > radius) {
            continue;
        }
        threat = proj;
        threatTime = t;
        threatBlast = blast;
        threatMiss = miss;
    }

    if (threat == NULL) {
        return false;
    }
    botEvent(c, BOT_EVENT_THREAT);

    // 1. shoot it away if the collision would happen far enough away
    bool shotAtThreat = false;
    if (drone->weapon.cooldown <= 0.0f && drone->weapon.weaponInfo->maxCharge == 0.0f) {
        b2Vec2 aim;
        float distance;
        if (interceptAim(drone, threat->pos, threat->velocity, threat->weaponInfo->damping, &aim, &distance) && distance > threatBlast && safeToFire(e, drone, aim) && shotPathClear(c, aim, distance, NULL)) {
            actions->aim = aim;
            scriptedAgentShoot(drone, actions);
            shotAtThreat = true;
            botEvent(c, BOT_EVENT_THREAT_SHOT);
        }
    }

    // 2. sidestep away from the projectile's path
    b2Vec2 dodge;
    if (b2Length(threatMiss) > 0.25f) {
        dodge = b2Normalize(b2MulSV(-1.0f, threatMiss));
    } else {
        dodge = dodgeDirection(c, b2Normalize(threat->velocity), c->state->dodgeSide);
    }
    if (hazardClearance(c, predictDronePos(drone, dodge, b2Vec2_zero, HAZARD_PREDICT_TIME)) < HAZARD_SAFE_CLEARANCE) {
        dodge = b2MulSV(-1.0f, dodge);
    }
    if (1) {
        actions->move = dodge;
    }
    if (!shotAtThreat && recoilMoveAllowed(c)) {
        const b2Vec2 recoilAim = b2MulSV(-1.0f, dodge);
        if (safeToFireMoving(e, drone, recoilAim, dodge)) {
            actions->aim = recoilAim;
            scriptedAgentShoot(drone, actions);
        }
    }

    // 3. burst when nothing else will work in time
    if (threatTime < THREAT_BURST_TIME && threat->weaponInfo->type != NUKE_WEAPON && canBurst(drone) && drone->energyLeft >= DRONE_BURST_BASE_COST * 2.0f) {
        scriptedAgentBurst(drone, actions);
        botEvent(c, BOT_EVENT_THREAT_BURST);
    }

    return true;
}

// get off the line of an enemy charging a sniper; returns true if the
// drone should only dodge this step
bool dodgeSniper(botCtx *c, agentActions *actions) {
    const droneEntity *drone = c->drone;
    const droneEntity *enemy = c->enemy;
    if (enemy == NULL || enemy->weapon.weaponInfo->type != SNIPER_WEAPON || drone->shield != NULL) {
        return false;
    }
    const float charge = enemy->weapon.charge / SNIPER_CHARGE;
    if (charge < SNIPER_TELL_CHARGE) {
        return false;
    }
    // only a threat if the enemy can see us
    const b2Vec2 translation = b2Sub(drone->pos, enemy->pos);
    const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
    const b2RayResult ray = b2World_CastRayClosest(c->e->worldID, enemy->pos, translation, filter);
    if (ray.hit) {
        return false;
    }

    // the shot can only come along the enemy's aim
    const b2Vec2 lineDir = b2MulSV(-1.0f, c->enemyDir);
    if (b2Dot(enemy->lastAim, lineDir) < SNIPER_AIM_COS) {
        return false;
    }
    int8_t side = 0;
    if (botStep(c->e) < c->state->dodgeUntilStep) {
        side = c->state->dodgeSide;
    }
    const b2Vec2 dodge = dodgeDirection(c, lineDir, side);
    actions->move = dodge;
    botEvent(c, BOT_EVENT_SNIPER_DODGE);

    // urgent if the shot is about to come or a death wall is behind us
    // along the line of fire
    bool urgent = charge >= SNIPER_URGENT_CHARGE;
    if (!urgent) {
        const b2Vec2 behind = b2MulSV(25.0f, lineDir);
        const b2QueryFilter wallFilter = {.categoryBits = DRONE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
        const b2RayResult behindRay = b2World_CastRayClosest(c->e->worldID, drone->pos, behind, wallFilter);
        if (behindRay.hit) {
            const entity *ent = b2Shape_GetUserData(behindRay.shapeId);
            urgent = ent != NULL && ent->type == DEATH_WALL_ENTITY;
        }
    }
    if (!urgent) {
        return false;
    }
    botEvent(c, BOT_EVENT_SNIPER_URGENT);
    if (recoilMoveAllowed(c)) {
        const b2Vec2 recoilAim = b2MulSV(-1.0f, dodge);
        if (safeToFireMoving(c->e, drone, recoilAim, dodge)) {
            actions->aim = recoilAim;
            scriptedAgentShoot(drone, actions);
        }
    }
    return true;
}

static inline float ammoFraction(const weaponSlot *slot) {
    if (slot->ammo == INFINITE) {
        return 1.0f;
    }
    return (float)slot->ammo / (float)slot->weaponInfo->ammo;
}

// how much a held weapon is worth keeping when a pickup would replace it
static float slotWorth(const botCtx *c, const weaponSlot *slot) {
    const enum weaponType type = slot->weaponInfo->type;
    if (type == UNSET_WEAPON || type == STANDARD_WEAPON || slot->ammo == 0) {
        return 0.0f;
    }
    return c->preset->weaponValue[type] * (0.3f + (0.7f * ammoFraction(slot)));
}

// what picking up a weapon adds: a weapon in place of the standard one, ammo
// for the held weapon, or a better weapon in place of the held one
static float pickupGain(const botCtx *c, const enum weaponType type) {
    const float value = c->preset->weaponValue[type];
    if (value == 0.0f) {
        return 0.0f;
    }
    const weaponSlot *weapon = &c->drone->weapon;
    const enum weaponType heldType = weapon->weaponInfo->type;
    if (type == heldType) {
        return value * (1.0f - ammoFraction(weapon));
    }
    if (heldType == STANDARD_WEAPON) {
        return value;
    }
    if (!c->preset->replacesWeapon) {
        return 0.0f;
    }
    return value - slotWorth(c, weapon) - PICKUP_REPLACE_MARGIN;
}

// what picking up a utility adds: a utility where there was none, ammo for
// the held one, or a better utility in place of the held one
static float utilityPickupGain(const botCtx *c, const enum utilityType type) {
    if (!botHas(c, BOT_FEAT_UTILITY)) {
        return 0.0f;
    }
    const float value = c->preset->utilityValue[type];
    if (value == 0.0f) {
        return 0.0f;
    }
    const utilitySlot *utility = &c->drone->utility;
    const enum utilityType heldType = utility->utilityInfo->type;
    if (heldType == UNSET_UTILITY) {
        return value;
    }
    if (type == heldType) {
        return value * (1.0f - ((float)utility->ammo / utility->utilityInfo->ammo));
    }
    return value - c->preset->utilityValue[heldType] - PICKUP_REPLACE_MARGIN;
}

// scores the weapon and utility pickups and returns the best one worth
// leaving the current fight for, or NULL
weaponPickupEntity *choosePickup(botCtx *c) {
    iwEnv *e = c->e;
    const droneEntity *drone = c->drone;

    // value of staying in the current fight
    float combatValue = 0.0f;
    if (c->enemy != NULL && c->enemyDistSq <= weaponIdealRangeSquared(drone) && shouldShootAtEnemy(e, drone, c->enemy, c->enemyDir)) {
        combatValue = 1.0f;
        if (killGeometryGood(c, BURST_KILL_WALL_DISTANCE)) {
            combatValue = 3.0f;
        }
    }

    weaponPickupEntity *best = NULL;
    float bestScore = combatValue;
    int8_t bestIdx = -1;
    for (uint8_t i = 0; i < cc_array_size(e->pickups); i++) {
        weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
        // bots don't seek ammo crates yet
        if (pickup->kind == CRATE_PICKUP || pickup->floatingWallsTouching > 0 || pickup->bodyDestroyed || !safeFromSuddenDeath(c, pickup->pos)) {
            continue;
        }
        float value;
        if (pickup->kind == UTILITY_PICKUP) {
            value = utilityPickupGain(c, pickup->utility);
        } else {
            value = pickupGain(c, pickup->weapon);
        }
        if (value <= 0.0f) {
            continue;
        }
        const float distance = b2Distance(pickup->pos, drone->pos);
        // the evader only takes pickups that keep it at least as far from
        // the enemy as it is now
        if (c->personality == BOT_EVADER && c->enemy != NULL) {
            const float enemyDistance = b2Distance(pickup->pos, c->enemy->pos);
            if (enemyDistance < distance || enemyDistance < c->enemyDist) {
                continue;
            }
        }
        float score = value - (distance / PICKUP_TRAVEL_SPEED);

        // pickups next to death walls are risky to grab
        const nearWall *walls = e->map->nearestWalls + (MAX_NEAREST_WALLS * pickup->mapCellIdx);
        for (uint8_t j = 0; j < MAX_NEAREST_WALLS; j++) {
            if (walls[j].type == DEATH_WALL_ENTITY && staticWallDistance(walls[j].pos, pickup->pos) < PICKUP_WALL_RISK_DISTANCE) {
                score -= 1.5f;
                break;
            }
        }
        // the enemy will likely get there first, or be waiting
        if (c->enemy != NULL) {
            const float enemyDistance = b2Distance(pickup->pos, c->enemy->pos);
            if (enemyDistance < distance) {
                score -= 2.0f * (1.0f - (enemyDistance / fmaxf(distance, 1.0f)));
            }
        }
        // stick with the pickup we already committed to
        if (i == c->state->pickupIdx && botStep(e) < c->state->pickupUntilStep) {
            score += 1.0f;
        }
        if (score > bestScore) {
            bestScore = score;
            best = pickup;
            bestIdx = i;
        }
    }

    if (bestIdx != c->state->pickupIdx) {
        c->state->pickupIdx = bestIdx;
        c->state->pickupUntilStep = botStep(e) + (uint16_t)(PICKUP_COMMIT_TIME * e->frameRate);
    }
    return best;
}

// decide whether to charge or release a burst at the enemy
void deliberateBurst(botCtx *c, agentActions *actions) {
    const droneEntity *drone = c->drone;
    if (!canBurst(drone) && !drone->chargingBurst) {
        return;
    }
    const float reach = c->enemyDist - DRONE_RADIUS;
    const bool geometry = killGeometryGood(c, BURST_KILL_WALL_DISTANCE);
    if (drone->chargingBurst) {
        // keep charging while the burst can't reach yet but soon will
        if (geometry && reach > burstRadius(drone->burstCharge) && reach < burstRadius(0.8f) && drone->burstCharge < 0.8f && drone->energyLeft > 0.05f) {
            actions->chargingBurst = true;
        }
        return;
    }
    if (geometry && reach <= burstRadius(0.6f) && drone->energyLeft >= 0.15f) {
        actions->chargingBurst = true;
        botEvent(c, BOT_EVENT_BURST_START);
    } else if (reach <= DRONE_BURST_RADIUS_MIN - 1.0f && drone->energyLeft >= BURST_ENERGY_RESERVE) {
        // separate from a drone that's right on top of us
        actions->chargingBurst = true;
        botEvent(c, BOT_EVENT_BURST_START);
    }
}

// charge a burst before firing and release it with the shot so the burst
// speeds up the just fired projectile
void burstBoost(botCtx *c, agentActions *actions) {
    const droneEntity *drone = c->drone;
    const enum weaponType type = drone->weapon.weaponInfo->type;
    bool worthIt = false;
    if (type == FLAK_CANNON_WEAPON) {
        worthIt = drone->energyLeft >= BOOST_ENERGY_RESERVE;
    } else if (type == STANDARD_WEAPON) {
        worthIt = drone->energyLeft >= BOOST_ENERGY_RESERVE && killGeometryGood(c, BURST_KILL_WALL_DISTANCE);
    }
    if (!worthIt) {
        return;
    }
    if (!drone->chargingBurst) {
        if (canBurst(drone)) {
            actions->chargingBurst = true;
        }
        return;
    }
    // release with the shot, keep charging until the weapon is ready
    if (drone->weapon.cooldown > 0.0f) {
        actions->chargingBurst = true;
    } else {
        botEvent(c, BOT_EVENT_BOOST_RELEASE);
    }
}

// charge a burst while holding a close shotgun shot, then fire and release
// the burst together so it speeds up the pellets
void shotgunBoost(botCtx *c, agentActions *actions) {
    const droneEntity *drone = c->drone;
    if (c->enemyDist > SHOTGUN_BOOST_RANGE) {
        return;
    }
    if (!drone->chargingBurst) {
        if (canBurst(drone) && drone->energyLeft >= BOOST_ENERGY_RESERVE) {
            actions->chargingBurst = true;
            actions->trigger = false;
        }
        return;
    }
    if (drone->weapon.cooldown > 0.0f || (drone->burstCharge < SHOTGUN_BOOST_CHARGE && drone->energyLeft > 0.05f)) {
        actions->chargingBurst = true;
        actions->trigger = false;
        return;
    }
    botEvent(c, BOT_EVENT_SHOTGUN_BOOST);
}

// fire against the direction of movement to speed up while below the top
// speed; when the enemy is roughly in the firing direction the shot is
// aimed at it instead so it attacks too
bool recoilBoost(botCtx *c, agentActions *actions, const b2Vec2 moveDir) {
    const droneEntity *drone = c->drone;
    if (b2Equal(moveDir, b2Vec2_zero) || !b2Equal(actions->aim, b2Vec2_zero) || !weaponSafeForMovement(drone)) {
        return false;
    }
    if (drone->weapon.cooldown > 0.0f || drone->weapon.charge < drone->weapon.weaponInfo->maxCharge - c->e->deltaTime) {
        return false;
    }
    if (b2Dot(drone->velocity, moveDir) >= BOOST_TOP_SPEED) {
        return false;
    }
    // don't spend shots charging into an enemy that's already in range
    if (c->enemy != NULL && c->enemyDistSq <= c->idealRangeSquared && b2Dot(moveDir, c->enemyDir) > 0.0f) {
        return false;
    }

    b2Vec2 aim = b2MulSV(-1.0f, moveDir);
    b2Vec2 enemyAim;
    if (c->enemy != NULL && b2Dot(c->enemyDir, aim) >= BOOST_AIM_COS && shotReaches(drone, c->enemyDist) && shouldShootAtEnemyIntercept(c, &enemyAim) && b2Dot(enemyAim, aim) >= BOOST_AIM_COS) {
        aim = enemyAim;
        c->state->lastShotStep = botStep(c->e);
    }
    if (!safeToFireMoving(c->e, drone, aim, moveDir)) {
        return false;
    }
    botEvent(c, BOT_EVENT_RECOIL_BOOST);
    actions->aim = aim;
    scriptedAgentShoot(drone, actions);
    return true;
}

static inline bool cellWalkable(const iwEnv *e, const uint16_t cellIdx) {
    const entity *ent = e->cells[cellIdx].ent;
    return ent == NULL || !entityTypeIsWall(ent->type);
}

// distances to the closest death wall and other static wall near a cell
static void cellWallDistances(const iwEnv *e, const uint16_t cellIdx, float *deathWall, float *coverWall) {
    *deathWall = FLT_MAX;
    *coverWall = FLT_MAX;
    const b2Vec2 pos = e->cells[cellIdx].pos;
    const nearWall *walls = e->map->nearestWalls + (MAX_NEAREST_WALLS * cellIdx);
    for (uint8_t i = 0; i < MAX_NEAREST_WALLS; i++) {
        const float dist = staticWallDistance(walls[i].pos, pos);
        if (walls[i].type == DEATH_WALL_ENTITY) {
            *deathWall = fminf(*deathWall, dist);
        } else {
            *coverWall = fminf(*coverWall, dist);
        }
    }
}

// scores a cell for a personality; -FLT_MAX means the cell can't be used
typedef float (*spotScoreFn)(botCtx *c, const uint16_t cellIdx);

#define MAX_SPOT_CANDIDATES 32

// picks the reachable cell with the best score and keeps it for holdTime
// seconds; returns -1 if no cell can be used. If refineFn is set, the
// best MAX_SPOT_CANDIDATES cells get its score added before picking, for
// checks too expensive to run on every cell
int16_t chooseSpot(botCtx *c, spotScoreFn scoreFn, spotScoreFn refineFn, const float holdTime, const float stickiness) {
    iwEnv *e = c->e;
    scriptedBotState *state = c->state;
    const uint16_t step = botStep(e);
    if (state->spotCellIdx != -1 && step < state->spotUntilStep && safeFromSuddenDeath(c, e->cells[state->spotCellIdx].pos)) {
        return state->spotCellIdx;
    }
    const int16_t srcIdx = c->drone->mapCellIdx;
    if (srcIdx == -1) {
        return state->spotCellIdx;
    }

    // the best candidates sorted by score, best first
    const uint8_t maxCandidates = refineFn != NULL ? MAX_SPOT_CANDIDATES : 1;
    float candScores[MAX_SPOT_CANDIDATES];
    uint16_t candCells[MAX_SPOT_CANDIDATES];
    uint8_t numCands = 0;
    const uint16_t numCells = e->map->columns * e->map->rows;
    for (uint16_t i = 0; i < numCells; i++) {
        if (!cellWalkable(e, i)) {
            continue;
        }
        if (i != srcIdx && e->map->paths[pathOffset(e, srcIdx, i)] >= 8) {
            continue;
        }
        if (!safeFromSuddenDeath(c, e->cells[i].pos)) {
            continue;
        }
        float score = scoreFn(c, i);
        if (score != -FLT_MAX && i == state->spotCellIdx) {
            score += stickiness;
        }
        if (score == -FLT_MAX || (numCands == maxCandidates && score <= candScores[numCands - 1])) {
            continue;
        }
        uint8_t j = numCands < maxCandidates ? numCands++ : numCands - 1;
        while (j > 0 && candScores[j - 1] < score) {
            candScores[j] = candScores[j - 1];
            candCells[j] = candCells[j - 1];
            j--;
        }
        candScores[j] = score;
        candCells[j] = i;
    }
    float bestScore = -FLT_MAX;
    int16_t best = -1;
    for (uint8_t i = 0; i < numCands; i++) {
        float score = candScores[i];
        if (refineFn != NULL) {
            score += refineFn(c, candCells[i]);
        }
        if (score > bestScore) {
            bestScore = score;
            best = candCells[i];
        }
    }
    if (best != state->spotCellIdx) {
        botEvent(c, BOT_EVENT_SPOT_MOVE);
    }
    state->spotCellIdx = best;
    state->spotUntilStep = (uint16_t)fminf(step + (holdTime * e->frameRate), (float)UINT16_MAX);
    return best;
}

static inline void forgetSpot(botCtx *c) {
    c->state->spotCellIdx = -1;
}

// head for the spot; once there the caller stabilizes against recoil
// instead of braking, which would drain energy
void goToSpot(botCtx *c, agentActions *actions, const int16_t spot) {
    const b2Vec2 spotPos = c->e->cells[spot].pos;
    if (b2Distance(spotPos, c->drone->pos) > SPOT_REACHED_DISTANCE) {
        moveTo(c, actions, spotPos);
        return;
    }
    botEvent(c, BOT_EVENT_HOLD_SPOT);
}

// shoot at the enemy with intercept aim if the shot is clear and reaches
bool shootAtEnemy(botCtx *c, agentActions *actions) {
    b2Vec2 aim;
    if (!shotReaches(c->drone, c->enemyDist) || !shouldShootAtEnemyIntercept(c, &aim)) {
        return false;
    }
    actions->aim = aim;
    scriptedAgentShoot(c->drone, actions);
    c->state->lastShotStep = botStep(c->e);
    return true;
}

// open cells around a cell, out of (2 * EVADER_OPENNESS_RADIUS + 1)^2 - 1
static uint8_t cellOpenness(const iwEnv *e, const uint16_t cellIdx) {
    const int16_t cols = e->map->columns;
    const int16_t rows = e->map->rows;
    const int16_t col = cellIdx % cols;
    const int16_t row = cellIdx / cols;
    uint8_t open = 0;
    for (int16_t dr = -EVADER_OPENNESS_RADIUS; dr <= EVADER_OPENNESS_RADIUS; dr++) {
        for (int16_t dc = -EVADER_OPENNESS_RADIUS; dc <= EVADER_OPENNESS_RADIUS; dc++) {
            const int16_t nc = col + dc, nr = row + dr;
            if ((dc != 0 || dr != 0) && nc >= 0 && nc < cols && nr >= 0 && nr < rows && cellWalkable(e, nc + (nr * cols))) {
                open++;
            }
        }
    }
    return open;
}

// as far from where the enemy is headed as possible, out of corners, away
// from death walls and only where the drone gets before the enemy can
float evaderSpotScore(botCtx *c, const uint16_t cellIdx) {
    iwEnv *e = c->e;
    const b2Vec2 pos = e->cells[cellIdx].pos;
    const float enemyDistance = b2Distance(pos, c->enemyPredPos);
    const float distance = b2Distance(pos, c->drone->pos);
    float score = enemyDistance - (EVADER_TRAVEL_COST * distance) + (EVADER_OPENNESS_BONUS * cellOpenness(e, cellIdx));
    if (enemyDistance <= distance) {
        score -= EVADER_CONTESTED_PENALTY;
    }
    float deathWall, coverWall;
    cellWallDistances(e, cellIdx, &deathWall, &coverWall);
    if (deathWall < SPOT_WALL_DISTANCE) {
        score -= EVADER_WALL_PENALTY;
    }
    return score;
}

// out of the enemy's line of sight
float evaderCoverScore(botCtx *c, const uint16_t cellIdx) {
    if (segmentClear(c->e, c->e->cells[cellIdx].pos, c->enemyPredPos, DRONE_RADIUS)) {
        return 0.0f;
    }
    return EVADER_COVER_BONUS;
}

// runs away, only firing to move faster or when the enemy is close
void evaderActions(botCtx *c, agentActions *actions) {
    if (c->enemyDist <= EVADER_DANGER_DISTANCE && shootAtEnemy(c, actions)) {
        return;
    }
    // when the enemy is closing in, don't wait where it's headed for it to
    // arrive; any other good spot beats staying
    float stickiness = EVADER_STICKINESS;
    const float closingSpeed = -b2Dot(b2Sub(c->enemy->velocity, c->drone->velocity), c->enemyDir);
    if (c->enemyDist < EVADER_THREAT_DISTANCE && closingSpeed > EVADER_THREAT_CLOSING_SPEED) {
        const b2Vec2 spotPos = c->state->spotCellIdx != -1 ? c->e->cells[c->state->spotCellIdx].pos : c->drone->pos;
        if (b2Distance(spotPos, c->drone->pos) <= SPOT_REACHED_DISTANCE) {
            stickiness = -EVADER_THREAT_SPOT_PENALTY;
        }
    }
    const int16_t spot = chooseSpot(c, evaderSpotScore, evaderCoverScore, 0.0f, stickiness);
    if (spot == -1) {
        return;
    }
    botEvent(c, BOT_EVENT_FLEE);
    goToSpot(c, actions, spot);
}

// within sniping range of the enemy with a clear line to it, next to cover
// and close to the drone
float camperSpotScore(botCtx *c, const uint16_t cellIdx) {
    iwEnv *e = c->e;
    const b2Vec2 pos = e->cells[cellIdx].pos;
    const float distance = b2Distance(pos, c->drone->pos);
    if (distance > CAMPER_SPOT_SEARCH_DISTANCE) {
        return -FLT_MAX;
    }
    const float enemyDistance = b2Distance(pos, c->enemy->pos);
    if (enemyDistance < CAMPER_SPOT_MIN_RANGE || enemyDistance > CAMPER_SPOT_MAX_RANGE) {
        return -FLT_MAX;
    }
    float deathWall, coverWall;
    cellWallDistances(e, cellIdx, &deathWall, &coverWall);
    if (deathWall < SPOT_WALL_DISTANCE) {
        return -FLT_MAX;
    }
    float score = -0.3f * distance;
    if (coverWall < SPOT_WALL_DISTANCE) {
        score += 5.0f;
    }
    // the raycast is the expensive part, so it's done last
    if (!segmentClear(e, pos, c->enemy->pos, 0.0f)) {
        return -FLT_MAX;
    }
    return score;
}

// the camper fights from cover with anything that isn't short range or
// area denial
static inline bool camperWeapon(const enum weaponType type) {
    return type == STANDARD_WEAPON || type == MACHINEGUN_WEAPON || type == SNIPER_WEAPON || type == IMPLODER_WEAPON || type == NUKE_WEAPON;
}

// with a charged weapon, waits at a spot with a line to the enemy and only
// takes fully charged shots; returns false to fall back to the default
// behavior
bool camperActions(botCtx *c, agentActions *actions) {
    if (!camperWeapon(c->drone->weapon.weaponInfo->type)) {
        return false;
    }
    // the spot is re-chosen every CAMPER_SHOT_WAIT seconds, or right away
    // if the enemy gets close
    if (c->enemyDist < CAMPER_MIN_RANGE) {
        forgetSpot(c);
    }
    const int16_t spot = chooseSpot(c, camperSpotScore, NULL, CAMPER_SHOT_WAIT, 0.0f);
    if (spot == -1) {
        return false;
    }
    const bool charged = c->drone->weapon.charge >= c->drone->weapon.weaponInfo->maxCharge - c->e->deltaTime;
    if (charged && c->enemyDist >= CAMPER_MIN_RANGE && shootAtEnemy(c, actions)) {
        return true;
    }
    goToSpot(c, actions, spot);
    return true;
}

// finds the aim of a mine shot at the wall nearest a point on the path
// between the drone and the enemy, far enough that the mine won't go off
// on the drone
bool mineAim(botCtx *c, b2Vec2 *aim) {
    iwEnv *e = c->e;
    const droneEntity *drone = c->drone;
    float pathFraction = 0.5f;
    if (c->preset->grappleCloses) {
        pathFraction = MINE_CLOSING_PATH_FRACTION;
    }
    // the mine has to be outside of its blast from the drone when the
    // enemy sets it off
    const float pathDistance = fminf(fmaxf(c->enemyDist * pathFraction, MINE_SELF_SAFE_DISTANCE + 3.0f), c->enemyDist);
    const b2Vec2 mid = b2MulAdd(drone->pos, pathDistance, c->enemyDir);
    const int16_t midIdx = entityPosToCellIdx(e, mid);
    if (midIdx == -1) {
        return false;
    }
    const nearWall *walls = e->map->nearestWalls + (MAX_NEAREST_WALLS * midIdx);
    for (uint8_t i = 0; i < MAX_NEAREST_WALLS; i++) {
        const b2Vec2 wallPos = walls[i].pos;
        const float distance = b2Distance(wallPos, drone->pos);
        if (distance > TRAPPER_MINE_RANGE || distance < MINE_SELF_SAFE_DISTANCE) {
            continue;
        }
        // the mine must reach this wall and not stop at a closer one; it's
        // cast as its full circle, as clipping a corner next to the drone
        // would weld it there and set it off on the drone
        const b2Vec2 wallAim = b2Normalize(b2Sub(wallPos, drone->pos));
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE};
        const b2ShapeProxy mineProxy = b2MakeProxy(&drone->pos, 1, MINE_LAUNCHER_RADIUS);
        castCircleCtx cast = {0};
        b2World_CastShape(e->worldID, &mineProxy, b2MulSV(distance, wallAim), filter, castCircleCallback, &cast);
        if (!cast.hit || b2Distance(cast.point, wallPos) > WALL_THICKNESS) {
            continue;
        }
        if (!safeToFireUtility(e, drone, wallAim)) {
            continue;
        }
        *aim = wallAim;
        return true;
    }
    return false;
}

// keep the enemy at range so it has to come through mines and black holes
void keepRange(botCtx *c, agentActions *actions, const float range) {
    if (c->enemyDist < range) {
        actions->move = b2Normalize(b2MulAdd(actions->move, -1.0f, c->enemyDir));
    } else if (c->enemyDist > range + 10.0f) {
        moveTo(c, actions, c->enemy->pos);
    }
}

// fires area weapons from mid range; mines are placed by the utility
// actions. Returns false to fall back to the default behavior
bool trapperActions(botCtx *c, agentActions *actions) {
    switch (c->drone->weapon.weaponInfo->type) {
    case BLACK_HOLE_WEAPON:
    case IMPLODER_WEAPON:
        if (c->enemyDist >= TRAPPER_MIN_RANGE && c->enemyDist <= TRAPPER_MAX_RANGE && shootAtEnemy(c, actions)) {
            return true;
        }
        keepRange(c, actions, (TRAPPER_MIN_RANGE + TRAPPER_MAX_RANGE) / 2.0f);
        return true;
    case NUKE_WEAPON:
        if (c->enemyDist >= TRAPPER_NUKE_MIN_RANGE && shootAtEnemy(c, actions)) {
            return true;
        }
        keepRange(c, actions, TRAPPER_NUKE_MIN_RANGE + 5.0f);
        return true;
    default:
        return false;
    }
}

// close to where the turtle is and away from death walls, ideally by cover
float turtleSpotScore(botCtx *c, const uint16_t cellIdx) {
    const b2Vec2 pos = c->e->cells[cellIdx].pos;
    float deathWall, coverWall;
    cellWallDistances(c->e, cellIdx, &deathWall, &coverWall);
    if (deathWall < SPOT_WALL_DISTANCE + 1.0f) {
        return -FLT_MAX;
    }
    float score = -b2Distance(pos, c->drone->pos);
    if (coverWall < SPOT_WALL_DISTANCE) {
        score += 2.0f;
    }
    return score;
}

// holds a home spot and only fires back after the enemy fires or when
// it's close
void turtleActions(botCtx *c, agentActions *actions) {
    iwEnv *e = c->e;
    scriptedBotState *state = c->state;
    const uint16_t step = botStep(e);
    for (uint8_t i = 0; i < cc_array_size(e->projectiles); i++) {
        const projectileEntity *proj = safe_array_get_at(e->projectiles, i);
        if (proj->droneIdx == c->enemy->idx && proj->weaponInfo->type != MINE_LAUNCHER_WEAPON && !proj->welded) {
            state->lastEnemyShotStep = step;
            break;
        }
    }
    const bool provoked = state->lastEnemyShotStep != 0 && step - state->lastEnemyShotStep <= (uint16_t)(TURTLE_COUNTER_TIME * e->frameRate);
    if ((provoked || c->enemyDist <= TURTLE_DANGER_DISTANCE) && shootAtEnemy(c, actions)) {
        botEvent(c, BOT_EVENT_COUNTER_SHOT);
    }
    // the home spot is chosen once per life
    const int16_t spot = chooseSpot(c, turtleSpotScore, NULL, (float)UINT16_MAX, 0.0f);
    if (spot != -1) {
        goToSpot(c, actions, spot);
    }
}

// move off the line from the enemy through the drone to a death wall
// behind it, as that's where hits knock the drone
void avoidKillGeometry(botCtx *c, agentActions *actions) {
    if (c->enemy == NULL) {
        return;
    }
    b2Vec2 wallDir;
    float wallDistance;
    if (!droneDeathWall(c->e, c->drone, SAFE_GEOMETRY_WALL_DISTANCE, &wallDir, &wallDistance)) {
        return;
    }
    const b2Vec2 pushDir = b2MulSV(-1.0f, c->enemyDir);
    const float alignment = b2Dot(pushDir, wallDir);
    if (alignment < SAFE_GEOMETRY_COS) {
        return;
    }
    botEvent(c, BOT_EVENT_SAFE_GEOMETRY);
    // away from the wall and to whichever side of the push line the drone
    // already leans
    b2Vec2 side = b2LeftPerp(pushDir);
    if (b2Dot(side, wallDir) > 0.0f) {
        side = b2MulSV(-1.0f, side);
    }
    const float weight = alignment * (1.0f - (wallDistance / SAFE_GEOMETRY_WALL_DISTANCE)) * 2.0f;
    const b2Vec2 away = b2Normalize(b2Add(b2MulSV(-1.0f, wallDir), side));
    actions->move = b2Normalize(b2MulAdd(actions->move, weight, away));
}

enum botGrappleIntent {
    GRAPPLE_INTENT_NONE,
    GRAPPLE_INTENT_CLOSE,
    GRAPPLE_INTENT_ESCAPE,
};

// whether the utility can be used now rather than acting on what's out
static inline bool utilityReady(const botCtx *c) {
    const utilitySlot *utility = &c->drone->utility;
    const int8_t maxActive = utility->utilityInfo->maxActive;
    return utility->ammo > 0 && utility->cooldown <= 0.0f && (maxActive == INFINITE || droneUtilityActive(c->e, c->drone) < maxActive);
}

// presses the utility button; a held button doesn't register a press, so
// it's released this step and pressed the next instead
static bool pressUtility(const droneEntity *drone, agentActions *actions) {
    actions->utility = !drone->utilityHeld;
    return actions->utility;
}

// fires the utility along aim this step instead of the weapon; a charged
// weapon keeps its charge held rather than firing in the wrong direction
static bool fireUtility(const droneEntity *drone, agentActions *actions, const b2Vec2 aim) {
    if (!pressUtility(drone, actions)) {
        return false;
    }
    actions->aim = aim;
    actions->trigger = drone->weapon.weaponInfo->maxCharge != 0.0f;
    return true;
}

// what the grapple would attach to fired along aim: returns false if it
// wouldn't reach anything or would attach to a death wall, otherwise sets
// distance and whether it would attach to a drone
static bool grappleTarget(const iwEnv *e, const droneEntity *drone, const b2Vec2 aim, float *distance, bool *hitDrone) {
    const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE};
    const b2RayResult ray = b2World_CastRayClosest(e->worldID, drone->pos, b2MulSV(GRAPPLE_MAX_DISTANCE, aim), filter);
    if (!ray.hit) {
        return false;
    }
    const entity *ent = b2Shape_GetUserData(ray.shapeId);
    if (ent == NULL || ent->type == DEATH_WALL_ENTITY || (ent->type != DRONE_ENTITY && !entityTypeIsWall(ent->type))) {
        return false;
    }
    if (ent->type == DRONE_ENTITY && ((const droneEntity *)ent->entity)->idx == drone->idx) {
        return false;
    }
    *distance = b2Distance(drone->pos, ray.point);
    *hitDrone = ent->type == DRONE_ENTITY;
    return true;
}

// grapples the enemy, or a wall behind it, when it's well out of the
// weapon's range
static bool grappleClose(botCtx *c, agentActions *actions) {
    if (!c->preset->grappleCloses || c->enemy->shield != NULL) {
        return false;
    }
    const float idealRange = sqrtf(weaponIdealRangeSquared(c->drone));
    if (c->enemyDist < fmaxf(idealRange + GRAPPLE_CLOSE_MARGIN, GRAPPLE_CLOSE_MIN_DISTANCE)) {
        return false;
    }
    const droneEntity *drone = c->drone;
    // lead the enemy by the hook's travel time
    const float travelTime = c->enemyDist / grapple.initialSpeed;
    const b2Vec2 target = b2MulAdd(c->enemy->pos, travelTime, c->enemy->velocity);
    const b2Vec2 aim = b2Normalize(b2Sub(target, drone->pos));
    float distance;
    bool hitDrone;
    if (!grappleTarget(c->e, drone, aim, &distance, &hitDrone) || (!hitDrone && distance < c->enemyDist)) {
        return false;
    }
    if (!safeToFireUtility(c->e, drone, aim) || !fireUtility(drone, actions, aim)) {
        return false;
    }
    botEvent(c, BOT_EVENT_GRAPPLE_CLOSE);
    c->state->grappleIntent = GRAPPLE_INTENT_CLOSE;
    c->state->grappleUntilStep = botStep(c->e) + (uint16_t)(GRAPPLE_CLOSE_TIME * c->e->frameRate);
    return true;
}

// grapples a wall away from a close enemy to get away from it
static bool grappleEscape(botCtx *c, agentActions *actions) {
    const droneEntity *drone = c->drone;
    if (!c->preset->grappleEscapes || c->enemyDist > GRAPPLE_ESCAPE_ENEMY_DISTANCE) {
        return false;
    }
    if (c->personality != BOT_EVADER && drone->energyLeft >= GRAPPLE_ESCAPE_ENERGY) {
        return false;
    }
    // straight away from the enemy, then further and further to the sides
    static const float angles[] = {0.0f, 0.6f, -0.6f, 1.2f, -1.2f};
    const b2Vec2 away = b2MulSV(-1.0f, c->enemyDir);
    for (uint8_t i = 0; i < sizeof(angles) / sizeof(angles[0]); i++) {
        const b2Vec2 aim = b2RotateVector(b2MakeRot(angles[i]), away);
        float distance;
        bool hitDrone;
        if (!grappleTarget(c->e, drone, aim, &distance, &hitDrone) || hitDrone || distance < GRAPPLE_ESCAPE_MIN_ANCHOR) {
            continue;
        }
        if (!safeToFireUtility(c->e, drone, aim) || !fireUtility(drone, actions, aim)) {
            continue;
        }
        botEvent(c, BOT_EVENT_GRAPPLE_ESCAPE);
        c->state->grappleIntent = GRAPPLE_INTENT_ESCAPE;
        c->state->grappleUntilStep = botStep(c->e) + (uint16_t)(GRAPPLE_ESCAPE_TIME * c->e->frameRate);
        return true;
    }
    return false;
}

// whether the grapple that's out has done what it was fired for, or should
// be let go of
static bool grappleDone(botCtx *c) {
    const droneEntity *drone = c->drone;
    const scriptedBotState *state = c->state;
    if (botStep(c->e) >= state->grappleUntilStep || c->enemy == NULL) {
        return true;
    }
    // don't get reeled or pulled into a death wall
    b2Vec2 wallDir;
    float wallDistance;
    if (droneDeathWall(c->e, drone, GRAPPLE_DANGER_WALL_DISTANCE, &wallDir, &wallDistance) && b2Dot(drone->velocity, wallDir) * GRAPPLE_DANGER_WALL_TIME > wallDistance) {
        botEvent(c, BOT_EVENT_GRAPPLE_DANGER_RELEASE);
        return true;
    }
    if (state->grappleIntent == GRAPPLE_INTENT_CLOSE) {
        const float idealRange = sqrtf(weaponIdealRangeSquared(drone));
        return c->enemyDist <= fmaxf(idealRange, GRAPPLE_CLOSE_STOP_DISTANCE);
    }
    return b2Distance(drone->pos, drone->grapple->pos) <= GRAPPLE_ESCAPE_DONE_ANCHOR || c->enemyDist >= GRAPPLE_ESCAPE_DONE_ENEMY;
}

// holds the grapple out until it's done what it was fired for, then lets
// go; the hook reels as soon as it attaches while held
static void holdGrapple(botCtx *c, agentActions *actions) {
    const droneEntity *drone = c->drone;
    scriptedBotState *state = c->state;
    // the hook was destroyed before attaching, or was shot off
    if (drone->grapple == NULL) {
        state->grappleIntent = GRAPPLE_INTENT_NONE;
        return;
    }
    if (drone->grapple->welded && grappleDone(c)) {
        botEvent(c, BOT_EVENT_GRAPPLE_RELEASE);
        state->grappleIntent = GRAPPLE_INTENT_NONE;
        return;
    }
    actions->utility = true;
}

// sets off the drone's welded mine when the enemy is close to it and the
// drone isn't
static bool detonateMine(botCtx *c, agentActions *actions) {
    const projectileEntity *mine = droneMine(c->e, c->drone);
    if (mine == NULL || !mine->welded) {
        return false;
    }
    if (b2Distance(mine->pos, c->enemy->pos) > MINE_DETONATE_DISTANCE || b2Distance(mine->pos, c->drone->pos) < MINE_SELF_SAFE_DISTANCE) {
        return false;
    }
    if (!pressUtility(c->drone, actions)) {
        return false;
    }
    botEvent(c, BOT_EVENT_MINE_DETONATE);
    return true;
}

// places a mine between the drone and the enemy when the weapon isn't
// being fired this step; the trapper places one whenever it can
static bool placeMine(botCtx *c, agentActions *actions) {
    const iwEnv *e = c->e;
    const droneEntity *drone = c->drone;
    if (c->enemyDist < MINE_PLACE_MIN_ENEMY_DISTANCE || !utilityReady(c)) {
        return false;
    }
    if (c->personality != BOT_TRAPPER && !b2Equal(actions->aim, b2Vec2_zero) && actions->trigger) {
        return false;
    }
    for (size_t i = 0; i < cc_array_size(e->projectiles); i++) {
        const projectileEntity *proj = safe_array_get_at(e->projectiles, i);
        if (proj->droneIdx != drone->idx && !proj->welded && b2DistanceSquared(proj->pos, drone->pos) < SQUARED(MINE_PLACE_PROJECTILE_CLEARANCE)) {
            return false;
        }
    }
    b2Vec2 aim;
    if (!mineAim(c, &aim)) {
        return false;
    }
    if (b2Dot(aim, c->enemyDir) > MINE_PLACE_ENEMY_AIM_COS && segmentClear(e, drone->pos, c->enemy->pos, 0.0f)) {
        return false;
    }
    if (!fireUtility(drone, actions, aim)) {
        return false;
    }
    botEvent(c, BOT_EVENT_MINE_PLACED);
    return true;
}

// keeps the drone from setting off its own mine: doesn't burst with the mine
// in reach of the burst, holds fire while the mine is in flight close by as
// any shot that hits it sets it off, and steers away from it once it's
// welded, out of its blast if the enemy is about to set it off
static void avoidOwnMine(botCtx *c, agentActions *actions) {
    const droneEntity *drone = c->drone;
    const projectileEntity *mine = droneMine(c->e, drone);
    if (mine == NULL) {
        return;
    }
    const float distance = b2Distance(mine->pos, drone->pos);
    if (distance < MINE_BURST_DISTANCE) {
        // releasing a burst being charged would set it off, so keep charging
        actions->chargingBurst = drone->chargingBurst;
    }
    if (!mine->welded) {
        if (distance < MINE_SELF_SAFE_DISTANCE) {
            // a charged weapon is kept charged rather than fired
            actions->trigger = actions->trigger && drone->weapon.weaponInfo->maxCharge != 0.0f;
        }
        return;
    }
    float avoidDistance = OWN_MINE_AVOID_DISTANCE;
    if (c->enemy != NULL && b2Distance(mine->pos, c->enemy->pos) < MINE_LAUNCHER_PROXIMITY_RADIUS + MINE_ENEMY_NEAR_MARGIN) {
        avoidDistance = MINE_SELF_SAFE_DISTANCE;
    }
    if (distance > avoidDistance) {
        return;
    }
    const b2Vec2 away = b2Normalize(b2Sub(drone->pos, mine->pos));
    // only the part of the move towards the mine is taken out, the rest of
    // where the drone was heading is kept
    const float towards = -b2Dot(actions->move, away);
    if (towards > 0.0f) {
        actions->move = b2MulAdd(actions->move, towards, away);
    }
    actions->move = b2Normalize(b2MulAdd(actions->move, 1.0f, away));
}

// everything the bot does with its utility, on top of what it decided to
// do with its weapon
static void utilityActions(botCtx *c, agentActions *actions) {
    const droneEntity *drone = c->drone;
    avoidOwnMine(c, actions);
    switch (drone->utility.utilityInfo->type) {
    case GRAPPLE_UTILITY:
        if (c->state->grappleIntent != GRAPPLE_INTENT_NONE) {
            holdGrapple(c, actions);
            return;
        }
        // detach an attached grapple the bot isn't using: releasing the
        // button only detaches it after holding it
        if (drone->grapple != NULL) {
            actions->utility = drone->grapple->welded && !drone->utilityHeld;
            return;
        }
        if (c->enemy == NULL || !utilityReady(c)) {
            return;
        }
        if (!grappleEscape(c, actions)) {
            grappleClose(c, actions);
        }
        return;
    case MINE_UTILITY:
        if (c->enemy == NULL) {
            return;
        }
        if (!detonateMine(c, actions)) {
            placeMine(c, actions);
        }
        return;
    default:
        return;
    }
}

static agentActions botActions(const botCtx *ctx);

agentActions scriptedAgentActions(iwEnv *e, droneEntity *drone) {
    agentActions actions = {0};
    // easy curriculum episodes always face a sitting duck
    if (e->botPolicy == BOT_SITTING_DUCK || e->easyEp) {
        return actions;
    }

    // pick a random action randomly if the bot noise is enabled
    if (e->botCLNoise > 0.0f && randFloat(&e->rng, 0.0f, 1.0f) < e->botCLNoise) {
        actions.move.x = randFloat(&e->rng, -1.0f, 1.0f);
        actions.move.y = randFloat(&e->rng, -1.0f, 1.0f);
        actions.aim.x = randFloat(&e->rng, -1.0f, 1.0f);
        actions.aim.y = randFloat(&e->rng, -1.0f, 1.0f);
        actions.brake = randInt(&e->rng, 0, 1) == 1;

        if (randInt(&e->rng, 0, 1) == 1) {
            scriptedAgentShoot(drone, &actions);
        }
        if (randInt(&e->rng, 0, 1) == 1) {
            scriptedAgentBurst(drone, &actions);
        }
        actions.utility = randInt(&e->rng, 0, 1) == 1;
        return actions;
    }

    enum botPolicy personality = e->botPersonality[drone->idx];
    if (personality < BOT_SCRIPTED || personality >= NUM_BOT_POLICIES) {
        personality = BOT_SCRIPTED;
    }
    botCtx c = {
        .e = e,
        .drone = drone,
        .feats = e->botFeatures[drone->idx],
        .state = &e->botState[drone->idx],
        .personality = personality,
        .preset = &botPresets[personality],
    };
    // spots are chosen per life
    if (c.state->livesLeft != drone->livesLeft) {
        c.state->livesLeft = drone->livesLeft;
        c.state->grappleIntent = GRAPPLE_INTENT_NONE;
        forgetSpot(&c);
    }

    // find closest enemy drone
    droneEntity *enemyDrone = NULL;
    float closestDistanceSquared = FLT_MAX;
    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        if (i == drone->idx) {
            continue;
        }
        droneEntity *otherDrone = safe_array_get_at(e->drones, i);
        if (otherDrone->dead || otherDrone->team == drone->team) {
            continue;
        }
        const float distanceSquared = b2DistanceSquared(otherDrone->pos, drone->pos);
        if (distanceSquared < closestDistanceSquared) {
            closestDistanceSquared = distanceSquared;
            enemyDrone = otherDrone;
        }
    }
    if (enemyDrone != NULL) {
        c.enemy = enemyDrone;
        c.enemyDistSq = closestDistanceSquared;
        c.enemyDist = sqrtf(closestDistanceSquared);
        c.enemyDir = b2Normalize(b2Sub(enemyDrone->pos, drone->pos));
        c.enemyPredPos = b2MulAdd(enemyDrone->pos, EVADER_PREDICT_TIME, enemyDrone->velocity);
    }
    actions = botActions(&c);
    if (botHas(&c, BOT_FEAT_UTILITY)) {
        utilityActions(&c, &actions);
    }
    return actions;
}

// everything the bot does with the weapon it has drawn
static agentActions botActions(const botCtx *ctx) {
    botCtx c = *ctx;
    iwEnv *e = c.e;
    droneEntity *drone = c.drone;
    droneEntity *enemyDrone = c.enemy;
    const float closestDistanceSquared = c.enemyDistSq;
    const enum botPolicy personality = c.personality;
    agentActions actions = {0};

    c.suddenDeath = suddenDeathSafeArea(e, SUDDEN_DEATH_MARGIN, &c.suddenDeathSafe);
    c.idealRangeSquared = weaponIdealRangeSquared(drone);
    if (drone->weapon.weaponInfo->type == FLAK_CANNON_WEAPON) {
        c.idealRangeSquared = SQUARED(FLAK_RANGE);
    }
    // flak shells need distance to arm and imploders pull the drone in too
    const enum weaponType weaponType = drone->weapon.weaponInfo->type;
    if (c.preset->rangeScale != 1.0f && c.idealRangeSquared != FLT_MAX && weaponType != FLAK_CANNON_WEAPON && weaponType != IMPLODER_WEAPON) {
        c.idealRangeSquared *= SQUARED(c.preset->rangeScale);
    }

    // stay out of reach of a shotgun
    bool kiting = false;
    if (botHas(&c, BOT_FEAT_KITE) && enemyDrone != NULL && weaponType != SHOTGUN_WEAPON && enemyDrone->weapon.weaponInfo->type == SHOTGUN_WEAPON && enemyDrone->weapon.ammo != 0) {
        c.idealRangeSquared = fmaxf(c.idealRangeSquared, SQUARED(KITE_RANGE));
        kiting = c.enemyDist < KITE_RANGE;
    }

    // keep the weapon charged and ready
    if (drone->weapon.weaponInfo->maxCharge != 0.0f) {
        actions.trigger = true;
    }

    // find the N nearest death walls or floating walls
    nearWall nearWalls[MAX_NEAREST_WALLS] = {0};
    findNearWalls(e, drone, nearWalls, NUM_NEAR_WALLS);

    // move away from and shoot at death walls if we're too close
    float closestWallDistance = FLT_MAX;
    for (uint8_t i = 0; i < NUM_NEAR_WALLS; i++) {
        if (nearWalls[i].type != DEATH_WALL_ENTITY) {
            continue;
        }

        const wallEntity *wall = safe_array_get_at(e->walls, nearWalls[i].wallIdx);
        const b2DistanceOutput output = closestPoint(drone->ent, wall->ent);
        closestWallDistance = fminf(closestWallDistance, output.distance);
        handleWallProximityVelocity(&c, &output, &actions);
    }

    for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        wallEntity *floatingWall = safe_array_get_at(e->floatingWalls, i);
        if (floatingWall->type != DEATH_WALL_ENTITY) {
            continue;
        }
        if (b2DistanceSquared(floatingWall->pos, drone->pos) > FLOATING_WALL_CHECK_DISTANCE_SQUARED) {
            continue;
        }

        const b2DistanceOutput output = closestPoint(drone->ent, floatingWall->ent);
        closestWallDistance = fminf(closestWallDistance, output.distance);
        handleWallProximityVelocity(&c, &output, &actions);
    }

    // get away from where the next sudden death walls will be placed
    if (c.suddenDeath) {
        const b2Vec2 safePos = clampToBounds(&c.suddenDeathSafe, drone->pos);
        const b2Vec2 toSafe = b2Sub(safePos, drone->pos);
        const float outside = b2Length(toSafe);
        if (outside > 0.0f) {
            botEvent(&c, BOT_EVENT_SUDDEN_DEATH_AVOID);
            const b2Vec2 safeDir = b2MulSV(1.0f / outside, toSafe);
            actions.move = b2MulAdd(actions.move, 2.0f + outside, safeDir);
            // the walls are about to land, nothing else matters; fire
            // outward so the recoil helps
            if ((float)e->suddenDeathSteps / e->frameRate < SUDDEN_DEATH_URGENT_TIME) {
                actions.move = b2Normalize(actions.move);
                const b2Vec2 aim = b2MulSV(-1.0f, safeDir);
                if (weaponSafeForMovement(drone) && safeToFireMoving(e, drone, aim, actions.move)) {
                    actions.aim = aim;
                    scriptedAgentShoot(drone, &actions);
                }
                return actions;
            }
        } else {
            // inside the safe area, push off its edges when close
            const float edges[4] = {
                drone->pos.x - c.suddenDeathSafe.min.x,
                c.suddenDeathSafe.max.x - drone->pos.x,
                drone->pos.y - c.suddenDeathSafe.min.y,
                c.suddenDeathSafe.max.y - drone->pos.y,
            };
            const b2Vec2 inward[4] = {{1.0f, 0.0f}, {-1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, -1.0f}};
            for (uint8_t i = 0; i < 4; i++) {
                if (edges[i] < SUDDEN_DEATH_AVOID_DISTANCE) {
                    actions.move = b2MulAdd(actions.move, 1.0f / fmaxf(edges[i], 0.25f), inward[i]);
                }
            }
        }
    }

    // keep clear of enemy mines stuck to walls
    findHazards(&c);
    for (uint8_t i = 0; i < c.numHazards; i++) {
        const hazard *h = &c.hazards[i];
        if (h->isStaticWall || h->floatingWall != NULL) {
            continue;
        }
        const float distance = b2Distance(h->pos, drone->pos) - h->radius;
        if (distance <= WALL_AVOID_DISTANCE) {
            const b2Vec2 away = b2Normalize(b2Sub(drone->pos, h->pos));
            actions.move = b2MulAdd(actions.move, 1.0f / fmaxf(distance, 0.25f), away);
        }
    }
    steerForClearance(&c, &actions);

    // if we are moving towards a death wall at a high speed, do everything
    // possible to avoid hitting it
    const float droneSpeed = b2Length(drone->velocity);
    if (drone->braking || drone->chargingBurst || closestWallDistance <= WALL_BURST_CHECK_DISTANCE || droneSpeed >= WALL_BURST_CHECK_SPEED) {
        float damping = DRONE_LINEAR_DAMPING;
        if (drone->braking || actions.brake) {
            damping *= DRONE_BRAKE_DAMPING_COEF;
        }
        const b2Vec2 recoilPos = positionWithDamping(e, drone, b2Vec2_zero, damping, 0.5f / e->deltaTime);
        const b2Vec2 pos = drone->pos;
        const b2Vec2 rayEnd = recoilPos;
        const b2Vec2 translation = b2Sub(rayEnd, pos);
        float radius = DRONE_RADIUS;
        if (drone->shield != NULL) {
            radius = DRONE_SHIELD_RADIUS;
        }
        const b2ShapeProxy cirProxy = b2MakeProxy(&pos, 1, radius);
        const b2QueryFilter filter = {.categoryBits = DRONE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};

        castCircleCtx ctx = {0};
        b2World_CastShape(e->worldID, &cirProxy, translation, filter, castCircleCallback, &ctx);
        if (ctx.hit) {
            const entity *ent = b2Shape_GetUserData(ctx.shapeID);
            if (entityTypeIsWall(ent->type) && ent->type == DEATH_WALL_ENTITY) {
                actions.brake = true;
                if (drone->shield == NULL) {
                    wallBurst(e, drone, b2Distance(drone->pos, ctx.point), &actions);
                }

                const b2Vec2 droneDirection = b2Normalize(drone->velocity);
                if (b2Equal(actions.move, b2Vec2_zero)) {
                    actions.move = b2MulSV(-1.0f, droneDirection);
                }
                if (b2Equal(actions.aim, b2Vec2_zero) && weaponSafeForMovement(drone)) {
                    actions.aim = droneDirection;
                    scriptedAgentShoot(drone, &actions);
                }
                return actions;
            }
        }
    }

    // if we're close enough to a wall to need to shoot at it, don't
    // worry about enemies
    if (!b2Equal(actions.aim, b2Vec2_zero)) {
        return actions;
    }

    // dangerous incoming projectiles take priority over everything else
    if (botHas(&c, BOT_FEAT_PROJ_DEFENSE)) {
        if (defendAgainstProjectiles(&c, &actions)) {
            return actions;
        }
    }
    if (botHas(&c, BOT_FEAT_SNIPER_DODGE) && dodgeSniper(&c, &actions)) {
        return actions;
    }

    // get a weapon, top up ammo or replace the held weapon with a better one
    if (cc_array_size(e->pickups) != 0) {
        const weaponPickupEntity *pickup = choosePickup(&c);
        if (pickup != NULL) {
            if (pickup->kind == UTILITY_PICKUP) {
                botEvent(&c, BOT_EVENT_UTILITY_PICKUP);
            }
            moveTo(&c, &actions, pickup->pos);
            return actions;
        }
        botEvent(&c, BOT_EVENT_PICKUP_SKIPPED);
    }

    if (enemyDrone == NULL) {
        // fight recoil if we're not otherwise moving
        if (b2Equal(actions.move, b2Vec2_zero) && droneSpeed >= STABILIZE_MOVE_SPEED) {
            actions.move = b2MulSV(-1.0f, b2Normalize(drone->velocity));
        }
        return actions;
    }

    // burst if the enemy drone is very close
    if (botHas(&c, BOT_FEAT_BURST)) {
        deliberateBurst(&c, &actions);
    } else if (closestDistanceSquared <= BURST_MIN_RADIUS_SQUARED) {
        scriptedAgentBurst(drone, &actions);
    }

    bool personalityActed = false;
    switch (personality) {
    case BOT_EVADER:
        evaderActions(&c, &actions);
        personalityActed = true;
        break;
    case BOT_CAMPER:
        personalityActed = camperActions(&c, &actions);
        break;
    case BOT_TRAPPER:
        personalityActed = trapperActions(&c, &actions);
        break;
    case BOT_TURTLE:
        turtleActions(&c, &actions);
        personalityActed = true;
        break;
    default:
        break;
    }
    if (personalityActed) {
        if (botHas(&c, BOT_FEAT_SAFE_GEOMETRY)) {
            avoidKillGeometry(&c, &actions);
        }
        if (!b2Equal(actions.move, b2Vec2_zero)) {
            steerForClearance(&c, &actions);
        } else if (!actions.brake && droneSpeed >= STABILIZE_MOVE_SPEED) {
            actions.move = b2MulSV(-1.0f, b2Normalize(drone->velocity));
        }
        return actions;
    }

    // move into ideal range for the current weapon, but take any clear
    // shot that will reach on the way
    const float idealRangeSquared = c.idealRangeSquared;
    const b2Vec2 enemyDroneDirection = c.enemyDir;
    b2Vec2 aim = b2Vec2_zero;
    bool shoot = false;
    const bool shotgunTooFar = botHas(&c, BOT_FEAT_SHOTGUN_RANGE) && weaponType == SHOTGUN_WEAPON && c.enemyDist > SHOTGUN_FIRE_RANGE;
    if (closestDistanceSquared > idealRangeSquared) {
        if (personality != BOT_RUSHER && !shotgunTooFar && shouldShootAtEnemyIntercept(&c, &aim) && shotReaches(drone, c.enemyDist)) {
            botEvent(&c, BOT_EVENT_LONG_SHOT);
            shoot = true;
        } else {
            moveTo(&c, &actions, enemyDrone->pos);
            return actions;
        }
    } else {
        shoot = shouldShootAtEnemyIntercept(&c, &aim);
    }

    if (shoot) {
        if (drone->weapon.cooldown <= 0.0f && drone->weapon.charge >= drone->weapon.weaponInfo->maxCharge - e->deltaTime) {
            actions.move.x += enemyDroneDirection.x;
            actions.move.y += enemyDroneDirection.y;
            actions.move = b2Normalize(actions.move);
        }
        actions.aim = aim;
        scriptedAgentShoot(drone, &actions);
        if (botHas(&c, BOT_FEAT_SHOTGUN_BOOST) && weaponType == SHOTGUN_WEAPON) {
            shotgunBoost(&c, &actions);
        } else if (botHas(&c, BOT_FEAT_BURST_BOOST) && !actions.chargingBurst) {
            burstBoost(&c, &actions);
        }
        // keep closing in to counter the recoil
        if (personality == BOT_RUSHER) {
            moveTo(&c, &actions, enemyDrone->pos);
        }
    } else {
        moveTo(&c, &actions, enemyDrone->pos);
    }

    if (shoot && botHas(&c, BOT_FEAT_KILL_GEOMETRY)) {
        // get on the far side of the enemy from its closest death wall so
        // hits push it into the wall
        b2Vec2 wallDir;
        float wallDistance;
        if (enemyDeathWall(&c, KILL_WALL_DISTANCE, &wallDir, &wallDistance)) {
            const float range = fminf(fmaxf(c.enemyDist, 8.0f), fminf(sqrtf(idealRangeSquared), 25.0f));
            const b2Vec2 desired = b2MulAdd(enemyDrone->pos, -range, wallDir);
            const b2Vec2 toDesired = b2Sub(desired, drone->pos);
            if (b2Length(toDesired) > 2.0f) {
                botEvent(&c, BOT_EVENT_KILL_GEOMETRY);
                actions.move = b2Normalize(b2Add(actions.move, b2MulSV(2.0f, b2Normalize(toDesired))));
            }
        }
    }

    if (kiting) {
        botEvent(&c, BOT_EVENT_KITE);
        actions.move = b2Normalize(b2MulAdd(actions.move, 2.0f, b2MulSV(-1.0f, enemyDroneDirection)));
    }
    if (botHas(&c, BOT_FEAT_SAFE_GEOMETRY)) {
        avoidKillGeometry(&c, &actions);
    }
    if (!b2Equal(actions.move, b2Vec2_zero)) {
        steerForClearance(&c, &actions);
    }

    // fight recoil if we're not otherwise moving
    if (b2Equal(actions.move, b2Vec2_zero) && droneSpeed >= STABILIZE_MOVE_SPEED) {
        actions.move = b2MulSV(-1.0f, b2Normalize(drone->velocity));
    }

    return actions;
}

#endif
