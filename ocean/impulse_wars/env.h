#ifndef IMPULSE_WARS_ENV_H
#define IMPULSE_WARS_ENV_H

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

static double lastFrameTime = 0.0;
static double accumulator = 0.0;
#endif

#include "game.h"
#include "map.h"
#include "render.h"
#include "scripted_agent.h"

static const uint8_t THREE_BIT_MASK = 0x7;
static const uint8_t FOUR_BIT_MASK = 0xf;

#define agentObs(e, i) (e->agents[i].observations)
#define agentActions(e, i) (e->agents[i].actions)
#define agentRewards(e, i) (e->agents[i].rewards)
#define agentTerminals(e, i) (e->agents[i].terminals)

// returns a cell index that is closest to pos that isn't cellIdx
uint16_t findNearestCell(const iwEnv *e, const b2Vec2 pos, const uint16_t cellIdx) {
    uint16_t closestCell = cellIdx;
    float minDistance = FLT_MAX;
    const uint8_t cellCol = cellIdx / e->map->columns;
    const uint8_t cellRow = cellIdx % e->map->columns;
    for (uint8_t i = 0; i < 8; i++) {
        const int8_t newCellCol = cellCol + cellOffsets[i][0];
        if (newCellCol < 0 || newCellCol >= e->map->columns) {
            continue;
        }
        const int8_t newCellRow = cellRow + cellOffsets[i][1];
        if (newCellRow < 0 || newCellRow >= e->map->rows) {
            continue;
        }
        const int16_t newCellIdx = cellIndex(e, newCellCol, newCellRow);
        const mapCell *cell = (&e->cells[newCellIdx]);
        if (minDistance != min(minDistance, b2DistanceSquared(pos, cell->pos))) {
            closestCell = newCellIdx;
        }
    }

    return closestCell;
}

// the type of a pickup in the obs: its weapon, AMMO_CRATE_OBS_TYPE for an
// ammo crate or UTILITY_PICKUP_OBS_TYPE of its utility
static inline float pickupObsType(const weaponPickupEntity *pickup) {
    switch (pickup->kind) {
    case CRATE_PICKUP:
        return AMMO_CRATE_OBS_TYPE;
    case UTILITY_PICKUP:
        return UTILITY_PICKUP_OBS_TYPE(pickup->utility);
    default:
        return pickup->weapon;
    }
}

// normalize a drone's ammo count, setting infinite ammo as no ammo
static inline float scaleAmmo(const weaponSlot *weapon) {
    int8_t maxAmmo = weapon->weaponInfo->ammo;
    float scaledAmmo = 0;
    if (weapon->ammo != INFINITE) {
        scaledAmmo = scaleValue(weapon->ammo, maxAmmo, true);
    }
    return scaledAmmo;
}

// the map grid window around a drone: the map column and row of its top
// left cell and the world position of that cell's top left corner
typedef struct gridWindow {
    int16_t startCol;
    int16_t startRow;
    b2Vec2 origin;
} gridWindow;

static inline b2AABB gridCellBox(const gridWindow *w, const uint8_t col, const uint8_t row) {
    const b2Vec2 lower = {
        .x = w->origin.x + (col * WALL_THICKNESS) + GRID_OVERLAP_MARGIN,
        .y = w->origin.y + (row * WALL_THICKNESS) + GRID_OVERLAP_MARGIN,
    };
    const float size = WALL_THICKNESS - (2.0f * GRID_OVERLAP_MARGIN);
    return (b2AABB){.lowerBound = lower, .upperBound = {.x = lower.x + size, .y = lower.y + size}};
}

// clamps the grid cells overlapped by aabb to the window, returns false if
// none are in it
static bool gridCellRange(const gridWindow *w, const b2AABB aabb, uint8_t *minCol, uint8_t *maxCol, uint8_t *minRow, uint8_t *maxRow) {
    const int16_t c0 = (int16_t)floorf((aabb.lowerBound.x - w->origin.x) / WALL_THICKNESS);
    const int16_t c1 = (int16_t)floorf((aabb.upperBound.x - w->origin.x) / WALL_THICKNESS);
    const int16_t r0 = (int16_t)floorf((aabb.lowerBound.y - w->origin.y) / WALL_THICKNESS);
    const int16_t r1 = (int16_t)floorf((aabb.upperBound.y - w->origin.y) / WALL_THICKNESS);
    if (c1 < 0 || r1 < 0 || c0 >= MAP_OBS_COLUMNS || r0 >= MAP_OBS_ROWS) {
        return false;
    }
    *minCol = (uint8_t)max(c0, 0);
    *maxCol = (uint8_t)min(c1, MAP_OBS_COLUMNS - 1);
    *minRow = (uint8_t)max(r0, 0);
    *maxRow = (uint8_t)min(r1, MAP_OBS_ROWS - 1);
    return true;
}

// separating axis test between a convex polygon and an axis aligned box
static bool polygonOverlapsBox(const b2Vec2 *verts, const uint8_t count, const b2AABB box) {
    b2AABB polyBox = {.lowerBound = verts[0], .upperBound = verts[0]};
    for (uint8_t i = 1; i < count; i++) {
        polyBox.lowerBound = b2Min(polyBox.lowerBound, verts[i]);
        polyBox.upperBound = b2Max(polyBox.upperBound, verts[i]);
    }
    if (polyBox.upperBound.x <= box.lowerBound.x || polyBox.lowerBound.x >= box.upperBound.x || polyBox.upperBound.y <= box.lowerBound.y || polyBox.lowerBound.y >= box.upperBound.y) {
        return false;
    }

    const b2Vec2 corners[4] = {
        box.lowerBound,
        {.x = box.upperBound.x, .y = box.lowerBound.y},
        box.upperBound,
        {.x = box.lowerBound.x, .y = box.upperBound.y},
    };
    for (uint8_t i = 0; i < count; i++) {
        const b2Vec2 edge = b2Sub(verts[(i + 1) % count], verts[i]);
        const b2Vec2 axis = {.x = -edge.y, .y = edge.x};
        float polyMin = FLT_MAX, polyMax = -FLT_MAX, boxMin = FLT_MAX, boxMax = -FLT_MAX;
        for (uint8_t j = 0; j < count; j++) {
            const float p = b2Dot(axis, verts[j]);
            polyMin = fminf(polyMin, p);
            polyMax = fmaxf(polyMax, p);
        }
        for (uint8_t j = 0; j < 4; j++) {
            const float p = b2Dot(axis, corners[j]);
            boxMin = fminf(boxMin, p);
            boxMax = fmaxf(boxMax, p);
        }
        if (polyMax <= boxMin || boxMax <= polyMin) {
            return false;
        }
    }
    return true;
}

static void markFloatingWall(const gridWindow *w, const wallEntity *wall, uint8_t *codes) {
    uint8_t minCol, maxCol, minRow, maxRow;
    if (!gridCellRange(w, b2Shape_GetAABB(wall->shapeID), &minCol, &maxCol, &minRow, &maxRow)) {
        return;
    }

    const b2Polygon poly = b2Shape_GetPolygon(wall->shapeID);
    const b2Transform xf = b2Body_GetTransform(wall->bodyID);
    b2Vec2 verts[B2_MAX_POLYGON_VERTICES];
    for (int i = 0; i < poly.count; i++) {
        verts[i] = b2TransformPoint(xf, poly.vertices[i]);
    }

    // death > bouncy > standard if floating walls of different types share a cell
    const uint8_t type = (uint8_t)(wall->type + 1);
    for (uint8_t row = minRow; row <= maxRow; row++) {
        for (uint8_t col = minCol; col <= maxCol; col++) {
            if (!polygonOverlapsBox(verts, (uint8_t)poly.count, gridCellBox(w, col, row))) {
                continue;
            }
            uint8_t *code = &codes[col + (row * MAP_OBS_COLUMNS)];
            const uint8_t curType = (*code >> GRID_FLOATING_WALL_SHIFT) & 0x3;
            if (type > curType) {
                *code = (uint8_t)((*code & ~(0x3 << GRID_FLOATING_WALL_SHIFT)) | (type << GRID_FLOATING_WALL_SHIFT));
            }
        }
    }
}

static void markDrone(const gridWindow *w, const droneEntity *drone, uint8_t *codes) {
    float radius = DRONE_RADIUS;
    if (drone->shield != NULL) {
        radius = DRONE_SHIELD_RADIUS;
    }
    const b2AABB aabb = {
        .lowerBound = {.x = drone->pos.x - radius, .y = drone->pos.y - radius},
        .upperBound = {.x = drone->pos.x + radius, .y = drone->pos.y + radius},
    };
    uint8_t minCol, maxCol, minRow, maxRow;
    if (!gridCellRange(w, aabb, &minCol, &maxCol, &minRow, &maxRow)) {
        return;
    }

    for (uint8_t row = minRow; row <= maxRow; row++) {
        for (uint8_t col = minCol; col <= maxCol; col++) {
            const b2AABB box = gridCellBox(w, col, row);
            const b2Vec2 closest = b2Clamp(drone->pos, box.lowerBound, box.upperBound);
            if (b2DistanceSquared(closest, drone->pos) < radius * radius) {
                codes[col + (row * MAP_OBS_COLUMNS)] |= GRID_ENEMY_BIT;
            }
        }
    }
}

// fills a grid centered on the agent's cell with packed cell codes of
// static walls, floating walls, available weapon pickups and enemy drones,
// see GRID_* in settings.h
void computeMapObs(iwEnv *e, const uint8_t agentIdx, const uint16_t obsStartOffset) {
    const droneEntity *drone = safe_array_get_at(e->drones, agentIdx);
    const int16_t droneCellCol = drone->mapCellIdx % e->map->columns;
    const int16_t droneCellRow = drone->mapCellIdx / e->map->columns;
    gridWindow w = {
        .startCol = droneCellCol - (MAP_OBS_COLUMNS / 2),
        .startRow = droneCellRow - (MAP_OBS_ROWS / 2),
    };
    w.origin = (b2Vec2){
        .x = (w.startCol * WALL_THICKNESS) - ((e->map->columns * WALL_THICKNESS) / 2.0f),
        .y = (w.startRow * WALL_THICKNESS) - ((e->map->rows * WALL_THICKNESS) / 2.0f),
    };

    // static walls
    uint8_t codes[MAP_OBS_SIZE];
    for (uint8_t row = 0; row < MAP_OBS_ROWS; row++) {
        for (uint8_t col = 0; col < MAP_OBS_COLUMNS; col++) {
            uint8_t *code = &codes[col + (row * MAP_OBS_COLUMNS)];
            const int16_t mapCol = w.startCol + col;
            const int16_t mapRow = w.startRow + row;
            if (mapCol < 0 || mapCol >= e->map->columns || mapRow < 0 || mapRow >= e->map->rows) {
                *code = GRID_OUT_OF_BOUNDS;
                continue;
            }

            const int16_t cellIdx = cellIndex(e, mapCol, mapRow);
            if (!e->suddenDeathWallsPlaced) {
                // precomputed map layout if sudden death walls haven't been placed
                *code = (uint8_t)e->map->packedLayout[cellIdx];
                continue;
            }
            const mapCell *cell = &e->cells[cellIdx];
            *code = 0;
            if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
                *code = (uint8_t)(cell->ent->type + 1);
            }
        }
    }

    for (size_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        const wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        markFloatingWall(&w, wall, codes);
    }

    for (size_t i = 0; i < cc_array_size(e->pickups); i++) {
        const weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
        if (pickup->floatingWallsTouching > 0 || pickup->respawnWait > 0.0f) {
            continue;
        }
        const int16_t col = (pickup->mapCellIdx % e->map->columns) - w.startCol;
        const int16_t row = (pickup->mapCellIdx / e->map->columns) - w.startRow;
        if (col < 0 || col >= MAP_OBS_COLUMNS || row < 0 || row >= MAP_OBS_ROWS) {
            continue;
        }
        codes[col + (row * MAP_OBS_COLUMNS)] |= GRID_PICKUP_BIT;
    }

    for (uint8_t i = 0; i < e->numDrones; i++) {
        const droneEntity *otherDrone = safe_array_get_at(e->drones, i);
        if (i == agentIdx || otherDrone->dead || otherDrone->team == drone->team) {
            continue;
        }
        markDrone(&w, otherDrone, codes);
    }

    float *obs = agentObs(e, agentIdx) + obsStartOffset;
    for (uint16_t i = 0; i < MAP_OBS_SIZE; i++) {
        obs[i] = (float)codes[i];
    }
}

// computes observations for N nearest walls, floating walls, and weapon pickups
void computeNearObs(iwEnv *e, const droneEntity *drone, float *continuousObs) {
    nearWall nearWalls[NUM_NEAR_WALL_OBS];
    findNearWalls(e, drone, nearWalls, NUM_NEAR_WALL_OBS);

    uint16_t offset;

    // compute type and position of N nearest walls
    for (uint8_t i = 0; i < NUM_NEAR_WALL_OBS; i++) {
        const nearWall *wall = &nearWalls[i];

        offset = NEAR_WALL_INFO_OBS_OFFSET + (i * NEAR_WALL_INFO_OBS_SIZE);
        ASSERTF(offset + NEAR_WALL_INFO_OBS_SIZE <= FLOATING_WALL_INFO_OBS_OFFSET, "offset: %d", offset);
        const b2Vec2 wallRelPos = squashVec(b2Sub(wall->pos, drone->pos), MAX_OBS_WALL_POS);

        continuousObs[offset++] = (float)(wall->type + 1);
        continuousObs[offset++] = wallRelPos.x;
        continuousObs[offset] = wallRelPos.y;
    }

    const size_t numFloatingWalls = cc_array_size(e->floatingWalls);
    if (numFloatingWalls != 0) {
        // find N nearest floating walls
        nearEntity nearFloatingWalls[MAX_FLOATING_WALLS] = {0};
        for (uint8_t i = 0; i < numFloatingWalls; i++) {
            wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
            const nearEntity nearEnt = {
                .entity = wall,
                .distanceSquared = b2DistanceSquared(wall->pos, drone->pos),
            };
            nearFloatingWalls[i] = nearEnt;
        }
        if (numFloatingWalls > NUM_FLOATING_WALL_OBS) {
            insertionSortEntities(nearFloatingWalls, numFloatingWalls);
        }

        // compute type, position, angle and velocity of N nearest floating walls
        for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
            if (i == NUM_FLOATING_WALL_OBS) {
                break;
            }
            const wallEntity *wall = nearFloatingWalls[i].entity;

            const b2Transform wallTransform = b2Body_GetTransform(wall->bodyID);
            const b2Vec2 wallRelPos = b2Sub(wallTransform.p, drone->pos);
            const b2Vec2 wallRelVel = squashVec(b2Sub(wall->velocity, drone->velocity), MAX_OBS_WALL_SPEED);
            // floating walls are squares, so 4x the angle makes all 4
            // orientations that look the same have the same obs
            const float angle = 4.0f * b2Rot_GetAngle(wallTransform.q);

            // DEBUG_LOGF("floating wall %d cell %d", i, wall->mapCellIdx);

            offset = FLOATING_WALL_INFO_OBS_OFFSET + (i * FLOATING_WALL_INFO_OBS_SIZE);
            ASSERTF(offset + FLOATING_WALL_INFO_OBS_SIZE <= WEAPON_PICKUP_INFO_OBS_OFFSET, "offset: %d", offset);
            continuousObs[offset++] = (float)(wall->type + 1);
            const b2Vec2 wallObsPos = squashVec(wallRelPos, MAX_OBS_RELATIVE_POS);
            continuousObs[offset++] = wallObsPos.x;
            continuousObs[offset++] = wallObsPos.y;
            continuousObs[offset++] = sinf(angle);
            continuousObs[offset++] = cosf(angle);
            continuousObs[offset++] = wallRelVel.x;
            continuousObs[offset] = wallRelVel.y;
        }
    }

    const size_t numPickups = cc_array_size(e->pickups);
    if (numPickups != 0) {
        // find N nearest weapon pickups
        nearEntity nearPickups[MAX_WEAPON_PICKUPS] = {0};
        uint8_t numActivePickups = 0;
        for (uint8_t i = 0; i < numPickups; i++) {
            weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
            if (pickup->floatingWallsTouching > 0 || pickup->respawnWait > 0.0f) {
                continue;
            }

            const nearEntity nearEnt = {
                .entity = pickup,
                .distanceSquared = b2DistanceSquared(pickup->pos, drone->pos),
            };
            nearPickups[numActivePickups++] = nearEnt;
        }
        if (numActivePickups > NUM_WEAPON_PICKUP_OBS) {
            insertionSortEntities(nearPickups, numActivePickups);
        }

        // compute type and location of N nearest weapon pickups
        for (uint8_t i = 0; i < numActivePickups; i++) {
            if (i == NUM_WEAPON_PICKUP_OBS) {
                break;
            }
            const weaponPickupEntity *pickup = nearPickups[i].entity;

            // DEBUG_LOGF("pickup %d cell %d", i, pickup->mapCellIdx);

            offset = WEAPON_PICKUP_INFO_OBS_OFFSET + (i * WEAPON_PICKUP_INFO_OBS_SIZE);
            ASSERTF(offset + WEAPON_PICKUP_INFO_OBS_SIZE <= PROJECTILE_INFO_OBS_OFFSET, "offset: %d", offset);
            continuousObs[offset++] = pickupObsType(pickup);
            const b2Vec2 pickupRelPos = squashVec(b2Sub(pickup->pos, drone->pos), MAX_OBS_RELATIVE_POS);
            continuousObs[offset++] = pickupRelPos.x;
            continuousObs[offset] = pickupRelPos.y;
        }
    }
}

static inline void computeWeaponObs(const weaponSlot *weapon, float *obs, uint16_t offset) {
    obs[offset++] = weapon->weaponInfo->type;
    obs[offset++] = scaleAmmo(weapon);
    obs[offset++] = scaleValue(fmaxf(weapon->cooldown, 0.0f), weapon->weaponInfo->cooldown, true);
    obs[offset++] = scaleValue(weapon->charge, weapon->weaponInfo->maxCharge, true);
    obs[offset++] = scaleValue(weapon->fullChargeHoldTime, MAX_OBS_DRONE_CHARGE_HOLD, true);
}

static inline void computeDroneGrappleObs(const droneEntity *drone, const b2Vec2 pos, float *obs, uint16_t offset) {
    if (drone->grapple == NULL || !drone->grapple->welded) {
        return;
    }

    const b2Vec2 grappleRelPos = b2Sub(drone->grapple->pos, pos);
    const float grappleTension = b2DistanceJoint_GetCurrentLength(drone->grappleJointID) - drone->grappleLength;
    const bool grappleReeling = drone->utilityHeld && b2DistanceJoint_GetMotorForce(drone->grappleJointID) > 0.0f;

    uint8_t grappleAttachedTo = 0;
    const enum entityType attachedType = drone->grapple->grappledEnt->type;
    if (entityTypeIsWall(attachedType)) {
        grappleAttachedTo = 1;
        const wallEntity *attachedWall = drone->grapple->grappledEnt->entity;
        if (attachedWall->isFloating) {
            grappleAttachedTo = 2;
        }
    } else if (attachedType == DRONE_ENTITY) {
        grappleAttachedTo = 3;
    }

    obs[offset++] = true;
    const b2Vec2 grappleObsPos = squashVec(grappleRelPos, MAX_OBS_RELATIVE_POS);
    obs[offset++] = grappleObsPos.x;
    obs[offset++] = grappleObsPos.y;
    obs[offset++] = scaleValue(grappleTension, GRAPPLE_MAX_STRETCH, true);
    obs[offset++] = grappleReeling;
    obs[offset++] = scaleValue(drone->grapple->grappleHealth, GRAPPLE_HEALTH, true);
    oneHotEncode(obs, offset, grappleAttachedTo, 4);
}

// the utility's state, then the state of each utility's projectiles in
// utility type order; pos is what the grapple and mine positions are
// relative to. Mine state is of the drone's mine whatever its utility, as
// placed mines outlast a switch to another utility
static inline void computeUtilityObs(const iwEnv *e, const droneEntity *drone, const b2Vec2 pos, float *obs, uint16_t offset) {
    const utilitySlot *utility = &drone->utility;
    const utilityInformation *info = utility->utilityInfo;

    obs[offset++] = info->type;
    obs[offset++] = scaleValue(utility->ammo, info->ammo, true);
    obs[offset++] = scaleValue(fminf(utility->regen, info->regenTime), info->regenTime, true);
    // utilities that can always be fired, like portals, have no active
    // count to show
    if (info->maxActive == INFINITE) {
        obs[offset++] = 0.0f;
    } else {
        obs[offset++] = scaleValue(fminf(droneUtilityActive(e, drone), info->maxActive), info->maxActive, true);
    }

    computeDroneGrappleObs(drone, pos, obs, offset);
    offset += GRAPPLE_OBS_SIZE;

    const projectileEntity *mine = droneMine(e, drone);
    if (mine != NULL) {
        const b2Vec2 mineRelPos = squashVec(b2Sub(mine->pos, pos), MAX_OBS_RELATIVE_POS);
        obs[offset++] = !mine->welded;
        obs[offset++] = mine->welded;
        obs[offset++] = mineRelPos.x;
        obs[offset++] = mineRelPos.y;
    } else {
        offset += MINE_OBS_SIZE;
    }

    for (uint8_t i = 0; i < drone->openPortals; i++) {
        const portalEntity *portal = drone->portals[i];
        const b2Vec2 portalRelPos = squashVec(b2Sub(portal->pos, pos), MAX_OBS_RELATIVE_POS);
        const b2Vec2 toDrone = b2Normalize(b2Sub(pos, portal->pos));

        obs[offset++] = 1.0f;
        obs[offset++] = portalRelPos.x;
        obs[offset++] = portalRelPos.y;
        obs[offset++] = portal->normal.x;
        obs[offset++] = portal->normal.y;
        obs[offset++] = clampValue(b2Dot(portal->normal, toDrone), -1.0f, 1.0f);
    }
}

static inline float rayHitType(const entity *ent) {
    if (entityTypeIsWall(ent->type)) {
        const wallEntity *wall = ent->entity;
        if (wall->isFloating) {
            return (float)(RAY_HIT_FLOATING_WALL + wall->type);
        }
        return (float)(RAY_HIT_WALL + wall->type);
    }
    if (ent->type == DRONE_ENTITY || ent->type == SHIELD_ENTITY) {
        return (float)RAY_HIT_DRONE;
    }
    return (float)RAY_HIT_NONE;
}

typedef struct rayObsContext {
    const droneEntity *drone;
    // optional drone the ray is cast at, ignored like the casting drone
    const droneEntity *target;
    const entity *hitEnt;
    float fraction;
} rayObsContext;

static inline bool entIsDroneOrShield(const entity *ent, const droneEntity *drone) {
    return drone != NULL && (ent == drone->ent || (drone->shield != NULL && ent == drone->shield->ent));
}

// records the closest hit that isn't the casting or target drone or
// their shields
float rayObsCallback(b2ShapeId shapeID, b2Vec2 point, b2Vec2 normal, float fraction, void *context) {
    // these are unused but required by the b2CastResultFcn callback prototype
    MAYBE_UNUSED(point);
    MAYBE_UNUSED(normal);

    rayObsContext *ctx = context;
    const entity *ent = b2Shape_GetUserData(shapeID);
    if (ent == NULL || entIsDroneOrShield(ent, ctx->drone) || entIsDroneOrShield(ent, ctx->target)) {
        return -1.0f;
    }
    if (fraction < ctx->fraction) {
        ctx->fraction = fraction;
        ctx->hitEnt = ent;
    }
    return fraction;
}

// casts a circle of radius from the drone along translation, returns the
// hit type and sets fraction to the fraction of translation traveled
// before the hit, or 1 if nothing was hit
static float castDroneCircle(const iwEnv *e, const droneEntity *drone, const float radius, const b2Vec2 translation, const b2QueryFilter filter, float *fraction) {
    const b2ShapeProxy proxy = b2MakeProxy(&drone->pos, 1, radius);
    rayObsContext ctx = {.drone = drone, .target = NULL, .hitEnt = NULL, .fraction = 1.0f};
    b2World_CastShape(e->worldID, &proxy, translation, filter, rayObsCallback, &ctx);
    *fraction = ctx.fraction;
    if (ctx.hitEnt == NULL) {
        return (float)RAY_HIT_NONE;
    }
    return rayHitType(ctx.hitEnt);
}

// casts the drone's wall colliding shape along the damped path it would
// travel in RAY_TIME_HORIZON seconds at velocity, and writes the hit
// type and time until the hit squashed by OBS_RAY_TIME_KNEE
static void computeVelocityRayObs(const iwEnv *e, const droneEntity *drone, const b2Vec2 velocity, float *obs) {
    obs[0] = (float)RAY_HIT_NONE;
    obs[1] = 1.0f;
    const float speed = b2Length(velocity);
    if (speed < 0.1f) {
        return;
    }

    float damping = DRONE_LINEAR_DAMPING;
    if (drone->braking) {
        damping = DRONE_BRAKE_DAMPING_COEF;
    }
    // with linear damping k, distance travelled after t seconds is
    // speed * (1 - e^(-kt)) / k
    float pathScale = RAY_TIME_HORIZON;
    if (damping > 0.0f) {
        pathScale = (1.0f - expf(-damping * RAY_TIME_HORIZON)) / damping;
    }
    const b2Vec2 translation = b2MulSV(pathScale, velocity);

    float radius = DRONE_RADIUS;
    if (drone->shield != NULL) {
        radius = DRONE_SHIELD_RADIUS;
    }
    const b2QueryFilter filter = {.categoryBits = DRONE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
    float fraction;
    obs[0] = castDroneCircle(e, drone, radius, translation, filter, &fraction);
    if (obs[0] == (float)RAY_HIT_NONE) {
        return;
    }

    // invert the damped distance to get the time of the hit
    float hitTime = fraction * RAY_TIME_HORIZON;
    if (damping > 0.0f) {
        hitTime = -logf(fmaxf(1.0f - (fraction * damping * pathScale), FLT_MIN)) / damping;
    }
    obs[1] = squashValue(fminf(hitTime, RAY_TIME_HORIZON), OBS_RAY_TIME_KNEE);
}

// writes ray observations for what the drone's shot and movement would hit
static void computeDroneRayObs(const iwEnv *e, const droneEntity *drone, float *obs) {
    float distance = drone->weapon.weaponInfo->maxDistance;
    if (distance == INFINITE) {
        distance = MAX_DISTANCE;
    } else {
        distance = fminf(distance, MAX_DISTANCE);
    }

    // what a shot fired now would hit first, walls or drones
    const b2Vec2 aimTranslation = b2MulSV(MAX_DISTANCE, drone->lastAim);
    const b2QueryFilter aimFilter = {
        .categoryBits = PROJECTILE_SHAPE,
        .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE | SHIELD_SHAPE,
    };
    float fraction;
    obs[0] = castDroneCircle(e, drone, drone->weapon.weaponInfo->radius, aimTranslation, aimFilter, &fraction);
    obs[1] = squashValue(fraction * MAX_DISTANCE, MAX_OBS_RAY_DISTANCE);

    // where the drone is drifting to
    computeVelocityRayObs(e, drone, drone->velocity, obs + RAY_OBS_SIZE);
}

static void computeAgentRayObs(const iwEnv *e, const droneEntity *drone, float *obs) {
    computeDroneRayObs(e, drone, obs);

    // where the drone would drift to if it fired now
    const b2Vec2 recoilVelocity = b2MulSV(-drone->weapon.weaponInfo->recoilMagnitude / DRONE_MASS, drone->lastAim);
    computeVelocityRayObs(e, drone, b2Add(drone->velocity, recoilVelocity), obs + (2 * RAY_OBS_SIZE));
}

static void computeEnemyRayObs(const iwEnv *e, const droneEntity *drone, const droneEntity *enemyDrone, float *obs) {
    // writes the type of the first wall or other drone between the
    // drone and the enemy and the distance to it
    const b2Vec2 translation = b2Sub(enemyDrone->pos, drone->pos);
    const float distance = b2Length(translation);
    const b2QueryFilter filter = {
        .categoryBits = PROJECTILE_SHAPE,
        .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | DRONE_SHAPE | SHIELD_SHAPE,
    };
    rayObsContext ctx = {.drone = drone, .target = enemyDrone, .hitEnt = NULL, .fraction = 1.0f};
    b2World_CastRay(e->worldID, drone->pos, translation, filter, rayObsCallback, &ctx);
    if (ctx.hitEnt == NULL) {
        obs[0] = (float)RAY_HIT_DRONE;
        obs[1] = squashValue(distance, MAX_OBS_RAY_DISTANCE);
        return;
    }
    obs[0] = rayHitType(ctx.hitEnt);
    obs[1] = squashValue(ctx.fraction * distance, MAX_OBS_RAY_DISTANCE);
}

// sort key of a projectile for the projectile obs: threats, projectiles
// whose closest approach to the drone at the current relative velocity is
// within hitting or blast range within PROJECTILE_THREAT_TIME, come first,
// soonest first, then the rest by distance
static float projectileObsKey(const droneEntity *drone, const projectileEntity *projectile, const float distance) {
    const b2Vec2 relPos = b2Sub(projectile->pos, drone->pos);
    const b2Vec2 relVel = b2Sub(projectile->velocity, drone->velocity);
    const float speedSq = b2LengthSquared(relVel);
    float t = 0.0f;

    if (speedSq > 1e-8f) {
        t = fmaxf(0.0f, -b2Dot(relPos, relVel) / speedSq);
    }

    float reach = DRONE_RADIUS + projectile->weaponInfo->radius;
    if (projectile->weaponInfo->explosive) {
        b2ExplosionDef explosion = b2DefaultExplosionDef();
        weaponExplosion(projectile->weaponInfo->type, &explosion);
        reach += explosion.radius;
    }

    if (t <= PROJECTILE_THREAT_TIME && b2Length(b2MulAdd(relPos, t, relVel)) <= reach) {
        return t;
    }

    return PROJECTILE_THREAT_TIME + distance;
}

typedef struct sortedProjectile {
    const projectileEntity *projectile;
    float distance;
    float key;
} sortedProjectile;

static void insertionSortProjectiles(sortedProjectile *arr, const uint16_t size) {
    for (int16_t i = 1; i < (int16_t)size; i++) {
        const sortedProjectile key = arr[i];
        int16_t j = i - 1;
        while (j >= 0 && arr[j].key > key.key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// writes the agent's own projectiles and other drones' projectiles into
// their slots, see NUM_OWN_PROJECTILE_OBS
static void computeProjectileObs(iwEnv *e, const uint8_t agentIdx, const droneEntity *agentDrone, float *continuousObs) {
    const size_t numProjectiles = cc_array_size(e->projectiles);
    if (numProjectiles == 0) {
        return;
    }

    sortedProjectile own[numProjectiles];
    sortedProjectile enemy[numProjectiles];
    uint16_t numOwn = 0;
    uint16_t numEnemy = 0;

    for (size_t i = 0; i < numProjectiles; i++) {
        const projectileEntity *projectile = safe_array_get_at(e->projectiles, i);
        const float distance = b2Distance(projectile->pos, agentDrone->pos);
        const sortedProjectile sp = {
            .projectile = projectile,
            .distance = distance,
            .key = projectileObsKey(agentDrone, projectile, distance),
        };

        if (projectile->droneIdx == agentIdx) {
            own[numOwn++] = sp;
        } else {
            enemy[numEnemy++] = sp;
        }
    }
    insertionSortProjectiles(own, numOwn);
    insertionSortProjectiles(enemy, numEnemy);

    for (uint8_t side = 0; side < 2; side++) {
        const sortedProjectile *sorted = own;
        uint16_t count = min(numOwn, NUM_OWN_PROJECTILE_OBS);
        uint16_t start = PROJECTILE_INFO_OBS_OFFSET;
        if (side == 1) {
            sorted = enemy;
            count = min(numEnemy, NUM_ENEMY_PROJECTILE_OBS);
            start = ENEMY_PROJECTILE_INFO_OBS_OFFSET;
        }
        for (uint16_t i = 0; i < count; i++) {
            const projectileEntity *projectile = sorted[i].projectile;
            const b2Vec2 relPos = squashVec(b2Sub(projectile->pos, agentDrone->pos), MAX_OBS_RELATIVE_POS);
            const b2Vec2 relVel = squashVec(b2Sub(projectile->velocity, agentDrone->velocity), MAX_OBS_PROJECTILE_SPEED);

            uint16_t offset = start + (i * PROJECTILE_INFO_OBS_SIZE);
            ASSERTF(offset + PROJECTILE_INFO_OBS_SIZE <= ENEMY_DRONE_OBS_OFFSET, "offset: %d", offset);
            continuousObs[offset++] = projectile->weaponInfo->type;
            continuousObs[offset++] = relPos.x;
            continuousObs[offset++] = relPos.y;
            continuousObs[offset++] = relVel.x;
            continuousObs[offset++] = relVel.y;
            continuousObs[offset] = squashValue(sorted[i].distance, MAX_OBS_RELATIVE_POS);
        }
    }
}

void computeObs(iwEnv *e) {
    for (uint8_t agentIdx = 0; agentIdx < e->numAgents; agentIdx++) {
        droneEntity *agentDrone = safe_array_get_at(e->drones, agentIdx);
        // if the drone is dead, only compute observations if it died
        // this step and it isn't out of bounds
        if (agentDrone->livesLeft == 0 && (!agentDrone->diedThisStep || agentDrone->mapCellIdx == -1)) {
            continue;
        }

        // compute discrete map observations
        float *obs = agentObs(e, agentIdx);
        const uint16_t discreteObsStart = 0;
        memset(obs, 0x0, e->obsSize * sizeof(float));
        computeMapObs(e, agentIdx, discreteObsStart);

        // compute continuous observations
        uint16_t continuousObsOffset;
        const uint16_t continuousObsStart = discreteObsStart + DISCRETE_OBS_SIZE;
        float *continuousObs = obs + continuousObsStart;

        computeNearObs(e, agentDrone, continuousObs);

        computeProjectileObs(e, agentIdx, agentDrone, continuousObs);

        // compute enemy drone observations
        bool hitShot = false;
        bool tookShot = false;
        uint8_t processedDrones = 0;
        for (uint8_t i = 0; i < e->numDrones; i++) {
            if (i == agentIdx) {
                continue;
            }

            if (agentDrone->stepInfo.shotHit[i] != 0.0f || agentDrone->stepInfo.explosionHit[i] != 0.0f) {
                hitShot = true;
            }
            if (agentDrone->stepInfo.shotTaken[i] != 0.0f || agentDrone->stepInfo.explosionTaken[i] != 0.0f) {
                tookShot = true;
            }

            droneEntity *enemyDrone = safe_array_get_at(e->drones, i);
            if (enemyDrone->livesLeft == 0) {
                processedDrones++;
                continue;
            }

            const b2Vec2 enemyDroneRelPos = b2Sub(enemyDrone->pos, agentDrone->pos);
            const float enemyDroneDistance = b2Distance(enemyDrone->pos, agentDrone->pos);
            const b2Vec2 enemyDroneRelVel = squashVec(b2Sub(enemyDrone->velocity, agentDrone->velocity), MAX_OBS_ENTITY_SPEED);
            const b2Vec2 enemyDroneRelNormPos = b2Normalize(b2Sub(enemyDrone->pos, agentDrone->pos));

            float enemyShieldHealth = 0.0f;
            float enemyShieldDuration = 0.0f;
            if (enemyDrone->shield != NULL) {
                enemyShieldHealth = enemyDrone->shield->health;
                enemyShieldDuration = enemyDrone->shield->duration;
            }

            continuousObsOffset = ENEMY_DRONE_OBS_OFFSET + (processedDrones * ENEMY_DRONE_OBS_SIZE);

            computeWeaponObs(&enemyDrone->weapon, continuousObs, continuousObsOffset);
            continuousObsOffset += WEAPON_OBS_SIZE;
            computeUtilityObs(e, enemyDrone, agentDrone->pos, continuousObs, continuousObsOffset);
            continuousObsOffset += UTILITY_OBS_SIZE;

            const b2Vec2 enemyDroneObsPos = squashVec(enemyDroneRelPos, MAX_OBS_RELATIVE_POS);
            continuousObs[continuousObsOffset++] = enemyDroneObsPos.x;
            continuousObs[continuousObsOffset++] = enemyDroneObsPos.y;
            continuousObs[continuousObsOffset++] = squashValue(enemyDroneDistance, MAX_OBS_DRONE_DISTANCE);
            continuousObs[continuousObsOffset++] = enemyDroneRelVel.x;
            continuousObs[continuousObsOffset++] = enemyDroneRelVel.y;
            continuousObs[continuousObsOffset++] = enemyDroneRelNormPos.x;
            continuousObs[continuousObsOffset++] = enemyDroneRelNormPos.y;
            continuousObs[continuousObsOffset++] = enemyDrone->lastAim.x;
            continuousObs[continuousObsOffset++] = enemyDrone->lastAim.y;
            continuousObs[continuousObsOffset++] = scaleValue(enemyShieldHealth, DRONE_SHIELD_HEALTH, true);
            continuousObs[continuousObsOffset++] = scaleValue(enemyShieldDuration, DRONE_SHIELD_RESPAWN_DURATION, true);

            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->energyLeft, DRONE_ENERGY_MAX, true);
            continuousObs[continuousObsOffset++] = enemyDrone->energyFullyDepleted;
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->energyRefillWait, DRONE_ENERGY_REFILL_EMPTY_WAIT, true);
            continuousObs[continuousObsOffset++] = enemyDrone->braking;
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->burstCooldown, DRONE_BURST_COOLDOWN, true);
            continuousObs[continuousObsOffset++] = enemyDrone->chargingBurst;
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->burstCharge, DRONE_ENERGY_MAX, true);

            continuousObs[continuousObsOffset++] = tookShot;
            continuousObs[continuousObsOffset++] = hitShot;
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->livesLeft, DRONE_LIVES, true);
            continuousObs[continuousObsOffset++] = !enemyDrone->dead;
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->respawnWait, DRONE_RESPAWN_WAIT, true);
            // sin and cos of the angle to turn our aim by to point at this enemy
            continuousObs[continuousObsOffset++] = b2Cross(agentDrone->lastAim, enemyDroneRelNormPos);
            continuousObs[continuousObsOffset++] = b2Dot(agentDrone->lastAim, enemyDroneRelNormPos);
            ASSERT(continuousObsOffset == ENEMY_DRONE_OBS_OFFSET + (processedDrones * ENEMY_DRONE_OBS_SIZE) + ENEMY_DRONE_RAY_OBS_IDX);
            if (!agentDrone->dead && !enemyDrone->dead) {
                computeEnemyRayObs(e, agentDrone, enemyDrone, continuousObs + continuousObsOffset);
            }
            continuousObsOffset += RAY_OBS_SIZE;

            processedDrones++;
            ASSERTF(continuousObsOffset == ENEMY_DRONE_OBS_OFFSET + (processedDrones * ENEMY_DRONE_OBS_SIZE), "offset: %d", continuousObsOffset);
        }

        // compute active drone observations
        continuousObsOffset = ENEMY_DRONE_OBS_OFFSET + ((e->numDrones - 1) * ENEMY_DRONE_OBS_SIZE);

        // sin and cos of the angle from our aim to our velocity, both 0
        // when barely moving
        float aimVelSin = 0.0f;
        float aimVelCos = 0.0f;
        if (b2Length(agentDrone->velocity) > 0.1f) {
            const b2Vec2 velDir = b2Normalize(agentDrone->velocity);
            aimVelSin = b2Cross(agentDrone->lastAim, velDir);
            aimVelCos = b2Dot(agentDrone->lastAim, velDir);
        }
        const b2Vec2 agentSquashedVel = squashVec(agentDrone->velocity, MAX_OBS_AGENT_SPEED);

        float shieldHealth = 0.0f;
        float shieldDuration = 0.0f;
        if (agentDrone->shield != NULL) {
            shieldHealth = agentDrone->shield->health;
            shieldDuration = agentDrone->shield->duration;
        }

        computeWeaponObs(&agentDrone->weapon, continuousObs, continuousObsOffset);
        continuousObsOffset += WEAPON_OBS_SIZE;
        computeUtilityObs(e, agentDrone, agentDrone->pos, continuousObs, continuousObsOffset);
        continuousObsOffset += UTILITY_OBS_SIZE;
        continuousObs[continuousObsOffset++] = agentDrone->triggerHeld;
        continuousObs[continuousObsOffset++] = agentDrone->utilityHeld;

        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->pos.x, MAX_OBS_AGENT_POS, false);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->pos.y, MAX_OBS_AGENT_POS, false);
        continuousObs[continuousObsOffset++] = agentSquashedVel.x;
        continuousObs[continuousObsOffset++] = agentSquashedVel.y;
        continuousObs[continuousObsOffset++] = agentDrone->lastAim.x;
        continuousObs[continuousObsOffset++] = agentDrone->lastAim.y;
        continuousObs[continuousObsOffset++] = aimVelSin;
        continuousObs[continuousObsOffset++] = aimVelCos;
        continuousObs[continuousObsOffset++] = scaleValue(shieldHealth, DRONE_SHIELD_HEALTH, true);
        continuousObs[continuousObsOffset++] = scaleValue(shieldDuration, DRONE_SHIELD_RESPAWN_DURATION, true);

        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->energyLeft, DRONE_ENERGY_MAX, true);
        continuousObs[continuousObsOffset++] = agentDrone->energyFullyDepleted;
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->energyRefillWait, DRONE_ENERGY_REFILL_EMPTY_WAIT, true);
        continuousObs[continuousObsOffset++] = agentDrone->braking;
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->burstCooldown, DRONE_BURST_COOLDOWN, true);
        continuousObs[continuousObsOffset++] = agentDrone->chargingBurst;
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->burstCharge, DRONE_ENERGY_MAX, true);

        continuousObs[continuousObsOffset++] = hitShot;
        continuousObs[continuousObsOffset++] = tookShot;
        continuousObs[continuousObsOffset++] = agentDrone->stepInfo.ownShotTaken;
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->livesLeft, DRONE_LIVES, true);
        continuousObs[continuousObsOffset++] = !agentDrone->dead;
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->respawnWait, DRONE_RESPAWN_WAIT, true);

        ASSERT(continuousObsOffset == ENEMY_DRONE_OBS_OFFSET + ((e->numDrones - 1) * ENEMY_DRONE_OBS_SIZE) + DRONE_RAY_OBS_IDX);
        if (!agentDrone->dead) {
            computeAgentRayObs(e, agentDrone, continuousObs + continuousObsOffset);
        }
        continuousObsOffset += NUM_DRONE_RAY_OBS * RAY_OBS_SIZE;

        ASSERTF(continuousObsOffset == ENEMY_DRONE_OBS_OFFSET + ((e->numDrones - 1) * ENEMY_DRONE_OBS_SIZE) + DRONE_OBS_SIZE, "offset: %d", continuousObsOffset);

        continuousObs[continuousObsOffset++] = !e->easyEp;
        continuousObs[continuousObsOffset++] = scaleValue(e->stepsLeft, e->totalSteps, true);
        continuousObs[continuousObsOffset++] = scaleValue(e->suddenDeathSteps, e->totalSuddenDeathSteps, true);
    }
}

// whether a weapon can be part of a combo loadout: the standard weapon is
// the fallback and utility projectiles aren't weapons, so only weapons that
// spawn as pickups are, and weapons with 1 ammo are excluded since ammo
// crates can't refill them once they're fired
static inline bool loadoutWeapon(const enum weaponType type) {
    return weaponInfos[type]->spawnWeight > 0.0f && weaponInfos[type]->ammo > 1;
}

// picks a weapon and a utility uniformly, ignoring spawn weights so rarely
// spawned weapons are practiced as much as common ones
static void randLoadout(iwEnv *e, enum weaponType *weapon, enum utilityType *utility) {
    enum weaponType candidates[_NUM_WEAPONS];
    uint8_t numCandidates = 0;
    for (uint8_t i = STANDARD_WEAPON + 1; i < NUM_WEAPONS; i++) {
        if (loadoutWeapon(i)) {
            candidates[numCandidates++] = i;
        }
    }
    ASSERT(numCandidates >= 1);

    *weapon = candidates[randInt(&e->rng, 0, numCandidates - 1)];
    *utility = randInt(&e->rng, UNSET_UTILITY + 1, NUM_UTILITIES - 1);
}

static void droneSetLoadout(droneEntity *drone, const enum weaponType weapon, const enum utilityType utility) {
    drone->weapon.weaponInfo = weaponInfos[weapon];
    resetWeapon(&drone->weapon);
    drone->weapon.ammo = weaponInfos[weapon]->ammo;
    droneSetUtility(drone, utility);
}

void setupEnv(iwEnv *e) {
    e->needsReset = false;

    e->stepsLeft = e->totalSteps;
    e->suddenDeathSteps = e->totalSuddenDeathSteps;
    e->suddenDeathWallCounter = 0;

    e->lastSpawnQuad = -1;
    for (uint8_t i = 0; i < _MAX_DRONES; i++) {
        e->botState[i] = (scriptedBotState){.pickupIdx = -1, .pathSrcIdx = -1, .pathDstIdx = -1, .spotCellIdx = -1};
    }

    e->easyEp = e->easyEpProb > 0.0f && randFloat(&e->rng, 0.0f, 1.0f) < e->easyEpProb;

    int8_t mapIdx = e->pinnedMapIdx;
    if (e->easyEp && e->numAgents != e->numDrones) {
        mapIdx = EASY_EP_MAP_IDX;
    } else if (e->pinnedMapIdx == -1) {
        uint8_t firstMap = 0;
        // don't evaluate on the boring empty map
        if (!e->isTraining) {
            firstMap = 1;
        }
        mapIdx = randInt(&e->rng, firstMap, NUM_MAPS - 1);
    }
    DEBUG_LOGF("setting up map %d", mapIdx);
    setupMap(e, mapIdx);

    DEBUG_LOG("creating drones");
    for (uint8_t i = 0; i < e->numDrones; i++) {
        createDrone(e, i);
    }

    // start drones with random weapon and utility loadouts so rare ones
    // and combos get practiced; easy episodes are left alone
    e->comboEp = e->isTraining && !e->easyEp && e->comboTrainingProb > 0.0f && randFloat(&e->rng, 0.0f, 1.0f) < e->comboTrainingProb;
    if (e->comboEp) {
        const bool symmetric = randInt(&e->rng, 0, 1) == 1;
        enum weaponType weapon = UNSET_WEAPON;
        enum utilityType utility = UNSET_UTILITY;
        for (uint8_t i = 0; i < e->numDrones; i++) {
            if (i == 0 || !symmetric) {
                randLoadout(e, &weapon, &utility);
            }
            droneEntity *drone = safe_array_get_at(e->drones, i);
            droneSetLoadout(drone, weapon, utility);
        }
    }

    DEBUG_LOG("placing floating walls");
    placeRandFloatingWalls(e, mapIdx);

    DEBUG_LOG("creating pickups");
    // the first pickups are utilities, one nearer each drone, and the rest
    // are weapons; once picked up they respawn as any type by spawn weight.
    // Drones in combo episodes already start with a utility
    uint8_t utilityPickups = 0;
    if (!e->comboEp) {
        utilityPickups = min(e->numDrones, maps[mapIdx]->weaponPickups);
    }
    // start spawning pickups in a random quadrant
    e->lastSpawnQuad = randInt(&e->rng, 0, 3);
    for (uint8_t i = 0; i < maps[mapIdx]->weaponPickups; i++) {
        if (i < utilityPickups) {
            createUtilityPickup(e, i);
        } else {
            createWeaponPickup(e);
        }
    }

    e->roundState = ROUND_STATE_PLAYING;
    e->tick_frames_left = 0;
    if (e->client != NULL) {
        e->roundState = ROUND_STATE_STARTING;
        e->tick_frames_left = START_READY_TIME * e->frameRate;

        setupEnvCamera(e);
    }

    computeObs(e);
}

// sets the timing related variables for the environment depending on
// the frame rate
void setEnvFrameRate(iwEnv *e) {
    float frameRate = TRAINING_FRAME_RATE;
    e->box2dSubSteps = TRAINING_BOX2D_SUBSTEPS;
    e->frameSkip = frameRate / TRAINING_ACTIONS_PER_SECOND;
    // set a higher frame rate and physics substeps when evaluating
    // to make it more enjoyable to play
    if (!e->isTraining) {
        frameRate = EVAL_FRAME_RATE;
        e->box2dSubSteps = EVAL_BOX2D_SUBSTEPS;
        e->frameSkip = 1;
    }

    e->frameRate = frameRate;
    e->deltaTime = 1.0f / (float)frameRate;

    e->totalSteps = ROUND_STEPS * frameRate;
    e->totalSuddenDeathSteps = SUDDEN_DEATH_STEPS * frameRate;
}

iwEnv *initEnv(iwEnv *e, uint8_t numDrones, uint8_t numAgents, int8_t mapIdx, uint64_t seed, bool enableTeams, bool isTraining, bool continuousActions, uint8_t botPolicy, float botCLNoise, float botCLDecay) {
    DEBUG_LOGF("seed: %lu", seed);

    e->numDrones = numDrones;
    e->numAgents = numAgents;
    e->num_agents = numAgents;
    e->teamsEnabled = enableTeams;
    e->numTeams = numDrones;
    if (e->teamsEnabled) {
        e->numTeams = 2;
    }
    e->isTraining = isTraining;
    e->botPolicy = botPolicy;
    e->botCLNoise = botCLNoise;
    e->botCLDecay = botCLDecay;
    setBotPersonality(e, botPolicy);

    e->obsSize = obsSize(e->numDrones);

    e->continuousActions = continuousActions;

    setEnvFrameRate(e);
    e->rng = seed;
    e->renderRng = e->rng;
    e->needsReset = false;

    b2WorldDef worldDef = b2DefaultWorldDef();
    worldDef.gravity = (b2Vec2){.x = 0.0f, .y = 0.0f};
    e->worldID = b2CreateWorld(&worldDef);
    e->pinnedMapIdx = mapIdx;
    e->mapIdx = -1;

    e->idPool = b2CreateIdPool();
    create_array(&e->entities, 128);

    e->cells = fastCalloc(MAX_CELLS, sizeof(mapCell));
    e->numCells = 0;
    create_array(&e->walls, 128);
    create_array(&e->floatingWalls, MAX_FLOATING_WALLS);
    create_array(&e->drones, e->numDrones);
    create_array(&e->pickups, MAX_WEAPON_PICKUPS);
    create_array(&e->projectiles, 64);
    create_array(&e->explosions, 8);
    create_array(&e->boostSparks, 8);
    create_array(&e->explodingProjectiles, 8);
    create_array(&e->dronePieces, 16);

    e->humanInput = false;
    e->humanDroneInput = 0;
    e->connectedControllers = 0;

    e->cachedActions = fastCalloc(e->numDrones, sizeof(agentActions));

#ifdef PUF_DEBUG
    create_array(&e->debugPoints, 4);
#endif

    return e;
}

void setRewards(iwEnv *e, float winReward, float lifeReward, float killReward, float selfKillPunishment, float teammateDeathPunishment, float teammateKillPunishment, float energyEmptiedPunishment, float weaponPickupReward, float shieldBreakReward, float hitRewardCoef) {
    e->winReward = winReward;
    e->lifeReward = lifeReward;
    e->killReward = killReward;
    e->selfKillPunishment = selfKillPunishment;
    e->teammateDeathPunishment = teammateDeathPunishment;
    e->teammateKillPunishment = teammateKillPunishment;
    e->energyEmptiedPunishment = energyEmptiedPunishment;
    e->weaponPickupReward = weaponPickupReward;
    e->shieldBreakReward = shieldBreakReward;
    e->hitRewardCoef = hitRewardCoef;
}

void clearEnv(iwEnv *e) {
    // rewards get cleared in stepEnv every step
    // memset(e->masks, 1, e->numAgents * sizeof(uint8_t));

    e->burstSerial = 0;
    e->episodeLength = 0;
    // pickups alive at the end of an episode are destroyed without being
    // disabled, so reset the spawned counts here
    memset(e->spawnedWeaponPickups, 0x0, sizeof(e->spawnedWeaponPickups));
    memset(e->spawnedUtilityPickups, 0x0, sizeof(e->spawnedUtilityPickups));
    e->spawnedCrates = 0;
    memset(e->stats, 0x0, sizeof(e->stats));
    memset(e->analysisStats, 0x0, sizeof(e->analysisStats));

    // portals need to be destroyed before the walls they're attached to
    // and the drones that own them
    for (size_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        destroyDronePortals(e, drone);
    }

    for (size_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
        wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        destroyWall(e, wall, false);
    }

    for (size_t i = 0; i < cc_array_size(e->pickups); i++) {
        weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
        destroyWeaponPickup(e, pickup);
    }

    for (size_t i = 0; i < cc_array_size(e->projectiles); i++) {
        projectileEntity *p = safe_array_get_at(e->projectiles, i);
        destroyProjectile(e, p, false, false);
    }

    for (size_t i = 0; i < cc_array_size(e->explosions); i++) {
        explosionInfo *explosion = safe_array_get_at(e->explosions, i);
        fastFree(explosion);
    }

    for (size_t i = 0; i < cc_array_size(e->boostSparks); i++) {
        boostSparksInfo *sparks = safe_array_get_at(e->boostSparks, i);
        fastFree(sparks);
    }

    for (size_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        destroyDrone(e, drone);
    }

    for (size_t i = 0; i < cc_array_size(e->dronePieces); i++) {
        dronePieceEntity *piece = safe_array_get_at(e->dronePieces, i);
        destroyDronePiece(e, piece);
    }

    cc_array_remove_all(e->floatingWalls);
    cc_array_remove_all(e->pickups);
    cc_array_remove_all(e->projectiles);
    cc_array_remove_all(e->explodingProjectiles);
    cc_array_remove_all(e->explosions);
    cc_array_remove_all(e->boostSparks);
    cc_array_remove_all(e->drones);
    cc_array_remove_all(e->dronePieces);

    memset(e->minesCreated, 0x0, sizeof(e->minesCreated));
    memset(e->droneSwapEffects, 0x0, sizeof(e->droneSwapEffects));
}

void puf_close(iwEnv *e) {
    clearEnv(e);

    fastFree(e->cachedActions);

    for (size_t i = 0; i < cc_array_size(e->walls); i++) {
        wallEntity *wall = safe_array_get_at(e->walls, i);
        destroyWall(e, wall, false);
    }

    for (size_t i = 0; i < cc_array_size(e->entities); i++) {
        entity *ent = safe_array_get_at(e->entities, i);
        fastFree(ent->id);
        fastFree(ent);
    }
    b2DestroyIdPool(&e->idPool);

    cc_array_destroy(e->entities);
    fastFree(e->cells);
    cc_array_destroy(e->walls);
    cc_array_destroy(e->drones);
    cc_array_destroy(e->floatingWalls);
    cc_array_destroy(e->pickups);
    cc_array_destroy(e->projectiles);
    cc_array_destroy(e->explosions);
    cc_array_destroy(e->boostSparks);
    cc_array_destroy(e->explodingProjectiles);
    cc_array_destroy(e->dronePieces);

    b2DestroyWorld(e->worldID);

#ifdef PUF_DEBUG
    cc_array_destroy(e->debugPoints);
#endif
}

void puf_reset(iwEnv *e) {
    clearEnv(e);
    setupEnv(e);
}

static inline bool droneDiedThisStep(const droneEntity *drone) {
    return drone->dead && drone->diedThisStep;
}

static inline float hitReward(float impulse) {
    return fminf(impulse * DRONE_INV_MASS / HIT_DV_REF, 1.0f);
}

float computeReward(iwEnv *e, droneEntity *drone) {
    float reward = 0.0f;

    if (droneDiedThisStep(drone)) {
        reward -= e->lifeReward;
        if (drone->killedBy == drone->idx) {
            reward += e->selfKillPunishment;
        }
    }

    if (drone->stepInfo.emptiedEnergy) {
        reward += e->energyEmptiedPunishment;
    }

    // only reward a pickup if it filled an unset slot, replaced the
    // standard weapon or refilled a weapon from empty to full; every
    // weapon is better than the standard weapon, but other weapons are
    // situationally better so don't reward replacing a non-standard
    // weapon or topping up ammo
    if (drone->stepInfo.pickedUpWeapon && drone->stepInfo.slotFilled) {
        reward += e->weaponPickupReward;
    }

    for (uint8_t i = 0; i < e->numDrones; i++) {
        if (i == drone->idx) {
            continue;
        }
        droneEntity *otherDrone = safe_array_get_at(e->drones, i);
        const bool onTeam = drone->team == otherDrone->team;

        // TODO: punish for hitting teammates?
        if (!onTeam) {
            // shot and explosion strengths are both the impulse delivered
            // to the hit drone, so they share a coefficient
            const float dealt = drone->stepInfo.shotHit[i] + drone->stepInfo.explosionHit[i];
            const float taken = drone->stepInfo.shotTaken[i] + drone->stepInfo.explosionTaken[i];

            // make hit and shield break rewards symmetric in self play
            if (e->numAgents == e->numDrones) {
                reward += (hitReward(dealt) - hitReward(taken)) * e->hitRewardCoef;
                if (otherDrone->stepInfo.brokeShield[drone->idx]) {
                    reward -= e->shieldBreakReward;
                }
            } else {
                reward += hitReward(dealt) * e->hitRewardCoef;
            }
            if (drone->stepInfo.brokeShield[i]) {
                reward += e->shieldBreakReward;
            }
        }

        if (droneDiedThisStep(otherDrone)) {
            if (!onTeam) {
                reward += e->lifeReward;
                if (drone->killed[i]) {
                    reward += e->killReward;
                }
            } else {
                reward += e->teammateDeathPunishment;
                if (drone->killed[i]) {
                    reward += e->teammateKillPunishment;
                }
            }
        }
    }

    return reward;
}

static const float REWARD_EPS = 1.0e-6f;

void computeRewards(iwEnv *e, const bool roundOver, const int8_t winner, const int8_t winningTeam) {
    for (uint8_t i = 0; i < e->numDrones; i++) {
        float reward = 0.0f;
        droneEntity *drone = safe_array_get_at(e->drones, i);
        reward = computeReward(e, drone);
        if (roundOver && (winner == i || winningTeam == drone->team)) {
            reward += e->winReward;
        } else if (roundOver && (winner != -1 || winningTeam != -1)) {
            // the round has a winner and it isn't this drone
            reward -= e->winReward;
        } else if (roundOver && (winner == -1 && winningTeam == -1)) {
            // punish ties like loses to discourage passivity
            reward -= e->winReward;
        }

        if (i < e->numAgents) {
            agentRewards(e, i)[0] += reward;
        }
        e->stats[i].returns += reward;
    }
}

static inline bool isActionNoop(const b2Vec2 action) {
    return b2Length(action) < ACTION_NOOP_MAGNITUDE;
}

// if manualActions is NULL the actions are drawn from the environment
agentActions _computeActions(iwEnv *e, droneEntity *drone, const agentActions *manualActions) {
    agentActions actions = {0};

    const float *envActions = agentActions(e, drone->idx);
    if (manualActions == NULL) {
        if (e->continuousActions) {
            actions.move = (b2Vec2){.x = tanhf(envActions[0]), .y = tanhf(envActions[1])};
            actions.aim = (b2Vec2){.x = tanhf(envActions[2]), .y = tanhf(envActions[3])};
            actions.trigger = envActions[4] > 0.0f;
            actions.utility = envActions[5] > 0.0f;
            actions.brake = envActions[6] > 0.0f;
            actions.chargingBurst = envActions[7] > 0.0f;
        } else {
            uint8_t move = envActions[0];
            // 0 is no-op for move
            ASSERT(move <= 8);
            if (move != 0) {
                move--;
                actions.move.x = discMoveToContMoveMap[0][move];
                actions.move.y = discMoveToContMoveMap[1][move];
            }
            const uint8_t aimCoarse = envActions[1];
            const uint8_t aimFine = envActions[2];
            ASSERT(aimCoarse < AIM_COARSE_SECTORS);
            ASSERT(aimFine < AIM_FINE_STEPS);
            // the fine offset is centered on the sector, so the middle
            // fine step aims exactly at the sector's center
            const int8_t fineOffset = (int8_t)aimFine - AIM_FINE_STEPS / 2;
            const float angle = (aimCoarse * AIM_FINE_STEPS + fineOffset) * AIM_FINE_STEP_ANGLE;
            actions.aim.x = cosf(angle);
            actions.aim.y = sinf(angle);

            actions.trigger = envActions[3] > 0.0f;
            actions.utility = envActions[4] > 0.0f;
            actions.brake = envActions[5] > 0.0f;
            actions.chargingBurst = envActions[6] > 0.0f;
        }
    } else {
        actions.move = manualActions->move;
        actions.aim = manualActions->aim;
        actions.trigger = manualActions->trigger;
        actions.brake = manualActions->brake;
        actions.chargingBurst = manualActions->chargingBurst;
        actions.utility = manualActions->utility;
    }

    // cap movement magnitude to 1.0
    if (b2Length(actions.move) > 1.0f) {
        actions.move = b2Normalize(actions.move);
    } else if (isActionNoop(actions.move)) {
        actions.move = b2Vec2_zero;
    }

    if (isActionNoop(actions.aim)) {
        actions.aim = b2Vec2_zero;
    } else {
        actions.aim = b2Normalize(actions.aim);
    }

    return actions;
}

agentActions computeActions(iwEnv *e, droneEntity *drone, const agentActions *manualActions) {
    const agentActions actions = _computeActions(e, drone, manualActions);
    drone->lastMove = actions.move;
    if (!b2Equal(actions.aim, b2Vec2_zero)) {
        drone->lastAim = actions.aim;
    }
    return actions;
}

bool droneControlledByHuman(const iwEnv *e, uint8_t i) {
    if (!e->humanInput) {
        return false;
    }
    return (e->connectedControllers > 1 && i >= e->humanDroneInput) || (e->connectedControllers <= 1 && i == e->humanDroneInput);
}

void addLog(iwEnv *e, Log *log) {
    e->log.length += log->length;
    e->log.ties += log->ties;
    e->log.botCLNoise += log->botCLNoise;
    e->log.easyEpProb += log->easyEpProb;
    e->log.easyEp += log->easyEp;

    for (uint8_t j = 0; j < e->numDrones; j++) {
        e->log.stats[j].returns += log->stats[j].returns;
        e->log.stats[j].score += log->stats[j].score;

        e->log.stats[j].distanceTraveled += log->stats[j].distanceTraveled;
        e->log.stats[j].absDistanceTraveled += log->stats[j].absDistanceTraveled;
        e->log.stats[j].brakeTime += log->stats[j].brakeTime;
        e->log.stats[j].totalBursts += log->stats[j].totalBursts;
        e->log.stats[j].burstsHit += log->stats[j].burstsHit;
        e->log.stats[j].energyEmptied += log->stats[j].energyEmptied;
        e->log.stats[j].shieldsBroken += log->stats[j].shieldsBroken;
        e->log.stats[j].ownShieldBroken += log->stats[j].ownShieldBroken;
        e->log.stats[j].selfKills += log->stats[j].selfKills;
        e->log.stats[j].kills += log->stats[j].kills;
        e->log.stats[j].unknownKills += log->stats[j].unknownKills;
        e->log.stats[j].cratesPickedUp += log->stats[j].cratesPickedUp;
        e->log.stats[j].cratesRefilled += log->stats[j].cratesRefilled;

        for (uint8_t k = 0; k < NUM_WEAPONS; k++) {
            e->log.stats[j].shotsFired[k] += log->stats[j].shotsFired[k];
            e->log.stats[j].shotsHit[k] += log->stats[j].shotsHit[k];
            e->log.stats[j].shotsTaken[k] += log->stats[j].shotsTaken[k];
            e->log.stats[j].ownShotsTaken[k] += log->stats[j].ownShotsTaken[k];
            e->log.stats[j].weaponsPickedUp[k] += log->stats[j].weaponsPickedUp[k];
            e->log.stats[j].shotDistances[k] += log->stats[j].shotDistances[k];
        }
        for (uint8_t k = 0; k < NUM_UTILITIES; k++) {
            e->log.stats[j].utilitiesPickedUp[k] += log->stats[j].utilitiesPickedUp[k];
        }

        e->log.stats[j].totalShotsFired += log->stats[j].totalShotsFired;
        e->log.stats[j].totalShotsHit += log->stats[j].totalShotsHit;
        e->log.stats[j].totalShotsTaken += log->stats[j].totalShotsTaken;
        e->log.stats[j].totalOwnShotsTaken += log->stats[j].totalOwnShotsTaken;
        e->log.stats[j].totalWeaponsPickedUp += log->stats[j].totalWeaponsPickedUp;
        e->log.stats[j].totalShotDistances += log->stats[j].totalShotDistances;
        e->log.stats[j].shotsAimed10 += log->stats[j].shotsAimed10;
        e->log.stats[j].shotsAimed30 += log->stats[j].shotsAimed30;
        e->log.stats[j].totalAimError += log->stats[j].totalAimError;
        e->log.stats[j].shotsAimedAway += log->stats[j].shotsAimedAway;
    }

    e->log.n += 1.0f;
}

void updateTrailPoints(trailPoints *tp, const uint8_t maxLen, const b2Vec2 pos) {
    const Vector2 v = (Vector2){.x = pos.x, .y = pos.y};
    if (tp->length < maxLen) {
        tp->points[tp->length++] = v;
        return;
    }

    for (uint8_t i = 0; i < maxLen - 1; i++) {
        tp->points[i] = tp->points[i + 1];
    }
    tp->points[maxLen - 1] = v;
}

void updateExplosions(iwEnv *e) {
    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->explosions);
    explosionInfo *explosion;

    while (cc_array_iter_next(&iter, (void **)&explosion) != CC_ITER_END) {
        if (explosion->renderSteps == UINT16_MAX) {
            const float duration = explosion->isBlackHole ? BLACK_HOLE_DISTORTION_FADE_TIME : explosionFadeTime(&explosion->def);
            explosion->maxRenderSteps = (uint16_t)min((float)(UINT16_MAX - 1), max(1.0f, ceilf(duration * e->frameRate)));
            explosion->renderSteps = explosion->maxRenderSteps;
        } else if (explosion->renderSteps == 0) {
            fastFree(explosion);
            cc_array_iter_remove(&iter, NULL);
        } else {
            explosion->renderSteps = max(explosion->renderSteps - 1, 0);
        }
    }
}

void updateBoostSparks(iwEnv *e) {
    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->boostSparks);
    boostSparksInfo *sparks;

    while (cc_array_iter_next(&iter, (void **)&sparks) != CC_ITER_END) {
        if (sparks->renderSteps == UINT16_MAX) {
            sparks->maxRenderSteps = (uint16_t)max(1.0f, ceilf(BOOST_SPARKS_FADE_TIME * e->frameRate));
            sparks->renderSteps = sparks->maxRenderSteps;
        } else if (sparks->renderSteps == 0) {
            fastFree(sparks);
            cc_array_iter_remove(&iter, NULL);
        } else {
            sparks->renderSteps--;
        }
    }
}

void updateDronePieces(iwEnv *e) {
    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->dronePieces);
    dronePieceEntity *piece;

    while (cc_array_iter_next(&iter, (void **)&piece) != CC_ITER_END) {
        if (piece->lifetime == UINT16_MAX) {
            piece->lifetime = e->client->maxDronePieceLifetime;
        } else if (piece->lifetime == 0) {
            destroyDronePiece(e, piece);
            cc_array_iter_remove_fast(&iter, NULL);
        } else {
            piece->lifetime--;
        }
    }
}

void updateDroneBrakeTrail(const iwEnv *e, droneEntity *drone) {
    // update lifetimes and prune expired points
    CC_ArrayIter iter;
    cc_array_iter_init(&iter, drone->brakeTrailPoints);
    brakeTrailPoint *pt;
    while (cc_array_iter_next(&iter, (void **)&pt) != CC_ITER_END) {
        if (pt->lifetime == UINT16_MAX) {
            pt->lifetime = e->client->maxBrakeTrailLifetime;
        } else if (pt->lifetime == 0) {
            fastFree(pt);
            cc_array_iter_remove(&iter, NULL);
            continue;
        } else {
            pt->lifetime--;
        }
    }
}

void updateDroneRespawnGuide(iwEnv *e, droneEntity *drone) {
    if (drone->respawnGuideLifetime == 0) {
        return;
    } else if (drone->respawnGuideLifetime == UINT16_MAX) {
        drone->respawnGuideLifetime = e->client->maxDroneRespawnGuideLifetime;
    } else {
        drone->respawnGuideLifetime -= 1;
    }
}

void updateDroneEnergyShake(iwEnv *e, droneEntity *drone) {
    if (drone->energyShakeRenderSteps == UINT16_MAX) {
        drone->energyShakeRenderSteps = energyShakeMaxSteps(e);
    } else if (drone->energyShakeRenderSteps != 0) {
        drone->energyShakeRenderSteps--;
    }
}

void updateDroneUtilityShake(iwEnv *e, droneEntity *drone) {
    if (drone->utilityShakeRenderSteps == UINT16_MAX) {
        drone->utilityShakeRenderSteps = utilityShakeMaxSteps(e);
    } else if (drone->utilityShakeRenderSteps != 0) {
        drone->utilityShakeRenderSteps--;
    }
}

void updatePortalTransitEffects(iwEnv *e) {
    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        for (uint8_t j = 0; j < drone->openPortals; j++) {
            portalEntity *portal = drone->portals[j];
            if (portal->transitRenderSteps == UINT16_MAX) {
                portal->transitRenderSteps = portalTransitMaxSteps(e);
            } else if (portal->transitRenderSteps != 0) {
                portal->transitRenderSteps--;
            }
        }
    }
}

void updateDroneSwapEffects(iwEnv *e) {
    for (uint8_t i = 0; i < MAX_DRONE_SWAP_EFFECTS; i++) {
        droneSwapEffect *effect = &e->droneSwapEffects[i];
        if (effect->renderSteps == UINT16_MAX) {
            effect->renderSteps = portalTransitMaxSteps(e);
        } else if (effect->renderSteps != 0) {
            effect->renderSteps--;
        }
    }
}

void updateVisuals(iwEnv *e) {
    updateExplosions(e);
    updateBoostSparks(e);
    updateDronePieces(e);
    updatePortalTransitEffects(e);
    updateDroneSwapEffects(e);

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        updateDroneBrakeTrail(e, drone);
        updateDroneRespawnGuide(e, drone);
        updateDroneUtilityShake(e, drone);
        updateDroneEnergyShake(e, drone);
    }
}

bool stepPhysicsFrame(iwEnv *e, const agentActions stepActions[], int8_t *winner, int8_t *winningTeam) {
    int8_t lastAlive = -1;
    int8_t lastAliveTeam = -1;

    for (uint8_t i = 0; i < e->numDrones; i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        agentActions actions = stepActions[i];

        if (actions.trigger) {
            droneShoot(e, drone, actions.aim, true);
        } else if (drone->triggerHeld && drone->weapon.charge > 0.0f) {
            droneShoot(e, drone, actions.aim, false);
        }
        drone->triggerHeld = actions.trigger;

        droneUtility(e, drone, actions.aim, actions.utility);

        const bool burstPressed = actions.chargingBurst && !drone->burstHeld;
        drone->burstHeld = actions.chargingBurst;
        if (actions.chargingBurst && !e->easyEp) {
            droneChargeBurst(e, drone, burstPressed);
        } else if (drone->chargingBurst) {
            droneBurst(e, drone);
        }
        if (!b2Equal(actions.move, b2Vec2_zero)) {
            droneMove(e, drone, actions.move);
        }
        droneBrake(e, drone, actions.brake);

        // update shield velocity if its active
        if (drone->shield != NULL) {
            b2Body_SetLinearVelocity(drone->shield->bodyID, b2Body_GetLinearVelocity(drone->bodyID));
        }
    }

    b2World_Step(e->worldID, e->deltaTime, e->box2dSubSteps);

    // update dynamic body positions and velocities
    handleBodyMoveEvents(e);

    // handle collisions
    handleContactEvents(e);
    handleSensorEvents(e);
    weaponPickupsOverlapStep(e);

    // handle sudden death
    e->stepsLeft = max(e->stepsLeft - 1, 0);
    if ((!e->isTraining || e->numDrones == e->numAgents) && e->stepsLeft == 0) {
        e->suddenDeathSteps = max(e->suddenDeathSteps - 1, 0);
        if (e->suddenDeathSteps == 0) {
            DEBUG_LOG("placing sudden death walls");
            handleSuddenDeath(e);
            e->suddenDeathSteps = e->totalSuddenDeathSteps;
        }
    }

    projectilesStep(e);

    bool allAliveOnSameTeam = false;
    bool roundOver = false;
    uint8_t deadDrones = 0;
    for (uint8_t i = 0; i < e->numDrones; i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->livesLeft != 0) {
            if (!droneStep(e, drone)) {
                // couldn't find a respawn position, end the round
                deadDrones++;
                roundOver = true;
            } else {
                lastAlive = i;

                if (e->teamsEnabled) {
                    if (lastAliveTeam == -1) {
                        lastAliveTeam = drone->team;
                        allAliveOnSameTeam = true;
                    } else if (drone->team != lastAliveTeam) {
                        allAliveOnSameTeam = false;
                    }
                }
            }
        } else {
            deadDrones++;
            if (i < e->numAgents) {
                if (drone->diedThisStep) {
                    agentTerminals(e, i)[0] = 1.0f;
                }
            }
        }
    }

    weaponPickupsStep(e);

    if (!roundOver) {
        roundOver = deadDrones >= e->numDrones - 1;
    }
    if (e->teamsEnabled && allAliveOnSameTeam) {
        roundOver = true;
        lastAlive = -1;
    }
    // if the enemy drone(s) are scripted don't enable sudden death
    // so that the agent has to work for victories
    if (e->isTraining && e->numDrones != e->numAgents && e->stepsLeft == 0) {
        roundOver = true;
        lastAliveTeam = -1;
    }
    if (roundOver && deadDrones < e->numDrones - 1) {
        lastAlive = -1;
    }

    if (e->client != NULL) {
        updateVisuals(e);
    }

    *winner = lastAlive;
    *winningTeam = lastAliveTeam;

    return roundOver;
}

void stepPhysicsFrameMinimal(iwEnv *e) {
    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead || drone->shield == NULL) {
            continue;
        }

        // update shield velocity if its active
        b2Body_SetLinearVelocity(drone->shield->bodyID, b2Body_GetLinearVelocity(drone->bodyID));
    };

    b2World_Step(e->worldID, e->deltaTime, e->box2dSubSteps);

    handleBodyMoveEvents(e);
    handleContactEvents(e);
    handleSensorEvents(e);
    weaponPickupsOverlapStep(e);

    projectilesStep(e);

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        droneStep(e, drone);
    }

    updateVisuals(e);
}

void endEpisode(iwEnv *e, const int8_t lastAlive, const int8_t lastAliveTeam) {
    if (e->numDrones != e->numAgents && e->stepsLeft == 0) {
        DEBUG_LOG("truncating episode");
    } else {
        DEBUG_LOG("terminating episode");
    }

    for (uint8_t i = 0; i < e->numAgents; i++) {
        agentTerminals(e, i)[0] = 1.0f;
    }
    if (e->tag > 0) {
        e->boundary_reached = 1;
    }

    Log log = {0};
    log.length = e->episodeLength;
    if (lastAlive != -1) {
        e->stats[lastAlive].score = 1.0f;
    } else if (!e->teamsEnabled || (e->teamsEnabled && lastAliveTeam == -1)) {
        log.ties = 1.0f;
        for (uint8_t i = 0; i < e->numDrones; i++) {
            e->stats[i].score = 0.5f;
        }
    }
    log.botCLNoise = e->botCLNoise;
    log.easyEpProb = e->easyEpProb;
    log.easyEp = e->easyEp;

    const float shotsFired = e->stats[0].totalShotsFired;
    if (e->easyEp && shotsFired >= EASY_EP_MIN_SHOTS && e->stats[0].totalShotsHit / shotsFired >= e->easyEpHitRate) {
        e->easyEpProb -= e->easyEpDecay;
        e->easyEpProb = clamp(e->easyEpProb);
    }

    // TODO: handle multiple agents/teams correctly
    // kills in easy episodes are against a sitting duck, so don't count them
    if (e->numAgents != e->numDrones && !e->easyEp && e->botCLNoise > 0.0f && e->stats[0].kills > 0.0f) {
        e->botCLNoise -= e->botCLDecay;
        e->botCLNoise = clamp(e->botCLNoise);
    }

    for (uint8_t i = 0; i < e->numDrones; i++) {
        const droneEntity *drone = safe_array_get_at(e->drones, i);
        if (e->teamsEnabled && drone->team == lastAliveTeam) {
            e->stats[i].score = 1.0f;
        }
        // set absolute distance traveled of agent drones
        e->stats[i].absDistanceTraveled = b2Distance(drone->initalPos, drone->pos);
    }

    memcpy(log.stats, e->stats, sizeof(e->stats));
    addLog(e, &log);
    memcpy(e->lastAnalysisStats, e->analysisStats, sizeof(e->analysisStats));

    if (e->client != NULL) {
        e->roundState = ROUND_STATE_ENDING;
        e->tick_frames_left = END_WAIT_TIME * e->frameRate;

        e->winner = lastAlive;
        e->winningTeam = lastAliveTeam;
    }

    e->needsReset = true;
}

void puf_step(iwEnv *e) {
    // handle the start and end pause when rendering
    if (e->client != NULL) {
        if (e->roundState == ROUND_STATE_ENDING) {
            if (e->tick_frames_left > 0) {
                stepPhysicsFrameMinimal(e);
                e->tick_frames_left--;
            }

            if (e->tick_frames_left == 0) {
                DEBUG_LOG("Resetting environment");
                puf_reset(e);
            }
            return;
        }

        if (e->needsReset) {
            DEBUG_LOG("Resetting environment");
            puf_reset(e);
            return;
        }

        if (e->roundState == ROUND_STATE_STARTING) {
            if (e->tick_frames_left > 0) {
                e->tick_frames_left--;
                return;
            }

            e->roundState = ROUND_STATE_PLAYING;
        }
    }

    for (uint8_t i = 0; i < e->numAgents; i++) {
        agentTerminals(e, i)[0] = 0.0f;
    }

    if (e->client == NULL || e->tick_frames_left <= 0) {
        e->tick_frames_left = e->frameRate / TRAINING_ACTIONS_PER_SECOND;

#ifdef PUF_DEBUG
        for (uint8_t i = 0; i < cc_array_size(e->debugPoints); i++) {
            debugPoint *point = safe_array_get_at(e->debugPoints, i);
            fastFree(point);
        }
        cc_array_remove_all(e->debugPoints);
#endif

        // preprocess agent actions for the next frameSkip steps
        for (uint8_t i = 0; i < e->numDrones; i++) {
            droneEntity *drone = safe_array_get_at(e->drones, i);

            memset(&drone->stepInfo, 0x0, sizeof(droneStepInfo));
            if (drone->dead) {
                drone->diedThisStep = false;
            }
            memset(&drone->killed, 0x0, sizeof(drone->killed));

            if (drone->dead || droneControlledByHuman(e, i)) {
                continue;
            }

            if (i < e->numAgents) {
                e->cachedActions[i] = computeActions(e, drone, NULL);
            } else {
                const agentActions scriptedActions = scriptedAgentActions(e, drone);
                e->cachedActions[i] = computeActions(e, drone, &scriptedActions);
            }
        }

        // reset reward buffer
        for (uint8_t i = 0; i < e->numAgents; i++) {
            agentRewards(e, i)[0] = 0.0f;
        }
    }

    bool roundOver = false;
    int8_t lastAlive = -1;
    int8_t lastAliveTeam = -1;
    for (int i = 0; i < e->frameSkip; i++) {
        e->episodeLength++;
        roundOver = stepPhysicsFrame(e, e->cachedActions, &lastAlive, &lastAliveTeam);
        e->tick_frames_left--;

        if (e->client == NULL && e->episodeLength == TRAINING_MAX_STEPS && !roundOver) {
            roundOver = true;
            lastAlive = -1;
            lastAliveTeam = -1;
        }
        if (roundOver || e->needsReset) {
            break;
        }
    }

    // avoid computing and logging rewards every frame for interactive evals
    if (e->client == NULL || e->tick_frames_left == 0 || roundOver) {
        computeRewards(e, roundOver, lastAlive, lastAliveTeam);
    }

    if (roundOver) {
        endEpisode(e, lastAlive, lastAliveTeam);
    }

    if (e->client == NULL && e->needsReset) {
        puf_reset(e);
    } else if (e->client == NULL || e->tick_frames_left == 0) {
        computeObs(e);
    }

#ifdef PUF_DEBUG
    bool gotReward = false;
    for (uint8_t i = 0; i < e->numAgents; i++) {
        if (agentRewards(e, i)[0] > REWARD_EPS || agentRewards(e, i)[0] < -REWARD_EPS) {
            gotReward = true;
            break;
        }
    }
    if (gotReward) {
        DEBUG_RAW_LOG("!!! rewards: [");
        for (uint8_t i = 0; i < e->numAgents; i++) {
            const float reward = agentRewards(e, i)[0];
            DEBUG_RAW_LOGF("%f", reward);
            if (i < e->numAgents - 1) {
                DEBUG_RAW_LOG(", ");
            }
        }
        DEBUG_RAW_LOGF("] step %d\n", e->totalSteps - e->stepsLeft);
    }
#endif
}

#endif
