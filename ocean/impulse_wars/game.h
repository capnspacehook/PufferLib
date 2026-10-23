#ifndef IMPULSE_WARS_GAME_H
#define IMPULSE_WARS_GAME_H

#include "helpers.h"
#include "settings.h"
#include "types.h"

// these functions call each other so need to be forward declared
void destroyProjectile(iwEnv *e, projectileEntity *projectile, const bool processExplosions, const bool full);
void createExplosion(iwEnv *e, droneEntity *drone, const projectileEntity *projectile, const b2ExplosionDef *def);

void updateTrailPoints(trailPoints *tp, const uint8_t maxLen, const b2Vec2 pos);

entity *createEntity(iwEnv *e, enum entityType type, void *entityData) {
    int32_t id = b2AllocId(&e->idPool);
    entity *ent = NULL;
    if (id == (int64_t)cc_array_size(e->entities)) {
        ent = fastCalloc(1, sizeof(entity));
        ent->id = fastCalloc(1, sizeof(entityID));
        cc_array_add(e->entities, ent);
    } else {
        ent = safe_array_get_at(e->entities, id);
    }

    ent->generation += 1;
    ent->type = type;
    ent->entity = entityData;
    ent->id->id = id + 1,
    ent->id->generation = ent->generation;
    memset(ent->lastBurstHit, 0x0, sizeof(ent->lastBurstHit));

    return ent;
}

void destroyEntity(iwEnv *e, entity *ent) {
    b2FreeId(&e->idPool, ent->id->id - 1);
    ent->id->id = 0;
}

// will return a pointer to an entity or NULL if the given entity ID is
// invalid or orphaned
entity *getEntityByID(const iwEnv *e, const entityID *id) {
    if (id->id < 1 || (int64_t)cc_array_size(e->entities) < id->id) {
        // invalid index
        return NULL;
    }
    entity *ent = safe_array_get_at(e->entities, id->id - 1);
    if (ent->id->id == 0 || ent->generation != id->generation) {
        // orphaned entity
        return NULL;
    }
    return ent;
}

static inline bool entityTypeIsWall(const enum entityType type) {
    // walls are the first 3 entity types
    return type <= DEATH_WALL_ENTITY;
}

static inline int16_t cellIndex(const iwEnv *e, const int8_t col, const int8_t row) {
    return col + (row * e->map->columns);
}

// discretizes an entity's position into a cell index; -1 is returned if
// the position is out of bounds of the map
static inline int16_t entityPosToCellIdx(const iwEnv *e, const b2Vec2 pos) {
    const float cellX = pos.x + (((float)e->map->columns * WALL_THICKNESS) / 2.0f);
    const float cellY = pos.y + (((float)e->map->rows * WALL_THICKNESS) / 2.0f);
    const float cellCol = floorf(cellX / WALL_THICKNESS);
    const float cellRow = floorf(cellY / WALL_THICKNESS);

    // set the cell to -1 if it's out of bounds
    if (cellCol < 0.0f || cellCol >= e->map->columns || cellRow < 0.0f || cellRow >= e->map->rows) {
        return -1;
    }
    const int16_t cellIdx = cellIndex(e, (int8_t)cellCol, (int8_t)cellRow);
    if (cellIdx < 0 || (uint16_t)cellIdx >= e->numCells) {
        DEBUG_LOGF("invalid cell index: %d from position: (%f, %f)", cellIdx, pos.x, pos.y);
        return -1;
    }

    return cellIdx;
}

typedef struct overlapAABBCtx {
    bool overlaps;
} overlapAABBCtx;

bool overlapAABBCallback(b2ShapeId shapeID, void *context) {
    if (!b2Shape_IsValid(shapeID)) {
        return true;
    }

    overlapAABBCtx *ctx = context;
    ctx->overlaps = true;
    return false;
}

// returns true if the given position overlaps with shapes in a bounding
// box with a height and width of distance
bool isOverlappingAABB(const iwEnv *e, const b2Vec2 pos, const float distance, const b2QueryFilter filter) {
    b2AABB bounds = {
        .lowerBound = {.x = pos.x - distance, .y = pos.y - distance},
        .upperBound = {.x = pos.x + distance, .y = pos.y + distance},
    };
    overlapAABBCtx ctx = {.overlaps = false};

    b2World_OverlapAABB(e->worldID, bounds, filter, overlapAABBCallback, &ctx);
    return ctx.overlaps;
}

b2ShapeProxy makeDistanceProxy(const entity *ent, bool *isCircle) {
    b2ShapeProxy proxy = {0};
    float extent = 0.0f;
    wallEntity *wall = NULL;

    switch (ent->type) {
    case DRONE_ENTITY:
        *isCircle = true;
        proxy.count = 1;
        proxy.radius = DRONE_RADIUS;
        break;
    case SHIELD_ENTITY:
        *isCircle = true;
        proxy.count = 1;
        proxy.radius = DRONE_SHIELD_RADIUS;
        break;
    case PROJECTILE_ENTITY:
        *isCircle = true;
        const projectileEntity *proj = ent->entity;
        proxy.count = 1;
        proxy.radius = proj->weaponInfo->radius;
        break;
    case WEAPON_PICKUP_ENTITY:
        extent = PICKUP_THICKNESS / 2.0f;

        proxy.count = 4;
        proxy.points[0] = (b2Vec2){.x = -extent, .y = -extent};
        proxy.points[1] = (b2Vec2){.x = -extent, .y = +extent};
        proxy.points[2] = (b2Vec2){.x = +extent, .y = -extent};
        proxy.points[3] = (b2Vec2){.x = +extent, .y = +extent};
        break;
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        extent = WALL_THICKNESS;
        wall = ent->entity;
        if (wall->isFloating) {
            extent = FLOATING_WALL_THICKNESS;
        }
        extent /= 2.0f;

        proxy.count = 4;
        proxy.points[0] = (b2Vec2){.x = -extent, .y = -extent};
        proxy.points[1] = (b2Vec2){.x = -extent, .y = +extent};
        proxy.points[2] = (b2Vec2){.x = +extent, .y = -extent};
        proxy.points[3] = (b2Vec2){.x = +extent, .y = +extent};
        break;
    default:
        ERRORF("unknown entity type for shape distance: %d", ent->type);
    }

    return proxy;
}

b2Transform entityTransform(const entity *ent) {
    b2Transform transform;
    wallEntity *wall;
    projectileEntity *proj;
    droneEntity *drone;
    weaponPickupEntity *pickup;

    switch (ent->type) {
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        wall = ent->entity;
        transform.p = wall->pos;
        transform.q = wall->rot;
        return transform;
    case PROJECTILE_ENTITY:
        proj = ent->entity;
        transform.p = proj->pos;
        transform.q = b2Rot_identity;
        return transform;
    case DRONE_ENTITY:
        drone = ent->entity;
        transform.p = drone->pos;
        transform.q = b2Rot_identity;
        return transform;
    case WEAPON_PICKUP_ENTITY:
        pickup = ent->entity;
        transform.p = pickup->pos;
        transform.q = b2Rot_identity;
        return transform;
    default:
        ERRORF("unknown entity type: %d", ent->type);
    }
}

// returns the closest points between two entities
b2DistanceOutput closestPoint(const entity *srcEnt, const entity *dstEnt) {
    bool isCircle = false;
    b2DistanceInput input;
    input.proxyA = makeDistanceProxy(srcEnt, &isCircle);
    input.proxyB = makeDistanceProxy(dstEnt, &isCircle);
    input.transformA = entityTransform(srcEnt);
    input.transformB = entityTransform(dstEnt);
    if (input.proxyA.radius != 0.0f && input.proxyB.radius != 0.0f) {
        b2DistanceOutput output = {0};
        output.distance = b2Distance(input.transformB.p, input.transformA.p) - input.proxyA.radius - input.proxyB.radius;
        output.normal = b2Normalize(b2Sub(input.transformB.p, input.transformA.p));
        output.pointA = b2MulAdd(input.transformA.p, input.proxyA.radius, output.normal);
        output.pointB = b2MulAdd(input.transformB.p, input.proxyB.radius, output.normal);
        return output;
    }

    input.useRadii = isCircle;

    b2SimplexCache cache = {0};
    return b2ShapeDistance(&input, &cache, NULL, 0);
}

typedef struct behindWallContext {
    const entity *dstEnt;
    const enum entityType *targetType;
    bool hit;
} behindWallContext;

float posBehindWallCallback(b2ShapeId shapeID, b2Vec2 point, b2Vec2 normal, float fraction, void *context) {
    // these are unused but required by the b2CastResultFcn callback prototype
    MAYBE_UNUSED(point);
    MAYBE_UNUSED(normal);
    MAYBE_UNUSED(fraction);

    behindWallContext *ctx = context;
    const entity *ent = b2Shape_GetUserData(shapeID);
    if (ent == ctx->dstEnt || (ctx->targetType != NULL && ent->type == *ctx->targetType)) {
        return -1;
    }
    ctx->hit = true;
    return 0;
}

// returns true if there are shapes that match filter between startPos and endPos
bool posBehindWall(const iwEnv *e, const b2Vec2 srcPos, const b2Vec2 dstPos, const entity *dstEnt, const b2QueryFilter filter, const enum entityType *targetType) {
    const float rayDistance = b2Distance(srcPos, dstPos);
    // if the two points are extremely close we can safely assume the
    // entity isn't behind a wall
    if (rayDistance <= 1.0f) {
        return false;
    }

    const b2Vec2 translation = b2Sub(dstPos, srcPos);
    behindWallContext ctx = {
        .dstEnt = dstEnt,
        .targetType = targetType,
        .hit = false,
    };
    b2World_CastRay(e->worldID, srcPos, translation, filter, posBehindWallCallback, &ctx);
    return ctx.hit;
}

static inline void projectileAddDroneBehindWall(projectileEntity *projectile, const uint8_t droneIdx) {
    for (uint8_t i = 0; i < projectile->numDronesBehindWalls; i++) {
        if (projectile->dronesBehindWalls[i] == droneIdx) {
            return;
        }
    }
    ASSERT(projectile->numDronesBehindWalls < _MAX_DRONES);
    projectile->dronesBehindWalls[projectile->numDronesBehindWalls++] = droneIdx;
}

static inline void projectileRemoveDroneBehindWall(projectileEntity *projectile, const uint8_t droneIdx) {
    for (uint8_t i = 0; i < projectile->numDronesBehindWalls; i++) {
        if (projectile->dronesBehindWalls[i] == droneIdx) {
            projectile->dronesBehindWalls[i] = projectile->dronesBehindWalls[--projectile->numDronesBehindWalls];
            return;
        }
    }
}

typedef struct overlapCircleLineOfSightCtx {
    const iwEnv *e;
    const entity *ent;
    const enum entityType *targetType;
    const b2QueryFilter filter;
    bool overlaps;
} overlapCircleLineOfSightCtx;

bool isOverlappingCircleInLineOfSightCallback(b2ShapeId shapeID, void *context) {
    if (!b2Shape_IsValid(shapeID)) {
        return true;
    }

    overlapCircleLineOfSightCtx *ctx = context;
    const entity *overlappingEnt = b2Shape_GetUserData(shapeID);
    if (ctx->targetType != NULL && overlappingEnt->type != *ctx->targetType) {
        return true;
    }

    const b2DistanceOutput output = closestPoint(ctx->ent, overlappingEnt);
    const bool behind = posBehindWall(ctx->e, output.pointA, output.pointB, ctx->ent, ctx->filter, ctx->targetType);
    if (!behind) {
        ctx->overlaps = true;
    } else if (ctx->ent->type == PROJECTILE_ENTITY) {
        projectileEntity *proj = ctx->ent->entity;
        const droneEntity *drone = overlappingEnt->entity;
        projectileAddDroneBehindWall(proj, drone->idx);
    }
    return behind;
}

bool isOverlappingCircleInLineOfSight(const iwEnv *e, const entity *ent, const b2Vec2 startPos, const float radius, const b2QueryFilter filter, const enum entityType *targetType) {
    const b2ShapeProxy cirProxy = b2MakeProxy(&startPos, 1, radius);
    overlapCircleLineOfSightCtx ctx = {
        .e = e,
        .ent = ent,
        .targetType = targetType,
        .filter = (b2QueryFilter){
            .categoryBits = filter.categoryBits,
            .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE,
        },
        .overlaps = false,
    };
    b2World_OverlapShape(e->worldID, &cirProxy, filter, isOverlappingCircleInLineOfSightCallback, &ctx);
    return ctx.overlaps;
}

uint8_t cellOffsets[8][2] = {
    {-1, 0},  // left
    {1, 0},   // right
    {0, -1},  // up
    {0, 1},   // down
    {-1, -1}, // top-left
    {1, -1},  // top-right
    {-1, 1},  // bottom-left
    {1, 1},   // bottom-right
};

// returns true and sets emptyPos to the position of an empty cell
// that is an appropriate distance away from other entities if one exists;
// if quad is set to -1 a random valid position from anywhere on the map
// will be returned, otherwise a position within the specified quadrant
// will be returned
bool findOpenPos(iwEnv *e, const enum shapeCategory shapeType, b2Vec2 *emptyPos, int8_t quad) {
    uint8_t checkedCells[BITNSLOTS(MAX_CELLS)] = {0};
    uint16_t attempts = 0;
    bool laxDroneDistanceChecks = false;
    float minSpawnDistance = MIN_SPAWN_DISTANCE;

    size_t nCells = e->numCells - 1;
    if (quad != -1) {
        nCells = e->map->quadCells;
    }

    while (true) {
        if (attempts == nCells) {
            // if we're trying to find a position for a drone and sudden
            // death walls have been placed, try again this time ignoring
            // distance checks; the drone must be spawned next
            // to a death wall in this case
            if (shapeType == DRONE_SHAPE && e->suddenDeathWallsPlaced && !laxDroneDistanceChecks) {
                attempts = 0;
                memset(checkedCells, 0x0, BITNSLOTS(MAX_CELLS));
                laxDroneDistanceChecks = true;
                minSpawnDistance = MIN_SD_SPAWN_DISTANCE;
                continue;
            }
            return false;
        }

        uint16_t cellIdx;
        if (quad == -1) {
            cellIdx = randInt(&e->rng, 0, nCells);
        } else {
            const float minX = e->map->spawnQuads[quad].min.x;
            const float minY = e->map->spawnQuads[quad].min.y;
            const float maxX = e->map->spawnQuads[quad].max.x;
            const float maxY = e->map->spawnQuads[quad].max.y;

            b2Vec2 randPos = {.x = randFloat(&e->rng, minX, maxX), .y = randFloat(&e->rng, minY, maxY)};
            cellIdx = entityPosToCellIdx(e, randPos);
        }
        if (bitTest(checkedCells, cellIdx)) {
            continue;
        }
        bitSet(checkedCells, cellIdx);
        attempts++;

        const mapCell *cell = (&e->cells[cellIdx]);
        if (cell->ent != NULL) {
            continue;
        }

        bool tooClose = false;
        switch (shapeType) {
        case WEAPON_PICKUP_SHAPE:
            // ensure pickups don't spawn too close to other pickups
            for (uint8_t i = 0; i < cc_array_size(e->pickups); i++) {
                const weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
                if (b2DistanceSquared(cell->pos, pickup->pos) < PICKUP_SPAWN_DISTANCE_SQUARED) {
                    tooClose = true;
                    break;
                }
            }
            if (tooClose) {
                continue;
            }
            break;
        case DRONE_SHAPE:
            if (e->suddenDeathWallsPlaced) {
                if (!laxDroneDistanceChecks) {
                    // if sudden death walls have been placed, ignore the
                    // spawn points as they may be covered by death walls;
                    // instead just try and find a cell that doesn't neighbor
                    // a death wall
                    const uint8_t cellCol = cellIdx % e->map->columns;
                    const uint8_t cellRow = cellIdx / e->map->columns;
                    bool deathWallNeighboring = false;
                    for (uint8_t i = 0; i < 8; i++) {
                        const int8_t col = cellCol + cellOffsets[i][0];
                        const int8_t row = cellRow + cellOffsets[i][1];
                        if (row < 0 || row >= e->map->rows || col < 0 || col >= e->map->columns) {
                            continue;
                        }
                        const int16_t testCellIdx = cellIndex(e, col, row);
                        const mapCell *testCell = (&e->cells[testCellIdx]);
                        if (testCell->ent != NULL && testCell->ent->type == DEATH_WALL_ENTITY) {
                            deathWallNeighboring = true;
                            break;
                        }
                    }
                    if (deathWallNeighboring) {
                        continue;
                    }
                }

                // ensure drones aren't spawning on top of each other
                for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
                    const droneEntity *drone = safe_array_get_at(e->drones, i);
                    if (drone->dead) {
                        continue;
                    }
                    if (b2DistanceSquared(cell->pos, drone->pos) == 0.0f) {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose) {
                    continue;
                }
            } else {
                if (!e->map->droneSpawns[cellIdx]) {
                    continue;
                }

                // ensure drones don't spawn too close to other drones
                for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
                    const droneEntity *drone = safe_array_get_at(e->drones, i);
                    if (drone->dead) {
                        continue;
                    }
                    if (b2DistanceSquared(cell->pos, drone->pos) < DRONE_DRONE_SPAWN_DISTANCE_SQUARED) {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose) {
                    continue;
                }
            }
            break;
        default:
            break;
        }

        uint64_t maskBits = FLOATING_WALL_SHAPE | WEAPON_PICKUP_SHAPE | DRONE_SHAPE;
        if (shapeType != FLOATING_WALL_SHAPE && !laxDroneDistanceChecks) {
            maskBits &= ~shapeType;
        }
        const b2QueryFilter filter = {
            .categoryBits = shapeType,
            .maskBits = maskBits,
        };
        if (!isOverlappingAABB(e, cell->pos, minSpawnDistance, filter)) {
            *emptyPos = cell->pos;
            return true;
        }
    }
}

entity *createWall(iwEnv *e, const b2Vec2 pos, const float width, const float height, int16_t cellIdx, const enum entityType type, const bool floating) {
    ASSERT(cellIdx != -1);
    ASSERT(entityTypeIsWall(type));

    b2BodyDef wallBodyDef = b2DefaultBodyDef();
    wallBodyDef.position = pos;
    if (floating) {
        wallBodyDef.type = b2_dynamicBody;
        wallBodyDef.linearDamping = FLOATING_WALL_DAMPING;
        wallBodyDef.angularDamping = FLOATING_WALL_DAMPING;
        wallBodyDef.isAwake = false;
        wallBodyDef.sleepThreshold = FLOATING_WALL_SLEEP_THRESHOLD;
    }
    b2BodyId wallBodyID = b2CreateBody(e->worldID, &wallBodyDef);

    b2Vec2 extent = {.x = width / 2.0f, .y = height / 2.0f};
    b2ShapeDef wallShapeDef = b2DefaultShapeDef();
    wallShapeDef.invokeContactCreation = false;
    wallShapeDef.density = WALL_DENSITY;
    wallShapeDef.material.restitution = STANDARD_WALL_RESTITUTION;
    wallShapeDef.material.friction = STANDARD_WALL_FRICTION;
    wallShapeDef.filter.categoryBits = WALL_SHAPE;
    wallShapeDef.filter.maskBits = FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE | SHIELD_SHAPE | DRONE_PIECE_SHAPE;
    if (floating) {
        wallShapeDef.filter.categoryBits = FLOATING_WALL_SHAPE;
        wallShapeDef.filter.maskBits |= WALL_SHAPE | WEAPON_PICKUP_SHAPE;
        wallShapeDef.enableSensorEvents = true;
    }

    if (type == BOUNCY_WALL_ENTITY) {
        wallShapeDef.material.restitution = BOUNCY_WALL_RESTITUTION;
        wallShapeDef.material.friction = 0.0f;
    } else if (type == DEATH_WALL_ENTITY) {
        wallShapeDef.enableContactEvents = true;
    }

    wallEntity *wall = fastCalloc(1, sizeof(wallEntity));
    wall->bodyID = wallBodyID;
    wall->pos = pos;
    wall->rot = b2Rot_identity;
    wall->velocity = b2Vec2_zero;
    wall->extent = extent;
    wall->mapCellIdx = cellIdx;
    wall->isFloating = floating;
    wall->type = type;
    wall->isSuddenDeath = e->suddenDeathWallsPlaced;

    entity *ent = createEntity(e, type, wall);
    wall->ent = ent;

    wallShapeDef.userData = ent;
    const b2Polygon wallPolygon = b2MakeBox(extent.x, extent.y);
    wall->shapeID = b2CreatePolygonShape(wallBodyID, &wallShapeDef, &wallPolygon);
    b2Body_SetUserData(wall->bodyID, ent);

    if (floating) {
        cc_array_add(e->floatingWalls, wall);
        create_array(&wall->physicsTracking, 16);
    } else {
        cc_array_add(e->walls, wall);
    }

    return ent;
}

void destroyWall(iwEnv *e, wallEntity *wall, const bool full) {
    destroyEntity(e, wall->ent);

    if (full) {
        mapCell *cell = (&e->cells[wall->mapCellIdx]);
        cell->ent = NULL;
    }

    if (wall->isFloating) {
        for (size_t i = 0; i < cc_array_size(wall->physicsTracking); i++) {
            physicsStepInfo *physicsStep = safe_array_get_at(wall->physicsTracking, i);
            fastFree(physicsStep);
        }
        cc_array_destroy(wall->physicsTracking);
    }

    b2DestroyBody(wall->bodyID);
    fastFree(wall);
}

int8_t mapQuad(const iwEnv *e, const b2Vec2 pos) {
    return (int8_t)(pos.x >= 0.0f) + 2 * (int8_t)(pos.y >= 0.0f);
}

// sets the spawn weight of each utility, discounted by how many of each are
// currently spawned or held the same way weapons are, and returns the total
static float utilitySpawnWeights(const iwEnv *e, const float *heldWeights, float *spawnWeights) {
    float totalWeight = 0.0f;
    for (uint8_t i = 1; i < NUM_UTILITIES; i++) {
        const float discount = (float)e->spawnedUtilityPickups[i] + (heldWeights[i] * PICKUP_HELD_WEAPON_WEIGHT_COEF) + 1.0f;
        spawnWeights[i - 1] = e->map->utilityWeights[i] / discount;
        totalWeight += spawnWeights[i - 1];
    }
    return totalWeight;
}

// how much each utility is held, for discounting its spawn weight; utilities
// with the max weight aren't discounted so featured ones stay common
static void heldUtilityWeights(const iwEnv *e, float *heldWeights) {
    for (uint8_t i = 0; i < e->numDrones; i++) {
        const droneEntity *drone = safe_array_get_at(e->drones, i);
        const enum utilityType type = drone->utility.utilityInfo->type;
        if (type != UNSET_UTILITY && e->map->utilityWeights[type] < MAX_PICKUP_SPAWN_WEIGHT) {
            heldWeights[type] += 1.0f;
        }
    }
}

static void countSpawnedPickup(iwEnv *e, const weaponPickupEntity *pickup) {
    switch (pickup->kind) {
    case WEAPON_PICKUP:
        ASSERT(pickup->weapon != STANDARD_WEAPON && pickup->weapon != UNSET_WEAPON);
        e->spawnedWeaponPickups[pickup->weapon]++;
        break;
    case UTILITY_PICKUP:
        ASSERT(pickup->utility != UNSET_UTILITY);
        e->spawnedUtilityPickups[pickup->utility]++;
        break;
    case CRATE_PICKUP:
        e->spawnedCrates++;
        break;
    }
}

// makes a pickup a utility picked by spawn weight
void randUtilityPickupType(iwEnv *e, weaponPickupEntity *pickup) {
    float heldWeights[_NUM_UTILITIES] = {0};
    heldUtilityWeights(e, heldWeights);
    float spawnWeights[_NUM_UTILITIES - 1] = {0};
    const float totalWeight = utilitySpawnWeights(e, heldWeights, spawnWeights);

    pickup->kind = UTILITY_PICKUP;
    pickup->weapon = UNSET_WEAPON;
    pickup->utility = GRAPPLE_UTILITY;
    if (totalWeight > 0.0f) {
        const float randPick = randFloat(&e->rng, 0.0f, totalWeight);
        float cumulativeWeight = 0.0f;
        for (uint8_t i = 1; i < NUM_UTILITIES; i++) {
            cumulativeWeight += spawnWeights[i - 1];
            if (randPick < cumulativeWeight) {
                pickup->utility = i;
                break;
            }
        }
    } else {
        // every utility has a 0 weight on this map, pick evenly
        pickup->utility = randInt(&e->rng, 1, NUM_UTILITIES - 1);
    }

    countSpawnedPickup(e, pickup);
}

// picks what a pickup spawns as: in combo episodes always an ammo crate,
// otherwise a weapon, utility or crate by spawn weight and how many pickups
// of each are currently spawned; the map's spawn weights are used, which
// are the default weights unless the map overrides them. Pickups spawned
// at the start of an episode are always weapons, utilities get their own
// starting pickups
void randPickupType(iwEnv *e, weaponPickupEntity *pickup, const bool episodeStart) {
    pickup->kind = CRATE_PICKUP;
    pickup->weapon = UNSET_WEAPON;
    pickup->utility = UNSET_UTILITY;
    if (!e->comboEp) {
        float totalWeight = 0.0f;
        float heldWeights[_NUM_WEAPONS] = {0};
        float spawnWeights[_NUM_WEAPONS - 1] = {0};

        // reduce the weight of weapons currently being held
        for (uint8_t i = 0; i < e->numDrones; i++) {
            const droneEntity *drone = safe_array_get_at(e->drones, i);
            if (drone->dead) {
                continue;
            }

            // if the weapon has the max weight don't reduce it based
            // off of players holding it; this will make it very unlikely
            // for both players to have it at once
            const enum weaponType heldType = drone->weapon.weaponInfo->type;
            if (heldType != STANDARD_WEAPON && e->map->weaponWeights[heldType] < MAX_PICKUP_SPAWN_WEIGHT) {
                heldWeights[heldType] += 1.0f;
            }
        }

        for (uint8_t i = 1; i < NUM_WEAPONS; i++) {
            const float discount = (float)e->spawnedWeaponPickups[i] + (heldWeights[i] * PICKUP_HELD_WEAPON_WEIGHT_COEF) + 1.0f;
            spawnWeights[i - 1] = e->map->weaponWeights[i] / discount;
            totalWeight += spawnWeights[i - 1];
        }

        float utilityWeights[_NUM_UTILITIES - 1] = {0};
        float crateWeight = 0.0f;
        if (!episodeStart) {
            float heldUtilities[_NUM_UTILITIES] = {0};
            heldUtilityWeights(e, heldUtilities);
            totalWeight += utilitySpawnWeights(e, heldUtilities, utilityWeights);

            crateWeight = e->map->crateWeight / (e->spawnedCrates + 1);
            totalWeight += crateWeight;
        }

        // the crate is what's left over past the weapons and utilities
        const float randPick = randFloat(&e->rng, 0.0f, totalWeight);
        float cumulativeWeight = 0.0f;
        for (uint8_t i = 1; i < NUM_WEAPONS; i++) {
            cumulativeWeight += spawnWeights[i - 1];
            if (randPick < cumulativeWeight) {
                pickup->kind = WEAPON_PICKUP;
                pickup->weapon = i;
                break;
            }
        }
        if (pickup->kind == CRATE_PICKUP) {
            for (uint8_t i = 1; i < NUM_UTILITIES; i++) {
                cumulativeWeight += utilityWeights[i - 1];
                if (randPick < cumulativeWeight) {
                    pickup->kind = UTILITY_PICKUP;
                    pickup->utility = i;
                    break;
                }
            }
        }
    }

    countSpawnedPickup(e, pickup);
}

void createWeaponPickupBodyShape(const iwEnv *e, weaponPickupEntity *pickup) {
    pickup->bodyDestroyed = false;

    b2BodyDef pickupBodyDef = b2DefaultBodyDef();
    pickupBodyDef.position = pickup->pos;
    pickupBodyDef.userData = pickup->ent;
    pickup->bodyID = b2CreateBody(e->worldID, &pickupBodyDef);

    b2ShapeDef pickupShapeDef = b2DefaultShapeDef();
    pickupShapeDef.filter.categoryBits = WEAPON_PICKUP_SHAPE;
    pickupShapeDef.filter.maskBits = FLOATING_WALL_SHAPE | DRONE_SHAPE;
    pickupShapeDef.isSensor = true;
    // overlaps are resolved in weaponPickupsOverlapStep instead; box2d
    // re-derives every sensor's overlap set inside b2World_Step, which is far
    // more expensive than testing a handful of static boxes directly
    pickupShapeDef.enableSensorEvents = false;
    pickupShapeDef.userData = pickup->ent;
    const b2Polygon pickupPolygon = b2MakeBox(PICKUP_THICKNESS / 2.0f, PICKUP_THICKNESS / 2.0f);
    pickup->shapeID = b2CreatePolygonShape(pickup->bodyID, &pickupShapeDef, &pickupPolygon);
}

static void addWeaponPickup(iwEnv *e, weaponPickupEntity *pickup, const b2Vec2 pos) {
    pickup->respawnWait = 0.0f;
    pickup->floatingWallsTouching = 0;
    pickup->pos = pos;

    entity *ent = createEntity(e, WEAPON_PICKUP_ENTITY, pickup);
    pickup->ent = ent;

    const int16_t cellIdx = entityPosToCellIdx(e, pos);
    if (cellIdx == -1) {
        ERRORF("invalid position for weapon pickup spawn: (%f, %f)", pos.x, pos.y);
    }
    pickup->mapCellIdx = cellIdx;
    mapCell *cell = (&e->cells[cellIdx]);
    cell->ent = ent;

    createWeaponPickupBodyShape(e, pickup);

    cc_array_add(e->pickups, pickup);
}

void createWeaponPickup(iwEnv *e) {
    // ensure weapon pickups are initially spawned somewhat uniformly
    b2Vec2 pos;
    e->lastSpawnQuad = (e->lastSpawnQuad + 1) % 4;
    if (!findOpenPos(e, WEAPON_PICKUP_SHAPE, &pos, e->lastSpawnQuad)) {
        // if we couldn't find a position in a specific quad fallback to
        // searching across the whole map
        if (!findOpenPos(e, WEAPON_PICKUP_SHAPE, &pos, -1)) {
            ERRORF("no open position for weapon pickup %zu; map %d", cc_array_size(e->pickups), e->mapIdx);
        }
    }

    weaponPickupEntity *pickup = fastCalloc(1, sizeof(weaponPickupEntity));
    randPickupType(e, pickup, true);
    addWeaponPickup(e, pickup, pos);
}

// creates one of the utility pickups spawned at the start of an episode;
// each is placed nearer one drone than the others so every drone can reach
// one fairly. Once picked up it respawns as any type, like other pickups
void createUtilityPickup(iwEnv *e, const uint8_t droneIdx) {
    const droneEntity *drone = safe_array_get_at(e->drones, droneIdx);
    b2Vec2 pos;
    bool found = false;
    for (uint8_t attempt = 0; attempt < UTILITY_PICKUP_PLACE_ATTEMPTS && !found; attempt++) {
        if (!findOpenPos(e, WEAPON_PICKUP_SHAPE, &pos, -1)) {
            break;
        }
        const float distance = b2DistanceSquared(pos, drone->pos);
        if (distance < UTILITY_PICKUP_MIN_DRONE_DISTANCE_SQUARED || distance > UTILITY_PICKUP_MAX_DRONE_DISTANCE_SQUARED) {
            continue;
        }
        found = true;
        for (uint8_t i = 0; i < e->numDrones; i++) {
            const droneEntity *other = safe_array_get_at(e->drones, i);
            if (i != droneIdx && b2DistanceSquared(pos, other->pos) <= distance) {
                found = false;
                break;
            }
        }
    }
    if (!found && !findOpenPos(e, WEAPON_PICKUP_SHAPE, &pos, mapQuad(e, drone->pos))) {
        if (!findOpenPos(e, WEAPON_PICKUP_SHAPE, &pos, -1)) {
            ERRORF("no open position for utility pickup %zu; map %d", cc_array_size(e->pickups), e->mapIdx);
        }
    }

    weaponPickupEntity *pickup = fastCalloc(1, sizeof(weaponPickupEntity));
    randUtilityPickupType(e, pickup);
    addWeaponPickup(e, pickup, pos);
}

void destroyWeaponPickup(iwEnv *e, weaponPickupEntity *pickup) {
    destroyEntity(e, pickup->ent);

    if (!pickup->bodyDestroyed) {
        mapCell *cell = (&e->cells[pickup->mapCellIdx]);
        cell->ent = NULL;
        b2DestroyBody(pickup->bodyID);
    }

    fastFree(pickup);
}

// destroys the pickup body and shape while the pickup is waiting to
// respawn to avoid spurious sensor overlap checks; enabling/disabling
// a body is almost as expensive and creating a new body in box2d, and
// manually moving (teleporting) it is expensive as well so destroying
// the body now and re-creating it later is the fastest
void disableWeaponPickup(iwEnv *e, weaponPickupEntity *pickup) {
    DEBUG_LOGF("disabling weapon pickup at cell %d (%f, %f)", pickup->mapCellIdx, pickup->pos.x, pickup->pos.y);

    pickup->respawnWait = PICKUP_RESPAWN_WAIT;
    if (e->suddenDeathWallsPlaced) {
        pickup->respawnWait = SUDDEN_DEATH_PICKUP_RESPAWN_WAIT;
    }
    b2DestroyBody(pickup->bodyID);
    pickup->bodyDestroyed = true;

    mapCell *cell = (&e->cells[pickup->mapCellIdx]);
    ASSERT(cell->ent != NULL);
    cell->ent = NULL;

    switch (pickup->kind) {
    case WEAPON_PICKUP:
        e->spawnedWeaponPickups[pickup->weapon]--;
        break;
    case UTILITY_PICKUP:
        e->spawnedUtilityPickups[pickup->utility]--;
        break;
    case CRATE_PICKUP:
        e->spawnedCrates--;
        break;
    }
}

void createDroneShield(iwEnv *e, droneEntity *drone, const int8_t groupIdx) {
    // the shield is comprised of 2 shapes over 2 bodies:
    // 1. a kinematic body that allows the parent drone to be unaffected
    // by collisions since kinematic bodies have essentially infinite mass
    // 2. a shape on the drone body that collides with walls and other
    // shields, since kinematic bodies don't collide with other kinematic
    // bodies or static bodies

    b2BodyDef shieldBodyDef = b2DefaultBodyDef();
    shieldBodyDef.type = b2_kinematicBody;
    shieldBodyDef.motionLocks = (b2MotionLocks){.angularZ = true};
    shieldBodyDef.position = drone->pos;
    b2BodyId shieldBodyID = b2CreateBody(e->worldID, &shieldBodyDef);

    b2ShapeDef shieldShapeDef = b2DefaultShapeDef();
    shieldShapeDef.filter.categoryBits = SHIELD_SHAPE;
    shieldShapeDef.filter.maskBits = PROJECTILE_SHAPE | DRONE_SHAPE;
    shieldShapeDef.filter.groupIndex = groupIdx;
    shieldShapeDef.enableContactEvents = true;

    b2ShapeDef shieldBufferShapeDef = b2DefaultShapeDef();
    shieldBufferShapeDef.density = 0.0f;
    shieldBufferShapeDef.material.friction = DRONE_FRICTION;
    shieldBufferShapeDef.material.restitution = DRONE_RESTITUTION;
    shieldBufferShapeDef.filter.categoryBits = SHIELD_SHAPE;
    shieldBufferShapeDef.filter.maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | SHIELD_SHAPE;
    shieldBufferShapeDef.filter.groupIndex = groupIdx;

    shieldEntity *shield = fastCalloc(1, sizeof(shieldEntity));
    shield->drone = drone;
    shield->bodyID = shieldBodyID;
    shield->pos = drone->pos;
    shield->health = DRONE_SHIELD_HEALTH;
    float duration = DRONE_SHIELD_START_DURATION;
    if (drone->livesLeft != DRONE_LIVES) {
        duration = DRONE_SHIELD_RESPAWN_DURATION;
    }
    shield->duration = duration;

    entity *ent = createEntity(e, SHIELD_ENTITY, shield);
    shield->ent = ent;

    shieldShapeDef.userData = ent;
    const b2Circle shieldCircle = {.center = b2Vec2_zero, .radius = DRONE_SHIELD_RADIUS};
    shield->shapeID = b2CreateCircleShape(shieldBodyID, &shieldShapeDef, &shieldCircle);
    b2Body_SetUserData(shield->bodyID, ent);

    shieldBufferShapeDef.userData = ent;
    shield->bufferShapeID = b2CreateCircleShape(drone->bodyID, &shieldBufferShapeDef, &shieldCircle);

    drone->shield = shield;
}

void resetWeapon(weaponSlot *weapon) {
    weapon->cooldown = 0.0f;
    weapon->heat = 0.0f;
    weapon->charge = 0.0f;
    weapon->fullChargeHoldTime = 0.0f;
}

// fills the utility's ammo and restarts its regen
void droneRefillUtility(droneEntity *drone) {
    drone->utility.ammo = drone->utility.utilityInfo->ammo;
    drone->utility.regen = 0.0f;
}

void droneSetUtility(droneEntity *drone, const enum utilityType type) {
    drone->utility.utilityInfo = utilityInfos[type];
    drone->utility.cooldown = 0.0f;
    droneRefillUtility(drone);
}

void createDroneTrails(droneEntity *drone) {
    create_array(&drone->brakeTrailPoints, 64);
    drone->trailPoints = fastCalloc(1, sizeof(trailPoints));
}

void createDrone(iwEnv *e, const uint8_t idx) {
    const int8_t groupIdx = -(idx + 1);
    b2BodyDef droneBodyDef = b2DefaultBodyDef();
    droneBodyDef.type = b2_dynamicBody;

    int8_t spawnQuad = -1;
    if (!e->isTraining) {
        // spawn drones in diagonal quadrants from each other so that
        // they're more likely to be further apart if we're not training;
        // doing this while training will result in much slower learning
        // due to drones starting much farther apart
        if (e->lastSpawnQuad == -1) {
            spawnQuad = randInt(&e->rng, 0, 3);
        } else if (e->numDrones == 2) {
            spawnQuad = 3 - e->lastSpawnQuad;
        } else {
            spawnQuad = (e->lastSpawnQuad + 1) % 4;
        }
        e->lastSpawnQuad = spawnQuad;
    }
    if (!findOpenPos(e, DRONE_SHAPE, &droneBodyDef.position, spawnQuad)) {
        ERROR("no open position for drone");
    }

    droneBodyDef.motionLocks = (b2MotionLocks){.angularZ = true};
    droneBodyDef.linearDamping = DRONE_LINEAR_DAMPING;
    b2BodyId droneBodyID = b2CreateBody(e->worldID, &droneBodyDef);
    b2ShapeDef droneShapeDef = b2DefaultShapeDef();
    droneShapeDef.density = DRONE_DENSITY;
    droneShapeDef.material.friction = DRONE_FRICTION;
    droneShapeDef.material.restitution = DRONE_RESTITUTION;
    droneShapeDef.filter.categoryBits = DRONE_SHAPE;
    droneShapeDef.filter.maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | WEAPON_PICKUP_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE | SHIELD_SHAPE;
    droneShapeDef.filter.groupIndex = groupIdx;
    droneShapeDef.enableContactEvents = true;
    droneShapeDef.enableSensorEvents = true;
    const b2Circle droneCircle = {.center = b2Vec2_zero, .radius = DRONE_RADIUS};

    droneEntity *drone = fastCalloc(1, sizeof(droneEntity));
    drone->bodyID = droneBodyID;
    drone->weapon.weaponInfo = weaponInfos[STANDARD_WEAPON];
    drone->weapon.ammo = drone->weapon.weaponInfo->ammo;
    droneSetUtility(drone, UNSET_UTILITY);
    drone->energyLeft = DRONE_ENERGY_MAX;
    drone->idx = idx;
    drone->team = idx;
    if (e->teamsEnabled) {
        drone->team = idx / (e->numDrones / 2);
    }
    drone->initalPos = droneBodyDef.position;
    drone->lastPos = droneBodyDef.position;
    drone->pos = droneBodyDef.position;
    drone->mapCellIdx = entityPosToCellIdx(e, droneBodyDef.position);
    drone->lastAim = (b2Vec2){.x = 0.0f, .y = -1.0f};
    drone->livesLeft = DRONE_LIVES;
    drone->respawnGuideLifetime = UINT16_MAX;
    memset(&drone->stepInfo, 0x0, sizeof(droneStepInfo));
    create_array(&drone->physicsTracking, 128);
    drone->killedBy = -1;

    if (e->client != NULL) {
        createDroneTrails(drone);
    }

    entity *ent = createEntity(e, DRONE_ENTITY, drone);
    drone->ent = ent;

    droneShapeDef.userData = ent;
    drone->shapeID = b2CreateCircleShape(droneBodyID, &droneShapeDef, &droneCircle);
    b2Body_SetUserData(drone->bodyID, ent);

    cc_array_add(e->drones, drone);

    createDroneShield(e, drone, groupIdx);
}

void droneAddEnergy(droneEntity *drone, float energy) {
    // if a burst is charging, add the energy to the burst charge
    if (drone->chargingBurst) {
        drone->burstCharge = clamp(drone->burstCharge + energy);
    } else {
        drone->energyLeft = clamp(drone->energyLeft + energy);
    }
}

void createDronePiece(iwEnv *e, droneEntity *drone, const bool fromShield) {
    const float distance = randFloat(&e->rng, DRONE_PIECE_MIN_DISTANCE, DRONE_PIECE_MAX_DISTANCE);
    const b2Vec2 direction = {.x = randFloat(&e->rng, -1.0f, 1.0f), .y = randFloat(&e->rng, -1.0f, 1.0f)};
    const b2Vec2 pos = b2MulAdd(drone->pos, distance, direction);
    const b2Rot rot = b2MakeRot(randFloat(&e->rng, -PI, PI));
    const float bonus = 1.0f + min(b2Length(drone->velocity) / 15.0f, 5.0f);
    const float speed = randFloat(&e->rng, DRONE_PIECE_MIN_SPEED, DRONE_PIECE_MAX_SPEED) * bonus;
    const float angularVelocity = randFloat(&e->rng, -PI, PI);

    // only create drone pieces when we're rendering, but always draw
    // random values to keep the rng consistent
    if (e->client == NULL) {
        return;
    }

    dronePieceEntity *piece = fastCalloc(1, sizeof(dronePieceEntity));
    piece->droneIdx = drone->idx;
    piece->pos = pos;
    piece->rot = rot;
    piece->isShieldPiece = fromShield;
    piece->lifetime = UINT16_MAX;

    entity *ent = createEntity(e, DRONE_PIECE_ENTITY, piece);
    piece->ent = ent;

    b2BodyDef pieceBodyDef = b2DefaultBodyDef();
    pieceBodyDef.type = b2_dynamicBody;

    pieceBodyDef.position = pos;
    pieceBodyDef.rotation = rot;
    pieceBodyDef.linearDamping = DRONE_PIECE_LINEAR_DAMPING;
    pieceBodyDef.angularDamping = DRONE_PIECE_ANGULAR_DAMPING;
    pieceBodyDef.sleepThreshold = FLOATING_WALL_SLEEP_THRESHOLD;
    pieceBodyDef.linearVelocity = b2MulSV(speed, direction);
    pieceBodyDef.angularVelocity = angularVelocity;
    pieceBodyDef.userData = ent;
    piece->bodyID = b2CreateBody(e->worldID, &pieceBodyDef);

    b2ShapeDef pieceShapeDef = b2DefaultShapeDef();
    pieceShapeDef.filter.categoryBits = DRONE_PIECE_SHAPE;
    // debris must not push floating walls around: it only exists when a client
    // is attached, so letting it affect them would make rendered rounds play
    // differently from headless ones
    pieceShapeDef.filter.maskBits = WALL_SHAPE | DRONE_PIECE_SHAPE;
    pieceShapeDef.density = 1.0f;
    pieceShapeDef.material.friction = 0.5f;
    pieceShapeDef.userData = ent;

    // make pieces from the shield a bit smaller
    if (fromShield) {
        piece->vertices[0] = (b2Vec2){.x = 0.0f, .y = -1.0f};
        piece->vertices[1] = (b2Vec2){.x = -0.5f, .y = 0.0f};
        piece->vertices[2] = (b2Vec2){.x = 0.5f, .y = 0.0f};
    } else {
        piece->vertices[0] = (b2Vec2){.x = 0.0f, .y = -1.5f};
        piece->vertices[1] = (b2Vec2){.x = -0.75f, .y = 0.0f};
        piece->vertices[2] = (b2Vec2){.x = 0.75f, .y = 0.0f};
    }

    const b2Hull pieceHull = b2ComputeHull(piece->vertices, 3);
    const b2Polygon piecePolygon = b2MakePolygon(&pieceHull, 0.0f);
    piece->shapeID = b2CreatePolygonShape(piece->bodyID, &pieceShapeDef, &piecePolygon);

    cc_array_add(e->dronePieces, piece);
}

void destroyDronePiece(iwEnv *e, dronePieceEntity *piece) {
    b2DestroyBody(piece->bodyID);
    destroyEntity(e, piece->ent);
    fastFree(piece);
}

void destroyDroneShield(iwEnv *e, shieldEntity *shield, const bool createPieces) {
    droneEntity *drone = shield->drone;
    const float health = shield->health;
    if (health <= 0.0f) {
        droneAddEnergy(drone, DRONE_SHIELD_BREAK_ENERGY_COST);
        e->stats[drone->idx].ownShieldBroken++;
    }
    drone->shield = NULL;

    b2DestroyBody(shield->bodyID);
    b2DestroyShape(shield->bufferShapeID, false);
    destroyEntity(e, shield->ent);
    fastFree(shield);

    if (!createPieces || health > 0.0f) {
        return;
    }

    // only create pieces if the shield was broken early
    for (uint8_t i = 0; i < DRONE_PIECE_COUNT; i++) {
        createDronePiece(e, drone, true);
    }
}

void destroyDrone(iwEnv *e, droneEntity *drone) {
    for (size_t i = 0; i < cc_array_size(drone->physicsTracking); i++) {
        physicsStepInfo *physicsStep = safe_array_get_at(drone->physicsTracking, i);
        fastFree(physicsStep);
    }
    cc_array_destroy(drone->physicsTracking);

    destroyEntity(e, drone->ent);

    shieldEntity *shield = drone->shield;
    if (shield != NULL) {
        destroyDroneShield(e, shield, false);
    }

    if (drone->brakeTrailPoints != NULL) {
        for (size_t i = 0; i < cc_array_size(drone->brakeTrailPoints); i++) {
            brakeTrailPoint *trailPoint = safe_array_get_at(drone->brakeTrailPoints, i);
            fastFree(trailPoint);
        }
        cc_array_destroy(drone->brakeTrailPoints);
    }
    if (drone->trailPoints != NULL) {
        fastFree(drone->trailPoints);
    }

    b2DestroyBody(drone->bodyID);
    fastFree(drone);
}

void trackForce(const iwEnv *e, CC_Array *physicsTracking, const b2Vec2 force, const uint8_t srcIdx, const physicsSource source) {
    physicsStepInfo *physicsStep = fastCalloc(1, sizeof(physicsStepInfo));
    physicsStep->srcIdx = srcIdx;
    physicsStep->force = force;
    physicsStep->source = source;
    physicsStep->step = e->episodeLength;
    cc_array_add(physicsTracking, physicsStep);
}

void applyTrackedForce(const iwEnv *e, const b2BodyId bodyID, CC_Array *physicsTracking, const b2Vec2 force, const uint8_t srcIdx, const physicsSource source) {
    b2Body_ApplyForceToCenter(bodyID, force, true);
    trackForce(e, physicsTracking, force, srcIdx, source);
}

void trackImpulse(const iwEnv *e, CC_Array *physicsTracking, const b2Vec2 impulse, const uint8_t srcIdx, const physicsSource source) {
    physicsStepInfo *physicsStep = fastCalloc(1, sizeof(physicsStepInfo));
    physicsStep->srcIdx = srcIdx;
    physicsStep->impulse = impulse;
    physicsStep->source = source;
    physicsStep->step = e->episodeLength;
    cc_array_add(physicsTracking, physicsStep);
}

void applyTrackedImpulse(const iwEnv *e, const b2BodyId bodyID, CC_Array *physicsTracking, const b2Vec2 impulse, const uint8_t srcIdx, const physicsSource source) {
    b2Body_ApplyLinearImpulseToCenter(bodyID, impulse, true);
    trackImpulse(e, physicsTracking, impulse, srcIdx, source);
}

void droneTrackBrake(const iwEnv *e, const droneEntity *drone) {
    physicsStepInfo *physicsStep = fastCalloc(1, sizeof(physicsStepInfo));
    physicsStep->brakeToggled = true;
    physicsStep->step = e->episodeLength;
    cc_array_add(drone->physicsTracking, physicsStep);
}

// get the physics tracking of the body that bounced off of a static wall,
// or NULL if the contact isn't a bounce that should be tracked
CC_Array *wallBounceTracking(const entity *ent, const entity *wallEnt) {
    if (!entityTypeIsWall(wallEnt->type)) {
        return NULL;
    }
    // reflecting assumes the wall doesn't move
    const wallEntity *wall = wallEnt->entity;
    if (wall->isFloating) {
        return NULL;
    }

    switch (ent->type) {
    case DRONE_ENTITY:
        if (wallEnt->type == DEATH_WALL_ENTITY) {
            return NULL;
        }
        return ((droneEntity *)ent->entity)->physicsTracking;
    case SHIELD_ENTITY:
        return ((shieldEntity *)ent->entity)->drone->physicsTracking;
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY: {
        wallEntity *bouncingWall = ent->entity;
        if (bouncingWall->isFloating) {
            return bouncingWall->physicsTracking;
        }
        return NULL;
    }
    default:
        return NULL;
    }
}

// track a body bouncing off of a static wall so findBiggestContributor
// can reflect the pushes that moved the body into the wall
void trackWallBounce(const iwEnv *e, const entity *e1, const entity *e2, const b2ContactId contactID) {
    bool wallIsShapeA = false;
    CC_Array *physicsTracking = wallBounceTracking(e1, e2);
    if (physicsTracking == NULL) {
        physicsTracking = wallBounceTracking(e2, e1);
        wallIsShapeA = true;
    }
    if (physicsTracking == NULL || !b2Contact_IsValid(contactID)) {
        return;
    }
    const b2ContactData contactData = b2Contact_GetData(contactID);
    const b2Manifold manifold = contactData.manifold;
    if (manifold.pointCount == 0) {
        return;
    }

    // the manifold normal points from shape A to shape B, make it
    // point out of the wall
    b2Vec2 normal = manifold.normal;
    if (!wallIsShapeA) {
        normal = b2Neg(normal);
    }

    // match box2d: restitution is the max of both shapes' restitution,
    // and is only applied if the shapes are approaching faster than
    // the restitution threshold
    float approachSpeed = 0.0f;
    for (uint8_t i = 0; i < manifold.pointCount; i++) {
        approachSpeed = fmaxf(approachSpeed, -manifold.points[i].normalVelocity);
    }
    float restitution = 0.0f;
    if (approachSpeed > b2World_GetRestitutionThreshold(e->worldID)) {
        restitution = fmaxf(b2Shape_GetRestitution(contactData.shapeIdA), b2Shape_GetRestitution(contactData.shapeIdB));
    }

    physicsStepInfo *physicsStep = fastCalloc(1, sizeof(physicsStepInfo));
    physicsStep->bounce = true;
    physicsStep->normal = normal;
    physicsStep->restitution = restitution;
    physicsStep->step = e->episodeLength;
    cc_array_add(physicsTracking, physicsStep);
}

// a pickup always replaces the weapon, or tops up its ammo if it's the
// same weapon; returns true if it replaced the standard weapon
bool droneChangeWeapon(droneEntity *drone, const enum weaponType newWeapon) {
    weaponSlot *weapon = &drone->weapon;
    const bool replacedStandard = weapon->weaponInfo->type == STANDARD_WEAPON;
    if (weapon->weaponInfo->type != newWeapon) {
        weapon->weaponInfo = weaponInfos[newWeapon];
        resetWeapon(weapon);
    }
    weapon->ammo = weapon->weaponInfo->ammo;
    return replacedStandard;
}

// returns true if the new utility filled an unset slot; the new utility
// starts with full ammo. Refilling a utility isn't counted since utility
// ammo regenerates on its own
bool droneChangeUtility(iwEnv *e, droneEntity *drone, const enum utilityType newUtility) {
    const enum utilityType curType = drone->utility.utilityInfo->type;
    if (curType == newUtility) {
        droneRefillUtility(drone);
        return false;
    }

    // a grapple can't be held without the grapple utility; placed mines
    // are kept and can still go off by proximity
    if (drone->grapple != NULL) {
        destroyProjectile(e, drone->grapple, false, true);
    }
    droneSetUtility(drone, newUtility);
    return curType == UNSET_UTILITY;
}

// an ammo crate fills the weapon, unless it's the standard weapon, and the
// utility to full ammo; returns true if it added any ammo
bool droneRefillAmmo(droneEntity *drone) {
    bool refilled = false;
    weaponSlot *weapon = &drone->weapon;
    if (weapon->weaponInfo->type != STANDARD_WEAPON && weapon->ammo < weapon->weaponInfo->ammo) {
        weapon->ammo = weapon->weaponInfo->ammo;
        refilled = true;
    }

    if (drone->utility.ammo < drone->utility.utilityInfo->ammo) {
        droneRefillUtility(drone);
        refilled = true;
    }

    return refilled;
}

void droneEmptyCharge(droneEntity *drone) {
    drone->weapon.charge = 0.0f;
    drone->weapon.fullChargeHoldTime = 0.0f;
}

// contribution of a single drone from a single physics source; kept
// separately so the source of a kill can be determined the same way
// the killer is
typedef struct sourceContrib {
    b2Vec2 contrib;
    physicsSource source;
    uint8_t srcIdx;
} sourceContrib;

void applyContribDamping(const iwEnv *e, const uint16_t stepDiff, const float damping, b2Vec2 *contrib, sourceContrib *sources, const uint16_t numSources) {
    const float perStepDamping = 1.0f / (1.0f + damping * e->deltaTime);
    const float damp = powf(perStepDamping, stepDiff);
    for (uint8_t k = 0; k < e->numDrones; k++) {
        contrib[k] = b2MulSV(damp, contrib[k]);
    }
    for (uint16_t k = 0; k < numSources; k++) {
        sources[k].contrib = b2MulSV(damp, sources[k].contrib);
    }
}

// reflect pushes into the wall the same way box2d reflects the
// body's velocity; pushes away from the wall are left alone
b2Vec2 reflectContrib(const b2Vec2 contrib, const physicsStepInfo *physicsStep) {
    const float intoWall = b2Dot(contrib, physicsStep->normal);
    if (intoWall < 0.0f) {
        return b2MulSub(contrib, (1.0f + physicsStep->restitution) * intoWall, physicsStep->normal);
    }
    return contrib;
}

int8_t findBiggestContributor(const iwEnv *e, const enum entityType type, const CC_Array *physicsTracking, const b2Vec2 lastVelocity, float *maxMoveContrib, physicsSource *source) {
    b2Vec2 contrib[e->numDrones];
    memset(contrib, 0x0, e->numDrones * sizeof(b2Vec2));
    const uint16_t maxSources = e->numDrones * NUM_PHYSICS_TYPES * _NUM_WEAPONS;
    sourceContrib sources[maxSources];
    int16_t sourceSlots[maxSources];
    memset(sourceSlots, 0xff, maxSources * sizeof(int16_t));
    uint16_t numSources = 0;
    uint16_t step = 0;
    bool braking = false;
    float defaultDamping = DRONE_LINEAR_DAMPING;
    float invMass = DRONE_INV_MASS;
    if (type != DRONE_ENTITY) {
        defaultDamping = FLOATING_WALL_DAMPING;
        invMass = FLOATING_WALL_INV_MASS;
    }
    float damping = defaultDamping;

    // calculate the contribution of forces and impulses of each drone
    for (size_t i = 0; i < cc_array_size(physicsTracking); i++) {
        physicsStepInfo *physicsStep = safe_array_get_at(physicsTracking, i);
        // if the step has changed, apply damping to contributions
        if (physicsStep->step != step) {
            applyContribDamping(e, physicsStep->step - step, damping, contrib, sources, numSources);
            step = physicsStep->step;
        }

        if (physicsStep->brakeToggled) {
            braking = !braking;
            if (braking) {
                damping = DRONE_BRAKE_DAMPING_COEF;
            } else {
                damping = defaultDamping;
            }
        }
        if (physicsStep->bounce) {
            for (uint8_t k = 0; k < e->numDrones; k++) {
                contrib[k] = reflectContrib(contrib[k], physicsStep);
            }
            for (uint16_t k = 0; k < numSources; k++) {
                sources[k].contrib = reflectContrib(sources[k].contrib, physicsStep);
            }
            continue;
        }

        const uint8_t srcIdx = physicsStep->srcIdx;
        const b2Vec2 invForce = b2MulSV(invMass, physicsStep->force);
        const b2Vec2 stepContrib = b2MulAdd(b2MulSV(invMass, physicsStep->impulse), e->deltaTime, invForce);
        if (b2Equal(stepContrib, b2Vec2_zero)) {
            continue;
        }
        contrib[srcIdx] = b2Add(contrib[srcIdx], stepContrib);

        const physicsSource stepSource = physicsStep->source;
        const uint16_t slotIdx = (srcIdx * NUM_PHYSICS_TYPES + stepSource.type) * _NUM_WEAPONS + stepSource.weapon;
        ASSERT(slotIdx < maxSources);
        if (sourceSlots[slotIdx] == -1) {
            sourceSlots[slotIdx] = numSources;
            sources[numSources] = (sourceContrib){.contrib = b2Vec2_zero, .source = stepSource, .srcIdx = srcIdx};
            numSources++;
        }
        sourceContrib *src = &sources[sourceSlots[slotIdx]];
        src->contrib = b2Add(src->contrib, stepContrib);

        // DEBUG_LOGF("# step %d drone %d contrib %f (%f, %f)", physicsStep->step, physicsStep->srcIdx, b2Length(contrib[physicsStep->srcIdx]), contrib[physicsStep->srcIdx].x, contrib[physicsStep->srcIdx].y);
    }
    applyContribDamping(e, e->episodeLength - step, damping, contrib, sources, numSources);

    // determine the killer by finding the drone that pushed the dead
    // drone towards the wall that killed it the most
    const b2Vec2 deathNormal = b2Normalize(lastVelocity);
    DEBUG_LOGF("> death normal (%f, %f) velocity (%f, %f)", deathNormal.x, deathNormal.y, lastVelocity.x, lastVelocity.y);
    float maxContrib = 1e-6f;
    int8_t killer = -1;
    for (uint8_t i = 0; i < e->numDrones; i++) {
        DEBUG_LOGF("---\n> src drone contrib %d %f (%f, %f)", i, b2Length(contrib[i]), contrib[i].x, contrib[i].y);
        if (b2Equal(contrib[i], b2Vec2_zero)) {
            continue;
        }

        const b2Vec2 normContrib = b2Normalize(contrib[i]);
        const float dot = b2Dot(normContrib, deathNormal);
        DEBUG_LOGF("> > src drone %d norm contrib (%f, %f)", i, normContrib.x, normContrib.y);
        DEBUG_LOGF("> src drone %d death normal dot %f", i, dot);
        const float totalContrib = b2Length(contrib[i]) * dot;
        DEBUG_LOGF("> src drone %d total contrib %f", i, totalContrib);
        if (totalContrib > maxContrib) {
            maxContrib = totalContrib;
            killer = i;
        }
    }
    if (killer == -1) {
        return killer;
    }
    *maxMoveContrib = maxContrib;

    // the source of the kill is whichever of the killer's sources
    // pushed the dead drone towards the wall the most
    float maxSourceContrib = -FLT_MAX;
    for (uint16_t i = 0; i < numSources; i++) {
        if (sources[i].srcIdx != killer) {
            continue;
        }
        const float projected = b2Dot(sources[i].contrib, deathNormal);
        DEBUG_LOGF("> killer source type %d weapon %d contrib %f", sources[i].source.type, sources[i].source.weapon, projected);
        if (projected > maxSourceContrib) {
            maxSourceContrib = projected;
            *source = sources[i].source;
        }
    }

    return killer;
}

// returns the index of the killer, or -1 if it's unknown
int8_t findDroneKiller(iwEnv *e, droneEntity *drone, const wallEntity *killWall) {
    float maxMoveContrib = -FLT_MAX;
    physicsSource source = {0};
    DEBUG_LOG("finding drone killer");
    int8_t killer = findBiggestContributor(e, DRONE_ENTITY, drone->physicsTracking, drone->lastVelocity, &maxMoveContrib, &source);
    if (killWall != NULL && killWall->isFloating) {
        float wallContrib = -FLT_MAX;
        physicsSource wallSource = {0};
        DEBUG_LOG("finding mover of floating death wall");
        const int8_t wallMover = findBiggestContributor(e, DEATH_WALL_ENTITY, killWall->physicsTracking, killWall->velocity, &wallContrib, &wallSource);
        if (wallContrib > maxMoveContrib) {
            DEBUG_LOGF(">>> drone %d killed by drone %d pushing floating death wall", drone->idx, wallMover);
            killer = wallMover;
            source = wallSource;
        }
    }

    if (killer == -1) {
        DEBUG_LOGF(">>> drone %d killed by UNKNOWN", drone->idx);
        e->stats[drone->idx].unknownKills++;
        return -1;
    }

    DEBUG_LOGF(">>> drone %d killed by drone %d", drone->idx, killer);

    if (killer == drone->idx) {
        e->stats[drone->idx].selfKills++;
    } else {
        e->stats[killer].kills++;
    }
    drone->killedBy = killer;
    drone->killSource = source;
    droneEntity *killerDrone = safe_array_get_at(e->drones, killer);
    killerDrone->killed[drone->idx] = true;
    return killer;
}

static inline bool physicsSourceIsWeapon(const physicsSource source) {
    return source.type == RECOIL_PHYSICS || source.type == PROJECTILE_PHYSICS || source.type == EXPLOSION_PHYSICS || source.type == GRAPPLE_PHYSICS;
}

void recordDeathAnalysis(iwEnv *e, const droneEntity *drone, const int8_t killer, const wallEntity *killWall) {
    droneAnalysisStats *stats = &e->analysisStats[drone->idx];
    stats->diedHoldingActive[drone->weapon.weaponInfo->type]++;

    if (killer == -1) {
        return;
    }

    const physicsSource source = drone->killSource;
    const bool floatingWall = killWall != NULL && killWall->isFloating;
    const bool weapon = physicsSourceIsWeapon(source);
    if (killer == drone->idx) {
        stats->selfKillsBySource[source.type]++;
        stats->floatingWallSelfKills += floatingWall;
        if (weapon) {
            stats->selfKillsByWeapon[source.weapon]++;
            stats->boostedSelfKills[source.weapon] += source.boosted;
        }
        return;
    }

    stats->deathsBySource[source.type]++;
    stats->floatingWallDeaths += floatingWall;
    if (weapon) {
        droneAnalysisStats *killerStats = &e->analysisStats[killer];
        killerStats->killsByWeapon[source.weapon]++;
        killerStats->boostedKills[source.weapon] += source.boosted;
    }
}

// records a black hole pull on or grapple attachment to an enemy, which
// earns an assist if the enemy dies soon after
static inline void markAssistSetup(const iwEnv *e, droneEntity *attacker, const droneEntity *victim, const enum assistWeapon weapon) {
    if (attacker->team == victim->team) {
        return;
    }
    attacker->assistFrame[victim->idx][weapon] = e->episodeLength + 1;
}

// gives an assist to every enemy that pulled the dying drone with a black
// hole or had a grapple attached to it within DRONE_ASSIST_WINDOW, unless
// the kill itself is credited to that weapon, as the kill reward already
// covers it
void recordAssists(iwEnv *e, const droneEntity *drone, const int8_t killer) {
    static const enum weaponType assistWeapons[NUM_ASSIST_WEAPONS] = {
        [ASSIST_BLACK_HOLE] = BLACK_HOLE_WEAPON,
        [ASSIST_GRAPPLE] = GRAPPLE_WEAPON,
    };
    const uint32_t window = (uint32_t)(DRONE_ASSIST_WINDOW * e->frameRate);
    const physicsSource source = drone->killSource;
    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        if (i == drone->idx) {
            continue;
        }

        droneEntity *other = safe_array_get_at(e->drones, i);
        for (uint8_t w = 0; w < NUM_ASSIST_WEAPONS; w++) {
            const uint32_t frame = other->assistFrame[drone->idx][w];
            other->assistFrame[drone->idx][w] = 0;
            if (frame == 0 || (uint32_t)e->episodeLength + 1 - frame > window) {
                continue;
            }
            if (killer == i && physicsSourceIsWeapon(source) && source.weapon == assistWeapons[w]) {
                continue;
            }
            other->stepInfo.assists++;
            e->stats[i].assists++;
            e->analysisStats[i].assistsByWeapon[assistWeapons[w]]++;
        }
    }
}

void destroyGrapplesAttachedTo(iwEnv *e, const entity *ent) {
    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->grapple != NULL && drone->grapple->grappledEnt == ent) {
            destroyProjectile(e, drone->grapple, false, true);
        }
    }
}

// falls back to the standard weapon once the weapon runs dry
void droneUnsetWeapon(droneEntity *drone) {
    drone->weapon.weaponInfo = weaponInfos[STANDARD_WEAPON];
    resetWeapon(&drone->weapon);
    drone->weapon.ammo = drone->weapon.weaponInfo->ammo;
}

void killDrone(iwEnv *e, droneEntity *drone, const wallEntity *killWall) {
    if (drone->dead || drone->livesLeft == 0) {
        return;
    }
    DEBUG_LOGF("drone %d died", drone->idx);

    const int8_t killer = findDroneKiller(e, drone, killWall);
    recordDeathAnalysis(e, drone, killer, killWall);
    recordAssists(e, drone, killer);

    for (size_t i = 0; i < cc_array_size(drone->physicsTracking); i++) {
        physicsStepInfo *physicsStep = safe_array_get_at(drone->physicsTracking, i);
        fastFree(physicsStep);
    }
    cc_array_remove_all(drone->physicsTracking);

    drone->livesLeft--;
    drone->dead = true;
    drone->diedThisStep = true;
    drone->respawnWait = DRONE_RESPAWN_WAIT;

    if (drone->grapple != NULL) {
        e->analysisStats[drone->idx].grappleAttachedDeaths += drone->grapple->welded;
        destroyProjectile(e, drone->grapple, false, true);
    }
    destroyGrapplesAttachedTo(e, drone->ent);

    for (uint8_t i = 0; i < DRONE_PIECE_COUNT; i++) {
        createDronePiece(e, drone, false);
    }

    shieldEntity *shield = drone->shield;
    if (shield != NULL) {
        destroyDroneShield(e, shield, false);
    }

    // terminate any active brake trail so it won't connect to the
    // drone's respawn position the next time it brakes
    if (drone->braking && e->client != NULL && drone->brakeTrailPoints != NULL) {
        brakeTrailPoint *trailPoint = fastCalloc(1, sizeof(brakeTrailPoint));
        trailPoint->pos = drone->pos;
        trailPoint->lifetime = UINT16_MAX;
        trailPoint->isEnd = true;
        cc_array_add(drone->brakeTrailPoints, trailPoint);
    }

    resetWeapon(&drone->weapon);

    b2Body_Disable(drone->bodyID);
    drone->triggerHeld = false;
    drone->utilityHeld = false;
    drone->braking = false;
    drone->chargingBurst = false;
    drone->burstCharge = 0.0f;
    drone->burstChargeUsed = 0.0f;
    drone->burstCooldown = 0.0f;
    drone->burstRadius = 0.0f;
    drone->burstImpulse = 0.0f;
    drone->burstFramesLeft = 0;
    drone->burstStaticWallsHit = false;
    drone->energyFullyDepleted = false;
    drone->shotThisStep = false;
    drone->velocity = b2Vec2_zero;
    drone->lastVelocity = b2Vec2_zero;
}

bool respawnDrone(iwEnv *e, droneEntity *drone) {
    b2Vec2 pos = b2Vec2_zero;
    uint8_t deadDrones = 0;
    uint8_t openQuads = 4;
    int8_t quads[4];

    for (uint8_t i = 0; i < 4; i++) {
        quads[i] = i;
    }

    // attempt to respawn the drone in a different part of the map than
    // living drones
    for (uint8_t i = 0; i < e->numDrones; i++) {
        if (i == drone->idx) {
            continue;
        }

        const droneEntity *otherDrone = safe_array_get_at(e->drones, i);
        if (otherDrone->dead) {
            deadDrones++;
            continue;
        }
        const int8_t quad = mapQuad(e, otherDrone->pos);
        if (quads[quad] != -1) {
            openQuads--;
            quads[quad] = -1;
        }
    }

    // if all drones are dead just pick a random position
    if (deadDrones == e->numDrones - 1 || openQuads == 0) {
        if (!findOpenPos(e, DRONE_SHAPE, &pos, -1)) {
            return false;
        }
    } else {
        bool foundPos = false;
        shuffle(&e->rng, quads, 4);
        for (uint8_t i = 0; i < 4; i++) {
            if (quads[i] == -1) {
                continue;
            }
            if (findOpenPos(e, DRONE_SHAPE, &pos, quads[i])) {
                foundPos = true;
                break;
            }
        }
        // no open spots in open quads, search the whole map
        if (!foundPos) {
            if (!findOpenPos(e, DRONE_SHAPE, &pos, -1)) {
                return false;
            }
        }
    }

    b2Body_SetTransform(drone->bodyID, pos, b2Rot_identity);
    b2Body_Enable(drone->bodyID);
    b2Body_SetLinearDamping(drone->bodyID, DRONE_LINEAR_DAMPING);

    drone->dead = false;
    drone->respawnWait = 0.0f;
    drone->pos = pos;
    drone->lastPos = pos;
    drone->mapCellIdx = entityPosToCellIdx(e, drone->pos);
    drone->respawnGuideLifetime = UINT16_MAX;
    drone->killedBy = -1;

    droneAddEnergy(drone, DRONE_ENERGY_RESPAWN_REFILL);
    // utilities survive death and come back with full ammo
    droneRefillUtility(drone);
    drone->utility.cooldown = 0.0f;

    createDroneShield(e, drone, -(drone->idx + 1));

    if (drone->trailPoints != NULL) {
        drone->trailPoints->length = 0;
    }

    return true;
}

void createProjectile(iwEnv *e, droneEntity *drone, const weaponInformation *weaponInfo, const b2Vec2 normAim) {
    ASSERT_VEC_NORMALIZED(normAim);

    const float radius = weaponInfo->radius;
    float droneRadius = DRONE_RADIUS;
    if (drone->shield != NULL) {
        droneRadius = DRONE_SHIELD_RADIUS;
    }
    // spawn the projectile just outside the drone so they don't
    // immediately collide
    b2Vec2 pos = b2MulAdd(drone->pos, droneRadius + (radius * 1.5f), normAim);
    // if the projectile is inside a wall or out of the map, move the
    // projectile to be just outside the wall
    bool projectileInWall = false;
    int16_t cellIdx = entityPosToCellIdx(e, pos);
    if (cellIdx == -1) {
        projectileInWall = true;
    } else {
        const mapCell *cell = (&e->cells[cellIdx]);
        if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
            projectileInWall = true;
        }
    }
    if (projectileInWall) {
        const b2Vec2 rayEnd = b2MulAdd(drone->pos, droneRadius + (radius * 2.5f), normAim);
        const b2Vec2 translation = b2Sub(rayEnd, drone->pos);
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE};
        const b2RayResult rayRes = b2World_CastRayClosest(e->worldID, drone->pos, translation, filter);
        if (rayRes.hit) {
            const b2Vec2 invNormAim = b2MulSV(-1.0f, normAim);
            pos = b2MulAdd(rayRes.point, radius * 1.5f, invNormAim);
        }
    }

    b2BodyDef projectileBodyDef = b2DefaultBodyDef();
    projectileBodyDef.type = b2_dynamicBody;
    projectileBodyDef.isBullet = weaponInfo->isPhysicsBullet;
    projectileBodyDef.linearDamping = weaponInfo->damping;
    projectileBodyDef.enableSleep = weaponInfo->canSleep;
    projectileBodyDef.position = pos;
    b2BodyId projectileBodyID = b2CreateBody(e->worldID, &projectileBodyDef);
    b2ShapeDef projectileShapeDef = b2DefaultShapeDef();
    projectileShapeDef.enableContactEvents = true;
    projectileShapeDef.density = weaponInfo->density;
    projectileShapeDef.material.restitution = PROJECTILE_RESTITUTION;
    projectileShapeDef.material.friction = 0.0f;
    projectileShapeDef.filter.categoryBits = PROJECTILE_SHAPE;
    projectileShapeDef.filter.maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE | SHIELD_SHAPE;
    const b2Circle projectileCircle = {.center = b2Vec2_zero, .radius = radius};

    b2ShapeId projectileShapeID = b2CreateCircleShape(projectileBodyID, &projectileShapeDef, &projectileCircle);

    // add a bit of lateral drone velocity to projectile
    b2Vec2 forwardVel = b2MulSV(b2Dot(drone->velocity, normAim), normAim);
    b2Vec2 lateralVel = b2Sub(drone->velocity, forwardVel);
    lateralVel = b2MulSV(projectileShapeDef.density * DRONE_MOVE_AIM_COEF, lateralVel);
    b2Vec2 fire = b2MulAdd(lateralVel, weaponFire(&e->rng, weaponInfo->type), normAim);
    b2Body_ApplyLinearImpulseToCenter(projectileBodyID, fire, true);

    projectileEntity *projectile = fastCalloc(1, sizeof(projectileEntity));
    projectile->droneIdx = drone->idx;
    projectile->bodyID = projectileBodyID;
    projectile->shapeID = projectileShapeID;
    projectile->weaponInfo = (weaponInformation *)weaponInfo;
    projectile->pos = projectileBodyDef.position;
    projectile->lastPos = projectileBodyDef.position;
    projectile->velocity = b2Body_GetLinearVelocity(projectileBodyID);
    projectile->lastVelocity = projectile->velocity;
    projectile->speed = b2Length(projectile->velocity);
    projectile->lastSpeed = projectile->speed;
    if (projectile->weaponInfo->type == BLACK_HOLE_WEAPON) {
        create_array(&projectile->entsInBlackHole, 4);
    }
    if (e->client != NULL) {
        projectile->trailPoints = fastCalloc(1, sizeof(trailPoints));
    }
    cc_array_add(e->projectiles, projectile);

    entity *ent = createEntity(e, PROJECTILE_ENTITY, projectile);
    projectile->ent = ent;
    b2Body_SetUserData(projectile->bodyID, ent);
    b2Shape_SetUserData(projectile->shapeID, ent);

    // create a sensor shape if needed
    if (projectile->weaponInfo->hasSensor) {
        projectile->sensorID = weaponSensor(projectile->bodyID, projectile->weaponInfo->type);
        b2Shape_SetUserData(projectile->sensorID, ent);
    }

    if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON) {
        e->minesCreated[drone->idx]++;
        drone->mine = projectile;
    } else if (projectile->weaponInfo->type == GRAPPLE_WEAPON) {
        projectile->grappleHealth = GRAPPLE_HEALTH;
        ASSERT(drone->grapple == NULL);
        drone->grapple = projectile;
    }
}

// simplified and copied from box2d/src/shape.c
float getShapeProjectedPerimeter(const b2ShapeId shapeID, const b2Vec2 line) {
    if (b2Shape_GetType(shapeID) == b2_circleShape) {
        const b2Circle circle = b2Shape_GetCircle(shapeID);
        return circle.radius * 2.0f;
    }

    const b2Polygon polygon = b2Shape_GetPolygon(shapeID);
    const b2Vec2 *points = polygon.vertices;
    int count = polygon.count;
    B2_ASSERT(count > 0);
    float value = b2Dot(points[0], line);
    float lower = value;
    float upper = value;
    for (int i = 1; i < count; ++i) {
        value = b2Dot(points[i], line);
        lower = b2MinFloat(lower, value);
        upper = b2MaxFloat(upper, value);
    }

    return upper - lower;
}

// explodes projectile and ensures any other projectiles that are caught
// in the explosion are also destroyed if necessary
void createProjectileExplosion(iwEnv *e, projectileEntity *projectile, const bool initalProjectile) {
    if (projectile->needsToBeDestroyed) {
        return;
    }
    projectile->needsToBeDestroyed = true;
    cc_array_add(e->explodingProjectiles, projectile);

    b2ExplosionDef explosion;
    weaponExplosion(projectile->weaponInfo->type, &explosion);
    explosion.position = projectile->pos;
    explosion.maskBits = FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE;
    droneEntity *parentDrone = safe_array_get_at(e->drones, projectile->droneIdx);
    createExplosion(e, parentDrone, projectile, &explosion);

    if (e->client != NULL) {
        explosionInfo *explInfo = fastCalloc(1, sizeof(explosionInfo));
        explInfo->def = explosion;
        explInfo->renderSteps = UINT16_MAX;
        explInfo->sparkSeed = rand_r(&e->renderRng);
        cc_array_add(e->explosions, explInfo);
    }
    if (!initalProjectile) {
        return;
    }

    // if we're not destroying the projectiles now, we need to remove the initial projectile
    // from the list of exploding projectiles so it's not destroyed twice
    const enum cc_stat res = cc_array_remove_fast(e->explodingProjectiles, projectile, NULL);
    MAYBE_UNUSED(res);
    ASSERT(res == CC_OK);
}

// ensures projectiles don't slow down when an impulse is applied to them
void fixProjectileSpeed(projectileEntity *projectile) {
    b2Vec2 newVel = b2Body_GetLinearVelocity(projectile->bodyID);
    float newSpeed = b2Length(newVel);
    if (newSpeed < projectile->lastSpeed) {
        newSpeed = projectile->lastSpeed;
        newVel = b2MulSV(newSpeed, b2Normalize(newVel));
        b2Body_SetLinearVelocity(projectile->bodyID, newVel);
    }

    projectile->velocity = newVel;
    projectile->lastSpeed = newSpeed;
    projectile->speed = newSpeed;
}

#define MAX_WALL_HITS 24

typedef struct wallBurstImpulse {
    float distance;
    b2Vec2 direction;
    float magnitude;
    uint16_t wallCellIdx;
} wallBurstImpulse;

typedef struct explosionCtx {
    iwEnv *e;
    const bool isBurst;
    droneEntity *parentDrone;
    const projectileEntity *projectile;
    const b2ExplosionDef *def;

    wallBurstImpulse wallImpulses[MAX_WALL_HITS];
    int8_t closestWallIdx;
    uint8_t wallsHit;
} explosionCtx;

// utility projectiles aren't shots, so they're kept out of the shot stats;
// hits with them are tracked separately in utilityHits
static inline void recordShotHit(iwEnv *e, const uint8_t shooterIdx, const enum weaponType type) {
    if (weaponIsUtility(type)) {
        e->analysisStats[shooterIdx].utilityHits[weaponUtility(type)]++;
        return;
    }
    e->stats[shooterIdx].shotsHit[type]++;
    e->stats[shooterIdx].totalShotsHit++;
}

static inline void recordShotTaken(iwEnv *e, const uint8_t droneIdx, const enum weaponType type) {
    if (weaponIsUtility(type)) {
        return;
    }
    e->stats[droneIdx].shotsTaken[type]++;
    e->stats[droneIdx].totalShotsTaken++;
}

static inline void recordOwnShotTaken(iwEnv *e, const uint8_t droneIdx, const enum weaponType type) {
    if (weaponIsUtility(type)) {
        return;
    }
    e->stats[droneIdx].ownShotsTaken[type]++;
    e->stats[droneIdx].totalOwnShotsTaken++;
}

static inline void recordShotDistance(iwEnv *e, const uint8_t droneIdx, const enum weaponType type, const float distance) {
    if (weaponIsUtility(type)) {
        return;
    }
    e->stats[droneIdx].shotDistances[type] += distance;
    e->stats[droneIdx].totalShotDistances += distance;
}

const float COS_85_DEGREES = 0.087155743f;

// b2World_Explode doesn't support filtering on shapes of the same category,
// so we have to do it manually
// mostly copied from box2d/src/world.c
bool explodeCallback(b2ShapeId shapeID, void *context) {
    if (!b2Shape_IsValid(shapeID) || b2Shape_IsSensor(shapeID)) {
        return true;
    }

    explosionCtx *ctx = context;
    entity *entity = b2Shape_GetUserData(shapeID);

    // allow only a single burst to hit an entity one time
    if (ctx->isBurst && ctx->parentDrone != NULL) {
        const uint16_t lastBurstHit = entity->lastBurstHit[ctx->parentDrone->idx];
        if (lastBurstHit == ctx->parentDrone->burstSerial) {
            return true;
        }
    }

    projectileEntity *projectile = NULL;
    droneEntity *drone = NULL;
    wallEntity *wall = NULL;
    bool isStaticWall = false;
    b2Transform transform;

    switch (entity->type) {
    case PROJECTILE_ENTITY:
        // don't explode the parent projectile
        projectile = entity->entity;
        if (ctx->projectile != NULL && (ctx->projectile == projectile || projectile->needsToBeDestroyed)) {
            return true;
        }
        transform.p = projectile->pos;
        transform.q = b2Rot_identity;
        break;
    case DRONE_ENTITY:
        drone = entity->entity;
        // the explosion shouldn't affect the parent drone if this is a burst
        if (drone->idx == ctx->parentDrone->idx && ctx->isBurst) {
            return true;
        }
        transform.p = drone->pos;
        transform.q = b2Rot_identity;
        break;
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        wall = entity->entity;
        isStaticWall = !wall->isFloating;
        // normal explosions don't affect static walls
        if (!ctx->isBurst && isStaticWall) {
            return true;
        }
        transform.p = wall->pos;
        transform.q = wall->rot;
        break;
    default:
        ERRORF("invalid entity type %d to explode", entity->type);
    }

    // find the closest point from the entity to the explosion center
    const b2BodyId bodyID = b2Shape_GetBody(shapeID);
    ASSERT(b2Body_IsValid(bodyID));

    bool isCircle = false;
    b2DistanceInput input;
    input.proxyA = makeDistanceProxy(entity, &isCircle);
    input.proxyB = b2MakeProxy(&ctx->def->position, 1, 0.0f);
    input.transformA = transform;
    input.transformB = b2Transform_identity;
    input.useRadii = isCircle;

    b2SimplexCache cache = {0};
    const b2DistanceOutput output = b2ShapeDistance(&input, &cache, NULL, 0);

    if (output.distance > ctx->def->radius) {
        return true;
    }

    // don't explode the entity if it's behind a static wall
    const bool isImplosion = ctx->def->impulsePerLength < 0.0f;
    b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE};
    if (posBehindWall(ctx->e, ctx->def->position, output.pointA, entity, filter, NULL)) {
        return true;
    }

    if (!isStaticWall && ctx->isBurst) {
        entity->lastBurstHit[ctx->parentDrone->idx] = ctx->parentDrone->burstSerial;
    }

    const b2Vec2 closestPoint = output.pointA;
    b2Vec2 distanceToEntity;
    b2Vec2 direction;
    if (isStaticWall) {
        distanceToEntity = b2Sub(ctx->def->position, closestPoint);
        direction = b2Normalize(distanceToEntity);
    } else {
        distanceToEntity = b2Sub(closestPoint, ctx->def->position);
        direction = b2Normalize(distanceToEntity);
    }
    // if the direction is zero, the magnitude cannot be calculated
    // correctly so set the direction randomly
    if (b2Equal(direction, b2Vec2_zero)) {
        direction.x = randFloat(&ctx->e->rng, -1.0f, 1.0f);
        direction.y = randFloat(&ctx->e->rng, -1.0f, 1.0f);
        direction = b2Normalize(direction);
    }

    b2Vec2 localLine = b2Vec2_zero;
    if (entityTypeIsWall(entity->type)) {
        // the localLine isn't used in perimeter calculations for circles
        localLine = b2InvRotateVector(transform.q, b2LeftPerp(direction));
    }
    float perimeter = getShapeProjectedPerimeter(shapeID, localLine);
    const float relDistance = output.distance / ctx->def->radius;
    const float scale = 1.0f - (SQUARED(relDistance) * relDistance);

    // the parent drone or projecile's velocity affects the direction
    // and magnitude of the explosion
    b2Vec2 parentVelocity;
    float parentSpeed;
    if (ctx->projectile != NULL) {
        // use the projectile's last velocity and speed if it is in
        // contact with another body, as the current velocity will be
        // the velocity after the projectile rebounds which is not
        // what we want
        if (ctx->projectile->contacts != 0) {
            parentVelocity = ctx->projectile->lastVelocity;
            parentSpeed = ctx->projectile->lastSpeed;
        } else {
            parentVelocity = ctx->projectile->velocity;
            parentSpeed = ctx->projectile->speed;
        }
        if (isImplosion) {
            parentSpeed *= -1.0f;
        }

        parentVelocity = b2MulSV(0.75f, parentVelocity);
    } else {
        parentVelocity = ctx->parentDrone->lastVelocity;
        parentSpeed = b2Length(parentVelocity);
    }
    const b2Vec2 parentDirection = b2Normalize(parentVelocity);

    // scale the parent speed by how close the movement direction of
    // the parent is to where the entity is to the parent, except if
    // we're bursting off of a wall to make it more predictable and
    // to prevent taking a log of a negative number
    if (!isStaticWall) {
        parentSpeed *= b2Dot(direction, parentDirection);
    }
    // never let the projectile's direction weaken or reverse the
    // impulse, only strengthen it
    if (ctx->projectile != NULL) {
        if (isImplosion) {
            parentSpeed = min(parentSpeed, 0.0f);
        } else {
            parentSpeed = max(parentSpeed, 0.0f);
        }
    }
    // the parent entity's velocity affects the direction of the impulse
    // depending on the speed
    const b2Vec2 baseImpulse = b2MulSV(fabsf(ctx->def->impulsePerLength), direction);
    direction = b2Normalize(b2Add(baseImpulse, parentVelocity));

    float shieldReduction = 1.0f;
    if (drone != NULL && drone->shield != NULL) {
        shieldReduction = DRONE_SHIELD_EXPLOSION_REDUCTION;
    }

    float magnitude = (ctx->def->impulsePerLength + parentSpeed) * perimeter * scale * shieldReduction;
    if (isStaticWall) {
        if (ctx->closestWallIdx == -1) {
            ctx->closestWallIdx = 0;
        } else {
            // ensure this wall faces at least 85 degrees away from the
            // closest hit wall to prevent multiple walls facing roughly
            // the same direction greatly increasing the impulse magnitude
            const float closestWallDistance = ctx->wallImpulses[ctx->closestWallIdx].distance;
            if (output.distance < closestWallDistance) {
                for (int8_t i = 0; i < ctx->wallsHit; i++) {
                    const wallBurstImpulse impulse = ctx->wallImpulses[i];
                    if (impulse.distance != output.distance && b2Dot(impulse.direction, direction) > COS_85_DEGREES) {
                        ctx->wallImpulses[i] = ctx->wallImpulses[--ctx->wallsHit];
                        i--;
                    }
                }
            }
        }

        // reduce the magnitude when pushing a drone away from a wall
        magnitude = log2f(fmaxf(magnitude, 1.0f)) * (5.0f + (25.0f * ctx->parentDrone->burstChargeUsed));

        uint8_t idx = ctx->wallsHit;
        if (ctx->wallsHit == MAX_WALL_HITS) {
            // replace the farthest wall if this one is closer
            idx = 0;
            for (uint8_t i = 1; i < ctx->wallsHit; i++) {
                if (ctx->wallImpulses[idx].distance < ctx->wallImpulses[i].distance) {
                    idx = i;
                }
            }
            if (output.distance >= ctx->wallImpulses[idx].distance) {
                return true;
            }
        } else {
            ctx->wallsHit++;
        }
        if (output.distance < ctx->wallImpulses[ctx->closestWallIdx].distance) {
            ctx->closestWallIdx = idx;
        }

        ctx->wallImpulses[idx] = (wallBurstImpulse){
            .distance = output.distance,
            .direction = direction,
            .magnitude = magnitude,
            .wallCellIdx = wall->mapCellIdx,
        };
        return true;
    }
    const b2Vec2 impulse = b2MulSV(magnitude, direction);
    ASSERT(b2IsValidVec2(impulse));

    physicsSource source = {0};
    if (ctx->isBurst) {
        source.type = BURST_PHYSICS;
    } else {
        source.type = EXPLOSION_PHYSICS;
        source.weapon = ctx->projectile->weaponInfo->type;
        source.boosted = ctx->projectile->boosted;
    }

    switch (entity->type) {
    case STANDARD_WALL_ENTITY:
    case BOUNCY_WALL_ENTITY:
    case DEATH_WALL_ENTITY:
        if (wall->isFloating) {
            applyTrackedImpulse(ctx->e, wall->bodyID, wall->physicsTracking, impulse, ctx->parentDrone->idx, source);
        } else {
            b2Body_ApplyLinearImpulse(bodyID, impulse, output.pointA, true);
        }
        wall->velocity = b2Body_GetLinearVelocity(wall->bodyID);
        break;
    case PROJECTILE_ENTITY: {
        // a burst on the frame a projectile was fired boosts it
        const float distance = b2Length(distanceToEntity);
        const bool boost = ctx->isBurst && !projectile->boosted && projectile->droneIdx == ctx->parentDrone->idx && distance <= DRONE_BURST_RADIUS_MIN + DRONE_BURST_RADIUS_BASE;
        const float speedBefore = projectile->speed;
        droneAnalysisStats *stats = &ctx->e->analysisStats[projectile->droneIdx];
        if (boost) {
            projectile->boosted = true;
            stats->boostedShotsFired[projectile->weaponInfo->type]++;
        }

        b2Vec2 projImpulse = impulse;
        if (boost) {
            projImpulse = b2MulSV(projectile->weaponInfo->burstBoostCoef, impulse);
        }
        b2Body_ApplyLinearImpulseToCenter(bodyID, projImpulse, true);
        if (!isImplosion) {
            switch (projectile->weaponInfo->type) {
            case MINE_LAUNCHER_WEAPON:
                // mine launcher projectiles explode when caught in another
                // explosion, explode this mine only once
                createProjectileExplosion(ctx->e, projectile, false);
                return true;
            case GRAPPLE_WEAPON: {
                if (!projectile->welded) {
                    break;
                }

                // attached grapples take damage when hit by an explosion
                float damage = (ctx->def->impulsePerLength + parentSpeed) * scale;
                if (ctx->isBurst) {
                    damage *= GRAPPLE_BURST_REDUCTION;
                }

                projectile->grappleHealth -= damage;
                if (projectile->grappleHealth <= 0.0f && !projectile->needsToBeDestroyed) {
                    projectile->needsToBeDestroyed = true;
                    cc_array_add(ctx->e->explodingProjectiles, projectile);
                }
                return true;
            }
            default:
                break;
            }
        }

        fixProjectileSpeed(projectile);
        if (boost) {
            stats->boostedSpeedBefore[projectile->weaponInfo->type] += speedBefore;
            stats->boostedSpeedAfter[projectile->weaponInfo->type] += projectile->speed;

            if (ctx->e->client != NULL) {
                boostSparksInfo *sparks = fastCalloc(1, sizeof(boostSparksInfo));
                sparks->pos = projectile->pos;
                sparks->direction = b2Normalize(b2Body_GetLinearVelocity(bodyID));
                sparks->weapon = projectile->weaponInfo->type;
                sparks->renderSteps = UINT16_MAX;
                sparks->sparkSeed = rand_r(&ctx->e->renderRng);
                cc_array_add(ctx->e->boostSparks, sparks);
            }
        }

        break;
    }
    case DRONE_ENTITY:
        if (drone->idx == ctx->parentDrone->idx && !ctx->isBurst) {
            drone->stepInfo.ownShotTaken = true;
            recordOwnShotTaken(ctx->e, drone->idx, ctx->projectile->weaponInfo->type);
            DEBUG_LOGF("drone %d hit itself with explosion from weapon %d", drone->idx, ctx->projectile->weaponInfo->type);
        } else if (ctx->isBurst) {
            DEBUG_LOGF("drone %d hit drone %d with burst", ctx->parentDrone->idx, drone->idx);
            ctx->e->stats[ctx->parentDrone->idx].burstsHit++;
            DEBUG_LOGF("drone %d hit by burst from drone %d", drone->idx, ctx->parentDrone->idx);
        } else {
            DEBUG_LOGF("drone %d hit drone %d with explosion from weapon %d", ctx->parentDrone->idx, drone->idx, ctx->projectile->weaponInfo->type);
            recordShotHit(ctx->e, ctx->parentDrone->idx, ctx->projectile->weaponInfo->type);
            ctx->e->analysisStats[ctx->parentDrone->idx].boostedShotsHit[ctx->projectile->weaponInfo->type] += ctx->projectile->boosted;
            DEBUG_LOGF("drone %d hit by explosion from weapon %d from drone %d", drone->idx, ctx->projectile->weaponInfo->type, ctx->parentDrone->idx);
        }

        applyTrackedImpulse(ctx->e, drone->bodyID, drone->physicsTracking, impulse, ctx->parentDrone->idx, source);
        drone->lastVelocity = drone->velocity;
        drone->velocity = b2Body_GetLinearVelocity(drone->bodyID);

        // add energy to the drone that fired the projectile that is
        // currently exploding if it hit another drone
        shieldEntity *shield = drone->shield;
        const float explosionStrength = fabsf(magnitude);
        if (!ctx->isBurst && drone->team != ctx->parentDrone->team && shield == NULL && (ctx->projectile == NULL || !weaponIsUtility(ctx->projectile->weaponInfo->type))) {
            const float energyRefill = explosionStrength * (DRONE_INV_MASS / MAX_SPEED) * EXPLOSION_ENERGY_REFILL_COEF;
            droneAddEnergy(ctx->parentDrone, energyRefill);
        } else if (shield != NULL && shield->health > 0.0f) {
            const float damage = fabsf(magnitude) * DRONE_SHIELD_HEALTH_EXPLOSION_COEF;
            DEBUG_LOGF("shield explosion damage: %f", damage);
            shield->health -= damage;
            if (shield->health <= 0.0f) {
                if (shield->drone->idx != ctx->parentDrone->idx && shield->drone->team != ctx->parentDrone->team) {
                    droneAddEnergy(ctx->parentDrone, DRONE_SHIELD_BREAK_ENERGY_REFILL);
                }
                ctx->parentDrone->stepInfo.brokeShield[shield->drone->idx] = true;
                ctx->e->stats[ctx->parentDrone->idx].shieldsBroken++;
            }
        }

        ctx->parentDrone->stepInfo.explosionHit[drone->idx] += explosionStrength;
        drone->stepInfo.explosionTaken[ctx->parentDrone->idx] += explosionStrength;

        break;
    default:
        ERRORF("unknown entity type for burst impulse %d", entity->type);
    }

    return true;
}

void applyDroneBurstImpulse(iwEnv *e, explosionCtx *ctx, const droneEntity *drone) {
    wallBurstImpulse *hitWalls = ctx->wallImpulses;
    uint8_t wallsHit = ctx->wallsHit;
    const physicsSource source = {.type = BURST_PHYSICS};

    // sort by distance
    for (int8_t i = 1; i < wallsHit; i++) {
        wallBurstImpulse key = hitWalls[i];
        int j = i - 1;

        while (j >= 0 && hitWalls[j].distance > key.distance) {
            hitWalls[j + 1] = hitWalls[j];
            j = j - 1;
        }

        hitWalls[j + 1] = key;
    }

    for (uint8_t i = 0; i < wallsHit; i++) {
        const wallBurstImpulse impulseA = hitWalls[i];
        if (impulseA.distance == FLT_MAX) {
            continue;
        }
        for (uint8_t j = 0; j < wallsHit; j++) {
            if (j == i) {
                continue;
            }
            wallBurstImpulse impulseB = hitWalls[j];
            if (impulseB.distance == FLT_MAX) {
                continue;
            }
            const uint16_t cellIdxDiff = abs(impulseB.wallCellIdx - impulseA.wallCellIdx);
            if (cellIdxDiff == e->map->columns) {
                hitWalls[j].distance = FLT_MAX;
                continue;
            }
            if (cellIdxDiff == 1) {
                int8_t colA = impulseA.wallCellIdx / e->map->columns;
                int8_t colB = impulseB.wallCellIdx / e->map->columns;
                if (colA == colB) {
                    hitWalls[j].distance = FLT_MAX;
                }
            }
        }
    }

    uint8_t wallsUsed = 0;
    float magnitude = 0.0f;
    b2Vec2 direction = b2Vec2_zero;
    for (uint8_t i = 0; i < wallsHit; i++) {
        const wallBurstImpulse impulse = hitWalls[i];
        if (impulse.distance == FLT_MAX) {
            continue;
        }

        wallsUsed++;
        magnitude += impulse.magnitude;
        direction = b2Add(direction, impulse.direction);
    }

    if (wallsUsed > 2) {
        DEBUG_LOG("more than 2 walls used");
    }

    DEBUG_LOGF("walls used: %d magnitude: %f final: %f", wallsUsed, magnitude, magnitude / (float)wallsUsed);
    const b2Vec2 finalImpulse = b2MulSV(magnitude / (float)wallsUsed, b2Normalize(direction));
    ASSERT(b2IsValidVec2(finalImpulse));
    applyTrackedImpulse(e, drone->bodyID, drone->physicsTracking, finalImpulse, drone->idx, source);
}

void createExplosion(iwEnv *e, droneEntity *drone, const projectileEntity *projectile, const b2ExplosionDef *def) {
    b2AABB aabb = {
        .lowerBound.x = def->position.x - def->radius,
        .lowerBound.y = def->position.y - def->radius,
        .upperBound.x = def->position.x + def->radius,
        .upperBound.y = def->position.y + def->radius,
    };

    b2QueryFilter filter = b2DefaultQueryFilter();
    filter.categoryBits = PROJECTILE_SHAPE;
    filter.maskBits = def->maskBits;

    const bool isBurst = projectile == NULL;
    explosionCtx ctx = {
        .e = e,
        .isBurst = isBurst,
        .parentDrone = drone,
        .projectile = projectile,
        .def = def,
        .wallImpulses = {{0}},
        .closestWallIdx = -1,
        .wallsHit = 0,
    };
    b2World_OverlapAABB(e->worldID, aabb, filter, explodeCallback, &ctx);

    uint8_t wallsHit = ctx.wallsHit;
    if (isBurst && wallsHit != 0) {
        applyDroneBurstImpulse(e, &ctx, drone);
        drone->burstStaticWallsHit = true;
    }
}

void destroyProjectile(iwEnv *e, projectileEntity *projectile, const bool processExplosions, const bool full) {
    if (projectile->needsToBeDestroyed && full) {
        return;
    }

    if (e->client != NULL && projectile->weaponInfo->type == BLACK_HOLE_WEAPON && BLACK_HOLE_DISTORTION_FADE_TIME > 0.0f) {
        // Keep only a visual field at the final position; gameplay pull ends now.
        explosionInfo *field = fastCalloc(1, sizeof(explosionInfo));
        field->def.position = projectile->pos;
        field->def.radius = BLACK_HOLE_PROXIMITY_RADIUS;
        field->isBlackHole = true;
        field->renderSteps = UINT16_MAX;
        cc_array_add(e->explosions, field);
    }
    // explode projectile if necessary
    if (processExplosions && projectile->weaponInfo->explosive) {
        createProjectileExplosion(e, projectile, true);
    }

    destroyEntity(e, projectile->ent);

    b2DestroyBody(projectile->bodyID);

    if (full) {
        enum cc_stat res = cc_array_remove_fast(e->projectiles, projectile, NULL);
        MAYBE_UNUSED(res);
        ASSERT(res == CC_OK);
    }

    recordShotDistance(e, projectile->droneIdx, projectile->weaponInfo->type, projectile->distance);

    if (projectile->entsInBlackHole != NULL) {
        for (uint8_t i = 0; i < cc_array_size(projectile->entsInBlackHole); i++) {
            entityID *id = safe_array_get_at(projectile->entsInBlackHole, i);
            fastFree(id);
        }
        cc_array_destroy(projectile->entsInBlackHole);
    }
    if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON) {
        e->minesCreated[projectile->droneIdx]--;
        droneEntity *drone = safe_array_get_at(e->drones, projectile->droneIdx);
        if (drone->mine == projectile) {
            drone->mine = NULL;
        }
    } else if (projectile->weaponInfo->type == GRAPPLE_WEAPON) {
        droneEntity *drone = safe_array_get_at(e->drones, projectile->droneIdx);
        if (drone->grapple == projectile) {
            drone->grapple = NULL;
        }
    }

    if (projectile->trailPoints != NULL) {
        fastFree(projectile->trailPoints);
    }

    fastFree(projectile);
}

// destroy projectiles that were caught in an explosion; projectiles
// can't be destroyed in explodeCallback because box2d assumes all shapes
// and bodies are valid for the lifetime of an AABB query
static inline void destroyExplodedProjectiles(iwEnv *e) {
    if (cc_array_size(e->explodingProjectiles) == 0) {
        return;
    }

    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->explodingProjectiles);
    projectileEntity *projectile;
    while (cc_array_iter_next(&iter, (void **)&projectile) != CC_ITER_END) {
        destroyProjectile(e, projectile, false, false);
        const enum cc_stat res = cc_array_remove_fast(e->projectiles, projectile, NULL);
        MAYBE_UNUSED(res);
        ASSERT(res == CC_OK);
    }
    cc_array_remove_all(e->explodingProjectiles);
}

void createSuddenDeathWalls(iwEnv *e, const b2Vec2 startPos, const b2Vec2 size) {
    int16_t endIdx;
    uint8_t indexIncrement;
    if (size.y == WALL_THICKNESS) {
        // horizontal walls
        const b2Vec2 endPos = (b2Vec2){.x = startPos.x + size.x, .y = startPos.y};
        endIdx = entityPosToCellIdx(e, endPos);
        if (endIdx == -1) {
            ERRORF("invalid position for sudden death wall: (%f, %f)", endPos.x, endPos.y);
        }
        indexIncrement = 1;
    } else {
        // vertical walls
        const b2Vec2 endPos = (b2Vec2){.x = startPos.x, .y = startPos.y + size.y};
        endIdx = entityPosToCellIdx(e, endPos);
        if (endIdx == -1) {
            ERRORF("invalid position for sudden death wall: (%f, %f)", endPos.x, endPos.y);
        }
        indexIncrement = e->map->columns;
    }
    const int16_t startIdx = entityPosToCellIdx(e, startPos);
    if (startIdx == -1) {
        ERRORF("invalid position for sudden death wall: (%f, %f)", startPos.x, startPos.y);
    }
    for (uint16_t i = startIdx; i <= endIdx; i += indexIncrement) {
        mapCell *cell = (&e->cells[i]);
        if (cell->ent != NULL) {
            if (cell->ent->type == WEAPON_PICKUP_ENTITY) {
                weaponPickupEntity *pickup = cell->ent->entity;
                disableWeaponPickup(e, pickup);
            } else {
                continue;
            }
        }
        entity *ent = createWall(e, cell->pos, WALL_THICKNESS, WALL_THICKNESS, i, DEATH_WALL_ENTITY, false);
        cell->ent = ent;
    }
}

typedef struct overlapCircleCtx {
    const enum entityType *targetType;
    bool overlaps;
} overlapCircleCtx;

bool isOverlappingCircleCallback(b2ShapeId shapeID, void *context) {
    if (!b2Shape_IsValid(shapeID)) {
        return true;
    }

    overlapCircleCtx *ctx = context;
    const entity *overlappingEnt = b2Shape_GetUserData(shapeID);
    if (ctx->targetType != NULL && overlappingEnt->type != *ctx->targetType) {
        return true;
    }

    ctx->overlaps = true;
    return false;
}

bool isOverlappingCircle(const iwEnv *e, const b2Vec2 startPos, const float radius, const b2QueryFilter filter, const enum entityType *targetType) {
    const b2ShapeProxy cirProxy = b2MakeProxy(&startPos, 1, radius);
    overlapCircleCtx ctx = {
        .targetType = targetType,
        .overlaps = false,
    };
    b2World_OverlapShape(e->worldID, &cirProxy, filter, isOverlappingCircleCallback, &ctx);
    return ctx.overlaps;
}

void handleSuddenDeath(iwEnv *e) {
    ASSERT(e->suddenDeathSteps == 0);
    if (e->suddenDeathWallCounter >= e->map->maxSuddenDeathWalls) {
        return;
    }

    // create new walls that will close in on the arena
    e->suddenDeathWallCounter++;
    e->suddenDeathWallsPlaced = true;

    const float leftX = (e->suddenDeathWallCounter - 1) * WALL_THICKNESS;
    const float yOffset = (WALL_THICKNESS * (e->suddenDeathWallCounter - 1)) + (WALL_THICKNESS / 2);
    const float xWidth = WALL_THICKNESS * (e->map->columns - (e->suddenDeathWallCounter * 2) - 1);

    // top walls
    createSuddenDeathWalls(
        e,
        (b2Vec2){
            .x = e->map->bounds.min.x + leftX,
            .y = e->map->bounds.min.y + yOffset,
        },
        (b2Vec2){
            .x = xWidth,
            .y = WALL_THICKNESS,
        }
    );
    // bottom walls
    createSuddenDeathWalls(
        e,
        (b2Vec2){
            .x = e->map->bounds.min.x + leftX,
            .y = e->map->bounds.max.y - yOffset,
        },
        (b2Vec2){
            .x = xWidth,
            .y = WALL_THICKNESS,
        }
    );
    // left walls
    createSuddenDeathWalls(
        e,
        (b2Vec2){
            .x = e->map->bounds.min.x + leftX,
            .y = e->map->bounds.min.y + (e->suddenDeathWallCounter * WALL_THICKNESS),
        },
        (b2Vec2){
            .x = WALL_THICKNESS,
            .y = WALL_THICKNESS * (e->map->rows - (e->suddenDeathWallCounter * 2) - 2),
        }
    );
    // right walls
    createSuddenDeathWalls(
        e,
        (b2Vec2){
            .x = e->map->bounds.min.x + ((e->map->columns - e->suddenDeathWallCounter - 2) * WALL_THICKNESS),
            .y = e->map->bounds.min.y + (e->suddenDeathWallCounter * WALL_THICKNESS),
        },
        (b2Vec2){
            .x = WALL_THICKNESS,
            .y = WALL_THICKNESS * (e->map->rows - (e->suddenDeathWallCounter * 2) - 2),
        }
    );

    // mark drones as dead if they touch a newly placed wall
    const enum entityType deathWallType = DEATH_WALL_ENTITY;
    for (uint8_t i = 0; i < e->numDrones; i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        const b2QueryFilter filter = {
            .categoryBits = DRONE_SHAPE,
            .maskBits = WALL_SHAPE,
        };
        if (isOverlappingCircle(e, drone->pos, DRONE_RADIUS, filter, &deathWallType)) {
            killDrone(e, drone, NULL);
        }
    }

    // make floating walls static bodies if they are now overlapping with
    // a newly placed wall, but destroy them if they are fully inside a wall
    CC_ArrayIter floatingWallIter;
    cc_array_iter_init(&floatingWallIter, e->floatingWalls);
    wallEntity *wall;
    while (cc_array_iter_next(&floatingWallIter, (void **)&wall) != CC_ITER_END) {
        const mapCell *cell = (&e->cells[wall->mapCellIdx]);
        if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
            // floating wall is overlapping with a wall, destroy it
            const enum cc_stat res = cc_array_iter_remove_fast(&floatingWallIter, NULL);
            MAYBE_UNUSED(res);
            ASSERT(res == CC_OK);

            const b2Vec2 wallPos = wall->pos;
            MAYBE_UNUSED(wallPos);
            destroyGrapplesAttachedTo(e, wall->ent);
            destroyWall(e, wall, false);
            DEBUG_LOGF("destroyed floating wall at %f, %f", wallPos.x, wallPos.y);
            continue;
        }
    }

    // detroy all projectiles that are now overlapping with a newly placed wall
    CC_ArrayIter projectileIter;
    cc_array_iter_init(&projectileIter, e->projectiles);
    projectileEntity *projectile;
    while (cc_array_iter_next(&projectileIter, (void **)&projectile) != CC_ITER_END) {
        const mapCell *cell = (&e->cells[projectile->mapCellIdx]);
        if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
            cc_array_iter_remove_fast(&projectileIter, NULL);
            destroyProjectile(e, projectile, false, false);
        }
    }
}

void droneEmptyEnergy(iwEnv *e, droneEntity *drone) {
    drone->energyFullyDepleted = true;
    drone->energyFullyDepletedThisStep = true;
    drone->stepInfo.emptiedEnergy = true;
    drone->energyRefillWait = DRONE_ENERGY_REFILL_EMPTY_WAIT;

    e->stats[drone->idx].energyEmptied++;
}

void droneMove(const iwEnv *e, droneEntity *drone, b2Vec2 direction) {
    ASSERT_VEC_BOUNDED(direction);

    // if energy is fully depleted halve movement until energy starts
    // to refill again
    if (drone->energyFullyDepleted && drone->energyRefillWait != 0.0f) {
        direction = b2MulSV(0.5f, direction);
    }
    const b2Vec2 force = b2MulSV(DRONE_MOVE_MAGNITUDE, direction);
    const physicsSource source = {.type = MOVEMENT_PHYSICS};
    applyTrackedForce(e, drone->bodyID, drone->physicsTracking, force, drone->idx, source);
}

// records how far the aim is from the bearing to the nearest live enemy
static void recordAimError(iwEnv *e, droneEntity *drone, const b2Vec2 normAim) {
    float bestDist = FLT_MAX;
    b2Vec2 bearing = b2Vec2_zero;
    for (uint8_t i = 0; i < e->numDrones; i++) {
        const droneEntity *other = safe_array_get_at(e->drones, i);
        if (other->idx == drone->idx || other->dead || other->team == drone->team) {
            continue;
        }
        const float dist = b2DistanceSquared(drone->pos, other->pos);
        if (dist < bestDist) {
            bestDist = dist;
            bearing = b2Sub(other->pos, drone->pos);
        }
    }
    if (bestDist == FLT_MAX) {
        return;
    }
    const float err = fabsf(signedAngle(normAim, bearing));
    drone->stepInfo.aimAlignment += cosf(err);
    droneStats *stats = &e->stats[drone->idx];
    stats->totalAimError += err;
    stats->shotsAimed10 += err < 10.0f * PI / 180.0f;
    stats->shotsAimed30 += err < 30.0f * PI / 180.0f;
    stats->shotsAimedAway += err > 150.0f * PI / 180.0f;
}

void droneChargeEnergyDrain(const iwEnv *e, droneEntity *drone, const float holdTime) {
    const float drainRate = DRONE_CHARGE_HOLD_DRAIN_RATE * (1.0f + holdTime);
    drone->energyLeft = max(drone->energyLeft - (drainRate * e->deltaTime), 0.0f);
    // don't set the refill wait when depleted as that would
    // block charging and cancel this charge before it drains
    if (!drone->energyFullyDepleted) {
        drone->energyRefillWait = max(drone->energyRefillWait, DRONE_ENERGY_REFILL_WAIT);
    }

    if (drone->energyLeft == 0.0f) {
        droneEmptyCharge(drone);
        if (!drone->chargingBurst) {
            droneEmptyEnergy(e, drone);
        }
    }
}

// fires a weapon's projectiles along aim with recoil; used by both weapons
// and utilities
static void droneFireProjectiles(iwEnv *e, droneEntity *drone, const weaponInformation *weaponInfo, const uint16_t heat, const b2Vec2 aim) {
    b2Vec2 normAim = drone->lastAim;
    if (!b2Equal(aim, b2Vec2_zero)) {
        normAim = b2Normalize(aim);
    }
    ASSERT_VEC_NORMALIZED(normAim);
    const b2Vec2 recoil = b2MulSV(-weaponInfo->recoilMagnitude, normAim);
    const physicsSource source = {.type = RECOIL_PHYSICS, .weapon = weaponInfo->type};
    applyTrackedImpulse(e, drone->bodyID, drone->physicsTracking, recoil, drone->idx, source);

    for (int i = 0; i < weaponInfo->numProjectiles; i++) {
        const b2Vec2 projAim = weaponAdjustAim(&e->rng, weaponInfo->type, heat, normAim, i);
        createProjectile(e, drone, weaponInfo, projAim);
    }
}

void droneShoot(iwEnv *e, droneEntity *drone, const b2Vec2 aim, const bool triggerDown) {
    drone->shotThisStep = true;

    weaponSlot *weapon = &drone->weapon;
    if (weapon->cooldown > 0.0f || weapon->ammo == 0) {
        return;
    }
    const bool weaponNeedsCharge = weapon->weaponInfo->maxCharge != 0.0f;
    if (weaponNeedsCharge) {
        if (triggerDown) {
            // charges cannot be held with no energy, and can't be started
            // when energy is depleted
            if (drone->energyFullyDepleted && (drone->energyRefillWait != 0.0f || drone->chargingBurst)) {
                droneEmptyCharge(drone);
                return;
            }

            weapon->charge = min(weapon->charge + e->deltaTime, weapon->weaponInfo->maxCharge);
        } else if (weapon->charge < weapon->weaponInfo->maxCharge) {
            droneEmptyCharge(drone);
        }
    }
    // if the weapon needs to be charged, only fire the weapon if it's
    // fully charged and the agent released the trigger
    if (weaponNeedsCharge && (triggerDown || weapon->charge < weapon->weaponInfo->maxCharge)) {
        return;
    }

    if (weapon->ammo != INFINITE) {
        weapon->ammo--;
    }
    weapon->cooldown += weapon->weaponInfo->cooldown;
    droneEmptyCharge(drone);

    if (!b2Equal(aim, b2Vec2_zero)) {
        recordAimError(e, drone, b2Normalize(aim));
    } else {
        recordAimError(e, drone, drone->lastAim);
    }
    droneFireProjectiles(e, drone, weapon->weaponInfo, weapon->heat, aim);
    drone->stepInfo.firedShot = true;
    // only weapon shots are counted, utility projectiles aren't
    e->stats[drone->idx].shotsFired[weapon->weaponInfo->type] += weapon->weaponInfo->numProjectiles;
    e->stats[drone->idx].totalShotsFired += weapon->weaponInfo->numProjectiles;

    // heat builds up after the shot so it only affects the shots after it
    if (weapon->weaponInfo->type == MACHINEGUN_WEAPON) {
        weapon->heat += MACHINEGUN_HEAT_BUILDUP;
    }

    if (weapon->ammo == 0) {
        droneUnsetWeapon(drone);
    }
}

// sets off all of a drone's mines that are stuck to walls
static void droneDetonateMines(iwEnv *e, const droneEntity *drone) {
    e->analysisStats[drone->idx].minesDetonated++;
    CC_ArrayIter projIter;
    cc_array_iter_init(&projIter, e->projectiles);
    projectileEntity *projectile;
    while (cc_array_iter_next(&projIter, (void **)&projectile) != CC_ITER_END) {
        if (projectile->needsToBeDestroyed || projectile->droneIdx != drone->idx || projectile->weaponInfo->type != MINE_LAUNCHER_WEAPON || !projectile->welded) {
            continue;
        }

        // we have to destroy the projectile using the iterator so
        // we can continue to iterate correctly
        destroyProjectile(e, projectile, true, false);
        enum cc_stat res = cc_array_iter_remove_fast(&projIter, NULL);
        MAYBE_UNUSED(res);
        ASSERT(res == CC_OK);
    }
    destroyExplodedProjectiles(e);
}

// how many of the utility's projectiles are live
uint8_t droneUtilityActive(const iwEnv *e, const droneEntity *drone) {
    switch (drone->utility.utilityInfo->type) {
    case GRAPPLE_UTILITY:
        return drone->grapple != NULL;
    case MINE_UTILITY:
        return e->minesCreated[drone->idx];
    default:
        return 0;
    }
}

// the drone's live mine, or NULL; the mines utility has a max active of 1
// and placed mines are kept through utility switches, so a drone never has
// more than one
const projectileEntity *droneMine(const iwEnv *e, const droneEntity *drone) {
    MAYBE_UNUSED(e);
    const projectileEntity *mine = drone->mine;
    if (mine == NULL || mine->needsToBeDestroyed) {
        return NULL;
    }
    return mine;
}

static bool droneHasWeldedMine(const iwEnv *e, const droneEntity *drone) {
    const projectileEntity *mine = droneMine(e, drone);
    return mine != NULL && mine->welded;
}

// handles the utility button. A press uses the utility if it has ammo
// and isn't at max active, otherwise it acts on what's already out: welded
// mines are detonated. The button does nothing while the grapple is in
// flight; once it's attached the drone is tethered, holding the button
// reels it in and releasing it detaches it. A held button never repeats a
// press
void droneUtility(iwEnv *e, droneEntity *drone, const b2Vec2 aim, const bool held) {
    const bool pressed = held && !drone->utilityHeld;
    const bool released = !held && drone->utilityHeld;
    drone->utilityHeld = held;

    utilitySlot *utility = &drone->utility;
    const utilityInformation *info = utility->utilityInfo;
    if (info->type == UNSET_UTILITY) {
        return;
    }
    droneAnalysisStats *stats = &e->analysisStats[drone->idx];
    stats->utilityPresses[info->type] += pressed;

    if (info->type == GRAPPLE_UTILITY && drone->grapple != NULL) {
        if (!drone->grapple->welded) {
            return;
        }
        if (released) {
            destroyProjectile(e, drone->grapple, false, true);
        } else if (held) {
            b2DistanceJoint_SetMaxMotorForce(drone->grappleJointID, GRAPPLE_MOTOR_FORCE);
            stats->utilityPressesUseful[info->type] += pressed;
        }
        return;
    }
    if (!pressed) {
        return;
    }

    if (droneUtilityActive(e, drone) >= info->maxActive) {
        if (info->type == MINE_UTILITY && droneHasWeldedMine(e, drone)) {
            droneDetonateMines(e, drone);
            stats->utilityPressesUseful[info->type]++;
        }
        return;
    }
    if (utility->ammo == 0 || utility->cooldown > 0.0f) {
        return;
    }

    stats->utilityPressesUseful[info->type]++;
    stats->utilityAmmoSpent[info->type]++;
    utility->ammo--;
    utility->cooldown = info->projectileInfo->cooldown;
    droneFireProjectiles(e, drone, info->projectileInfo, 0, aim);
}

// regenerates utility ammo and cools down the utility
void utilityStep(const iwEnv *e, utilitySlot *utility) {
    utility->cooldown = max(utility->cooldown - e->deltaTime, 0.0f);

    const utilityInformation *info = utility->utilityInfo;
    if (utility->ammo >= info->ammo) {
        utility->regen = 0.0f;
        return;
    }
    utility->regen += e->deltaTime;
    if (utility->regen >= info->regenTime) {
        utility->regen -= info->regenTime;
        utility->ammo++;
    }
}

void droneBrake(iwEnv *e, droneEntity *drone, const bool brake) {
    if (drone->shield != NULL) {
        return;
    }
    // if the drone isn't braking or energy is fully depleted, return
    // unless the drone was braking during the last step
    if (!brake || drone->energyFullyDepleted) {
        if (drone->braking) {
            drone->braking = false;
            b2Body_SetLinearDamping(drone->bodyID, DRONE_LINEAR_DAMPING);
            if (drone->energyRefillWait == 0.0f && !drone->chargingBurst) {
                drone->energyRefillWait = DRONE_ENERGY_REFILL_WAIT;
            }
            droneTrackBrake(e, drone);

            if (e->client != NULL) {
                brakeTrailPoint *trailPoint = fastCalloc(1, sizeof(brakeTrailPoint));
                trailPoint->pos = drone->pos;
                trailPoint->lifetime = UINT16_MAX;
                trailPoint->isEnd = true;
                cc_array_add(drone->brakeTrailPoints, trailPoint);
            }
        }
        return;
    }
    ASSERT(!drone->energyFullyDepleted);

    // apply additional brake damping and decrease energy
    if (brake) {
        if (!drone->braking) {
            drone->braking = true;
            b2Body_SetLinearDamping(drone->bodyID, DRONE_LINEAR_DAMPING * DRONE_BRAKE_DAMPING_COEF);
            droneTrackBrake(e, drone);
        }
        drone->energyLeft = max(drone->energyLeft - (DRONE_BRAKE_DRAIN_RATE * e->deltaTime), 0.0f);
        e->stats[drone->idx].brakeTime += e->deltaTime;
    }

    // if energy is empty but burst is being charged, let burst functions
    // handle energy refill
    if (drone->energyLeft == 0.0f && !drone->chargingBurst) {
        droneEmptyEnergy(e, drone);
    }

    if (e->client != NULL) {
        brakeTrailPoint *trailPoint = fastCalloc(1, sizeof(brakeTrailPoint));
        trailPoint->pos = drone->pos;
        trailPoint->lifetime = UINT16_MAX;
        cc_array_add(drone->brakeTrailPoints, trailPoint);
    }
}

void droneChargeBurst(iwEnv *e, droneEntity *drone) {
    if (drone->energyFullyDepleted || drone->burstCooldown != 0.0f || drone->shield != NULL || (!drone->chargingBurst && drone->energyLeft < DRONE_BURST_BASE_COST)) {
        return;
    }

    // take energy and put it into burst charge
    if (drone->chargingBurst) {
        drone->burstCharge = min(drone->burstCharge + (DRONE_BURST_CHARGE_RATE * e->deltaTime), DRONE_ENERGY_MAX);
        drone->energyLeft = max(drone->energyLeft - (DRONE_BURST_CHARGE_RATE * e->deltaTime), 0.0f);
    } else {
        drone->burstCharge = min(drone->burstCharge + DRONE_BURST_BASE_COST, DRONE_ENERGY_MAX);
        drone->energyLeft = max(drone->energyLeft - DRONE_BURST_BASE_COST, 0.0f);
        drone->chargingBurst = true;
    }

    // droneEmptyEnergy is not used here as emptying energy while charging
    // a burst shouldn't apply the penalties until after the burst is released
    if (drone->energyLeft == 0.0f) {
        drone->energyFullyDepleted = true;
    }
}

void droneBurst(iwEnv *e, droneEntity *drone) {
    if (!drone->chargingBurst) {
        return;
    }

    drone->burstSerial = ++e->burstSerial;
    drone->burstChargeUsed = drone->burstCharge;
    drone->burstRadius = (DRONE_BURST_RADIUS_BASE * drone->burstCharge) + DRONE_BURST_RADIUS_MIN;
    drone->burstImpulse = (DRONE_BURST_IMPACT_BASE * drone->burstCharge) + DRONE_BURST_IMPACT_MIN;
    drone->burstStaticWallsHit = false;
    b2ExplosionDef explosion = {
        .position = drone->pos,
        .radius = drone->burstRadius,
        .falloff = 0.0f,
        .impulsePerLength = drone->burstImpulse,
        .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE,
    };
    createExplosion(e, drone, NULL, &explosion);
    destroyExplodedProjectiles(e);

    drone->burstFramesLeft = (uint8_t)roundf(DRONE_BURST_ACTIVE_TIME * e->frameRate) + 1;
    drone->chargingBurst = false;
    drone->burstCharge = 0.0f;
    drone->burstCooldown = DRONE_BURST_COOLDOWN;
    if (drone->energyLeft == 0.0f) {
        droneEmptyEnergy(e, drone);
    } else {
        drone->energyRefillWait = DRONE_ENERGY_REFILL_WAIT;
    }
    e->stats[drone->idx].totalBursts++;

    if (e->client != NULL) {
        explosionInfo *explInfo = fastCalloc(1, sizeof(explosionInfo));
        explInfo->def = explosion;
        explInfo->isBurst = true;
        explInfo->droneIdx = drone->idx;
        explInfo->renderSteps = UINT16_MAX;
        explInfo->sparkSeed = rand_r(&e->renderRng);
        cc_array_add(e->explosions, explInfo);
    }
}

void weaponStep(const iwEnv *e, weaponSlot *weapon) {
    if (weapon->cooldown > 0.0f) {
        weapon->cooldown = max(weapon->cooldown - e->deltaTime, -e->deltaTime);
    } else {
        weapon->cooldown = 0.0f;
        weapon->heat = max(weapon->heat - (MACHINEGUN_HEAT_COOL * e->deltaTime), 0);
    }
}

// update drone state, respawn the drone if necessary; false is returned
// if no position could be found to respawn the drone at, and true otherwise
bool droneStep(iwEnv *e, droneEntity *drone) {
    if (drone->dead) {
        drone->respawnWait -= e->deltaTime;
        if (drone->respawnWait <= 0.0f) {
            const bool foundPos = respawnDrone(e, drone);
            if (!foundPos) {
                return false;
            }
        }
        return true;
    }

    // burst until the burst time is up
    if (drone->burstFramesLeft > 0 && --drone->burstFramesLeft > 0) {
        uint64_t mask = FLOATING_WALL_SHAPE | PROJECTILE_SHAPE | DRONE_SHAPE;
        if (!drone->burstStaticWallsHit) {
            mask |= WALL_SHAPE;
        }
        b2ExplosionDef explosion = {
            .position = drone->pos,
            .radius = drone->burstRadius,
            .falloff = 0.0f,
            .impulsePerLength = drone->burstImpulse,
            .maskBits = mask,
        };
        createExplosion(e, drone, NULL, &explosion);
        destroyExplodedProjectiles(e);
    }

    // manage weapon charge and heat, and utility ammo
    weaponStep(e, &drone->weapon);
    utilityStep(e, &drone->utility);

    // slowly drain energy if a charge is being held for a long time
    weaponSlot *active = &drone->weapon;
    if (active->weaponInfo->maxCharge > 0.0f && active->charge == active->weaponInfo->maxCharge) {
        active->fullChargeHoldTime += e->deltaTime;
        if (active->fullChargeHoldTime > DRONE_CHARGE_FULL_DRAIN_WAIT) {
            const float holdTime = active->fullChargeHoldTime - DRONE_CHARGE_FULL_DRAIN_WAIT;
            droneChargeEnergyDrain(e, drone, holdTime);
        }
    }

    if (!drone->shotThisStep) {
        // if the trigger is released while charging, the charge is
        // emptied
        if (drone->weapon.charge < drone->weapon.weaponInfo->maxCharge) {
            droneEmptyCharge(drone);
        }
    } else {
        drone->shotThisStep = false;
    }
    ASSERT(!drone->shotThisStep);

    if (drone->grapple != NULL && drone->grapple->welded) {
        if (drone->utilityHeld) {
            // shorten the grapple line as it's being reeled in
            const float curLen = b2DistanceJoint_GetCurrentLength(drone->grappleJointID);
            drone->grappleLength = fminf(curLen + GRAPPLE_SLACK, drone->grappleLength);
            b2DistanceJoint_SetLength(drone->grappleJointID, drone->grappleLength);
            b2DistanceJoint_SetLengthRange(drone->grappleJointID, 0.0f, drone->grappleLength + GRAPPLE_MAX_STRETCH);
        } else {
            b2DistanceJoint_SetMaxMotorForce(drone->grappleJointID, 0.0f);
        }
    }

    if (drone->grapple != NULL && drone->grapple->welded) {
        const bool stretched = b2DistanceJoint_GetCurrentLength(drone->grappleJointID) > drone->grappleLength;
        const bool springOn = b2DistanceJoint_GetSpringHertz(drone->grappleJointID) > 0.0f;
        if (stretched != springOn) {
            b2DistanceJoint_SetSpringHertz(drone->grappleJointID, stretched ? GRAPPLE_SPRING_HERTZ : 0.0f);
            if (!stretched) {
                // prevent yo-yo'ing/oscillating when the spring is activated
                b2DistanceJoint_SetLength(drone->grappleJointID, drone->grappleLength);
            }
        }

        // track applied force to the owning drone and attached entity
        const b2Vec2 pullForce = b2Joint_GetConstraintForce(drone->grappleJointID);
        const physicsSource source = {.type = GRAPPLE_PHYSICS, .weapon = GRAPPLE_WEAPON};

        CC_Array *attachedPhysics = NULL;
        const entity *attachedEnt = drone->grapple->grappledEnt;
        if (entityTypeIsWall(attachedEnt->type)) {
            const wallEntity *wall = attachedEnt->entity;
            if (wall->isFloating) {
                attachedPhysics = wall->physicsTracking;
            }
        } else if (attachedEnt->type == DRONE_ENTITY) {
            const droneEntity *attachedDrone = attachedEnt->entity;
            attachedPhysics = attachedDrone->physicsTracking;
            markAssistSetup(e, drone, attachedDrone, ASSIST_GRAPPLE);
        }

        // impulses and forces applied to static walls are not tracked
        if (attachedPhysics != NULL) {
            trackForce(e, attachedPhysics, pullForce, drone->idx, source);
        }
        trackForce(e, drone->physicsTracking, b2Neg(pullForce), drone->idx, source);
    }

    // manage drone energy
    if (drone->burstCooldown != 0.0f) {
        drone->burstCooldown = max(drone->burstCooldown - e->deltaTime, 0.0f);
    }
    if (drone->energyFullyDepletedThisStep) {
        drone->energyFullyDepletedThisStep = false;
    } else if (drone->energyRefillWait != 0.0f) {
        drone->energyRefillWait = max(drone->energyRefillWait - e->deltaTime, 0.0f);
    } else if (drone->energyLeft != DRONE_ENERGY_MAX && !drone->chargingBurst && drone->weapon.fullChargeHoldTime <= DRONE_CHARGE_FULL_DRAIN_WAIT) {
        // don't start recharging energy until the burst charge is used
        // or while a held weapon charge is draining energy
        drone->energyLeft = min(drone->energyLeft + (DRONE_ENERGY_REFILL_RATE * e->deltaTime), DRONE_ENERGY_MAX);
    }
    if (drone->energyLeft == DRONE_ENERGY_MAX) {
        drone->energyFullyDepleted = false;
    }

    const float distance = b2Distance(drone->lastPos, drone->pos);
    e->stats[drone->idx].distanceTraveled += distance;
    drone->lastPos = drone->pos;

    shieldEntity *shield = drone->shield;
    if (shield != NULL) {
        shield->duration -= e->deltaTime;
        if (shield->duration <= 0.0f || shield->health <= 0.0f) {
            destroyDroneShield(e, shield, true);
        }
    }

    return true;
}

void handleBlackHolePull(iwEnv *e, projectileEntity *projectile) {
    ASSERT(projectile->weaponInfo->type == BLACK_HOLE_WEAPON);

    CC_ArrayIter entIter;
    cc_array_iter_init(&entIter, projectile->entsInBlackHole);
    entityID *id;
    while (cc_array_iter_next(&entIter, (void **)&id) != CC_ITER_END) {
        // check if the entity is still valid
        const entity *ent = getEntityByID(e, id);
        if (ent == NULL) {
            fastFree(id);
            enum cc_stat res = cc_array_iter_remove_fast(&entIter, NULL);
            MAYBE_UNUSED(res);
            ASSERT(res == CC_OK);
            continue;
        }
        const b2DistanceOutput output = closestPoint(projectile->ent, ent);
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE};
        if (posBehindWall(e, projectile->pos, output.pointB, ent, filter, NULL)) {
            continue;
        }

        b2BodyId bodyID;
        b2ShapeId shapeID;
        b2Rot rot;

        wallEntity *wall;
        droneEntity *drone;
        projectileEntity *proj;
        bool hasShield = false;

        switch (ent->type) {
        case STANDARD_WALL_ENTITY:
        case BOUNCY_WALL_ENTITY:
        case DEATH_WALL_ENTITY:
            wall = ent->entity;
            shapeID = wall->shapeID;
            rot = wall->rot;
            break;
        case DRONE_ENTITY:
            drone = ent->entity;
            shapeID = drone->shapeID;
            hasShield = drone->shield != NULL;
            break;
        case PROJECTILE_ENTITY:
            proj = ent->entity;
            bodyID = proj->bodyID;
            shapeID = proj->shapeID;
            break;
        default:
            ERRORF("unknown entity type %d for black hole suck", ent->type);
        }

        b2Vec2 direction = b2Normalize(b2Sub(output.pointB, projectile->pos));

        b2Vec2 localLine = b2Vec2_zero;
        if (entityTypeIsWall(ent->type)) {
            // the localLine isn't used in perimeter calculations for circles
            localLine = b2InvRotateVector(rot, b2LeftPerp(direction));
        }
        const float perimeter = getShapeProjectedPerimeter(shapeID, localLine);

        const float scale = 1.0f - (output.distance / BLACK_HOLE_PROXIMITY_RADIUS);

        // the projecile's velocity affects the direction
        // and magnitude of the force
        b2Vec2 parentVelocity = projectile->velocity;
        float parentSpeed = projectile->speed;
        if (projectile->contacts != 0) {
            parentVelocity = projectile->lastVelocity;
            parentSpeed = projectile->lastSpeed;
        }
        const b2Vec2 parentDirection = b2Normalize(parentVelocity);
        parentSpeed *= b2Dot(direction, parentDirection);
        // the parent entity's velocity affects the direction of the force
        // depending on the speed
        const b2Vec2 baseForce = b2MulSV(-1.0f * BLACK_HOLE_PULL_MAGNITUDE, direction);
        direction = b2Normalize(b2Add(baseForce, parentVelocity));

        float magnitude = (BLACK_HOLE_PULL_MAGNITUDE + (-1.0f * parentSpeed)) * perimeter * scale;
        if (hasShield) {
            magnitude *= DRONE_SHIELD_FORCE_REDUCTION;
        }
        const b2Vec2 force = b2MulSV(magnitude, direction);
        const physicsSource source = {.type = PROJECTILE_PHYSICS, .weapon = BLACK_HOLE_WEAPON};

        if (entityTypeIsWall(ent->type)) {
            wallEntity *wall = ent->entity;
            if (wall->isFloating) {
                applyTrackedForce(e, wall->bodyID, wall->physicsTracking, force, projectile->droneIdx, source);
            } else {
                b2Body_ApplyForce(wall->bodyID, force, output.pointB, true);
            }
        } else if (ent->type == DRONE_ENTITY) {
            applyTrackedForce(e, drone->bodyID, drone->physicsTracking, force, projectile->droneIdx, source);

            droneEntity *parentDrone = safe_array_get_at(e->drones, projectile->droneIdx);
            if (projectile->droneIdx != drone->idx && parentDrone->team != drone->team) {
                // add energy to the drone that fired the projectile
                const float energyRefill = fabsf(magnitude) * e->deltaTime * (DRONE_INV_MASS / MAX_SPEED) * BLACK_HOLE_PULL_ENERGY_REFILL_COEF;
                droneAddEnergy(parentDrone, energyRefill);

                markAssistSetup(e, parentDrone, drone, ASSIST_BLACK_HOLE);
            }
        } else {
            b2Body_ApplyForceToCenter(bodyID, force, true);
        }
    }
}

void projectilesStep(iwEnv *e) {
    CC_ArrayIter projIter;
    cc_array_iter_init(&projIter, e->projectiles);
    projectileEntity *projectile;
    while (cc_array_iter_next(&projIter, (void **)&projectile) != CC_ITER_END) {
        if (projectile->needsToBeDestroyed) {
            continue;
        }
        const float maxDistance = projectile->weaponInfo->maxDistance;
        if (!projectile->welded) {
            const float distance = b2Distance(projectile->pos, projectile->lastPos);
            projectile->distance += distance;
        }

        // if a drone is in a set mine's sensor range but behind a wall,
        // we need to check until the drone leaves the sensor range if
        // it's not behind the wall anymore as we normally only check if
        // we need to explode the mine when a drone touches the sensor
        if (projectile->numDronesBehindWalls != 0) {
            bool destroyed = false;
            for (uint8_t i = 0; i < projectile->numDronesBehindWalls; i++) {
                const uint8_t droneIdx = projectile->dronesBehindWalls[i];
                const droneEntity *drone = safe_array_get_at(e->drones, droneIdx);
                // TODO: do we need the closest point here?
                const b2DistanceOutput output = closestPoint(projectile->ent, drone->ent);
                const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
                if (posBehindWall(e, projectile->pos, output.pointB, NULL, filter, NULL)) {
                    continue;
                }

                // we have to destroy the projectile using the iterator so
                // we can continue to iterate correctly
                destroyProjectile(e, projectile, true, false);
                enum cc_stat res = cc_array_iter_remove_fast(&projIter, NULL);
                MAYBE_UNUSED(res);
                ASSERT(res == CC_OK);
                destroyed = true;
                break;
            }
            if (destroyed) {
                continue;
            }
        }

        if (projectile->entsInBlackHole != NULL) {
            handleBlackHolePull(e, projectile);
        }

        if (maxDistance == INFINITE) {
            continue;
        }
        if (projectile->distance >= maxDistance) {
            // we have to destroy the projectile using the iterator so
            // we can continue to iterate correctly
            destroyProjectile(e, projectile, true, false);
            enum cc_stat res = cc_array_iter_remove_fast(&projIter, NULL);
            MAYBE_UNUSED(res);
            ASSERT(res == CC_OK);
            continue;
        }
    }

    destroyExplodedProjectiles(e);
}

// a pickup and a floating wall are both boxes, so their centers have to be
// closer than the sum of their half diagonals to touch at all
static const float PICKUP_WALL_CHECK_DISTANCE_SQUARED =
    SQUARED(((PICKUP_THICKNESS + FLOATING_WALL_THICKNESS) * 0.70710678f) / 2.0f * 2.0f);

// squared distance from a point to the axis aligned pickup box
static inline float pickupDistanceSquared(const b2Vec2 pickupPos, const b2Vec2 pos) {
    const float halfExtent = PICKUP_THICKNESS / 2.0f;
    const float dx = max(fabsf(pos.x - pickupPos.x) - halfExtent, 0.0f);
    const float dy = max(fabsf(pos.y - pickupPos.y) - halfExtent, 0.0f);
    return SQUARED(dx) + SQUARED(dy);
}

static inline bool pickupOverlapsDrone(const b2Vec2 pickupPos, const b2Vec2 dronePos) {
    return pickupDistanceSquared(pickupPos, dronePos) <= SQUARED(DRONE_RADIUS);
}

// A drone can cross a whole pickup within one physics frame, so test the
// segment it actually travelled rather than only where it ended up. Distance to
// a box is convex along a segment, so a ternary search finds the closest
// approach exactly; the bounding box reject means it almost never runs.
static bool pickupOverlapsDronePath(const b2Vec2 pickupPos, const b2Vec2 from, const b2Vec2 to) {
    const float reach = (PICKUP_THICKNESS / 2.0f) + DRONE_RADIUS;
    if (min(from.x, to.x) - pickupPos.x > reach || pickupPos.x - max(from.x, to.x) > reach ||
        min(from.y, to.y) - pickupPos.y > reach || pickupPos.y - max(from.y, to.y) > reach) {
        return false;
    }

    float lo = 0.0f;
    float hi = 1.0f;
    for (uint8_t i = 0; i < 16; i++) {
        const float third = (hi - lo) / 3.0f;
        const float m1 = lo + third;
        const float m2 = hi - third;
        if (pickupDistanceSquared(pickupPos, b2Lerp(from, to, m1)) <
            pickupDistanceSquared(pickupPos, b2Lerp(from, to, m2))) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    return pickupDistanceSquared(pickupPos, b2Lerp(from, to, 0.5f * (lo + hi))) <= SQUARED(DRONE_RADIUS);
}

// Resolves weapon pickup overlaps without box2d sensors. Pickups are a handful
// of static 3x3 boxes, so testing them against the drones and floating walls
// directly costs a few dozen float compares, where box2d's sensor pass rebuilds
// the overlap set for every sensor on every step.
void weaponPickupsOverlapStep(iwEnv *e) {
    const size_t numPickups = cc_array_size(e->pickups);
    if (numPickups == 0) {
        return;
    }

    // gather the moving entities once; the inner loops below would otherwise
    // chase the same scattered pointers for every pickup
    const size_t numFloatingWalls = cc_array_size(e->floatingWalls);
    b2Vec2 wallPositions[MAX_FLOATING_WALLS];
    wallEntity *walls[MAX_FLOATING_WALLS];
    for (size_t i = 0; i < numFloatingWalls; i++) {
        wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
        walls[i] = wall;
        wallPositions[i] = wall->pos;
    }

    b2Vec2 dronePositions[_MAX_DRONES];
    b2Vec2 droneLastPositions[_MAX_DRONES];
    uint8_t liveDrones[_MAX_DRONES];
    uint8_t numLiveDrones = 0;
    for (uint8_t i = 0; i < e->numDrones; i++) {
        const droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        dronePositions[numLiveDrones] = drone->pos;
        droneLastPositions[numLiveDrones] = drone->lastPos;
        liveDrones[numLiveDrones] = i;
        numLiveDrones++;
    }

    for (size_t i = 0; i < numPickups; i++) {
        weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
        if (pickup->respawnWait != 0.0f) {
            continue;
        }
        const b2Vec2 pickupPos = pickup->pos;

        // a pickup covered by a floating wall can't be picked up
        uint8_t floatingWallsTouching = 0;
        for (size_t j = 0; j < numFloatingWalls; j++) {
            if (b2DistanceSquared(pickupPos, wallPositions[j]) > PICKUP_WALL_CHECK_DISTANCE_SQUARED) {
                continue;
            }
            const b2DistanceOutput output = closestPoint(pickup->ent, walls[j]->ent);
            if (output.distance <= 0.0f) {
                floatingWallsTouching++;
            }
        }
        pickup->floatingWallsTouching = floatingWallsTouching;
        if (floatingWallsTouching != 0) {
            continue;
        }

        for (uint8_t j = 0; j < numLiveDrones; j++) {
            if (!pickupOverlapsDrone(pickupPos, dronePositions[j]) &&
                !pickupOverlapsDronePath(pickupPos, droneLastPositions[j], dronePositions[j])) {
                continue;
            }

            droneEntity *drone = safe_array_get_at(e->drones, liveDrones[j]);
            disableWeaponPickup(e, pickup);
            drone->stepInfo.pickedUpWeapon = true;
            if (pickup->kind == CRATE_PICKUP) {
                const bool refilled = droneRefillAmmo(drone);
                e->stats[drone->idx].cratesPickedUp++;
                e->stats[drone->idx].cratesRefilled += refilled;
                DEBUG_LOGF("drone %d picked up an ammo crate", drone->idx);
                break;
            }
            if (pickup->kind == UTILITY_PICKUP) {
                drone->stepInfo.slotFilled |= droneChangeUtility(e, drone, pickup->utility);
                e->stats[drone->idx].utilitiesPickedUp[pickup->utility]++;
                DEBUG_LOGF("drone %d picked up utility %d", drone->idx, pickup->utility);
                break;
            }
            e->analysisStats[drone->idx].pickupsFrom[drone->weapon.weaponInfo->type][pickup->weapon]++;
            drone->stepInfo.slotFilled |= droneChangeWeapon(drone, pickup->weapon);

            e->stats[drone->idx].weaponsPickedUp[pickup->weapon]++;
            e->stats[drone->idx].totalWeaponsPickedUp++;
            DEBUG_LOGF("drone %d picked up weapon %d", drone->idx, pickup->weapon);
            break;
        }
    }
}

void weaponPickupsStep(iwEnv *e) {
    CC_ArrayIter iter;
    cc_array_iter_init(&iter, e->pickups);
    weaponPickupEntity *pickup;

    // respawn weapon pickups at a random location as a random weapon type
    // once the respawn wait has elapsed
    while (cc_array_iter_next(&iter, (void **)&pickup) != CC_ITER_END) {
        if (pickup->respawnWait == 0.0f) {
            continue;
        }
        pickup->respawnWait = max(pickup->respawnWait - e->deltaTime, 0.0f);
        if (pickup->respawnWait != 0.0f) {
            continue;
        }

        b2Vec2 pos;
        if (!findOpenPos(e, WEAPON_PICKUP_SHAPE, &pos, -1)) {
            const enum cc_stat res = cc_array_iter_remove_fast(&iter, NULL);
            MAYBE_UNUSED(res);
            ASSERT(res == CC_OK);
            DEBUG_LOG("destroying weapon pickup");
            destroyWeaponPickup(e, pickup);
            continue;
        }
        pickup->pos = pos;
        randPickupType(e, pickup, false);

        const int16_t cellIdx = entityPosToCellIdx(e, pos);
        if (cellIdx == -1) {
            ERRORF("invalid position for weapon pickup spawn: (%f, %f)", pos.x, pos.y);
        }
        DEBUG_LOGF("respawned weapon pickup at cell %d (%f, %f)", cellIdx, pos.x, pos.y);
        pickup->mapCellIdx = cellIdx;
        createWeaponPickupBodyShape(e, pickup);

        mapCell *cell = (&e->cells[cellIdx]);
        cell->ent = pickup->ent;
    }
}

// only update positions and velocities of dynamic bodies if they moved
// this step
void handleBodyMoveEvents(iwEnv *e) {
    b2BodyEvents events = b2World_GetBodyEvents(e->worldID);
    for (int i = 0; i < events.moveCount; i++) {
        const b2BodyMoveEvent *event = events.moveEvents + i;
        if (!b2Body_IsValid(event->bodyId)) {
            continue;
        }
        ASSERT(b2IsValidVec2(event->transform.p));
        const b2Vec2 newPos = event->transform.p;
        entity *ent = event->userData;
        if (ent == NULL) {
            continue;
        }

        wallEntity *wall;
        projectileEntity *proj;
        droneEntity *drone;
        shieldEntity *shield;
        dronePieceEntity *piece;
        int16_t mapIdx;

        // if the new position is out of bounds, destroy the entity unless
        // a drone is out of bounds, then just kill it
        switch (ent->type) {
        case STANDARD_WALL_ENTITY:
        case BOUNCY_WALL_ENTITY:
        case DEATH_WALL_ENTITY:
            wall = ent->entity;
            mapIdx = entityPosToCellIdx(e, newPos);
            if (mapIdx == -1) {
                DEBUG_LOGF("invalid position for floating wall: (%f, %f) destroying", newPos.x, newPos.y);
                cc_array_remove_fast(e->floatingWalls, wall, NULL);
                destroyGrapplesAttachedTo(e, wall->ent);
                destroyWall(e, wall, false);
                continue;
            }
            wall->mapCellIdx = mapIdx;
            wall->pos = newPos;
            wall->rot = event->transform.q;
            wall->velocity = b2Body_GetLinearVelocity(wall->bodyID);
            break;
        case PROJECTILE_ENTITY:
            proj = ent->entity;
            mapIdx = entityPosToCellIdx(e, newPos);
            if (mapIdx == -1) {
                DEBUG_LOGF("invalid position for projectile: (%f, %f) destroying", newPos.x, newPos.y);
                destroyProjectile(e, proj, false, true);
                continue;
            }
            proj->mapCellIdx = mapIdx;
            proj->lastPos = proj->pos;
            proj->pos = newPos;
            proj->lastVelocity = proj->velocity;
            proj->velocity = b2Body_GetLinearVelocity(proj->bodyID);
            // if the projectile doesn't have damping its speed will
            // only change when colliding with a dynamic body or getting
            // hit by an explosion, and if it's currently colliding with
            // something we don't care about the current speed
            if (proj->weaponInfo->damping != 0.0f && proj->contacts == 0) {
                proj->lastSpeed = proj->speed;
                proj->speed = b2Length(proj->velocity);
            }

            if (proj->trailPoints != NULL) {
                updateTrailPoints(proj->trailPoints, proj->boosted ? MAX_BOOSTED_PROJECTLE_TRAIL_POINTS : MAX_PROJECTLE_TRAIL_POINTS, newPos);
            }
            break;
        case DRONE_ENTITY:
            drone = ent->entity;
            mapIdx = entityPosToCellIdx(e, newPos);
            if (mapIdx == -1) {
                DEBUG_LOGF("invalid position for drone: (%f, %f) killing it", newPos.x, newPos.y);
                killDrone(e, drone, NULL);
                continue;
            }
            drone->mapCellIdx = mapIdx;
            drone->lastPos = drone->pos;
            drone->pos = newPos;
            drone->lastVelocity = drone->velocity;
            drone->velocity = b2Body_GetLinearVelocity(drone->bodyID);

            if (drone->trailPoints != NULL) {
                updateTrailPoints(drone->trailPoints, MAX_DRONE_TRAIL_POINTS, newPos);
            }
            break;
        case SHIELD_ENTITY:
            shield = ent->entity;
            shield->pos = newPos;
            break;
        case DRONE_PIECE_ENTITY:
            piece = ent->entity;
            piece->pos = newPos;
            piece->rot = event->transform.q;
            break;
        default:
            ERRORF("unknown entity type for move event %d", ent->type);
        }
    }
}

bool projectileIgnoresBouncyWalls(projectileEntity *projectile) {
    switch (projectile->weaponInfo->type) {
    case ACCELERATOR_WEAPON:
        // all accelerator wall bounces are counted
    case GRAPPLE_WEAPON:
        // grapples can attach to all walls
    case MINE_LAUNCHER_WEAPON:
        // mines can attach to all walls as well
        return true;
    default:
        return false;
    }
}

void createProjectileWeldJoint(iwEnv *e, projectileEntity *projectile, const b2BodyId bodyID, const b2Rot rot, const b2ContactId contactID, const bool projIsShapeA) {
    ASSERT(b2Contact_IsValid(contactID));
    const b2Manifold manifold = b2Contact_GetData(contactID).manifold;
    ASSERT(manifold.pointCount == 1);

    b2WeldJointDef jointDef = b2DefaultWeldJointDef();
    const b2Rot projRot = b2Body_GetRotation(projectile->bodyID);
    if (projIsShapeA) {
        jointDef.base.bodyIdA = projectile->bodyID;
        jointDef.base.bodyIdB = bodyID;
        jointDef.base.localFrameA.p = b2InvRotateVector(projRot, manifold.points[0].anchorA);
        jointDef.base.localFrameA.q = b2InvMulRot(projRot, rot);
        jointDef.base.localFrameB.p = b2InvRotateVector(rot, manifold.points[0].anchorB);
        jointDef.base.localFrameB.q = b2Rot_identity;
    } else {
        jointDef.base.bodyIdA = bodyID;
        jointDef.base.bodyIdB = projectile->bodyID;
        jointDef.base.localFrameA.p = b2InvRotateVector(rot, manifold.points[0].anchorA);
        jointDef.base.localFrameA.q = b2InvMulRot(rot, projRot);
        jointDef.base.localFrameB.p = b2InvRotateVector(projRot, manifold.points[0].anchorB);
        jointDef.base.localFrameB.q = b2Rot_identity;
    }

    b2CreateWeldJoint(e->worldID, &jointDef);
    projectile->welded = true;
}

// destroy the projectile if it has traveled enough or has bounced enough
// times, and update drone stats if a drone was hit
uint8_t handleProjectileBeginContact(iwEnv *e, const entity *proj, const entity *ent, const b2ContactId contactID, const bool projIsShapeA) {
    projectileEntity *projectile = proj->entity;
    projectile->contacts++;

    bool bodyIDSet = false;
    b2BodyId bodyID = {0};
    const physicsSource source = {.type = PROJECTILE_PHYSICS, .weapon = projectile->weaponInfo->type, .boosted = projectile->boosted};

    // e (shape B in the collision) will be NULL if it's another
    // projectile that was just destroyed
    if (ent == NULL || ent->type == PROJECTILE_ENTITY) {
        // explode mines when hit by a projectile
        if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON) {
            uint8_t numDestroyed = 1;
            if (ent != NULL) {
                const projectileEntity *projectile2 = ent->entity;
                // if both entities are mines both will be destroyed
                if (projectile2->weaponInfo->type == MINE_LAUNCHER_WEAPON) {
                    numDestroyed = 2;
                }
            }
            destroyProjectile(e, projectile, true, true);
            return numDestroyed;
        }

        // always allow all other projectiles to bounce off each other
        return 0;
    } else if (entityTypeIsWall(ent->type)) {
        wallEntity *wall = ent->entity;
        bodyID = wall->bodyID;
        bodyIDSet = true;

        if (wall->isFloating && b2Contact_IsValid(contactID)) {
            const b2Manifold manifold = b2Contact_GetData(contactID).manifold;
            ASSERT(manifold.pointCount == 1);
            b2Vec2 hitImpulse = b2MulSV(manifold.points[0].normalImpulse, manifold.normal);
            if (!projIsShapeA) {
                hitImpulse = b2Neg(hitImpulse);
            }
            b2Body_ApplyLinearImpulseToCenter(wall->bodyID, hitImpulse, true);
            trackImpulse(e, wall->physicsTracking, b2MulSV(SHOT_HIT_IMPULSE_SCALE, hitImpulse), projectile->droneIdx, source);
        }

        if (ent->type == BOUNCY_WALL_ENTITY && !projectileIgnoresBouncyWalls(projectile)) {
            return 0;
        }
    } else if (ent->type == SHIELD_ENTITY) {
        // always allow projectiles to bounce off shields, and update shield health
        shieldEntity *shield = ent->entity;
        if (shield->health <= 0.0f) {
            return 0;
        }

        const float damage = projectile->lastSpeed * projectile->weaponInfo->mass * DRONE_SHIELD_HEALTH_IMPULSE_COEF;
        DEBUG_LOGF("shield projectile damage: %f", damage);
        shield->health -= damage;
        if (shield->health <= 0.0f) {
            droneEntity *parentDrone = safe_array_get_at(e->drones, projectile->droneIdx);
            if (shield->drone->idx != parentDrone->idx && shield->drone->team != parentDrone->team) {
                droneAddEnergy(parentDrone, DRONE_SHIELD_BREAK_ENERGY_REFILL);
            }
            parentDrone->stepInfo.brokeShield[shield->drone->idx] = true;
            e->stats[parentDrone->idx].shieldsBroken++;
        }

        // accelerator projectiles speed up off of shields, so shields
        // count towards their max bounces
        if (projectile->weaponInfo->type == ACCELERATOR_WEAPON) {
            projectile->bounces++;
            if (projectile->bounces == projectile->weaponInfo->maxBounces) {
                destroyProjectile(e, projectile, true, true);
                return 1;
            }
        }

        return 0;
    }

    if (projectile->weaponInfo->type != BLACK_HOLE_WEAPON || ent->type != DRONE_ENTITY) {
        projectile->bounces++;
    }
    if (ent->type == DRONE_ENTITY) {
        droneEntity *hitDrone = ent->entity;
        bodyID = hitDrone->bodyID;
        bodyIDSet = true;

        float hitStrength = 0.0f;
        if (b2Contact_IsValid(contactID)) {
            const b2Manifold manifold = b2Contact_GetData(contactID).manifold;
            ASSERT(manifold.pointCount == 1);
            b2Vec2 hitImpulse = b2MulSV(manifold.points[0].normalImpulse, manifold.normal);
            if (!projIsShapeA) {
                hitImpulse = b2Neg(hitImpulse);
            }
            hitStrength = b2AbsFloat(b2Length(hitImpulse));

            b2Body_ApplyLinearImpulseToCenter(hitDrone->bodyID, hitImpulse, true);
            trackImpulse(e, hitDrone->physicsTracking, b2MulSV(SHOT_HIT_IMPULSE_SCALE, hitImpulse), projectile->droneIdx, source);
        }

        if (projectile->droneIdx != hitDrone->idx) {
            droneEntity *shooterDrone = safe_array_get_at(e->drones, projectile->droneIdx);
            if (shooterDrone->team != hitDrone->team) {
                const float impulseEnergy = projectile->lastSpeed * projectile->weaponInfo->mass * projectile->weaponInfo->energyRefillCoef;
                droneAddEnergy(shooterDrone, impulseEnergy);
            }
            shooterDrone->stepInfo.shotHit[hitDrone->idx] += hitStrength * SHOT_HIT_IMPULSE_SCALE;
            recordShotHit(e, shooterDrone->idx, projectile->weaponInfo->type);
            e->analysisStats[shooterDrone->idx].boostedShotsHit[projectile->weaponInfo->type] += projectile->boosted;
            DEBUG_LOGF("drone %d hit drone %d with weapon %d", shooterDrone->idx, hitDrone->idx, projectile->weaponInfo->type);
            hitDrone->stepInfo.shotTaken[shooterDrone->idx] += hitStrength * SHOT_HIT_IMPULSE_SCALE;
            recordShotTaken(e, hitDrone->idx, projectile->weaponInfo->type);
            DEBUG_LOGF("drone %d hit by drone %d with weapon %d", hitDrone->idx, shooterDrone->idx, projectile->weaponInfo->type);
        } else {
            hitDrone->stepInfo.ownShotTaken = true;
            recordOwnShotTaken(e, hitDrone->idx, projectile->weaponInfo->type);
            DEBUG_LOGF("drone %d hit by own weapon %d", hitDrone->idx, projectile->weaponInfo->type);
        }

        if (projectile->weaponInfo->destroyedOnDroneHit) {
            destroyProjectile(e, projectile, projectile->weaponInfo->explodesOnDroneHit, true);
            return 1;
        }
    } else if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON && !projectile->welded) {
        if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON) {
            // if the mine is in explosion proximity of a drone now,
            // destroy it
            const b2QueryFilter filter = {
                .categoryBits = PROJECTILE_SHAPE,
                .maskBits = DRONE_SHAPE,
            };
            if (isOverlappingCircleInLineOfSight(e, projectile->ent, projectile->pos, MINE_LAUNCHER_PROXIMITY_RADIUS, filter, NULL)) {
                destroyProjectile(e, projectile, true, true);
                return 1;
            }
        }

        ASSERT(entityTypeIsWall(ent->type));
        wallEntity *wall = ent->entity;

        createProjectileWeldJoint(e, projectile, wall->bodyID, wall->rot, contactID, projIsShapeA);
        projectile->velocity = b2Vec2_zero;
        projectile->lastVelocity = b2Vec2_zero;
        projectile->speed = 0.0f;
        projectile->lastSpeed = 0.0f;
    }

    if (projectile->weaponInfo->type == GRAPPLE_WEAPON && !projectile->welded && bodyIDSet) {
        if (!b2Contact_IsValid(contactID)) {
            destroyProjectile(e, projectile, false, true);
            return 1;
        }

        // don't let a grapple attach to the drone that shot it;
        // destroy it instead
        if (ent->type == DRONE_ENTITY) {
            const droneEntity *hitDrone = ent->entity;
            if (hitDrone->idx == projectile->droneIdx) {
                destroyProjectile(e, projectile, false, true);
                return 1;
            }
        }

        createProjectileWeldJoint(e, projectile, bodyID, b2Body_GetRotation(bodyID), contactID, projIsShapeA);
        projectile->grappledEnt = ent;
        droneEntity *drone = safe_array_get_at(e->drones, projectile->droneIdx);
        uint8_t attachedTo = 2;
        if (entityTypeIsWall(ent->type)) {
            attachedTo = ((const wallEntity *)ent->entity)->isFloating;
        }
        e->analysisStats[drone->idx].grappleAttaches[attachedTo]++;
        drone->grappleLength = b2Distance(drone->pos, projectile->pos) + GRAPPLE_SLACK;

        b2DistanceJointDef def = b2DefaultDistanceJointDef();
        def.base.bodyIdA = drone->bodyID;
        def.base.bodyIdB = projectile->bodyID;
        def.enableSpring = true;
        def.hertz = 0.0f;
        def.dampingRatio = GRAPPLE_DAMPING_RATIO;
        def.enableLimit = true;
        def.minLength = 0.0f;
        def.maxLength = drone->grappleLength + GRAPPLE_MAX_STRETCH;
        def.length = drone->grappleLength;
        def.enableMotor = true;
        // start reeling right away if the utility button is held
        def.maxMotorForce = drone->utilityHeld ? GRAPPLE_MOTOR_FORCE : 0.0f;
        def.motorSpeed = -GRAPPLE_MOTOR_SPEED;

        drone->grappleJointID = b2CreateDistanceJoint(e->worldID, &def);
    }

    const uint8_t maxBounces = projectile->weaponInfo->maxBounces;
    if (projectile->bounces == maxBounces) {
        destroyProjectile(e, projectile, true, true);
        return 1;
    }

    return 0;
}

// ensure speed is maintained when a projectile hits a dynamic body
void handleProjectileEndContact(const entity *proj, const entity *ent) {
    projectileEntity *projectile = proj->entity;
    projectile->contacts--;

    // mines stick to walls, explode when hitting another projectile
    // and are destroyed when hitting a drone so no matter what we don't
    // need to do anything here; any projectile with a weld joint doesn't
    // need the velocity or speed messed with
    if (projectile->weaponInfo->type == MINE_LAUNCHER_WEAPON || projectile->welded) {
        return;
    }

    if (ent != NULL) {
        if (ent->type == PROJECTILE_ENTITY) {
            const projectileEntity *projectile2 = ent->entity;
            // allow projectile speeds to increase when two different
            // projectile types collide
            if (projectile->weaponInfo->type != projectile2->weaponInfo->type) {
                b2Vec2 newVel = b2Body_GetLinearVelocity(projectile->bodyID);
                float newSpeed = b2Length(newVel);
                if (newSpeed < projectile->lastSpeed) {
                    newSpeed = projectile->lastSpeed;
                    newVel = b2MulSV(newSpeed, b2Normalize(newVel));
                    b2Body_SetLinearVelocity(projectile->bodyID, newVel);
                }
                projectile->velocity = newVel;
                projectile->speed = newSpeed;
                projectile->lastSpeed = newSpeed;
                return;
            }
        }
    }

    // the last speed is used here instead of the current speed because
    // the current speed will be the speed box2d set the projectile to
    // after a collision and we want to keep the speed consistent
    float newSpeed = projectile->lastSpeed;
    if (projectile->weaponInfo->type == ACCELERATOR_WEAPON) {
        newSpeed = min(projectile->lastSpeed * ACCELERATOR_BOUNCE_SPEED_COEF, ACCELERATOR_MAX_SPEED);
        projectile->speed = newSpeed;
    }

    // ensure the projectile's speed doesn't change after bouncing off
    // something
    b2Vec2 newVel = b2Body_GetLinearVelocity(projectile->bodyID);
    newVel = b2MulSV(newSpeed, b2Normalize(newVel));
    b2Body_SetLinearVelocity(projectile->bodyID, newVel);
    projectile->velocity = newVel;
    projectile->speed = newSpeed;
    projectile->lastSpeed = newSpeed;
}

static inline bool entityLive(const entity *ent) {
    if (ent == NULL || (ent->type == PROJECTILE_ENTITY && ((projectileEntity *)ent->entity)->needsToBeDestroyed)) {
        return false;
    }
    return true;
}

void handleContactEvents(iwEnv *e) {
    b2ContactEvents events = b2World_GetContactEvents(e->worldID);
    for (int i = 0; i < events.beginCount; ++i) {
        const b2ContactBeginTouchEvent *event = events.beginEvents + i;
        entity *e1 = NULL;
        entity *e2 = NULL;

        if (b2Shape_IsValid(event->shapeIdA)) {
            e1 = b2Shape_GetUserData(event->shapeIdA);
            ASSERT(e1 != NULL);
        }
        if (b2Shape_IsValid(event->shapeIdB)) {
            e2 = b2Shape_GetUserData(event->shapeIdB);
            ASSERT(e2 != NULL);
        }
        // projectiles destroyed by an earlier explosion this step are
        // destroyed after all begin events are handled
        if (!entityLive(e1)) {
            e1 = NULL;
        }
        if (!entityLive(e2)) {
            e2 = NULL;
        }

        if (e1 != NULL && e1->type == DRONE_ENTITY && e2 != NULL && e2->type == DRONE_ENTITY) {
            droneEntity *drone1 = e1->entity;
            droneEntity *drone2 = e2->entity;

            if (b2Contact_IsValid(event->contactId)) {
                const b2Manifold manifold = b2Contact_GetData(event->contactId).manifold;
                ASSERT(manifold.pointCount == 1);
                b2Vec2 hitImpulse = b2MulSV(manifold.points[0].normalImpulse * (1.0f + DRONE_RESTITUTION), manifold.normal);
                const physicsSource source = {.type = DRONE_PHYSICS};
                trackImpulse(e, drone2->physicsTracking, hitImpulse, drone1->idx, source);
                trackImpulse(e, drone1->physicsTracking, b2Neg(hitImpulse), drone2->idx, source);
                // TODO: continue here?
            }
        }

        // track bounces off of walls before shields touching death walls
        // are destroyed below
        if (e1 != NULL && e2 != NULL) {
            trackWallBounce(e, e1, e2, event->contactId);
        }

        // TODO: drone on drone collisions should reduce shield health

        if (e1 != NULL) {
            if (e1->type == PROJECTILE_ENTITY) {
                uint8_t numDestroyed = handleProjectileBeginContact(e, e1, e2, event->contactId, true);
                if (numDestroyed == 2) {
                    continue;
                } else if (numDestroyed == 1) {
                    e1 = NULL;
                    if (!entityLive(e2)) {
                        e2 = NULL;
                    }
                }
            } else if (entityTypeIsWall(e1->type) && e2 != NULL) {
                if (e1->type == DEATH_WALL_ENTITY) {
                    if (e2->type == DRONE_ENTITY) {
                        droneEntity *drone = e2->entity;
                        const wallEntity *wall = e1->entity;
                        killDrone(e, drone, wall);
                    } else if (e2->type == SHIELD_ENTITY) {
                        shieldEntity *shield = e2->entity;
                        shield->health = 0.0f;
                        destroyDroneShield(e, shield, true);
                        e2 = NULL;
                    }
                }
            }
        }

        if (e2 != NULL) {
            if (e2->type == PROJECTILE_ENTITY) {
                handleProjectileBeginContact(e, e2, e1, event->contactId, false);
            } else if (entityTypeIsWall(e2->type) && e1 != NULL) {
                if (e2->type == DEATH_WALL_ENTITY) {
                    if (e1->type == DRONE_ENTITY) {
                        droneEntity *drone = e1->entity;
                        const wallEntity *wall = e2->entity;
                        killDrone(e, drone, wall);
                    } else if (e1->type == SHIELD_ENTITY) {
                        shieldEntity *shield = e1->entity;
                        shield->health = 0.0f;
                        destroyDroneShield(e, shield, true);
                    }
                }
            }
        }
    }

    destroyExplodedProjectiles(e);

    for (int i = 0; i < events.endCount; ++i) {
        const b2ContactEndTouchEvent *event = events.endEvents + i;
        entity *e1 = NULL;
        entity *e2 = NULL;
        if (b2Shape_IsValid(event->shapeIdA)) {
            e1 = b2Shape_GetUserData(event->shapeIdA);
            ASSERT(e1 != NULL);
        }
        if (b2Shape_IsValid(event->shapeIdB)) {
            e2 = b2Shape_GetUserData(event->shapeIdB);
            ASSERT(e2 != NULL);
        }
        if (e1 != NULL && e1->type == PROJECTILE_ENTITY) {
            handleProjectileEndContact(e1, e2);
        }
        if (e2 != NULL && e2->type == PROJECTILE_ENTITY) {
            handleProjectileEndContact(e2, e1);
        }
    }
}

// set pickup to respawn somewhere else randomly if a drone touched it,
// mark the pickup as disabled if a floating wall is touching it
// explode proximity detonating projectiles
void handleProjectileBeginTouch(iwEnv *e, const entity *sensor, entity *visitor) {
    projectileEntity *projectile = sensor->entity;

    switch (projectile->weaponInfo->type) {
    case FLAK_CANNON_WEAPON:
        if (projectile->distance < FLAK_CANNON_SAFE_DISTANCE) {
            return;
        }
        destroyProjectile(e, projectile, true, true);
        destroyExplodedProjectiles(e);
        break;
    case MINE_LAUNCHER_WEAPON:
        if (!projectile->welded) {
            return;
        }

        ASSERT(visitor->type == DRONE_ENTITY);
        const b2DistanceOutput output = closestPoint(sensor, visitor);
        const b2QueryFilter filter = {.categoryBits = PROJECTILE_SHAPE, .maskBits = WALL_SHAPE | FLOATING_WALL_SHAPE};
        if (posBehindWall(e, projectile->pos, output.pointB, NULL, filter, NULL)) {
            const droneEntity *drone = visitor->entity;
            projectileAddDroneBehindWall(projectile, drone->idx);
            return;
        }

        destroyProjectile(e, projectile, true, true);
        destroyExplodedProjectiles(e);
        break;
    case BLACK_HOLE_WEAPON:
        if (visitor->type == DRONE_ENTITY) {
            const droneEntity *drone = visitor->entity;
            // TODO: does this work?
            if (projectile->droneIdx == drone->idx && projectile->distance < BLACK_HOLE_PARENT_IGNORE_DISTANCE) {
                return;
            }
        }

        // copy the entity ID so it won't be changed if the entity is
        // destroyed and reused later
        const entityID *visitorID = visitor->id;
        entityID *id = fastCalloc(1, sizeof(entityID));
        id->id = visitorID->id;
        id->generation = visitorID->generation;
        cc_array_add(projectile->entsInBlackHole, id);
        break;
    default:
        ERRORF("invalid projectile type %d for begin touch event", sensor->type);
    }
}

void handleProjectileEndTouch(const entity *sensor, entity *visitor) {
    projectileEntity *projectile = sensor->entity;

    switch (projectile->weaponInfo->type) {
    case FLAK_CANNON_WEAPON:
        break;
    case MINE_LAUNCHER_WEAPON:
        if (visitor == NULL) {
            return;
        }
        ASSERT(visitor->type == DRONE_ENTITY);
        const droneEntity *drone = visitor->entity;
        projectileRemoveDroneBehindWall(projectile, drone->idx);
        break;
    case BLACK_HOLE_WEAPON:
        if (visitor == NULL) {
            return;
        }

        const entityID *visitorID = visitor->id;
        for (uint8_t i = 0; i < cc_array_size(projectile->entsInBlackHole); ++i) {
            entityID *id = safe_array_get_at(projectile->entsInBlackHole, i);
            if (id->id == visitorID->id) {
                fastFree(id);
                cc_array_remove_fast_at(projectile->entsInBlackHole, i, NULL);
                return;
            }
        }
        break;
    default:
        ERRORF("invalid projectile type %d for end touch event", projectile->weaponInfo->type);
    }
}

void handleSensorEvents(iwEnv *e) {
    b2SensorEvents events = b2World_GetSensorEvents(e->worldID);
    for (int i = 0; i < events.beginCount; ++i) {
        const b2SensorBeginTouchEvent *event = events.beginEvents + i;
        if (!b2Shape_IsValid(event->sensorShapeId)) {
            DEBUG_LOG("could not find sensor shape for begin touch event");
            continue;
        }
        entity *s = b2Shape_GetUserData(event->sensorShapeId);
        ASSERT(s != NULL);

        if (!b2Shape_IsValid(event->visitorShapeId)) {
            DEBUG_LOG("could not find visitor shape for begin touch event");
            continue;
        }
        entity *v = b2Shape_GetUserData(event->visitorShapeId);
        ASSERT(v != NULL);

        if (s->type != PROJECTILE_ENTITY) {
            ERRORF("unknown entity type %d for sensor begin touch event", s->type);
        }
        handleProjectileBeginTouch(e, s, v);
    }

    for (int i = 0; i < events.endCount; ++i) {
        const b2SensorEndTouchEvent *event = events.endEvents + i;
        if (!b2Shape_IsValid(event->sensorShapeId)) {
            DEBUG_LOG("could not find sensor shape for end touch event");
            continue;
        }
        entity *s = b2Shape_GetUserData(event->sensorShapeId);
        ASSERT(s != NULL);
        entity *v = NULL;
        if (b2Shape_IsValid(event->visitorShapeId)) {
            v = b2Shape_GetUserData(event->visitorShapeId);
            ASSERT(v != NULL);
        }

        ASSERT(s->type == PROJECTILE_ENTITY);
        handleProjectileEndTouch(s, v);
    }
}

// index of the (destination, source) pair in a map's flow field
static inline uint32_t pathOffset(const iwEnv *e, const uint16_t srcCellIdx, const uint16_t destCellIdx) {
    const uint32_t cellCount = e->map->rows * e->map->columns;
    return (uint32_t)destCellIdx * cellCount + srcCellIdx;
}

// flood fills flatPaths with the direction each cell should move in to
// reach destCellIdx; buffer is scratch space for the BFS queue and must
// hold at least 8 entries per cell
void pathfindBFS(const iwEnv *e, uint8_t *flatPaths, int8_t (*buffer)[3], const uint16_t destCellIdx) {
    uint8_t (*paths)[e->map->columns] = (uint8_t (*)[e->map->columns])flatPaths;

    uint16_t start = 0;
    uint16_t end = 1;

    const mapCell *cell = (&e->cells[destCellIdx]);
    if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
        return;
    }
    const int8_t destCol = destCellIdx % e->map->columns;
    const int8_t destRow = destCellIdx / e->map->columns;

    buffer[start][0] = 8;
    buffer[start][1] = destCol;
    buffer[start][2] = destRow;
    while (start < end) {
        const int8_t direction = buffer[start][0];
        const int8_t startCol = buffer[start][1];
        const int8_t startRow = buffer[start][2];
        start++;

        if (startCol < 0 || startCol >= e->map->columns || startRow < 0 || startRow >= e->map->rows || paths[startRow][startCol] != UINT8_MAX) {
            continue;
        }
        int16_t cellIdx = cellIndex(e, startCol, startRow);
        const mapCell *cell = (&e->cells[cellIdx]);
        if (cell->ent != NULL && entityTypeIsWall(cell->ent->type)) {
            paths[startRow][startCol] = 8;
            continue;
        }

        paths[startRow][startCol] = direction;

        buffer[end][0] = 6; // up
        buffer[end][1] = startCol;
        buffer[end][2] = startRow + 1;
        end++;

        buffer[end][0] = 2; // down
        buffer[end][1] = startCol;
        buffer[end][2] = startRow - 1;
        end++;

        buffer[end][0] = 0; // right
        buffer[end][1] = startCol - 1;
        buffer[end][2] = startRow;
        end++;

        buffer[end][0] = 4; // left
        buffer[end][1] = startCol + 1;
        buffer[end][2] = startRow;
        end++;

        buffer[end][0] = 5; // up left
        buffer[end][1] = startCol + 1;
        buffer[end][2] = startRow + 1;
        end++;

        buffer[end][0] = 3; // down left
        buffer[end][1] = startCol + 1;
        buffer[end][2] = startRow - 1;
        end++;

        buffer[end][0] = 1; // down right
        buffer[end][1] = startCol - 1;
        buffer[end][2] = startRow - 1;
        end++;

        buffer[end][0] = 7; // up right
        buffer[end][1] = startCol - 1;
        buffer[end][2] = startRow + 1;
        end++;
    }
}

static inline void considerNearWallCell(const iwEnv *e, const droneEntity *drone, const int16_t col, const int16_t row, nearWall nearestWalls[], const uint8_t nWalls, uint8_t *found) {
    if (col < 0 || col >= e->map->columns || row < 0 || row >= e->map->rows) {
        return;
    }
    const mapCell *cell = &e->cells[col + row * e->map->columns];
    if (cell->ent == NULL || !entityTypeIsWall(cell->ent->type)) {
        return;
    }
    const wallEntity *wall = cell->ent->entity;
    if (wall->isFloating) {
        return;
    }

    const nearWall candidate = {
        .pos = wall->pos,
        .type = wall->type,
        .distanceSquared = b2DistanceSquared(drone->pos, wall->pos),
    };
    uint8_t insert = 0;

    while (insert < *found) {
        const nearWall *other = &nearestWalls[insert];
        // Break distance ties by grid order, independently of spiral order.
        if (candidate.distanceSquared < other->distanceSquared ||
            (candidate.distanceSquared == other->distanceSquared &&
             (candidate.pos.y < other->pos.y ||
              (candidate.pos.y == other->pos.y && candidate.pos.x < other->pos.x)))) {
            break;
        }
        insert++;
    }
    if (insert == nWalls) {
        return;
    }

    if (*found < nWalls) {
        (*found)++;
    }
    for (uint8_t i = *found - 1; i > insert; i--) {
        nearestWalls[i] = nearestWalls[i - 1];
    }
    nearestWalls[insert] = candidate;
}

// Visit complete square rings of live cells. All static walls, including
// sudden-death walls, are at cell centers; floating walls are observed separately.
static void findNearWallsSpiral(const iwEnv *e, const droneEntity *drone, nearWall nearestWalls[], const uint8_t nWalls) {
    const b2Vec2 origin = e->cells[0].pos;
    const int16_t col = (int16_t)fminf(e->map->columns - 1, fmaxf(0.0f, floorf((drone->pos.x - origin.x) / WALL_THICKNESS + 0.5f)));
    const int16_t row = (int16_t)fminf(e->map->rows - 1, fmaxf(0.0f, floorf((drone->pos.y - origin.y) / WALL_THICKNESS + 0.5f)));
    const int16_t maxRadius = max(max(col, e->map->columns - 1 - col), max(row, e->map->rows - 1 - row));
    uint8_t found = 0;

    for (uint8_t i = 0; i < nWalls; i++) {
        nearestWalls[i] = (nearWall){.distanceSquared = FLT_MAX};
    }
    considerNearWallCell(e, drone, col, row, nearestWalls, nWalls, &found);

    for (int16_t radius = 0; radius <= maxRadius; radius++) {
        if (radius != 0) {
            for (int16_t x = max(0, col - radius); x <= min(e->map->columns - 1, col + radius); x++) {
                considerNearWallCell(e, drone, x, row - radius, nearestWalls, nWalls, &found);
                considerNearWallCell(e, drone, x, row + radius, nearestWalls, nWalls, &found);
            }
            // Exclude corners already visited along the horizontal edges.
            for (int16_t y = max(0, row - radius + 1); y <= min(e->map->rows - 1, row + radius - 1); y++) {
                considerNearWallCell(e, drone, col - radius, y, nearestWalls, nWalls, &found);
                considerNearWallCell(e, drone, col + radius, y, nearestWalls, nWalls, &found);
            }
        }

        // Every unvisited cell is beyond at least one searched edge. Its
        // distance is at least the axis distance to that side's next cell
        // center. Ignore sides exhausted by the map boundary. Use strict >
        // so equal-distance candidates still receive deterministic tie ordering.
        float remainingDistance = FLT_MAX;
        if (col - radius > 0) {
            remainingDistance = min(remainingDistance, drone->pos.x - e->cells[col - radius - 1].pos.x);
        }
        if (col + radius + 1 < e->map->columns) {
            remainingDistance = min(remainingDistance, e->cells[col + radius + 1].pos.x - drone->pos.x);
        }
        if (row - radius > 0) {
            remainingDistance = min(remainingDistance, drone->pos.y - e->cells[(row - radius - 1) * e->map->columns].pos.y);
        }
        if (row + radius + 1 < e->map->rows) {
            remainingDistance = min(remainingDistance, e->cells[(row + radius + 1) * e->map->columns].pos.y - drone->pos.y);
        }
        if (found == nWalls && (remainingDistance == FLT_MAX ||
                                remainingDistance * remainingDistance > nearestWalls[nWalls - 1].distanceSquared)) {
            break;
        }
    }
}

void findNearWalls(const iwEnv *e, const droneEntity *drone, nearWall nearestWalls[], const uint8_t nWalls) {
    ASSERT(nWalls <= MAX_NEAREST_WALLS);
    if (nWalls == 0) {
        return;
    }
    if (e->suddenDeathWallsPlaced) {
        findNearWallsSpiral(e, drone, nearestWalls, nWalls);
        return;
    }
    nearWall nearWalls[MAX_NEAREST_WALLS];

    const nearWall *candidates = e->map->nearestWalls + (MAX_NEAREST_WALLS * drone->mapCellIdx);
    for (uint8_t i = 0; i < MAX_NEAREST_WALLS; ++i) {
        nearWalls[i] = candidates[i];
        nearWalls[i].distanceSquared = b2DistanceSquared(drone->pos, candidates[i].pos);
    }
    insertionSortWalls(nearWalls, MAX_NEAREST_WALLS);
    memcpy(nearestWalls, nearWalls, nWalls * sizeof(nearWall));
}

#endif
