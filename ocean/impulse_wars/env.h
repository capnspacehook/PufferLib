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

// normalize a drone's ammo count, setting infinite ammo as no ammo
static inline float scaleAmmo(const iwEnv *e, const droneEntity *drone) {
    int8_t maxAmmo = weaponAmmo(e->defaultWeapon->type, drone->weaponInfo->type);
    float scaledAmmo = 0;
    if (drone->ammo != INFINITE) {
        scaledAmmo = scaleValue(drone->ammo, maxAmmo, true);
    }
    return scaledAmmo;
}

// fills a small 2D grid centered around the agent with discretized
// walls, floating walls, weapon pickups, and drone positions
void computeMapObs(iwEnv *e, const uint8_t agentIdx, const uint16_t obsStartOffset) {
    droneEntity *drone = safe_array_get_at(e->drones, agentIdx);
    const uint8_t droneCellCol = drone->mapCellIdx % e->map->columns;
    const uint8_t droneCellRow = drone->mapCellIdx / e->map->columns;

    const int8_t obsStartCol = droneCellCol - (MAP_OBS_COLUMNS / 2);
    const int8_t startCol = max(obsStartCol, 0);
    const int8_t obsStartRow = droneCellRow - (MAP_OBS_ROWS / 2);
    const int8_t startRow = max(obsStartRow, 0);

    const int8_t obsEndCol = droneCellCol + (MAP_OBS_COLUMNS / 2);
    const int8_t endCol = min(obsEndCol, e->map->columns - 1);
    const int8_t endRow = min(droneCellRow + (MAP_OBS_ROWS / 2), e->map->rows - 1);

    const int8_t obsColOffset = startCol - obsStartCol;
    const int8_t obsRowOffset = startRow - obsStartRow;
    uint16_t startOffset = obsStartOffset;
    if (obsColOffset == 0 && obsRowOffset != 0) {
        startOffset += obsRowOffset * MAP_OBS_COLUMNS;
    } else if (obsColOffset != 0 && obsRowOffset == 0) {
        startOffset += obsColOffset;
    } else if (obsColOffset != 0 && obsRowOffset != 0) {
        startOffset += obsColOffset + (obsRowOffset * MAP_OBS_COLUMNS);
    }
    uint16_t offset = startOffset;

    // compute map layout, and discretized positions of weapon pickups
    if (!e->suddenDeathWallsPlaced) {
        // copy precomputed map layout if sudden death walls haven't been placed
        const int8_t numCols = endCol - startCol + 1;
        for (int8_t row = startRow; row <= endRow; row++) {
            const int16_t cellIdx = cellIndex(e, startCol, row);
            memcpy(agentObs(e, agentIdx) + offset, e->map->packedLayout + cellIdx, numCols * sizeof(float));
            offset += MAP_OBS_COLUMNS;
        }

        // // compute discretized location of weapon pickups on grid
        // for (size_t i = 0; i < cc_array_size(e->pickups); i++) {
        //     const weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
        //     const uint8_t cellCol = pickup->mapCellIdx % e->map->columns;
        //     if (cellCol < startCol || cellCol > endCol) {
        //         continue;
        //     }
        //     const uint8_t cellRow = pickup->mapCellIdx / e->map->columns;
        //     if (cellRow < startRow || cellRow > endRow) {
        //         continue;
        //     }

        //     offset = startOffset + ((cellCol - startCol) + ((cellRow - startRow) * MAP_OBS_COLUMNS));
        //     ASSERTF(offset <= startOffset + MAP_OBS_SIZE, "offset: %d", offset);
        //     e->observations[offset] |= 1 << 3;
        // }
    } else {
        // sudden death walls have been placed so compute may layout manually
        const int8_t colPadding = obsColOffset + (obsEndCol - endCol);
        for (int8_t row = startRow; row <= endRow; row++) {
            for (int8_t col = startCol; col <= endCol; col++) {
                const int16_t cellIdx = cellIndex(e, col, row);
                const mapCell *cell = (&e->cells[cellIdx]);
                if (cell->ent == NULL) {
                    offset++;
                    continue;
                }

                if (entityTypeIsWall(cell->ent->type)) {
                    agentObs(e, agentIdx)[offset] = (float)(cell->ent->type + 1) / 3.0f;
                }
                // else if (cell->ent->type == WEAPON_PICKUP_ENTITY) {
                //     e->observations[offset] |= 1 << 3;
                // }

                offset++;
            }
            offset += colPadding;
        }
        ASSERTF(offset <= startOffset + MAP_OBS_SIZE, "offset %u startOffset %u", offset, startOffset);
    }

    // // compute discretized locations of floating walls on grid
    // for (size_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
    //     const wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
    //     const uint8_t cellCol = wall->mapCellIdx % e->map->columns;
    //     if (cellCol < startCol || cellCol > endCol) {
    //         continue;
    //     }
    //     const uint8_t cellRow = wall->mapCellIdx / e->map->columns;
    //     if (cellRow < startRow || cellRow > endRow) {
    //         continue;
    //     }

    //     offset = startOffset + ((cellCol - startCol) + ((cellRow - startRow) * MAP_OBS_COLUMNS));
    //     ASSERTF(offset <= startOffset + MAP_OBS_SIZE, "offset: %d", offset);
    //     e->observations[offset] = ((wall->type + 1) & TWO_BIT_MASK) << 5;
    //     e->observations[offset] |= 1 << 4;
    // }

    // // compute discretized location and index of drones on grid
    // uint8_t newDroneIdx = 1;
    // uint16_t droneCells[e->numDrones];
    // memset(droneCells, 0x0, sizeof(droneCells));
    // for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
    //     if (i == agentIdx) {
    //         continue;
    //     }

    //     // ensure drones do not share cells in the observation
    //     droneEntity *otherDrone = safe_array_get_at(e->drones, i);
    //     if (i != 0) {
    //         for (uint8_t j = 0; j < i; j++) {
    //             if (droneCells[j] == otherDrone->mapCellIdx) {
    //                 otherDrone->mapCellIdx = findNearestCell(e, otherDrone->pos, otherDrone->mapCellIdx);
    //                 break;
    //             }
    //         }
    //     }
    //     const uint8_t cellCol = otherDrone->mapCellIdx % e->map->columns;
    //     if (cellCol < startCol || cellCol > endCol) {
    //         continue;
    //     }
    //     const uint8_t cellRow = otherDrone->mapCellIdx / e->map->columns;
    //     if (cellRow < startRow || cellRow > endRow) {
    //         continue;
    //     }
    //     droneCells[i] = otherDrone->mapCellIdx;

    //     offset = startOffset + ((cellCol - startCol) + ((cellRow - startRow) * MAP_OBS_COLUMNS));
    //     ASSERTF(offset <= startOffset + MAP_OBS_SIZE, "offset: %d", offset);
    //     e->observations[offset] |= (newDroneIdx++ & THREE_BIT_MASK);
    // }
}

// computes observations for N nearest walls, floating walls, and weapon pickups
void computeNearObs(iwEnv *e, const droneEntity *drone, const uint16_t discreteObsStart, float *continuousObs) {
    nearWall nearWalls[NUM_NEAR_WALL_OBS];
    findNearWalls(e, drone, nearWalls, NUM_NEAR_WALL_OBS);

    uint16_t offset;

    // compute type and position of N nearest walls
    for (uint8_t i = 0; i < NUM_NEAR_WALL_OBS; i++) {
        const nearWall *wall = &nearWalls[i];

        offset = discreteObsStart + NEAR_WALL_TYPES_OBS_OFFSET + i;
        ASSERTF(offset <= discreteObsStart + FLOATING_WALL_TYPES_OBS_OFFSET, "offset: %d", offset);
        agentObs(e, drone->idx)[offset] = (float)wall->type / 2.0f;

        offset = NEAR_WALL_POS_OBS_OFFSET + (i * NEAR_WALL_POS_OBS_SIZE);
        ASSERTF(offset <= FLOATING_WALL_INFO_OBS_OFFSET, "offset: %d", offset);
        const b2Vec2 wallRelPos = b2Sub(wall->pos, drone->pos);

        continuousObs[offset++] = scaleValue(wallRelPos.x, MAX_X_POS, false);
        continuousObs[offset] = scaleValue(wallRelPos.y, MAX_Y_POS, false);
    }

    if (cc_array_size(e->floatingWalls) != 0) {
        // find N nearest floating walls
        nearEntity nearFloatingWalls[MAX_FLOATING_WALLS] = {0};
        for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
            wallEntity *wall = safe_array_get_at(e->floatingWalls, i);
            const nearEntity nearEnt = {
                .entity = wall,
                .distanceSquared = b2DistanceSquared(wall->pos, drone->pos),
            };
            nearFloatingWalls[i] = nearEnt;
        }
        insertionSortEntities(nearFloatingWalls, cc_array_size(e->floatingWalls));

        // compute type, position, angle and velocity of N nearest floating walls
        for (uint8_t i = 0; i < cc_array_size(e->floatingWalls); i++) {
            if (i == NUM_FLOATING_WALL_OBS) {
                break;
            }
            const wallEntity *wall = nearFloatingWalls[i].entity;

            const b2Transform wallTransform = b2Body_GetTransform(wall->bodyID);
            const b2Vec2 wallRelPos = b2Sub(wallTransform.p, drone->pos);
            const float angle = b2Rot_GetAngle(wallTransform.q);

            offset = discreteObsStart + FLOATING_WALL_TYPES_OBS_OFFSET + i;
            ASSERTF(offset <= discreteObsStart + PROJECTILE_DRONE_OBS_OFFSET, "offset: %d", offset);
            agentObs(e, drone->idx)[offset] = (float)(wall->type + 1) / 3.0f;

            // DEBUG_LOGF("floating wall %d cell %d", i, wall->mapCellIdx);

            offset = FLOATING_WALL_INFO_OBS_OFFSET + (i * FLOATING_WALL_INFO_OBS_SIZE);
            ASSERTF(offset <= WEAPON_PICKUP_POS_OBS_OFFSET, "offset: %d", offset);
            continuousObs[offset++] = scaleValue(wallRelPos.x, MAX_X_POS, false);
            continuousObs[offset++] = scaleValue(wallRelPos.y, MAX_Y_POS, false);
            continuousObs[offset++] = scaleValue(angle, MAX_ANGLE, false);
            continuousObs[offset++] = scaleValue(wall->velocity.x, MAX_SPEED, false);
            continuousObs[offset] = scaleValue(wall->velocity.y, MAX_SPEED, false);
        }
    }

    if (cc_array_size(e->pickups) != 0) {
        // find N nearest weapon pickups
        nearEntity nearPickups[MAX_WEAPON_PICKUPS] = {0};
        for (uint8_t i = 0; i < cc_array_size(e->pickups); i++) {
            weaponPickupEntity *pickup = safe_array_get_at(e->pickups, i);
            const nearEntity nearEnt = {
                .entity = pickup,
                .distanceSquared = b2DistanceSquared(pickup->pos, drone->pos),
            };
            nearPickups[i] = nearEnt;
        }
        insertionSortEntities(nearPickups, cc_array_size(e->pickups));

        // compute type and location of N nearest weapon pickups
        for (uint8_t i = 0; i < cc_array_size(e->pickups); i++) {
            if (i == NUM_WEAPON_PICKUP_OBS) {
                break;
            }
            const weaponPickupEntity *pickup = nearPickups[i].entity;

            offset = discreteObsStart + WEAPON_PICKUP_WEAPONS_OBS_OFFSET + i;
            ASSERTF(offset <= discreteObsStart + ENEMY_DRONE_WEAPONS_OBS_OFFSET, "offset: %d", offset);
            agentObs(e, drone->idx)[offset] = (float)(pickup->weapon + 1) / NUM_WEAPONS;

            // DEBUG_LOGF("pickup %d cell %d", i, pickup->mapCellIdx);

            offset = WEAPON_PICKUP_POS_OBS_OFFSET + (i * WEAPON_PICKUP_POS_OBS_SIZE);
            ASSERTF(offset <= PROJECTILE_INFO_OBS_OFFSET, "offset: %d", offset);
            const b2Vec2 pickupRelPos = b2Sub(pickup->pos, drone->pos);
            continuousObs[offset++] = scaleValue(pickupRelPos.x, MAX_X_POS, false);
            continuousObs[offset] = scaleValue(pickupRelPos.y, MAX_Y_POS, false);
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
        uint16_t discreteObsOffset;
        uint16_t continuousObsOffset;
        const uint16_t continuousObsStart = discreteObsStart + e->discreteObsSize;
        float *continuousObs = obs + continuousObsStart;

        computeNearObs(e, agentDrone, discreteObsStart, continuousObs);

        // sort projectiles by distance to the current agent
        const b2Vec2 agentPos = agentDrone->pos;
        const size_t numProjectiles = cc_array_size(e->projectiles);
        if (numProjectiles > 0) {
            projectileEntity *sortedProjectiles[numProjectiles];
            memcpy(sortedProjectiles, e->projectiles->buffer, numProjectiles * sizeof(projectileEntity *));

            for (int16_t i = 1; i < (int64_t)numProjectiles; i++) {
                projectileEntity *key = sortedProjectiles[i];
                const float keyDistance = b2DistanceSquared(agentPos, key->pos);
                int16_t j = i - 1;

                while (j >= 0 && b2DistanceSquared(agentPos, sortedProjectiles[j]->pos) > keyDistance) {
                    sortedProjectiles[j + 1] = sortedProjectiles[j];
                    j = j - 1;
                }

                sortedProjectiles[j + 1] = key;
            }

            // compute type and location of N projectiles
            for (size_t i = 0; i < numProjectiles; i++) {
                if (i == NUM_PROJECTILE_OBS) {
                    break;
                }
                const projectileEntity *projectile = sortedProjectiles[i];

                discreteObsOffset = discreteObsStart + PROJECTILE_DRONE_OBS_OFFSET + i;
                ASSERTF(discreteObsOffset <= discreteObsStart + PROJECTILE_WEAPONS_OBS_OFFSET, "offset: %d", discreteObsOffset);
                obs[discreteObsOffset] = (float)(projectile->droneIdx + 1) / e->numDrones;

                discreteObsOffset = discreteObsStart + PROJECTILE_WEAPONS_OBS_OFFSET + i;
                ASSERTF(discreteObsOffset <= discreteObsStart + WEAPON_PICKUP_WEAPONS_OBS_OFFSET, "offset: %d", discreteObsOffset);
                obs[discreteObsOffset] = (float)(projectile->weaponInfo->type + 1) / NUM_WEAPONS;

                continuousObsOffset = PROJECTILE_INFO_OBS_OFFSET + (i * PROJECTILE_INFO_OBS_SIZE);
                ASSERTF(continuousObsOffset <= ENEMY_DRONE_OBS_OFFSET, "offset: %d", continuousObsOffset);
                const b2Vec2 projectileRelPos = b2Sub(projectile->pos, agentDrone->pos);
                continuousObs[continuousObsOffset++] = scaleValue(projectileRelPos.x, MAX_X_POS, false);
                continuousObs[continuousObsOffset++] = scaleValue(projectileRelPos.y, MAX_Y_POS, false);
                continuousObs[continuousObsOffset++] = scaleValue(projectile->velocity.x, MAX_SPEED, false);
                continuousObs[continuousObsOffset] = scaleValue(projectile->velocity.y, MAX_SPEED, false);
            }
        }

        // compute enemy drone observations
        bool hitShot = false;
        bool tookShot = false;
        uint8_t processedDrones = 0;
        for (uint8_t i = 0; i < e->numDrones; i++) {
            if (i == agentIdx) {
                continue;
            }

            if (agentDrone->stepInfo.shotHit[i] != 0.0f) {
                hitShot = true;
            }
            if (agentDrone->stepInfo.shotTaken[i] != 0.0f) {
                tookShot = true;
            }

            droneEntity *enemyDrone = safe_array_get_at(e->drones, i);
            if (enemyDrone->livesLeft == 0) {
                processedDrones++;
                continue;
            }

            const b2Vec2 enemyDroneRelPos = b2Sub(enemyDrone->pos, agentDrone->pos);
            const float enemyDroneDistance = b2Distance(enemyDrone->pos, agentDrone->pos);
            const b2Vec2 enemyDroneAccel = b2Sub(enemyDrone->velocity, enemyDrone->lastVelocity);
            const b2Vec2 enemyDroneRelNormPos = b2Normalize(b2Sub(enemyDrone->pos, agentDrone->pos));
            const float enemyDroneAimAngle = atan2f(enemyDrone->lastAim.y, enemyDrone->lastAim.x);
            float enemyDroneBraking = 0.0f;
            if (enemyDrone->braking) {
                enemyDroneBraking = 1.0f;
            }

            discreteObsOffset = discreteObsStart + ENEMY_DRONE_WEAPONS_OBS_OFFSET + processedDrones;
            obs[discreteObsOffset] = (float)(enemyDrone->weaponInfo->type + 1) / NUM_WEAPONS;

            continuousObsOffset = ENEMY_DRONE_OBS_OFFSET + (processedDrones * ENEMY_DRONE_OBS_SIZE);
            continuousObs[continuousObsOffset++] = enemyDrone->team == agentDrone->team;
            continuousObs[continuousObsOffset++] = scaleValue(enemyDroneRelPos.x, MAX_X_POS, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDroneRelPos.y, MAX_Y_POS, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDroneDistance, MAX_DISTANCE, true);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->velocity.x, MAX_SPEED, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->velocity.y, MAX_SPEED, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDroneAccel.x, MAX_ACCEL, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDroneAccel.y, MAX_ACCEL, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDroneRelNormPos.x, 1.0f, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDroneRelNormPos.y, 1.0f, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->lastAim.x, 1.0f, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->lastAim.y, 1.0f, false);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDroneAimAngle, PI, false);
            continuousObs[continuousObsOffset++] = scaleAmmo(e, enemyDrone);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->weaponCooldown, enemyDrone->weaponInfo->coolDown, true);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->weaponCharge, enemyDrone->weaponInfo->charge, true);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->energyLeft, DRONE_ENERGY_MAX, true);
            continuousObs[continuousObsOffset++] = (float)enemyDrone->energyFullyDepleted;
            continuousObs[continuousObsOffset++] = enemyDroneBraking;
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->burstCooldown, DRONE_BURST_COOLDOWN, true);
            continuousObs[continuousObsOffset++] = (float)enemyDrone->chargingBurst;
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->burstCharge, DRONE_ENERGY_MAX, true);
            continuousObs[continuousObsOffset++] = scaleValue(enemyDrone->livesLeft, DRONE_LIVES, true);
            continuousObs[continuousObsOffset++] = !enemyDrone->dead;

            processedDrones++;
            ASSERTF(continuousObsOffset == ENEMY_DRONE_OBS_OFFSET + (processedDrones * ENEMY_DRONE_OBS_SIZE), "offset: %d", continuousObsOffset);
        }

        // compute active drone observations
        continuousObsOffset = ENEMY_DRONE_OBS_OFFSET + ((e->numDrones - 1) * ENEMY_DRONE_OBS_SIZE);
        const b2Vec2 agentDroneAccel = b2Sub(agentDrone->velocity, agentDrone->lastVelocity);
        float agentDroneBraking = 0.0f;
        if (agentDrone->braking) {
            agentDroneBraking = 1.0f;
        }

        discreteObsOffset = discreteObsStart + ENEMY_DRONE_WEAPONS_OBS_OFFSET + e->numDrones - 1;
        obs[discreteObsOffset] = (float)(agentDrone->weaponInfo->type + 1) / NUM_WEAPONS;

        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->pos.x, MAX_X_POS, false);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->pos.y, MAX_Y_POS, false);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->velocity.x, MAX_SPEED, false);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->velocity.y, MAX_SPEED, false);
        continuousObs[continuousObsOffset++] = scaleValue(agentDroneAccel.x, MAX_ACCEL, false);
        continuousObs[continuousObsOffset++] = scaleValue(agentDroneAccel.y, MAX_ACCEL, false);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->lastAim.x, 1.0f, false);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->lastAim.y, 1.0f, false);
        continuousObs[continuousObsOffset++] = scaleAmmo(e, agentDrone);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->weaponCooldown, agentDrone->weaponInfo->coolDown, true);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->weaponCharge, agentDrone->weaponInfo->charge, true);
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->energyLeft, DRONE_ENERGY_MAX, true);
        continuousObs[continuousObsOffset++] = (float)agentDrone->energyFullyDepleted;
        continuousObs[continuousObsOffset++] = agentDroneBraking;
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->burstCooldown, DRONE_BURST_COOLDOWN, true);
        continuousObs[continuousObsOffset++] = (float)agentDrone->chargingBurst;
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->burstCharge, DRONE_ENERGY_MAX, true);
        continuousObs[continuousObsOffset++] = hitShot;
        continuousObs[continuousObsOffset++] = tookShot;
        continuousObs[continuousObsOffset++] = agentDrone->stepInfo.ownShotTaken;
        continuousObs[continuousObsOffset++] = scaleValue(agentDrone->livesLeft, DRONE_LIVES, true);
        continuousObs[continuousObsOffset++] = !agentDrone->dead;

        ASSERTF(continuousObsOffset == ENEMY_DRONE_OBS_OFFSET + ((e->numDrones - 1) * ENEMY_DRONE_OBS_SIZE) + DRONE_OBS_SIZE, "offset: %d", continuousObsOffset);
        continuousObs[continuousObsOffset] = scaleValue(e->stepsLeft, e->totalSteps, true);
    }
}

void setupEnv(iwEnv *e) {
    e->needsReset = false;

    e->stepsLeft = e->totalSteps;
    e->suddenDeathSteps = e->totalSuddenDeathSteps;
    e->suddenDeathWallCounter = 0;

    e->lastSpawnQuad = -1;

    int8_t mapIdx = e->pinnedMapIdx;
    if (e->pinnedMapIdx == -1) {
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

    DEBUG_LOG("placing floating walls");
    placeRandFloatingWalls(e, mapIdx);

    DEBUG_LOG("creating weapon pickups");
    // start spawning pickups in a random quadrant
    e->lastSpawnQuad = randInt(&e->rng, 0, 3);
    for (uint8_t i = 0; i < maps[mapIdx]->weaponPickups; i++) {
        createWeaponPickup(e);
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

iwEnv *initEnv(iwEnv *e, uint8_t numDrones, uint8_t numAgents, int8_t mapIdx, uint64_t seed, bool enableTeams, bool sittingDuck, bool isTraining, bool continuousActions, float botCLNoise, float botCLDecay) {
    DEBUG_LOGF("seed: %lu", seed);

    e->numDrones = numDrones;
    e->numAgents = numAgents;
    e->num_agents = numAgents;
    e->teamsEnabled = enableTeams;
    e->numTeams = numDrones;
    if (e->teamsEnabled) {
        e->numTeams = 2;
    }
    e->sittingDuck = sittingDuck;
    e->isTraining = isTraining;
    e->botCLNoise = botCLNoise;
    e->botCLDecay = botCLDecay;

    e->winReward = WIN_REWARD;
    e->selfKillPunishment = SELF_KILL_PUNISHMENT;
    e->enemyDeathReward = ENEMY_DEATH_REWARD;
    e->enemyKillReward = ENEMY_KILL_REWARD;
    e->teammateDeathPunishment = TEAMMATE_DEATH_PUNISHMENT;
    e->teammateKillPunishment = TEAMMATE_KILL_PUNISHMENT;
    e->deathPunishment = DEATH_PUNISHMENT;
    e->energyEmptiedPunishment = ENERGY_EMPTY_PUNISHMENT;
    e->weaponPickupReward = WEAPON_PICKUP_REWARD;
    e->shieldBreakReward = SHIELD_BREAK_REWARD;
    e->shotHitRewardCoef = SHOT_HIT_REWARD_COEF;
    e->explosionHitRewardCoef = EXPLOSION_HIT_REWARD_COEF;

    e->obsSize = obsSize(e->numDrones);
    e->discreteObsSize = discreteObsSize(e->numDrones);

    e->continuousActions = continuousActions;

    setEnvFrameRate(e);
    e->rng = seed;
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

void setRewards(iwEnv *e, float winReward, float selfKillPunishment, float enemyDeathReward, float enemyKillReward, float teammateDeathPunishment, float teammateKillPunishment, float deathPunishment, float energyEmptiedPunishment, float weaponPickupReward, float shieldBreakReward, float shotHitRewardCoef, float explosionHitRewardCoef) {
    e->winReward = winReward;
    e->selfKillPunishment = selfKillPunishment;
    e->enemyDeathReward = enemyDeathReward;
    e->enemyKillReward = enemyKillReward;
    e->teammateDeathPunishment = teammateDeathPunishment;
    e->teammateKillPunishment = teammateKillPunishment;
    e->deathPunishment = deathPunishment;
    e->energyEmptiedPunishment = energyEmptiedPunishment;
    e->weaponPickupReward = weaponPickupReward;
    e->shieldBreakReward = shieldBreakReward;
    e->shotHitRewardCoef = shotHitRewardCoef;
    e->explosionHitRewardCoef = explosionHitRewardCoef;
}

void clearEnv(iwEnv *e) {
    // rewards get cleared in stepEnv every step
    // memset(e->masks, 1, e->numAgents * sizeof(uint8_t));

    e->episodeLength = 0;
    memset(e->stats, 0x0, sizeof(e->stats));

    for (size_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        destroyDrone(e, drone);
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

    for (size_t i = 0; i < cc_array_size(e->dronePieces); i++) {
        dronePieceEntity *piece = safe_array_get_at(e->dronePieces, i);
        destroyDronePiece(e, piece);
    }

    cc_array_remove_all(e->drones);
    cc_array_remove_all(e->floatingWalls);
    cc_array_remove_all(e->pickups);
    cc_array_remove_all(e->projectiles);
    cc_array_remove_all(e->explodingProjectiles);
    cc_array_remove_all(e->explosions);
    cc_array_remove_all(e->dronePieces);
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

float computeReward(iwEnv *e, droneEntity *drone) {
    float reward = 0.0f;

    if (drone->energyFullyDepleted && drone->energyRefillWait == DRONE_ENERGY_REFILL_EMPTY_WAIT) {
        reward += e->energyEmptiedPunishment;
    }

    // only reward picking up a weapon if the standard weapon was
    // previously held; every weapon is better than the standard
    // weapon, but other weapons are situational better so don't
    // reward switching a non-standard weapon
    if (drone->stepInfo.pickedUpWeapon && drone->stepInfo.prevWeapon == STANDARD_WEAPON) {
        reward += e->weaponPickupReward;
    }

    for (uint8_t i = 0; i < e->numDrones; i++) {
        if (i == drone->idx) {
            continue;
        }
        droneEntity *enemyDrone = safe_array_get_at(e->drones, i);
        const bool onTeam = drone->team == enemyDrone->team;

        // TODO: punish for hitting teammates?
        if (drone->stepInfo.shotHit[i] != 0.0f && !onTeam) {
            reward += drone->stepInfo.shotHit[i] * e->shotHitRewardCoef;
        }
        if (drone->stepInfo.explosionHit[i] != 0.0f && !onTeam) {
            reward += drone->stepInfo.explosionHit[i] * e->explosionHitRewardCoef;
        }
        if (drone->stepInfo.brokeShield[i] && !onTeam) {
            reward += e->shieldBreakReward;
        }

        if (e->numAgents == e->numDrones) {
            if (drone->stepInfo.shotTaken[i] != 0) {
                reward -= drone->stepInfo.shotTaken[i] * e->shotHitRewardCoef;
            }
            if (drone->stepInfo.explosionTaken[i]) {
                reward -= drone->stepInfo.explosionTaken[i] * e->explosionHitRewardCoef;
            }
        }

        if (enemyDrone->dead && enemyDrone->diedThisStep) {
            if (!onTeam) {
                reward += e->enemyDeathReward;
                if (drone->killed[i]) {
                    reward += e->enemyKillReward;
                }
            } else {
                reward += e->teammateDeathPunishment;
                if (drone->killed[i]) {
                    reward += e->teammateKillPunishment;
                }
            }
            continue;
        }

        // const b2Vec2 enemyDirection = b2Normalize(b2Sub(enemyDrone->pos, drone->pos));
        // const float velocityToEnemy = b2Dot(drone->lastVelocity, enemyDirection);
        // const float enemyDistance = b2Distance(enemyDrone->pos, drone->pos);
        // // stop rewarding approaching an enemy if they're very close
        // // to avoid constant clashing; always reward approaching when
        // // the current weapon is the shotgun, it greatly benefits from
        // // being close to enemies
        // if (velocityToEnemy > 0.1f && (drone->weaponInfo->type == SHOTGUN_WEAPON || enemyDistance > DISTANCE_CUTOFF)) {
        //     reward += APPROACH_REWARD;
        // }
    }

    return reward;
}

static const float REWARD_EPS = 1.0e-6f;

void computeRewards(iwEnv *e, const bool roundOver, const int8_t winner, const int8_t winningTeam) {
    for (uint8_t i = 0; i < e->numDrones; i++) {
        float reward = 0.0f;
        droneEntity *drone = safe_array_get_at(e->drones, i);
        reward = computeReward(e, drone);
        if (!drone->dead && roundOver && (winner == i || winningTeam == drone->team)) {
            reward += e->winReward;
        } else if (drone->diedThisStep) {
            reward = e->deathPunishment;
            if (drone->killedBy == drone->idx) {
                reward += e->selfKillPunishment;
            }
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
            actions.chargingWeapon = envActions[4] > 0.0f;
            actions.brake = envActions[5] > 0.0f;
            actions.chargingBurst = envActions[6] > 0.0f;
        } else {
            uint8_t move = envActions[0];
            // 0 is no-op for both move and aim
            ASSERT(move <= 8);
            if (move != 0) {
                move--;
                actions.move.x = discMoveToContMoveMap[0][move];
                actions.move.y = discMoveToContMoveMap[1][move];
            }
            uint8_t aim = envActions[1];
            ASSERT(aim <= 16);
            if (aim != 0) {
                aim--;
                actions.aim.x = discAimToContAimMap[0][aim];
                actions.aim.y = discAimToContAimMap[1][aim];
            }

            actions.chargingWeapon = envActions[2] > 0.0f;
            actions.brake = envActions[3] > 0.0f;
            actions.chargingBurst = envActions[4] > 0.0f;
        }

        actions.shoot = actions.chargingWeapon;
        if (!actions.chargingWeapon && drone->chargingWeapon) {
            actions.shoot = true;
        }
    } else {
        actions.move = manualActions->move;
        actions.aim = manualActions->aim;
        actions.chargingWeapon = manualActions->chargingWeapon;
        actions.shoot = manualActions->shoot;
        actions.brake = manualActions->brake;
        actions.chargingBurst = manualActions->chargingBurst;
        actions.discardWeapon = manualActions->discardWeapon;
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
    if (!b2VecEqual(actions.aim, b2Vec2_zero)) {
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

    for (uint8_t j = 0; j < e->numDrones; j++) {
        e->log.stats[j].returns += log->stats[j].returns;
        e->log.stats[j].wins += log->stats[j].wins;

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

        for (uint8_t k = 0; k < NUM_WEAPONS; k++) {
            e->log.stats[j].shotsFired[k] += log->stats[j].shotsFired[k];
            e->log.stats[j].shotsHit[k] += log->stats[j].shotsHit[k];
            e->log.stats[j].shotsTaken[k] += log->stats[j].shotsTaken[k];
            e->log.stats[j].ownShotsTaken[k] += log->stats[j].ownShotsTaken[k];
            e->log.stats[j].weaponsPickedUp[k] += log->stats[j].weaponsPickedUp[k];
            e->log.stats[j].shotDistances[k] += log->stats[j].shotDistances[k];
        }

        e->log.stats[j].totalShotsFired += log->stats[j].totalShotsFired;
        e->log.stats[j].totalShotsHit += log->stats[j].totalShotsHit;
        e->log.stats[j].totalShotsTaken += log->stats[j].totalShotsTaken;
        e->log.stats[j].totalOwnShotsTaken += log->stats[j].totalOwnShotsTaken;
        e->log.stats[j].totalWeaponsPickedUp += log->stats[j].totalWeaponsPickedUp;
        e->log.stats[j].totalShotDistances += log->stats[j].totalShotDistances;
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
            explosion->renderSteps = e->client->maxExplosionLifetime;
        } else if (explosion->renderSteps == 0) {
            fastFree(explosion);
            cc_array_iter_remove(&iter, NULL);
        } else {
            explosion->renderSteps = max(explosion->renderSteps - 1, 0);
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

void updateVisuals(iwEnv *e) {
    updateExplosions(e);
    updateDronePieces(e);

    for (uint8_t i = 0; i < cc_array_size(e->drones); i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        updateDroneBrakeTrail(e, drone);
        updateDroneRespawnGuide(e, drone);
    }
}

// TODO: 2nd agent doesn't seem to work right
void stepPhysicsFrame(iwEnv *e, const agentActions stepActions[]) {
    for (uint8_t i = 0; i < e->numDrones; i++) {
        droneEntity *drone = safe_array_get_at(e->drones, i);
        if (drone->dead) {
            continue;
        }
        agentActions actions = stepActions[i];

        if (actions.discardWeapon) {
            droneDiscardWeapon(e, drone);
        }
        if (actions.shoot) {
            droneShoot(e, drone, actions.aim, actions.chargingWeapon);
        }
        if (actions.chargingBurst) {
            droneChargeBurst(e, drone);
        } else if (drone->chargingBurst) {
            droneBurst(e, drone);
        }
        if (!b2VecEqual(actions.move, b2Vec2_zero)) {
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

    int8_t lastAlive = -1;
    int8_t lastAliveTeam = -1;
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
            }
            lastAlive = i;

            if (e->teamsEnabled) {
                if (lastAliveTeam == -1) {
                    lastAliveTeam = drone->team;
                    allAliveOnSameTeam = true;
                } else if (drone->team != lastAliveTeam) {
                    allAliveOnSameTeam = false;
                }
            }
        } else {
            deadDrones++;
            if (i < e->numAgents) {
                if (drone->diedThisStep) {
                    agentTerminals(e, i)[0] = 1.0f;
                }
                // else {
                //     e->masks[i] = 0;
                // }
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
    computeRewards(e, roundOver, lastAlive, lastAliveTeam);

    if (roundOver) {
        if (e->numDrones != e->numAgents && e->stepsLeft == 0) {
            DEBUG_LOG("truncating episode");
        } else {
            DEBUG_LOG("terminating episode");
        }

        for (uint8_t t = 0; t < e->numAgents; t++) {
            agentTerminals(e, t)[0] = 1.0f;
        }

        Log log = {0};
        log.length = e->episodeLength;
        if (lastAlive != -1) {
            e->stats[lastAlive].wins = 1.0f;
        } else if (!e->teamsEnabled || (e->teamsEnabled && lastAliveTeam == -1)) {
            log.ties = 1.0f;
        }
        log.botCLNoise = e->botCLNoise;

        // TODO: handle multiple agents/teams correctly
        if (e->botCLNoise > 0.0f && lastAlive == 0) {
            e->botCLNoise -= e->botCLDecay;
            e->botCLNoise = clamp(e->botCLNoise);
        }

        for (uint8_t i = 0; i < e->numDrones; i++) {
            const droneEntity *drone = safe_array_get_at(e->drones, i);
            if (!drone->dead && e->teamsEnabled && drone->team == lastAliveTeam) {
                e->stats[i].wins = 1.0f;
            }
            // set absolute distance traveled of agent drones
            e->stats[i].absDistanceTraveled = b2Distance(drone->initalPos, drone->pos);
        }

        memcpy(log.stats, e->stats, sizeof(e->stats));
        addLog(e, &log);

        if (e->client != NULL) {
            e->roundState = ROUND_STATE_ENDING;
            e->tick_frames_left = END_WAIT_TIME * e->frameRate;

            e->winner = lastAlive;
            e->winningTeam = lastAliveTeam;
        }

        e->needsReset = true;
    }

    if (e->client != NULL) {
        updateVisuals(e);
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

    for (int i = 0; i < e->frameSkip; i++) {
        e->episodeLength++;
        stepPhysicsFrame(e, e->cachedActions);
        e->tick_frames_left--;

        if (e->needsReset) {
            break;
        }
    }

    if (e->client == NULL && e->needsReset) {
        puf_reset(e);
    } else if (e->client == NULL || e->tick_frames_left == 0) {
        computeObs(e);
    }
}

#endif
