#ifndef IMPULSE_WARS_TYPES_H
#define IMPULSE_WARS_TYPES_H

#include "box2d/box2d.h"
#include "id_pool.h"
#include "raylib.h"
#include "rlights.h"

#define PUF_HAS_BOT_POLICY
#include "pufferenv.h"

#include "include/cc_array.h"

#define _MAX_DRONES 4

// discrete aim is split into a coarse sector and a fine offset within
// it, giving AIM_COARSE_SECTORS * AIM_FINE_STEPS evenly spaced directions
// while only needing AIM_COARSE_SECTORS + AIM_FINE_STEPS logits; lives
// here so the ACT_SIZES in impulse_wars.h can use it
#define AIM_COARSE_SECTORS 16
#define AIM_FINE_STEPS 8

static const uint8_t NUM_WALL_TYPES = 3;

#define MAX_TRAIL_POINTS 20
#define MAX_DRONE_TRAIL_POINTS 20
#define MAX_PROJECTLE_TRAIL_POINTS 10

enum entityType {
    STANDARD_WALL_ENTITY,
    BOUNCY_WALL_ENTITY,
    DEATH_WALL_ENTITY,
    WEAPON_PICKUP_ENTITY,
    PROJECTILE_ENTITY,
    DRONE_ENTITY,
    SHIELD_ENTITY,
    DRONE_PIECE_ENTITY,
};

// the category bit that will be set on each entity's shape; this is
// used to control what entities can collide with each other
enum shapeCategory {
    WALL_SHAPE = 1,
    FLOATING_WALL_SHAPE = 2,
    PROJECTILE_SHAPE = 4,
    WEAPON_PICKUP_SHAPE = 8,
    DRONE_SHAPE = 16,
    SHIELD_SHAPE = 32,
    DRONE_PIECE_SHAPE = 64,
};

typedef struct entityID {
    int32_t id;
    uint16_t generation;
} entityID;

// general purpose entity object
typedef struct entity {
    entityID *id;
    uint32_t generation;
    enum entityType type;
    void *entity;
} entity;

#define _NUM_WEAPONS 10
static const uint8_t NUM_WEAPONS = _NUM_WEAPONS;

enum weaponType {
    STANDARD_WEAPON,
    MACHINEGUN_WEAPON,
    SNIPER_WEAPON,
    SHOTGUN_WEAPON,
    IMPLODER_WEAPON,
    ACCELERATOR_WEAPON,
    FLAK_CANNON_WEAPON,
    MINE_LAUNCHER_WEAPON,
    BLACK_HOLE_WEAPON,
    NUKE_WEAPON,
};

typedef struct mapBounds {
    b2Vec2 min;
    b2Vec2 max;
} mapBounds;

// used for N near entities observations
typedef struct nearEntity {
    uint16_t idx;
    void *entity;
    float distanceSquared;
} nearEntity;

// One of the nearest static walls to a map cell. Static wall geometry is
// fixed per map, so position and type are stored alongside the index here
// and callers never have to chase a pointer into the per-env wall array to
// compute distances or fill observations.
typedef struct nearWall {
    float distanceSquared;
    uint16_t wallIdx;
    uint8_t type;
    b2Vec2 pos;
} nearWall;

typedef struct mapEntry {
    const char *layout;
    const uint8_t columns;
    const uint8_t rows;
    const uint8_t randFloatingStandardWalls;
    const uint8_t randFloatingBouncyWalls;
    const uint8_t randFloatingDeathWalls;
    // are there any floating walls that have consistent starting positions
    const bool hasSetFloatingWalls;
    const uint16_t weaponPickups;
    const enum weaponType defaultWeapon;
    const uint8_t maxSuddenDeathWalls;

    mapBounds bounds;
    mapBounds spawnQuads[4];
    bool *droneSpawns;
    float *packedLayout;
    nearWall *nearestWalls;
    // flow field for every (destination, source) cell pair; identical for
    // every env on this map so it's built once in initMaps and read-only
    // afterwards
    uint8_t *paths;
} mapEntry;

// a cell in the map; ent will be NULL if the cell is empty
typedef struct mapCell {
    entity *ent;
    b2Vec2 pos;
} mapCell;

typedef struct wallEntity {
    b2BodyId bodyID;
    b2ShapeId shapeID;
    b2Vec2 pos;
    b2Rot rot;
    b2Vec2 velocity;
    b2Vec2 extent;
    int16_t mapCellIdx;
    bool isFloating;
    enum entityType type;
    bool isSuddenDeath;
    CC_Array *physicsTracking;

    entity *ent;
} wallEntity;

typedef struct weaponInformation {
    const enum weaponType type;
    // should the body be treated as a bullet by box2d; if so CCD
    // (continuous collision detection) will be enabled to prevent
    // tunneling through static bodies which is expensive so it's
    // only enabled for fast moving projectiles
    const bool isPhysicsBullet;
    // can the projectile ever be stationary? if so, it should be
    // allowed to sleep to save on physics updates
    const bool canSleep;
    const uint8_t numProjectiles;
    const float fireMagnitude;
    const float recoilMagnitude;
    const float damping;
    const float charge;
    const float coolDown;
    const float maxDistance;
    const float radius;
    const float density;
    const float mass;
    const float invMass;
    const float initialSpeed;
    const uint8_t maxBounces;
    const bool explosive;
    const bool destroyedOnDroneHit;
    const bool explodesOnDroneHit;
    const bool hasSensor;
    const float energyRefillCoef;
    const float spawnWeight;
} weaponInformation;

typedef struct weaponPickupEntity {
    b2BodyId bodyID;
    b2ShapeId shapeID;
    enum weaponType weapon;
    float respawnWait;
    // how many floating walls are touching this pickup
    uint8_t floatingWallsTouching;
    b2Vec2 pos;
    int16_t mapCellIdx;

    entity *ent;
    bool bodyDestroyed;
} weaponPickupEntity;

typedef struct droneEntity droneEntity;

typedef struct trailPoints {
    Vector2 points[MAX_TRAIL_POINTS];
    uint8_t length;
} trailPoints;

typedef struct projectileEntity {
    uint8_t droneIdx;

    b2BodyId bodyID;
    b2ShapeId shapeID;
    // used for proximity explosive projectiles
    b2ShapeId sensorID;
    weaponInformation *weaponInfo;
    b2Vec2 pos;
    int16_t mapCellIdx;
    b2Vec2 lastPos;
    b2Vec2 velocity;
    b2Vec2 lastVelocity;
    float speed;
    float lastSpeed;
    float distance;
    uint8_t bounces;
    uint8_t contacts;
    bool setMine;
    uint8_t numDronesBehindWalls;
    uint8_t dronesBehindWalls[_MAX_DRONES];
    CC_Array *entsInBlackHole;
    bool needsToBeDestroyed;

    entity *ent;

    trailPoints *trailPoints;
} projectileEntity;

// used to keep track of what happened each step for reward purposes
typedef struct droneStepInfo {
    bool firedShot;
    bool pickedUpWeapon;
    bool emptiedEnergy;
    enum weaponType prevWeapon;
    float shotHit[_MAX_DRONES];
    float explosionHit[_MAX_DRONES];
    float shotTaken[_MAX_DRONES];
    float explosionTaken[_MAX_DRONES];
    bool brokeShield[_MAX_DRONES];
    bool ownShotTaken;
    // summed cos of the aim error of shots fired this step
    float aimAlignment;
} droneStepInfo;

typedef struct shieldEntity {
    droneEntity *drone;

    b2BodyId bodyID;
    b2ShapeId shapeID;
    b2ShapeId bufferShapeID;
    b2Vec2 pos;
    float health;
    float duration;

    entity *ent;
} shieldEntity;

typedef struct dronePieceEntity {
    uint8_t droneIdx;

    b2BodyId bodyID;
    b2ShapeId shapeID;
    b2Vec2 pos;
    b2Rot rot;
    b2Vec2 vertices[3];
    bool isShieldPiece;

    entity *ent;

    uint16_t lifetime;
} dronePieceEntity;

enum physicsType {
    INVALID_PHYSICS,
    MOVEMENT_PHYSICS,
    RECOIL_PHYSICS,
    BURST_PHYSICS,
    PROJECTILE_PHYSICS,
    EXPLOSION_PHYSICS,
    DRONE_PHYSICS,
};
#define NUM_PHYSICS_TYPES (DRONE_PHYSICS + 1)

typedef struct physicsSource {
    enum physicsType type;
    enum weaponType weapon;
} physicsSource;

typedef struct physicsStepInfo {
    uint8_t srcIdx;
    b2Vec2 impulse;
    b2Vec2 force;
    bool brakeToggled;
    // set when the body bounced off of a static wall; normal points out
    // of the wall and restitution is what box2d used for the bounce
    bool bounce;
    b2Vec2 normal;
    float restitution;
    physicsSource source;
    uint16_t step;
} physicsStepInfo;

typedef struct droneEntity {
    b2BodyId bodyID;
    b2ShapeId shapeID;
    weaponInformation *weaponInfo;
    int8_t ammo;
    float weaponCooldown;
    uint16_t heat;
    bool chargingWeapon;
    float weaponCharge;
    float energyLeft;
    bool braking;
    bool chargingBurst;
    float burstCharge;
    float burstCooldown;
    bool energyFullyDepleted;
    bool energyFullyDepletedThisStep;
    float energyRefillWait;
    bool shotThisStep;
    bool diedThisStep;

    uint8_t idx;
    uint8_t team;
    b2Vec2 initalPos;
    b2Vec2 pos;
    int16_t mapCellIdx;
    b2Vec2 lastPos;
    b2Vec2 lastMove;
    b2Vec2 lastAim;
    b2Vec2 velocity;
    b2Vec2 lastVelocity;
    droneStepInfo stepInfo;
    float respawnWait;
    uint8_t livesLeft;
    bool dead;

    CC_Array *physicsTracking;
    int8_t killedBy;
    physicsSource killSource;
    bool killed[_MAX_DRONES];

    shieldEntity *shield;
    entity *ent;

    trailPoints *trailPoints;
    CC_Array *brakeTrailPoints;
    uint16_t respawnGuideLifetime;
} droneEntity;

// stats for the whole episode
typedef struct droneStats {
    float returns;
    float distanceTraveled;
    float absDistanceTraveled;
    float brakeTime;
    float totalBursts;
    float burstsHit;
    float energyEmptied;
    float shieldsBroken;
    float ownShieldBroken;
    float selfKills;
    float kills;
    float unknownKills;
    float score;

    float shotsFired[_NUM_WEAPONS];
    float shotsHit[_NUM_WEAPONS];
    float shotsTaken[_NUM_WEAPONS];
    float ownShotsTaken[_NUM_WEAPONS];
    float weaponsPickedUp[_NUM_WEAPONS];
    float shotDistances[_NUM_WEAPONS];

    float totalShotsFired;
    float totalShotsHit;
    float totalShotsTaken;
    float totalOwnShotsTaken;
    float totalWeaponsPickedUp;
    float totalShotDistances;
    // aim error vs the bearing to the nearest enemy, sampled when firing
    float shotsAimed10;
    float shotsAimed30;
    float totalAimError;
    float shotsAimedAway;
} droneStats;

struct Log {
    float length;
    float ties;
    float botCLNoise;
    float easyEpProb;
    float easyEp;
    // agent shots in normal (non easy) episodes, to see if aiming
    // carries over once bursts are enabled
    float normalShotsFired;
    float normalShotsAimed10;
    droneStats stats[_MAX_DRONES];

    float n;
};

enum roundState {
    ROUND_STATE_INVALID,
    ROUND_STATE_STARTING,
    ROUND_STATE_PLAYING,
    ROUND_STATE_ENDING,
};

typedef struct gameCamera {
    Camera3D camera3D;
    Camera2D camera2D;
    Vector2 targetPos;
    float maxZoom;
    bool orthographic;
} gameCamera;

typedef struct rayClient {
    float scale;
    uint16_t width;
    uint16_t height;
    uint16_t halfWidth;
    uint16_t halfHeight;

    gameCamera *camera;

    Shader blurShader;
    int32_t blurShaderDirLoc;
    Shader bloomShader;
    int32_t bloomIntensityLoc;
    int32_t bloomTexColorLoc;
    int32_t bloomTexBloomBlurLoc;
    Shader gridShader;
    int32_t gridShaderPosLoc[4];
    int32_t gridShaderColorLoc[4];
    Shader explosionShader;
    int32_t explosionResolutionLoc;
    int32_t explosionBlastsLoc;
    int32_t explosionAmplitudesLoc;
    int32_t explosionCountLoc;
    Texture2D wallTexture;
    RenderTexture2D blurSrcTexture;
    RenderTexture2D blurDstTexture;
    RenderTexture2D projRawTex;
    RenderTexture2D projBloomTex;
    RenderTexture2D droneRawTex;
    RenderTexture2D droneBloomTex;
    RenderTexture2D explosionSceneTex;

    float maxDronePieceLifetime;
    float maxBrakeTrailLifetime;
    float maxDroneRespawnGuideLifetime;
} rayClient;

typedef struct brakeTrailPoint {
    b2Vec2 pos;
    uint16_t lifetime;
    bool isEnd;
} brakeTrailPoint;

typedef struct explosionInfo {
    b2ExplosionDef def;
    bool isBurst;
    bool isBlackHole;
    uint8_t droneIdx;
    uint16_t renderSteps;
    uint16_t maxRenderSteps;
} explosionInfo;

typedef struct agentActions {
    b2Vec2 move;
    b2Vec2 aim;
    bool chargingWeapon;
    bool shoot;
    bool brake;
    bool chargingBurst;
    bool discardWeapon;
} agentActions;

typedef struct debugPoint {
    b2Vec2 pos;
    float size;
    Color color;
} debugPoint;

// Scripted opponents, weakest first. These are the rungs of [selfplay] eval_bots.
enum botPolicy {
    BOT_SITTING_DUCK = 0,
    BOT_SCRIPTED = 1,
};

// scripted bot behaviors, toggled per drone so variants of the bot can be
// played against each other in the same env (see bot_tournament.c)
enum botFeature {
    BOT_FEAT_AMMO_RESERVE = 1 << 0,
    BOT_FEAT_PICKUP_SCORE = 1 << 1,
    BOT_FEAT_PROJ_DEFENSE = 1 << 2,
    BOT_FEAT_FLAK_RANGE = 1 << 3,
    BOT_FEAT_AIM = 1 << 4,
    BOT_FEAT_WALL_AVOID = 1 << 5,
    BOT_FEAT_BURST = 1 << 6,
    BOT_FEAT_BURST_BOOST = 1 << 7,
    BOT_FEAT_KILL_GEOMETRY = 1 << 8,
    BOT_FEAT_SNIPER_DODGE = 1 << 9,
    BOT_FEAT_ESCAPE_FIRE = 1 << 10,
    BOT_FEAT_LONG_SHOTS = 1 << 11,
    BOT_FEAT_SMART_RESERVE = 1 << 12,
};
#define BOT_FEAT_ALL 0x1FFF

// how many cells ahead along a planned route the scripted bot remembers
#define BOT_PATH_WAYPOINTS 4

// per drone scripted bot state that persists between actions
typedef struct scriptedBotState {
    // pickup the bot committed to, and the step it gives up on it
    int8_t pickupIdx;
    uint16_t pickupUntilStep;
    // side the bot sidesteps to when dodging, kept for a few steps to
    // avoid oscillating
    int8_t dodgeSide;
    uint16_t dodgeUntilStep;
    // last planned route, reused until the drone changes cells, the
    // destination changes or it expires
    int16_t pathSrcIdx;
    int16_t pathDstIdx;
    uint16_t pathUntilStep;
    int16_t pathWaypoints[BOT_PATH_WAYPOINTS];
    uint8_t pathNumWaypoints;
} scriptedBotState;

// counts of scripted bot decisions, for tuning the bot; never reset by
// the env
enum botEvent {
    BOT_EVENT_THREAT,
    BOT_EVENT_THREAT_SHOT,
    BOT_EVENT_THREAT_BURST,
    BOT_EVENT_SNIPER_DODGE,
    BOT_EVENT_SNIPER_URGENT,
    BOT_EVENT_BURST_START,
    BOT_EVENT_BOOST_RELEASE,
    BOT_EVENT_KILL_GEOMETRY,
    BOT_EVENT_PICKUP_SKIPPED,
    BOT_EVENT_AMMO_RESERVED,
    BOT_EVENT_CLEARANCE_STEER,
    BOT_EVENT_ESCAPE_FIRE,
    BOT_EVENT_LONG_SHOT,
    BOT_EVENT_SPARE_AMMO_MOVE,
    BOT_EVENT_DYNAMIC_PATH,
    BOT_EVENT_DYNAMIC_PATH_FAILED,
    NUM_BOT_EVENTS,
};

struct Env {
    Agent agents[_MAX_DRONES];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;

    uint8_t numDrones;
    uint8_t numAgents;
    uint8_t numTeams;
    bool teamsEnabled;
    enum botPolicy botPolicy;
    bool isTraining;

    float botCLNoise;
    float botCLDecay;
    uint32_t botFeatures[_MAX_DRONES];
    scriptedBotState botState[_MAX_DRONES];
    uint32_t botEvents[_MAX_DRONES][NUM_BOT_EVENTS];

    // easy episode curriculum: with probability easyEpProb an episode is
    // played on the empty map against a sitting duck with bursts disabled;
    // the probability decays
    // each easy episode the agent shoots accurately enough
    float easyEpProb;
    float easyEpDecay;
    float easyEpHitRate;
    bool easyEp;

    float winReward;
    float lifeReward;
    float killReward;
    float selfKillPunishment;
    float teammateDeathPunishment;
    float teammateKillPunishment;
    float energyEmptiedPunishment;
    float weaponPickupReward;
    float shieldBreakReward;
    float hitRewardCoef;
    float aimDrillReward;

    uint16_t obsSize;
    bool continuousActions;

    uint8_t frameRate;
    float deltaTime;
    uint8_t frameSkip;
    uint8_t box2dSubSteps;
    bool needsReset;

    uint16_t episodeLength;
    Log log;
    droneStats stats[_MAX_DRONES];

    b2WorldId worldID;
    int8_t pinnedMapIdx;
    int8_t mapIdx;
    mapEntry *map;
    int8_t lastSpawnQuad;
    uint8_t spawnedWeaponPickups[_NUM_WEAPONS];
    weaponInformation *defaultWeapon;
    b2IdPool idPool;
    CC_Array *entities;
    // flat, allocated once at MAX_CELLS; map layouts only ever shrink it
    mapCell *cells;
    uint16_t numCells;
    CC_Array *walls;
    CC_Array *floatingWalls;
    CC_Array *drones;
    CC_Array *pickups;
    CC_Array *projectiles;
    CC_Array *explodingProjectiles;
    CC_Array *dronePieces;

    uint16_t totalSteps;
    uint16_t totalSuddenDeathSteps;
    // steps left until sudden death
    uint16_t stepsLeft;
    // steps left until the next set of sudden death walls are spawned
    uint16_t suddenDeathSteps;
    // the amount of sudden death walls that have been spawned
    uint8_t suddenDeathWallCounter;
    bool suddenDeathWallsPlaced;

    bool humanInput;
    uint8_t humanDroneInput;
    uint8_t connectedControllers;
    agentActions *cachedActions;

    // used for rendering
    rayClient *client;
    float renderScale;
    CC_Array *explosions;
    CC_Array *debugPoints;
    enum roundState roundState;
    uint16_t tick_frames_left;
    int8_t winner;
    int8_t winningTeam;
};
typedef struct Env iwEnv;

#endif
