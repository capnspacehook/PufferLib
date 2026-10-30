#ifndef IMPULSE_WARS_TYPES_H
#define IMPULSE_WARS_TYPES_H

#include "box2d/box2d.h"
#include "id_pool.h"
#include "raylib.h"
#include "rlights.h"

#define PUF_HAS_BOT_POLICY
#include "pufferenv.h"

#include "include/cc_array.h"

// set by IW_NUM_DRONES in impulse_wars.h
#define _MAX_DRONES IW_NUM_DRONES

#define _MAX_PORTALS 2

#define _MAX_MIRRORS 2

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
#define MAX_BOOSTED_PROJECTLE_TRAIL_POINTS 15

enum entityType {
    STANDARD_WALL_ENTITY,
    BOUNCY_WALL_ENTITY,
    DEATH_WALL_ENTITY,
    WEAPON_PICKUP_ENTITY,
    PROJECTILE_ENTITY,
    DRONE_ENTITY,
    SHIELD_ENTITY,
    DRONE_PIECE_ENTITY,
    PORTAL_ENTITY,
    MIRROR_ENTITY,
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
    PORTAL_SHAPE = 128,
    MIRROR_SHAPE = 256,
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
    uint16_t lastBurstHit[_MAX_DRONES];
} entity;

#define _NUM_WEAPONS 13
static const uint8_t NUM_WEAPONS = _NUM_WEAPONS;

enum weaponType {
    UNSET_WEAPON,
    STANDARD_WEAPON,
    MACHINEGUN_WEAPON,
    SNIPER_WEAPON,
    SHOTGUN_WEAPON,
    IMPLODER_WEAPON,
    FLAK_CANNON_WEAPON,
    BLACK_HOLE_WEAPON,
    NUKE_WEAPON,
    GRAPPLE_WEAPON,
    MINE_LAUNCHER_WEAPON,
    PORTAL_WEAPON,
    MIRROR_WEAPON,
};

// utilities live in their own slot next to the weapon and survive death;
// their ammo regenerates over time. Their projectiles are
// still weapon types (see utilityInformation) so physics, kill credit and
// per weapon stats treat them like any other projectile. The mirror isn't a
// projectile, but it has a weapon type with no projectiles for its recoil
#define _NUM_UTILITIES 5
static const uint8_t NUM_UTILITIES = _NUM_UTILITIES;

enum utilityType {
    UNSET_UTILITY,
    GRAPPLE_UTILITY,
    MINE_UTILITY,
    PORTAL_UTILITY,
    MIRROR_UTILITY,
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

// an optional per map override of a pickup spawn weight; zero initialized
// overrides aren't set so the default weight is used
typedef struct spawnWeightOverride {
    bool set;
    float weight;
} spawnWeightOverride;

#define SPAWN_WEIGHT(w) {.set = true, .weight = (w)}

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
    const uint8_t maxSuddenDeathWalls;
    // overrides of the default pickup spawn weights, indexed by weapon
    // and utility type
    const spawnWeightOverride weaponSpawnWeights[_NUM_WEAPONS];
    const spawnWeightOverride utilitySpawnWeights[_NUM_UTILITIES];
    const spawnWeightOverride crateSpawnWeight;

    // pickup spawn weights with overrides applied, set in initMaps
    float weaponWeights[_NUM_WEAPONS];
    float utilityWeights[_NUM_UTILITIES];
    float crateWeight;
    mapBounds bounds;
    mapBounds spawnQuads[4];
    uint16_t quadCells;
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
    // max ammo, INFINITE for the standard weapon
    const int8_t ammo;
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
    const float maxCharge;
    const float cooldown;
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
    const float burstBoostCoef;
    const float spawnWeight;
} weaponInformation;

typedef struct utilityInformation {
    const enum utilityType type;
    // the weapon whose projectile the utility fires
    const weaponInformation *projectileInfo;
    const uint8_t ammo;
    // seconds to regenerate one ammo
    const float regenTime;
    // how many of its projectiles can exist at once
    const int8_t maxActive;
    const float spawnWeight;
} utilityInformation;

enum pickupKind {
    WEAPON_PICKUP,
    UTILITY_PICKUP,
    CRATE_PICKUP,
};

typedef struct weaponPickupEntity {
    b2BodyId bodyID;
    b2ShapeId shapeID;
    enum pickupKind kind;
    // UNSET_WEAPON unless kind is WEAPON_PICKUP
    enum weaponType weapon;
    // UNSET_UTILITY unless kind is UTILITY_PICKUP
    enum utilityType utility;
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
    // set once an enemy turned the projectile enough to take it over;
    // its pushes on its new owner are credited to prevDroneIdx
    bool deflected;
    uint8_t prevDroneIdx;
    // the drone that fired it, which droneIdx no longer is once deflected
    uint8_t firerIdx;
    // direction and the other projectile's owner when a contact with
    // another projectile began, to check for a deflection when it ends
    bool inProjContact;
    b2Vec2 projContactDir;
    uint8_t projContactOwner;

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
    bool welded;
    bool needsToBeDestroyed;
    bool boosted;

    uint8_t numDronesBehindWalls;
    uint8_t dronesBehindWalls[_MAX_DRONES];
    CC_Array *entsInBlackHole;

    entity *grappledEnt;
    float grappleHealth;

    entity *ent;

    trailPoints *trailPoints;
} projectileEntity;

// a mote drifting into a portal and out of the linked one, purely visual;
// drawn from renderRng when it starts so motes are random but stable across
// frames
#define _NUM_PORTAL_MOTES 4
typedef struct portalMote {
    // game time in seconds the mote starts going in, and how long it takes
    // to come out of the linked portal; 0 duration until it's first started
    float start;
    float duration;
    float offset;
    float height;
} portalMote;

typedef struct portalEntity portalEntity;
struct portalEntity {
    uint8_t droneIdx;

    // the sensor lives on the attached wall's body so it follows
    // floating walls exactly
    b2ShapeId sensorID;
    wallEntity *attachedWall;
    // center of the portal on the wall face and the face's outward
    // normal, in the attached wall's local frame
    b2Vec2 localPos;
    b2Vec2 localNormal;
    // world space versions of the above, updated when the wall moves
    b2Vec2 pos;
    b2Vec2 normal;
    int16_t mapCellIdx;
    portalEntity *linkedPortal;
    // drones overlapping the sensor that weren't allowed through yet, so
    // they can be teleported once they turn towards the portal
    bool droneInRange[_MAX_DRONES];
    // frames left of the flash and swirl shown when a drone goes through,
    // UINT16_MAX until the render code starts it
    uint16_t transitRenderSteps;
    // the drone went in this portal, which swirls the opposite way to the
    // exit portal
    bool transitEntry;
    // frames left of the line drawn to the linked portal when this portal
    // formed the link, UINT16_MAX until the render code starts it
    uint16_t linkRenderSteps;
    // motes going into this portal, reset when it's unlinked
    portalMote motes[_NUM_PORTAL_MOTES];

    entity *ent;
};

// a panel that drifts in a straight line, through everything but
// projectiles, and reverses every projectile that hits it; a kinematic body
// so nothing can push it
typedef struct mirrorEntity {
    uint8_t droneIdx;

    b2BodyId bodyID;
    b2ShapeId shapeID;
    b2Vec2 pos;
    b2Rot rot;
    b2Vec2 velocity;
    int16_t mapCellIdx;
    // seconds since it was placed
    float age;
    // half its width along the face, which grows to MIRROR_WIDTH / 2
    float halfWidth;

    entity *ent;
} mirrorEntity;

// used to keep track of what happened each step for reward purposes
typedef struct droneStepInfo {
    bool firedShot;
    bool pickedUpWeapon;
    bool emptiedEnergy;
    // a pickup filled an unset slot, replaced the standard weapon, or
    // refilled a weapon from empty to full
    bool slotFilled;
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
    // the duration the shield started with
    float maxDuration;

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
    GRAPPLE_PHYSICS,
    DRONE_PHYSICS,
    SWAP_PHYSICS,
};
#define NUM_PHYSICS_TYPES (SWAP_PHYSICS + 1)

typedef struct physicsSource {
    enum physicsType type;
    enum weaponType weapon;
    // the projectile was boosted by its owner's burst on the frame it
    // was fired
    bool boosted;
    // the projectile was deflected and the push is credited to the drone
    // that deflected it or the previous owner
    bool deflected;
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
    // set when the body went through a portal; rot is the rotation its
    // velocity was given
    bool portalTransit;
    b2Rot rot;
    // set when drone srcIdx swapped places with the body; srcIdx claims
    // every other drone's contributions as source
    bool claim;
    physicsSource source;
    uint16_t step;
} physicsStepInfo;

typedef struct weaponSlot {
    weaponInformation *weaponInfo;

    int8_t ammo;
    float cooldown;
    uint16_t heat;
    float charge;
    float fullChargeHoldTime;
} weaponSlot;

typedef struct utilitySlot {
    const utilityInformation *utilityInfo;

    uint8_t ammo;
    // seconds spent regenerating the next ammo
    float regen;
    float cooldown;
} utilitySlot;

// what turned an enemy projectile enough to take it over
enum deflectType {
    BURST_DEFLECT,
    EXPLOSION_DEFLECT,
    PROJECTILE_DEFLECT,
    MIRROR_DEFLECT,
    NUM_DEFLECT_TYPES,
};

// setup weapons that earn assists
enum assistWeapon {
    ASSIST_BLACK_HOLE,
    ASSIST_GRAPPLE,
    NUM_ASSIST_WEAPONS,
};

typedef struct droneEntity {
    b2BodyId bodyID;
    b2ShapeId shapeID;

    weaponSlot weapon;
    utilitySlot utility;
    // the weapon the drone's loadout started it with in combo training
    // episodes, UNSET_WEAPON if it had none; in combo episodes ammo
    // crates restore it once it runs dry
    enum weaponType loadoutWeapon;
    // the utility button was held last step
    bool utilityHeld;
    bool triggerHeld;
    // frames left of the utility ammo pips shaking after a utility press
    // that did nothing, UINT16_MAX until the render code starts it
    uint16_t utilityShakeRenderSteps;
    // the brake and burst buttons were held last step
    bool brakeHeld;
    bool burstHeld;
    // frames left of the energy meter shaking after brake or burst was
    // pressed when energy couldn't be used, UINT16_MAX until the render
    // code starts it
    uint16_t energyShakeRenderSteps;

    projectileEntity *grapple;
    // the drone's live mine, see droneMine
    projectileEntity *mine;
    // live attached portals
    portalEntity *portals[_MAX_PORTALS];
    uint8_t openPortals;
    // touching a wall inside an open portal's area, which touches what
    // blocks the linked portal's exit instead of the wall; checked each
    // step as leaving the area or the exit becoming blocked while still
    // touching starts no new contact
    bool inWallPortal;

    float grappleLength;
    b2JointId grappleJointID;

    // episode frame + 1 this drone last pulled each drone with a black
    // hole or had its grapple attached to it, 0 if it hasn't
    uint16_t assistFrame[_MAX_DRONES][NUM_ASSIST_WEAPONS];

    float energyLeft;
    bool braking;
    bool chargingBurst;
    float burstCharge;
    float burstChargeUsed;
    float burstCooldown;
    uint16_t burstSerial;
    float burstRadius;
    float burstImpulse;
    bool burstStaticWallsHit;
    uint8_t burstFramesLeft;
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
    // set when an explosion overwrote lastVelocity mid step; tracked pushes
    // from lastVelocityTrackIdx on aren't part of lastVelocity
    uint16_t lastVelocityStep;
    uint32_t lastVelocityTrackIdx;
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
    float teammateKills;
    float unknownKills;
    // ammo crates picked up, and those that added ammo
    float cratesPickedUp;
    float cratesRefilled;
    float score;

    float shotsFired[_NUM_WEAPONS];
    float shotsHit[_NUM_WEAPONS];
    float shotsTaken[_NUM_WEAPONS];
    float ownShotsTaken[_NUM_WEAPONS];
    float weaponsPickedUp[_NUM_WEAPONS];
    float shotDistances[_NUM_WEAPONS];
    float utilitiesPickedUp[_NUM_UTILITIES];

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

// per episode stats for offline analysis; kept out of Log so they aren't
// aggregated or logged during training
typedef struct droneAnalysisStats {
    // weapon kills credit the weapon that made the killing push; kills by
    // movement, drone collisions and bursts are only in deathsBySource
    float killsByWeapon[_NUM_WEAPONS];
    float selfKillsByWeapon[_NUM_WEAPONS];
    float deathsBySource[NUM_PHYSICS_TYPES];
    float selfKillsBySource[NUM_PHYSICS_TYPES];
    float floatingWallDeaths;
    float floatingWallSelfKills;
    float assistsByWeapon[_NUM_WEAPONS];
    float diedHoldingActive[_NUM_WEAPONS];
    // pickups by the weapon held before and the weapon picked up
    float pickupsFrom[_NUM_WEAPONS][_NUM_WEAPONS];
    // projectiles boosted by their owner's burst on the frame they were fired
    float boostedShotsFired[_NUM_WEAPONS];
    float boostedShotsHit[_NUM_WEAPONS];
    float boostedKills[_NUM_WEAPONS];
    float boostedSelfKills[_NUM_WEAPONS];
    // projectile speed right before and after the boost; mines explode
    // when boosted so they're excluded
    float boostedSpeedBefore[_NUM_WEAPONS];
    float boostedSpeedAfter[_NUM_WEAPONS];
    // utility button presses by utility, and presses that did something:
    // fired, started reeling in an attached grapple or detonated mines
    float utilityPresses[_NUM_UTILITIES];
    float utilityPressesUseful[_NUM_UTILITIES];
    float utilityAmmoSpent[_NUM_UTILITIES];
    // utility projectile hits on other drones, kept out of the shot stats
    float utilityHits[_NUM_UTILITIES];
    // grapples attached to a static wall, a floating wall and a drone
    float grappleAttaches[3];
    float grappleAttachedDeaths;
    float minesDetonated;
    // enemy projectiles this drone deflected by what deflected them and
    // the deflected weapon, and the hits and weapon kills credited to it
    // from deflected projectiles; deflected hits and kills are kept out of
    // the shot hit and per weapon kill stats
    float deflections[NUM_DEFLECT_TYPES][_NUM_WEAPONS];
    float deflectedHits[_NUM_WEAPONS];
    float deflectedKills[_NUM_WEAPONS];
    // mirrors placed, and projectiles this drone's mirrors reversed by the
    // reversed weapon, own projectiles included
    float mirrorsPlaced;
    float mirrorReflections[_NUM_WEAPONS];
} droneAnalysisStats;

struct Log {
    float length;
    float ties;
    float botCLNoise;
    float easyEpProb;
    float easyEp;
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
    int32_t explosionSwirlsLoc;
    int32_t explosionCountLoc;
    Texture2D wallTexture;
    // only has the infinity glyph, shown for infinite ammo as the default
    // font doesn't have it
    Font infinityFont;
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
    // drawn from renderRng on creation so sparks are random but
    // stable across frames
    unsigned int sparkSeed;
} explosionInfo;

// sparks shown where a projectile was boosted by its owner's burst,
// purely visual
// where two drones swapped places from a direct portal shot hit, shown
// as a swirl at both positions
#define MAX_DRONE_SWAP_EFFECTS 4
typedef struct droneSwapEffect {
    // where the shooting drone was, then where the hit drone was
    b2Vec2 pos[2];
    // 0 when unused, UINT16_MAX until the render code starts it
    uint16_t renderSteps;
} droneSwapEffect;

typedef struct boostSparksInfo {
    b2Vec2 pos;
    b2Vec2 direction;
    enum weaponType weapon;
    uint16_t renderSteps;
    uint16_t maxRenderSteps;
    // drawn from renderRng on creation so sparks are random but
    // stable across frames
    unsigned int sparkSeed;
} boostSparksInfo;

typedef struct agentActions {
    b2Vec2 move;
    b2Vec2 aim;
    bool trigger;
    bool brake;
    bool chargingBurst;
    bool utility;
} agentActions;

typedef struct debugPoint {
    b2Vec2 pos;
    float size;
    Color color;
} debugPoint;

// Scripted opponents. These are the rungs of [selfplay] eval_bots. Every
// policy after BOT_SCRIPTED is a personality of the scripted bot with its
// own playstyle, used to evaluate learned policies against varied opponents
enum botPolicy {
    BOT_SITTING_DUCK = 0,
    BOT_SCRIPTED = 1,
    // flees and only fires to move faster or when cornered
    BOT_EVADER = 2,
    // prefers long range charged weapons and waits for shots near cover
    BOT_CAMPER = 3,
    // prefers short range weapons and closes in fast with bursts
    BOT_RUSHER = 4,
    // prefers mines and black holes and makes the enemy come through them
    BOT_TRAPPER = 5,
    // holds a spot and only fires back
    BOT_TURTLE = 6,
    // never seeks weapon pickups
    BOT_FUNDAMENTALS = 7,
    NUM_BOT_POLICIES,
};

// scripted bot behaviors, toggled per drone so variants of the bot can be
// played against each other in the same env (see bot_tournament.c). Ammo
// reserve, safe geometry and shotgun boosting are on by default, and
// personalities add others for their playstyle (see botPresets); the rest
// are experiments that measured neutral or worse and are kept for
// evaluating against learned policies. Smart reserve only has an effect
// with ammo reserve on
enum botFeature {
    BOT_FEAT_AMMO_RESERVE = 1 << 0,
    BOT_FEAT_PROJ_DEFENSE = 1 << 1,
    BOT_FEAT_BURST = 1 << 2,
    BOT_FEAT_BURST_BOOST = 1 << 3,
    BOT_FEAT_KILL_GEOMETRY = 1 << 4,
    BOT_FEAT_SNIPER_DODGE = 1 << 5,
    BOT_FEAT_SMART_RESERVE = 1 << 6,
    // keep death walls from being behind the drone as seen from the enemy,
    // so hits don't knock it into them
    BOT_FEAT_SAFE_GEOMETRY = 1 << 7,
    // stay out of shotgun range of an enemy with a shotgun drawn
    BOT_FEAT_KITE = 1 << 8,
    // charge a burst before a close shotgun shot and release it with the
    // shot so the burst speeds up the pellets
    BOT_FEAT_SHOTGUN_BOOST = 1 << 9,
    // only fire the shotgun from close enough for the pellets to keep
    // their speed and stay together
    BOT_FEAT_SHOTGUN_RANGE = 1 << 10,
    // use the utility: place and detonate mines, and grapple to close in
    // on or escape the enemy as the personality does
    BOT_FEAT_UTILITY = 1 << 11,
};
#define BOT_FEAT_ALL 0xFFF

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
    // cell a personality is heading to or holding (flee target, sniping
    // spot, home), and the step it's re-chosen
    int16_t spotCellIdx;
    uint16_t spotUntilStep;
    // last step the bot fired at the enemy or the enemy fired
    uint16_t lastShotStep;
    uint16_t lastEnemyShotStep;
    // lives left when the spot was chosen; a respawn re-chooses it
    uint8_t livesLeft;
    // what the grapple is out for (see botGrappleIntent), and the step the
    // bot gives up on it
    uint8_t grappleIntent;
    uint16_t grappleUntilStep;
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
    BOT_EVENT_FLEE,
    BOT_EVENT_HOLD_SPOT,
    BOT_EVENT_SPOT_MOVE,
    BOT_EVENT_RECOIL_BOOST,
    BOT_EVENT_MINE_PLACED,
    BOT_EVENT_COUNTER_SHOT,
    BOT_EVENT_SUDDEN_DEATH_AVOID,
    BOT_EVENT_SAFE_GEOMETRY,
    BOT_EVENT_KITE,
    BOT_EVENT_SHOTGUN_BOOST,
    BOT_EVENT_GRAPPLE_CLOSE,
    BOT_EVENT_GRAPPLE_ESCAPE,
    BOT_EVENT_GRAPPLE_RELEASE,
    BOT_EVENT_GRAPPLE_DANGER_RELEASE,
    BOT_EVENT_MINE_DETONATE,
    BOT_EVENT_UTILITY_PICKUP,
    NUM_BOT_EVENTS,
};

struct Env {
    Agent agents[_MAX_DRONES];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;

    uint8_t numAgents;
    uint8_t numTeams;
    bool teamsEnabled;
    enum botPolicy botPolicy;
    bool isTraining;

    float botCLNoise;
    float botCLDecay;
    // personality of each scripted drone, BOT_SCRIPTED or later
    enum botPolicy botPersonality[_MAX_DRONES];
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

    // with probability comboTrainingProb a training episode starts every
    // drone with a random two weapon loadout, symmetric half of the time
    float comboTrainingProb;
    // this episode started with combo loadouts, so pickups are ammo crates
    bool comboEp;

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

    bool continuousActions;

    uint8_t frameRate;
    float deltaTime;
    uint8_t frameSkip;
    uint8_t box2dSubSteps;
    bool needsReset;

    uint16_t episodeLength;
    Log log;
    droneStats stats[_MAX_DRONES];
    droneAnalysisStats analysisStats[_MAX_DRONES];
    // analysisStats of the last finished episode, saved before the reset
    droneAnalysisStats lastAnalysisStats[_MAX_DRONES];

    b2WorldId worldID;
    int8_t pinnedMapIdx;
    int8_t mapIdx;
    mapEntry *map;
    int8_t lastSpawnQuad;
    uint8_t spawnedWeaponPickups[_NUM_WEAPONS];
    uint8_t spawnedUtilityPickups[_NUM_UTILITIES];
    uint8_t spawnedCrates;
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
    // TODO: rename to deferred destroyed or something?
    CC_Array *explodingProjectiles;
    CC_Array *dronePieces;
    // live mirrors, oldest first
    CC_Array *mirrors;

    uint16_t burstSerial;
    uint8_t minesCreated[_MAX_DRONES];

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
    unsigned int renderRng;
    CC_Array *explosions;
    CC_Array *boostSparks;
    droneSwapEffect droneSwapEffects[MAX_DRONE_SWAP_EFFECTS];
    CC_Array *debugPoints;
    enum roundState roundState;
    uint16_t tick_frames_left;
    int8_t winner;
    int8_t winningTeam;
};
typedef struct Env iwEnv;

#endif
