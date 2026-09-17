#include "env.h"

void randActions(iwEnv *e) {
    // e->lastRandState = e->rng;
    for (uint8_t i = 0; i < e->numDrones; i++) {
        float* actions = agentActions(e, i);
        actions[0] = randFloat(&e->rng, -1.0f, 1.0f);
        actions[1] = randFloat(&e->rng, -1.0f, 1.0f);
        actions[2] = randFloat(&e->rng, -1.0f, 1.0f);
        actions[3] = randFloat(&e->rng, -1.0f, 1.0f);
        actions[4] = randFloat(&e->rng, -1.0f, 1.0f);
        actions[5] = randFloat(&e->rng, -1.0f, 1.0f);
        actions[6] = randFloat(&e->rng, -1.0f, 1.0f);
    }
}

void perfTest(const uint32_t numSteps) {
    const uint8_t NUM_DRONES = 2;

    iwEnv *e = fastCalloc(1, sizeof(iwEnv));

    // rayClient *client = createRayClient();
    // e->client = client;

    uint64_t seed = time(NULL);
    printf("seed: %lu\n", seed);
    initEnv(e, NUM_DRONES, NUM_DRONES, -1, seed, false, false, true, false);
    for (uint8_t i = 0; i < e->numAgents; i++) {
        posix_memalign((void **)&agentObs(e, i), sizeof(void *),
            alignedSize(e->obsSize, sizeof(float)));
        agentActions(e, i) = fastCalloc(CONTINUOUS_ACTION_SIZE, sizeof(float));
        agentRewards(e, i) = fastCalloc(1, sizeof(float));
        agentTerminals(e, i) = fastCalloc(1, sizeof(uint8_t));
    }
    initMaps(e);

    // randActions(e);
    setupEnv(e);
    puf_step(e);

    uint32_t steps = 0;
    while (steps != numSteps) {
        // randActions(e);
        puf_step(e);
        steps++;
    }

    puf_close(e);
    destroyMaps();
    for (uint8_t i = 0; i < e->numAgents; i++) {
        free(agentObs(e, i));
        fastFree(agentActions(e, i));
        fastFree(agentRewards(e, i));
        fastFree(agentTerminals(e, i));
    }
    fastFree(e);
}

int main(void) {
    perfTest(2500000);
    return 0;
}
