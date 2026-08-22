#pragma once

#include "pufferenv.h"

#ifdef TEST_NARROW_CATEGORICAL
#define NUM_ATNS 2
#define ACT_SIZES {31, 32}
#else
#define NUM_ATNS 5
#define ACT_SIZES {31, 32, 33, 155, 212}
#endif
#define OBS_SIZE 1

typedef float obs_t;
struct Log {
    float score;
    float n;
};

struct Env {
    Log log;
    int num_agents;
    unsigned int rng;
    Agent agents[1];
};

void puf_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = nullptr;
}

void puf_reset(Env* env) {
    ((obs_t*)env->agents[0].observations)[0] = 0.0f;
}

void puf_step(Env* env) {
    (void)env;
}

void puf_render(Env* env) {
    (void)env;
}

void puf_close(Env* env) {
    (void)env;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
}
