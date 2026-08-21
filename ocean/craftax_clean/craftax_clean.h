// Full native Craftax port.

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "constants.h"
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
#include <stdio.h>
#include <stdlib.h>

#define ACT_SIZES {NUM_ACTIONS}
#define NUM_ATNS 1

// Data structures 
typedef struct {
    int wood;
    int stone;
    int coal;
    int iron;
    int diamond;
    int sapling;
    int pickaxe;
    int sword;
    int bow;
    int arrows;
    int armour[4];
    int torches;
    int ruby;
    int sapphire;
    int potions[NUM_POTIONS];
    int books;
} Inventory;

typedef struct {
    int position[3][2];
    float health[3];
    bool mask[3];
    int attack_cooldown[3];
    int type_id[3];
} Mobs;

typedef struct {
    int map[NUM_LEVELS][MAP_SIZE][MAP_SIZE];
    int item_map[NUM_LEVELS][MAP_SIZE][MAP_SIZE];
    int mob_map[NUM_LEVELS][MAP_SIZE][MAP_SIZE];
    unsigned char light_map[NUM_LEVELS][MAP_SIZE][MAP_SIZE];
    int down_ladders[NUM_LEVELS][2];
    int up_ladders[NUM_LEVELS][2];
    int chests_opened[NUM_LEVELS];
    int monsters_killed[NUM_LEVELS];

    int player_position[2];
    int player_level;
    int player_direction;

    // Intrinsics 
    float player_health;
    int player_food;
    int player_drink;
    int player_energy;
    int player_mana;
    int is_sleeping;
    int is_resting;

    // Second order intrinsics
    float player_recover;
    float player_hunger;
    float player_thirst;
    float player_fatigue;
    float player_recover_mana;

    // Attributes
    int player_xp;
    int player_dexterity;
    int player_strength;
    int player_intelligence;

    Inventory inventory;

    Mobs melee_mobs[NUM_LEVELS];
    Mobs passive_mobs[NUM_LEVELS];
    Mobs ranged_mobs[NUM_LEVELS];
    Mobs mob_projectiles[NUM_LEVELS];

    int mob_projectile_directions[NUM_LEVELS][MAX_MOB_PROJECTILES][2];
    Mobs player_projectiles[NUM_LEVELS];
    int player_projectile_directions[NUM_LEVELS][MAX_PLAYER_PROJECTILES][2];
    int growing_plants_positions[MAX_GROWING_PLANTS][2];
    int growing_plants_age[MAX_GROWING_PLANTS];
    int growing_plants_mask[MAX_GROWING_PLANTS];
    int potion_mapping[NUM_POTIONS];
    int learned_spells[2];
    int sword_enchantment;
    int bow_enchantment;
    int armour_enchantments[4];
    int boss_progress;
    int boss_timestep_to_spawn_this_round;
    float light_level;
    int achievements[NUM_ACHIEVEMENTS];
    uint32_t state_rng[2];
    int timestep;
    int fractal_noise_angles[4];
} State;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float floors[NUM_LEVELS];
    float achievements[NUM_ACHIEVEMENTS];
    float n;
};

// Rendering 
typedef struct {
    int cell_size;
    int screen_width;
    int screen_height;
    bool window_ready;
} Client;

// Fast SplitMix64/MurmurHash3 RNG, matching ocean/craftax threefry.h.
typedef struct {
    uint32_t word[2];
} Rng;

struct Env {
    Client* client;
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    State state;
    int timestep;
    unsigned int rng;
    uint64_t seed;
    Rng env_rng;
    float episode_return_accum;
    int episode_length_accum;
    int max_floor_accum;
    int achievements[NUM_ACHIEVEMENTS];
};

typedef Env Craftax;

// Init function
void c_init(Craftax* env) {
    memset(&env->state, 0, sizeof(State));

    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    env->max_floor_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(Log));
    env->client = NULL;
}

void c_allocate(Craftax* env) {
    env->agents[0].observations = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
    env->agents[0].actions = (float*)calloc(1, sizeof(float));
    env->agents[0].rewards = (float*)calloc(1, sizeof(float));
    env->agents[0].terminals = (float*)calloc(1, sizeof(float));
}

// World generation
static inline uint64_t rng_to_u64(Rng key) {
    return ((uint64_t)key.word[1] << 32) | key.word[0];
}

static inline Rng rng_from_u64(uint64_t x) {
    return (Rng){{(uint32_t)x, (uint32_t)(x >> 32)}};
}

static inline Rng rng_seed(uint32_t seed) {
    return (Rng){{seed, seed ^ 0x9E3779B9u}};
}

static inline uint64_t rng_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline uint64_t rng_hash64(Rng key, uint64_t counter) {
    return rng_mix64(rng_to_u64(key) ^ counter);
}

static inline Rng rng_counter(Rng key, uint32_t count_hi, uint32_t count_lo) {
    return rng_from_u64(rng_hash64(key, ((uint64_t)count_hi << 32) | count_lo));
}

static inline void rng_split(Rng key, Rng* left, Rng* right) {
    uint64_t state = rng_to_u64(key);
    uint64_t s1 = state * 6364136223846793005ULL + 1;
    uint64_t s2 = s1 * 6364136223846793005ULL + 1;
    *left = rng_from_u64(s1);
    *right = rng_from_u64(s2);
}

static inline void rng_split_n(Rng key, Rng* out, int count) {
    uint64_t state = rng_to_u64(key);
    for (int i = 0; i < count; i++) {
        state = state * 6364136223846793005ULL + 1;
        out[i] = rng_from_u64(state);
    }
}

static inline uint32_t rng_u32_at(Rng key, uint64_t index) {
    uint64_t h = rng_hash64(key, index);
    return (uint32_t)h ^ (uint32_t)(h >> 32);
}

static inline uint32_t rng_u32(Rng key) {
    return rng_u32_at(key, 0u);
}

static inline float rng_f32_at(Rng key, uint64_t index) {
    uint32_t bits = rng_u32_at(key, index);
    uint32_t float_bits = (bits >> 9u) | 0x3F800000u;
    float value;
    memcpy(&value, &float_bits, sizeof(value));
    return value - 1.0f;
}

static inline float rng_f32(Rng key) {
    return rng_f32_at(key, 0u);
}

static inline Rng load_rng(const State* state) {
    return (Rng){{state->state_rng[0], state->state_rng[1]}};
}

static inline void store_rng(State* state, Rng rng) {
    state->state_rng[0] = rng.word[0];
    state->state_rng[1] = rng.word[1];
}

static inline Rng rng_key(Rng* rng) {
    Rng next;
    Rng draw;
    rng_split(*rng, &next, &draw);
    *rng = next;
    return draw;
}

static inline int rand_range(Rng* rng, int low, int high) {
    uint32_t span = (uint32_t)(high - low + 1);
    if (span == 0) {
        return low;
    }
    return low + (int)(rng_u32_at(rng_key(rng), 0u) % span);
}

static inline float rand_unit(Rng* rng) {
    return rng_f32(rng_key(rng));
}

static inline int weighted_choice(Rng* rng, const float* weights, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; i++) {
        total += weights[i];
    }
    float draw = total * (1.0f - rand_unit(rng));
    float cumulative = 0.0f;
    for (int i = 0; i < count; i++) {
        cumulative += weights[i];
        if (cumulative >= draw) {
            return i;
        }
    }
    return count - 1;
}

static inline int choice_valid(Rng key, const bool* valid, int count) {
    int valid_count = 0;
    int last_valid = 0;
    for (int i = 0; i < count; i++) {
        if (valid[i]) {
            valid_count++;
            last_valid = i;
        }
    }
    if (valid_count == 0) {
        return 0;
    }
    float draw = (float)valid_count * (1.0f - rng_f32(key));
    float cumulative = 0.0f;
    for (int i = 0; i < count; i++) {
        if (valid[i]) {
            cumulative += 1.0f;
        }
        if (cumulative >= draw) {
            return i;
        }
    }
    return last_valid;
}

static inline uint32_t randint_u32_at(Rng key, uint64_t index, uint32_t minval, uint32_t maxval) {
    uint32_t span = maxval > minval ? maxval - minval : 1u;
    if ((span & (span - 1)) == 0) {
        uint32_t bits = rng_u32_at(key, index);
        return minval + (bits & (span - 1));
    }
    uint64_t h = rng_hash64(key, index);
    return minval + (uint32_t)(((h >> 32) * (uint64_t)span) >> 32);
}

static inline int randint_at(Rng key, uint64_t index, int minval, int maxval) {
    return (int)randint_u32_at(key, index, (uint32_t)minval, (uint32_t)maxval);
}

static inline bool walkable(int block) {
    return block == BLOCK_GRASS
        || block == BLOCK_PATH
        || block == BLOCK_FIRE_GRASS
        || block == BLOCK_ICE_GRASS
        || block == BLOCK_SAND
        || block == BLOCK_GRAVEL
        || block == BLOCK_FOUNTAIN
        || block == BLOCK_ENCHANTMENT_TABLE_FIRE
        || block == BLOCK_ENCHANTMENT_TABLE_ICE
        || block == BLOCK_GRAVE
        || block == BLOCK_GRAVE2
        || block == BLOCK_GRAVE3
        || block == BLOCK_NECROMANCER
        || block == BLOCK_NECROMANCER_VULNERABLE;
}

static inline void init_mobs(Mobs* mobs) {
    memset(mobs, 0, sizeof(*mobs));
}

static inline void add_light(State* state, int level, int center_row, int center_col) {
    static const float torch_light[9][9] = {
        {0.0f, 0.0f, 0.10557288f, 0.17537886f, 0.19999999f, 0.17537886f, 0.10557288f, 0.0f, 0.0f},
        {0.0f, 0.15147191f, 0.27888972f, 0.36754447f, 0.39999998f, 0.36754447f, 0.27888972f, 0.15147191f, 0.0f},
        {0.10557288f, 0.27888972f, 0.43431455f, 0.55278647f, 0.6f, 0.55278647f, 0.43431455f, 0.27888972f, 0.10557288f},
        {0.17537886f, 0.36754447f, 0.55278647f, 0.71715724f, 0.8f, 0.71715724f, 0.55278647f, 0.36754447f, 0.17537886f},
        {0.19999999f, 0.39999998f, 0.6f, 0.8f, 1.0f, 0.8f, 0.6f, 0.39999998f, 0.19999999f},
        {0.17537886f, 0.36754447f, 0.55278647f, 0.71715724f, 0.8f, 0.71715724f, 0.55278647f, 0.36754447f, 0.17537886f},
        {0.10557288f, 0.27888972f, 0.43431455f, 0.55278647f, 0.6f, 0.55278647f, 0.43431455f, 0.27888972f, 0.10557288f},
        {0.0f, 0.15147191f, 0.27888972f, 0.36754447f, 0.39999998f, 0.36754447f, 0.27888972f, 0.15147191f, 0.0f},
        {0.0f, 0.0f, 0.10557288f, 0.17537886f, 0.19999999f, 0.17537886f, 0.10557288f, 0.0f, 0.0f},
    };

    for (int dr = -4; dr <= 4; dr++) {
        int row = center_row + dr;
        if (row < 0 || row >= MAP_SIZE) {
            continue;
        }
        for (int dc = -4; dc <= 4; dc++) {
            int col = center_col + dc;
            if (col < 0 || col >= MAP_SIZE) {
                continue;
            }
            float light = (float)state->light_map[level][row][col] / 255.0f
                + torch_light[dr + 4][dc + 4];
            if (light < 0.0f) {
                light = 0.0f;
            }
            if (light > 1.0f) {
                light = 1.0f;
            }
            state->light_map[level][row][col] = (unsigned char)(light * 255.0f);
        }
    }
}

#define MAP_CELLS (MAP_SIZE * MAP_SIZE)
#define NOISE_PI2 6.28318530717958647692f
#define NOISE_SQRT2 1.41421356237309504880f

static inline float noise_interpolant(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static inline void generate_perlin(
    Rng rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    float* out
) {
    Rng unused;
    Rng angle_key;
    rng_split(rng, &unused, &angle_key);
    int cell_rows = rows / res_rows;
    int cell_cols = cols / res_cols;
    int width = res_cols + 1;

    for (int row = 0; row < rows; row++) {
        int grad_row = row / cell_rows;
        float local_row = (float)(row - grad_row * cell_rows) / (float)cell_rows;
        float interp_row = noise_interpolant(local_row);
        for (int col = 0; col < cols; col++) {
            int grad_col = col / cell_cols;
            float local_col = (float)(col - grad_col * cell_cols) / (float)cell_cols;
            float interp_col = noise_interpolant(local_col);
            float gx[2][2];
            float gy[2][2];
            for (int dr = 0; dr < 2; dr++) {
                for (int dc = 0; dc < 2; dc++) {
                    uint64_t index = (uint64_t)(grad_row + dr) * (uint64_t)width
                        + (uint64_t)(grad_col + dc);
                    float angle = NOISE_PI2 * rng_f32_at(angle_key, index);
                    gx[dr][dc] = cosf(angle);
                    gy[dr][dc] = sinf(angle);
                }
            }
            float n00 = local_row * gx[0][0] + local_col * gy[0][0];
            float n10 = (local_row - 1.0f) * gx[1][0] + local_col * gy[1][0];
            float n01 = local_row * gx[0][1] + (local_col - 1.0f) * gy[0][1];
            float n11 = (local_row - 1.0f) * gx[1][1] + (local_col - 1.0f) * gy[1][1];
            float n0 = n00 * (1.0f - interp_row) + interp_row * n10;
            float n1 = n01 * (1.0f - interp_row) + interp_row * n11;
            out[(size_t)row * (size_t)cols + (size_t)col] =
                NOISE_SQRT2 * ((1.0f - interp_col) * n0 + interp_col * n1);
        }
    }
}

static inline void generate_fractal(
    Rng rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    int octaves,
    float persistence,
    int lacunarity,
    float* out
) {
    size_t size = (size_t)rows * (size_t)cols;
    memset(out, 0, size * sizeof(float));
    int frequency = 1;
    float amplitude = 1.0f;
    float perlin[MAP_CELLS];
    for (int octave = 0; octave < octaves; octave++) {
        Rng next_rng;
        Rng noise_key;
        rng_split(rng, &next_rng, &noise_key);
        rng = next_rng;
        generate_perlin(
            noise_key,
            rows,
            cols,
            frequency * res_rows,
            frequency * res_cols,
            perlin
        );
        for (size_t i = 0; i < size; i++) {
            out[i] += amplitude * perlin[i];
        }
        frequency *= lacunarity;
        amplitude *= persistence;
    }
    float min_value = out[0];
    float max_value = out[0];
    for (size_t i = 1; i < size; i++) {
        if (out[i] < min_value) min_value = out[i];
        if (out[i] > max_value) max_value = out[i];
    }
    float scale = max_value - min_value;
    for (size_t i = 0; i < size; i++) {
        out[i] = (out[i] - min_value) / scale;
    }
}

static inline void apply_ladder_light(
    unsigned char light_map[MAP_SIZE][MAP_SIZE],
    const int ladder_up[2],
    float default_light
) {
    int start_row = ladder_up[0] - 4;
    int start_col = ladder_up[1] - 4;
    if (start_row < 0) start_row += MAP_SIZE;
    if (start_col < 0) start_col += MAP_SIZE;
    if (start_row > MAP_SIZE - 9) start_row = MAP_SIZE - 9;
    if (start_col > MAP_SIZE - 9) start_col = MAP_SIZE - 9;
    if (start_row < 0) start_row = 0;
    if (start_col < 0) start_col = 0;
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            float dr = (float)(row - 4);
            float dc = (float)(col - 4);
            float torch = 1.0f - sqrtf(dr * dr + dc * dc) / 5.0f;
            if (torch < 0.0f) torch = 0.0f;
            float light = torch * (1.0f - default_light) + default_light;
            light_map[start_row + row][start_col + col] = (unsigned char)(light * 255.0f);
        }
    }
}

static inline void add_lava_light(
    unsigned char light_map[MAP_SIZE][MAP_SIZE],
    const bool lava_map[MAP_SIZE][MAP_SIZE],
    bool lava_emits_light
) {
    if (!lava_emits_light) {
        return;
    }
    static const float kernel[3][3] = {
        {0.2f, 0.7f, 0.2f},
        {0.7f, 1.0f, 0.7f},
        {0.2f, 0.7f, 0.2f},
    };
    for (int row = 0; row < MAP_SIZE; row++) {
        for (int col = 0; col < MAP_SIZE; col++) {
            float add = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                int src_row = row + kr - 1;
                if (src_row < 0 || src_row >= MAP_SIZE) continue;
                for (int kc = 0; kc < 3; kc++) {
                    int src_col = col + kc - 1;
                    if (src_col < 0 || src_col >= MAP_SIZE) continue;
                    if (lava_map[src_row][src_col]) add += kernel[kr][kc];
                }
            }
            float light = (float)light_map[row][col] / 255.0f + add;
            if (light < 0.0f) light = 0.0f;
            if (light > 1.0f) light = 1.0f;
            light_map[row][col] = (unsigned char)(light * 255.0f);
        }
    }
}

static inline int cell_index(int row, int col) {
    return row * MAP_SIZE + col;
}

static inline void generate_smooth_level(
    State* state,
    int level,
    Rng rng,
    const SmoothGenConfig* config
) {
    const int player_row = MAP_SIZE / 2;
    const int player_col = MAP_SIZE / 2;
    float water[MAP_CELLS];
    float mountain[MAP_CELLS];
    float path_x[MAP_CELLS];
    float tree_noise[MAP_CELLS];
    bool lava_map[MAP_SIZE][MAP_SIZE];
    Rng subkey;

    rng_split(rng, &rng, &subkey);
    generate_fractal(subkey, MAP_SIZE, MAP_SIZE, 3, 3, 1, 0.5f, 2, water);
    rng_split(rng, &rng, &subkey);
    rng_split(rng, &rng, &subkey);
    generate_fractal(subkey, MAP_SIZE, MAP_SIZE, 3, 3, 1, 0.5f, 2, mountain);
    rng_split(rng, &rng, &subkey);
    generate_fractal(subkey, MAP_SIZE, MAP_SIZE, 6, 24, 1, 0.5f, 2, path_x);
    rng_split(rng, &rng, &subkey);
    rng_split(rng, &rng, &subkey);
    Rng tree_uniform_key = rng;
    generate_fractal(subkey, MAP_SIZE, MAP_SIZE, 12, 12, 1, 0.5f, 2, tree_noise);

    for (int row = 0; row < MAP_SIZE; row++) {
        int dr = row > player_row ? row - player_row : player_row - row;
        for (int col = 0; col < MAP_SIZE; col++) {
            int dc = col > player_col ? col - player_col : player_col - col;
            float distance = sqrtf((float)(dr * dr + dc * dc));
            float proximity_water = distance / config->water_strength;
            if (proximity_water < 0.0f) proximity_water = 0.0f;
            if (proximity_water > config->water_max) proximity_water = config->water_max;
            float proximity_mountain = distance / config->mountain_strength;
            if (proximity_mountain < 0.0f) proximity_mountain = 0.0f;
            if (proximity_mountain > config->mountain_max) proximity_mountain = config->mountain_max;
            int idx = cell_index(row, col);

            water[idx] = water[idx] + proximity_water - 1.0f;
            int block = water[idx] > config->water_threshold
                ? config->sea_block
                : config->default_block;
            if (water[idx] > config->sand_threshold && block != config->sea_block) {
                block = config->coast_block;
            }

            mountain[idx] = mountain[idx] + 0.05f + proximity_mountain - 1.0f;
            if (mountain[idx] > 0.7f) {
                block = config->mountain_block;
            }
            if (mountain[idx] > 0.7f && path_x[idx] > 0.8f) {
                block = config->path_block;
            }
            if (mountain[idx] > 0.7f && path_x[cell_index(col, row)] > 0.8f) {
                block = config->path_block;
            }
            if (mountain[idx] > 0.85f && water[idx] > 0.4f) {
                block = config->inner_mountain_block;
            }
            if (tree_noise[idx] > config->tree_threshold_perlin
                && rng_f32_at(tree_uniform_key, (uint64_t)idx) > config->tree_threshold_uniform
                && block == config->tree_requirement_block) {
                block = config->tree;
            }

            state->map[level][row][col] = block;
            state->item_map[level][row][col] = ITEM_NONE;
            state->light_map[level][row][col] = (unsigned char)(config->default_light * 255.0f);
            state->mob_map[level][row][col] = 0;
        }
    }

    Rng ore_rng;
    rng_split(rng, &rng, &ore_rng);
    for (int ore_index = 0; ore_index < 5; ore_index++) {
        Rng ore_key;
        rng_split(ore_rng, &ore_rng, &ore_key);
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                int idx = cell_index(row, col);
                if (state->map[level][row][col] == config->ore_requirement_blocks[ore_index]
                    && rng_f32_at(ore_key, (uint64_t)idx) < config->ore_chances[ore_index]) {
                    state->map[level][row][col] = config->ores[ore_index];
                }
            }
        }
    }

    for (int row = 0; row < MAP_SIZE; row++) {
        for (int col = 0; col < MAP_SIZE; col++) {
            int idx = cell_index(row, col);
            lava_map[row][col] = mountain[idx] > 0.85f && tree_noise[idx] > 0.7f;
            if (lava_map[row][col]) {
                state->map[level][row][col] = config->lava;
            }
        }
    }

    rng_split(rng, &rng, &subkey);
    bool valid_diamond[MAP_CELLS];
    for (int row = 0; row < MAP_SIZE; row++) {
        for (int col = 0; col < MAP_SIZE; col++) {
            valid_diamond[cell_index(row, col)] = state->map[level][row][col] == BLOCK_STONE;
        }
    }
    int diamond_index = choice_valid(subkey, valid_diamond, MAP_CELLS);
    state->map[level][diamond_index / MAP_SIZE][diamond_index % MAP_SIZE] = BLOCK_STONE;
    state->map[level][player_row][player_col] = config->player_spawn;

    bool valid_ladder[MAP_CELLS];
    for (int row = 0; row < MAP_SIZE; row++) {
        for (int col = 0; col < MAP_SIZE; col++) {
            valid_ladder[cell_index(row, col)] =
                state->map[level][row][col] == config->valid_ladder;
        }
    }

    rng_split(rng, &rng, &subkey);
    int ladder_down_index = choice_valid(subkey, valid_ladder, MAP_CELLS);
    state->down_ladders[level][0] = ladder_down_index / MAP_SIZE;
    state->down_ladders[level][1] = ladder_down_index % MAP_SIZE;
    if (config->ladder_down) {
        state->item_map[level][state->down_ladders[level][0]][state->down_ladders[level][1]] =
            ITEM_LADDER_DOWN;
    }

    rng_split(rng, &rng, &subkey);
    int ladder_up_index = choice_valid(subkey, valid_ladder, MAP_CELLS);
    state->up_ladders[level][0] = ladder_up_index / MAP_SIZE;
    state->up_ladders[level][1] = ladder_up_index % MAP_SIZE;
    apply_ladder_light(state->light_map[level], state->up_ladders[level], config->default_light);
    add_lava_light(state->light_map[level], lava_map, config->lava == BLOCK_LAVA);
    if (config->ladder_up) {
        state->item_map[level][state->up_ladders[level][0]][state->up_ladders[level][1]] =
            ITEM_LADDER_UP;
    }
}

static inline void generate_dungeon_level(
    State* state,
    int level,
    Rng rng,
    const DungeonConfig* config
) {
    const int chunk_size = DUNGEON_CHUNK_SIZE;
    const int world_chunk_height = MAP_SIZE / chunk_size;
    const int num_rooms = DUNGEON_ROOM_COUNT;
    const int min_room_size = DUNGEON_MIN_ROOM_SIZE;
    const int max_room_size = DUNGEON_MAX_ROOM_SIZE;
    const int padded_size = MAP_SIZE + 2 * max_room_size;

    int padded_map[68][68];
    int padded_item[68][68];
    bool room_occupancy[9];
    int room_sizes[8][2];
    int room_positions[8][2];

    for (int row = 0; row < padded_size; row++) {
        for (int col = 0; col < padded_size; col++) {
            bool inner = row >= max_room_size
                && row < max_room_size + MAP_SIZE
                && col >= max_room_size
                && col < max_room_size + MAP_SIZE;
            padded_map[row][col] = inner ? BLOCK_WALL : 0;
            padded_item[row][col] = ITEM_NONE;
        }
    }
    for (int i = 0; i < 9; i++) {
        room_occupancy[i] = true;
    }

    Rng ignored;
    Rng room_size_key;
    Rng keys3[3];
    rng_split_n(rng, keys3, 3);
    rng = keys3[0];
    ignored = keys3[1];
    room_size_key = keys3[2];
    (void)ignored;
    for (int room = 0; room < num_rooms; room++) {
        room_sizes[room][0] = randint_at(room_size_key, (uint64_t)room * 2u, min_room_size, max_room_size);
        room_sizes[room][1] = randint_at(room_size_key, (uint64_t)room * 2u + 1u, min_room_size, max_room_size);
    }

    Rng room_rng;
    rng_split(rng, &rng, &room_rng);
    for (int room_index = 0; room_index < num_rooms; room_index++) {
        Rng choice_key;
        rng_split(room_rng, &room_rng, &choice_key);
        int room_chunk = choice_valid(choice_key, room_occupancy, 9);
        room_occupancy[room_chunk] = false;
        int room_row = (room_chunk % world_chunk_height) * chunk_size + max_room_size;
        int room_col = (room_chunk / world_chunk_height) * chunk_size + max_room_size;
        Rng position_key;
        rng_split(room_rng, &room_rng, &position_key);
        room_row += randint_at(position_key, 0, 0, chunk_size - min_room_size);
        room_col += randint_at(position_key, 1, 0, chunk_size - min_room_size);
        room_positions[room_index][0] = room_row;
        room_positions[room_index][1] = room_col;

        for (int row = 0; row < max_room_size; row++) {
            for (int col = 0; col < max_room_size; col++) {
                if (row < room_sizes[room_index][0] && col < room_sizes[room_index][1]) {
                    padded_map[room_row + row][room_col + col] = BLOCK_PATH;
                }
            }
        }

        padded_item[room_row][room_col] = ITEM_TORCH;
        padded_item[room_row + room_sizes[room_index][0] - 1][room_col] = ITEM_TORCH;
        padded_item[room_row][room_col + room_sizes[room_index][1] - 1] = ITEM_TORCH;
        padded_item[room_row + room_sizes[room_index][0] - 1][room_col + room_sizes[room_index][1] - 1] = ITEM_TORCH;

        Rng chest_key;
        rng_split(room_rng, &room_rng, &chest_key);
        int chest_row = randint_at(chest_key, 0, 1, room_sizes[room_index][0] - 1);
        int chest_col = randint_at(chest_key, 1, 1, room_sizes[room_index][1] - 1);
        padded_map[room_row + chest_row][room_col + chest_col] = BLOCK_CHEST;

        Rng fountain_keys[3];
        rng_split_n(room_rng, fountain_keys, 3);
        room_rng = fountain_keys[0];
        int fountain_row = randint_at(fountain_keys[1], 0, 1, room_sizes[room_index][0] - 1);
        int fountain_col = randint_at(fountain_keys[1], 1, 1, room_sizes[room_index][1] - 1);
        if (rng_f32(fountain_keys[2]) > 0.5f) {
            padded_map[room_row + fountain_row][room_col + fountain_col] = config->fountain_block;
        }
    }

    Rng path_rng;
    rng_split(rng, &rng, &path_rng);
    bool included_rooms[8] = {false, false, false, false, false, false, false, true};
    for (int path_index = 0; path_index < num_rooms; path_index++) {
        int source_row = room_positions[path_index][0];
        int source_col = room_positions[path_index][1];
        Rng sink_key;
        rng_split(path_rng, &path_rng, &sink_key);
        int sink_index = choice_valid(sink_key, included_rooms, num_rooms);
        int sink_row = room_positions[sink_index][0];
        int sink_col = room_positions[sink_index][1];

        int horizontal_distance = sink_col - source_col;
        int horizontal_sign = (horizontal_distance > 0) - (horizontal_distance < 0);
        if (horizontal_sign != 0) {
            int abs_distance = horizontal_distance > 0 ? horizontal_distance : -horizontal_distance;
            for (int col = 0; col < padded_size; col++) {
                int path_index_col = (col - source_col) * horizontal_sign;
                if (path_index_col >= 0 && path_index_col <= abs_distance
                    && padded_map[source_row][col] == BLOCK_WALL) {
                    padded_map[source_row][col] = BLOCK_PATH;
                }
            }
        }
        int vertical_distance = sink_row - source_row;
        int vertical_sign = (vertical_distance > 0) - (vertical_distance < 0);
        if (vertical_sign != 0) {
            int abs_distance = vertical_distance > 0 ? vertical_distance : -vertical_distance;
            for (int row = 0; row < padded_size; row++) {
                int path_index_row = (row - source_row) * vertical_sign;
                if (path_index_row >= 0 && path_index_row <= abs_distance
                    && padded_map[row][sink_col] == BLOCK_WALL) {
                    padded_map[row][sink_col] = BLOCK_PATH;
                }
            }
        }

        Rng unused_left;
        Rng next_path_rng;
        rng_split(path_rng, &unused_left, &next_path_rng);
        path_rng = next_path_rng;
        included_rooms[path_index] = true;
    }

    padded_map[room_positions[0][0] + 2][room_positions[0][1] + 2] = config->special_block;

    for (int row = 0; row < MAP_SIZE; row++) {
        for (int col = 0; col < MAP_SIZE; col++) {
            state->map[level][row][col] =
                padded_map[row + max_room_size][col + max_room_size];
            state->item_map[level][row][col] =
                padded_item[row + max_room_size][col + max_room_size];
            state->mob_map[level][row][col] = 0;
        }
    }

    bool adjacent_path[MAP_SIZE][MAP_SIZE];
    for (int row = 0; row < MAP_SIZE; row++) {
        for (int col = 0; col < MAP_SIZE; col++) {
            bool adjacent = state->map[level][row][col] != BLOCK_WALL;
            adjacent = adjacent || (row > 0 && state->map[level][row - 1][col] != BLOCK_WALL);
            adjacent = adjacent || (row + 1 < MAP_SIZE && state->map[level][row + 1][col] != BLOCK_WALL);
            adjacent = adjacent || (col > 0 && state->map[level][row][col - 1] != BLOCK_WALL);
            adjacent = adjacent || (col + 1 < MAP_SIZE && state->map[level][row][col + 1] != BLOCK_WALL);
            adjacent_path[row][col] = adjacent;
        }
    }

    Rng rare_key;
    rng_split(rng, &rng, &rare_key);
    for (int row = 0; row < MAP_SIZE; row++) {
        for (int col = 0; col < MAP_SIZE; col++) {
            size_t idx = (size_t)cell_index(row, col);
            bool rare = (1.0f - rng_f32_at(rare_key, idx)) > 0.9f;
            int wall_map = rare ? BLOCK_WALL_MOSS : BLOCK_WALL;
            bool rare_path = rare
                && state->map[level][row][col] == BLOCK_PATH
                && state->item_map[level][row][col] == ITEM_NONE;
            int path_map = rare_path ? config->rare_path_replacement_block : state->map[level][row][col];
            bool is_wall_map = state->map[level][row][col] == BLOCK_WALL && adjacent_path[row][col];
            if (!adjacent_path[row][col]) {
                state->map[level][row][col] = BLOCK_DARKNESS;
            } else if (is_wall_map) {
                state->map[level][row][col] = wall_map;
            } else {
                state->map[level][row][col] = path_map;
            }
            state->light_map[level][row][col] = 255;
        }
    }

    bool valid_ladder[MAP_CELLS];
    for (int row = 0; row < MAP_SIZE; row++) {
        for (int col = 0; col < MAP_SIZE; col++) {
            valid_ladder[cell_index(row, col)] = state->map[level][row][col] == BLOCK_PATH;
        }
    }
    Rng ladder_down_key;
    rng_split(rng, &rng, &ladder_down_key);
    int ladder_down_index = choice_valid(ladder_down_key, valid_ladder, MAP_CELLS);
    state->down_ladders[level][0] = ladder_down_index / MAP_SIZE;
    state->down_ladders[level][1] = ladder_down_index % MAP_SIZE;
    state->item_map[level][state->down_ladders[level][0]][state->down_ladders[level][1]] =
        ITEM_LADDER_DOWN;

    Rng ladder_up_key;
    rng_split(rng, &rng, &ladder_up_key);
    int ladder_up_index = choice_valid(ladder_up_key, valid_ladder, MAP_CELLS);
    state->up_ladders[level][0] = ladder_up_index / MAP_SIZE;
    state->up_ladders[level][1] = ladder_up_index % MAP_SIZE;
    state->item_map[level][state->up_ladders[level][0]][state->up_ladders[level][1]] =
        ITEM_LADDER_UP;
}

static inline void permute_potions(Rng key, int out[6]) {
    Rng carry;
    Rng sort_key;
    rng_split(key, &carry, &sort_key);
    (void)carry;
    uint32_t keys[6];
    for (int i = 0; i < 6; i++) {
        keys[i] = rng_u32_at(sort_key, (uint64_t)i);
        out[i] = i;
    }
    for (int i = 1; i < 6; i++) {
        uint32_t key_value = keys[i];
        int value = out[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > key_value) {
            keys[j + 1] = keys[j];
            out[j + 1] = out[j];
            j--;
        }
        keys[j + 1] = key_value;
        out[j + 1] = value;
    }
}

static inline Rng worldgen_key_from_seed(uint32_t seed) {
    Rng key = rng_seed(seed);
    Rng carry;
    Rng reset_key;
    rng_split(key, &carry, &reset_key);
    Rng reset_carry;
    Rng world_key;
    rng_split(reset_key, &reset_carry, &world_key);
    return world_key;
}

void generate_world_from_key(State* state, Rng rng) {
    memset(state, 0, sizeof(*state));
    Rng smooth_split[7];
    rng_split_n(rng, smooth_split, 7);
    rng = smooth_split[0];

    static const int smooth_floor_order[6] = {0, 2, 5, 6, 7, 8};
    for (int i = 0; i < 6; i++) {
        generate_smooth_level(state, smooth_floor_order[i], smooth_split[i + 1], &SMOOTH_LEVEL_CONFIGS[i]);
    }

    Rng dungeon_split[4];
    rng_split_n(rng, dungeon_split, 4);
    rng = dungeon_split[0];
    static const int dungeon_floor_order[3] = {1, 3, 4};
    for (int i = 0; i < 3; i++) {
        generate_dungeon_level(state, dungeon_floor_order[i], dungeon_split[i + 1], &DUNGEON_LEVEL_CONFIGS[i]);
    }

    for (int level = 0; level < NUM_LEVELS; level++) {
        init_mobs(&state->melee_mobs[level]);
        init_mobs(&state->passive_mobs[level]);
        init_mobs(&state->ranged_mobs[level]);
        init_mobs(&state->mob_projectiles[level]);
        init_mobs(&state->player_projectiles[level]);
        for (int i = 0; i < 3; i++) {
            state->melee_mobs[level].health[i] = 1.0f;
            state->passive_mobs[level].health[i] = 1.0f;
            state->mob_projectiles[level].health[i] = 1.0f;
            state->player_projectiles[level].health[i] = 1.0f;
        }
        for (int i = 0; i < 2; i++) {
            state->ranged_mobs[level].health[i] = 1.0f;
        }
        for (int projectile = 0; projectile < MAX_MOB_PROJECTILES; projectile++) {
            state->mob_projectile_directions[level][projectile][0] = 1;
            state->mob_projectile_directions[level][projectile][1] = 1;
        }
        for (int projectile = 0; projectile < MAX_PLAYER_PROJECTILES; projectile++) {
            state->player_projectile_directions[level][projectile][0] = 1;
            state->player_projectile_directions[level][projectile][1] = 1;
        }
    }

    Rng potion_key;
    rng_split(rng, &rng, &potion_key);
    permute_potions(potion_key, state->potion_mapping);

    Rng state_key;
    rng_split(rng, &rng, &state_key);
    store_rng(state, state_key);

    state->monsters_killed[0] = 10;
    state->player_position[0] = MAP_SIZE / 2;
    state->player_position[1] = MAP_SIZE / 2;
    state->player_level = 0;
    state->player_direction = ACTION_UP;
    state->player_health = 9.0f;
    state->player_food = 9;
    state->player_drink = 9;
    state->player_energy = 9;
    state->player_mana = 9;
    state->player_dexterity = 1;
    state->player_strength = 1;
    state->player_intelligence = 1;
    state->boss_timestep_to_spawn_this_round = BOSS_SPAWN_TURNS;
    float cosine = cosf(3.14159265358979323846f * 0.3f);
    state->light_level = 1.0f - powf(fabsf(cosine), 3.0f);
}

void generate_world(State* state, int seed) {
    generate_world_from_key(state, worldgen_key_from_seed((uint32_t)seed));
}

static int g_clean_reset_pool_size = 0;
static State* g_clean_reset_pool = NULL;
static int g_clean_reset_pool_ready = 0;

static inline void craftax_clean_set_reset_pool_size(int n) {
    if (g_clean_reset_pool_ready) {
        return;
    }
    g_clean_reset_pool_size = n;
    if (n > 0) {
        g_clean_reset_pool = (State*)calloc((size_t)n, sizeof(State));
        for (int i = 0; i < n; i++) {
            Rng init_key = rng_seed((uint32_t)i);
            Rng discard;
            Rng reset_key;
            rng_split(init_key, &discard, &reset_key);
            Rng unused;
            Rng world_key;
            rng_split(reset_key, &unused, &world_key);
            generate_world_from_key(&g_clean_reset_pool[i], world_key);
        }
    }
    g_clean_reset_pool_ready = 1;
}

static inline void set_one_hot(float* obs, int* obs_idx, int index, int count) {
    if (index < 0) {
        index = 0;
    }
    if (index >= count) {
        index = count - 1;
    }
    for (int i = 0; i < count; i++) {
        obs[(*obs_idx)++] = i == index ? 1.0f : 0.0f;
    }
}

static inline bool scatter_index(int index, int size, int* mapped) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped = index < 0 ? index + size : index;
    return true;
}

static inline void scatter_mobs3(
    const State* state,
    const Mobs* mobs,
    int slots,
    int mob_class,
    uint8_t raw[OBS_ROWS][OBS_COLS][NUM_MOB_CLASSES][NUM_MOB_TYPES]
) {
    for (int i = 0; i < slots; i++) {
        int type_id = mobs->type_id[i];
        int local_row = mobs->position[i][0] - state->player_position[0] + OBS_ROWS / 2;
        int local_col = mobs->position[i][1] - state->player_position[1] + OBS_COLS / 2;
        int scatter_row;
        int scatter_col;
        if (!scatter_index(local_row, OBS_ROWS, &scatter_row)
            || !scatter_index(local_col, OBS_COLS, &scatter_col)
            || type_id < 0
            || type_id >= NUM_MOB_TYPES) {
            continue;
        }
        bool on_screen = local_row >= 0 && local_row < OBS_ROWS
            && local_col >= 0 && local_col < OBS_COLS;
        if (mobs->mask[i] && on_screen) {
            raw[scatter_row][scatter_col][mob_class][type_id] = 1u;
        }
    }
}

static inline void set_packed_mobs(float* obs, int* obs_idx, const State* state, int row, int col) {
    uint8_t raw[OBS_ROWS][OBS_COLS][NUM_MOB_CLASSES][NUM_MOB_TYPES];
    memset(raw, 0, sizeof(raw));
    int level = state->player_level;
    scatter_mobs3(state, &state->melee_mobs[level], MAX_MELEE_MOBS, 0, raw);
    scatter_mobs3(state, &state->passive_mobs[level], MAX_PASSIVE_MOBS, 1, raw);
    scatter_mobs3(state, &state->ranged_mobs[level], MAX_RANGED_MOBS, 2, raw);
    scatter_mobs3(state, &state->mob_projectiles[level], MAX_MOB_PROJECTILES, 3, raw);
    scatter_mobs3(state, &state->player_projectiles[level], MAX_PLAYER_PROJECTILES, 4, raw);

    int local_row = row - state->player_position[0] + OBS_ROWS / 2;
    int local_col = col - state->player_position[1] + OBS_COLS / 2;
    for (int cls = 0; cls < NUM_MOB_CLASSES; cls++) {
        int found = -1;
        if (local_row >= 0 && local_row < OBS_ROWS && local_col >= 0 && local_col < OBS_COLS) {
            for (int type_id = 0; type_id < NUM_MOB_TYPES; type_id++) {
                if (raw[local_row][local_col][cls][type_id]) {
                    found = type_id;
                    break;
                }
            }
        }
        obs[(*obs_idx)++] = (float)(found + 1);
    }
}

static inline bool boss_vulnerable(const State* state);

void compute_observations(Craftax* env) {
    State* state = &env->state;
    int obs_idx = 0;
    const int row_radius = OBS_ROWS / 2;
    const int col_radius = OBS_COLS / 2;

    int level = state->player_level;
    int (*map)[MAP_SIZE] = state->map[level];
    int (*item_map)[MAP_SIZE] = state->item_map[level];

    int row = state->player_position[0];
    int col = state->player_position[1];

    // Packed symbolic cells: block, item+1, visibility, then one type+1
    // value for melee, passive, ranged, mob projectile, player projectile.
    for (int r = -row_radius; r <= row_radius; r++) {
        for (int c = -col_radius; c <= col_radius; c++) {
            int obs_row = row + r;
            int obs_col = col + c;
            bool in_bounds = obs_row >= 0 && obs_row < MAP_SIZE
                && obs_col >= 0 && obs_col < MAP_SIZE;
            bool visible = in_bounds && state->light_map[level][obs_row][obs_col] > 12;
            int block = visible ? map[obs_row][obs_col] : 0;
            int item = visible ? item_map[obs_row][obs_col] + 1 : 0;

            env->agents[0].observations[obs_idx++] = (float)block;
            env->agents[0].observations[obs_idx++] = (float)item;
            env->agents[0].observations[obs_idx++] = visible ? 1.0f : 0.0f;
            if (visible) {
                set_packed_mobs(env->agents[0].observations, &obs_idx, state, obs_row, obs_col);
            } else {
                for (int i = 0; i < NUM_MOB_CLASSES; i++) {
                    env->agents[0].observations[obs_idx++] = 0.0f;
                }
            }
        }
    }

    // Player inventory (normalized)
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.wood) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.stone) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.coal) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.iron) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.diamond) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.sapphire) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.ruby) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.sapling) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.torches) / 10.0f;
    env->agents[0].observations[obs_idx++] = sqrtf((float)state->inventory.arrows) / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->inventory.books / 2.0f;
    env->agents[0].observations[obs_idx++] = (float)state->inventory.pickaxe / 4.0f;
    env->agents[0].observations[obs_idx++] = (float)state->inventory.sword / 4.0f;
    env->agents[0].observations[obs_idx++] = (float)state->sword_enchantment;
    env->agents[0].observations[obs_idx++] = (float)state->bow_enchantment;
    env->agents[0].observations[obs_idx++] = (float)state->inventory.bow;
    for (int i = 0; i < NUM_POTIONS; i++) {
        env->agents[0].observations[obs_idx++] =
            sqrtf((float)state->inventory.potions[i]) / 10.0f;
    }
    
    // Player intrinsic values (normalized)
    env->agents[0].observations[obs_idx++] = state->player_health / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_food / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_drink / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_energy / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_mana / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_xp / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_dexterity / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_strength / 10.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_intelligence / 10.0f;

    // One-hot encoded player direction (has 4 categories)
    int direction_index = state->player_direction - ACTION_LEFT;
    for (int i = 0; i < 4; i++) {
        env->agents[0].observations[obs_idx++] = i == direction_index ? 1.0f : 0.0f;
    }

    for (int i = 0; i < 4; i++) {
        env->agents[0].observations[obs_idx++] = (float)state->inventory.armour[i] / 2.0f;
    }

    for (int i = 0; i < 4; i++) {
        env->agents[0].observations[obs_idx++] = (float)state->armour_enchantments[i];
    }

    // Special values
    env->agents[0].observations[obs_idx++] = state->light_level;
    env->agents[0].observations[obs_idx++] = state->is_sleeping ? 1.0f : 0.0f;
    env->agents[0].observations[obs_idx++] = state->is_resting ? 1.0f : 0.0f;
    env->agents[0].observations[obs_idx++] = state->learned_spells[0] ? 1.0f : 0.0f;
    env->agents[0].observations[obs_idx++] = state->learned_spells[1] ? 1.0f : 0.0f;
    env->agents[0].observations[obs_idx++] = (float)state->player_level / 10.0f;
    env->agents[0].observations[obs_idx++] =
        state->monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL ? 1.0f : 0.0f;
    env->agents[0].observations[obs_idx++] = boss_vulnerable(state) ? 1.0f : 0.0f;

    if (obs_idx != OBS_SIZE) {
        fprintf(stderr, "craftax_clean: encoded %d values, expected %d\n", obs_idx, OBS_SIZE);
        abort();
    }
} 

#include "game_logic.h"

static inline void c_update_log_state(Craftax* env) {
    if (env->state.player_level > env->max_floor_accum) {
        env->max_floor_accum = env->state.player_level;
    }
}

// Reset function
static inline void reset_from_key(Craftax* env, Rng reset_key) {
    if (g_clean_reset_pool_size > 0) {
        uint32_t idx = reset_key.word[0] % (uint32_t)g_clean_reset_pool_size;
        memcpy(&env->state, &g_clean_reset_pool[idx], sizeof(State));
        refresh_mob_map(&env->state, env->state.player_level);
        return;
    }
    Rng unused;
    Rng world_key;
    rng_split(reset_key, &unused, &world_key);
    generate_world_from_key(&env->state, world_key);
    refresh_mob_map(&env->state, env->state.player_level);
}

void c_reset(Craftax* env) {
    if (env->agents[0].rewards != NULL) {
        env->agents[0].rewards[0] = 0.0f;
    }
    if (env->agents[0].terminals != NULL) {
        env->agents[0].terminals[0] = 0.0f;
    }
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    env->max_floor_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));

    Rng initial = rng_seed((uint32_t)env->seed);
    if (g_clean_reset_pool_size > 0) {
        Rng discard;
        rng_split(initial, &env->env_rng, &discard);
        int idx = (int)(env->seed % (uint64_t)g_clean_reset_pool_size);
        memcpy(&env->state, &g_clean_reset_pool[idx], sizeof(State));
        refresh_mob_map(&env->state, env->state.player_level);
        compute_observations(env);
        c_update_log_state(env);
        return;
    }
    Rng reset_key;
    rng_split(initial, &env->env_rng, &reset_key);
    reset_from_key(env, reset_key);

    compute_observations(env);
    c_update_log_state(env);
}

// Step function
void c_step(Craftax* env) {
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;

    int action = (int)env->agents[0].actions[0];
    if (action < 0) {
        action = ACTION_NOOP;
    }
    if (action >= NUM_ACTIONS) {
        action = NUM_ACTIONS - 1;
    }

    Rng step_key;
    rng_split(env->env_rng, &env->env_rng, &step_key);
    Rng step_rng;
    Rng reset_key;
    rng_split(step_key, &step_rng, &reset_key);

    State* state = &env->state;
    int initial_achievements[NUM_ACHIEVEMENTS];
    memcpy(initial_achievements, state->achievements, sizeof(initial_achievements));
    float initial_health = state->player_health;

    if (state->is_sleeping) {
        action = ACTION_NOOP;
    }
    if (state->is_resting) {
        action = ACTION_NOOP;
    }

    change_floor(state, action);
    craft_tools_and_items(state, action);
    interact_facing_tile(state, action, rng_key(&step_rng));
    place_block(state, action);
    use_projectile_or_spell(state, action);
    drink_potion(state, action);
    read_book(state, action, rng_key(&step_rng));
    enchant_items(state, action, rng_key(&step_rng));
    update_boss_logic(state);
    level_up_attributes(state, action);
    move_player(state, action);
    update_mobs(state, rng_key(&step_rng));
    spawn_mobs(state, rng_key(&step_rng));
    update_plants(state);
    update_intrinsics(state, action);
    clip_inventory_and_intrinsics(state);
    calculate_inventory_achievements(state);
    c_update_log_state(env);

    float reward = 0.0f;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        int delta = state->achievements[i] - initial_achievements[i];
        reward += (float)delta * ACHIEVEMENT_REWARD_MAP[i];
    }
    reward += (state->player_health - initial_health) * 0.1f;

    store_rng(state, rng_key(&step_rng));
    state->timestep += 1;
    state->light_level = calculate_light_level(state->timestep);

    bool done = is_game_over(&env->state);

    memcpy(env->achievements, env->state.achievements, sizeof(env->achievements));

    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;

    if (done) {
        int unlocked = 0;
        for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
            if (env->achievements[i]) {
                unlocked++;
                env->log.achievements[i] += 1.0f;
            }
        }
        env->log.perf += (float)unlocked / (float)NUM_ACHIEVEMENTS;
        env->log.score += env->episode_return_accum;
        env->log.episode_return += env->episode_return_accum;
        env->log.episode_length += (float)env->episode_length_accum;
        for (int floor = 0; floor <= env->max_floor_accum; floor++) {
            env->log.floors[floor] += 1.0f;
        }
        env->log.n += 1.0f;

        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        env->max_floor_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        reset_from_key(env, reset_key);
    }

    compute_observations(env);
}

#define TEX_TILE_PX 16
#define TEX_SCALE 3
#define TEX_DRAW_PX (TEX_TILE_PX * TEX_SCALE)
#define TEX_NUM (37 + 5 + 5 + 3 + 4)
#define RENDER_ROWS 14
#define RENDER_COLS 16
#define ACTION_PANEL_W 280
#define TEX_PLAYER_DOWN 37
#define TEX_PLAYER_UP 38
#define TEX_PLAYER_LEFT 39
#define TEX_PLAYER_RIGHT 40
#define TEX_PLAYER_SLEEP 41
#define TEX_ITEM_BASE 42
#define TEX_MOB_ZOMBIE 47
#define TEX_MOB_SKELETON 48
#define TEX_MOB_COW 49
#define TEX_ARROW_DOWN 50
#define TEX_ARROW_UP 51
#define TEX_ARROW_LEFT 52
#define TEX_ARROW_RIGHT 53

static Texture2D craftax_clean_textures[TEX_NUM];
static bool craftax_clean_textures_loaded = false;

static void craftax_clean_load_textures(void) {
    if (craftax_clean_textures_loaded) {
        return;
    }
    const char* candidates[] = {
        "resources/craftax/textures.bin",
        "../resources/craftax/textures.bin",
        "../../resources/craftax/textures.bin",
    };
    FILE* f = NULL;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        f = fopen(candidates[i], "rb");
        if (f != NULL) {
            break;
        }
    }
    if (f == NULL) {
        fprintf(stderr, "craftax_clean: textures.bin not found in resources/craftax\n");
        exit(1);
    }

    const size_t tile_bytes = TEX_TILE_PX * TEX_TILE_PX * 4;
    uint8_t* buf = (uint8_t*)malloc(tile_bytes);
    for (int i = 0; i < TEX_NUM; i++) {
        if (fread(buf, 1, tile_bytes, f) != tile_bytes) {
            fprintf(stderr, "craftax_clean: short read on textures.bin at tile %d\n", i);
            exit(1);
        }
        Image img = {
            .data = buf,
            .width = TEX_TILE_PX,
            .height = TEX_TILE_PX,
            .mipmaps = 1,
            .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
        };
        craftax_clean_textures[i] = LoadTextureFromImage(img);
        SetTextureFilter(craftax_clean_textures[i], TEXTURE_FILTER_POINT);
    }
    free(buf);
    fclose(f);
    craftax_clean_textures_loaded = true;
}

static int craftax_clean_player_tex_id(int direction, bool sleeping) {
    if (sleeping) {
        return TEX_PLAYER_SLEEP;
    }
    switch (direction) {
    case ACTION_LEFT: return TEX_PLAYER_LEFT;
    case ACTION_RIGHT: return TEX_PLAYER_RIGHT;
    case ACTION_UP: return TEX_PLAYER_UP;
    case ACTION_DOWN: return TEX_PLAYER_DOWN;
    default: return TEX_PLAYER_DOWN;
    }
}

static void craftax_clean_draw_tile(int tex_id, int dst_x, int dst_y, float tint_alpha) {
    if (tex_id < 0 || tex_id >= TEX_NUM) {
        return;
    }
    Rectangle src = {0, 0, TEX_TILE_PX, TEX_TILE_PX};
    Rectangle dst = {(float)dst_x, (float)dst_y, TEX_DRAW_PX, TEX_DRAW_PX};
    Color tint = {255, 255, 255, (unsigned char)(tint_alpha * 255.0f)};
    DrawTexturePro(craftax_clean_textures[tex_id], src, dst, (Vector2){0, 0}, 0.0f, tint);
}

static void craftax_clean_draw_tile_tinted(int tex_id, int dst_x, int dst_y, Color tint) {
    if (tex_id < 0 || tex_id >= TEX_NUM) {
        return;
    }
    Rectangle src = {0, 0, TEX_TILE_PX, TEX_TILE_PX};
    Rectangle dst = {(float)dst_x, (float)dst_y, TEX_DRAW_PX, TEX_DRAW_PX};
    DrawTexturePro(craftax_clean_textures[tex_id], src, dst, (Vector2){0, 0}, 0.0f, tint);
}

static void craftax_clean_draw_icon_count(int tex_id, int value, int x, int y) {
    const int icon_px = 20;
    if (tex_id >= 0 && tex_id < TEX_NUM) {
        Rectangle src = {0, 0, TEX_TILE_PX, TEX_TILE_PX};
        Rectangle dst = {(float)x, (float)y, icon_px, icon_px};
        DrawTexturePro(
            craftax_clean_textures[tex_id],
            src,
            dst,
            (Vector2){0, 0},
            0.0f,
            WHITE
        );
    }
    DrawText(TextFormat("%d", value), x + icon_px + 3, y + 4, 14, RAYWHITE);
}

static const char* craftax_clean_action_name(int action) {
    static const char* names[NUM_ACTIONS] = {
        "NOOP",
        "LEFT",
        "RIGHT",
        "UP",
        "DOWN",
        "DO",
        "SLEEP",
        "PLACE_STONE",
        "PLACE_TABLE",
        "PLACE_FURNACE",
        "PLACE_PLANT",
        "MAKE_WOOD_PICKAXE",
        "MAKE_STONE_PICKAXE",
        "MAKE_IRON_PICKAXE",
        "MAKE_WOOD_SWORD",
        "MAKE_STONE_SWORD",
        "MAKE_IRON_SWORD",
        "REST",
        "DESCEND",
        "ASCEND",
        "MAKE_DIAMOND_PICKAXE",
        "MAKE_DIAMOND_SWORD",
        "MAKE_IRON_ARMOUR",
        "MAKE_DIAMOND_ARMOUR",
        "SHOOT_ARROW",
        "MAKE_ARROW",
        "CAST_FIREBALL",
        "CAST_ICEBALL",
        "PLACE_TORCH",
        "DRINK_POTION_RED",
        "DRINK_POTION_GREEN",
        "DRINK_POTION_BLUE",
        "DRINK_POTION_PINK",
        "DRINK_POTION_CYAN",
        "DRINK_POTION_YELLOW",
        "READ_BOOK",
        "ENCHANT_SWORD",
        "ENCHANT_ARMOUR",
        "MAKE_TORCH",
        "LEVEL_UP_DEXTERITY",
        "LEVEL_UP_STRENGTH",
        "LEVEL_UP_INTELLIGENCE",
        "ENCHANT_BOW",
    };
    if (action < 0 || action >= NUM_ACTIONS) {
        return "NONE";
    }
    return names[action];
}

static const char* craftax_clean_action_key(int action) {
    switch (action) {
    case ACTION_NOOP: return "Q";
    case ACTION_LEFT: return "A";
    case ACTION_RIGHT: return "D";
    case ACTION_UP: return "W";
    case ACTION_DOWN: return "S";
    case ACTION_DO: return "Space";
    case ACTION_SLEEP: return "Tab";
    case ACTION_PLACE_STONE: return "R";
    case ACTION_PLACE_TABLE: return "T";
    case ACTION_PLACE_FURNACE: return "F";
    case ACTION_PLACE_PLANT: return "P";
    case ACTION_MAKE_WOOD_PICKAXE: return "1";
    case ACTION_MAKE_STONE_PICKAXE: return "2";
    case ACTION_MAKE_IRON_PICKAXE: return "3";
    case ACTION_MAKE_DIAMOND_PICKAXE: return "4";
    case ACTION_MAKE_WOOD_SWORD: return "5";
    case ACTION_MAKE_STONE_SWORD: return "6";
    case ACTION_MAKE_IRON_SWORD: return "7";
    case ACTION_MAKE_DIAMOND_SWORD: return "8";
    case ACTION_REST: return "E";
    case ACTION_ASCEND: return ",";
    case ACTION_DESCEND: return ".";
    case ACTION_MAKE_IRON_ARMOUR: return "Y";
    case ACTION_MAKE_DIAMOND_ARMOUR: return "U";
    case ACTION_SHOOT_ARROW: return "I";
    case ACTION_MAKE_ARROW: return "O";
    case ACTION_CAST_FIREBALL: return "G";
    case ACTION_CAST_ICEBALL: return "H";
    case ACTION_PLACE_TORCH: return "J";
    case ACTION_DRINK_POTION_RED: return "Z";
    case ACTION_DRINK_POTION_GREEN: return "X";
    case ACTION_DRINK_POTION_BLUE: return "C";
    case ACTION_DRINK_POTION_PINK: return "V";
    case ACTION_DRINK_POTION_CYAN: return "B";
    case ACTION_DRINK_POTION_YELLOW: return "N";
    case ACTION_READ_BOOK: return "M";
    case ACTION_ENCHANT_SWORD: return "K";
    case ACTION_ENCHANT_ARMOUR: return "L";
    case ACTION_MAKE_TORCH: return "[";
    case ACTION_LEVEL_UP_DEXTERITY: return "]";
    case ACTION_LEVEL_UP_STRENGTH: return "-";
    case ACTION_LEVEL_UP_INTELLIGENCE: return "=";
    case ACTION_ENCHANT_BOW: return ";";
    default: return "-";
    }
}

static Color craftax_clean_mob_tint(int mob_class, int type_id) {
    static const Color tints[NUM_MOB_TYPES] = {
        {255, 255, 255, 255},
        {220, 235, 255, 255},
        {235, 220, 255, 255},
        {255, 225, 190, 255},
        {210, 255, 210, 255},
        {190, 230, 255, 255},
        {255, 185, 145, 255},
        {210, 210, 210, 255},
    };
    (void)mob_class;
    return tints[clampi(type_id, 0, NUM_MOB_TYPES - 1)];
}

static void craftax_clean_draw_mob_marker(State* state, int level, int code, int x, int y) {
    int mob_class;
    int slot;
    decode_mob_map_code(code, &mob_class, &slot);

    int tex_id = TEX_MOB_ZOMBIE;
    if (mob_class == MOB_PASSIVE) {
        tex_id = TEX_MOB_COW;
    } else if (mob_class == MOB_MELEE) {
        tex_id = TEX_MOB_ZOMBIE;
    } else if (mob_class == MOB_RANGED) {
        tex_id = TEX_MOB_SKELETON;
    }

    int type_id = 0;
    Mobs* mobs = mobs_for_class(state, level, mob_class);
    if (slot >= 0 && slot < mob_slot_count(mob_class)) {
        type_id = mobs->type_id[slot];
    }
    craftax_clean_draw_tile_tinted(tex_id, x, y, craftax_clean_mob_tint(mob_class, type_id));
}

static int craftax_clean_projectile_tex(int dir_row, int dir_col) {
    if (dir_row < 0) return TEX_ARROW_UP;
    if (dir_row > 0) return TEX_ARROW_DOWN;
    if (dir_col < 0) return TEX_ARROW_LEFT;
    return TEX_ARROW_RIGHT;
}

static Color craftax_clean_projectile_tint(int projectile_type, bool from_player) {
    (void)from_player;
    if (projectile_type == PROJECTILE_FIREBALL || projectile_type == PROJECTILE_FIREBALL2) {
        return (Color){255, 135, 55, 255};
    }
    if (projectile_type == PROJECTILE_ICEBALL || projectile_type == PROJECTILE_ICEBALL2) {
        return (Color){115, 210, 255, 255};
    }
    if (projectile_type == PROJECTILE_SLIMEBALL) {
        return (Color){120, 235, 95, 255};
    }
    return WHITE;
}

static void craftax_clean_draw_projectiles(
    State* state,
    bool from_player,
    int level,
    int top_row,
    int left_col
) {
    Mobs* projectiles = from_player ? &state->player_projectiles[level] : &state->mob_projectiles[level];
    int (*directions)[MAX_PLAYER_PROJECTILES][2] =
        from_player ? state->player_projectile_directions : state->mob_projectile_directions;
    for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
        if (!projectiles->mask[i]) {
            continue;
        }
        int row = projectiles->position[i][0];
        int col = projectiles->position[i][1];
        int vr = row - top_row;
        int vc = col - left_col;
        if (vr < 0 || vr >= RENDER_ROWS || vc < 0 || vc >= RENDER_COLS) {
            continue;
        }
        int tex_id = craftax_clean_projectile_tex(directions[level][i][0], directions[level][i][1]);
        Color tint = craftax_clean_projectile_tint(projectiles->type_id[i], from_player);
        craftax_clean_draw_tile_tinted(tex_id, vc * TEX_DRAW_PX, vr * TEX_DRAW_PX, tint);
    }
}

void c_render(Craftax* env) {
    const int view_w = RENDER_COLS * TEX_DRAW_PX;
    const int view_h = RENDER_ROWS * TEX_DRAW_PX;
    const int hud_h = 122;
    const int window_w = view_w + ACTION_PANEL_W;

    if (env->client == NULL) {
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->cell_size = TEX_DRAW_PX;
        env->client->screen_width = window_w;
        env->client->screen_height = view_h + hud_h;
    }

    Client* client = env->client;
    if (!client->window_ready) {
        InitWindow(client->screen_width, client->screen_height, "Craftax Clean");
        SetTargetFPS(30);
        client->window_ready = true;
    }
    if (!craftax_clean_textures_loaded) {
        craftax_clean_load_textures();
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    int level = clampi(env->state.player_level, 0, NUM_LEVELS - 1);
    int player_row = clampi(env->state.player_position[0], 0, MAP_SIZE - 1);
    int player_col = clampi(env->state.player_position[1], 0, MAP_SIZE - 1);
    int half_r = RENDER_ROWS / 2;
    int half_c = RENDER_COLS / 2;
    int top_row = player_row - half_r;
    int left_col = player_col - half_c;

    BeginDrawing();
    ClearBackground(BLACK);

    for (int vr = 0; vr < RENDER_ROWS; vr++) {
        for (int vc = 0; vc < RENDER_COLS; vc++) {
            int wr = top_row + vr;
            int wc = left_col + vc;
            int dst_x = vc * TEX_DRAW_PX;
            int dst_y = vr * TEX_DRAW_PX;

            int block = BLOCK_OUT_OF_BOUNDS;
            if (wr >= 0 && wr < MAP_SIZE && wc >= 0 && wc < MAP_SIZE) {
                block = env->state.map[level][wr][wc];
            }
            if (block < 0 || block >= NUM_BLOCK_TYPES) {
                block = BLOCK_INVALID;
            }
            craftax_clean_draw_tile(block, dst_x, dst_y, 1.0f);

            if (wr >= 0 && wr < MAP_SIZE && wc >= 0 && wc < MAP_SIZE) {
                int item = env->state.item_map[level][wr][wc];
                if (item > ITEM_NONE && item < NUM_ITEM_TYPES) {
                    craftax_clean_draw_tile(TEX_ITEM_BASE + item, dst_x, dst_y, 1.0f);
                }
                int mob_code = env->state.mob_map[level][wr][wc];
                if (mob_code != 0) {
                    craftax_clean_draw_mob_marker(&env->state, level, mob_code, dst_x, dst_y);
                }
            }
        }
    }

    craftax_clean_draw_projectiles(&env->state, true, level, top_row, left_col);
    craftax_clean_draw_projectiles(&env->state, false, level, top_row, left_col);

    int player_tex = craftax_clean_player_tex_id(env->state.player_direction, env->state.is_sleeping);
    craftax_clean_draw_tile(player_tex, half_c * TEX_DRAW_PX, half_r * TEX_DRAW_PX, 1.0f);

    if (env->state.light_level < 1.0f) {
        unsigned char alpha = (unsigned char)((1.0f - env->state.light_level) * 140.0f);
        DrawRectangle(0, 0, view_w, view_h, (Color){0, 0, 40, alpha});
    }

    int hud_y = view_h;
    Inventory* inv = &env->state.inventory;
    DrawRectangle(0, hud_y, view_w, hud_h, (Color){20, 20, 20, 255});

    int health_max = max_health(&env->state);
    float health_frac = health_max > 0
        ? clampf(env->state.player_health / (float)health_max, 0.0f, 1.0f)
        : 0.0f;
    int bar_x = 4;
    int bar_y = hud_y + 4;
    int bar_w = view_w - 8;
    int bar_h = 18;
    DrawRectangle(bar_x, bar_y, bar_w, bar_h, (Color){115, 25, 25, 255});
    DrawRectangle(bar_x, bar_y, (int)((float)bar_w * health_frac), bar_h, (Color){35, 190, 75, 255});
    DrawRectangleLines(bar_x, bar_y, bar_w, bar_h, (Color){220, 220, 220, 255});
    DrawText(
        TextFormat("HP %.0f / %d", env->state.player_health, health_max),
        bar_x + 8,
        bar_y + 2,
        14,
        WHITE
    );

    DrawText(
        TextFormat(
            "Food:%d/%d  Drink:%d/%d  Energy:%d/%d  Mana:%d/%d  L:%d  t:%d",
            env->state.player_food,
            max_food(&env->state),
            env->state.player_drink,
            max_drink(&env->state),
            env->state.player_energy,
            max_energy(&env->state),
            env->state.player_mana,
            max_mana(&env->state),
            env->state.player_level,
            env->state.timestep
        ),
        4,
        hud_y + 26,
        14,
        WHITE
    );
    DrawText(
        TextFormat(
            "XP:%d  DEX:%d  STR:%d  INT:%d  light:%.2f  sleep:%d rest:%d",
            env->state.player_xp,
            env->state.player_dexterity,
            env->state.player_strength,
            env->state.player_intelligence,
            env->state.light_level,
            env->state.is_sleeping,
            env->state.is_resting
        ),
        4,
        hud_y + 44,
        14,
        (Color){200, 200, 200, 255}
    );
    int achievements = 0;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        achievements += env->state.achievements[i] ? 1 : 0;
    }
    int inv_y = hud_y + 62;
    int inv_x = 4;
    const int inv_step = 70;
    craftax_clean_draw_icon_count(BLOCK_TREE, inv->wood, inv_x + inv_step * 0, inv_y);
    craftax_clean_draw_icon_count(BLOCK_STONE, inv->stone, inv_x + inv_step * 1, inv_y);
    craftax_clean_draw_icon_count(BLOCK_COAL, inv->coal, inv_x + inv_step * 2, inv_y);
    craftax_clean_draw_icon_count(BLOCK_IRON, inv->iron, inv_x + inv_step * 3, inv_y);
    craftax_clean_draw_icon_count(BLOCK_DIAMOND, inv->diamond, inv_x + inv_step * 4, inv_y);
    craftax_clean_draw_icon_count(BLOCK_PLANT, inv->sapling, inv_x + inv_step * 5, inv_y);
    craftax_clean_draw_icon_count(TEX_ITEM_BASE + ITEM_TORCH, inv->torches, inv_x + inv_step * 6, inv_y);
    craftax_clean_draw_icon_count(BLOCK_RUBY, inv->ruby, inv_x + inv_step * 7, inv_y);
    craftax_clean_draw_icon_count(BLOCK_SAPPHIRE, inv->sapphire, inv_x + inv_step * 8, inv_y);
    craftax_clean_draw_icon_count(BLOCK_CHEST, inv->books, inv_x + inv_step * 9, inv_y);
    DrawText(
        TextFormat(
            "pick:%d sword:%d bow:%d arrows:%d armour:%d/%d/%d/%d",
            inv->pickaxe,
            inv->sword,
            inv->bow,
            inv->arrows,
            inv->armour[0],
            inv->armour[1],
            inv->armour[2],
            inv->armour[3]
        ),
        4,
        hud_y + 86,
        14,
        (Color){190, 210, 230, 255}
    );
    DrawText(
        TextFormat(
            "potions:%d/%d/%d/%d/%d/%d  ach:%d/%d  ret:%.2f len:%d",
            inv->potions[0],
            inv->potions[1],
            inv->potions[2],
            inv->potions[3],
            inv->potions[4],
            inv->potions[5],
            achievements,
            NUM_ACHIEVEMENTS,
            env->episode_return_accum,
            env->episode_length_accum
        ),
        4,
        hud_y + 98,
        14,
        (Color){200, 200, 140, 255}
    );

    int panel_x = view_w;
    int panel_h = view_h + hud_h;
    int taken_action = (int)env->agents[0].actions[0];
    DrawRectangle(panel_x, 0, ACTION_PANEL_W, panel_h, (Color){12, 18, 22, 255});
    DrawRectangleLines(panel_x, 0, ACTION_PANEL_W, panel_h, (Color){55, 70, 76, 255});
    DrawText("Actions", panel_x + 10, 8, 18, RAYWHITE);
    DrawText("key", panel_x + 12, 32, 11, (Color){140, 160, 166, 255});
    DrawText("action", panel_x + 78, 32, 11, (Color){140, 160, 166, 255});
    for (int action = 0; action < NUM_ACTIONS; action++) {
        int y = 48 + action * 15;
        bool selected = action == taken_action;
        if (selected) {
            DrawRectangle(panel_x + 6, y - 2, ACTION_PANEL_W - 12, 15, (Color){0, 210, 220, 255});
        }
        Color text_color = selected ? BLACK : (Color){220, 230, 230, 255};
        DrawText(craftax_clean_action_key(action), panel_x + 12, y, 10, text_color);
        DrawText(TextFormat("%02d %s", action, craftax_clean_action_name(action)),
            panel_x + 78, y, 10, text_color);
    }

    EndDrawing();
    puf_web_vsync();
}

void c_close(Craftax* env) {
    if (env->client == NULL) {
        return;
    }

    if (env->client->window_ready) {
        CloseWindow();
    }

    free(env->client);
    env->client = NULL;
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    uint64_t seed_offset = 0;
    int reset_pool_size = 0;
    for (int i = 0; i < kwargs->size; i++) {
        if (strcmp(kwargs->items[i].key, "seed_offset") == 0) {
            seed_offset = (uint64_t)kwargs->items[i].value;
        }
        if (strcmp(kwargs->items[i].key, "reset_pool_size") == 0) {
            reset_pool_size = (int)kwargs->items[i].value;
        }
    }
    env->seed = seed_offset + (uint64_t)env->rng;
    craftax_clean_set_reset_pool_size(reset_pool_size);
    c_init(env);
}

void puf_reset(Env* env) {
    c_reset(env);
}

void puf_step(Env* env) {
    c_step(env);
}

void puf_render(Env* env) {
    c_render(env);
}

void puf_close(Env* env) {
    c_close(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "floor_0_overworld", log->floors[0]);
    dict_set(out, "floor_1_dungeon", log->floors[1]);
    dict_set(out, "floor_2_gnomish_mines", log->floors[2]);
    dict_set(out, "floor_3_sewers", log->floors[3]);
    dict_set(out, "floor_4_vault", log->floors[4]);
    dict_set(out, "floor_5_troll_mines", log->floors[5]);
    dict_set(out, "floor_6_fire_realm", log->floors[6]);
    dict_set(out, "floor_7_ice_realm", log->floors[7]);
    dict_set(out, "floor_8_graveyard", log->floors[8]);
    dict_set(out, "n", log->n);
}
