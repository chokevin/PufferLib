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
typedef Env Craftax;

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
    unsigned char light_map[NUM_LEVELS][MAP_SIZE][MAP_SIZE];
    uint64_t mob_bits[NUM_LEVELS][MAP_SIZE];
    uint64_t spawn_land[NUM_LEVELS][MAP_SIZE];
    uint64_t spawn_grave[NUM_LEVELS][MAP_SIZE];
    uint64_t spawn_water[NUM_LEVELS][MAP_SIZE];
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

static void c_init(Craftax* env) {
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

// World generation

// RNG functions to achieve parity with JAX version
uint64_t rng_to_u64(Rng key) {
    return ((uint64_t)key.word[1] << 32) | key.word[0];
}

Rng rng_from_u64(uint64_t x) {
    return (Rng){{(uint32_t)x, (uint32_t)(x >> 32)}};
}

Rng rng_seed(uint32_t seed) {
    return (Rng){{seed, seed ^ 0x9E3779B9u}};
}

uint64_t rng_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t rng_hash64(Rng key, uint64_t counter) {
    return rng_mix64(rng_to_u64(key) ^ counter);
}

void rng_split(Rng key, Rng* left, Rng* right) {
    uint64_t state = rng_to_u64(key);
    uint64_t s1 = state * 6364136223846793005ULL + 1;
    uint64_t s2 = s1 * 6364136223846793005ULL + 1;
    *left = rng_from_u64(s1);
    *right = rng_from_u64(s2);
}

void rng_split_n(Rng key, Rng* out, int count) {
    uint64_t state = rng_to_u64(key);
    for (int i = 0; i < count; i++) {
        state = state * 6364136223846793005ULL + 1;
        out[i] = rng_from_u64(state);
    }
}

uint32_t rng_u32_at(Rng key, uint64_t index) {
    uint64_t h = rng_hash64(key, index);
    return (uint32_t)h ^ (uint32_t)(h >> 32);
}

float rng_f32_at(Rng key, uint64_t index) {
    uint32_t bits = rng_u32_at(key, index);
    uint32_t float_bits = (bits >> 9u) | 0x3F800000u;
    float value;
    memcpy(&value, &float_bits, sizeof(value));
    return value - 1.0f;
}

float rng_f32(Rng key) {
    return rng_f32_at(key, 0u);
}

void store_rng(State* state, Rng rng) {
    state->state_rng[0] = rng.word[0];
    state->state_rng[1] = rng.word[1];
}

Rng rng_key(Rng* rng) {
    Rng next;
    Rng draw;
    rng_split(*rng, &next, &draw);
    *rng = next;
    return draw;
}

static int choice_valid(Rng key, const bool* valid, int count) {
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

static uint32_t randint_u32_at(Rng key, uint64_t index, uint32_t minval, uint32_t maxval) {
    uint32_t span = maxval > minval ? maxval - minval : 1u;
    if ((span & (span - 1)) == 0) {
        uint32_t bits = rng_u32_at(key, index);
        return minval + (bits & (span - 1));
    }
    uint64_t h = rng_hash64(key, index);
    return minval + (uint32_t)(((h >> 32) * (uint64_t)span) >> 32);
}

static int randint_at(Rng key, uint64_t index, int minval, int maxval) {
    return (int)randint_u32_at(key, index, (uint32_t)minval, (uint32_t)maxval);
}

static void refresh_spawn_cell(State* state, int level, int row, int col) {
    int block = state->map[level][row][col];
    uint64_t bit = 1ull << col;
    uint64_t* land = &state->spawn_land[level][row];
    uint64_t* grave = &state->spawn_grave[level][row];
    uint64_t* water = &state->spawn_water[level][row];
    *land = (*land & ~bit) | ((block == BLOCK_GRASS || block == BLOCK_PATH
        || block == BLOCK_FIRE_GRASS || block == BLOCK_ICE_GRASS) ? bit : 0);
    *grave = (*grave & ~bit) | ((block == BLOCK_GRAVE || block == BLOCK_GRAVE2
        || block == BLOCK_GRAVE3) ? bit : 0);
    *water = (*water & ~bit) | (block == BLOCK_WATER ? bit : 0);
}

static void set_block(State* state, int level, int row, int col, int block) {
    state->map[level][row][col] = block;
    refresh_spawn_cell(state, level, row, col);
}

static void refresh_spawn_maps(State* state) {
    memset(state->spawn_land, 0, sizeof(state->spawn_land));
    memset(state->spawn_grave, 0, sizeof(state->spawn_grave));
    memset(state->spawn_water, 0, sizeof(state->spawn_water));
    for (int level = 0; level < NUM_LEVELS; level++) {
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                refresh_spawn_cell(state, level, row, col);
            }
        }
    }
}

static float torch_light_at(int dr, int dc) {
    float torch = 1.0f - sqrtf((float)(dr * dr + dc * dc)) / 5.0f;
    return torch < 0.0f ? 0.0f : torch;
}

void add_light(State* state, int level, int center_row, int center_col) {
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
                + torch_light_at(dr, dc);
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

float noise_interpolant(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

void generate_perlin(
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

void generate_fractal(
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

void apply_ladder_light(
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
            float light = torch_light_at(row - 4, col - 4) * (1.0f - default_light) + default_light;
            light_map[start_row + row][start_col + col] = (unsigned char)(light * 255.0f);
        }
    }
}

void add_lava_light(
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

int cell_index(int row, int col) {
    return row * MAP_SIZE + col;
}

static void generate_smooth_level(
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

static void generate_dungeon_level(
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

static void permute_potions(Rng key, int out[6]) {
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
        for (int i = 0; i < MAX_MELEE_MOBS; i++) {
            state->melee_mobs[level].health[i] = 1.0f;
            state->passive_mobs[level].health[i] = 1.0f;
            state->mob_projectiles[level].health[i] = 1.0f;
            state->player_projectiles[level].health[i] = 1.0f;
        }
        for (int i = 0; i < MAX_RANGED_MOBS; i++) {
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
    refresh_spawn_maps(state);
}

// TODO: do we need this reset pool stuff?
static int g_clean_reset_pool_size = 0;
static State* g_clean_reset_pool = NULL;
static int g_clean_reset_pool_ready = 0;

void craftax_clean_set_reset_pool_size(int n) {
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

static void write_mob_obs(
    float* obs,
    const State* state,
    const Mobs* mobs,
    int slots,
    int channel
) {
    int level = state->player_level;
    for (int i = 0; i < slots; i++) {
        if (!mobs->mask[i]) {
            continue;
        }
        int type_id = mobs->type_id[i];
        if (type_id < 0 || type_id >= NUM_MOB_TYPES) {
            continue;
        }
        int world_row = mobs->position[i][0];
        int world_col = mobs->position[i][1];
        int local_row = world_row - state->player_position[0] + OBS_ROWS / 2;
        int local_col = world_col - state->player_position[1] + OBS_COLS / 2;
        if (local_row < 0 || local_row >= OBS_ROWS
            || local_col < 0 || local_col >= OBS_COLS) {
            continue;
        }
        if (world_row < 0 || world_row >= MAP_SIZE
            || world_col < 0 || world_col >= MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }
        int base = (local_row * OBS_COLS + local_col) * OBS_TILE_CHANNELS;
        obs[base + 3 + channel] = (float)(type_id + 1);
    }
}

int clampi(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

float clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

int max_health(const State* state) {
    return 8 + state->player_strength;
}

int max_food(const State* state) {
    return 7 + 2 * state->player_dexterity;
}

int max_drink(const State* state) {
    return 7 + 2 * state->player_dexterity;
}

int max_energy(const State* state) {
    return 7 + 2 * state->player_dexterity;
}

int max_mana(const State* state) {
    return 6 + 3 * state->player_intelligence;
}

bool fighting_boss(const State* state) {
    return state->player_level == NUM_LEVELS - 1;
}

bool boss_vulnerable(const State* state) {
    if (state->boss_timestep_to_spawn_this_round > 0) {
        return false;
    }
    int level = state->player_level;
    for (int i = 0; i < MAX_MELEE_MOBS; i++) {
        if (state->melee_mobs[level].mask[i]) {
            return false;
        }
    }
    for (int i = 0; i < MAX_RANGED_MOBS; i++) {
        if (state->ranged_mobs[level].mask[i]) {
            return false;
        }
    }
    return true;
}

bool has_beaten_boss(const State* state) {
    return state->boss_progress >= NUM_LEVELS - 1;
}

void action_to_direction(int action, int direction[2]) {
    direction[0] = 0;
    direction[1] = 0;

    if (action == ACTION_LEFT) {
        direction[1] = -1;
    } else if (action == ACTION_RIGHT) {
        direction[1] = 1;
    } else if (action == ACTION_UP) {
        direction[0] = -1;
    } else if (action == ACTION_DOWN) {
        direction[0] = 1;
    }
}

bool is_solid_block(int block) {
    switch (block) {
    case BLOCK_STONE:
    case BLOCK_TREE:
    case BLOCK_COAL:
    case BLOCK_IRON:
    case BLOCK_DIAMOND:
    case BLOCK_CRAFTING_TABLE:
    case BLOCK_FURNACE:
    case BLOCK_PLANT:
    case BLOCK_RIPE_PLANT:
    case BLOCK_WALL:
    case BLOCK_WALL_MOSS:
    case BLOCK_STALAGMITE:
    case BLOCK_RUBY:
    case BLOCK_SAPPHIRE:
    case BLOCK_CHEST:
    case BLOCK_FOUNTAIN:
    case BLOCK_FIRE_TREE:
    case BLOCK_ENCHANTMENT_TABLE_FIRE:
    case BLOCK_ENCHANTMENT_TABLE_ICE:
    case BLOCK_GRAVE:
    case BLOCK_GRAVE2:
    case BLOCK_GRAVE3:
    case BLOCK_NECROMANCER:
        return true;
    default:
        return false;
    }
}

bool mob_at(const State* state, int level, int row, int col) {
    if ((unsigned)row >= MAP_SIZE || (unsigned)col >= MAP_SIZE) {
        return false;
    }
    return (state->mob_bits[level][row] >> col) & 1ull;
}

void set_mob_bit(State* state, int level, int row, int col, bool on) {
    if ((unsigned)row >= MAP_SIZE || (unsigned)col >= MAP_SIZE) {
        return;
    }
    uint64_t bit = 1ull << col;
    if (on) {
        state->mob_bits[level][row] |= bit;
    } else {
        state->mob_bits[level][row] &= ~bit;
    }
}

void move_mob_occupancy(
    State* state, int level, int old_row, int old_col, int new_row, int new_col, bool keep
) {
    set_mob_bit(state, level, old_row, old_col, false);
    if (keep) {
        set_mob_bit(state, level, new_row, new_col, true);
    }
}

bool mobs_at(const Mobs* mobs, int slots, int row, int col, int* slot) {
    for (int i = 0; i < slots; i++) {
        if (mobs->mask[i]
            && mobs->position[i][0] == row
            && mobs->position[i][1] == col) {
            *slot = i;
            return true;
        }
    }
    return false;
}

bool valid_player_position(const State* state, int row, int col) {
    if (row < 0 || row >= MAP_SIZE || col < 0 || col >= MAP_SIZE) {
        return false;
    }

    int level = state->player_level;
    int block = state->map[level][row][col];
    if (is_solid_block(block)) {
        return false;
    }
    if (block == BLOCK_WATER || block == BLOCK_LAVA) {
        return false;
    }
    return !mob_at(state, level, row, col);
}

Mobs* mobs_for_class(State* state, int level, int mob_class) {
    if (mob_class == MOB_PASSIVE) {
        return &state->passive_mobs[level];
    }
    if (mob_class == MOB_RANGED) {
        return &state->ranged_mobs[level];
    }
    return &state->melee_mobs[level];
}

bool find_mob_at(
    const State* state,
    int level,
    int row,
    int col,
    int* mob_class,
    int* slot
) {
    if (mobs_at(&state->melee_mobs[level], MAX_MELEE_MOBS, row, col, slot)) {
        *mob_class = MOB_MELEE;
        return true;
    }
    if (mobs_at(&state->passive_mobs[level], MAX_PASSIVE_MOBS, row, col, slot)) {
        *mob_class = MOB_PASSIVE;
        return true;
    }
    if (mobs_at(&state->ranged_mobs[level], MAX_RANGED_MOBS, row, col, slot)) {
        *mob_class = MOB_RANGED;
        return true;
    }
    return false;
}

bool mob_can_move_on(int mob_class, int type_id, int block) {
    static const bool blocked[NUM_MOB_TYPES][3][3] = {
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,0,0},{0,1,1},{0,1,1}},
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,1,1},{0,0,1},{0,1,1}},
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,1,1},{0,1,1},{1,0,1}},
        {{0,1,1},{0,1,1},{0,0,0}}, {{0,1,1},{0,1,1},{0,0,0}},
    };
    if (is_solid_block(block)) return false;
    int terrain = block == BLOCK_WATER ? 1 : (block == BLOCK_LAVA ? 2 : 0);
    return !blocked[clampi(type_id,0,7)][clampi(mob_class,0,2)][terrain];
}

bool valid_typed_mob_position(
    const State* state,
    int level,
    int mob_class,
    int type_id,
    int row,
    int col,
    int old_row,
    int old_col
) {
    if (row < 0 || row >= MAP_SIZE || col < 0 || col >= MAP_SIZE) {
        return false;
    }
    if (row == state->player_position[0] && col == state->player_position[1]) {
        return false;
    }
    if (!mob_can_move_on(mob_class, type_id, state->map[level][row][col])) {
        return false;
    }
    return !mob_at(state, level, row, col) || (row == old_row && col == old_col);
}

float mob_base_health(int mob_class, int type_id) {
    static const float passive_health[NUM_MOB_TYPES] = {3, 4, 6, 8, 0, 0, 0, 0};
    static const float melee_health[NUM_MOB_TYPES] = {5, 7, 9, 11, 12, 20, 20, 24};
    static const float ranged_health[NUM_MOB_TYPES] = {3, 5, 6, 8, 12, 4, 14, 16};
    int idx = clampi(type_id, 0, NUM_MOB_TYPES - 1);
    if (mob_class == MOB_PASSIVE) return passive_health[idx];
    if (mob_class == MOB_RANGED) return ranged_health[idx];
    return melee_health[idx];
}

typedef struct { float physical, fire, ice; } Damage;

Damage player_attack_damage_vector(const State* state) {
    static const float base_damage[5] = {1, 2, 3, 5, 8};
    float base = base_damage[clampi(state->inventory.sword, 0, 4)];
    float physical = base * (1.0f + 0.25f * (float)(state->player_strength - 1));
    float magic = base * 0.5f * (1.0f + 0.05f * (float)(state->player_intelligence - 1));
    return (Damage){physical, state->sword_enchantment == 1 ? magic : 0, state->sword_enchantment == 2 ? magic : 0};
}

Damage mob_damage_vector(int type, int mob_class) {
    static const float damage[NUM_MOB_TYPES][4][3] = {
        {{0,0,0},{2,0,0},{0,0,0},{2,0,0}}, {{0,0,0},{4,0,0},{0,0,0},{4,0,0}},
        {{0,0,0},{3,0,0},{0,0,0},{0,3,0}}, {{0,0,0},{5,0,0},{0,0,0},{0,0,3}},
        {{0,0,0},{6,0,0},{0,0,0},{5,0,0}}, {{0,0,0},{6,1,1},{0,0,0},{4,3,3}},
        {{0,0,0},{3,5,0},{0,0,0},{3,5,0}}, {{0,0,0},{4,0,5},{0,0,0},{4,0,5}},
    };
    const float* d = damage[clampi(type, 0, 7)][clampi(mob_class, 0, 3)];
    return (Damage){d[0], d[1], d[2]};
}

float damage_to_mob(Damage damage, int type, int mob_class) {
    static const float defense[NUM_MOB_TYPES][4][3] = {
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{.5f,0,0},{.5f,0,0},{0,0,0}},
        {{0,0,0},{.2f,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{.9f,1,0},{.9f,1,0},{0,0,0}},
        {{0,0,0},{.9f,0,1},{.9f,0,1},{0,0,0}},
    };
    const float* d = defense[clampi(type, 0, 7)][clampi(mob_class, 0, 3)];
    return damage.physical * (1-d[0]) + damage.fire * (1-d[1]) + damage.ice * (1-d[2]);
}

float damage_to_player(const State* state, Damage damage) {
    float physical_defense = 0, fire_defense = 0, ice_defense = 0;
    for (int i = 0; i < 4; i++) {
        physical_defense += 0.1f * state->inventory.armour[i];
        fire_defense += 0.2f * (state->armour_enchantments[i] == 1);
        ice_defense += 0.2f * (state->armour_enchantments[i] == 2);
    }
    float coeff = fighting_boss(state) ? 1.5f : 1.0f;
    return coeff * (damage.physical * (1-physical_defense) + damage.fire * (1-fire_defense) + damage.ice * (1-ice_defense));
}

int defeat_achievement(int mob_class, int type_id, int level) {
    (void)level;
    static const int achievements[3][8] = {
        {ACH_EAT_COW, ACH_EAT_BAT, ACH_EAT_SNAIL, 0,0,0,0,0},
        {ACH_DEFEAT_ZOMBIE, ACH_DEFEAT_GNOME_WARRIOR, ACH_DEFEAT_ORC_SOLIDER, ACH_DEFEAT_LIZARD, ACH_DEFEAT_KNIGHT, ACH_DEFEAT_TROLL, ACH_DEFEAT_PIGMAN, ACH_DEFEAT_FROST_TROLL},
        {ACH_DEFEAT_SKELETON, ACH_DEFEAT_GNOME_ARCHER, ACH_DEFEAT_ORC_MAGE, ACH_DEFEAT_KOBOLD, ACH_DEFEAT_ARCHER, ACH_DEFEAT_DEEP_THING, ACH_DEFEAT_FIRE_ELEMENTAL, ACH_DEFEAT_ICE_ELEMENTAL},
    };
    return achievements[clampi(mob_class, 0, 2)][clampi(type_id, 0, 7)];
}

bool damage_mob_at(
    State* state, int level, int row, int col, float damage,
    bool can_eat, bool can_get_achievement
) {
    int mob_class;
    int slot;
    if (!find_mob_at(state, level, row, col, &mob_class, &slot)) {
        return false;
    }
    Mobs* mobs = mobs_for_class(state, level, mob_class);
    if (!mobs->mask[slot]) {
        return false;
    }

    mobs->health[slot] -= damage;
    if (mobs->health[slot] > 0.0f) {
        return true;
    }

    int type_id = mobs->type_id[slot];
    mobs->mask[slot] = false;
    set_mob_bit(state, level, row, col, false);
    state->monsters_killed[level] += mob_class == MOB_PASSIVE ? 0 : 1;
    if (can_get_achievement) {
        state->achievements[defeat_achievement(mob_class, type_id, level)] = 1;
    }

    if (mob_class == MOB_PASSIVE && can_eat) {
        state->player_food = clampi(state->player_food + 6, 0, max_food(state));
        state->player_hunger = 0.0f;
    }
    return true;
}

bool attack_mob_at(State* state, int level, int row, int col, bool can_eat) {
    int mob_class;
    int slot;
    if (!find_mob_at(state, level, row, col, &mob_class, &slot)) {
        return false;
    }
    Mobs* mobs = mobs_for_class(state, level, mob_class);
    Damage vector = player_attack_damage_vector(state);
    return damage_mob_at(
        state, level, row, col,
        damage_to_mob(vector, mobs->type_id[slot], mob_class),
        can_eat, true
    );
}

float projectile_damage(int projectile_type, bool from_player) {
    (void)from_player;
    Damage d = mob_damage_vector(projectile_type, MOB_PROJECTILE);
    return d.physical + d.fire + d.ice;
}

Damage player_projectile_damage(const State* state, int type) {
    Damage damage = mob_damage_vector(type, MOB_PROJECTILE);
    bool arrow = type == PROJECTILE_ARROW || type == PROJECTILE_ARROW2;
    if (arrow && state->bow_enchantment == 1) damage.fire += damage.physical * 0.5f;
    if (arrow && state->bow_enchantment == 2) damage.ice += damage.physical * 0.5f;
    float coeff = arrow ? 1.0f + 0.2f * (state->player_dexterity - 1) :
        ((type == PROJECTILE_FIREBALL || type == PROJECTILE_ICEBALL) ? 1.0f + 0.5f * (state->player_intelligence - 1) : 1.0f);
    damage.physical *= coeff; damage.fire *= coeff; damage.ice *= coeff;
    return damage;
}

bool spawn_projectile(
    State* state,
    bool from_player,
    int projectile_type,
    int row,
    int col,
    int dir_row,
    int dir_col
) {
    int level = state->player_level;
    Mobs* projectiles = from_player ? &state->player_projectiles[level] : &state->mob_projectiles[level];
    int (*directions)[MAX_PLAYER_PROJECTILES][2] =
        from_player ? state->player_projectile_directions : state->mob_projectile_directions;
    for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
        if (projectiles->mask[i]) {
            continue;
        }
        projectiles->position[i][0] = row;
        projectiles->position[i][1] = col;
        projectiles->health[i] = projectile_damage(projectile_type, from_player);
        projectiles->attack_cooldown[i] = 0;
        projectiles->type_id[i] = projectile_type;
        projectiles->mask[i] = true;
        directions[level][i][0] = dir_row;
        directions[level][i][1] = dir_col;
        return true;
    }
    return false;
}

bool projectile_in_mob(const State* state, int level, int row, int col) {
    bool player_here = state->player_position[0] == row
        && state->player_position[1] == col;
    return mob_at(state, level, row, col) || player_here;
}

void scatter_set_block(State* state, int level, int row, int col, int block) {
    if ((unsigned)level >= NUM_LEVELS || (unsigned)row >= MAP_SIZE || (unsigned)col >= MAP_SIZE) {
        return;
    }
    set_block(state, level, row, col, block);
}

void move_mob_projectile_slot(State* state, int level, int slot) {
    Mobs* projectiles = &state->mob_projectiles[level];
    bool alive = projectiles->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = projectiles->position[slot][0];
    int old_col = projectiles->position[slot][1];
    int proposed_row = old_row + state->mob_projectile_directions[level][slot][0];
    int proposed_col = old_col + state->mob_projectile_directions[level][slot][1];
    bool proposed_in_player = proposed_row == state->player_position[0]
        && proposed_col == state->player_position[1];
    bool proposed_in_bounds = proposed_row >= 0 && proposed_row < MAP_SIZE
        && proposed_col >= 0 && proposed_col < MAP_SIZE;
    int proposed_block = proposed_in_bounds ? state->map[level][proposed_row][proposed_col] : 0;
    bool in_wall = is_solid_block(proposed_block) && proposed_block != BLOCK_WATER;
    bool in_mob = projectile_in_mob(state, level, proposed_row, proposed_col);
    bool keep_moving = proposed_in_bounds && !in_wall && !in_mob;
    bool hit_player = (
        (old_row == state->player_position[0] && old_col == state->player_position[1])
        || proposed_in_player
    ) && alive;
    keep_moving = keep_moving && !hit_player;
    bool keep = keep_moving && alive;
    bool hit_bench = proposed_block == BLOCK_FURNACE
        || proposed_block == BLOCK_CRAFTING_TABLE;
    int new_block = (hit_bench && alive) ? BLOCK_PATH : proposed_block;

    projectiles->position[slot][0] = proposed_row;
    projectiles->position[slot][1] = proposed_col;
    projectiles->mask[slot] = keep;
    if (hit_player) {
        state->player_health -= damage_to_player(
            state, mob_damage_vector(projectiles->type_id[slot], MOB_PROJECTILE));
        state->is_sleeping = false;
        state->is_resting = false;
    }
    scatter_set_block(state, level, proposed_row, proposed_col, new_block);
}

void move_player_projectile_slot(State* state, int level, int slot) {
    Mobs* projectiles = &state->player_projectiles[level];
    bool alive = projectiles->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = projectiles->position[slot][0];
    int old_col = projectiles->position[slot][1];
    int proposed_row = old_row + state->player_projectile_directions[level][slot][0];
    int proposed_col = old_col + state->player_projectile_directions[level][slot][1];
    Damage vector = player_projectile_damage(state, projectiles->type_id[slot]);
    if (!alive) {
        vector.physical = 0.0f;
        vector.fire = 0.0f;
        vector.ice = 0.0f;
    }

    bool hit_old = false;
    int mob_class;
    int mob_slot;
    if (find_mob_at(state, level, old_row, old_col, &mob_class, &mob_slot)) {
        Mobs* target = mobs_for_class(state, level, mob_class);
        hit_old = damage_mob_at(
            state, level, old_row, old_col,
            damage_to_mob(vector, target->type_id[mob_slot], mob_class),
            false, true
        );
    }

    Damage second = vector;
    if (hit_old) {
        second.physical = 0.0f;
        second.fire = 0.0f;
        second.ice = 0.0f;
    }
    bool hit_new = false;
    if (find_mob_at(state, level, proposed_row, proposed_col, &mob_class, &mob_slot)) {
        Mobs* target = mobs_for_class(state, level, mob_class);
        hit_new = damage_mob_at(
            state, level, proposed_row, proposed_col,
            damage_to_mob(second, target->type_id[mob_slot], mob_class),
            false, true
        );
    }

    bool proposed_in_bounds = proposed_row >= 0 && proposed_row < MAP_SIZE
        && proposed_col >= 0 && proposed_col < MAP_SIZE;
    int proposed_block = proposed_in_bounds ? state->map[level][proposed_row][proposed_col] : 0;
    bool in_wall = is_solid_block(proposed_block) && proposed_block != BLOCK_WATER;
    bool keep = proposed_in_bounds && !in_wall && !hit_old && !hit_new && alive;
    projectiles->position[slot][0] = proposed_row;
    projectiles->position[slot][1] = proposed_col;
    projectiles->mask[slot] = keep;
}

void update_projectile_set(State* state, bool from_player) {
    int level = state->player_level;
    for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
        if (from_player) {
            move_player_projectile_slot(state, level, i);
        } else {
            move_mob_projectile_slot(state, level, i);
        }
    }
}

int floor_mob_type(int level, int mob_class) {
    static const int types[NUM_LEVELS][3] = {
        {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, {2, 3, 3}, {2, 4, 4},
        {1, 5, 5}, {1, 6, 6}, {1, 7, 7}, {0, 0, 0},
    };
    return types[clampi(level, 0, NUM_LEVELS - 1)][clampi(mob_class, 0, 2)];
}

int pick_kth(int count, Rng key) {
    float draw = (float)count * (1.0f - rng_f32(key));
    float cumulative = 0.0f;
    for (int k = 0; k < count; k++) {
        cumulative += 1.0f;
        if (cumulative >= draw) {
            return k;
        }
    }
    return count - 1;
}

int collect_spawn_cells(
    const State* state,
    int level,
    int min_exclusive,
    int max_exclusive,
    bool boss,
    bool water_only,
    int* rows,
    int* cols
) {
    const uint64_t* terrain = boss
        ? state->spawn_grave[level]
        : (water_only ? state->spawn_water[level] : state->spawn_land[level]);

    int pr = state->player_position[0];
    int pc = state->player_position[1];
    int limit = MOB_DESPAWN_DISTANCE - 1;
    int count = 0;
    for (int dr = -limit; dr <= limit; dr++) {
        int row = pr + dr;
        if ((unsigned)row >= MAP_SIZE) {
            continue;
        }
        uint64_t bits = terrain[row] & ~state->mob_bits[level][row];
        if (!bits) {
            continue;
        }
        for (int dc = -limit; dc <= limit; dc++) {
            int distance2 = dr * dr + dc * dc;
            if (distance2 <= min_exclusive || distance2 >= max_exclusive) {
                continue;
            }
            int col = pc + dc;
            if ((unsigned)col >= MAP_SIZE) {
                continue;
            }
            if (bits & (1ull << col)) {
                rows[count] = row;
                cols[count] = col;
                count++;
            }
        }
    }
    return count;
}

bool pick_spawn_cell(
    const int* rows,
    const int* cols,
    int count,
    Rng key,
    int* out_row,
    int* out_col
) {
    if (count <= 0) {
        return false;
    }
    int chosen = pick_kth(count, key);
    *out_row = rows[chosen];
    *out_col = cols[chosen];
    return true;
}

void spawn_into_slot(
    State* state,
    int level,
    Mobs* mobs,
    int slot,
    int mob_class,
    int type_id,
    int row,
    int col
) {
    mobs->position[slot][0] = row;
    mobs->position[slot][1] = col;
    mobs->health[slot] = mob_base_health(mob_class, type_id);
    mobs->mask[slot] = true;
    set_mob_bit(state, level, row, col, true);
}

void count_and_empty(const Mobs* mobs, int slots, int* count, int* empty) {
    int n = 0;
    int first = 0;
    bool found = false;
    for (int i = 0; i < slots; i++) {
        n += mobs->mask[i] ? 1 : 0;
        if (!mobs->mask[i] && !found) {
            first = i;
            found = true;
        }
    }
    *count = n;
    *empty = first;
}

void spawn_mobs(State* state, Rng rng) {
    int level = state->player_level;
    bool boss = fighting_boss(state);
    int coeff = 1 + (state->monsters_killed[level] < MONSTERS_KILLED_TO_CLEAR_LEVEL ? 2 : 0);
    if (boss) {
        coeff *= (state->boss_timestep_to_spawn_this_round >= 1) ? 1000 : 0;
    }

    static const float chances[NUM_LEVELS][4] = {
        {0.1f, 0.02f, 0.05f, 0.1f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.0f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
    };

    int passive_count;
    int passive_slot;
    count_and_empty(&state->passive_mobs[level], MAX_PASSIVE_MOBS, &passive_count, &passive_slot);
    Rng passive_prob = rng_key(&rng);
    Rng passive_pos = rng_key(&rng);
    int passive_type = floor_mob_type(level, MOB_PASSIVE);
    state->passive_mobs[level].type_id[passive_slot] = passive_type;

    int melee_count;
    int melee_slot;
    count_and_empty(&state->melee_mobs[level], MAX_MELEE_MOBS, &melee_count, &melee_slot);
    int melee_type = floor_mob_type(boss ? state->boss_progress : level, MOB_MELEE);
    Rng melee_prob = rng_key(&rng);
    float night = 1.0f - state->light_level;
    float melee_chance = chances[level][1] + chances[level][3] * night * night;
    Rng melee_pos = rng_key(&rng);
    state->melee_mobs[level].type_id[melee_slot] = melee_type;

    int ranged_count;
    int ranged_slot;
    count_and_empty(&state->ranged_mobs[level], MAX_RANGED_MOBS, &ranged_count, &ranged_slot);
    int ranged_type = floor_mob_type(boss ? state->boss_progress : level, MOB_RANGED);
    Rng ranged_prob = rng_key(&rng);
    Rng ranged_pos = rng_key(&rng);
    state->ranged_mobs[level].type_id[ranged_slot] = ranged_type;

    bool try_passive = !boss && passive_count < MAX_PASSIVE_MOBS
        && rng_f32(passive_prob) < chances[level][0];
    bool try_melee = melee_count < MAX_MELEE_MOBS
        && rng_f32(melee_prob) < melee_chance * (float)coeff;
    bool try_ranged = ranged_count < MAX_RANGED_MOBS
        && rng_f32(ranged_prob) < chances[level][2] * (float)coeff;
    if (!try_passive && !try_melee && !try_ranged) {
        return;
    }

    int min_hostile = boss ? -1 : 81;
    int max_hostile = boss ? 37 : MOB_DESPAWN_DISTANCE * MOB_DESPAWN_DISTANCE;
    int spawn_rows[729];
    int spawn_cols[729];
    int row;
    int col;
    if (try_passive) {
        int n = collect_spawn_cells(
            state, level, 9, MOB_DESPAWN_DISTANCE * MOB_DESPAWN_DISTANCE,
            false, false, spawn_rows, spawn_cols);
        if (pick_spawn_cell(spawn_rows, spawn_cols, n, passive_pos, &row, &col)) {
            spawn_into_slot(
                state, level, &state->passive_mobs[level],
                passive_slot, MOB_PASSIVE, passive_type, row, col);
        }
    }
    if (try_melee) {
        int n = collect_spawn_cells(
            state, level, min_hostile, max_hostile,
            boss, false, spawn_rows, spawn_cols);
        if (pick_spawn_cell(spawn_rows, spawn_cols, n, melee_pos, &row, &col)) {
            spawn_into_slot(
                state, level, &state->melee_mobs[level],
                melee_slot, MOB_MELEE, melee_type, row, col);
        }
    }
    if (try_ranged) {
        int n = collect_spawn_cells(
            state, level, min_hostile, max_hostile,
            boss, ranged_type == 5, spawn_rows, spawn_cols);
        if (pick_spawn_cell(spawn_rows, spawn_cols, n, ranged_pos, &row, &col)) {
            spawn_into_slot(
                state, level, &state->ranged_mobs[level],
                ranged_slot, MOB_RANGED, ranged_type, row, col);
        }
    }
}

void choose_direction(Rng key, int count, int direction[2]) {
    int choice = randint_at(key, 0u, 0, count);
    direction[0] = 0;
    direction[1] = 0;
    if (choice == 0) {
        direction[1] = -1;
    } else if (choice == 1) {
        direction[1] = 1;
    } else if (choice == 2) {
        direction[0] = -1;
    } else if (choice == 3) {
        direction[0] = 1;
    }
}

int choose_player_axis(Rng key, int distance_row, int distance_col) {
    int total = distance_row + distance_col;
    if (total == 0) {
        return 1;
    }
    int maximum = distance_row > distance_col ? distance_row : distance_col;
    float weights[2] = {
        distance_row == maximum ? 1.0f / (float)total : 0.0f,
        distance_col == maximum ? 1.0f / (float)total : 0.0f,
    };
    float sum = weights[0] + weights[1];
    float draw = sum * (1.0f - rng_f32(key));
    return (weights[0] >= draw || sum == 0.0f) ? 0 : 1;
}

int signi(int value) {
    if (value < 0) {
        return -1;
    }
    return value > 0 ? 1 : 0;
}

void move_melee_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->melee_mobs[level];
    bool alive = mobs->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int cooldown = mobs->attack_cooldown[slot];

    int random_dir[2];
    choose_direction(rng_key(rng), 4, random_dir);
    int distance_row = abs(state->player_position[0] - old_row);
    int distance_col = abs(state->player_position[1] - old_col);
    int axis = choose_player_axis(rng_key(rng), distance_row, distance_col);
    int player_dir[2] = {0, 0};
    if (axis == 0) {
        player_dir[0] = signi(state->player_position[0] - old_row);
    } else {
        player_dir[1] = signi(state->player_position[1] - old_col);
    }
    int dist = distance_row + distance_col;
    float chase_roll = rng_f32(rng_key(rng));
    bool chase = (dist < 10 || fighting_boss(state)) && chase_roll < 0.75f;
    int proposed_row = chase ? old_row + player_dir[0] : old_row + random_dir[0];
    int proposed_col = chase ? old_col + player_dir[1] : old_col + random_dir[1];
    bool attacking = dist == 1 && cooldown <= 0 && alive;
    if (attacking) {
        proposed_row = old_row;
        proposed_col = old_col;
        Damage damage = mob_damage_vector(type_id, MOB_MELEE);
        float sleep = 1.0f + 2.5f * (float)state->is_sleeping;
        damage.physical *= sleep;
        damage.fire *= sleep;
        damage.ice *= sleep;
        state->player_health -= damage_to_player(state, damage);
        state->achievements[ACH_WAKE_UP] = state->achievements[ACH_WAKE_UP] || state->is_sleeping;
        state->is_sleeping = false;
        state->is_resting = false;
    }
    int new_cooldown = attacking ? 5 : cooldown - 1;
    bool valid = valid_typed_mob_position(state, level, MOB_MELEE, type_id, proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    bool keep = alive && (dist < MOB_DESPAWN_DISTANCE || fighting_boss(state));
    Rng unused;
    rng_split(*rng, &unused, rng);

    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->attack_cooldown[slot] = new_cooldown;
    mobs->mask[slot] = keep;
}

void move_passive_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->passive_mobs[level];
    bool alive = mobs->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int direction[2];
    choose_direction(rng_key(rng), 8, direction);
    int proposed_row = old_row + direction[0];
    int proposed_col = old_col + direction[1];
    bool valid = valid_typed_mob_position(state, level, MOB_PASSIVE, type_id, proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    int dist = abs(state->player_position[0] - old_row) + abs(state->player_position[1] - old_col);
    bool keep = alive && dist < MOB_DESPAWN_DISTANCE;
    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->mask[slot] = keep;
}

void move_ranged_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->ranged_mobs[level];
    bool alive = mobs->mask[slot];
    if (!alive) {
        return;
    }
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int cooldown = mobs->attack_cooldown[slot];

    int random_dir[2];
    choose_direction(rng_key(rng), 4, random_dir);
    int distance_row = abs(state->player_position[0] - old_row);
    int distance_col = abs(state->player_position[1] - old_col);
    int axis = choose_player_axis(rng_key(rng), distance_row, distance_col);
    int player_dir[2] = {0, 0};
    if (axis == 0) {
        player_dir[0] = signi(state->player_position[0] - old_row);
    } else {
        player_dir[1] = signi(state->player_position[1] - old_col);
    }
    int dist = distance_row + distance_col;
    int proposed_row = dist >= 6 ? old_row + player_dir[0] : old_row + random_dir[0];
    int proposed_col = dist >= 6 ? old_col + player_dir[1] : old_col + random_dir[1];
    if (dist <= 3) {
        proposed_row = old_row - player_dir[0];
        proposed_col = old_col - player_dir[1];
    }
    if (!(rng_f32(rng_key(rng)) > 0.85f)) {
        proposed_row = old_row + random_dir[0];
        proposed_col = old_col + random_dir[1];
    }
    bool valid = valid_typed_mob_position(state, level, MOB_RANGED, type_id, proposed_row, proposed_col, old_row, old_col);
    bool attacking = ((dist >= 4 && dist <= 5) || (dist <= 3 && !valid)) && cooldown <= 0 && alive;
    if (attacking) {
        static const int projectile[8] = {
            PROJECTILE_ARROW, PROJECTILE_ARROW, PROJECTILE_FIREBALL, PROJECTILE_DAGGER,
            PROJECTILE_ARROW2, PROJECTILE_SLIMEBALL, PROJECTILE_FIREBALL2, PROJECTILE_ICEBALL2
        };
        spawn_projectile(state, false, projectile[clampi(type_id, 0, 7)], old_row, old_col, player_dir[0], player_dir[1]);
        proposed_row = old_row;
        proposed_col = old_col;
    }
    int new_cooldown = attacking ? 4 : cooldown - 1;
    valid = valid_typed_mob_position(state, level, MOB_RANGED, type_id, proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    bool keep = alive && (dist < MOB_DESPAWN_DISTANCE || fighting_boss(state));
    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->attack_cooldown[slot] = new_cooldown;
    mobs->mask[slot] = keep;
}

void update_mobs(State* state, Rng rng) {
    int level = state->player_level;
    rng_key(&rng);
    move_melee_slot(state, level, 0, &rng);
    move_melee_slot(state, level, 1, &rng);
    move_melee_slot(state, level, 2, &rng);
    rng_key(&rng);
    move_passive_slot(state, level, 0, &rng);
    move_passive_slot(state, level, 1, &rng);
    move_passive_slot(state, level, 2, &rng);
    rng_key(&rng);
    move_ranged_slot(state, level, 0, &rng);
    move_ranged_slot(state, level, 1, &rng);
    rng_key(&rng);
    update_projectile_set(state, false);
    rng_key(&rng);
    update_projectile_set(state, true);
}

void move_player(State* state, int action) {
    int direction[2];
    action_to_direction(action, direction);

    int proposed_row = state->player_position[0] + direction[0];
    int proposed_col = state->player_position[1] + direction[1];
    if (valid_player_position(state, proposed_row, proposed_col)) {
        state->player_position[0] = proposed_row;
        state->player_position[1] = proposed_col;
    }

    if (direction[0] != 0 || direction[1] != 0) {
        state->player_direction = action;
    }
}

int level_achievement(int level) {
    switch (level) {
    case 1: return ACH_ENTER_DUNGEON;
    case 2: return ACH_ENTER_GNOMISH_MINES;
    case 3: return ACH_ENTER_SEWERS;
    case 4: return ACH_ENTER_VAULT;
    case 5: return ACH_ENTER_TROLL_MINES;
    case 6: return ACH_ENTER_FIRE_REALM;
    case 7: return ACH_ENTER_ICE_REALM;
    case 8: return ACH_ENTER_GRAVEYARD;
    default: return -1;
    }
}

void change_floor(State* state, int action) {
    int level = state->player_level;
    int row = clampi(state->player_position[0], 0, MAP_SIZE - 1);
    int col = clampi(state->player_position[1], 0, MAP_SIZE - 1);

    bool on_down_ladder = state->item_map[level][row][col] == ITEM_LADDER_DOWN;
    bool can_move_down = action == ACTION_DESCEND
        && on_down_ladder
        && state->monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
        && level < NUM_LEVELS - 1;

    bool on_up_ladder = state->item_map[level][row][col] == ITEM_LADDER_UP;
    bool can_move_up = action == ACTION_ASCEND
        && on_up_ladder
        && level > 0;

    if (!can_move_down && !can_move_up) {
        return;
    }

    int new_level = level + (can_move_down ? 1 : -1);
    if (can_move_down) {
        state->player_position[0] = state->up_ladders[new_level][0];
        state->player_position[1] = state->up_ladders[new_level][1];
    } else {
        state->player_position[0] = state->down_ladders[new_level][0];
        state->player_position[1] = state->down_ladders[new_level][1];
    }

    state->player_level = new_level;
    int achievement = level_achievement(new_level);
    if (achievement >= 0 && !state->achievements[achievement]) {
        state->achievements[achievement] = 1;
        state->player_xp += 1;
    }
}

void level_up_attributes(State* state, int action) {
    if (state->player_xp < 1) {
        return;
    }

    bool leveled = false;
    if (action == ACTION_LEVEL_UP_DEXTERITY && state->player_dexterity < MAX_ATTRIBUTE) {
        state->player_dexterity += 1;
        leveled = true;
    } else if (action == ACTION_LEVEL_UP_STRENGTH && state->player_strength < MAX_ATTRIBUTE) {
        state->player_strength += 1;
        leveled = true;
    } else if (action == ACTION_LEVEL_UP_INTELLIGENCE && state->player_intelligence < MAX_ATTRIBUTE) {
        state->player_intelligence += 1;
        leveled = true;
    }

    if (leveled) {
        state->player_xp -= 1;
    }
}

bool near_block(const State* state, int block_type) {
    static const int offsets[8][2] = {
        {0, -1}, {0, 1}, {-1, 0}, {1, 0},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
    };
    int level = state->player_level;
    for (int i = 0; i < 8; i++) {
        int row = state->player_position[0] + offsets[i][0];
        int col = state->player_position[1] + offsets[i][1];
        if (row >= 0 && row < MAP_SIZE && col >= 0 && col < MAP_SIZE
            && state->map[level][row][col] == block_type) {
            return true;
        }
    }
    return false;
}

int first_armour_below(const Inventory* inventory, int threshold, int* count) {
    int first = 0;
    *count = 0;
    for (int i = 0; i < 4; i++) {
        bool below = inventory->armour[i] < threshold;
        if (*count == 0 && below) {
            first = i;
        }
        *count += below ? 1 : 0;
    }
    return first;
}

void craft_tools_and_items(State* state, int action) {
    bool at_table = near_block(state, BLOCK_CRAFTING_TABLE);
    bool at_furnace = near_block(state, BLOCK_FURNACE);
    Inventory* inv = &state->inventory;

    if (action == ACTION_MAKE_WOOD_PICKAXE && at_table && inv->wood >= 1 && inv->pickaxe < 1) {
        inv->wood -= 1;
        inv->pickaxe = 1;
    } else if (action == ACTION_MAKE_STONE_PICKAXE && at_table && inv->wood >= 1 && inv->stone >= 1 && inv->pickaxe < 2) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->pickaxe = 2;
    } else if (action == ACTION_MAKE_IRON_PICKAXE && at_table && at_furnace && inv->wood >= 1 && inv->stone >= 1 && inv->iron >= 1 && inv->coal >= 1 && inv->pickaxe < 3) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->iron -= 1;
        inv->coal -= 1;
        inv->pickaxe = 3;
    } else if (action == ACTION_MAKE_DIAMOND_PICKAXE && at_table && inv->wood >= 1 && inv->diamond >= 3 && inv->pickaxe < 4) {
        inv->wood -= 1;
        inv->diamond -= 3;
        inv->pickaxe = 4;
    } else if (action == ACTION_MAKE_WOOD_SWORD && at_table && inv->wood >= 1 && inv->sword < 1) {
        inv->wood -= 1;
        inv->sword = 1;
    } else if (action == ACTION_MAKE_STONE_SWORD && at_table && inv->wood >= 1 && inv->stone >= 1 && inv->sword < 2) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->sword = 2;
    } else if (action == ACTION_MAKE_IRON_SWORD && at_table && at_furnace && inv->wood >= 1 && inv->stone >= 1 && inv->iron >= 1 && inv->coal >= 1 && inv->sword < 3) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->iron -= 1;
        inv->coal -= 1;
        inv->sword = 3;
    } else if (action == ACTION_MAKE_DIAMOND_SWORD && at_table && inv->wood >= 1 && inv->diamond >= 2 && inv->sword < 4) {
        inv->wood -= 1;
        inv->diamond -= 2;
        inv->sword = 4;
    } else if (action == ACTION_MAKE_ARROW && at_table && inv->wood >= 1 && inv->stone >= 1 && inv->arrows < 99) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->arrows += 2;
    } else if (action == ACTION_MAKE_TORCH && at_table && inv->wood >= 1 && inv->coal >= 1 && inv->torches < 99) {
        inv->wood -= 1;
        inv->coal -= 1;
        inv->torches += 4;
    }

    int count = 0;
    int idx = first_armour_below(inv, 1, &count);
    if (action == ACTION_MAKE_IRON_ARMOUR && at_table && at_furnace && count > 0 && inv->iron >= 3 && inv->coal >= 3) {
        inv->iron -= 3;
        inv->coal -= 3;
        inv->armour[idx] = 1;
        state->achievements[ACH_MAKE_IRON_ARMOUR] = 1;
    }

    idx = first_armour_below(inv, 2, &count);
    if (action == ACTION_MAKE_DIAMOND_ARMOUR && at_table && count > 0 && inv->diamond >= 3) {
        inv->diamond -= 3;
        inv->armour[idx] = 2;
        state->achievements[ACH_MAKE_DIAMOND_ARMOUR] = 1;
    }
}

bool can_place_item_on(int block) {
    return block == BLOCK_GRASS || block == BLOCK_SAND || block == BLOCK_PATH
        || block == BLOCK_FIRE_GRASS || block == BLOCK_ICE_GRASS;
}

void add_growing_plant(State* state, int row, int col) {
    for (int i = 0; i < MAX_GROWING_PLANTS; i++) {
        if (!state->growing_plants_mask[i]) {
            state->growing_plants_positions[i][0] = row;
            state->growing_plants_positions[i][1] = col;
            state->growing_plants_age[i] = 0;
            state->growing_plants_mask[i] = 1;
            return;
        }
    }
}

void place_block(State* state, int action) {
    int direction[2];
    action_to_direction(state->player_direction, direction);
    int row = state->player_position[0] + direction[0];
    int col = state->player_position[1] + direction[1];
    if (row < 0 || row >= MAP_SIZE || col < 0 || col >= MAP_SIZE) {
        return;
    }

    int level = state->player_level;
    int block = state->map[level][row][col];
    bool occupied = is_solid_block(block) || state->item_map[level][row][col] != ITEM_NONE
        || mob_at(state, level, row, col);
    Inventory* inv = &state->inventory;

    if (action == ACTION_PLACE_TABLE && !occupied && inv->wood >= 2) {
        set_block(state, level, row, col, BLOCK_CRAFTING_TABLE);
        inv->wood -= 2;
        state->achievements[ACH_PLACE_TABLE] = 1;
    } else if (action == ACTION_PLACE_FURNACE && !occupied && inv->stone >= 1) {
        set_block(state, level, row, col, BLOCK_FURNACE);
        inv->stone -= 1;
        state->achievements[ACH_PLACE_FURNACE] = 1;
    } else if (action == ACTION_PLACE_STONE && (block == BLOCK_WATER || !occupied) && inv->stone >= 1) {
        set_block(state, level, row, col, BLOCK_STONE);
        inv->stone -= 1;
        state->achievements[ACH_PLACE_STONE] = 1;
    } else if (action == ACTION_PLACE_TORCH && can_place_item_on(block) && state->item_map[level][row][col] == ITEM_NONE && inv->torches >= 1) {
        state->item_map[level][row][col] = ITEM_TORCH;
        add_light(state, level, row, col);
        inv->torches -= 1;
        state->achievements[ACH_PLACE_TORCH] = 1;
    } else if (action == ACTION_PLACE_PLANT && block == BLOCK_GRASS && state->item_map[level][row][col] == ITEM_NONE && inv->sapling >= 1) {
        set_block(state, level, row, col, BLOCK_PLANT);
        inv->sapling -= 1;
        add_growing_plant(state, row, col);
        state->achievements[ACH_PLACE_PLANT] = 1;
    }
}

int choose_weighted_key(Rng key, const float* weights, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; i++) {
        total += weights[i];
    }
    float draw = total * (1.0f - rng_f32(key));
    float cumulative = 0.0f;
    for (int i = 0; i < count; i++) {
        cumulative += weights[i];
        if (cumulative >= draw) {
            return i;
        }
    }
    return count - 1;
}

void add_chest_loot(State* state, Rng rng) {
    Inventory* inv = &state->inventory;
    rng_key(&rng);
    (void)randint_at(rng_key(&rng), 0u, 1, 6);
    bool torch = rng_f32(rng_key(&rng)) < 0.6f;
    int torches = randint_at(rng_key(&rng), 0u, 4, 8);
    bool ore = rng_f32(rng_key(&rng)) < 0.6f;
    const float ore_weights[5] = {0.3f, 0.3f, 0.15f, 0.125f, 0.125f};
    int ore_id = choose_weighted_key(rng_key(&rng), ore_weights, 5);
    Rng amount_key = rng_key(&rng);
    int coal = randint_at(amount_key, 0u, 1, 4);
    int iron = randint_at(amount_key, 0u, 1, 3);
    int diamond = randint_at(amount_key, 0u, 1, 2);
    int sapphire = randint_at(amount_key, 0u, 1, 2);
    int ruby = randint_at(amount_key, 0u, 1, 2);
    bool potion = rng_f32(rng_key(&rng)) < 0.5f;
    int potion_id = randint_at(rng_key(&rng), 0u, 0, 6);
    int potion_amount = randint_at(rng_key(&rng), 0u, 1, 3);
    bool arrows = rng_f32(rng_key(&rng)) < 0.25f;
    int arrow_amount = randint_at(rng_key(&rng), 0u, 1, 5);
    bool tool = rng_f32(rng_key(&rng)) < 0.2f;
    int tool_id = randint_at(rng_key(&rng), 0u, 0, 2);
    const float tool_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    int pickaxe = choose_weighted_key(rng_key(&rng), tool_weights, 4) + 1;
    int sword = choose_weighted_key(rng_key(&rng), tool_weights, 4) + 1;
    int level = state->player_level;

    if (torch) inv->torches += torches;
    if (ore && ore_id == 0) inv->coal += coal;
    if (ore && ore_id == 1) inv->iron += iron;
    if (ore && ore_id == 2) inv->diamond += diamond;
    if (ore && ore_id == 3) inv->sapphire += sapphire;
    if (ore && ore_id == 4) inv->ruby += ruby;
    if (potion) inv->potions[potion_id] += potion_amount;
    if (arrows) inv->arrows += arrow_amount;
    if (tool && tool_id == 0 && pickaxe > inv->pickaxe) inv->pickaxe = pickaxe;
    if (tool && tool_id == 1 && sword > inv->sword) inv->sword = sword;
    if (state->player_level == 1 && !state->chests_opened[level]) inv->bow = 1;
    if (!state->chests_opened[level] && (state->player_level == 3 || state->player_level == 4)) {
        inv->books += 1;
    }
}

void interact_facing_tile(State* state, int action, Rng rng) {
    if (action != ACTION_DO) {
        return;
    }

    int direction[2];
    action_to_direction(state->player_direction, direction);
    int row = state->player_position[0] + direction[0];
    int col = state->player_position[1] + direction[1];
    bool in_bounds = row >= 0 && row < MAP_SIZE && col >= 0 && col < MAP_SIZE;
    int level = state->player_level;
    Inventory* inv = &state->inventory;

    bool did_attack = attack_mob_at(state, level, row, col, true);
    Rng sapling_key = rng_key(&rng);
    Rng chest_key = rng_key(&rng);
    if (did_attack || !in_bounds) {
        return;
    }
    int block = state->map[level][row][col];

    if (block == BLOCK_TREE || block == BLOCK_FIRE_TREE || block == BLOCK_ICE_SHRUB) {
        set_block(
            state,
            level,
            row,
            col,
            block == BLOCK_TREE ? BLOCK_GRASS :
                (block == BLOCK_FIRE_TREE ? BLOCK_FIRE_GRASS : BLOCK_ICE_GRASS)
        );
        inv->wood += 1;
    } else if (block == BLOCK_STONE && inv->pickaxe >= 1) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->stone += 1;
    } else if (block == BLOCK_COAL && inv->pickaxe >= 1) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->coal += 1;
    } else if (block == BLOCK_IRON && inv->pickaxe >= 2) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->iron += 1;
    } else if (block == BLOCK_DIAMOND && inv->pickaxe >= 3) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->diamond += 1;
    } else if (block == BLOCK_SAPPHIRE && inv->pickaxe >= 4) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->sapphire += 1;
    } else if (block == BLOCK_RUBY && inv->pickaxe >= 4) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->ruby += 1;
    } else if (block == BLOCK_STALAGMITE && inv->pickaxe >= 1) {
        set_block(state, level, row, col, BLOCK_PATH);
        inv->stone += 1;
    } else if (block == BLOCK_CRAFTING_TABLE || block == BLOCK_FURNACE) {
        set_block(state, level, row, col, BLOCK_PATH);
    } else if (block == BLOCK_WATER || block == BLOCK_FOUNTAIN) {
        state->player_drink = clampi(state->player_drink + 1, 0, max_drink(state));
        state->player_thirst = 0.0f;
        state->achievements[ACH_COLLECT_DRINK] = 1;
    } else if (block == BLOCK_RIPE_PLANT) {
        set_block(state, level, row, col, BLOCK_PLANT);
        for (int i = 0; i < MAX_GROWING_PLANTS; i++) {
            if (state->growing_plants_positions[i][0] == row
                && state->growing_plants_positions[i][1] == col) {
                state->growing_plants_age[i] = 0;
                break;
            }
        }
        state->player_food = clampi(state->player_food + 4, 0, max_food(state));
        state->player_hunger = 0.0f;
        state->achievements[ACH_EAT_PLANT] = 1;
    } else if (block == BLOCK_CHEST) {
        set_block(state, level, row, col, BLOCK_PATH);
        add_chest_loot(state, chest_key);
        state->achievements[ACH_OPEN_CHEST] = 1;
    } else if (block == BLOCK_NECROMANCER && boss_vulnerable(state) && fighting_boss(state)) {
        state->boss_progress += 1;
        state->boss_timestep_to_spawn_this_round = BOSS_SPAWN_TURNS;
        state->achievements[ACH_DAMAGE_NECROMANCER] = 1;
    }
    if (block == BLOCK_GRASS && rng_f32(sapling_key) < 0.1f) inv->sapling += 1;
    state->chests_opened[level] |= block == BLOCK_CHEST;
}

void drink_potion(State* state, int action) {
    int potion = action - ACTION_DRINK_POTION_RED;
    if (potion < 0 || potion >= NUM_POTIONS || state->inventory.potions[potion] <= 0) {
        return;
    }
    int effect = state->potion_mapping[potion];
    state->inventory.potions[potion] -= 1;
    if (effect == 0) state->player_health += 8.0f;
    else if (effect == 1) state->player_health -= 3.0f;
    else if (effect == 2) state->player_mana += 8;
    else if (effect == 3) state->player_mana -= 3;
    else if (effect == 4) state->player_energy += 8;
    else state->player_energy -= 3;
    state->achievements[ACH_DRINK_POTION] = 1;
}

void read_book(State* state, int action, Rng rng) {
    bool reading = action == ACTION_READ_BOOK && state->inventory.books > 0;
    Rng unused;
    Rng choice_key;
    rng_split(rng, &unused, &choice_key);
    float p0 = state->learned_spells[0] ? 0.0f : 1.0f;
    float p1 = state->learned_spells[1] ? 0.0f : 1.0f;
    int spell = 0;
    if (p0 + p1 != 0.0f) {
        float r = 1.0f - rng_f32(choice_key);
        spell = r <= (p0 / (p0 + p1)) ? 0 : 1;
    }
    if (reading) {
        state->inventory.books -= 1;
        state->learned_spells[spell] = 1;
        state->achievements[spell == 0 ? ACH_LEARN_FIREBALL : ACH_LEARN_ICEBALL] = 1;
    }
}

void enchant_items(State* state, int action, Rng rng) {
    int direction[2];
    action_to_direction(state->player_direction, direction);
    int level = state->player_level;
    int row = state->player_position[0] + direction[0];
    int col = state->player_position[1] + direction[1];
    int block = 0;
    if (row >= 0 && row < MAP_SIZE && col >= 0 && col < MAP_SIZE) {
        block = state->map[level][row][col];
    }
    int enchant = block == BLOCK_ENCHANTMENT_TABLE_FIRE ? 1 :
        (block == BLOCK_ENCHANTMENT_TABLE_ICE ? 2 : 0);
    int gems = enchant == 1 ? state->inventory.ruby : state->inventory.sapphire;
    bool could = state->player_mana >= 9 && enchant != 0 && gems >= 1;
    bool enchanting_sword = could && action == ACTION_ENCHANT_SWORD && state->inventory.sword > 0;
    bool enchanting_bow = could && action == ACTION_ENCHANT_BOW && state->inventory.bow > 0;
    int armour_count = 0;
    for (int i = 0; i < 4; i++) {
        armour_count += state->inventory.armour[i];
    }
    bool enchanting_armour = could && action == ACTION_ENCHANT_ARMOUR && armour_count > 0;
    Rng armour_key = rng_key(&rng);
    int unenchanted = 0;
    for (int i = 0; i < 4; i++) {
        unenchanted += state->armour_enchantments[i] == 0;
    }
    float candidates[4];
    for (int i = 0; i < 4; i++) {
        bool opposite = state->armour_enchantments[i] != 0 && state->armour_enchantments[i] != enchant;
        candidates[i] = (state->armour_enchantments[i] == 0 || (unenchanted == 0 && opposite)) ? 1.0f : 0.0f;
    }
    int armour_target = choose_weighted_key(armour_key, candidates, 4);
    if (enchanting_sword) {
        state->sword_enchantment = enchant;
        state->achievements[ACH_ENCHANT_SWORD] = 1;
    }
    if (enchanting_bow) {
        state->bow_enchantment = enchant;
    }
    if (enchanting_armour) {
        state->armour_enchantments[armour_target] = enchant;
        state->achievements[ACH_ENCHANT_ARMOUR] = 1;
    }
    bool enchanting = enchanting_sword || enchanting_bow || enchanting_armour;
    if (enchanting) {
        if (enchant == 1) {
            state->inventory.ruby -= 1;
        } else {
            state->inventory.sapphire -= 1;
        }
        state->player_mana -= 9;
    }
}

void use_projectile_or_spell(State* state, int action) {
    int direction[2];
    action_to_direction(state->player_direction, direction);
    if (direction[0] == 0 && direction[1] == 0) {
        direction[0] = 1;
    }

    if (action == ACTION_SHOOT_ARROW && state->inventory.bow > 0 && state->inventory.arrows > 0) {
        bool fired = spawn_projectile(
            state,
            true,
            PROJECTILE_ARROW2,
            state->player_position[0],
            state->player_position[1],
            direction[0],
            direction[1]
        );
        if (fired) {
            state->inventory.arrows -= 1;
            state->achievements[ACH_FIRE_BOW] = 1;
        }
    } else if (action == ACTION_CAST_FIREBALL && state->learned_spells[0] && state->player_mana >= 2) {
        bool cast = spawn_projectile(
            state,
            true,
            PROJECTILE_FIREBALL,
            state->player_position[0],
            state->player_position[1],
            direction[0],
            direction[1]
        );
        if (cast) {
            state->player_mana -= 2;
            state->achievements[ACH_CAST_FIREBALL] = 1;
        }
    } else if (action == ACTION_CAST_ICEBALL && state->learned_spells[1] && state->player_mana >= 2) {
        bool cast = spawn_projectile(
            state,
            true,
            PROJECTILE_ICEBALL,
            state->player_position[0],
            state->player_position[1],
            direction[0],
            direction[1]
        );
        if (cast) {
            state->player_mana -= 2;
            state->achievements[ACH_CAST_ICEBALL] = 1;
        }
    }
}

void update_plants(State* state) {
    for (int plant = 0; plant < MAX_GROWING_PLANTS; plant++) {
        if (!state->growing_plants_mask[plant]) {
            continue;
        }

        state->growing_plants_age[plant] += 1;
        if (state->growing_plants_age[plant] < 600) {
            continue;
        }

        int row = state->growing_plants_positions[plant][0];
        int col = state->growing_plants_positions[plant][1];
        if (state->growing_plants_age[plant] >= 600) {
            set_block(state, 0, row, col, BLOCK_RIPE_PLANT);
        }
    }
}

void update_intrinsics(State* state, int action) {
    bool start_sleep = action == ACTION_SLEEP && state->player_energy < max_energy(state);
    state->is_sleeping = state->is_sleeping || start_sleep;

    bool wake_from_sleep = state->is_sleeping && state->player_energy >= max_energy(state);
    state->is_sleeping = state->is_sleeping && !wake_from_sleep;
    state->achievements[ACH_WAKE_UP] = state->achievements[ACH_WAKE_UP] || wake_from_sleep;

    bool start_rest = action == ACTION_REST && state->player_health < (float)max_health(state);
    state->is_resting = state->is_resting || start_rest;

    bool wake_from_rest = state->is_resting && (
        state->player_health >= (float)max_health(state)
        || state->player_food <= 0
        || state->player_drink <= 0
    );
    state->is_resting = state->is_resting && !wake_from_rest;

    bool not_boss = !fighting_boss(state);
    float decay = 1.0f - 0.125f * (float)(state->player_dexterity - 1);

    state->player_hunger += (state->is_sleeping ? 0.5f : 1.0f) * decay;
    if (state->player_hunger > 25.0f) {
        state->player_hunger = 0.0f;
        state->player_food = clampi(state->player_food - (not_boss ? 1 : 0), 0, max_food(state));
    }

    state->player_thirst += (state->is_sleeping ? 0.5f : 1.0f) * decay;
    if (state->player_thirst > 20.0f) {
        state->player_thirst = 0.0f;
        state->player_drink = clampi(state->player_drink - (not_boss ? 1 : 0), 0, max_drink(state));
    }

    if (state->is_sleeping) {
        state->player_fatigue = state->player_fatigue - 1.0f;
        if (state->player_fatigue > 0.0f) {
            state->player_fatigue = 0.0f;
        }
    } else {
        state->player_fatigue += decay;
    }
    if (state->player_fatigue > 30.0f) {
        state->player_fatigue = 0.0f;
        state->player_energy = clampi(state->player_energy - (not_boss ? 1 : 0), 0, max_energy(state));
    } else if (state->player_fatigue < -10.0f) {
        state->player_fatigue = 0.0f;
        state->player_energy = clampi(state->player_energy + 1, 0, max_energy(state));
    }

    bool all_necessities = state->player_food > 0
        && state->player_drink > 0
        && (state->player_energy > 0 || state->is_sleeping);
    state->player_recover += all_necessities
        ? (state->is_sleeping ? 2.0f : 1.0f)
        : (state->is_sleeping ? -0.5f : -1.0f) * (not_boss ? 1.0f : 0.0f);

    if (state->player_recover > 25.0f) {
        state->player_recover = 0.0f;
        state->player_health = clampf(state->player_health + 1.0f, 0.0f, (float)max_health(state));
    } else if (state->player_recover < -15.0f) {
        state->player_recover = 0.0f;
        state->player_health -= 1.0f;
    }

    float mana_gain = state->is_sleeping ? 2.0f : 1.0f;
    float mana_coeff = 1.0f + 0.25f * (float)(state->player_intelligence - 1);
    state->player_recover_mana = (state->player_recover_mana + mana_gain) * mana_coeff;
    if (state->player_recover_mana > 30.0f) {
        state->player_recover_mana = 0.0f;
        state->player_mana = clampi(state->player_mana + 1, 0, max_mana(state));
    }
}

void clip_inventory_and_intrinsics(State* state) {
    state->inventory.wood = clampi(state->inventory.wood, 0, 99);
    state->inventory.stone = clampi(state->inventory.stone, 0, 99);
    state->inventory.coal = clampi(state->inventory.coal, 0, 99);
    state->inventory.iron = clampi(state->inventory.iron, 0, 99);
    state->inventory.diamond = clampi(state->inventory.diamond, 0, 99);
    state->inventory.sapling = clampi(state->inventory.sapling, 0, 99);
    state->inventory.pickaxe = clampi(state->inventory.pickaxe, 0, 99);
    state->inventory.sword = clampi(state->inventory.sword, 0, 99);
    state->inventory.bow = clampi(state->inventory.bow, 0, 99);
    state->inventory.arrows = clampi(state->inventory.arrows, 0, 99);
    state->inventory.torches = clampi(state->inventory.torches, 0, 99);
    state->inventory.ruby = clampi(state->inventory.ruby, 0, 99);
    state->inventory.sapphire = clampi(state->inventory.sapphire, 0, 99);
    state->inventory.books = clampi(state->inventory.books, 0, 99);
    for (int i = 0; i < 4; i++) {
        state->inventory.armour[i] = clampi(state->inventory.armour[i], 0, 99);
    }
    for (int i = 0; i < NUM_POTIONS; i++) {
        state->inventory.potions[i] = clampi(state->inventory.potions[i], 0, 99);
    }

    state->player_health = clampf(state->player_health, 0.0f, (float)max_health(state));
    state->player_food = clampi(state->player_food, 0, max_food(state));
    state->player_drink = clampi(state->player_drink, 0, max_drink(state));
    state->player_energy = clampi(state->player_energy, 0, max_energy(state));
    state->player_mana = clampi(state->player_mana, 0, max_mana(state));
}

void calculate_inventory_achievements(State* state) {
    state->achievements[ACH_COLLECT_WOOD] |= state->inventory.wood > 0;
    state->achievements[ACH_COLLECT_STONE] |= state->inventory.stone > 0;
    state->achievements[ACH_COLLECT_COAL] |= state->inventory.coal > 0;
    state->achievements[ACH_COLLECT_IRON] |= state->inventory.iron > 0;
    state->achievements[ACH_COLLECT_DIAMOND] |= state->inventory.diamond > 0;
    state->achievements[ACH_COLLECT_SAPPHIRE] |= state->inventory.sapphire > 0;
    state->achievements[ACH_COLLECT_RUBY] |= state->inventory.ruby > 0;
    state->achievements[ACH_COLLECT_SAPLING] |= state->inventory.sapling > 0;
    state->achievements[ACH_FIND_BOW] |= state->inventory.bow > 0;
    state->achievements[ACH_MAKE_ARROW] |= state->inventory.arrows > 0;
    state->achievements[ACH_MAKE_TORCH] |= state->inventory.torches > 0;
    state->achievements[ACH_MAKE_WOOD_PICKAXE] |= state->inventory.pickaxe >= 1;
    state->achievements[ACH_MAKE_STONE_PICKAXE] |= state->inventory.pickaxe >= 2;
    state->achievements[ACH_MAKE_IRON_PICKAXE] |= state->inventory.pickaxe >= 3;
    state->achievements[ACH_MAKE_DIAMOND_PICKAXE] |= state->inventory.pickaxe >= 4;
    state->achievements[ACH_MAKE_WOOD_SWORD] |= state->inventory.sword >= 1;
    state->achievements[ACH_MAKE_STONE_SWORD] |= state->inventory.sword >= 2;
    state->achievements[ACH_MAKE_IRON_SWORD] |= state->inventory.sword >= 3;
    state->achievements[ACH_MAKE_DIAMOND_SWORD] |= state->inventory.sword >= 4;
}

void update_boss_logic(State* state) {
    state->achievements[ACH_DEFEAT_NECROMANCER] |= has_beaten_boss(state);
    if (fighting_boss(state)) {
        state->boss_timestep_to_spawn_this_round -= 1;
    }
}

float calculate_light_level(int timestep) {
    float progress = fmodf((float)timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
    float cosine = cosf(3.14159265358979323846f * progress);
    return 1.0f - powf(fabsf(cosine), 3.0f);
}

bool is_game_over(const State* state) {
    return state->player_health <= 0.0f || state->timestep >= DEFAULT_MAX_TIMESTEPS;
}

void compute_observations(Craftax* env) {
    State* state = &env->state;
    float* obs = env->agents[0].observations;
    const int map_obs = OBS_ROWS * OBS_COLS * OBS_TILE_CHANNELS;
    memset(obs, 0, (size_t)map_obs * sizeof(float));

    int level = state->player_level;
    int row = state->player_position[0];
    int col = state->player_position[1];
    const int row_radius = OBS_ROWS / 2;
    const int col_radius = OBS_COLS / 2;

    for (int r = -row_radius; r <= row_radius; r++) {
        int obs_row = row + r;
        bool row_in = obs_row >= 0 && obs_row < MAP_SIZE;
        for (int c = -col_radius; c <= col_radius; c++) {
            int obs_col = col + c;
            int base = ((r + row_radius) * OBS_COLS + (c + col_radius)) * OBS_TILE_CHANNELS;
            if (row_in && obs_col >= 0 && obs_col < MAP_SIZE
                && state->light_map[level][obs_row][obs_col] > 12) {
                obs[base] = (float)state->map[level][obs_row][obs_col];
                obs[base + 1] = (float)(state->item_map[level][obs_row][obs_col] + 1);
                obs[base + 2] = 1.0f;
            }
        }
    }

    write_mob_obs(obs, state, &state->melee_mobs[level], MAX_MELEE_MOBS, 0);
    write_mob_obs(obs, state, &state->passive_mobs[level], MAX_PASSIVE_MOBS, 1);
    write_mob_obs(obs, state, &state->ranged_mobs[level], MAX_RANGED_MOBS, 2);
    write_mob_obs(obs, state, &state->mob_projectiles[level], MAX_MOB_PROJECTILES, 3);
    write_mob_obs(obs, state, &state->player_projectiles[level], MAX_PLAYER_PROJECTILES, 4);

    int obs_idx = map_obs;

    // Player inventory (normalized)
    obs[obs_idx++] = sqrtf((float)state->inventory.wood) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.stone) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.coal) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.iron) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.diamond) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.sapphire) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.ruby) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.sapling) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.torches) / 10.0f;
    obs[obs_idx++] = sqrtf((float)state->inventory.arrows) / 10.0f;
    obs[obs_idx++] = (float)state->inventory.books / 2.0f;
    obs[obs_idx++] = (float)state->inventory.pickaxe / 4.0f;
    obs[obs_idx++] = (float)state->inventory.sword / 4.0f;
    obs[obs_idx++] = (float)state->sword_enchantment;
    obs[obs_idx++] = (float)state->bow_enchantment;
    obs[obs_idx++] = (float)state->inventory.bow;
    for (int i = 0; i < NUM_POTIONS; i++) {
        obs[obs_idx++] = sqrtf((float)state->inventory.potions[i]) / 10.0f;
    }

    obs[obs_idx++] = state->player_health / 10.0f;
    obs[obs_idx++] = (float)state->player_food / 10.0f;
    obs[obs_idx++] = (float)state->player_drink / 10.0f;
    obs[obs_idx++] = (float)state->player_energy / 10.0f;
    obs[obs_idx++] = (float)state->player_mana / 10.0f;
    obs[obs_idx++] = (float)state->player_xp / 10.0f;
    obs[obs_idx++] = (float)state->player_dexterity / 10.0f;
    obs[obs_idx++] = (float)state->player_strength / 10.0f;
    obs[obs_idx++] = (float)state->player_intelligence / 10.0f;

    int direction_index = state->player_direction - ACTION_LEFT;
    for (int i = 0; i < 4; i++) {
        obs[obs_idx++] = i == direction_index ? 1.0f : 0.0f;
    }
    for (int i = 0; i < 4; i++) {
        obs[obs_idx++] = (float)state->inventory.armour[i] / 2.0f;
    }
    for (int i = 0; i < 4; i++) {
        obs[obs_idx++] = (float)state->armour_enchantments[i];
    }

    obs[obs_idx++] = state->light_level;
    obs[obs_idx++] = state->is_sleeping ? 1.0f : 0.0f;
    obs[obs_idx++] = state->is_resting ? 1.0f : 0.0f;
    obs[obs_idx++] = state->learned_spells[0] ? 1.0f : 0.0f;
    obs[obs_idx++] = state->learned_spells[1] ? 1.0f : 0.0f;
    obs[obs_idx++] = (float)state->player_level / 10.0f;
    obs[obs_idx++] = state->monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL ? 1.0f : 0.0f;
    obs[obs_idx++] = boss_vulnerable(state) ? 1.0f : 0.0f;

    if (obs_idx != OBS_SIZE) {
        fprintf(stderr, "craftax_clean: encoded %d values, expected %d\n", obs_idx, OBS_SIZE);
        abort();
    }
}

void update_log_state(Craftax* env) {
    if (env->state.player_level > env->max_floor_accum) {
        env->max_floor_accum = env->state.player_level;
    }
}

// Reset function
void reset_from_key(Craftax* env, Rng reset_key) {
    if (g_clean_reset_pool_size > 0) {
        uint32_t idx = reset_key.word[0] % (uint32_t)g_clean_reset_pool_size;
        memcpy(&env->state, &g_clean_reset_pool[idx], sizeof(State));
        return;
    }
    Rng unused;
    Rng world_key;
    rng_split(reset_key, &unused, &world_key);
    generate_world_from_key(&env->state, world_key);
}

void puf_reset(Craftax* env) {
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
        compute_observations(env);
        update_log_state(env);
        return;
    }
    Rng reset_key;
    rng_split(initial, &env->env_rng, &reset_key);
    reset_from_key(env, reset_key);

    compute_observations(env);
    update_log_state(env);
}

void puf_step(Craftax* env) {
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
    update_log_state(env);

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

static void craftax_clean_draw_mob_marker(State* state, int level, int row, int col, int x, int y) {
    int mob_class;
    int slot;
    if (!find_mob_at(state, level, row, col, &mob_class, &slot)) {
        return;
    }

    int tex_id = TEX_MOB_ZOMBIE;
    if (mob_class == MOB_PASSIVE) {
        tex_id = TEX_MOB_COW;
    } else if (mob_class == MOB_MELEE) {
        tex_id = TEX_MOB_ZOMBIE;
    } else if (mob_class == MOB_RANGED) {
        tex_id = TEX_MOB_SKELETON;
    }

    int type_id = mobs_for_class(state, level, mob_class)->type_id[slot];
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

void puf_render(Craftax* env) {
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
                if (mob_at(&env->state, level, wr, wc)) {
                    craftax_clean_draw_mob_marker(&env->state, level, wr, wc, dst_x, dst_y);
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

void puf_close(Craftax* env) {
    if (env->client == NULL) {
        return;
    }

    if (env->client->window_ready) {
        CloseWindow();
    }

    free(env->client);
    env->client = NULL;
}

