#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "raylib.h"
#include "boxoban_maps.h"

static const unsigned char NOOP = 0;
static const unsigned char DOWN = 1;
static const unsigned char UP = 2;
static const unsigned char LEFT = 3;
static const unsigned char RIGHT = 4;

static const unsigned char AGENT = 0;
static const unsigned char WALLS = 1;
static const unsigned char BOXES = 2;
static const unsigned char TARGET = 3;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float on_targets; // Fraction of targets currently boxed
    float puzzle_ticks; // Steps spent on the final puzzle of the episode
    float n; // Required as the last field
} Log;

typedef struct {
    Texture2D wall;
    Texture2D box;
    Texture2D target;
    Texture2D floor;
    Texture2D agent;
    Texture2D box_on_target;
} Client;

typedef struct State {
    int tick;
    int puzzle_tick;
    int agent_x;
    int agent_y;
    int on_target;
    int n_boxes;
    int n_targets;
    int win;
    int curriculum_difficulty;
    int largest_solved_difficulty;
    int episode_maps_solved;
    int sequence_pos;
    bool initialized;
    float episode_return;
    unsigned char observations[BOXOBAN_PUZZLE_OBS_BYTES];
} State;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. Exposed obs are floats; map state is byte-packed.
    float* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    float* terminals; // Required. We don't yet have truncations as standard yet
    unsigned int rng;
    int size;
    int num_agents;
    int max_steps;
    int num_levels; // -1 uses every puzzle in the selected map bin
    int map_idx; // -1 samples normally; >=0 always loads this puzzle index
    int map_sequence_len; // 0 disables ordered map progression
    int* map_sequence;
    float int_r_coeff;
    int difficulty_id; // 0=basic,1=easy,2=medium,3=hard,4=unfiltered,5=incremental
    int curriculum_mode; // 1 when using incremental difficulty mode
    Client* client;
    State state;
} Boxoban;

void ensure_map_loaded(void);

static bool boxoban_ends_with(const char* value, const char* suffix) {
    size_t value_len = strlen(value);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > value_len) {
        return false;
    }
    return strcmp(value + value_len - suffix_len, suffix) == 0;
}

static char* boxoban_read_text_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);

    char* text = (char*)malloc((size_t)size + 1);
    if (text == NULL) {
        fclose(f);
        return NULL;
    }
    size_t nread = fread(text, 1, (size_t)size, f);
    fclose(f);
    text[nread] = '\0';
    return text;
}

static int boxoban_parse_map_sequence_text(const char* text, int** out_maps) {
    int count = 0;
    int cap = 0;
    int* maps = NULL;
    const char* p = text;
    while (*p != '\0') {
        while (*p != '\0' && (*p < '0' || *p > '9')) {
            p++;
        }
        if (*p == '\0') {
            break;
        }

        char* end = NULL;
        long value = strtol(p, &end, 10);
        if (end == p || value < 0 || value > INT32_MAX) {
            free(maps);
            return -1;
        }
        if (count == cap) {
            int next_cap = cap == 0 ? 16 : cap * 2;
            int* next = (int*)realloc(maps, (size_t)next_cap * sizeof(int));
            if (next == NULL) {
                free(maps);
                return -1;
            }
            maps = next;
            cap = next_cap;
        }
        maps[count++] = (int)value;
        p = end;
    }

    *out_maps = maps;
    return count;
}

static void boxoban_free_map_sequence(Boxoban* env) {
    free(env->map_sequence);
    env->map_sequence = NULL;
    env->map_sequence_len = 0;
}

static int boxoban_set_map_sequence(Boxoban* env, const char* spec) {
    boxoban_free_map_sequence(env);
    if (spec == NULL || spec[0] == '\0' || strcmp(spec, "-") == 0) {
        return 0;
    }

    char* file_text = boxoban_read_text_file(spec);
    if (file_text == NULL && (strchr(spec, '/') != NULL ||
            boxoban_ends_with(spec, ".txt"))) {
        fprintf(stderr, "Failed to read Boxoban map_sequence file: %s\n", spec);
        return -1;
    }

    const char* text = file_text != NULL ? file_text : spec;
    int* maps = NULL;
    int count = boxoban_parse_map_sequence_text(text, &maps);
    free(file_text);
    if (count <= 0) {
        free(maps);
        fprintf(stderr, "Boxoban map_sequence must contain at least one map index\n");
        return -1;
    }

    env->map_sequence = maps;
    env->map_sequence_len = count;
    return 0;
}

static int boxoban_configure_maps_from_env(Boxoban* env) {
    env->curriculum_mode = (env->difficulty_id == BOXOBAN_DIFFICULTY_INCREMENTAL);
    if (env->curriculum_mode) {
        reset_incremental_map_cache();
        if (boxoban_load_incremental_bins() != 0) {
            fprintf(stderr, "Failed to load incremental Boxoban map bins\n");
            return -1;
        }
        return 0;
    }

    if (env->difficulty_id == -1) {
        return 0;
    }

    if (env->difficulty_id < -1) {
        fprintf(stderr, "Invalid Boxoban difficulty id %d\n", env->difficulty_id);
        return -1;
    }

    const char* difficulty_name = boxoban_difficulty_name_from_id(env->difficulty_id);
    if (difficulty_name == NULL) {
        fprintf(stderr, "Invalid Boxoban difficulty id %d\n", env->difficulty_id);
        return -1;
    }
    char prepared_path[512];
    if (boxoban_prepare_maps_for_difficulty(difficulty_name, prepared_path, sizeof(prepared_path)) != 0) {
        return -1;
    }

    return 0;
}

// Entity,x,y convention. y moves top to bottom.
static inline void set_entity(Boxoban* env, int entity, int x, int y, unsigned char value) {
    env->state.observations[(entity)*env->size*env->size + (y)*env->size + (x)] = value;
}

static inline unsigned char get_entity(Boxoban* env, int entity, int x, int y) {
    return env->state.observations[(entity)*env->size*env->size + (y)*env->size + (x)];
}

static inline uint32_t get_random_puzzle_idx(Boxoban* env, size_t puzzle_count) {
    int idx = (int)(rand_r(&env->rng) % (unsigned int)puzzle_count);
    return (uint32_t)idx;
}

void refresh_state(Boxoban* env) {
    if (env->observations != NULL) {
        for (size_t i = 0; i < PUZZLE_OBS_BYTES; i++) {
            env->observations[i] = (float)env->state.observations[i];
        }
        float remaining = 0.0f;
        if (env->max_steps > 0) {
            int remaining_ticks = env->max_steps - env->state.puzzle_tick;
            if (remaining_ticks < 0) {
                remaining_ticks = 0;
            } else if (remaining_ticks > env->max_steps) {
                remaining_ticks = env->max_steps;
            }
            remaining = (float)remaining_ticks / (float)env->max_steps;
        }
        env->observations[PUZZLE_OBS_BYTES] = remaining;
    }
}

void init(Boxoban* env) {
    static int boxoban_maps_ready = 0;
    static int configured_difficulty_id = -9999;

    if (env->size != BOXOBAN_EXPECTED_ROWS) {
        fprintf(stderr, "Boxoban size must be %d, got %d\n", BOXOBAN_EXPECTED_ROWS, env->size);
        abort();
    }

    env->curriculum_mode = (env->difficulty_id == BOXOBAN_DIFFICULTY_INCREMENTAL);
    if (env->curriculum_mode && env->map_sequence_len > 0) {
        fprintf(stderr, "Boxoban map_sequence is not supported with incremental difficulty\n");
        abort();
    }
    if (!boxoban_maps_ready || configured_difficulty_id != env->difficulty_id) {
        if (boxoban_configure_maps_from_env(env) != 0) {
            fprintf(stderr, "Failed to configure Boxoban maps\n");
            abort();
        }
        if (!env->curriculum_mode) {
            ensure_map_loaded();
        }
        boxoban_maps_ready = 1;
        configured_difficulty_id = env->difficulty_id;
    }

    memset(&env->state, 0, sizeof(State));
    env->state.largest_solved_difficulty = -1;
}

void add_log(Boxoban* env) {
    State* s = &env->state;
    float perf;
    float score;
    float targets_hit = 0.0f;
    if (s->n_targets > 0) {
        targets_hit = (float)s->on_target / (float)s->n_targets;
    }
    if (env->map_sequence_len > 0) {
        score = (float)s->episode_maps_solved;
        perf = score / (float)env->map_sequence_len;
    } else if (env->curriculum_mode) {
        score = 0.0f;
        if (s->largest_solved_difficulty >= 0) {
            score = (float)(s->largest_solved_difficulty + 1);
        }
        perf = (score + targets_hit) /
            (float)(BOXOBAN_INCREMENTAL_NUM_DIFFICULTIES + 1);
    } else {
        perf = (s->win == 1) ? 1.0f : 0.0f;
        score = perf;
    }

    env->log.perf += perf;
    env->log.score += score;
    env->log.episode_length += (float)s->tick;
    env->log.episode_return += s->episode_return;
    env->log.on_targets += targets_hit;
    env->log.puzzle_ticks += (float)s->puzzle_tick;
    env->log.n++;
}

bool clear(Boxoban* env, int x, int y) {
    if (x < 0 || y < 0 || x >= env->size || y >= env->size) {
        return false;
    }
    return (get_entity(env, WALLS, x, y) == 0) && (get_entity(env, BOXES, x, y) == 0);
}

static void load_random_puzzle(Boxoban* env) {
    State* s = &env->state;
    const uint8_t* map_base = MAP_BASE;
    size_t puzzle_count = PUZZLE_COUNT;
    if (env->curriculum_mode) {
        int slot = s->curriculum_difficulty;
        map_base = INCREMENTAL_MAP_BASES[slot];
        puzzle_count = INCREMENTAL_PUZZLE_COUNTS[slot];
    }
    uint32_t i;
    if (env->map_sequence_len > 0) {
        if (s->sequence_pos < 0 || s->sequence_pos >= env->map_sequence_len) {
            fprintf(stderr, "Boxoban sequence_pos %d out of range for sequence length %d\n",
                s->sequence_pos, env->map_sequence_len);
            abort();
        }
        i = (uint32_t)env->map_sequence[s->sequence_pos];
        if ((size_t)i >= puzzle_count) {
            fprintf(stderr, "Boxoban map_sequence index %u out of range for %zu puzzles\n",
                i, puzzle_count);
            abort();
        }
    } else if (env->map_idx >= 0) {
        if ((size_t)env->map_idx >= puzzle_count) {
            fprintf(stderr, "Boxoban map_idx %d out of range for %zu puzzles\n",
                env->map_idx, puzzle_count);
            abort();
        }
        i = (uint32_t)env->map_idx;
    } else {
        if (env->num_levels > 0 && (size_t)env->num_levels < puzzle_count) {
            puzzle_count = (size_t)env->num_levels;
        }
        i = get_random_puzzle_idx(env, puzzle_count);
    }

    const uint8_t* puzzle = map_base + (size_t)i * PUZZLE_SIZE;
    memcpy(s->observations, puzzle, PUZZLE_OBS_BYTES);

    const uint8_t* meta = puzzle + PUZZLE_OBS_BYTES;
    s->agent_x = (int)meta[0];
    s->agent_y = (int)meta[1];
    s->n_boxes = (int)meta[2];
    s->n_targets = (int)meta[3];
    s->on_target = (int)meta[4];

    s->puzzle_tick = 0;
}

// Required function
void c_reset(Boxoban* env, const State* state) {
    State* s = &env->state;
    if (state != NULL) {
        env->state = *state;
        refresh_state(env);
        return;
    }

    s->tick = 0;
    s->puzzle_tick = 0;
    s->win = 0;
    s->episode_return = 0;
    if (env->curriculum_mode) {
        s->curriculum_difficulty = 0;
        s->largest_solved_difficulty = -1;
        s->episode_maps_solved = 0;
    }
    if (env->map_sequence_len > 0) {
        s->sequence_pos = 0;
        s->episode_maps_solved = 0;
    }

    load_random_puzzle(env);

    if (!s->initialized) {
        s->tick = rand_r(&env->rng) % env->max_steps;
        s->puzzle_tick = s->tick;
        s->initialized = true;
    }
    refresh_state(env);
}

// Updates OBS for moved entity
void move_entity(Boxoban* env, unsigned char entity, int x, int y, int dx, int dy) {
    set_entity(env, entity, x, y, 0);
    set_entity(env, entity, x + dx, y + dy, 1);
}

// Updates state in place. Returns target transition count for the pushed box.
int take_action(Boxoban* env, int action) {
    State* s = &env->state;
    int dx = 0;
    int dy = 0;
    int target_delta = 0;

    if (action == NOOP) {
        return 0;
    } else if (action == DOWN) {
        dy = 1;
    } else if (action == UP) {
        dy = -1;
    } else if (action == LEFT) {
        dx = -1;
    } else if (action == RIGHT) {
        dx = 1;
    }

    // If move space is clear, move agent.
    if (clear(env, s->agent_x + dx, s->agent_y + dy)) {
        move_entity(env, AGENT, s->agent_x, s->agent_y, dx, dy);
        s->agent_y += dy;
        s->agent_x += dx;
        return 0;
    }

    int box_x = s->agent_x + dx;
    int box_y = s->agent_y + dy;
    int box_dest_x = box_x + dx;
    int box_dest_y = box_y + dy;

    // If its not clear, but its a box and box is clear to move, move both.
    if (clear(env, box_dest_x, box_dest_y) &&
            get_entity(env, BOXES, box_x, box_y) == 1) {
        if (get_entity(env, TARGET, box_x, box_y) == 1) {
            s->on_target -= 1;
            target_delta -= 1;
        }

        move_entity(env, BOXES, box_x, box_y, dx, dy);
        move_entity(env, AGENT, s->agent_x, s->agent_y, dx, dy);
        s->agent_y += dy;
        s->agent_x += dx;

        if (get_entity(env, TARGET, box_dest_x, box_dest_y) == 1) {
            s->on_target += 1;
            target_delta += 1;
        }
        return target_delta;
    }
    return 0;
}

// Required function
void c_step(Boxoban* env) {
    State* s = &env->state;
    s->tick += 1;
    s->puzzle_tick += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;

    int action = (int)env->actions[0];

    int target_delta = take_action(env, action);
    env->rewards[0] += (float)target_delta * env->int_r_coeff;

    // Terminals
    if (s->on_target == s->n_targets) {
        env->rewards[0] += 1.0f;
        s->win = 1;
        s->episode_return += env->rewards[0];
        if (env->curriculum_mode) {
            if (s->curriculum_difficulty > s->largest_solved_difficulty) {
                s->largest_solved_difficulty = s->curriculum_difficulty;
            }
            s->episode_maps_solved += 1;
            if (s->curriculum_difficulty < BOXOBAN_INCREMENTAL_MAX_DIFFICULTY) {
                s->curriculum_difficulty += 1;
            }
            load_random_puzzle(env);
            refresh_state(env);
            return;
        }
        if (env->map_sequence_len > 0) {
            s->episode_maps_solved += 1;
            if (s->sequence_pos + 1 < env->map_sequence_len) {
                s->sequence_pos += 1;
                s->win = 0;
                load_random_puzzle(env);
                refresh_state(env);
                return;
            }
        }
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env, NULL);
        return;
    }

    if (s->puzzle_tick >= env->max_steps) {
        env->terminals[0] = 1;
        //env->rewards[0] -= 1.0f;
        s->episode_return += env->rewards[0];
        add_log(env);
        c_reset(env, NULL);
        return;
    }

    s->episode_return += env->rewards[0];
    refresh_state(env);
}

Client* c_create(Boxoban* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->wall = LoadTexture("resources/boxoban/Wall_Black.jpg");
    client->box = LoadTexture("resources/boxoban/Crate_Black.jpg");
    client->target = LoadTexture("resources/boxoban/EndPoint_Black.jpg");
    client->floor = LoadTexture("resources/boxoban/GroundGravel_Concrete.jpg");
    client->box_on_target = LoadTexture("resources/boxoban/EndPoint_Blue.jpg");
    client->agent = LoadTexture("resources/shared/puffers_128.png");
    env->client = client;
    return client;
}

#define TILE 32

Texture2D choose_sprite(Client* c, Boxoban* env, int x, int y) {
    int a = get_entity(env, AGENT, x, y);
    int w = get_entity(env, WALLS, x, y);
    int b = get_entity(env, BOXES, x, y);
    int t = get_entity(env, TARGET, x, y);

    if (w) return c->wall;
    if (b && t) return c->box_on_target;
    if (b) return c->box;
    if (a) return c->agent;
    if (t) return c->target;

    return c->floor;
}

static inline Rectangle boxoban_texture_src(Texture2D tex) {
    Rectangle src = {0.0f, 0.0f, (float)tex.width, (float)tex.height};
    return src;
}

void draw_tile(Boxoban* env, int x, int y) {
    Client* c = env->client;
    Rectangle dest = {(float)(x * TILE), (float)(y * TILE), (float)TILE, (float)TILE};
    Vector2 origin = {0.0f, 0.0f};

    // Always lay down the base tile.
    DrawTexturePro(c->floor, boxoban_texture_src(c->floor), dest, origin, 0.0f, WHITE);

    if (get_entity(env, TARGET, x, y)) {
        DrawTexturePro(c->target, boxoban_texture_src(c->target), dest, origin, 0.0f, WHITE);
    }
    if (get_entity(env, BOXES, x, y)) {
        Texture2D tex = get_entity(env, TARGET, x, y) ? c->box_on_target : c->box;
        DrawTexturePro(tex, boxoban_texture_src(tex), dest, origin, 0.0f, WHITE);
    }
    if (get_entity(env, WALLS, x, y)) {
        DrawTexturePro(c->wall, boxoban_texture_src(c->wall), dest, origin, 0.0f, WHITE);
    }
    if (get_entity(env, AGENT, x, y)) {
        Rectangle src = {0.0f, 0.0f, c->agent.width / 2.0f, (float)c->agent.height};
        DrawTexturePro(c->agent, src, dest, origin, 0.0f, WHITE);
    }
}

// Required function. Should handle creating the client on first call
void c_render(Boxoban* env) {
    if (!IsWindowReady()) {
        InitWindow(TILE*env->size, TILE*env->size, "PufferLib Boxoban");
        SetTargetFPS(10);
    }

    // Standard across our envs so exiting is always the same.
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (env->client == NULL) {
        env->client = c_create(env);
    }

    BeginDrawing();
    Color background = {6, 24, 24, 255};
    ClearBackground(background);

    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            draw_tile(env, x, y);
        }
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated.
// Do not free env->observations, actions, rewards, terminals.
void c_close(Boxoban* env) {
    boxoban_free_map_sequence(env);
    if (IsWindowReady()) {
        if (env->client) {
            UnloadTexture(env->client->wall);
            UnloadTexture(env->client->box);
            UnloadTexture(env->client->target);
            UnloadTexture(env->client->floor);
            UnloadTexture(env->client->agent);
            UnloadTexture(env->client->box_on_target);
            free(env->client);
            env->client = NULL;
        }
        CloseWindow();
    }
}
