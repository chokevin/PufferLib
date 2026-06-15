/* Pure C demo file for Boxoban. Usage:
 *   ./build.sh boxoban --local
 *   ./boxoban [difficulty|path_to_bin]
 *
 * If you pass one of the known difficulty names (basic, easy, medium,
 * hard, unfiltered) the demo prepares that difficulty bin. Otherwise the
 * argument is treated as an explicit path to a bin file.
 */

#define BOXOBAN_MAPS_IMPLEMENTATION
#include <time.h>
#include "boxoban.h"

static int is_named_difficulty(const char* arg) {
    return strcmp(arg, "basic") == 0 ||
        strcmp(arg, "easy") == 0 ||
        strcmp(arg, "medium") == 0 ||
        strcmp(arg, "hard") == 0 ||
        strcmp(arg, "unfiltered") == 0;
}

static const char* resolve_map_path(int argc, char** argv, char* buffer, size_t buf_sz) {
    const char* arg = argc > 1 ? argv[1] : NULL;
    if (arg == NULL) {
        if (boxoban_prepare_maps_for_difficulty("easy", buffer, buf_sz) != 0) {
            return NULL;
        }
        return buffer;
    }
    if (strchr(arg, '/')) {
        return arg;
    }
    if (is_named_difficulty(arg)) {
        if (boxoban_prepare_maps_for_difficulty(arg, buffer, buf_sz) != 0) {
            return NULL;
        }
        return buffer;
    }
    snprintf(buffer, buf_sz, "resources/boxoban/boxoban_maps_%s.bin", arg);
    return buffer;
}

static int setup_demo_env(Boxoban* env, const char* chosen_path) {
    memset(env, 0, sizeof(Boxoban));
    env->size = 10;
    env->num_agents = 1;
    env->max_steps = 500;
    env->num_levels = -1;
    env->map_idx = -1;
    env->int_r_coeff = 0.1f;
    env->difficulty_id = -1;

    env->observations = (float*)calloc(BOXOBAN_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    if (env->observations == NULL || env->actions == NULL ||
            env->rewards == NULL || env->terminals == NULL) {
        return -1;
    }

    if (boxoban_set_map_path(chosen_path) != 0) {
        fprintf(stderr, "Failed to set map path: %s\n", chosen_path);
        return -1;
    }

    init(env);
    c_reset(env, NULL);
    return 0;
}

static void free_demo_env(Boxoban* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

int demo(int argc, char** argv) {
    char path_buffer[512];
    const char* chosen_path = resolve_map_path(argc, argv, path_buffer, sizeof(path_buffer));
    if (chosen_path == NULL) {
        fprintf(stderr, "Failed to prepare map path\n");
        return 1;
    }

    Boxoban env;
    if (setup_demo_env(&env, chosen_path) != 0) {
        free_demo_env(&env);
        return 1;
    }

    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_LEFT_SHIFT) || IsKeyPressed(KEY_RIGHT_SHIFT)) {
            TraceLog(LOG_INFO, "Shift key pressed");
        }
        bool manual = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
        bool stepped = false;
        if (manual) {
            int new_action = -1;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) new_action = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) new_action = DOWN;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) new_action = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) new_action = RIGHT;

            if (new_action >= 0) {
                env.actions[0] = (float)new_action;
                c_step(&env);
                stepped = true;
            }
        } else {
            env.actions[0] = (float)(rand() % 5);
            c_step(&env);
            stepped = true;
        }

        if (!stepped) {
            // Manual mode with no direction: stay paused.
        }
        c_render(&env);
    }

    free_demo_env(&env);
    return 0;
}

void test_performance(int argc, char** argv, int timeout) {
    char path_buffer[512];
    const char* chosen_path = resolve_map_path(argc, argv, path_buffer, sizeof(path_buffer));
    if (chosen_path == NULL) {
        fprintf(stderr, "Failed to prepare map path\n");
        return;
    }
    printf("Loaded map: %s\n", chosen_path);

    Boxoban env;
    if (setup_demo_env(&env, chosen_path) != 0) {
        free_demo_env(&env);
        return;
    }

    printf("Starting test...\n");
    int start = (int)time(NULL);
    int num_steps = 0;
    while ((int)time(NULL) - start < timeout) {
        env.actions[0] = (float)(rand() % 5);
        c_step(&env);
        num_steps++;
    }

    int end = (int)time(NULL);
    float sps = (float)num_steps / (float)(end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_demo_env(&env);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    return demo(argc, argv);
}
