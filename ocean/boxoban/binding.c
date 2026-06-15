#define BOXOBAN_MAPS_IMPLEMENTATION //enables mmap
#include "boxoban.h"
#define OBS_SIZE BOXOBAN_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T FloatTensor


#define Env Boxoban
#define PUFFER_RESET_WITH_STATE
#define PUFFER_CURRICULUM_DIAG_SEQUENCE_OUTCOME
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->difficulty_id = (int)dict_get(kwargs, "difficulty")->value;
    env->size = 10;
    env->num_agents = 1;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    DictItem* num_levels = dict_get_unsafe(kwargs, "num_levels");
    env->num_levels = num_levels ? (int)num_levels->value : -1;
    DictItem* map_idx = dict_get_unsafe(kwargs, "map_idx");
    env->map_idx = map_idx ? (int)map_idx->value : -1;
    DictItem* map_sequence = dict_get_unsafe(kwargs, "map_sequence");
    if (map_sequence != NULL && map_sequence->ptr != NULL) {
        if (boxoban_set_map_sequence(env, (const char*)map_sequence->ptr) != 0) {
            abort();
        }
    }
    env->int_r_coeff = (float)dict_get(kwargs, "int_r_coeff")->value;
    env->curriculum_mode = (env->difficulty_id == BOXOBAN_DIFFICULTY_INCREMENTAL);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "targets_hit", log->on_targets);
    dict_set(out, "final_puzzle_tick", log->puzzle_ticks);
}
