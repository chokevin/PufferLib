// Reward-only best-trajectory curriculum implementation.

#include <cub/device/device_scan.cuh>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PRIO_WARP_SIZE 32
#define PRIO_FULL_MASK 0xffffffff
#define PRIO_BLOCK_SIZE 256

#ifdef PUFFER_CURRICULUM_TYPES

struct PrioBuffers {
    FloatTensor prio_weights, cdf;
    ByteTensor cdf_temp;
    PrecisionTensor mb_prio;
    IntTensor idx, sample_done;
};

struct BestTrajectoryMeta {
    int len;
    int episode_len;
    float episode_return;
    long long generation;
};

struct ActiveTrajectoryRow {
    int best_slot;
    int sample_offset;
    int history_count;
    int segment_start;
    float episode_return;
    float prefix_return;
    int episode_len;
    int prefix_step;
    long long best_generation;
    int segment_count;
};

struct CompletedSegment {
    int best_slot;
    int sample_offset;
    int start;
    int count;
    float episode_return;
    float prefix_return;
    int episode_len;
    int prefix_step;
    long long best_generation;
};

struct StateBuffer {
    PufferState* best_states;
    PufferState* active_states;
    float* best_state_return;
    float* active_state_return;
    int* best_state_step;
    int* active_state_step;
    BestTrajectoryMeta* best_meta;
    ActiveTrajectoryRow* active_rows;
    CompletedSegment* segments;
    unsigned int* rng_states;
    unsigned int host_rng_state;
    unsigned int rng_seed;
    int max_active_envs;
    int max_segments_per_row;
    int rng_workers_per_buffer;
    int rng_state_count;
    int num_start_states;
    int trajectory_max_len;
    int checkpoint_interval;
    int num_vanilla_envs;
    int num_fresh_envs;
    int num_cl_envs;
};

void close_state_buffer(StateBuffer* buf);

void register_state_buffer(StateBuffer* buf,
        int num_envs, int max_active_envs,
        int num_start_states, int trajectory_max_len, int checkpoint_interval,
        int rollout_horizon, int num_buffers, int num_threads,
        unsigned int rng_seed) {
    int rng_workers_per_buffer = num_threads / num_buffers;
    if (rng_workers_per_buffer < 1) {
        fprintf(stderr,
            "Invalid curriculum thread config: num_threads=%d num_buffers=%d\n",
            num_threads, num_buffers);
        abort();
    }

    buf->max_active_envs = max_active_envs;
    buf->max_segments_per_row = rollout_horizon;
    buf->rng_workers_per_buffer = rng_workers_per_buffer;
    buf->rng_state_count = num_buffers * rng_workers_per_buffer;
    buf->rng_seed = rng_seed;
    buf->host_rng_state = rng_seed + (unsigned int)buf->rng_state_count;
    buf->num_start_states = num_start_states;
    buf->trajectory_max_len = trajectory_max_len;
    buf->checkpoint_interval = checkpoint_interval;
    buf->num_vanilla_envs = num_envs;
    buf->num_fresh_envs = 0;
    buf->num_cl_envs = 0;
}

void init_state_buffer(StateBuffer* buf) {
    size_t best_rows = (size_t)buf->num_start_states;
    size_t traj_len = (size_t)buf->trajectory_max_len;
    size_t active_rows = (size_t)buf->max_active_envs;
    size_t segment_entries = active_rows * (size_t)buf->max_segments_per_row;
    size_t best_entries = best_rows * traj_len;
    size_t active_entries = active_rows * traj_len;
    size_t rng_entries = (size_t)buf->rng_state_count;

    buf->best_states = (PufferState*)calloc(best_entries, sizeof(PufferState));
    buf->active_states = (PufferState*)calloc(active_entries, sizeof(PufferState));
    buf->best_state_return = (float*)calloc(best_entries, sizeof(float));
    buf->active_state_return = (float*)calloc(active_entries, sizeof(float));
    buf->best_state_step = (int*)calloc(best_entries, sizeof(int));
    buf->active_state_step = (int*)calloc(active_entries, sizeof(int));
    buf->best_meta = (BestTrajectoryMeta*)calloc(
        best_rows, sizeof(BestTrajectoryMeta));
    buf->active_rows = (ActiveTrajectoryRow*)calloc(
        active_rows, sizeof(ActiveTrajectoryRow));
    buf->segments = (CompletedSegment*)calloc(
        segment_entries, sizeof(CompletedSegment));
    buf->rng_states = (unsigned int*)calloc(
        rng_entries, sizeof(unsigned int));

    if (buf->best_states == NULL || buf->active_states == NULL
            || buf->best_state_return == NULL || buf->active_state_return == NULL
            || buf->best_state_step == NULL || buf->active_state_step == NULL
            || buf->best_meta == NULL || buf->active_rows == NULL
            || buf->segments == NULL || buf->rng_states == NULL) {
        fprintf(stderr,
            "Failed to allocate curriculum trajectory buffer: starts=%d len=%d active=%d state_size=%d\n",
            buf->num_start_states, buf->trajectory_max_len,
            buf->max_active_envs, (int)sizeof(PufferState));
        close_state_buffer(buf);
        abort();
    }

    for (int i = 0; i < buf->rng_state_count; i++) {
        buf->rng_states[i] = buf->rng_seed + (unsigned int)i;
    }
}

void close_state_buffer(StateBuffer* buf) {
    free(buf->best_states);
    free(buf->active_states);
    free(buf->best_state_return);
    free(buf->active_state_return);
    free(buf->best_state_step);
    free(buf->active_state_step);
    free(buf->best_meta);
    free(buf->active_rows);
    free(buf->segments);
    free(buf->rng_states);
}

#endif

#ifdef PUFFER_CURRICULUM_TYPES

void register_prio_buffers(PrioBuffers& bufs, Allocator* alloc, int B, int minibatch_segments) {
    size_t cdf_temp_bytes = 0;
    cudaError_t err = cub::DeviceScan::InclusiveSum(
        NULL, cdf_temp_bytes, (float*)NULL, (float*)NULL, B);
    assert(err == cudaSuccess);

    bufs = (PrioBuffers){
        .prio_weights = {.shape = {B}},
        .cdf = {.shape = {B}},
        .cdf_temp = {.shape = {(int64_t)cdf_temp_bytes}},
        .mb_prio = {.shape = {minibatch_segments}},
        .idx = {.shape = {minibatch_segments}},
        .sample_done = {.shape = {1}},
    };
    alloc_register(alloc, &bufs.prio_weights);
    alloc_register(alloc, &bufs.cdf);
    alloc_register(alloc, &bufs.cdf_temp);
    alloc_register(alloc, &bufs.idx);
    alloc_register(alloc, &bufs.sample_done);
    alloc_register(alloc, &bufs.mb_prio);
}

#endif

#ifdef PUFFER_CURRICULUM_IMPL

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static int fixed_agents_per_env(StaticVec* vec) {
    assert(vec->size > 0 && "state curriculum requires at least one env");
    int agents_per_env = vec->envs[0].num_agents;
    assert(agents_per_env > 0 && "env num_agents must be positive");
    for (int i = 0; i < vec->size; i++) {
        assert(vec->envs[i].num_agents == agents_per_env
            && "state curriculum currently requires fixed num_agents per env");
    }
    assert(vec->size * agents_per_env == vec->total_agents
        && "env agent counts must sum to total_agents");
    assert(vec->agents_per_buffer % agents_per_env == 0
        && "state curriculum requires agents_per_buffer to be divisible by num_agents");
    return agents_per_env;
}

static inline int curriculum_best_base(StateBuffer* buf, int slot) {
    return slot * buf->trajectory_max_len;
}

static inline int curriculum_segment_base(StateBuffer* buf, int row) {
    return row * buf->max_segments_per_row;
}

static inline void curriculum_record_state(StateBuffer* buf, int row,
        const PufferState* state) {
    ActiveTrajectoryRow* active = &buf->active_rows[row];
    int count = active->history_count;
    if (count >= buf->trajectory_max_len) {
        return;
    }
    int base = row * buf->trajectory_max_len + count;
    buf->active_states[base] = *state;
    buf->active_state_return[base] = active->episode_return;
    buf->active_state_step[base] = active->episode_len;
    active->history_count = count + 1;
}

static inline void curriculum_start_env(PuffeRL* pufferl, int env_idx,
        int is_cl, unsigned int* rng_state, int reset_history,
        int copy_to_gpu) {
    StateBuffer* buf = &pufferl->state_buf;
    StaticVec* vec = pufferl->vec;
    // curriculum_active_row
    int active_row = env_idx - buf->num_vanilla_envs;
    ActiveTrajectoryRow* active = &buf->active_rows[active_row];

    int slot = (int)(rand_r(rng_state) % (unsigned int)buf->num_start_states);
    int offset = 0;
    float prefix_return = 0.0f;
    int prefix_step = 0;
    long long generation = -1;
    PufferState start_state;

    if (is_cl) {
        // curriculum_sample_offset
        int len = buf->best_meta[slot].len;
        offset = len <= 1
            ? 0
            : (int)(rand_r(rng_state) % (unsigned int)len);
    }
    int best_base = curriculum_best_base(buf, slot);
    start_state = buf->best_states[best_base + offset];
    if (is_cl) {
        prefix_return = buf->best_state_return[best_base + offset];
        prefix_step = buf->best_state_step[best_base + offset];
    }
    generation = buf->best_meta[slot].generation;

    Env* env = &vec->envs[env_idx];
    if (copy_to_gpu) {
        static_vec_reset(vec, env_idx, 1, &start_state);
    } else {
        puffer_env_reset(env, &start_state);
    }
    if (reset_history) {
        active->history_count = 0;
        active->segment_start = 0;
        active->segment_count = 0;
    } else {
        active->segment_start = active->history_count;
    }
    active->best_slot = slot;
    active->sample_offset = offset;
    active->episode_return = 0.0f;
    active->prefix_return = prefix_return;
    active->episode_len = 0;
    active->prefix_step = prefix_step;
    active->best_generation = generation;
    curriculum_record_state(buf, active_row, &env->state);
}

static inline void curriculum_post_active_step(PuffeRL* pufferl,
        int buffer_idx, int env_idx, int row, int is_cl) {
    StateBuffer* buf = &pufferl->state_buf;
    StaticVec* vec = pufferl->vec;
    Env* env = &vec->envs[env_idx];
    ActiveTrajectoryRow* active = &buf->active_rows[row];
    // curriculum_observe_step
    float reward = 0.0f;
    for (int a = 0; a < env->num_agents; a++) {
        float value = env->rewards[a];
        if (isnan(value) || isinf(value)) {
            abort();
        }
        reward += value;
    }
    active->episode_return += reward;
    active->episode_len += 1;
    if (env->terminals[0] > 0.5f) {
        // curriculum_process_terminal
        int slot = active->best_slot;
        int start = active->segment_start;
        int count = active->history_count - start;
        int seg_count = active->segment_count;
        if (count > 0) {
            int seg_idx = curriculum_segment_base(buf, row) + seg_count;
            CompletedSegment* segment = &buf->segments[seg_idx];
            active->segment_count = seg_count + 1;
            segment->best_slot = slot;
            segment->sample_offset = active->sample_offset;
            segment->start = start;
            segment->count = count;
            segment->episode_return = active->episode_return;
            if (is_cl) {
                segment->episode_return += active->prefix_return;
            }
            if (isnan(segment->episode_return)
                    || isinf(segment->episode_return)) {
                abort();
            }
            segment->prefix_return = active->prefix_return;
            segment->episode_len = active->episode_len;
            if (is_cl) {
                segment->episode_len += active->prefix_step;
            }
            segment->prefix_step = active->prefix_step;
            segment->best_generation = active->best_generation;
        }
        // curriculum_worker_rng_state
        int rng_idx = buffer_idx * buf->rng_workers_per_buffer
            + omp_get_thread_num();
        unsigned int* rng_state = &buf->rng_states[rng_idx];
        curriculum_start_env(pufferl, env_idx, is_cl,
            rng_state, 0, 0);
        return;
    }
    // curriculum_should_record_checkpoint
    int interval = buf->checkpoint_interval;
    int should_record = interval <= 1;
    if (!should_record) {
        int len = active->episode_len;
        should_record = len > 0 && (len % interval) == 0;
    }
    if (should_record) {
        curriculum_record_state(buf, row, &env->state);
    }
}

void curriculum_init_active_envs(PuffeRL* pufferl) {
    StateBuffer* buf = &pufferl->state_buf;
    StaticVec* vec = pufferl->vec;
    HypersT* h = &pufferl->hypers;
    int total_envs = vec->size;

    // curriculum_seed_best
    for (int slot = 0; slot < buf->num_start_states; slot++) {
        Env* env = &vec->envs[slot % vec->size];
        int best_base = curriculum_best_base(buf, slot);
        PufferState* dst = buf->best_states + best_base;
        dst[0] = env->state;
        buf->best_meta[slot].len = 1;
        buf->best_meta[slot].episode_len = 0;
        buf->best_meta[slot].episode_return = 0.0f;
        buf->best_state_return[best_base] = 0.0f;
        buf->best_state_step[best_base] = 0;
        buf->best_meta[slot].generation = 1;
    }

    int num_cl_envs = h->cl_frac * total_envs;
    int num_fresh_envs = h->fresh_frac * total_envs;
    int active_envs = num_cl_envs + num_fresh_envs;

    buf->num_cl_envs = num_cl_envs;
    buf->num_fresh_envs = num_fresh_envs;
    buf->num_vanilla_envs = total_envs - active_envs;
    vec->log_env_limit = buf->num_vanilla_envs;

    int fresh_start = buf->num_vanilla_envs;
    int cl_start = fresh_start + num_fresh_envs;
    for (int i = 0; i < num_fresh_envs; i++) {
        int env_idx = fresh_start + i;
        curriculum_start_env(pufferl, env_idx,
            0, &buf->host_rng_state, 1, 1);
    }
    for (int i = 0; i < num_cl_envs; i++) {
        int env_idx = cl_start + i;
        curriculum_start_env(pufferl, env_idx,
            1, &buf->host_rng_state, 1, 1);
    }
}

void curriculum_rollout_end(PuffeRL* pufferl) {
    StateBuffer* buf = &pufferl->state_buf;
    // curriculum_apply_completed_segments
    for (int slot = 0; slot < buf->num_start_states; slot++) {
        int best_row = -1;
        int best_seg = -1;
        BestTrajectoryMeta* meta = &buf->best_meta[slot];
        float candidate_return = meta->episode_return;
        int candidate_len = meta->episode_len;
        long long generation = meta->generation;
        for (int row = 0; row < buf->max_active_envs; row++) {
            int seg_base = curriculum_segment_base(buf, row);
            int seg_count = buf->active_rows[row].segment_count;
            for (int i = 0; i < seg_count; i++) {
                int seg_idx = seg_base + i;
                CompletedSegment* segment = &buf->segments[seg_idx];
                if (segment->best_slot != slot
                        || segment->best_generation != generation) {
                    continue;
                }
                // curriculum_segment_beats
                float terminal_return = segment->episode_return;
                int segment_beats =
                    terminal_return > candidate_return + 1e-6f
                    || (fabsf(terminal_return - candidate_return) <= 1e-6f
                        && segment->episode_len < candidate_len);
                if (segment_beats) {
                    best_row = row;
                    best_seg = seg_idx;
                    candidate_return = segment->episode_return;
                    candidate_len = segment->episode_len;
                }
            }
        }
        if (best_seg < 0) {
            continue;
        }
        CompletedSegment* segment = &buf->segments[best_seg];
        // curriculum_row_is_fresh
        if (best_row < buf->num_fresh_envs) {
            // curriculum_apply_full_segment
            int apply_slot = segment->best_slot;
            int apply_count = segment->count;
            int active_base = best_row * buf->trajectory_max_len
                + segment->start;
            int best_base = curriculum_best_base(buf, apply_slot);
            memcpy(buf->best_states + best_base,
                buf->active_states + active_base,
                (size_t)apply_count * sizeof(PufferState));
            memcpy(buf->best_state_return + best_base,
                buf->active_state_return + active_base,
                (size_t)apply_count * sizeof(float));
            memcpy(buf->best_state_step + best_base,
                buf->active_state_step + active_base,
                (size_t)apply_count * sizeof(int));
            BestTrajectoryMeta* apply_meta = &buf->best_meta[apply_slot];
            apply_meta->len = apply_count;
            apply_meta->episode_len = segment->episode_len;
            apply_meta->episode_return = segment->episode_return;
            apply_meta->generation++;
        } else {
            // curriculum_apply_tail_segment
            int apply_slot = segment->best_slot;
            int offset = segment->sample_offset;
            int apply_count = segment->count;
            BestTrajectoryMeta* apply_meta = &buf->best_meta[apply_slot];
            if (offset + apply_count > buf->trajectory_max_len) {
                apply_count = buf->trajectory_max_len - offset;
            }
            if (apply_count > 0) {
                int active_base = best_row * buf->trajectory_max_len
                    + segment->start;
                int best_base = curriculum_best_base(buf, apply_slot);
                memcpy(buf->best_states + best_base + offset,
                    buf->active_states + active_base,
                    (size_t)apply_count * sizeof(PufferState));
                for (int i = 0; i < apply_count; i++) {
                    buf->best_state_return[best_base + offset + i] =
                        segment->prefix_return
                        + buf->active_state_return[active_base + i];
                    buf->best_state_step[best_base + offset + i] =
                        segment->prefix_step
                        + buf->active_state_step[active_base + i];
                }
                apply_meta->len = offset + apply_count;
                apply_meta->episode_len = segment->episode_len;
                apply_meta->episode_return = segment->episode_return;
                apply_meta->generation++;
            }
        }
    }

    // curriculum_compact_active_rows
    for (int row = 0; row < buf->max_active_envs; row++) {
        ActiveTrajectoryRow* active = &buf->active_rows[row];
        int start = active->segment_start;
        int count = active->history_count - start;
        int base = row * buf->trajectory_max_len;
        if (start > 0 && count > 0) {
            memmove(buf->active_states + base,
                buf->active_states + base + start,
                (size_t)count * sizeof(PufferState));
            memmove(buf->active_state_return + base,
                buf->active_state_return + base + start,
                (size_t)count * sizeof(float));
            memmove(buf->active_state_step + base,
                buf->active_state_step + base + start,
                (size_t)count * sizeof(int));
        }
        active->history_count = count;
        active->segment_start = 0;
        active->segment_count = 0;
    }
}

__device__ __forceinline__ float priority_power(float value, float alpha) {
    if (alpha == 0.0f) {
        return 1.0f;
    }
    if (value <= 0.0f) {
        return 0.0f;
    }
    value = __powf(value, alpha);
    if (isnan(value) || isinf(value)) {
        return 0.0f;
    }
    return value;
}

__global__ void compute_prio_abs(
        const precision_t* __restrict__ advantages,
        float* prio_weights, float prio_alpha, float eps, int rows, int stride) {
    if (stride == 1) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows) {
            prio_weights[idx] = priority_power(fabsf(to_float(advantages[idx])), prio_alpha) + eps;
        }
        return;
    }

    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;

    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(to_float(advantages[offset + t]));
    }

    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (tx == 0) {
        prio_weights[row] = priority_power(local_sum, prio_alpha) + eps;
    }
}

__global__ void multinomial_sample_advance(int* __restrict__ out_idx,
        precision_t* __restrict__ out_importance,
        const float* __restrict__ prio_weights, const float* __restrict__ cdf,
        int B, int num_samples, uint64_t seed, float beta,
        int64_t* __restrict__ offset_ptr, int* __restrict__ done_counter,
        int launch_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t base_off = *offset_ptr;
    float total_weight = cdf[B - 1];
    if (tid < num_samples) {
        curandStatePhilox4_32_10_t rng_state;
        curand_init(seed, (uint64_t)base_off + tid, 0, &rng_state);
        float u = curand_uniform(&rng_state);

        int lo;
        int use_uniform = total_weight <= 0.0f || isnan(total_weight) || isinf(total_weight);
        if (use_uniform) {
            lo = (int)(u * (float)B);
            if (lo >= B) {
                lo = B - 1;
            }
        } else {
            float target = u * total_weight;
            lo = 0;
            int hi = B - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (cdf[mid] < target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
        }
        if (out_importance != NULL) {
            float weight = 1.0f;
            if (!use_uniform) {
                float value = prio_weights[lo] * (float)B / total_weight;
                weight = __powf(value, -beta);
                if (isnan(weight) || isinf(weight)) {
                    weight = 1.0f;
                }
            }
            precision_t value = from_float(weight);
            out_importance[tid] = value;
        }
        out_idx[tid] = lo;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence();
        int ticket = atomicAdd(done_counter, 1);
        if (ticket == launch_blocks - 1) {
            *offset_ptr = base_off + num_samples;
            *done_counter = 0;
        }
    }
}

static inline void sample_prio_indices(PrioBuffers* bufs, int population,
        int samples, ulong seed, long* offset_ptr, precision_t* out_importance,
        float beta, cudaStream_t stream) {
    size_t cdf_temp_bytes = (size_t)bufs->cdf_temp.shape[0];
    cub::DeviceScan::InclusiveSum(bufs->cdf_temp.data, cdf_temp_bytes,
        bufs->prio_weights.data, bufs->cdf.data, population, stream);
    int blocks = (samples + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    multinomial_sample_advance<<<blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs->idx.data, out_importance,
        bufs->prio_weights.data, bufs->cdf.data, population, samples, seed, beta,
        offset_ptr, bufs->sample_done.data, blocks);
}

#endif
