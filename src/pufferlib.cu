#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <nvml.h>
#include <nccl.h>
#include <vector>

#include <time.h>
#include "models.cu"
#include "ocean.cu"
#include "muon.cu"
#include "vecenv.h"

static double wall_clock() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

enum LossIdx {
    LOSS_PG = 0, LOSS_VF = 1, LOSS_ENT = 2, LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4, LOSS_APPROX_KL = 5, LOSS_CLIPFRAC = 6,
    LOSS_PPO_ACTIVE_HEADS = 7, LOSS_PPO_ACTIVE_GROUPS = 8,
    LOSS_PPO_OPTIMIZED_KL = 9, LOSS_PPO_OPTIMIZED_CLIPFRAC = 10,
    LOSS_PPO_OPTIMIZED_ENTROPY = 11,
    LOSS_PPO_JOINT_OLD_KL = 12, LOSS_PPO_JOINT_KL = 13,
    LOSS_PPO_JOINT_CLIPFRAC = 14,
    LOSS_N = 15, NUM_LOSSES = 16,
};

enum PPOClipMode {
    PPO_CLIP_JOINT = 0,
    PPO_CLIP_PER_HEAD = 1,
    PPO_CLIP_GROUP_SUM = 2,
    PPO_CLIP_GROUP_MEAN = 3,
};

enum ProfileIdx {
    PROF_ROLLOUT = 0,
    PROF_EVAL_GPU,
    PROF_EVAL_ENV,
    PROF_TRAIN_MISC,
    PROF_TRAIN_FORWARD,
    NUM_PROF,
};

static const char* PROF_NAMES[NUM_PROF] = {
    "rollout",
    "eval_gpu",
    "eval_env",
    "train_misc",
    "train_forward",
};

#define NUM_TRAIN_EVENTS 5
typedef struct {
    cudaEvent_t events[NUM_TRAIN_EVENTS];
    float accum[NUM_PROF];
} ProfileT;

// Data collected by parallel environment workers. Each worker handles
// a constant subset of agents 
struct RolloutBuf {
    PrecisionTensor observations;  // (horizon, agents, input_size)
    PrecisionTensor actions;       // (horizon, agents, num_atns)
    PrecisionTensor values;        // (horizon, agents)
    PrecisionTensor logprobs;      // ...
    PrecisionTensor rewards;
    PrecisionTensor terminals;
    PrecisionTensor ratio;
    PrecisionTensor importance;
    PrecisionTensor action_mask;   // (horizon, agents, mask_size); .data=nullptr when env opts out
};

// Buffers are initialized as raw structs with only shape information. alloc_register
// stores the shape and data pointer. Memory is only allocated after all buffers are registered.
void register_rollout_buffers(RolloutBuf& bufs, Allocator* alloc, int T, int B, int input_size,
        int num_atns, int mask_size) {
    bufs = (RolloutBuf){
        .observations = {.shape = {T, B, input_size}},
        .actions      = {.shape = {T, B, num_atns}},
        .values       = {.shape = {T, B}},
        .logprobs     = {.shape = {T, B, num_atns}},
        .rewards      = {.shape = {T, B}},
        .terminals    = {.shape = {T, B}},
        .ratio        = {.shape = {T, B}},
        .importance   = {.shape = {T, B}},
        .action_mask  = {},
    };
    alloc_register(alloc, &bufs.observations);
    alloc_register(alloc, &bufs.actions);
    alloc_register(alloc, &bufs.values);
    alloc_register(alloc, &bufs.logprobs);
    alloc_register(alloc, &bufs.rewards);
    alloc_register(alloc, &bufs.terminals);
    alloc_register(alloc, &bufs.ratio);
    alloc_register(alloc, &bufs.importance);
    if (mask_size > 0) {
        bufs.action_mask = {.shape = {T, B, mask_size}};
        alloc_register(alloc, &bufs.action_mask);
    }
}

// Train data layout is transposed to (B, T) from rollouts layout (T, B)
// This allows env workers to collect data with contiguous writes and
// training to perform several (though not all) ops in contiguous memory
struct TrainGraph {
    PrecisionTensor mb_state;       // (layers, B, hidden)
    PrecisionTensor mb_obs;         // (B, T, input_size)
    PrecisionTensor mb_actions;     // (B, T, num_atns)
    PrecisionTensor mb_logprobs;    // (B, T, num_atns)
    PrecisionTensor mb_advantages;  // ...
    PrecisionTensor mb_values;
    PrecisionTensor mb_returns;
    PrecisionTensor mb_ratio;
    PrecisionTensor mb_newvalue;
    PrecisionTensor mb_prio;        // (B,)
    PrecisionTensor mb_action_mask; // (B, T, mask_size); .data=nullptr when disabled
    PrecisionTensor mb_trainable;   // (B,), zero for frozen-opponent segments
};

void register_train_buffers(TrainGraph& bufs, Allocator* alloc, int B, int T, int input_size,
        int hidden_size, int num_atns, int num_layers, int mask_size) {
    bufs = (TrainGraph){
        .mb_state =         {.shape = {num_layers, B, hidden_size}},
        .mb_obs =           {.shape = {B, T, input_size}},
        .mb_actions =       {.shape = {B, T, num_atns}},
        .mb_logprobs =      {.shape = {B, T, num_atns}},
        .mb_advantages =    {.shape = {B, T}},
        .mb_values =        {.shape = {B, T}},
        .mb_returns =       {.shape = {B, T}},
        .mb_ratio =         {.shape = {B, T}},
        .mb_newvalue =      {.shape = {B, T}},
        .mb_prio =          {.shape = {B}},
        .mb_action_mask =   {},
        .mb_trainable =     {.shape = {B}},
    };
    alloc_register(alloc, &bufs.mb_obs);
    alloc_register(alloc, &bufs.mb_state);
    alloc_register(alloc, &bufs.mb_actions);
    alloc_register(alloc, &bufs.mb_logprobs);
    alloc_register(alloc, &bufs.mb_advantages);
    alloc_register(alloc, &bufs.mb_prio);
    alloc_register(alloc, &bufs.mb_values);
    alloc_register(alloc, &bufs.mb_returns);
    alloc_register(alloc, &bufs.mb_ratio);
    alloc_register(alloc, &bufs.mb_newvalue);
    alloc_register(alloc, &bufs.mb_trainable);
    if (mask_size > 0) {
        bufs.mb_action_mask = {.shape = {B, T, mask_size}};
        alloc_register(alloc, &bufs.mb_action_mask);
    }
}

// PPO buffers + args are quite complex. We do the entire
// forward + backwards pass for the full loss function in one kernel
struct PPOGraphArgs {
    precision_t* out_ratio;
    precision_t* out_newvalue;
    const precision_t* actions;
    const precision_t* old_logprobs;
    const precision_t* advantages;
    const precision_t* prio;
    const precision_t* values;
    const precision_t* returns;
    const precision_t* trainable;
};

struct PPOKernelArgs {
    float* grad_logits;
    float* grad_logstd; // For continuous actions
    float* grad_values_pred;
    const precision_t* logits;
    const precision_t* logstd; // Continuous only
    const precision_t* values_pred;
    const float* adv_mean;
    const float* adv_var;
    const int* act_sizes;
    const precision_t* action_mask; // (N, T, A_total) or nullptr
    int mask_stride_n, mask_stride_t;
    const signed char* head_consume; // legacy (nverbs, num_atns) or nullptr
    int hc_stride;
    const int* head_to_group;
    const int* head_selector;
    const int* head_used_offsets;
    const unsigned char* head_used_values;
    int num_groups;
    int ppo_clip_mode;
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    const float* ent_coef; // device ptr, by-value args get baked into the cuda graph
    int T_seq, A_total, N;
    int logits_stride_n, logits_stride_t, logits_stride_a;
    int values_stride_n, values_stride_t;
    bool is_continuous;
};

struct PPOBuffersPuf {
    FloatTensor loss_output, grad_loss;
    FloatTensor saved_for_bwd;
    FloatTensor grad_logits, grad_values, grad_logstd, adv_scratch;
    FloatTensor ent_coef;
};

void register_ppo_buffers(PPOBuffersPuf& bufs, Allocator* alloc, int N, int T, int A_total, bool is_continuous) {
    long total = (long)N * T;
    bufs = (PPOBuffersPuf){
        .loss_output = {.shape = {1}},
        .grad_loss = {.shape = {1}},
        .saved_for_bwd = {.shape = {total, 5}},
        .grad_logits = {.shape = {N, T, A_total}},
        .grad_values = {.shape = {N, T, 1}},
        .grad_logstd = {.shape = {N, T, A_total}},
        .adv_scratch = {.shape = {3}},
        .ent_coef = {.shape = {1}},
    };
    alloc_register(alloc, &bufs.loss_output);
    alloc_register(alloc, &bufs.saved_for_bwd);
    alloc_register(alloc, &bufs.grad_loss);
    alloc_register(alloc, &bufs.grad_logits);
    alloc_register(alloc, &bufs.grad_values);
    if (is_continuous) {
        alloc_register(alloc, &bufs.grad_logstd);
    }
    alloc_register(alloc, &bufs.adv_scratch);
    alloc_register(alloc, &bufs.ent_coef);
}

// Prioritized replay over single-epoch data. These kernels are
// the least cleaned because we will likely have a better method in 5.0
struct PrioBuffers {
    FloatTensor prio_probs, cdf, mb_prio;
    IntTensor idx;
};

void register_prio_buffers(PrioBuffers& bufs, Allocator* alloc, int B, int minibatch_segments) {
    bufs = (PrioBuffers){
        .prio_probs = {.shape = {B}},
        .cdf = {.shape = {B}},
        .mb_prio = {.shape = {minibatch_segments}},
        .idx = {.shape = {minibatch_segments}},
    };
    alloc_register(alloc, &bufs.prio_probs);
    alloc_register(alloc, &bufs.cdf);
    alloc_register(alloc, &bufs.idx);
    alloc_register(alloc, &bufs.mb_prio);
}

// Slice: select dim0 index t, then narrow dim0 from start for count.
// 3D (T, B, F) -> (count, F); 2D (T, B) -> (count,)
inline PrecisionTensor puf_slice(PrecisionTensor& p, int t, int start, int count) {
    if (ndim(p.shape) == 3) {
        long B = p.shape[1], F = p.shape[2];
        return {.data = p.data + (t*B + start)*F, .shape = {count, F}};
    } else {
        long B = p.shape[1];
        return {.data = p.data + (t*B + start), .shape = {count}};
    }
}

struct EnvBuf {
    OBS_TENSOR_T obs;      // (total_agents, obs_size) - type defined per-env in binding.c
    FloatTensor actions;   // (total_agents, num_atns)
    FloatTensor rewards;   // (total_agents,)
    FloatTensor terminals; // (total_agents,)
    ByteTensor action_mask; // (total_agents, mask_size); .data=nullptr when env opts out
};

StaticVec* create_environments(int num_buffers, int total_agents,
        const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs, EnvBuf& env) {
    StaticVec* vec = create_static_vec(total_agents, num_buffers, 1, vec_kwargs, env_kwargs);
    env.obs = {
        .data = (decltype(env.obs.data))vec->gpu_observations,
        .shape = {total_agents, get_obs_size()},
    };
    env.actions = { .data = (float*)vec->gpu_actions, .shape = {total_agents, get_num_atns()} };
    env.rewards = { .data = (float*)vec->gpu_rewards, .shape = {total_agents} };
    env.terminals = { .data = (float*)vec->gpu_terminals, .shape = {total_agents} };
    if (vec->action_mask_size > 0) {
        env.action_mask = { .data = vec->gpu_action_mask,
                            .shape = {total_agents, vec->action_mask_size} };
    } else {
        env.action_mask = { .data = nullptr, .shape = {0} };
    }
    return vec;
}

typedef struct {
    // Layout
    int horizon;
    int total_agents;
    int num_buffers;
    // Model architecture
    int num_atns;
    int hidden_size;
    int num_layers;
    // Learning rate
    float lr;
    float min_lr_ratio;
    bool anneal_lr;
    // Optimizer
    float beta1;
    float beta2;
    float eps;
    // Training
    int minibatch_size;
    float replay_ratio;
    long total_timesteps;
    float max_grad_norm;
    // PPO
    float clip_coef;
    int ppo_clip_mode;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    // Entropy coefficient anneal — mirrors lr annealing. When anneal_ent_coef
    // is set, ent_coef cosine-decays from its base value to
    // min_ent_coef_ratio * ent_coef over total_timesteps.
    float min_ent_coef_ratio;
    bool anneal_ent_coef;
    // GAE
    float gamma;
    float gae_lambda;
    // VTrace
    float vtrace_rho_clip;
    float vtrace_c_clip;
    // Priority
    float prio_alpha;
    float prio_beta0;
    // Flags
    bool reset_state;
    int cudagraphs;
    bool profile;
    // Multi-GPU
    int rank;
    int world_size;
    int gpu_id;
    std::string nccl_id;  // raw bytes of ncclUniqueId (empty for single-GPU)
    // Threading
    int num_threads;
    int seed;
} HypersT;

// A frozen weight bank: same shape as the primary, but its own params buffer
// (and per-buffer rollout states/activations). Used for match (eval) and league
// (frozen historical opponents). Not trained; updated only via load.
typedef struct {
    Policy policy;  // Bank-owned Policy; lets banks have different arch than primary.
    PolicyWeights weights;
    Allocator params_alloc;
    Allocator acts_alloc;
    PrecisionTensor param_puf;
    FloatTensor master_weights;
    PrecisionTensor* buffer_states;         // [num_buffers]
    PolicyActivations* buffer_activations;  // [num_buffers]
    int slice_size;  // # agents per buffer this bank owns; sets activation/state batch dim
    int hidden_size;
    int num_layers;
} WeightBank;

typedef struct {
    Policy policy;
    PolicyWeights weights;       // current precision_t weights (structured)
    PolicyActivations train_activations;
    Allocator params_alloc;
    Allocator grads_alloc;
    Allocator activations_alloc;
    StaticVec* vec;
    Muon muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    HypersT hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    PrecisionTensor* buffer_states;  // Per-buffer states for contiguous access
    PolicyActivations* buffer_activations;  // Per-buffer inference activations
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;  // Pre-allocated transposed copy for train_impl
    EnvBuf env;
    TrainGraph train_buf;
    PrecisionTensor advantages_puf;  // Pre-allocated for train_impl (B, T)
    cudaGraphExec_t* fused_rollout_cudagraphs;  // [horizon][num_buffers]
    cudaGraphExec_t train_cudagraph;
    cudaStream_t* streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    IntTensor act_sizes_puf;    // CUDA int32 tensor of action head sizes
    FloatTensor losses_puf;     // (NUM_LOSSES,) f32 accumulator
    PPOBuffersPuf ppo_bufs_puf; // Pre-allocated buffers for ppo_loss_fwd_bwd
    PrioBuffers prio_bufs;      // Pre-allocated buffers for prio_replay
    FloatTensor master_weights;  // fp32 master weights (flat); same buffer as param_puf in fp32 mode
    PrecisionTensor param_puf;
    PrecisionTensor grad_puf;
    LongTensor rng_offset_puf;   // (num_buffers+1,) int64 CUDA device counters
    int ppo_head_adapter_version;
    int ppo_num_groups;
    int* ppo_head_to_group;
    int* ppo_head_selector;
    int* ppo_head_used_offsets;
    unsigned char* ppo_head_used_values;
    ProfileT profile;
    nvmlDevice_t nvml_device;
    long epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    int train_warmup;
    bool rollout_captured;
    bool train_captured;
    ulong seed;
    curandStatePhilox4_32_10_t** rng_states;  // per-buffer persistent RNG states [num_buffers]
    // Optional frozen weight banks for match / league.
    WeightBank* frozen_banks;  // [num_frozen_banks]
    int num_frozen_banks;
    std::string env_name;  // Kept for post-init bank adds (needs create_custom_encoder).
    // Per-buffer-relative bank layout: bank_layout[b] = first agent within each
    // buffer chunk owned by bank b. Length num_banks+1; ends at agents_per_buffer.
    // Same shape applied to every buffer (each buffer hosts every bank), so each
    // worker thread only writes inside its own physical chunk.
    // Bank 0 = primary (learner). NULL = no layout set (primary owns full chunk).
    int* bank_layout;
} PuffeRL;

Dict* log_environments_impl(PuffeRL& pufferl) {
    // Capacity raised from 32 to 64 to accommodate chess's per-bank
    // hist_score_bank_<b> / hist_n_bank_<b> entries (16 keys for 8 banks).
    Dict* out = create_dict(64);
    static_vec_log(pufferl.vec, out);
    return out;
}

inline void profile_begin(const char* tag, bool enable) {
    if (enable) nvtxRangePushA(tag);
}

inline void profile_end(bool enable) {
    if (enable) nvtxRangePop();
}

// Thread-local stream for per-buffer threads (set once by thread_init_wrapper)
static thread_local cudaStream_t tl_stream = 0;

// Thread initialization callback - sets thread-local stream once per thread
extern "C" void thread_init_wrapper(void* ctx, int buf) {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    tl_stream = pufferl->streams[buf];
}

__global__ void rng_init(curandStatePhilox4_32_10_t* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ __forceinline__ float safe_logit(const precision_t* logits,
        int logits_base, int logits_offset, int offset) {
    float l = to_float(logits[logits_base + logits_offset + offset]);
    if (isnan(l)) {
        l = 0.0f;
    }
    if (isinf(l)) {
        l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
    }
    return l;
}

__device__ __forceinline__ float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) {
        return 0.0f;
    }
    if (isinf(x)) {
        return x > 0.0f ? hi : lo;
    }
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ float safe_continuous_mean(const precision_t* logits, int idx) {
    return finite_or_clamp(to_float(logits[idx]), -1.0e6f, 1.0e6f);
}

__device__ __forceinline__ float safe_continuous_logstd(const precision_t* logstd, int idx) {
    return finite_or_clamp(to_float(logstd[idx]), -20.0f, 2.0f);
}

__device__ __forceinline__ float masked_logit(const precision_t* logits,
        int logits_base, int logits_offset, int offset,
        const precision_t* mask, int mask_base) {
    float l = safe_logit(logits, logits_base, logits_offset, offset);
    if (mask != nullptr) {
        float m = to_float(mask[mask_base + logits_offset + offset]);
        if (m == 0.0f) l = -INFINITY;
    }
    return l;
}

// Expects action logits and values to be in the same contiguous buffer. See default decoder
// ---- factorized PPO environment adapter + legacy consumed-head gating ----
extern "C" __attribute__((weak)) const PufferPPOHeadAdapter* puffer_ppo_head_adapter(void);
extern "C" __attribute__((weak)) const signed char* env_head_consume_map(int*, int*);

struct ExpandedPPOHeadAdapter {
    int version = 0;
    int num_groups = 0;
    std::vector<int> head_to_group;
    std::vector<int> head_selector;
    std::vector<int> head_used_offsets;
    std::vector<unsigned char> head_used_values;
};

static ExpandedPPOHeadAdapter expand_ppo_head_adapter(
        int num_heads, const int* act_sizes, bool required) {
    ExpandedPPOHeadAdapter out;
    const PufferPPOHeadAdapter* adapter =
        puffer_ppo_head_adapter ? puffer_ppo_head_adapter() : nullptr;
    if (!adapter) {
        if (required) {
            throw std::runtime_error(
                "non-joint train.ppo_clip_mode requires "
                "puffer_ppo_head_adapter() from the environment binding");
        }
        return out;
    }
    if (adapter->struct_size < sizeof(PufferPPOHeadAdapter)
            || adapter->version != PUFFER_PPO_HEAD_ADAPTER_VERSION
            || adapter->num_heads != num_heads || adapter->num_groups <= 0
            || !adapter->head_to_group || !adapter->head_selector
            || !adapter->head_used) {
        throw std::runtime_error("invalid puffer_ppo_head_adapter() contract");
    }

    out.version = adapter->version;
    out.num_groups = adapter->num_groups;
    out.head_to_group.assign(adapter->head_to_group,
        adapter->head_to_group + num_heads);
    out.head_selector.assign(adapter->head_selector,
        adapter->head_selector + num_heads);
    out.head_used_offsets.resize(num_heads + 1);
    std::vector<int> group_sizes(out.num_groups, 0);
    for (int h = 0; h < num_heads; ++h) {
        int group = out.head_to_group[h];
        int selector = out.head_selector[h];
        if (group < 0 || group >= out.num_groups
                || selector < 0 || selector >= num_heads
                || act_sizes[selector] <= 0) {
            throw std::runtime_error(
                "puffer_ppo_head_adapter() contains an invalid group or selector");
        }
        group_sizes[group]++;
        if (out.head_to_group[selector] != group) {
            throw std::runtime_error(
                "PPO head selectors must belong to the selected head's group");
        }
        out.head_used_offsets[h] = out.head_used_values.size();
        for (int action = 0; action < act_sizes[selector]; ++action) {
            int used = adapter->head_used(h, action);
            if (used != 0 && used != 1) {
                throw std::runtime_error(
                    "puffer_ppo_head_adapter().head_used must return 0 or 1");
            }
            out.head_used_values.push_back((unsigned char)used);
        }
    }
    for (int size : group_sizes) {
        if (size == 0) {
            throw std::runtime_error(
                "puffer_ppo_head_adapter() contains an empty group");
        }
    }
    out.head_used_offsets[num_heads] = out.head_used_values.size();
    return out;
}
static const signed char* g_hc_dev = nullptr;
static int g_hc_stride = 0;
static bool g_hc_init = false;
static const signed char* get_head_consume_dev(int* stride) {
    if (!g_hc_init) {
        g_hc_init = true;
        const char* hg = getenv("PUFFER_HEAD_GATING");
        if (hg && hg[0] && hg[0] != '0' && env_head_consume_map) {
            int nv = 0, na = 0;
            const signed char* host = env_head_consume_map(&nv, &na);
            if (host && nv > 0 && na > 0) {
                signed char* dev = nullptr;
                cudaMalloc(&dev, (size_t)nv * na);
                cudaMemcpy(dev, host, (size_t)nv * na, cudaMemcpyHostToDevice);
                g_hc_dev = dev; g_hc_stride = na;
            }
        }
    }
    *stride = g_hc_stride;
    return g_hc_dev;
}

__global__ void sample_logits(
        PrecisionTensor dec_out,              // (B, logits_dim + 1 for values)
        PrecisionTensor logstd_puf,           // (1, od) - continuous actions only
        IntTensor act_sizes_puf,              // (num_atns,) action head sizes
        precision_t* __restrict__ actions,    // (B, num_atns)
        precision_t* __restrict__ logprobs,   // (B, num_atns)
        precision_t* __restrict__ value_out,  // (B,)
        curandStatePhilox4_32_10_t* __restrict__ rng_states,
        const precision_t* __restrict__ action_mask, // (B, A_total) or nullptr
        int mask_stride) {                    // 0 when action_mask is nullptr
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int num_atns = numel(act_sizes_puf.shape);
    const int* act_sizes = act_sizes_puf.data;
    const precision_t* logits = dec_out.data;
    int logits_stride = fused_cols;
    int value_stride = fused_cols;
    bool is_continuous = logstd_puf.data != nullptr && numel(logstd_puf.shape) > 0;
    const precision_t* logstd = logstd_puf.data;
    int logstd_stride = is_continuous ? 0 : 0;  // 1D broadcast: stride 0
    const precision_t* value = logits + (fused_cols - 1);  // last column

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }

    // Load persistent RNG state (advanced in-place each call)
    curandStatePhilox4_32_10_t state = rng_states[idx];

    int logits_base = idx * logits_stride;
    if (is_continuous) {
        // Continuous action sampling from Normal(mean, exp(logstd))
        constexpr float LOG_2PI = 1.8378770664093453f;  // log(2*pi)
        int logstd_base = idx * logstd_stride;  // separate stride for logstd (may be 0 for broadcast)

        for (int h = 0; h < num_atns; ++h) {
            float mean = safe_continuous_mean(logits, logits_base + h);
            float log_std = safe_continuous_logstd(logstd, logstd_base + h);
            float std = expf(log_std);

            // Sample from N(0,1) and transform: action = mean + std * noise
            float noise = curand_normal(&state);
            float action = finite_or_clamp(mean + std * noise, -1.0e6f, 1.0e6f);

            precision_t stored_action_p = from_float(action);
            float stored_action = to_float(stored_action_p);
            // Log probability: -0.5 * ((action - mean) / std)^2 - 0.5 * log(2*pi) - log(std)
            float normalized = (stored_action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - 0.5f * LOG_2PI - log_std;

            actions[idx * num_atns + h] = stored_action_p;
            logprobs[idx * num_atns + h] = from_float(log_prob);
        }
    } else {
        // Discrete action sampling (original multinomial logic)
        int logits_offset = 0;  // offset within row for current action head
        int mask_base = (action_mask != nullptr) ? idx * mask_stride : 0;

        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];  // size of this action head

            // Step 1: Find max and sum for numerical stability (with nan_to_num)
            float max_val = -INFINITY;
            float sum_exp = 0.0f;
            for (int a = 0; a < A; ++a) {
                if (action_mask != nullptr
                        && to_float(action_mask[mask_base + logits_offset + a]) == 0.0f) {
                    continue;
                }
                float l = masked_logit(logits, logits_base, logits_offset, a, action_mask, mask_base);
                if (l > max_val) {
                    sum_exp *= expf(max_val - l);
                    max_val = l;
                }
                sum_exp += expf(l - max_val);
            }
            float logsumexp = max_val + logf(sum_exp);

            // Step 3: Generate random value for this action head
            float rand_val = curand_uniform(&state);

            // Step 4: Multinomial sampling using inverse CDF
            float cumsum = 0.0f;
            int sampled_action = -1;  // sentinel: no action chosen yet

            for (int a = 0; a < A; ++a) {
                if (action_mask != nullptr
                        && to_float(action_mask[mask_base + logits_offset + a]) == 0.0f) {
                    continue;
                }
                float l = masked_logit(logits, logits_base, logits_offset, a, action_mask, mask_base);
                float prob = expf(l - logsumexp);
                cumsum += prob;
                if (rand_val < cumsum) {
                    sampled_action = a;
                    break;
                }
            }

            // Float rounding can leave cumsum < 1.0; fall back to the last legal action.
            if (sampled_action < 0) {
                sampled_action = A - 1;
                if (action_mask != nullptr) {
                    for (int a = A - 1; a >= 0; --a) {
                        if (to_float(action_mask[mask_base + logits_offset + a]) != 0.0f) {
                            sampled_action = a;
                            break;
                        }
                    }
                }
            }

            // Step 5: Gather log probability of sampled action
            float sampled_logit = masked_logit(logits, logits_base, logits_offset, sampled_action, action_mask, mask_base);
            float log_prob = sampled_logit - logsumexp;

            // Write action for this head
            actions[idx * num_atns + h] = from_float(sampled_action);
            logprobs[idx * num_atns + h] = from_float(log_prob);

            // Advance to next action head
            logits_offset += A;
        }
    }

    // Copy value (fused to avoid separate elementwise kernel for strided->contiguous copy)
    value_out[idx] = value[idx * value_stride];

    // Save RNG state back for next call
    rng_states[idx] = state;
}

// Single step rollout forward pass. Called by each environment worker in their
// own buffer thread. This operation is cudagraphed.
extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    HypersT& hypers = pufferl->hypers;
    int graph = t * hypers.num_buffers + buf;
    profile_begin("fused_rollout", hypers.profile);

    cudaStream_t current_stream = tl_stream;
    if (pufferl->rollout_captured) {
        assert(cudaGraphLaunch(pufferl->fused_rollout_cudagraphs[graph], current_stream) == cudaSuccess
                && "cudaGraphLaunch failed");
        profile_end(hypers.profile);
        return;
    }

    bool capturing = pufferl->epoch == hypers.cudagraphs;
    if (capturing) {
        assert(cudaStreamBeginCapture(current_stream, cudaStreamCaptureModeGlobal) == cudaSuccess
                && "cudaStreamBeginCapture failed");
    }

    RolloutBuf& rollouts = pufferl->rollouts;
    EnvBuf& env = pufferl->env;
    int block_size = pufferl->vec->total_agents / hypers.num_buffers;
    int start = buf * block_size;
    cudaStream_t stream = current_stream;

    // Copy observations, rewards, terminals from GPU env buffers to rollout buffer
    OBS_TENSOR_T& obs_env = env.obs;
    int n = block_size * obs_env.shape[1];
    PrecisionTensor obs_dst = puf_slice(rollouts.observations, t, start, block_size);
    cast_dispatch(obs_dst.data, obs_env.data + (long)start*obs_env.shape[1], n, stream);

    PrecisionTensor rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    n = block_size;
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        rew_dst.data, env.rewards.data + start, n);

    PrecisionTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        term_dst.data, env.terminals.data + start, n);

    // Copy action mask from env into rollout buffer (if env opted in)
    PrecisionTensor mask_slice = {};
    int mask_stride = 0;
    if (rollouts.action_mask.data != nullptr) {
        int mask_size = rollouts.action_mask.shape[2];
        mask_stride = mask_size;
        mask_slice = puf_slice(rollouts.action_mask, t, start, block_size);
        int mask_n = block_size * mask_size;
        cast<<<grid_size(mask_n), BLOCK_SIZE, 0, stream>>>(
            mask_slice.data,
            env.action_mask.data + (long)start * mask_size,
            mask_n);
    }

    // Per-bank policy forward + sampling. Each bank owns a contiguous sub-range
    // [bank_layout[b], bank_layout[b+1]) within every buffer's chunk; layout is
    // per-buffer-relative so each worker writes only inside its own chunk.
    // Cudagraph capture absorbs the extra kernel launches.
    int num_banks = 1 + pufferl->num_frozen_banks;
    long act_cols = env.actions.shape[1];
    for (int b = 0; b < num_banks; b++) {
        int bank_off = pufferl->bank_layout ? pufferl->bank_layout[b] : 0;
        int bank_end = pufferl->bank_layout ? pufferl->bank_layout[b + 1] : block_size;
        int bank_size = bank_end - bank_off;
        if (bank_size == 0) continue;

        Policy* p_bank;
        PolicyWeights* w_bank;
        PolicyActivations* a_bank;
        PrecisionTensor* s_bank;
        if (b == 0) {
            p_bank = &pufferl->policy;
            w_bank = &pufferl->weights;
            a_bank = &pufferl->buffer_activations[buf];
            s_bank = &pufferl->buffer_states[buf];
        } else {
            WeightBank* fb = &pufferl->frozen_banks[b - 1];
            p_bank = &fb->policy;
            w_bank = &fb->weights;
            a_bank = &fb->buffer_activations[buf];
            s_bank = &fb->buffer_states[buf];
        }

        int sub_start = start + bank_off;
        PrecisionTensor obs_b   = puf_slice(rollouts.observations, t, sub_start, bank_size);
        PrecisionTensor act_b   = puf_slice(rollouts.actions,      t, sub_start, bank_size);
        PrecisionTensor lp_b    = puf_slice(rollouts.logprobs,     t, sub_start, bank_size);
        PrecisionTensor val_b   = puf_slice(rollouts.values,       t, sub_start, bank_size);
        PrecisionTensor mask_b  = {};
        int mask_stride_b = 0;
        if (rollouts.action_mask.data != nullptr) {
            mask_b = puf_slice(rollouts.action_mask, t, sub_start, bank_size);
            mask_stride_b = mask_stride;
        }

        PrecisionTensor dec_puf = policy_forward(p_bank, *w_bank, *a_bank, obs_b, *s_bank, stream);

        PrecisionTensor p_logstd = {};
        DecoderWeights* dw = (DecoderWeights*)w_bank->decoder;
        if (dw->continuous) {
            p_logstd = dw->logstd;
        }

        // Offset RNG by bank_off so banks don't collide on per-buffer rng slots.
        sample_logits<<<grid_size(bank_size), BLOCK_SIZE, 0, stream>>>(
            dec_puf, p_logstd, pufferl->act_sizes_puf,
            act_b.data, lp_b.data, val_b.data,
            pufferl->rng_states[buf] + bank_off,
            mask_b.data, mask_stride_b);

        cast<<<grid_size(numel(act_b.shape)), BLOCK_SIZE, 0, stream>>>(
                env.actions.data + (long)sub_start * act_cols,
                act_b.data, numel(act_b.shape));
    }

    if (capturing) {
        cudaGraph_t _graph;
        assert(cudaStreamEndCapture(current_stream, &_graph) == cudaSuccess
                && "cudaStreamEndCapture failed");
        assert(cudaGraphInstantiate(&pufferl->fused_rollout_cudagraphs[graph], _graph, 0) == cudaSuccess
                && "cudaGraphInstantiate failed");
        assert(cudaGraphDestroy(_graph) == cudaSuccess && "cudaGraphDestroy failed");
        cudaDeviceSynchronize();
    }
    profile_end(hypers.profile);
}


__device__ __forceinline__ float load_logit_masked(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_stride_a, int logits_offset, int a,
        const precision_t* __restrict__ mask, int mask_base) {
    float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
    if (mask != nullptr) {
        float m = to_float(mask[mask_base + logits_offset + a]);
        if (m == 0.0f) {
            l = -INFINITY;
            return l;
        }
    }
    return l;
}

__device__ __forceinline__ void ppo_discrete_head(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_stride_a, int logits_offset, int A, int act,
        const precision_t* __restrict__ mask, int mask_base,
        float* out_logsumexp, float* out_entropy, float* out_logp) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    float act_logit = 0.0f;

    for (int a = 0; a < A; ++a) {
        if (mask != nullptr
                && to_float(mask[mask_base + logits_offset + a]) == 0.0f) {
            continue;
        }
        float l = load_logit_masked(logits, logits_base, logits_stride_a, logits_offset, a, mask, mask_base);
        if (a == act) {
            act_logit = l;
        }
        if (l > max_logit) {
            sum *= __expf(max_logit - l);
            max_logit = l;
        }
        sum += __expf(l - max_logit);
    }
    float logsumexp = max_logit + __logf(sum);

    float ent = 0.0f;
    for (int a = 0; a < A; ++a) {
        if (mask != nullptr
                && to_float(mask[mask_base + logits_offset + a]) == 0.0f) {
            continue;
        }
        float l = load_logit_masked(logits, logits_base, logits_stride_a, logits_offset, a, mask, mask_base);
        float logp = l - logsumexp;
        float p = __expf(logp);
        ent -= p * logp;
    }

    *out_logsumexp = logsumexp;
    *out_entropy = ent;
    *out_logp = act_logit - logsumexp;
}

__device__ __forceinline__ void ppo_continuous_head(
        float mean, float log_std, float action,
        float* out_logp, float* out_entropy) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = __expf(log_std);
    float normalized = (action - mean) / std;
    *out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    *out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

__device__ __forceinline__ int ppo_head_offset(const int* act_sizes, int head) {
    int offset = 0;
    for (int h = 0; h < head; ++h) offset += act_sizes[h];
    return offset;
}

__device__ __forceinline__ int ppo_head_legal_count(
        PPOKernelArgs a, int mask_base, int head) {
    int A = a.act_sizes[head];
    if (a.action_mask == nullptr) return A;
    int offset = ppo_head_offset(a.act_sizes, head);
    int count = 0;
    for (int action = 0; action < A; ++action) {
        count += to_float(a.action_mask[mask_base + offset + action]) != 0.0f;
    }
    return count;
}

__device__ __forceinline__ int ppo_head_used(
        PPOKernelArgs a, PPOGraphArgs g, int nt, int head) {
    if (a.head_selector != nullptr) {
        int selector = a.head_selector[head];
        int selector_action =
            static_cast<int>(to_float(g.actions[nt * a.num_atns + selector]));
        int begin = a.head_used_offsets[head];
        int end = a.head_used_offsets[head + 1];
        if (selector_action < 0 || begin + selector_action >= end) return 0;
        return a.head_used_values[begin + selector_action] != 0;
    }
    if (a.head_consume != nullptr && head != 0) {
        int verb = static_cast<int>(to_float(g.actions[nt * a.num_atns]));
        return a.head_consume[verb * a.hc_stride + head] != 0;
    }
    return 1;
}

__device__ __forceinline__ int ppo_head_active(
        PPOKernelArgs a, PPOGraphArgs g, int nt, int mask_base, int head) {
    return ppo_head_used(a, g, nt, head)
        && ppo_head_legal_count(a, mask_base, head) > 1;
}

__device__ __forceinline__ void ppo_unit_loss_grad(
        float logratio, float wa, float clip_coef,
        float* loss, float* grad_logratio) {
    float ratio = __expf(fmaxf(-20.0f, fminf(20.0f, logratio)));
    float ratio_clipped =
        fmaxf(1.0f - clip_coef, fminf(1.0f + clip_coef, ratio));
    float loss_unclipped = wa * ratio;
    float loss_clipped = wa * ratio_clipped;
    *loss = fmaxf(loss_unclipped, loss_clipped);
    float grad = wa * ratio;
    if (loss_clipped > loss_unclipped
            && (ratio <= 1.0f - clip_coef || ratio >= 1.0f + clip_coef)) {
        grad = 0.0f;
    }
    *grad_logratio = grad;
}

__global__ void ppo_loss_compute(
        float* __restrict__ ppo_partials,
        PPOKernelArgs a, PPOGraphArgs g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_elements = a.N * a.T_seq;
    float trainable_segments = fmaxf(1.0f, a.adv_var[2]);
    float inv_trainable = 1.0f / (trainable_segments * float(a.T_seq));

    __shared__ float block_losses[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    if (idx >= total_elements) {
        goto reduce;
    }

    {
    int n = idx / a.T_seq;
    int t = idx % a.T_seq;
    int nt = n * a.T_seq + t;

    int logits_base = n * a.logits_stride_n + t * a.logits_stride_t;
    int values_idx = n * a.values_stride_n + t * a.values_stride_t;
    int grad_logits_base = nt * a.A_total;

    // Shared computation (used by both forward and backward)

    float adv = to_float(g.advantages[nt]);
    float w = to_float(g.prio[n]);
    float val = to_float(g.values[nt]);
    float ret = to_float(g.returns[nt]);
    float val_pred = to_float(a.values_pred[values_idx]);
    g.out_newvalue[nt] = from_float(val_pred);

    bool trainable = to_float(g.trainable[n]) != 0.0f;
    float adv_std = sqrtf(float(a.adv_var[0]));
    float adv_normalized = (adv - float(a.adv_mean[0])) / (adv_std + 1e-8f);

    float dL = inv_trainable;
    float ent_coef = *a.ent_coef;
    float d_entropy_term = dL * (-ent_coef);

    if (!trainable) {
        g.out_ratio[nt] = from_float(1.0f);
        a.grad_values_pred[nt] = 0.0f;
        for (int h = 0; h < a.A_total; ++h) {
            a.grad_logits[grad_logits_base + h] = 0.0f;
        }
        if (a.is_continuous) {
            for (int h = 0; h < a.num_atns; ++h) {
                a.grad_logstd[nt * a.num_atns + h] = 0.0f;
            }
        }
    } else {

    // Value loss (forward) + value gradient (backward)

    float v_error = val_pred - val;
    float v_clipped = val + fmaxf(-a.vf_clip_coef, fminf(a.vf_clip_coef, v_error));
    float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
    float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);

    // Value gradient
    bool use_clipped_vf = (v_loss_clipped > v_loss_unclipped);
    float d_val_pred = 0.0f;
    if (use_clipped_vf) {
        if (v_error >= -a.vf_clip_coef && v_error <= a.vf_clip_coef) {
            d_val_pred = v_clipped - ret;
        }
    } else {
        d_val_pred = val_pred - ret;
    }
    a.grad_values_pred[nt] = dL * a.vf_coef * d_val_pred;

    // Policy loss + gradients

    float pg_loss = 0.0f;
    float total_entropy = 0.0f;
    float joint_logratio = 0.0f;
    int active_heads = 0;
    int active_groups = 0;

    int mask_base = (a.action_mask != nullptr)
        ? n * a.mask_stride_n + t * a.mask_stride_t : 0;

    if (!a.is_continuous) {
        for (int h = 0; h < a.num_atns; ++h) {
            int logits_offset = ppo_head_offset(a.act_sizes, h);
            int A = a.act_sizes[h];
            int act = static_cast<int>(g.actions[nt * a.num_atns + h]);
            float lse, ent, lp;
            ppo_discrete_head(a.logits, logits_base, a.logits_stride_a, logits_offset, A, act,
                              a.action_mask, mask_base, &lse, &ent, &lp);
            if (!ppo_head_active(a, g, nt, mask_base, h)) continue;
            joint_logratio += lp - to_float(
                g.old_logprobs[nt * a.num_atns + h]);
            total_entropy += ent;
            active_heads++;
            int group = a.head_to_group ? a.head_to_group[h] : h;
            bool first = true;
            for (int prev = 0; prev < h; ++prev) {
                int prev_group = a.head_to_group ? a.head_to_group[prev] : prev;
                if (prev_group == group
                        && ppo_head_active(a, g, nt, mask_base, prev)) {
                    first = false;
                    break;
                }
            }
            if (first) active_groups++;
        }
    } else {
        for (int h = 0; h < a.num_atns; ++h) {
            float mean = safe_continuous_mean(a.logits, logits_base + h * a.logits_stride_a);
            float log_std = safe_continuous_logstd(a.logstd, h);
            float action = finite_or_clamp(float(g.actions[nt * a.num_atns + h]), -1.0e6f, 1.0e6f);
            float lp, ent;
            ppo_continuous_head(mean, log_std, action, &lp, &ent);
            joint_logratio += lp - to_float(
                g.old_logprobs[nt * a.num_atns + h]);
            total_entropy += ent;
            active_heads++;
        }
        active_groups = active_heads;
    }

    float wa = -w * adv_normalized;
    float joint_ratio = __expf(fmaxf(-20.0f, fminf(20.0f, joint_logratio)));
    g.out_ratio[nt] = from_float(joint_ratio);
    float optimized_kl = 0.0f;
    float optimized_clip = 0.0f;
    int num_units = 0;

    if (a.ppo_clip_mode == PPO_CLIP_JOINT) {
        float unit_grad;
        ppo_unit_loss_grad(
            joint_logratio, wa, a.clip_coef, &pg_loss, &unit_grad);
        optimized_kl = (joint_ratio - 1.0f) - joint_logratio;
        optimized_clip = fabsf(joint_ratio - 1.0f) > a.clip_coef;
        num_units = 1;
    } else if (a.ppo_clip_mode == PPO_CLIP_PER_HEAD) {
        for (int h = 0; h < a.num_atns; ++h) {
            if (!ppo_head_active(a, g, nt, mask_base, h)) continue;
            int offset = ppo_head_offset(a.act_sizes, h);
            int act = static_cast<int>(g.actions[nt * a.num_atns + h]);
            float lse, ent, lp;
            ppo_discrete_head(a.logits, logits_base, a.logits_stride_a,
                offset, a.act_sizes[h], act, a.action_mask, mask_base,
                &lse, &ent, &lp);
            float lr = lp - to_float(g.old_logprobs[nt * a.num_atns + h]);
            float unit_loss, unit_grad;
            ppo_unit_loss_grad(lr, wa, a.clip_coef, &unit_loss, &unit_grad);
            float ratio = __expf(fmaxf(-20.0f, fminf(20.0f, lr)));
            pg_loss += unit_loss;
            optimized_kl += (ratio - 1.0f) - lr;
            optimized_clip += fabsf(ratio - 1.0f) > a.clip_coef;
            num_units++;
        }
    } else {
        for (int leader = 0; leader < a.num_atns; ++leader) {
            if (!ppo_head_active(a, g, nt, mask_base, leader)) continue;
            int group = a.head_to_group[leader];
            bool first = true;
            for (int prev = 0; prev < leader; ++prev) {
                if (a.head_to_group[prev] == group
                        && ppo_head_active(a, g, nt, mask_base, prev)) {
                    first = false;
                    break;
                }
            }
            if (!first) continue;
            float lr = 0.0f;
            int members = 0;
            for (int h = leader; h < a.num_atns; ++h) {
                if (a.head_to_group[h] != group
                        || !ppo_head_active(a, g, nt, mask_base, h)) continue;
                int offset = ppo_head_offset(a.act_sizes, h);
                int act = static_cast<int>(g.actions[nt * a.num_atns + h]);
                float lse, ent, lp;
                ppo_discrete_head(a.logits, logits_base, a.logits_stride_a,
                    offset, a.act_sizes[h], act, a.action_mask, mask_base,
                    &lse, &ent, &lp);
                lr += lp - to_float(g.old_logprobs[nt * a.num_atns + h]);
                members++;
            }
            if (a.ppo_clip_mode == PPO_CLIP_GROUP_MEAN) lr /= members;
            float unit_loss, unit_grad;
            ppo_unit_loss_grad(lr, wa, a.clip_coef, &unit_loss, &unit_grad);
            float ratio = __expf(fmaxf(-20.0f, fminf(20.0f, lr)));
            pg_loss += unit_loss;
            optimized_kl += (ratio - 1.0f) - lr;
            optimized_clip += fabsf(ratio - 1.0f) > a.clip_coef;
            num_units++;
        }
    }
    if (num_units > 0) {
        pg_loss /= num_units;
        optimized_kl /= num_units;
        optimized_clip /= num_units;
    }

    if (!a.is_continuous) {
        for (int h = 0; h < a.num_atns; ++h) {
            int logits_offset = ppo_head_offset(a.act_sizes, h);
            int A = a.act_sizes[h];
            if (!ppo_head_active(a, g, nt, mask_base, h)) {
                for (int j = 0; j < A; ++j)
                    a.grad_logits[grad_logits_base + logits_offset + j] = 0.0f;
                continue;
            }
            int act = static_cast<int>(g.actions[nt * a.num_atns + h]);
            float logsumexp, ent, lp;
            ppo_discrete_head(a.logits, logits_base, a.logits_stride_a,
                logits_offset, A, act, a.action_mask, mask_base,
                &logsumexp, &ent, &lp);

            float unit_lr = joint_logratio;
            int group_members = 1;
            if (a.ppo_clip_mode == PPO_CLIP_PER_HEAD) {
                unit_lr = lp - to_float(g.old_logprobs[nt * a.num_atns + h]);
            } else if (a.ppo_clip_mode == PPO_CLIP_GROUP_SUM
                    || a.ppo_clip_mode == PPO_CLIP_GROUP_MEAN) {
                unit_lr = 0.0f;
                group_members = 0;
                int group = a.head_to_group[h];
                for (int member = 0; member < a.num_atns; ++member) {
                    if (a.head_to_group[member] != group
                            || !ppo_head_active(a, g, nt, mask_base, member)) continue;
                    int member_offset = ppo_head_offset(a.act_sizes, member);
                    int member_act = static_cast<int>(
                        g.actions[nt * a.num_atns + member]);
                    float member_lse, member_ent, member_lp;
                    ppo_discrete_head(a.logits, logits_base, a.logits_stride_a,
                        member_offset, a.act_sizes[member], member_act,
                        a.action_mask, mask_base,
                        &member_lse, &member_ent, &member_lp);
                    unit_lr += member_lp - to_float(
                        g.old_logprobs[nt * a.num_atns + member]);
                    group_members++;
                }
                if (a.ppo_clip_mode == PPO_CLIP_GROUP_MEAN) {
                    unit_lr /= group_members;
                }
            }
            float unit_loss, d_new_logp;
            ppo_unit_loss_grad(
                unit_lr, wa, a.clip_coef, &unit_loss, &d_new_logp);
            d_new_logp *= dL / fmaxf(1, num_units);
            if (a.ppo_clip_mode == PPO_CLIP_GROUP_MEAN) {
                d_new_logp /= group_members;
            }

            for (int j = 0; j < A; ++j) {
                if (a.action_mask != nullptr
                        && to_float(a.action_mask[
                            mask_base + logits_offset + j]) == 0.0f) {
                    a.grad_logits[grad_logits_base + logits_offset + j] = 0.0f;
                    continue;
                }
                float l = load_logit_masked(a.logits, logits_base, a.logits_stride_a,
                                            logits_offset, j, a.action_mask, mask_base);
                float logp = l - logsumexp;
                float p = __expf(logp);
                float d_logit = (j == act) ? d_new_logp : 0.0f;
                d_logit -= p * d_new_logp;
                d_logit += d_entropy_term * p * (-ent - logp);
                a.grad_logits[grad_logits_base + logits_offset + j] = d_logit;
            }
        }
    } else {
        float unit_loss, d_new_logp;
        ppo_unit_loss_grad(
            joint_logratio, wa, a.clip_coef, &unit_loss, &d_new_logp);
        d_new_logp *= dL;
        for (int h = 0; h < a.num_atns; ++h) {
            float mean = safe_continuous_mean(a.logits, logits_base + h * a.logits_stride_a);
            float log_std = safe_continuous_logstd(a.logstd, h);
            float std = __expf(log_std);
            float var = std * std;
            float action = finite_or_clamp(float(g.actions[nt * a.num_atns + h]), -1.0e6f, 1.0e6f);
            float diff = action - mean;

            a.grad_logits[grad_logits_base + h] = d_new_logp * diff / var;
            a.grad_logstd[nt * a.num_atns + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
        }
    }

    // Forward: loss partials
    float thread_loss = (pg_loss + a.vf_coef * v_loss - ent_coef * total_entropy) * inv_trainable;
    block_losses[LOSS_PG][tid] = pg_loss * inv_trainable;
    block_losses[LOSS_VF][tid] = v_loss * inv_trainable;
    block_losses[LOSS_ENT][tid] = total_entropy * inv_trainable;
    block_losses[LOSS_TOTAL][tid] = thread_loss;
    block_losses[LOSS_OLD_APPROX_KL][tid] = (-joint_logratio) * inv_trainable;
    block_losses[LOSS_APPROX_KL][tid] = ((joint_ratio - 1.0f) - joint_logratio) * inv_trainable;
    block_losses[LOSS_CLIPFRAC][tid] = (fabsf(joint_ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_trainable;
    block_losses[LOSS_PPO_ACTIVE_HEADS][tid] = active_heads * inv_trainable;
    block_losses[LOSS_PPO_ACTIVE_GROUPS][tid] = active_groups * inv_trainable;
    block_losses[LOSS_PPO_OPTIMIZED_KL][tid] = optimized_kl * inv_trainable;
    block_losses[LOSS_PPO_OPTIMIZED_CLIPFRAC][tid] = optimized_clip * inv_trainable;
    block_losses[LOSS_PPO_OPTIMIZED_ENTROPY][tid] =
        (active_heads > 0 ? total_entropy / active_heads : 0.0f) * inv_trainable;
    block_losses[LOSS_PPO_JOINT_OLD_KL][tid] = (-joint_logratio) * inv_trainable;
    block_losses[LOSS_PPO_JOINT_KL][tid] =
        ((joint_ratio - 1.0f) - joint_logratio) * inv_trainable;
    block_losses[LOSS_PPO_JOINT_CLIPFRAC][tid] =
        (fabsf(joint_ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_trainable;
    }
    } // end if (idx < total_elements)

// Deterministic aggregation
reduce:
    __syncthreads();

    for (int stride = PPO_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int c = 0; c < LOSS_N; c++) {
                block_losses[c][tid] += block_losses[c][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = blockIdx.x * (LOSS_N + 1);
        ppo_partials[base] = block_losses[LOSS_TOTAL][0];
        for (int c = 0; c < LOSS_N; c++) {
            ppo_partials[base + 1 + c] = block_losses[c][0];
        }
    }
}

// Deterministic reduction of per-block PPO loss partials + count increment
__global__ void ppo_loss_reduce(
        float* __restrict__ loss,
        float* __restrict__ losses_acc,
        const float* __restrict__ partials,
        int num_blocks) {
    int tid = threadIdx.x;
    if (tid > LOSS_N) {
        return;
    }

    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        sum += partials[b * (LOSS_N + 1) + tid];
    }

    if (tid == 0) {
        *loss += sum;
    } else {
        losses_acc[tid - 1] += sum;
    }

    // Fold add_scalar: increment epoch count
    if (tid == 0) {
        losses_acc[LOSS_N] += 1.0f;
    }
}

__global__ void ppo_var_mean(const precision_t* __restrict__ src,
        const precision_t* __restrict__ trainable,
        float* __restrict__ var_out, float* __restrict__ mean_out,
        float* __restrict__ count_out, int N, int T) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    int trainable_segments = 0;
    for (int n = 0; n < N; ++n) {
        if (to_float(trainable[n]) == 0.0f) continue;
        trainable_segments++;
        for (int t = tid; t < T; t += blockDim.x) {
            sum += to_float(src[n * T + t]);
        }
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float count = float(trainable_segments * T);
    float mean = count > 0 ? sdata[0] / count : 0.0f;
    if (tid == 0) {
        *mean_out = mean;
        *count_out = trainable_segments;
    }
    __syncthreads();
    float ss = 0.0f;
    for (int n = 0; n < N; ++n) {
        if (to_float(trainable[n]) == 0.0f) continue;
        for (int t = tid; t < T; t += blockDim.x) {
            float d = to_float(src[n * T + t]) - mean;
            ss += d * d;
        }
    }
    sdata[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *var_out = count > 1 ? sdata[0] / (count - 1.0f) : 0.0f;
    }
}

// This is a huge kernel for a relatively cheap operation. But without this,
// it's death by a thousand cuts with repeated kernel launches. Even graphed, you
// blow up the memory bandwidth.
void ppo_loss_fwd_bwd(
        PrecisionTensor& dec_out,    // (N, T, fused_cols) — fused logits+value from decoder
        PrecisionTensor& logstd,     // continuous logstd or empty
        TrainGraph& graph,
        IntTensor& act_sizes, FloatTensor& losses_acc,
        float clip_coef, float vf_clip_coef, float vf_coef, const float* ent_coef,
        int ppo_clip_mode, int num_groups,
        const int* head_to_group, const int* head_selector,
        const int* head_used_offsets, const unsigned char* head_used_values,
        PPOBuffersPuf& bufs, bool is_continuous,
        cudaStream_t stream) {
    int N = dec_out.shape[0], T = dec_out.shape[1], fused_cols = dec_out.shape[2];
    int A_total = fused_cols - 1;  // last column is value
    int total = N * T;

    // Pointers into fused decoder output
    const precision_t* logits_ptr = dec_out.data;

    float* adv_var_ptr = bufs.adv_scratch.data;
    float* adv_mean_ptr = adv_var_ptr + 1;
    float* trainable_count_ptr = adv_var_ptr + 2;
    ppo_var_mean<<<1, 256, 0, stream>>>(
        graph.mb_advantages.data, graph.mb_trainable.data,
        adv_var_ptr, adv_mean_ptr, trainable_count_ptr, N, T);

    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;

    static float* ppo_partials_buf = nullptr;
    static int ppo_partials_capacity = 0;
    int ppo_partials_needed = ppo_grid * (LOSS_N + 1);
    if (!ppo_partials_buf || ppo_partials_needed > ppo_partials_capacity) {
        if (ppo_partials_buf) cudaFree(ppo_partials_buf);
        ppo_partials_capacity = ppo_partials_needed;
        cudaMalloc(&ppo_partials_buf, ppo_partials_capacity * sizeof(float));
    }

    cudaMemsetAsync(bufs.loss_output.data, 0, sizeof(float), stream);

    PPOGraphArgs graph_args = {
        .out_ratio = graph.mb_ratio.data,
        .out_newvalue = graph.mb_newvalue.data,
        .actions = graph.mb_actions.data,
        .old_logprobs = graph.mb_logprobs.data,
        .advantages = graph.mb_advantages.data,
        .prio = graph.mb_prio.data,
        .values = graph.mb_values.data,
        .returns = graph.mb_returns.data,
        .trainable = graph.mb_trainable.data,
    };

    bool has_mask = (graph.mb_action_mask.data != nullptr);
            int hc_stride_l = 0;
        const signed char* hc_dev_l = get_head_consume_dev(&hc_stride_l);
PPOKernelArgs args = {
        .grad_logits = bufs.grad_logits.data,
        .grad_logstd = is_continuous ? bufs.grad_logstd.data : nullptr,
        .grad_values_pred = bufs.grad_values.data,
        .logits = logits_ptr,
        .logstd = is_continuous ? logstd.data : nullptr,
        .values_pred = logits_ptr + A_total,
        .adv_mean = adv_mean_ptr,
        .adv_var = adv_var_ptr,
        .act_sizes = act_sizes.data,
        .action_mask = has_mask ? graph.mb_action_mask.data : nullptr,
        .mask_stride_n = has_mask ? T * A_total : 0,
        .mask_stride_t = has_mask ? A_total : 0,
        .head_consume = hc_dev_l,
        .hc_stride = hc_stride_l,
        .head_to_group = head_to_group,
        .head_selector = head_selector,
        .head_used_offsets = head_used_offsets,
        .head_used_values = head_used_values,
        .num_groups = num_groups,
        .ppo_clip_mode = ppo_clip_mode,
        .num_atns = (int)numel(act_sizes.shape),
        .clip_coef = clip_coef, .vf_clip_coef = vf_clip_coef,
        .vf_coef = vf_coef, .ent_coef = ent_coef,
        .T_seq = T, .A_total = A_total, .N = N,
        .logits_stride_n = T * fused_cols, .logits_stride_t = fused_cols, .logits_stride_a = 1,
        .values_stride_n = T * fused_cols, .values_stride_t = fused_cols,
        .is_continuous = is_continuous,
    };

    ppo_loss_compute<<<ppo_grid, PPO_THREADS, 0, stream>>>(ppo_partials_buf, args, graph_args);

    ppo_loss_reduce<<<1, LOSS_N + 1, 0, stream>>>(
        bufs.loss_output.data, losses_acc.data, ppo_partials_buf, ppo_grid);
}

#define PRIO_WARP_SIZE 32
#define PRIO_FULL_MASK 0xffffffff
#define PRIO_BLOCK_SIZE 256
#define PRIO_NUM_WARPS (PRIO_BLOCK_SIZE / PRIO_WARP_SIZE)
__global__ void compute_prio_adv_reduction(
        const precision_t* __restrict__ advantages,
        float* prio_weights, float prio_alpha, int stride) {
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
        float pw = __powf(local_sum, prio_alpha);
        if (isnan(pw) || isinf(pw)) {
            pw = 0.0f;
        }
        prio_weights[row] = pw;
    }
}

__global__ void compute_prio_normalize(float* prio_weights, int length) {
    __shared__ float shmem[PRIO_NUM_WARPS];
    __shared__ float block_sum;

    int tx = threadIdx.x;
    int lane = tx % PRIO_WARP_SIZE;
    int warp_id = tx / PRIO_WARP_SIZE;
    const float eps = 1e-6f;

    float local_sum = 0.0f;
    for (int t = tx; t < length; t += blockDim.x) {
        local_sum += prio_weights[t];
    }
    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (lane == 0) {
        shmem[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < PRIO_NUM_WARPS) ? shmem[lane] : 0.0f;
        for (int s = PRIO_NUM_WARPS / 2; s >= 1; s /= 2) {
            val += __shfl_down_sync(PRIO_FULL_MASK, val, s);
        }
        if (tx == 0) {
            block_sum = val + eps;
        }
    }
    __syncthreads();

    for (int t = tx; t < length; t += blockDim.x) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
}

// mb_prio[i] = pow(total_agents * prio_probs[idx[i]], -anneal_beta)
__global__ void compute_prio_imp_weights(
        const int* __restrict__ indices,
        const float* __restrict__ prio_probs,
        float* mb_prio, int total_agents,
        float anneal_beta, int minibatch_segments) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx < minibatch_segments) {
        float value = prio_probs[indices[tx]] * (float)total_agents;
        mb_prio[tx] = __powf(value, -anneal_beta);
    }
}

__global__ void build_cdf(
    float* __restrict__ cdf, const float* __restrict__ probs, int B) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float cum = 0.0f;
        for (int i = 0; i < B; i++) {
            cum += probs[i];
            cdf[i] = cum;
        }
    }
}

__global__ void advance_rng_offset(int64_t* __restrict__ offset_ptr, int64_t delta) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *offset_ptr += delta;
    }
}

// Multinomial with replacement (uses cuRAND)
__global__ void multinomial_sample(int* __restrict__ out_idx, const float* __restrict__ cdf,
        int B, int num_samples, uint64_t seed, const int64_t* __restrict__ offset_ptr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;

    uint64_t base_off = (uint64_t)(*offset_ptr);
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, base_off + tid, 0, &rng_state);
    float u = curand_uniform(&rng_state);

    int lo = 0, hi = B - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    out_idx[tid] = lo;
}

// Prioritize high absolute advantage trajectories
// This is a form of implicit curriculum learning
// It is a major improvement in some complex environments
// The values of alpha and beta found by sweeps will tell you
// whether it is important for your task
void prio_replay_cuda(PrecisionTensor& advantages, float prio_alpha,
        int minibatch_segments, int total_agents, float anneal_beta,
        PrioBuffers& bufs, ulong seed, long* offset_ptr, cudaStream_t stream) {
    int B = advantages.shape[0], T = advantages.shape[1];
    compute_prio_adv_reduction<<<B, PRIO_WARP_SIZE, 0, stream>>>(
        advantages.data, bufs.prio_probs.data, prio_alpha, T);
    compute_prio_normalize<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.prio_probs.data, B);
    //int block = fmaxf(((minibatch_segments + 31) / 32) * 32, 32);
    build_cdf<<<1, 1, 0, stream>>>(bufs.cdf.data, bufs.prio_probs.data, B);
    int threads = 256;
    int blocks = (minibatch_segments + threads - 1) / threads;
    multinomial_sample<<<blocks, threads, 0, stream>>>(
        bufs.idx.data, bufs.cdf.data, B, minibatch_segments, seed, offset_ptr);
    advance_rng_offset<<<1, 1, 0, stream>>>(offset_ptr, (int64_t)minibatch_segments);

    int p3_blocks = (minibatch_segments + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    compute_prio_imp_weights<<<p3_blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.idx.data, bufs.prio_probs.data,
        bufs.mb_prio.data, total_agents, anneal_beta, minibatch_segments);
}

// Experience the puffer advantage! Generalized advantage estimation + V-Trace
// importance sampling correction in a single streamlined operation
__device__ void puff_advantage_row_scalar(
        const precision_t* values, const precision_t* rewards, const precision_t* dones,
        const precision_t* importance, precision_t* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - to_float(dones[t_next]);
        float imp = to_float(importance[t]);
        float rho_t = fminf(imp, rho_clip);
        float c_t = fminf(imp, c_clip);
        float r_nxt = to_float(rewards[t_next]);
        float v = to_float(values[t]);
        float v_nxt = to_float(values[t_next]);
        float delta = rho_t*r_nxt + gamma*v_nxt*nextnonterminal - v;
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = from_float(lastpufferlam);
    }
}

// These loading fns just optimize bandwidth for advantage since we call it on all
// the data every minibatch. This should change in 5.0
__device__ __forceinline__ void adv_vec_load(const float* ptr, float* out) {
    float4 v = *reinterpret_cast<const float4*>(ptr);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

__device__ __forceinline__ void adv_vec_load(const __nv_bfloat16* ptr, float* out) {
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = __bfloat162float(bf[i]);
    }
}

// Store N floats as precision_t via 128-bit writes (float4 for f32, uint4 for bf16)
__device__ __forceinline__ void adv_vec_store(float* ptr, const float* vals) {
    *reinterpret_cast<float4*>(ptr) = make_float4(vals[0], vals[1], vals[2], vals[3]);
}

__device__ __forceinline__ void adv_vec_store(__nv_bfloat16* ptr, const float* vals) {
    // N=8 for bf16: all 8 elements fit in one uint4 (128 bits)
    __nv_bfloat16 tmp[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) tmp[i] = __float2bfloat16(vals[i]);
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(tmp);
}

__device__ __forceinline__ void puff_advantage_row_vec(
        const precision_t* values, const precision_t* rewards, const precision_t* dones,
        const precision_t* importance, precision_t* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    constexpr int N = 16 / sizeof(precision_t);

    float lastpufferlam = 0.0f;
    int num_chunks = horizon / N;

    float next_value = to_float(values[horizon - 1]);
    float next_done = to_float(dones[horizon - 1]);
    float next_reward = to_float(rewards[horizon - 1]);

    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int base = chunk * N;

        float v[N], r[N], d[N], imp[N];
        adv_vec_load(values + base, v);
        adv_vec_load(rewards + base, r);
        adv_vec_load(dones + base, d);
        adv_vec_load(importance + base, imp);

        float adv[N] = {0};
        int start_idx = (chunk == num_chunks - 1) ? (N - 2) : (N - 1);

        #pragma unroll
        for (int i = start_idx; i >= 0; i--) {
            float nextnonterminal = 1.0f - next_done;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_reward + gamma * next_value * nextnonterminal - v[i]);
            lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
            adv[i] = lastpufferlam;
            next_value = v[i];
            next_done = d[i];
            next_reward = r[i];
        }

        adv_vec_store(advantages + base, adv);
    }
}

__global__ void puff_advantage(const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, precision_t* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_vec(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

__global__ void puff_advantage_scalar(const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, precision_t* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_scalar(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

void puff_advantage_cuda(PrecisionTensor& values, PrecisionTensor& rewards,
        PrecisionTensor& dones, PrecisionTensor& importance, PrecisionTensor& advantages,
        float gamma, float lambda, float rho_clip, float c_clip, cudaStream_t stream) {
    int num_steps = values.shape[0], horizon = values.shape[1];
    int blocks = grid_size(num_steps);
    constexpr int N = 16 / sizeof(precision_t);
    auto kernel = (horizon % N == 0) ? puff_advantage : puff_advantage_scalar;
    kernel<<<blocks, 256, 0, stream>>>(
        values.data, rewards.data, dones.data, importance.data,
        advantages.data, gamma, lambda, rho_clip, c_clip, num_steps, horizon);
}

// Zero advantages on frozen-bank rows so prio_replay never samples them. Frozen
// rollout rows hold actions/logprobs from the frozen policy — training the
// primary's PPO on them produces garbage ratios and poisoned gradients.
__global__ void zero_frozen_advantages_kernel(precision_t* advantages,
        int agents_per_buffer, int primary_per_buffer, int total_rows, int horizon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_rows * horizon;
    if (idx >= total) return;
    int row = idx / horizon;
    int rel = row % agents_per_buffer;
    if (rel >= primary_per_buffer) {
        advantages[idx] = from_float(0.0f);
    }
}

void zero_frozen_advantages_cuda(PrecisionTensor& advantages,
        int agents_per_buffer, int primary_per_buffer, cudaStream_t stream) {
    int total_rows = advantages.shape[0];
    int horizon = advantages.shape[1];
    int total = total_rows * horizon;
    zero_frozen_advantages_kernel<<<grid_size(total), BLOCK_SIZE, 0, stream>>>(
        advantages.data, agents_per_buffer, primary_per_buffer, total_rows, horizon);
}

// Minor copy bandwidth optimizations
__global__ void index_copy(char* __restrict__ dst, const int* __restrict__ idx,
        const char* __restrict__ src, int num_idx, int row_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_idx) {
        int dst_row = idx[i];
        memcpy(dst + (int64_t)dst_row * row_bytes, src + (int64_t)i * row_bytes, row_bytes);
    }
}

__device__ __forceinline__ void copy_values_adv_returns(
        const precision_t* __restrict__ src_values, precision_t* __restrict__ dst_values,
        const precision_t* __restrict__ src_advantages, precision_t* __restrict__ dst_advantages,
        precision_t* __restrict__ dst_returns,
        int src_row, int dst_row, int horizon) {
    int srh = (int64_t)src_row * horizon;
    int drh = (int64_t)dst_row * horizon;
    const precision_t* s_values = src_values + srh;
    const precision_t* s_adv = src_advantages + srh;
    precision_t* d_values = dst_values + drh;
    precision_t* d_adv = dst_advantages + drh;
    precision_t* d_returns = dst_returns + drh;
    for (int i = threadIdx.x; i < horizon; i += blockDim.x) {
        precision_t val = s_values[i];
        precision_t adv = s_adv[i];
        d_values[i] = val;
        d_adv[i] = adv;
        d_returns[i] = from_float(to_float(val) + to_float(adv));
    }
}

__global__ void select_copy(RolloutBuf rollouts, TrainGraph graph,
        const int* __restrict__ idx, const precision_t* __restrict__ advantages,
        const float* __restrict__ mb_prio,
        int agents_per_buffer, int primary_per_buffer) {
    int mb = blockIdx.x;
    int ch = blockIdx.y;
    int src_row = idx[mb];

    // Compute row byte counts from tensor shapes
    int obs_row_bytes = (numel(rollouts.observations.shape) / rollouts.observations.shape[0]) * sizeof(precision_t);
    int act_row_bytes = (numel(rollouts.actions.shape) / rollouts.actions.shape[0]) * sizeof(precision_t);
    int lp_row_bytes = (numel(rollouts.logprobs.shape) / rollouts.logprobs.shape[0]) * sizeof(precision_t);
    int horizon = rollouts.values.shape[1];

    switch (ch) {
    case 0:
        copy_bytes((const char*)rollouts.observations.data, (char*)graph.mb_obs.data, src_row, mb, obs_row_bytes);
        break;
    case 1:
        copy_bytes((const char*)rollouts.actions.data, (char*)graph.mb_actions.data, src_row, mb, act_row_bytes);
        break;
    case 2:
        copy_bytes((const char*)rollouts.logprobs.data, (char*)graph.mb_logprobs.data, src_row, mb, lp_row_bytes);
        break;
    case 3:
        copy_values_adv_returns(rollouts.values.data, graph.mb_values.data,
                advantages, graph.mb_advantages.data,
                graph.mb_returns.data, src_row, mb, horizon);
        break;
    case 4:
        if (threadIdx.x == 0) {
            graph.mb_prio.data[mb] = from_float(mb_prio[mb]);
        }
        break;
    case 5:
        if (graph.mb_action_mask.data != nullptr) {
            int mask_row_bytes = (numel(rollouts.action_mask.shape)
                / rollouts.action_mask.shape[0]) * sizeof(precision_t);
            copy_bytes((const char*)rollouts.action_mask.data,
                       (char*)graph.mb_action_mask.data, src_row, mb, mask_row_bytes);
        }
        break;
    case 6:
        if (threadIdx.x == 0) {
            int rel = src_row % agents_per_buffer;
            graph.mb_trainable.data[mb] =
                from_float(rel < primary_per_buffer ? 1.0f : 0.0f);
        }
        break;
    }
}

inline float cosine_annealing(float lr_base, float lr_min, long t, long T) {
    if (T == 0) return lr_base;
    float ratio = (double )t / (double) T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

void train_impl(PuffeRL& pufferl) {
    // Update to HypersT& p
    HypersT& hypers = pufferl.hypers;

    cudaEventRecord(pufferl.profile.events[0]);  // pre-loop start
    cudaStream_t train_stream = pufferl.default_stream;

    // Transpose from rollout layout (T, B, ...) to train layout (B, T, ...)
    RolloutBuf& src = pufferl.rollouts;
    RolloutBuf& rollouts = pufferl.train_rollouts;
    PrecisionTensor& advantages_puf = pufferl.advantages_puf;

    int T = src.observations.shape[0], B = src.observations.shape[1];
    int obs_size = (ndim(src.observations.shape) >= 3) ? src.observations.shape[2] : 1;
    int num_atns = (ndim(src.actions.shape) >= 3) ? src.actions.shape[2] : 1;

    transpose_102<<<grid_size(T*B*obs_size), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.observations.data, src.observations.data, T, B, obs_size);
    transpose_102<<<grid_size(T*B*num_atns), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.actions.data, src.actions.data, T, B, num_atns);
    transpose_102<<<grid_size(T*B*num_atns), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.logprobs.data, src.logprobs.data, T, B, num_atns);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.rewards.data, src.rewards.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.terminals.data, src.terminals.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.ratio.data, src.ratio.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.values.data, src.values.data, T, B, 1);
    if (src.action_mask.data != nullptr) {
        int mask_size = src.action_mask.shape[2];
        transpose_102<<<grid_size(T*B*mask_size), BLOCK_SIZE, 0, train_stream>>>(
            rollouts.action_mask.data, src.action_mask.data, T, B, mask_size);
    }

    // We hard-clamp rewards to -1, 1. Our envs are mostly designed to respect this range
    clamp_precision_kernel<<<grid_size(numel(rollouts.rewards.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.rewards.data, -1.0f, 1.0f, numel(rollouts.rewards.shape));

    // Set importance weights to 1.0
    fill_precision_kernel<<<grid_size(numel(rollouts.ratio.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.ratio.data, from_float(1.0f), numel(rollouts.ratio.shape));

    // Inline any of these only used once
    int minibatch_size = hypers.minibatch_size;
    int batch_size = hypers.total_agents * hypers.horizon;
    int minibatch_segments = minibatch_size / hypers.horizon;
    float prio_beta0 = hypers.prio_beta0;
    float prio_alpha = hypers.prio_alpha;
    bool anneal_lr = hypers.anneal_lr;
    int current_epoch = pufferl.epoch;

    Muon* muon = &pufferl.muon;
    int total_epochs = hypers.total_timesteps / batch_size;
    if (anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, total_epochs);
        cudaMemcpy(muon->lr_ptr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Annealed entropy coefficient — same cosine shape as lr. With PG signal
    // alive, the entropy bonus that kept early-training exploratory becomes
    // load-bearing dead weight late in training; cosine-decay frees the policy
    // to commit harder on what it has already learned.
    float current_ent_coef = hypers.ent_coef;
    if (hypers.anneal_ent_coef) {
        float ent_min = hypers.min_ent_coef_ratio * hypers.ent_coef;
        current_ent_coef = cosine_annealing(hypers.ent_coef, ent_min,
                                            current_epoch, total_epochs);
    }
    // copy ent_coef to the device buffer read by the loss kernel
    cudaMemcpy(pufferl.ppo_bufs_puf.ent_coef.data, &current_ent_coef,
               sizeof(float), cudaMemcpyHostToDevice);

    // Annealed priority exponent
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;
    TrainGraph& graph = pufferl.train_buf;
    cudaEventRecord(pufferl.profile.events[1]);  // pre-loop end

    int total_minibatches = hypers.replay_ratio * batch_size / hypers.minibatch_size;
    for (int mb = 0; mb < total_minibatches; ++mb) {
        cudaEventRecord(pufferl.profile.events[2]);  // start of misc (overwritten each iter)
        puf_zero(&advantages_puf, train_stream);

        profile_begin("compute_advantage", hypers.profile);
        puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
            rollouts.ratio, advantages_puf, hypers.gamma, hypers.gae_lambda,
            hypers.vtrace_rho_clip, hypers.vtrace_c_clip, train_stream);
        if (pufferl.num_frozen_banks > 0 && pufferl.bank_layout != NULL) {
            int apb = hypers.total_agents / hypers.num_buffers;
            zero_frozen_advantages_cuda(advantages_puf, apb,
                pufferl.bank_layout[1], train_stream);
        }
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        // Use the training RNG offset slot (last slot, index num_buffers)
        long* train_rng_offset = pufferl.rng_offset_puf.data + hypers.num_buffers;
        prio_replay_cuda(advantages_puf, prio_alpha, minibatch_segments,
            hypers.total_agents, anneal_beta,
            pufferl.prio_bufs, pufferl.seed, train_rng_offset, train_stream);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        if (hypers.reset_state) puf_zero(&graph.mb_state, train_stream);
        {
            RolloutBuf sel_src = rollouts;
            sel_src.values = rollouts.values;
            int mb_segs = pufferl.prio_bufs.idx.shape[0];
            int channels = 7;
            int agents_per_buffer = hypers.total_agents / hypers.num_buffers;
            int primary_per_buffer = pufferl.bank_layout
                ? pufferl.bank_layout[1] : agents_per_buffer;
            select_copy<<<dim3(mb_segs, channels), SELECT_COPY_THREADS, 0, train_stream>>>(
                sel_src, graph, pufferl.prio_bufs.idx.data,
                advantages_puf.data, pufferl.prio_bufs.mb_prio.data,
                agents_per_buffer, primary_per_buffer);
        }
        profile_end(hypers.profile);

        cudaEventRecord(pufferl.profile.events[3]);  // end misc / start forward
        profile_begin("train_forward_backward", hypers.profile);
        if (pufferl.train_captured) {
            cudaGraphLaunch(pufferl.train_cudagraph, train_stream);
        } else {
            bool capturing = pufferl.train_warmup == hypers.cudagraphs;
            if (capturing) {
                assert(cudaStreamBeginCapture(train_stream, cudaStreamCaptureModeGlobal) == cudaSuccess
                        && "cudaStreamBeginCapture failed");
            }

            cudaStream_t stream = train_stream;
            PrecisionTensor obs_puf = graph.mb_obs;
            PrecisionTensor state_puf = graph.mb_state;
            PrecisionTensor dec_puf = policy_forward_train(&pufferl.policy, pufferl.weights, pufferl.train_activations, obs_puf, state_puf, stream);
            DecoderWeights* dw_train = (DecoderWeights*)pufferl.weights.decoder;
            PrecisionTensor p_logstd;
            if (dw_train->continuous) {
                p_logstd = dw_train->logstd;
            }

            ppo_loss_fwd_bwd(dec_puf, p_logstd, graph,
                pufferl.act_sizes_puf, pufferl.losses_puf,
                hypers.clip_coef, hypers.vf_clip_coef, hypers.vf_coef,
                pufferl.ppo_bufs_puf.ent_coef.data,
                hypers.ppo_clip_mode, pufferl.ppo_num_groups,
                pufferl.ppo_head_to_group, pufferl.ppo_head_selector,
                pufferl.ppo_head_used_offsets, pufferl.ppo_head_used_values,
                pufferl.ppo_bufs_puf, pufferl.is_continuous, stream);

            FloatTensor grad_logits_puf = pufferl.ppo_bufs_puf.grad_logits;
            FloatTensor grad_logstd_puf = pufferl.is_continuous ? pufferl.ppo_bufs_puf.grad_logstd : FloatTensor();
            FloatTensor grad_values_puf = pufferl.ppo_bufs_puf.grad_values;
            policy_backward(&pufferl.policy, pufferl.weights, pufferl.train_activations,
                grad_logits_puf, grad_logstd_puf, grad_values_puf, stream);

            muon_step(&pufferl.muon, pufferl.master_weights, pufferl.grad_puf, hypers.max_grad_norm, stream);
            if (USE_BF16) {
                int n = numel(pufferl.param_puf.shape);
                cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                    pufferl.param_puf.data, pufferl.master_weights.data, n);
            }
            if (capturing) {
                cudaGraph_t _graph;
                assert(cudaStreamEndCapture(train_stream, &_graph) == cudaSuccess
                        && "cudaStreamEndCapture failed");
                assert(cudaGraphInstantiate(&pufferl.train_cudagraph, _graph, 0) == cudaSuccess
                        && "cudaGraphInstantiate failed");
                assert(cudaGraphDestroy(_graph) == cudaSuccess && "cudaGraphDestroy failed");
                cudaDeviceSynchronize();
                pufferl.train_captured = true;
            }
            pufferl.train_warmup++;
        }
        profile_end(hypers.profile);

        // This version is consistent with PufferLib 3.0. One of the major algorithmic
        // questions remaining is how and when to update value and advantage estimates.
        {
            int num_idx = numel(pufferl.prio_bufs.idx.shape);
            int row_bytes = (numel(graph.mb_ratio.shape) / graph.mb_ratio.shape[0]) * sizeof(precision_t);
            index_copy<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                (char*)rollouts.ratio.data, pufferl.prio_bufs.idx.data,
                (const char*)graph.mb_ratio.data, num_idx, row_bytes);
        }
        {
            int num_idx = numel(pufferl.prio_bufs.idx.shape);
            int row_bytes = graph.mb_newvalue.shape[1] * sizeof(precision_t);
            index_copy<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                (char*)rollouts.values.data, pufferl.prio_bufs.idx.data,
                (const char*)graph.mb_newvalue.data, num_idx, row_bytes);
        }
        cudaEventRecord(pufferl.profile.events[4]);  // end forward
    }
    pufferl.epoch += 1;

    cudaStreamSynchronize(pufferl.default_stream);

    if (total_minibatches > 0) {
        float ms;
        // Pre-loop setup (transpose, advantage, allocs)
        cudaEventElapsedTime(&ms, pufferl.profile.events[0], pufferl.profile.events[1]);
        pufferl.profile.accum[PROF_TRAIN_MISC] += ms;
        // In-loop misc (last iteration, representative) scaled by count
        cudaEventElapsedTime(&ms, pufferl.profile.events[2], pufferl.profile.events[3]);
        pufferl.profile.accum[PROF_TRAIN_MISC] += ms * total_minibatches;
        // In-loop forward (last iteration, representative) scaled by count
        cudaEventElapsedTime(&ms, pufferl.profile.events[3], pufferl.profile.events[4]);
        pufferl.profile.accum[PROF_TRAIN_FORWARD] += ms * total_minibatches;
    }

}

// Build a Policy value for a given env + arch. Encoder/decoder algorithms are
// fixed by the env; hidden_size/num_layers/horizon parameterize shape. Policy
// has no heap state so this returns by value; callers store it wherever.
static Policy build_policy(const char* env_name, int input_size, int hidden_size,
                           int num_layers, int decoder_output_size, int act_n,
                           bool is_continuous, int horizon) {
    Encoder encoder = {
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
        .create_weights = encoder_create_weights,
        .free_weights = encoder_free_weights,
        .free_activations = encoder_free_activations,
        .in_dim = input_size, .out_dim = hidden_size,
        .activation_size = sizeof(EncoderActivations),
    };
    create_custom_encoder(env_name, &encoder);
    Decoder decoder = {
        .forward = decoder_forward,
        .backward = decoder_backward,
        .init_weights = decoder_init_weights,
        .reg_params = decoder_reg_params,
        .reg_train = decoder_reg_train,
        .reg_rollout = decoder_reg_rollout,
        .create_weights = decoder_create_weights,
        .free_weights = decoder_free_weights,
        .free_activations = decoder_free_activations,
        .hidden_dim = hidden_size, .output_dim = decoder_output_size, .continuous = is_continuous,
        .activation_size = (int)sizeof(DecoderActivations),
    };
    create_custom_decoder(env_name, &decoder);
    Network network = {
        .forward = mingru_forward,
        .forward_train = mingru_forward_train,
        .backward = mingru_backward,
        .init_weights = mingru_init_weights,
        .reg_params = mingru_reg_params,
        .reg_train = mingru_reg_train,
        .reg_rollout = mingru_reg_rollout,
        .create_weights = mingru_create_weights,
        .free_weights = mingru_free_weights,
        .free_activations = mingru_free_activations,
        .hidden = hidden_size, .num_layers = num_layers, .horizon = horizon,
    };
    return Policy{
        .encoder = encoder, .decoder = decoder, .network = network,
        .input_dim = input_size, .hidden_dim = hidden_size, .output_dim = decoder_output_size,
        .num_atns = act_n,
    };
}

// Allocate a fresh frozen WeightBank with its own Policy (may differ in
// hidden_size/num_layers from primary). slice_size = how many agents per buffer
// this bank will own. Weights are uninitialized — caller must load before use.
static void weight_bank_create_for_pufferl(WeightBank* bank, PuffeRL* pufferl,
        int slice_size, int hidden_size, int num_layers) {
    int num_buffers = pufferl->hypers.num_buffers;

    // Rebuild arch-varying Policy from env metadata already on pufferl.
    int input_size = pufferl->env.obs.shape[1];
    int num_action_heads = pufferl->env.actions.shape[1];
    int* raw_act_sizes = get_act_sizes();
    int act_n = 0;
    for (int i = 0; i < num_action_heads; i++) act_n += raw_act_sizes[i];
    int decoder_output_size = pufferl->is_continuous ? num_action_heads : act_n;
    bank->policy = build_policy(pufferl->env_name.c_str(), input_size, hidden_size,
        num_layers, decoder_output_size, act_n, pufferl->is_continuous, pufferl->hypers.horizon);
    bank->hidden_size = hidden_size;
    bank->num_layers = num_layers;

    Allocator* params = &bank->params_alloc;
    Allocator* acts = &bank->acts_alloc;

    bank->slice_size = slice_size;
    bank->weights = policy_weights_create(&bank->policy, params);
    bank->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
    bank->buffer_states = (PrecisionTensor*)calloc(num_buffers, sizeof(PrecisionTensor));
    for (int i = 0; i < num_buffers; i++) {
        bank->buffer_activations[i] = policy_reg_rollout(&bank->policy, bank->weights, acts, slice_size);
        bank->buffer_states[i] = {.shape = {num_layers, slice_size, hidden_size}};
        alloc_register(acts, &bank->buffer_states[i]);
    }

    alloc_create(params);
    alloc_create(acts);

    bank->param_puf = {.data = (precision_t*)params->mem, .shape = {params->total_elems}};
    if (USE_BF16) {
        bank->master_weights = {.shape = {params->total_elems}};
        cudaMalloc(&bank->master_weights.data, params->total_elems * sizeof(float));
    } else {
        bank->master_weights = {.data = (float*)bank->param_puf.data, .shape = {params->total_elems}};
    }
}

// Mirror of weight_bank_create_for_pufferl. Frees the bank's weights, per-buffer
// activations, allocators, and master_weights (BF16 only). Does not free the
// WeightBank struct itself — caller owns that.
static void weight_bank_destroy(WeightBank* bank, PuffeRL* pufferl) {
    int num_buffers = pufferl->hypers.num_buffers;
    policy_weights_free(&bank->policy, &bank->weights);
    if (bank->buffer_activations != NULL) {
        for (int i = 0; i < num_buffers; i++) {
            policy_activations_free(&bank->policy, bank->buffer_activations[i]);
        }
        free(bank->buffer_activations);
    }
    free(bank->buffer_states);
    alloc_free(&bank->params_alloc);
    alloc_free(&bank->acts_alloc);
    if (USE_BF16 && bank->master_weights.data != NULL) {
        cudaFree(bank->master_weights.data);
    }
}

// Append a fresh frozen bank with the given per-buffer slice size; returns its
// index. Rebuilds bank_layout sequentially (primary first, then frozen banks in
// add order). Must be called BEFORE cudagraph capture (pointers get baked in).
extern "C" int pufferl_add_frozen_bank(PuffeRL* pufferl, int slice_size,
        int hidden_size, int num_layers) {
    int idx = pufferl->num_frozen_banks;
    pufferl->frozen_banks = (WeightBank*)realloc(
        pufferl->frozen_banks, (idx + 1) * sizeof(WeightBank));
    memset(&pufferl->frozen_banks[idx], 0, sizeof(WeightBank));
    weight_bank_create_for_pufferl(&pufferl->frozen_banks[idx], pufferl,
        slice_size, hidden_size, num_layers);
    pufferl->num_frozen_banks++;

    // Rebuild sequential layout from declared slice_sizes.
    int agents_per_buffer = pufferl->vec->total_agents / pufferl->hypers.num_buffers;
    int frozen_total = 0;
    for (int b = 0; b < pufferl->num_frozen_banks; b++) {
        frozen_total += pufferl->frozen_banks[b].slice_size;
    }
    if (frozen_total > agents_per_buffer) {
        fprintf(stderr, "pufferl_add_frozen_bank: total frozen slice (%d) exceeds "
            "agents_per_buffer (%d)\n", frozen_total, agents_per_buffer);
    }
    int num_banks = 1 + pufferl->num_frozen_banks;
    pufferl->bank_layout = (int*)realloc(pufferl->bank_layout, (num_banks + 1) * sizeof(int));
    pufferl->bank_layout[0] = 0;
    pufferl->bank_layout[1] = agents_per_buffer - frozen_total;  // primary
    int cumul = pufferl->bank_layout[1];
    for (int b = 0; b < pufferl->num_frozen_banks; b++) {
        cumul += pufferl->frozen_banks[b].slice_size;
        pufferl->bank_layout[2 + b] = cumul;
    }
    return idx;
}

// Load a frozen bank's weights from a file (same format as save_weights — flat fp32).
// Safe to call between rollouts (in-place cudaMemcpy; cudagraphs hold the pointer,
// not a copy of the data).
extern "C" void pufferl_load_frozen_bank(PuffeRL* pufferl, int bank_idx, const char* path) {
    if (bank_idx < 0 || bank_idx >= pufferl->num_frozen_banks) {
        fprintf(stderr, "pufferl_load_frozen_bank: bank_idx %d out of range\n", bank_idx);
        return;
    }
    WeightBank* bank = &pufferl->frozen_banks[bank_idx];
    int64_t nbytes = numel(bank->master_weights.shape) * sizeof(float);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "pufferl_load_frozen_bank: failed to open %s\n", path);
        return;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size != nbytes) {
        fprintf(stderr, "pufferl_load_frozen_bank: size mismatch (expected %lld, got %ld)\n",
            (long long)nbytes, file_size);
        fclose(f);
        return;
    }
    std::vector<char> buf(nbytes);
    size_t nread = fread(buf.data(), 1, nbytes, f);
    fclose(f);
    if ((int64_t)nread != nbytes) {
        fprintf(stderr, "pufferl_load_frozen_bank: short read on %s\n", path);
        return;
    }
    cudaMemcpy(bank->master_weights.data, buf.data(), nbytes, cudaMemcpyHostToDevice);
    if (USE_BF16) {
        int n = numel(bank->param_puf.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
            bank->param_puf.data, bank->master_weights.data, n);
    }
    cudaDeviceSynchronize();
}

// Set the agent permutation. Validates that the perm respects buffer boundaries:
// each buffer's range [buf_start, buf_start+buf_size) must map onto itself (no
// cross-buffer writes, since each worker only owns its physical chunk).
extern "C" void pufferl_set_agent_perm(PuffeRL* pufferl, const int* perm) {
    int total = pufferl->vec->total_agents;
    int num_buffers = pufferl->hypers.num_buffers;
    int buf_size = total / num_buffers;
    for (int b = 0; b < num_buffers; b++) {
        int lo = b * buf_size;
        int hi = lo + buf_size;
        for (int i = lo; i < hi; i++) {
            if (perm[i] < lo || perm[i] >= hi) {
                fprintf(stderr,
                    "pufferl_set_agent_perm: perm[%d]=%d crosses buffer %d range [%d,%d)\n",
                    i, perm[i], b, lo, hi);
                return;
            }
        }
    }
    static_vec_set_perm(pufferl->vec, perm);
}

// Set per-env tags (e.g. selfplay vs historical). tags array length must equal
// pufferl_num_envs(). Also clears each env's boundary_reached flag.
extern "C" void pufferl_set_env_tags(PuffeRL* pufferl, const int* tags) {
    static_vec_set_env_tags(pufferl->vec, tags);
}

// Returns count of envs with tag == tag_value AND boundary_reached. If
// reset_flags != 0, clears boundary_reached only on envs whose tag matches
// tag_value (so multi-bank swaps don't trample each other's alignment).
extern "C" int pufferl_count_aligned(PuffeRL* pufferl, int tag_value, int reset_flags) {
    return static_vec_count_aligned(pufferl->vec, tag_value, reset_flags);
}

extern "C" int pufferl_num_envs(PuffeRL* pufferl) {
    return pufferl->vec->size;
}

std::unique_ptr<PuffeRL> create_pufferl_impl(HypersT& hypers,
        const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs) {
    auto pufferl = std::make_unique<PuffeRL>();
    pufferl->hypers = hypers;
    pufferl->nccl_comm = nullptr;
    pufferl->default_stream = 0;
    pufferl->env_name = env_name;

    cudaSetDevice(hypers.gpu_id);

    // Multi-GPU: initialize NCCL
    if (hypers.world_size > 1) {
        if (hypers.nccl_id.size() != sizeof(ncclUniqueId))
            throw std::runtime_error("nccl_id must be " + std::to_string(sizeof(ncclUniqueId)) + " bytes");
        ncclUniqueId nccl_id;
        memcpy(&nccl_id, hypers.nccl_id.data(), sizeof(nccl_id));
        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    ulong seed = hypers.seed + hypers.rank;
    pufferl->seed = seed;

    // Load environment first to get input_size and action info from env
    // Create environments and set up action sizes
    StaticVec* vec = create_environments(hypers.num_buffers, hypers.total_agents,
        env_name, vec_kwargs, env_kwargs, pufferl->env);
    pufferl->vec = vec;

    // Sanity check action space
    int num_action_heads = pufferl->env.actions.shape[1];
    int* raw_act_sizes = get_act_sizes();  // CPU int32 pointer from env
    int act_n = 0;
    int num_continuous = 0;
    int num_discrete = 0;
    for (int i = 0; i < num_action_heads; i++) {
        int val = raw_act_sizes[i];
        if (val == 1) {
            num_continuous++;
        } else {
            num_discrete++;
        }
        act_n += val;
    }
    if (num_action_heads > MAX_ATN_HEADS) {
        throw std::runtime_error(
            "factorized action space exceeds MAX_ATN_HEADS");
    }
    assert((num_continuous == 0 || num_discrete == 0) &&
        "Mixed continuous/discrete action spaces not supported");
    pufferl->is_continuous = (num_continuous > 0);
    if (pufferl->is_continuous && hypers.ppo_clip_mode != PPO_CLIP_JOINT) {
        throw std::runtime_error(
            "non-joint train.ppo_clip_mode is only supported for discrete actions");
    }
    ExpandedPPOHeadAdapter ppo_adapter = expand_ppo_head_adapter(
        num_action_heads, raw_act_sizes, hypers.ppo_clip_mode != PPO_CLIP_JOINT);
    pufferl->ppo_head_adapter_version = ppo_adapter.version;
    pufferl->ppo_num_groups = ppo_adapter.version
        ? ppo_adapter.num_groups : num_action_heads;
    if (pufferl->is_continuous) {
        printf("Detected continuous action space with %d dimensions\n", num_action_heads);
    } else {
        printf("Detected discrete action space with %d heads\n", num_action_heads);
    }

    // Create profiling events
    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventCreate(&pufferl->profile.events[i]);
    }
    memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));
    nvmlInit();
    nvmlDeviceGetHandleByIndex(hypers.gpu_id, &pufferl->nvml_device);

    // Create policy
    int input_size = pufferl->env.obs.shape[1];
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    bool is_continuous = pufferl->is_continuous;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;
    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int inf_batch = vec->total_agents / hypers.num_buffers;
    int B_TT = minibatch_segments * hypers.horizon;
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    pufferl->policy = build_policy(env_name.c_str(), input_size, hidden_size,
        num_layers, decoder_output_size, act_n, is_continuous, hypers.horizon);

    // Create and allocate params
    Allocator* params = &pufferl->params_alloc;
    Allocator* acts = &pufferl->activations_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // Buffers for weights, grads, and activations
    pufferl->weights = policy_weights_create(&pufferl->policy, params);
    pufferl->train_activations = policy_reg_train(&pufferl->policy, pufferl->weights, acts, grads, B_TT);
    pufferl->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
    pufferl->buffer_states = (PrecisionTensor*)calloc(num_buffers, sizeof(PrecisionTensor));
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_activations[i] = policy_reg_rollout(
            &pufferl->policy, pufferl->weights, acts, inf_batch);
        pufferl->buffer_states[i] = {
            .shape = {num_layers, batch, hidden_size},
        };
        alloc_register(acts, &pufferl->buffer_states[i]);
    }
    int mask_size = pufferl->vec->action_mask_size;
    register_rollout_buffers(pufferl->rollouts,
        acts, horizon, total_agents, input_size, num_action_heads, mask_size);
    register_train_buffers(pufferl->train_buf,
        acts, minibatch_segments, horizon, input_size,
        hidden_size, num_action_heads, num_layers, mask_size);
    register_rollout_buffers(pufferl->train_rollouts,
        acts, total_agents, horizon, input_size, num_action_heads, mask_size);
    register_ppo_buffers(pufferl->ppo_bufs_puf,
        acts, minibatch_segments, hypers.horizon, decoder_output_size, is_continuous);
    register_prio_buffers(pufferl->prio_bufs,
        acts, hypers.total_agents, minibatch_segments);

    // Extra cuda buffers just reuse activ allocator
    pufferl->rng_offset_puf = {.shape = {num_buffers + 1}};
    alloc_register(acts, &pufferl->rng_offset_puf);

    pufferl->act_sizes_puf  = {.shape = {num_action_heads}};
    alloc_register(acts, &pufferl->act_sizes_puf);

    pufferl->losses_puf = {.shape = {NUM_LOSSES}};
    alloc_register(acts, &pufferl->losses_puf);

    pufferl->advantages_puf = {.shape = {total_agents, horizon}};
    alloc_register(acts, &pufferl->advantages_puf);

    muon_init(&pufferl->muon, params, hypers.lr, hypers.beta1, hypers.eps, 0.0, acts);
    pufferl->muon.nccl_comm = pufferl->nccl_comm;
    pufferl->muon.world_size = hypers.world_size;

    // All buffers allocated here
    if (alloc_create(params) != cudaSuccess) {
        return nullptr;
    }
    if (alloc_create(grads) != cudaSuccess) {
        return nullptr;
    }
    if (alloc_create(acts) != cudaSuccess) {
        return nullptr;
    }

    pufferl->grad_puf = {.data = (precision_t*)grads->mem, .shape = {grads->total_elems}};
    pufferl->param_puf = {.data = (precision_t*)params->mem, .shape = {params->total_elems}};

    ulong init_seed = hypers.seed;
    policy_init_weights(&pufferl->policy, pufferl->weights, &init_seed, pufferl->default_stream);
    pufferl->master_weights = {.data = (float*)pufferl->param_puf.data, .shape = {params->total_elems}};
    if (USE_BF16) {
        pufferl->master_weights = {.shape = {params->total_elems}};
        cudaMalloc(&pufferl->master_weights.data, params->total_elems * sizeof(float));
        int n = numel(pufferl->param_puf.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
            pufferl->master_weights.data, pufferl->param_puf.data, n);
    }

    // Per-buffer persistent RNG states
    int agents_per_buf = total_agents / num_buffers;
    pufferl->rng_states = (curandStatePhilox4_32_10_t**)calloc(num_buffers, sizeof(curandStatePhilox4_32_10_t*));
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc(&pufferl->rng_states[i], agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
            pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
    }

    // Post-create initialization
    cudaMemcpy(pufferl->act_sizes_puf.data, raw_act_sizes, num_action_heads * sizeof(int), cudaMemcpyHostToDevice);
    if (ppo_adapter.version) {
        cudaMalloc(&pufferl->ppo_head_to_group,
            num_action_heads * sizeof(int));
        cudaMalloc(&pufferl->ppo_head_selector,
            num_action_heads * sizeof(int));
        cudaMalloc(&pufferl->ppo_head_used_offsets,
            (num_action_heads + 1) * sizeof(int));
        cudaMalloc(&pufferl->ppo_head_used_values,
            ppo_adapter.head_used_values.size() * sizeof(unsigned char));
        cudaMemcpy(pufferl->ppo_head_to_group, ppo_adapter.head_to_group.data(),
            num_action_heads * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(pufferl->ppo_head_selector, ppo_adapter.head_selector.data(),
            num_action_heads * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(pufferl->ppo_head_used_offsets, ppo_adapter.head_used_offsets.data(),
            (num_action_heads + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(pufferl->ppo_head_used_values, ppo_adapter.head_used_values.data(),
            ppo_adapter.head_used_values.size() * sizeof(unsigned char),
            cudaMemcpyHostToDevice);
    }
    cudaMemset(pufferl->losses_puf.data, 0, NUM_LOSSES * sizeof(float));
    float one = 1.0f;
    cudaMemcpy(pufferl->ppo_bufs_puf.grad_loss.data, &one, sizeof(float), cudaMemcpyHostToDevice);
    muon_post_create(&pufferl->muon);

    // Set up frozen banks declared in vec_kwargs (num_frozen_banks +
    // frozen_bank_pct: each bank gets floor(agents_per_buffer * pct) agents).
    // Must happen BEFORE cudagraph capture so the graph bakes in their pointers
    // and per-bank loop iterations.
    DictItem* nb_item = dict_get_unsafe(vec_kwargs, "num_frozen_banks");
    DictItem* fbp_item = dict_get_unsafe(vec_kwargs, "frozen_bank_pct");
    DictItem* fbh_item = dict_get_unsafe(vec_kwargs, "frozen_bank_hidden_size");
    DictItem* fbl_item = dict_get_unsafe(vec_kwargs, "frozen_bank_num_layers");
    int num_frozen = nb_item ? (int)nb_item->value : 0;
    float frozen_pct = fbp_item ? (float)fbp_item->value : 0.0f;
    int frozen_hidden = fbh_item ? (int)fbh_item->value : hidden_size;
    int frozen_layers = fbl_item ? (int)fbl_item->value : num_layers;
    if (num_frozen > 0) {
        int agents_per_buffer = total_agents / num_buffers;
        int frozen_size = (int)((float)agents_per_buffer * frozen_pct);  // truncates = floor for positive
        int frozen_total = num_frozen * frozen_size;
        if (frozen_size <= 0 || frozen_total > agents_per_buffer) {
            fprintf(stderr, "create_pufferl: invalid frozen bank config "
                "(num=%d, pct=%.4f -> size=%d, total=%d, agents_per_buffer=%d)\n",
                num_frozen, frozen_pct, frozen_size, frozen_total, agents_per_buffer);
            return nullptr;
        }
        // add_frozen_bank auto-builds the sequential bank_layout.
        for (int b = 0; b < num_frozen; b++) {
            pufferl_add_frozen_bank(pufferl.get(), frozen_size, frozen_hidden, frozen_layers);
        }
    }

    // Cudagraph rolluts and entire training step
    if (hypers.cudagraphs >= 0) {
        pufferl->fused_rollout_cudagraphs = (cudaGraphExec_t*)calloc(horizon*num_buffers, sizeof(cudaGraphExec_t));
        pufferl->train_warmup = 0;

        // Snapshot weights + optimizer state before init-time capture
        long wb_bytes = numel(pufferl->master_weights.shape) * sizeof(float);
        void* saved_weights;
        cudaMalloc(&saved_weights, wb_bytes);
        cudaMemcpy(saved_weights, pufferl->master_weights.data, wb_bytes, cudaMemcpyDeviceToDevice);
        void* saved_momentum;
        cudaMalloc(&saved_momentum, wb_bytes);
        cudaMemcpy(saved_momentum, pufferl->muon.mb_puf.data, wb_bytes, cudaMemcpyDeviceToDevice);

        // Create per-buffer streams before capture so graphs are
        // captured and replayed on the same streams.
        pufferl->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamCreate(&pufferl->streams[i]);
            vec->streams[i] = pufferl->streams[i];
        }

        cudaStream_t saved_default = pufferl->default_stream;
        cudaStream_t saved_tl = tl_stream;
        cudaStream_t warmup_stream;
        cudaStreamCreate(&warmup_stream);
        pufferl->default_stream = warmup_stream;

        for (pufferl->epoch = 0; pufferl->epoch <= hypers.cudagraphs; pufferl->epoch++) {
            for (int i = 0; i < num_buffers * horizon; ++i) {
                int buf = i % num_buffers;
                tl_stream = pufferl->streams[buf];
                net_callback_wrapper(pufferl.get(), buf, i / num_buffers);
                cudaDeviceSynchronize();
            }
        }
        pufferl->rollout_captured = true;

        tl_stream = warmup_stream;
        for (int i = 0; i <= hypers.cudagraphs; i++) {
            train_impl(*pufferl);
        }

        cudaStreamSynchronize(warmup_stream);
        cudaDeviceSynchronize();
        pufferl->default_stream = saved_default;
        tl_stream = saved_tl;
        cudaStreamDestroy(warmup_stream);

        // Restore weights + optimizer state corrupted by warmup/capture
        cudaMemcpy(pufferl->master_weights.data, saved_weights, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_weights);
        cudaMemcpy(pufferl->muon.mb_puf.data, saved_momentum, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_momentum);
        if (USE_BF16) {
            int n = numel(pufferl->param_puf.shape);
            cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
                pufferl->param_puf.data, pufferl->master_weights.data, n);
        }

        // Re-init RNG states corrupted by warmup
        for (int i = 0; i < num_buffers; i++) {
            rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
                pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
        }
        cudaDeviceSynchronize();

        pufferl->epoch = 0;
        pufferl->global_step = 0;
    }

    // Create per-buffer streams if not already created by cudagraph path
    if (!pufferl->streams) {
        pufferl->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamCreate(&pufferl->streams[i]);
            vec->streams[i] = pufferl->streams[i];
        }
    }

    create_static_threads(vec, hypers.num_threads, horizon, pufferl.get(),
        net_callback_wrapper, thread_init_wrapper);
    static_vec_reset(vec);

    if (hypers.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
    }

    double now = wall_clock();
    pufferl->start_time = now;
    pufferl->last_log_time = now;
    pufferl->last_log_step = 0;

    return pufferl;
}

void close_impl(PuffeRL& pufferl) {
    cudaDeviceSynchronize();
    if (pufferl.hypers.profile) {
        cudaProfilerStop();
    }

    cudaGraphExecDestroy(pufferl.train_cudagraph);
    for (int i = 0; i < pufferl.hypers.horizon * pufferl.hypers.num_buffers; i++) {
        cudaGraphExecDestroy(pufferl.fused_rollout_cudagraphs[i]);
    }

    policy_weights_free(&pufferl.policy, &pufferl.weights);
    policy_activations_free(&pufferl.policy, pufferl.train_activations);
    for (int buf = 0; buf < pufferl.hypers.num_buffers; buf++) {
        policy_activations_free(&pufferl.policy, pufferl.buffer_activations[buf]);
    }

    for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
        cudaFree(pufferl.rng_states[i]);
    }
    free(pufferl.rng_states);

    if (USE_BF16) {
        cudaFree(pufferl.master_weights.data);
    }

    alloc_free(&pufferl.params_alloc);
    alloc_free(&pufferl.grads_alloc);
    alloc_free(&pufferl.activations_alloc);

    for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
        cudaStreamDestroy(pufferl.streams[i]);
    }
    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventDestroy(pufferl.profile.events[i]);
    }
    nvmlShutdown();

    static_vec_close(pufferl.vec);

    free(pufferl.buffer_states);
    free(pufferl.buffer_activations);
    free(pufferl.fused_rollout_cudagraphs);
    free(pufferl.streams);

    for (int b = 0; b < pufferl.num_frozen_banks; b++) {
        weight_bank_destroy(&pufferl.frozen_banks[b], &pufferl);
    }
    free(pufferl.frozen_banks);
    free(pufferl.bank_layout);
    cudaFree(pufferl.ppo_head_to_group);
    cudaFree(pufferl.ppo_head_selector);
    cudaFree(pufferl.ppo_head_used_offsets);
    cudaFree(pufferl.ppo_head_used_values);

    if (pufferl.nccl_comm != nullptr) {
        ncclCommDestroy(pufferl.nccl_comm);
    }
}
