// Host-executable checks for the production categorical primitive
// (src/categorical.cuh), run through the 32-lane warp emulator in
// tests/cuda_stub/. This is the coverage that can run on a machine without
// CUDA; tests/test_categorical_mask.py is the real-GPU counterpart.
//
// Build (float):  c++ -std=c++17 -O1 -Itests/cuda_stub -DPRECISION_FLOAT \
//                     -o build/categorical_host_float tests/categorical_host_exec.cc
// Build (bf16):   same without -DPRECISION_FLOAT
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "../src/categorical.cuh"

thread_local PufHostWarp* puf_host_warp = nullptr;
thread_local int puf_host_lane = 0;
thread_local PufHostDim3 threadIdx = {0, 0, 0};

static std::mutex g_atomic_mutex;

int puf_host_atomic_cas(int* addr, int cmp, int val) {
    std::lock_guard<std::mutex> lock(g_atomic_mutex);
    int old = *addr;
    if (old == cmp) {
        *addr = val;
    }
    return old;
}

void puf_host_warp_run(const std::function<void(int)>& body) {
    PufHostWarp warp;
    std::vector<std::thread> threads;
    threads.reserve(PufHostWarp::W);
    for (int lane = 0; lane < PufHostWarp::W; lane++) {
        threads.emplace_back([&warp, &body, lane]() {
            puf_host_warp = &warp;
            puf_host_lane = lane;
            threadIdx.x = (unsigned)lane;
            body(lane);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}

// --------------------------------------------------------------------------
// Test harness
// --------------------------------------------------------------------------
static int g_failures = 0;
static int g_checks = 0;
static std::string g_case;

static void check(bool ok, const char* what, double got = 0.0,
        double expect = 0.0) {
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL [%s] %s (got %.9g expect %.9g)\n", g_case.c_str(), what,
            got, expect);
    }
}

static void begin_case(const std::string& name) {
    g_case = name;
}

static bool close(double a, double b, double tol) {
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

static bool is_positive_zero(float v) {
    return v == 0.0f && !std::signbit(v);
}

static double tolerance() {
    return sizeof(precision_t) == sizeof(float) ? 3e-6 : 5e-2;
}

// --------------------------------------------------------------------------
// float64 legal-only reference
// --------------------------------------------------------------------------
struct RefHead {
    std::vector<double> logp;
    double entropy;
    double max_logit;
    double log_sum;
};

static RefHead reference(const std::vector<float>& logits,
        const std::vector<unsigned char>& mask) {
    RefHead r;
    r.logp.assign(logits.size(), 0.0);
    double m = -INFINITY;
    for (size_t i = 0; i < logits.size(); i++) {
        if (mask[i] && (double)logits[i] > m) {
            m = (double)logits[i];
        }
    }
    double s = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        if (mask[i]) {
            s += std::exp((double)logits[i] - m);
        }
    }
    r.max_logit = m;
    r.log_sum = std::log(s);
    r.entropy = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        if (mask[i]) {
            r.logp[i] = ((double)logits[i] - m) - std::log(s);
            r.entropy -= std::exp(r.logp[i]) * r.logp[i];
        }
    }
    return r;
}

// Round host floats through the build precision so the reference and the
// primitive see identical inputs.
static std::vector<float> round_trip(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = to_float(from_float(v[i]));
    }
    return out;
}

static std::vector<precision_t> to_prec(const std::vector<float>& v) {
    std::vector<precision_t> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = from_float(v[i]);
    }
    return out;
}

struct HeadResult {
    PufCatNorm norm;
    std::vector<float> logps;
    float entropy;
    float selected_logp;
    int sampled;
};

// Mirrors the learner cache stage for one head.
static HeadResult run_head(const std::vector<float>& logits_f,
        const std::vector<unsigned char>& mask_b, int width, int action = -1,
        float uniform = -1.0f, float eps = 0.0f) {
    std::vector<precision_t> logits = to_prec(logits_f);
    std::vector<float> mask_f(mask_b.begin(), mask_b.end());
    std::vector<precision_t> mask = to_prec(mask_f);
    HeadResult out;
    out.logps.assign(logits_f.size(), -12345.0f);
    out.entropy = 0.0f;
    out.selected_logp = 0.0f;
    out.sampled = -1;
    puf_host_warp_run([&](int lane) {
        PufCatNorm n = puf_cat_head_norm(
            logits.data(), mask.data(), width, eps > 0.0f);
        if (n.code == PUF_CAT_OK) {
            puf_cat_write_logps(logits.data(), mask.data(), width, n,
                out.logps.data());
            float ent = puf_cat_entropy_from_logps(
                out.logps.data(), mask.data(), width);
            int sampled = -1;
            if (uniform >= 0.0f) {
                float inv_legal = eps > 0.0f ? 1.0f / (float)n.legal_count : 0.0f;
                sampled = puf_cat_sample(logits.data(), mask.data(), width, n,
                    uniform, eps, inv_legal);
            }
            if (lane == 0) {
                out.entropy = ent;
                out.sampled = sampled;
                if (action >= 0) {
                    out.selected_logp = puf_cat_logp_at(
                        logits.data(), action, n);
                }
            }
        }
        if (lane == 0) {
            out.norm = n;
        }
    });
    return out;
}

// --------------------------------------------------------------------------
// Cases
// --------------------------------------------------------------------------
static unsigned g_rng = 12345u;

static float next_uniform() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)((g_rng >> 8) & 0xffffff) / (float)0x1000000;
}

static float next_normal(float scale) {
    float u1 = next_uniform() + 1e-7f;
    float u2 = next_uniform();
    return scale * std::sqrt(-2.0f * std::log(u1))
        * std::cos(6.2831853f * u2);
}

static void case_reference_match() {
    const int widths[] = {1, 2, 3, 31, 32, 33, 63, 64, 65, 97, 155, 212, 257};
    for (int wi = 0; wi < (int)(sizeof(widths) / sizeof(widths[0])); wi++) {
        int width = widths[wi];
        begin_case("reference width=" + std::to_string(width));
        std::vector<float> logits(width);
        std::vector<unsigned char> mask(width, 0);
        for (int i = 0; i < width; i++) {
            logits[i] = next_normal(4.0f);
            mask[i] = next_uniform() > 0.4f ? 1 : 0;
        }
        mask[0] = 1;
        mask[width - 1] = 1;
        logits = round_trip(logits);
        RefHead ref = reference(logits, mask);
        HeadResult got = run_head(logits, mask, width, 0);
        check(got.norm.code == PUF_CAT_OK, "status ok", got.norm.code, 0);
        double tol = tolerance();
        for (int i = 0; i < width; i++) {
            if (mask[i]) {
                check(close(got.logps[i], ref.logp[i], tol), "logp", got.logps[i],
                    ref.logp[i]);
            } else {
                check(is_positive_zero(got.logps[i]), "masked logp is +0",
                    got.logps[i], 0.0);
            }
        }
        check(close(got.entropy, ref.entropy, tol), "entropy", got.entropy,
            ref.entropy);
        check(close(got.selected_logp, ref.logp[0], tol), "selected logp",
            got.selected_logp, ref.logp[0]);
    }
}

static void case_masked_extremes() {
    begin_case("legal below -1e4 vs masked extremes");
    int width = 64;
    std::vector<float> logits(width, -3.0e4f);
    std::vector<unsigned char> mask(width, 0);
    logits[10] = -1.1e4f;
    logits[20] = -1.2e4f;
    mask[10] = 1;
    mask[20] = 1;
    logits[0] = 3.4e38f;
    logits[1] = std::nanf("");
    logits[2] = INFINITY;
    logits[3] = -INFINITY;
    logits[40] = 1.0e30f;
    std::vector<float> rt = round_trip(logits);
    RefHead ref = reference(rt, mask);
    HeadResult got = run_head(rt, mask, width, 10);
    check(got.norm.code == PUF_CAT_OK, "masked extremes ignored", got.norm.code, 0);
    check(close(got.logps[10], ref.logp[10], tolerance()), "best legal logp",
        got.logps[10], ref.logp[10]);
    check(close(got.logps[20], ref.logp[20], tolerance()), "other legal logp",
        got.logps[20], ref.logp[20]);
    check(is_positive_zero(got.logps[0]), "masked FLT_MAX logp", got.logps[0], 0);
    check(is_positive_zero(got.logps[1]), "masked NaN logp", got.logps[1], 0);
    check(is_positive_zero(got.logps[2]), "masked +Inf logp", got.logps[2], 0);
    check(is_positive_zero(got.logps[3]), "masked -Inf logp", got.logps[3], 0);
    check(std::isfinite(got.entropy), "entropy finite", got.entropy, 0);

    for (int i = 0; i <= 64; i++) {
        float u = std::max(1e-7f, (float)i / 64.0f);
        HeadResult s = run_head(rt, mask, width, 10, u);
        check(s.sampled == 10 || s.sampled == 20, "sampled legal", s.sampled, 10);
    }
}

static void case_single_legal() {
    begin_case("single legal category");
    int width = 155;
    std::vector<float> logits(width, -3.0e4f);
    logits[77] = -2.5e4f;
    std::vector<unsigned char> mask(width, 0);
    mask[77] = 1;
    std::vector<float> rt = round_trip(logits);
    HeadResult got = run_head(rt, mask, width, 77, 0.999999f);
    check(got.norm.code == PUF_CAT_OK, "status ok", got.norm.code, 0);
    check(got.logps[77] == 0.0f, "certain logp is 0", got.logps[77], 0);
    check(is_positive_zero(got.entropy), "entropy exactly +0", got.entropy, 0);
    check(got.sampled == 77, "sampled only legal", got.sampled, 77);
}

static void case_errors() {
    begin_case("empty mask");
    int width = 155;
    std::vector<float> logits(width, 0.0f);
    std::vector<unsigned char> mask(width, 0);
    HeadResult got = run_head(logits, mask, width);
    check(got.norm.code == PUF_CAT_ERR_EMPTY, "empty legal set", got.norm.code,
        PUF_CAT_ERR_EMPTY);

    begin_case("nonpositive width");
    std::vector<unsigned char> ones(width, 1);
    got = run_head(logits, ones, 0);
    check(got.norm.code == PUF_CAT_ERR_WIDTH, "width <= 0", got.norm.code,
        PUF_CAT_ERR_WIDTH);
    got = run_head(logits, ones, -7);
    check(got.norm.code == PUF_CAT_ERR_WIDTH, "negative width", got.norm.code,
        PUF_CAT_ERR_WIDTH);

    const float bad[] = {std::nanf(""), INFINITY, -INFINITY};
    for (int i = 0; i < 3; i++) {
        begin_case("legal nonfinite logit " + std::to_string(i));
        std::vector<float> bad_logits(width, 0.0f);
        bad_logits[100] = bad[i];
        got = run_head(bad_logits, ones, width);
        check(got.norm.code == PUF_CAT_ERR_LEGAL_NONFINITE, "nonfinite legal",
            got.norm.code, PUF_CAT_ERR_LEGAL_NONFINITE);
        check(got.norm.detail == 100, "offending category", got.norm.detail, 100);
    }

    begin_case("invalid actions");
    const float actions[] = {0.5f, -1.0f, 212.0f, 1e9f, std::nanf(""), INFINITY,
        -0.0f, 211.0f};
    const int expect_ok[] = {0, 0, 0, 0, 0, 0, 1, 1};
    for (int i = 0; i < 8; i++) {
        int code = PUF_CAT_OK;
        int idx = puf_cat_check_action(actions[i], 212, &code);
        if (expect_ok[i]) {
            check(code == PUF_CAT_OK, "valid action accepted", code, 0);
            check(idx == (int)actions[i], "action index", idx, actions[i]);
        } else {
            check(code == PUF_CAT_ERR_ACTION, "invalid action rejected", code,
                PUF_CAT_ERR_ACTION);
            check(idx == -1, "invalid action index", idx, -1);
        }
    }

    begin_case("status first-error payload");
    PufCatStatus status;
    std::memset(&status, 0, sizeof(status));
    puf_cat_report(&status, PUF_CAT_ERR_EMPTY, 7, 3, 155);
    puf_cat_report(&status, PUF_CAT_ERR_ACTION, 9, 1, 2);
    check(status.code == PUF_CAT_ERR_EMPTY, "first error wins", status.code,
        PUF_CAT_ERR_EMPTY);
    check(status.row == 7 && status.head == 3 && status.detail == 155,
        "payload", status.row, 7);
    check(std::string(puf_cat_code_name(PUF_CAT_ERR_EMPTY))
        == "empty legal action set", "code name", 0, 0);
}

static void case_shift_invariance() {
    // Integer logits in [-16, 16] shifted by integers that keep |value| <= 128
    // are exactly representable in both float32 and bfloat16 (bf16 has 8 bits
    // of precision, so integers stay exact only below 256). float32 also gets
    // the large shifts. A legal-uniform shift must leave the normalized result
    // bitwise unchanged.
    std::vector<float> shifts = {-96.0f, -32.0f, -1.0f, 0.0f, 1.0f, 32.0f,
        96.0f};
    if (sizeof(precision_t) == sizeof(float)) {
        shifts.push_back(-1.0e4f);
        shifts.push_back(-8192.0f);
        shifts.push_back(8192.0f);
        shifts.push_back(1.0e4f);
    }
    int width = 155;
    std::vector<float> base(width);
    std::vector<unsigned char> mask(width, 0);
    for (int i = 0; i < width; i++) {
        float v = std::round(next_normal(6.0f));
        base[i] = std::max(-16.0f, std::min(16.0f, v));
        mask[i] = next_uniform() > 0.3f ? 1 : 0;
    }
    mask[0] = 1;
    HeadResult ref = run_head(base, mask, width, 0);
    for (int s = 0; s < (int)shifts.size(); s++) {
        begin_case("uniform shift " + std::to_string(shifts[s]));
        std::vector<float> shifted(width);
        for (int i = 0; i < width; i++) {
            shifted[i] = base[i] + shifts[s];
        }
        HeadResult got = run_head(shifted, mask, width, 0);
        check(got.norm.code == PUF_CAT_OK, "status ok", got.norm.code, 0);
        for (int i = 0; i < width; i++) {
            if (!mask[i]) {
                check(is_positive_zero(got.logps[i]), "masked stays +0",
                    got.logps[i], 0);
                continue;
            }
            check(got.logps[i] == ref.logps[i], "shift-invariant logp",
                got.logps[i], ref.logps[i]);
        }
        check(got.entropy == ref.entropy, "shift-invariant entropy",
            got.entropy, ref.entropy);
        check(got.selected_logp == ref.selected_logp,
            "shift-invariant selected logp", got.selected_logp,
            ref.selected_logp);
    }
}

static void case_sampling_cdf() {
    begin_case("sampling matches reference CDF");
    int width = 97;
    std::vector<float> logits(width);
    std::vector<unsigned char> mask(width, 0);
    for (int i = 0; i < width; i++) {
        logits[i] = next_normal(1.5f);
        mask[i] = next_uniform() > 0.5f ? 1 : 0;
    }
    mask[0] = 1;
    mask[width - 1] = 1;
    std::vector<float> rt = round_trip(logits);
    RefHead ref = reference(rt, mask);
    std::vector<double> cdf(width);
    double acc = 0.0;
    for (int i = 0; i < width; i++) {
        acc += mask[i] ? std::exp(ref.logp[i]) : 0.0;
        cdf[i] = acc;
    }
    // Compare positions among legal categories: masked gaps mean neighbouring
    // legal categories can be many raw indices apart, and float rounding of the
    // CDF boundary may legitimately move the draw by one legal step.
    std::vector<int> rank(width, -1);
    int legal_count = 0;
    for (int i = 0; i < width; i++) {
        if (mask[i]) {
            rank[i] = legal_count++;
        }
    }
    int mismatches = 0;
    const int trials = 129;
    for (int t = 0; t < trials; t++) {
        float u = (float)((t + 0.5) / trials);
        HeadResult got = run_head(rt, mask, width, -1, u);
        check(mask[got.sampled] != 0, "sample is legal", got.sampled, 0);
        int expect = 0;
        while (expect < width - 1 && cdf[expect] <= (double)u) {
            expect++;
        }
        if (std::abs(rank[got.sampled] - rank[expect]) > 1) {
            mismatches++;
        }
    }
    check(mismatches == 0, "inverse CDF selection", mismatches, 0);

    begin_case("epsilon mixture stays legal");
    for (int t = 0; t < 64; t++) {
        float u = std::max(1e-7f, (float)t / 64.0f);
        HeadResult got = run_head(rt, mask, width, -1, u, 0.4f);
        check(mask[got.sampled] != 0, "eps sample legal", got.sampled, 0);
    }

    begin_case("epsilon mixture admits a legal -Inf base logit");
    std::vector<float> mixed_logits = {0.0f, -INFINITY};
    std::vector<unsigned char> mixed_mask = {1, 1};
    HeadResult mixed = run_head(mixed_logits, mixed_mask, 2, 1, 0.9f, 0.4f);
    check(mixed.norm.code == PUF_CAT_OK, "mixed norm accepts -Inf",
        mixed.norm.code, PUF_CAT_OK);
    check(mixed.sampled == 1, "epsilon can sample zero-base-probability action",
        mixed.sampled, 1);
    check(std::isfinite(mixed.entropy), "base entropy remains finite",
        mixed.entropy, 0.0);

    HeadResult strict = run_head(mixed_logits, mixed_mask, 2, 1);
    check(strict.norm.code == PUF_CAT_ERR_LEGAL_NONFINITE,
        "generic distribution still rejects legal -Inf", strict.norm.code,
        PUF_CAT_ERR_LEGAL_NONFINITE);

    std::vector<float> all_negative_infinity = {-INFINITY, -INFINITY};
    HeadResult all_bad = run_head(
        all_negative_infinity, mixed_mask, 2, -1, 0.9f, 0.4f);
    check(all_bad.norm.code == PUF_CAT_ERR_LEGAL_NONFINITE,
        "all -Inf legal logits remain fatal", all_bad.norm.code,
        PUF_CAT_ERR_LEGAL_NONFINITE);
}

static void case_ratio_one() {
    begin_case("unchanged policy ratio is exactly 1");
    const int sizes[] = {155, 212, 3, 64, 1, 31, 32, 33};
    int num_heads = 8;
    for (int row = 0; row < 4; row++) {
        float old_lp = 0.0f;
        float new_lp = 0.0f;
        for (int h = 0; h < num_heads; h++) {
            int width = sizes[h];
            std::vector<float> logits(width);
            std::vector<unsigned char> mask(width, 0);
            for (int i = 0; i < width; i++) {
                logits[i] = next_normal(2.0f);
                mask[i] = next_uniform() > 0.3f ? 1 : 0;
            }
            mask[0] = 1;
            std::vector<float> rt = round_trip(logits);
            float u = std::max(1e-7f, next_uniform());
            HeadResult sampled = run_head(rt, mask, width, -1, u);
            check(sampled.norm.code == PUF_CAT_OK, "sample ok",
                sampled.norm.code, 0);
            HeadResult learner = run_head(rt, mask, width, sampled.sampled);
            // Rollout logp and learner recomputation must be bit identical.
            HeadResult rollout = run_head(rt, mask, width, sampled.sampled, u);
            check(rollout.selected_logp == learner.selected_logp,
                "bitwise identical logp", rollout.selected_logp,
                learner.selected_logp);
            old_lp += rollout.selected_logp;
            new_lp += learner.selected_logp;
        }
        check(new_lp == old_lp, "row logprob identical", new_lp, old_lp);
        check(std::exp(new_lp - old_lp) == 1.0f, "ratio exactly 1",
            std::exp(new_lp - old_lp), 1.0);
    }

    begin_case("mixed rollout and learner ratio is exactly 1");
    const float eps = 0.35f;
    const float inv_legal = 0.2f;
    const float base_logps[] = {-INFINITY, -7.0f, -1.25f, 0.0f};
    for (float base_logp : base_logps) {
        PufCatUniformMix rollout = puf_cat_uniform_mix(
            base_logp, eps, inv_legal);
        PufCatUniformMix learner = puf_cat_uniform_mix(
            base_logp, eps, inv_legal);
        check(rollout.logp == learner.logp, "mixed logprob bit identical",
            rollout.logp, learner.logp);
        check(std::exp(learner.logp - rollout.logp) == 1.0f,
            "mixed ratio exactly 1",
            std::exp(learner.logp - rollout.logp), 1.0);
        check(rollout.base_scale == learner.base_scale,
            "mixed gradient scale bit identical",
            rollout.base_scale, learner.base_scale);
    }

    begin_case("async slot keeps rollout epsilon");
    float stored_slot_eps = 0.35f;
    float current_global_eps = 0.05f;
    PufCatUniformMix rollout = puf_cat_uniform_mix(
        -1.25f, stored_slot_eps, inv_legal);
    PufCatUniformMix stored_learner = puf_cat_uniform_mix(
        -1.25f, stored_slot_eps, inv_legal);
    PufCatUniformMix stale_global_learner = puf_cat_uniform_mix(
        -1.25f, current_global_eps, inv_legal);
    check(stored_learner.logp == rollout.logp,
        "slot epsilon reproduces rollout logp",
        stored_learner.logp, rollout.logp);
    check(stale_global_learner.logp != rollout.logp,
        "changed global epsilon would mismatch",
        stale_global_learner.logp, rollout.logp);
}

static void case_gradients() {
    begin_case("gradients: masked exactly +0, legal analytic");
    int width = 212;
    std::vector<float> logits(width);
    std::vector<unsigned char> mask(width, 0);
    for (int i = 0; i < width; i++) {
        logits[i] = next_normal(2.0f);
        mask[i] = next_uniform() > 0.5f ? 1 : 0;
    }
    mask[3] = 1;
    // Masked entries carry adversarial raw logits.
    for (int i = 0; i < width; i++) {
        if (!mask[i]) {
            logits[i] = (i % 3 == 0) ? 3.4e38f
                : (i % 3 == 1 ? std::nanf("") : 1.0e5f);
        }
    }
    std::vector<float> rt = round_trip(logits);
    RefHead ref = reference(rt, mask);
    HeadResult cached = run_head(rt, mask, width, 3);
    check(cached.norm.code == PUF_CAT_OK, "status ok", cached.norm.code, 0);

    std::vector<float> grad = cached.logps;
    std::vector<float> mask_f(mask.begin(), mask.end());
    std::vector<precision_t> mask_p = to_prec(mask_f);
    const float d_logp = -0.75f;
    const float d_ent = -1.0e-3f;
    float entropy = 0.0f;
    puf_host_warp_run([&](int lane) {
        float e = puf_cat_entropy_from_logps(grad.data(), mask_p.data(), width);
        if (lane == 0) {
            entropy = e;
        }
        __syncwarp();
        puf_cat_write_grads(grad.data(), mask_p.data(), width, 3, d_logp,
            d_ent, e);
    });
    check(close(entropy, ref.entropy, tolerance()), "entropy", entropy,
        ref.entropy);
    double tol = tolerance() * 10.0;
    for (int i = 0; i < width; i++) {
        if (!mask[i]) {
            check(is_positive_zero(grad[i]), "masked gradient is +0", grad[i], 0);
            continue;
        }
        double p = std::exp(ref.logp[i]);
        double expect = ((i == 3 ? 1.0 : 0.0) - p) * d_logp
            + d_ent * p * (-ref.entropy - ref.logp[i]);
        check(close(grad[i], expect, tol), "legal gradient", grad[i], expect);
    }

    begin_case("zeroed (gated-off) head");
    std::vector<float> zeroed(width, 7.0f);
    puf_host_warp_run([&](int) { puf_cat_zero_head(zeroed.data(), width); });
    for (int i = 0; i < width; i++) {
        check(is_positive_zero(zeroed[i]), "gated head gradient is +0",
            zeroed[i], 0);
    }
}

static void case_many_heads() {
    begin_case("42 heterogeneous heads");
    std::vector<int> sizes;
    sizes.push_back(155);
    sizes.push_back(212);
    sizes.push_back(1);
    sizes.push_back(2);
    sizes.push_back(31);
    sizes.push_back(32);
    sizes.push_back(33);
    sizes.push_back(64);
    sizes.push_back(65);
    sizes.push_back(97);
    while ((int)sizes.size() < 42) {
        sizes.push_back(7);
    }
    double tol = tolerance();
    double total_ref_entropy = 0.0;
    double total_got_entropy = 0.0;
    for (size_t h = 0; h < sizes.size(); h++) {
        int width = sizes[h];
        std::vector<float> logits(width);
        std::vector<unsigned char> mask(width, 0);
        for (int i = 0; i < width; i++) {
            logits[i] = next_normal(3.0f);
            mask[i] = next_uniform() > 0.35f ? 1 : 0;
        }
        mask[0] = 1;
        std::vector<float> rt = round_trip(logits);
        RefHead ref = reference(rt, mask);
        HeadResult got = run_head(rt, mask, width, 0);
        check(got.norm.code == PUF_CAT_OK, "status ok", got.norm.code, 0);
        for (int i = 0; i < width; i++) {
            if (mask[i]) {
                check(close(got.logps[i], ref.logp[i], tol), "logp",
                    got.logps[i], ref.logp[i]);
            } else {
                check(is_positive_zero(got.logps[i]), "masked +0", got.logps[i], 0);
            }
        }
        total_ref_entropy += ref.entropy;
        total_got_entropy += got.entropy;
    }
    check(close(total_got_entropy, total_ref_entropy, tol), "summed entropy",
        total_got_entropy, total_ref_entropy);
}

static void case_any_nonzero_mask_byte_is_legal() {
    begin_case("any nonzero mask byte is legal");
    int width = 212;
    std::vector<float> logits(width);
    std::vector<unsigned char> ones(width, 0);
    std::vector<unsigned char> varied(width, 0);
    for (int i = 0; i < width; i++) {
        logits[i] = next_normal(2.0f);
        ones[i] = next_uniform() > 0.5f ? 1 : 0;
        varied[i] = ones[i] ? (unsigned char)(1 + (i % 254)) : 0;
    }
    ones[5] = 1;
    varied[5] = 200;
    std::vector<float> rt = round_trip(logits);
    HeadResult a = run_head(rt, ones, width, 5);
    HeadResult b = run_head(rt, varied, width, 5);
    check(a.norm.code == PUF_CAT_OK, "status ok", a.norm.code, 0);
    check(b.norm.code == PUF_CAT_OK, "status ok", b.norm.code, 0);
    for (int i = 0; i < width; i++) {
        check(a.logps[i] == b.logps[i], "mask byte value irrelevant",
            a.logps[i], b.logps[i]);
    }
    check(a.entropy == b.entropy, "entropy unchanged", a.entropy, b.entropy);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "abort") == 0) {
        // Host fail-before-consume helper must surface and abort, never return.
        PufCatStatus status;
        std::memset(&status, 0, sizeof(status));
        status.code = PUF_CAT_ERR_ACTION_ILLEGAL;
        status.row = 11;
        status.head = 5;
        status.detail = 93;
        puf_cat_status_check(&status, "unit test consumer");
        printf("puf_cat_status_check returned instead of aborting\n");
        return 2;
    }
    printf("precision: %s\n",
        sizeof(precision_t) == sizeof(float) ? "float32" : "bfloat16");
    case_reference_match();
    case_masked_extremes();
    case_single_legal();
    case_errors();
    case_shift_invariance();
    case_sampling_cdf();
    case_ratio_one();
    case_gradients();
    case_many_heads();
    case_any_nonzero_mask_byte_is_legal();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
