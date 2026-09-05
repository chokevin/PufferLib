from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "pufferl.cu"


class RewardClipSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = SOURCE.read_text()
        parser = source.split("static bool puf_read_clip_rewards(", 1)[1].split(
            "\nPuffeRL* create_pufferl(", 1
        )[0]
        kernel = source.split("__global__ void clamp_precision_kernel(", 1)[1].split(
            "// 1-thread graph node.", 1
        )[0]
        guard = re.search(
            r"if \(hypers->clip_rewards\) \{\s*"
            r"clamp_precision_kernel<<<.*?;\s*\}", source, re.DOTALL
        )
        if guard is None:
            raise AssertionError("learner reward clipping must be conditional")
        host_guard = re.sub(r"<<<.*?>>>", "", guard.group(0), flags=re.DOTALL)
        program = """
#include <stdbool.h>
#include <math.h>
#include "ini.h"
typedef float precision_t;
#define to_float(x) (x)
#define from_float(x) (x)
struct Dim { int x; };
static struct Dim blockIdx = {0}, threadIdx = {0}, blockDim = {1}, gridDim = {1};
static int numel(int n) { return n; }
static bool puf_read_clip_rewards(""" + parser + """
void clamp_precision_kernel(""" + kernel + """
static void apply(float* values, int count, bool enabled) {
    struct Hypers { bool clip_rewards; } config = {enabled};
    struct Hypers* hypers = &config;
    struct Rollout { struct { float* data; int shape; } rewards; } buffer = {{values, count}};
    struct Rollout* rollouts = &buffer;
""" + host_guard + """
}
int main(int argc, char** argv) {
    Ini ini = {0};
    puf_ini_load_file(&ini, argv[1]);
    if (argc > 2) puf_ini_put(&ini, "train.clip_rewards", argv[2]);
    bool enabled = puf_read_clip_rewards(&ini);
    float baseline[] = {-1.0f, 5.0f};
    float investment[] = {-1.0f, -1.0f, 7.0f};
    float bounds[] = {-100.0f, -0.5f, 0.0f, 0.5f, 100.0f};
    apply(baseline, 2, enabled);
    apply(investment, 3, enabled);
    apply(bounds, 5, enabled);
    printf("%d %.9g %.9g %.9g %.9g %.9g %.9g %.9g\\n", enabled,
        baseline[0] + baseline[1], investment[0] + investment[1] + investment[2],
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4]);
    return 0;
}
"""
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.executable = Path(cls.directory.name) / "reward-clip"
        subprocess.run(
            ["cc", "-std=c11", "-I", str(ROOT / "src"), "-x", "c", "-",
             "-lm", "-o", str(cls.executable)],
            input=program, text=True, check=True, capture_output=True,
        )

    def run_config(self, value=None):
        command = [str(self.executable), str(ROOT / "config" / "default.ini")]
        if value is not None:
            command.append(value)
        return subprocess.run(command, text=True, capture_output=True)

    def test_stock_default_keeps_clipping(self):
        result = self.run_config()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["1", "0", "-1", "-1", "-0.5", "0", "0.5", "1"])

    def test_disabled_clipping_preserves_settlement_and_profit_order(self):
        for value in ("0", "false"):
            with self.subTest(value=value):
                result = self.run_config(value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.split(), ["0", "4", "5", "-100", "-0.5", "0", "0.5", "100"])

    def test_enabled_clipping_keeps_legacy_bounds(self):
        for value in ("1", "true"):
            with self.subTest(value=value):
                result = self.run_config(value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.split()[0:3], ["1", "0", "-1"])

    def test_invalid_setting_fails_closed(self):
        for value in ("-1", "2", "nan", "inf", "banana", "0,1"):
            with self.subTest(value=value):
                result = self.run_config(value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("clip_rewards must be", result.stderr)

    def test_native_wiring_precedes_advantage_computation(self):
        source = SOURCE.read_text()
        train = source.split("static void train_epoch_gpu(", 1)[1].split("\nvoid train_impl(", 1)[0]
        self.assertLess(train.index("if (hypers->clip_rewards)"), train.index("puff_advantage<<<"))
        self.assertIn(".clip_rewards = puf_read_clip_rewards(ini)", source)


if __name__ == "__main__":
    unittest.main()
