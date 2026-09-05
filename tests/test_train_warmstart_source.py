from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "pufferl.cu"


class TrainWarmstartSourceTest(unittest.TestCase):
    def test_load_receipt_survives_nonzero_rank_stdout_redirection(self):
        source = SOURCE.read_text()
        helper = source.split("void pufferl_load_initial_policy(", 1)[1].split(
            "// fp32 master weights:", 1
        )[0]
        receipt = re.search(
            r'(?:printf\(|fprintf\(stderr,)\s*"NATIVE_INITIAL_MODEL_LOADED'
            r'.*?fflush\((?:stdout|stderr)\);',
            helper,
            re.DOTALL,
        )
        self.assertIsNotNone(receipt)
        program = """
#include <stdio.h>
#include <sys/stat.h>
int main(void) {
    struct Runtime {
        struct { int rank, world_size, async; } hypers;
    } runtime = {{3, 8, 1}};
    struct Runtime* pufferl = &runtime;
    struct stat st = {0};
    st.st_size = 10755072;
    long long params = 2688768;
    const char* path = "/models/checkpoint.bin";
    if (!freopen("/dev/null", "w", stdout)) return 2;
""" + receipt.group(0) + """
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "load-receipt"
            subprocess.run(
                ["cc", "-std=c11", "-x", "c", "-", "-o", str(executable)],
                input=program, text=True, check=True, capture_output=True,
            )
            result = subprocess.run(
                [str(executable)], text=True, check=True, capture_output=True
            )
        self.assertEqual(result.stdout, "")
        self.assertIn("NATIVE_INITIAL_MODEL_LOADED rank=3 world_size=8", result.stderr)
        self.assertIn("async_actor_synced=1", result.stderr)

    def test_train_loads_primary_policy_and_syncs_async_actor(self):
        source = SOURCE.read_text()
        helper_start = source.index("void pufferl_load_initial_policy(")
        helper_end = source.index(
            "// fp32 master weights:", helper_start
        )
        helper = source[helper_start:helper_end]
        self.assertIn(
            'puf_checkpoint_path_key(\n'
            '        ini, "load_model_path"',
            helper,
        )
        self.assertIn("pufferl_load_policy(pufferl, 0, path);", helper)
        self.assertIn("&pufferl->actor_param, &primary->param", helper)
        self.assertIn("NATIVE_INITIAL_MODEL_LOADED", helper)

        train_start = source.index("TrainResult run_train(")
        train_end = source.index("TrainResult launch_train(", train_start)
        train = source[train_start:train_end]
        create = train.index("PuffeRL* pufferl = create_pufferl(ini, ctx);")
        load = train.index("pufferl_load_initial_policy(pufferl, ini);")
        self.assertLess(create, load)


if __name__ == "__main__":
    unittest.main()
