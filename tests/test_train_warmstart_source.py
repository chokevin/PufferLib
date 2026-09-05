from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "pufferl.cu"


class TrainWarmstartSourceTest(unittest.TestCase):
    def receipt_program(self, main):
        source = SOURCE.read_text()
        helper = source.split("void pufferl_load_initial_policy(", 1)[1].split(
            "// fp32 master weights:", 1
        )[0]
        receipt = re.search(
            r'char receipt\[PIPE_BUF\];.*?'
            r'assert\(written == receipt_length.*?\);',
            helper,
            re.DOTALL,
        )
        self.assertIsNotNone(receipt)
        return """
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
static void emit(int rank, const char* path) {
    struct Runtime {
        struct { int rank, world_size, async; } hypers;
    } runtime = {{3, 8, 1}};
    struct Runtime* pufferl = &runtime;
    runtime.hypers.rank = rank;
    struct stat st = {0};
    st.st_size = 10755072;
    long long params = 2688768;
""" + receipt.group(0) + """
}
""" + main

    def compile_and_run(self, program):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "load-receipt"
            subprocess.run(
                ["cc", "-std=c11", "-x", "c", "-", "-o", str(executable)],
                input=program, text=True, check=True, capture_output=True,
            )
            return subprocess.run(
                [str(executable)], text=True, check=True, capture_output=True
            )

    def test_load_receipt_survives_nonzero_rank_stdout_redirection(self):
        program = self.receipt_program("""
int main(void) {
    if (!freopen("/dev/null", "w", stdout)) return 2;
    emit(3, "/models/checkpoint.bin");
    return 0;
}
""")
        result = self.compile_and_run(program)
        self.assertEqual(result.stdout, "")
        self.assertIn("NATIVE_INITIAL_MODEL_LOADED rank=3 world_size=8", result.stderr)
        self.assertIn("async_actor_synced=1", result.stderr)

    def test_concurrent_rank_receipts_are_complete(self):
        path = (
            "/data/tau-workspaces/default/kaggriculture/direct/"
            "representative-long-named-training-run/initial-model.bin"
        )
        program = self.receipt_program("""
int main(void) {
    int gate[2];
    pid_t children[8];
    assert(pipe(gate) == 0);
    for (int rank = 0; rank < 8; rank++) {
        children[rank] = fork();
        assert(children[rank] >= 0);
        if (children[rank] == 0) {
            close(gate[1]);
            char ready;
            assert(read(gate[0], &ready, 1) == 1);
            for (int n = 0; n < 256; n++)
                emit(rank, "CHECKPOINT_PATH");
            _exit(0);
        }
    }
    close(gate[0]);
    assert(write(gate[1], "12345678", 8) == 8);
    close(gate[1]);
    for (int rank = 0; rank < 8; rank++) {
        int status;
        assert(waitpid(children[rank], &status, 0) == children[rank]);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    return 0;
}
""".replace("CHECKPOINT_PATH", path))
        result = self.compile_and_run(program)
        self.assertEqual(result.stdout, "")
        lines = result.stderr.splitlines()
        self.assertEqual(len(lines), 8 * 256)
        for rank in range(8):
            expected = (
                f"NATIVE_INITIAL_MODEL_LOADED rank={rank} world_size=8 "
                f"path={path} bytes=10755072 params=2688768 async_actor_synced=1"
            )
            self.assertEqual(lines.count(expected), 256)

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
