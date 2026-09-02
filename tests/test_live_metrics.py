import json
import pathlib
import subprocess
import textwrap


def test_live_metrics_publish_atomic_scalar_jsonl(tmp_path):
    source = tmp_path / "live_metrics_test.c"
    binary = tmp_path / "live_metrics_test"
    source.write_text(textwrap.dedent(
        r"""
        #include <math.h>
        #include "live_metrics.h"

        int main(int argc, char** argv) {
            assert(argc == 2);
            Dict log = {0};
            dict_set(&log, "SPS", 1234.5);
            dict_set(&log, "env/score", -2.25);
            dict_set(&log, "quoted\"key", 7.0);
            dict_set(&log, "_step", 999.0);
            dict_set(&log, "_timestamp", 999.0);
            dict_set(&log, "not_finite", NAN);
            dict_set_str(&log, "label", "ignored");

            puf_live_metrics_write(NULL, 10, &log);
            puf_live_metrics_write("None", 11, &log);
            puf_live_metrics_write(argv[1], 12, &log);
            dict_clear(&log);
            return 0;
        }
        """
    ))
    repo_root = pathlib.Path(__file__).parents[1]
    subprocess.run([
        "cc",
        "-std=c11",
        "-D_POSIX_C_SOURCE=200809L",
        f"-I{repo_root / 'src'}",
        str(source),
        "-o",
        str(binary),
    ], check=True)
    subprocess.run([str(binary), str(tmp_path)], check=True)

    chunks = sorted(tmp_path.glob("epoch-*.jsonl"))
    assert [chunk.name for chunk in chunks] == ["epoch-000012.jsonl"]
    assert not list(tmp_path.glob("*.tmp.*"))
    raw = chunks[0].read_text()
    assert raw.endswith("\n")
    assert raw.count("\n") == 1

    metric = json.loads(raw)
    assert metric["_step"] == 12
    assert metric["_timestamp"] > 0
    assert metric["SPS"] == 1234.5
    assert metric["env/score"] == -2.25
    assert metric['quoted"key'] == 7
    assert "not_finite" not in metric
    assert "label" not in metric
