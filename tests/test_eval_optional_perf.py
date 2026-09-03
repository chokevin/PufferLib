from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_eval_requires_score_but_allows_missing_perf():
    source = (ROOT / "src" / "pufferl.cu").read_text()
    eval_loop = source[
        source.index("EvalResult eval_loop(") : source.index(
            "PuffeRL* eval_make("
        )
    ]

    assert 'dict_get(&el, "env/score")' in eval_loop
    assert 'result.perf = dash_num(&el, "env/perf", 0.0);' in eval_loop
