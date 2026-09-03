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


def test_metric_curves_allow_sparse_final_rows():
    source = (ROOT / "src" / "pufferl.cu").read_text()
    history = source[
        source.index("static void log_history_bin_mean(") : source.index(
            "double rollout_start("
        )
    ]

    assert "DictItem* item = dict_find(log, key);" in history
    assert "out[points - 1] = latest;" in history
    assert "bin_sum += dict_get(log, key);" not in history


def test_failed_sweep_workers_count_toward_run_limit():
    source = (ROOT / "src" / "pufferl.cu").read_text()
    failure = source[
        source.index('fprintf(stderr, "sweep worker run=%d failed;') :
        source.index("// points[]:")
    ]

    assert "completed++;" in failure


def test_train_result_uses_latest_nonempty_objective_row():
    source = (ROOT / "src" / "pufferl.cu").read_text()
    result = source[
        source.index("// TrainResult curve:") : source.index(
            "int points = use_selfplay"
        )
    ]

    assert "else if (log_history.size > 0)" in result
    assert "log_history_bin_mean(&log_history, target_key, 1, &latest_score);" in result
