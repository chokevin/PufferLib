from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "pufferl.cu"


class TrainWarmstartSourceTest(unittest.TestCase):
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
