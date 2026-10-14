"""Independent tree-role roots, shared period contracts and physical-file discovery."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from pih3l.config import DEFAULTS, load_config, period_configs, validate
from pih3l.errors import AnalysisError
from pih3l.io import discover_files


class PeriodConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()

    def write_config(self, value):
        path = self.base / "config/analysis.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(value))
        return path

    def touch(self, relative):
        path = self.base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"toy file-discovery input; no ROOT decoding")
        return path.resolve()

    def test_legacy_discovery_and_single_output_are_unchanged(self):
        file = self.touch("data/Femto/AO2D.root")
        path = self.write_config({"data_root": "../data", "output": "../out"})
        config = load_config(path)
        self.assertNotIn("root", config["inputs"]["femto"])
        self.assertEqual(discover_files(config, "femto"), [file])
        self.assertEqual(period_configs(config), [])
        self.assertEqual(config["output"], str(self.base / "out"))
        self.assertFalse(config["synthetic_repetition"])

    def test_independent_role_roots_resolve_without_repeating_role_folders(self):
        femto = self.touch("separate/pairs/Femto/AO2D.root")
        background = self.touch("separate/data_hyper/AO2D_1.root")
        signal = self.touch("separate/signal_mc/AO2D_2.root")
        config = load_config(
            self.write_config(
                {
                    "data_root": "../unused",
                    "inputs": {
                        "femto": {
                            "root": "../separate/pairs/Femto",
                            "patterns": ["AO2D*.root"],
                        },
                        "background": {
                            "root": "../separate/data_hyper",
                            "patterns": ["AO2D*.root"],
                        },
                        "signal": {
                            "root": "../separate/signal_mc",
                            "patterns": ["AO2D*.root"],
                        },
                        "mc_femto": {"root": None, "required": False},
                    },
                }
            )
        )
        self.assertEqual(discover_files(config, "femto"), [femto])
        self.assertEqual(discover_files(config, "background"), [background])
        self.assertEqual(discover_files(config, "signal"), [signal])
        self.assertEqual(discover_files(config, "mc_femto"), [])
        self.assertEqual(config["inputs"]["femto"]["root"], str(femto.parent))

    def test_multiple_roots_globs_symlinks_and_hard_links_use_each_physical_file_once(
        self,
    ):
        first = self.touch("one/Femto/AO2D_1.root")
        second = self.touch("two/Femto/AO2D_2.root")
        nested = self.touch("two/Femto/nested/AO2D_3.root")
        (second.parent / "AO2D_symlink.root").symlink_to(first)
        os.link(first, second.parent / "AO2D_hardlink.root")
        config = load_config(
            self.write_config(
                {
                    "inputs": {
                        "femto": {
                            "root": ["../one/Femto", "../two/Femto"],
                            "patterns": ["**/AO2D*.root", "AO2D*.root"],
                        }
                    }
                }
            )
        )
        paths = discover_files(config, "femto")
        self.assertEqual(len(paths), 3)
        self.assertEqual(
            {(path.stat().st_dev, path.stat().st_ino) for path in paths},
            {
                (path.stat().st_dev, path.stat().st_ino)
                for path in (first, second, nested)
            },
        )
        self.assertEqual(paths, discover_files(config, "femto"))

    def test_null_optional_input_cannot_fall_back_to_existing_legacy_files(self):
        self.touch("data/MCFemto/AO2D.root")
        config = copy.deepcopy(DEFAULTS)
        config["data_root"] = str(self.base / "data")
        config["inputs"]["mc_femto"]["root"] = None
        self.assertEqual(discover_files(config, "mc_femto"), [])
        config["inputs"]["mc_femto"]["required"] = True
        with self.assertRaisesRegex(AnalysisError, "null for a required"):
            validate(config)
        with self.assertRaisesRegex(AnalysisError, "explicitly disabled"):
            discover_files(config, "mc_femto")

    def test_period_overrides_resolve_to_yaml_and_keep_shared_physics_and_default_seed(
        self,
    ):
        path = self.write_config(
            {
                "data_root": "../data",
                "output": "../out",
                "seed": 71,
                "synthetic_repetition": True,
                "inputs": {
                    "femto": {"root": "../shared/Femto", "patterns": ["AO2D*.root"]}
                },
                "periods": [
                    {
                        "id": "p1",
                        "inputs": {"femto": {"root": "../p1/Femto"}},
                        "training": {"model_params": {"max_depth": 3}},
                        "working_point": {"yield_eff_range": [0.5, 0.9]},
                    },
                    {
                        "id": "p2",
                        "seed": 72,
                        "inputs": {"femto": {"root": ["../p2a/Femto", "../p2b/Femto"]}},
                    },
                ],
            }
        )
        config = load_config(path)
        before = copy.deepcopy(config)
        children = period_configs(config)
        self.assertEqual([id for id, _ in children], ["p1", "p2"])
        p1, p2 = [child for _, child in children]
        self.assertEqual(p1["seed"], 71)
        self.assertEqual(p2["seed"], 72)
        self.assertEqual(p1["output"], str(self.base / "out/periods/p1"))
        self.assertEqual(p2["output"], str(self.base / "out/periods/p2"))
        self.assertEqual(p1["inputs"]["femto"]["root"], str(self.base / "p1/Femto"))
        self.assertEqual(
            p2["inputs"]["femto"]["root"],
            [str(self.base / "p2a/Femto"), str(self.base / "p2b/Femto")],
        )
        self.assertEqual(p1["training"]["model_params"]["max_depth"], 3)
        for child in (p1, p2):
            self.assertEqual(child["periods"], [])
            self.assertTrue(child["synthetic_repetition"])
            for key in ("analysis", "physics", "systematics"):
                self.assertEqual(child[key], config[key])
        p1["analysis"]["candidate_cuts"]["DcaHe"] = {"min": 0.1}
        self.assertEqual(config, before)
        self.assertNotIn("DcaHe", p2["analysis"]["candidate_cuts"])

    def test_period_specs_reject_unsafe_ids_unknown_fields_and_invalid_merged_children(
        self,
    ):
        invalid = [
            [{"id": "../escape"}],
            [{"id": "Period"}],
            [{"id": "p"}, {"id": "p"}],
            [{"id": "p", "analysis": {}}],
            [{"id": "p", "physics": {}}],
            [{"id": "p", "systematics": {}}],
            [{"id": "p", "output": "/tmp/escape"}],
            [{"id": "p", "training": {"features": ["ct"]}}],
            [{"id": "p", "working_point": {"mass_fit": {"background": "expo"}}}],
            [{"id": "p", "inputs": {"femto": {"root": None}}}],
            [{"id": "p", "inputs": {"unknown": {}}}],
            [{"id": "p", "seed": True}],
            [{"id": "p", "training": "not a mapping"}],
            [None],
            {"p": {}},
            None,
        ]
        for periods in invalid:
            with self.subTest(periods=periods):
                config = copy.deepcopy(DEFAULTS)
                config["periods"] = periods
                with self.assertRaises(AnalysisError):
                    validate(config)

    def test_root_schema_and_synthetic_marker_fail_early(self):
        for root in ("", " ", [], ["/tmp", None], [1], 1, {}, Path("/tmp")):
            with self.subTest(root=root):
                config = copy.deepcopy(DEFAULTS)
                config["inputs"]["femto"]["root"] = root
                with self.assertRaises(AnalysisError):
                    validate(config)
        config = copy.deepcopy(DEFAULTS)
        config["synthetic_repetition"] = "true"
        with self.assertRaisesRegex(AnalysisError, "synthetic_repetition"):
            validate(config)

    def test_environment_expansion_and_absolute_patterns_remain_supported(self):
        absolute = self.touch("absolute/AO2D.root")
        self.touch("env/Femto/AO2D.root")
        with patch.dict(
            os.environ, {"PIH3L_PERIOD_INPUT": str(self.base / "env")}
        ), patch("os.path.expanduser", wraps=os.path.expanduser):
            config = load_config(
                self.write_config(
                    {
                        "inputs": {
                            "femto": {
                                "root": "$PIH3L_PERIOD_INPUT/Femto",
                                "patterns": [str(absolute)],
                            }
                        }
                    }
                )
            )
        self.assertEqual(
            config["inputs"]["femto"]["root"], str(self.base / "env/Femto")
        )
        self.assertEqual(discover_files(config, "femto"), [absolute])

    def test_examples_keep_single_testset_and_explicit_repeated_period_semantics(self):
        project = Path(__file__).resolve().parents[1]
        single = load_config(project / "configs/testset.yaml")
        repeated = load_config(project / "configs/testset.periods.yaml")
        production = load_config(project / "configs/production.example.yaml")
        self.assertEqual(period_configs(single), [])
        self.assertEqual(Path(single["output"]).name, "testset")
        self.assertIsNone(single["inputs"]["mc_femto"]["root"])
        self.assertFalse(single["inputs"]["mc_femto"]["required"])
        children = period_configs(repeated)
        self.assertEqual(
            [id for id, _ in children],
            ["test_period_1", "test_period_2", "test_period_3"],
        )
        self.assertTrue(repeated["synthetic_repetition"])
        for id, child in children:
            self.assertEqual(child["inputs"], single["inputs"])
            self.assertEqual(child["training"], single["training"])
            self.assertEqual(child["working_point"], single["working_point"])
            self.assertEqual(child["seed"], single["seed"])
            self.assertEqual(
                Path(child["output"]), Path(repeated["output"]) / "periods" / id
            )
        self.assertGreaterEqual(len(period_configs(production)), 1)
        for role in single["inputs"]:
            self.assertEqual(single["inputs"][role]["patterns"], ["AO2D*.root"])


if __name__ == "__main__":
    unittest.main()
