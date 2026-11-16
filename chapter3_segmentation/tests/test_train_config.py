"""Configuration contracts for original Meta SAM3 training, runnable without CUDA."""
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sam3_training", ROOT / "train.py")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


class Sam3TrainingConfigTests(unittest.TestCase):
    def test_mask_training_and_validation_contract(self):
        args = training.parse_args(["--sam3-root", "/opt/sam3", "--epochs", "7"])
        cfg = training.build_config(args)
        self.assertEqual(cfg["trainer"]["max_epochs"], 7)
        self.assertTrue(cfg["scratch"]["enable_segmentation"])
        self.assertFalse(cfg["trainer"]["skip_saving_ckpts"])
        self.assertEqual(cfg["trainer"]["_target_"], "chapter3_segmentation.train.build_trainer")
        self.assertEqual(cfg["trainer"]["model"]["_target_"], "chapter3_segmentation.train.build_train_model")
        self.assertEqual(cfg["trainer"]["model"]["device"], "cpu")
        self.assertFalse(cfg["trainer"]["model"]["load_from_HF"])
        self.assertEqual(cfg["launcher"]["gpus_per_node"], 5)
        self.assertFalse(cfg["submitit"]["use_cluster"])
        self.assertEqual(cfg["trainer"]["data"]["train"]["dataset"]["ann_file"],
                         (Path(args.data_dir).resolve() / "annotations/train.json").as_posix())
        self.assertEqual(cfg["trainer"]["data"]["val"]["dataset"]["ann_file"],
                         (Path(args.data_dir).resolve() / "annotations/val.json").as_posix())
        for split in ("train", "val"):
            ds = cfg["trainer"]["data"][split]["dataset"]
            self.assertEqual(ds["transforms"][0]["transforms"][0]["_target_"],
                             "sam3.train.transforms.segmentation.DecodeRle")
            self.assertTrue(ds["load_segmentation"])
            self.assertTrue(cfg["trainer"]["data"][split]["collate_fn"]["with_seg_masks"])
        losses = cfg["trainer"]["loss"]["all"]
        self.assertTrue(any(x["_target_"].endswith(".Masks") for x in losses["loss_fns_find"]))
        self.assertEqual(losses["loss_fn_semantic_seg"]["_target_"],
                         "sam3.train.loss.loss_fns.SemanticSegCriterion")
        self.assertEqual(cfg["trainer"]["meters"]["val"]["tea"]["segmentation"]["iou_type"], "segm")

    def test_dry_run_writes_portable_configuration_without_model_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = training.main(["--sam3-root", str(Path(tmp) / "missing-sam3"),
                                    "--output-dir", tmp, "--dry-run"])
            self.assertEqual(result, 0)
            self.assertTrue((Path(tmp) / "sam3_train.yaml").is_file())
            self.assertTrue((Path(tmp) / "preflight.json").is_file())

    def test_invalid_batch_size_rejected(self):
        with self.assertRaises(SystemExit):
            training.parse_args(["--sam3-root", "/opt/sam3", "--batch-size", "0"])

    def test_hydra_interpolation_references_are_complete(self):
        cfg = training.build_config(training.parse_args([]))
        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, str):
                for reference in re.findall(r"\$\{([A-Za-z_][\w.]*)\}", value):
                    owner = cfg
                    for part in reference.split("."):
                        self.assertIn(part, owner, reference)
                        owner = owner[part]
        walk(cfg)

    def test_five_gpu_configuration_and_launch_use_same_world_size(self):
        args = training.parse_args(["--num-gpus", "5", "--batch-size", "2"])
        cfg = training.build_config(args)
        self.assertEqual(cfg["launcher"]["gpus_per_node"], 5)
        self.assertEqual(cfg["scratch"]["train_batch_size"], 2)
        self.assertEqual(cfg["scratch"]["gradient_accumulation_steps"], 1)
        self.assertEqual(training.batch_configuration(args)["effective_global_batch_size"], 10)
        command = training.build_launch_command(args, Path("/tmp/run/sam3_train.yaml"))
        self.assertEqual(command[0], training.sys.executable)
        self.assertEqual(command[-1], "5")
        self.assertIn("num_gpus=int(sys.argv[2])", command[2])
        self.assertNotIn("torchrun", command)
        self.assertNotIn("torch.distributed.run", command)

    def test_single_gpu_override_preserves_learning_rates(self):
        one = training.build_config(training.parse_args(["--num-gpus", "1"]))
        five = training.build_config(training.parse_args(["--num-gpus", "5"]))
        self.assertEqual(one["launcher"]["gpus_per_node"], 1)
        for field in ("lr_scale", "lr_transformer", "lr_vision_backbone", "lr_language_backbone"):
            self.assertEqual(one["scratch"][field], five["scratch"][field])

    def test_gpu_inventory_checks_every_selected_device(self):
        devices = [{"index": i, "name": "RTX 4090", "total_gib": 24,
                    "capability": [8, 9], "free_gib": 22} for i in range(5)]
        report = training.assess_gpus(devices, 5)
        self.assertEqual(report["blockers"], [])
        self.assertEqual(len(report["selected_gpus"]), 5)
        self.assertTrue(training.assess_gpus(devices[:4], 5)["blockers"])
        devices[4] = dict(devices[4], capability=[7, 5], total_gib=8)
        report = training.assess_gpus(devices, 5)
        self.assertTrue(any("4" in reason and "bfloat16" in reason for reason in report["blockers"]))
        self.assertTrue(any("4" in reason and "memory" in reason for reason in report["warnings"]))

    def test_nonpositive_gpu_count_rejected(self):
        with self.assertRaises(SystemExit):
            training.parse_args(["--num-gpus", "0"])


if __name__ == "__main__":
    unittest.main()
