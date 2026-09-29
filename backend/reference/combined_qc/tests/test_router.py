"""Coordinator behavior with child models replaced by lightweight test doubles."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
import copy
import inspect
import io
from pathlib import Path
import tempfile
from threading import Barrier
import time
import types
import unittest
from unittest.mock import MagicMock, patch


class RouterTests(unittest.TestCase):
    def setUp(self):
        from combined_qc.router import coordinator
        self.coordinator = coordinator
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.classifications = {}
        self.handles = {
            name: types.SimpleNamespace(provenance={"test_model": name})
            for name in ("classifier", "hip", "spine")
        }
        self.responses = {
            "hip": {
                "region": "hip", "source": "child-hip", "labels": {"roi": 0},
                "scores": {"roi": 0.1}, "any_violation": 0,
                "laterality": "left_hip", "needs_review": False,
                "laterality_prediction": {"laterality": "left_hip", "confidence": 0.99,
                                          "scores": {"left_hip": 0.99, "right_hip": 0.01}},
                "metadata": {"laterality": "left_hip", "laterality_source": "model", "model": "hip-test"},
                "geometry": {"diagnostic_region": [1, 2, 3, 4]},
                "annotated_image": {"encoding": "base64_png", "data": "hip-png"},
            },
            "spine": {
                "region": "spine", "source": "child-spine", "labels": {"axis": 1},
                "scores": {"axis": 0.9}, "any_violation": 1,
                "metadata": {"model": "spine-test"},
                "geometry": {"vertebra_centers": [[5, 6], [7, 8]]},
                "annotated_image": {"encoding": "base64_png", "data": "spine-png"},
            },
        }
        self.modules = {}
        self.converters = {}
        for name in self.handles:
            self.modules[name] = types.SimpleNamespace(
                load_checkpoints=MagicMock(return_value=self.handles[name]),
                infer=MagicMock(), train=MagicMock(return_value={"trained": name}),
            )
            self.converters[name] = types.SimpleNamespace(
                convert_all=MagicMock(return_value={"converted": name}),
            )
        self.modules["classifier"].infer.side_effect = self._classify
        for name in ("hip", "spine"):
            self.modules[name].infer.side_effect = (
                lambda *args, _name=name, **kwargs: copy.deepcopy(self.responses[_name])
            )
        self.addCleanup(patch.stopall)
        patch.object(coordinator, "_module", side_effect=self.modules.__getitem__).start()
        patch.object(coordinator, "_converter", side_effect=self.converters.__getitem__).start()

    def _classify(self, source, **kwargs):
        return copy.deepcopy(self.classifications[Path(source).name])

    def _image(self, name, region, *, needs_review=False, confidence=0.99):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        self.classifications[path.name] = {
            "source": str(path), "region": region,
            "label": None if region is None else int(region == "lumbar_spine"),
            "confidence": None if region is None else confidence,
            "scores": None if region is None else {
                "hip": 0.99 if region == "hip" else 0.01,
                "lumbar_spine": 0.99 if region == "lumbar_spine" else 0.01,
            },
            "needs_review": needs_review,
            "metadata": {"status": "invalid_image" if region is None else "classified"},
        }
        return path

    def _pipeline(self, **kwargs):
        kwargs.setdefault("checkpoints_dir", self.root / "classifier_checkpoints")
        kwargs.setdefault("hip_checkpoints_dir", self.root / "hip_checkpoints")
        kwargs.setdefault("spine_checkpoints_dir", self.root / "spine_checkpoints")
        return self.coordinator.QCPipeline(**kwargs)

    def test_explicit_load_all_is_idempotent(self):
        pipeline = self._pipeline()
        self.assertIs(pipeline.load_checkpoints(), pipeline)
        self.assertIs(pipeline.load_checkpoints(), pipeline)
        self.assertEqual(set(pipeline.models), {"classifier", "hip", "spine"})
        for name, module in self.modules.items():
            self.assertIs(pipeline.models[name], self.handles[name])
            module.load_checkpoints.assert_called_once()

    def test_classifier_functions_match_profile_interfaces(self):
        from combined_qc import hip, router, spine
        for name in ("train", "load_checkpoints", "infer"):
            with self.subTest(function=name):
                expected = inspect.signature(getattr(hip, name))
                self.assertEqual(inspect.signature(getattr(spine, name)), expected)
                self.assertEqual(inspect.signature(getattr(router, name)), expected)

    def test_concurrent_first_load_creates_each_handle_once(self):
        pipeline = self._pipeline()
        start = Barrier(4)
        for name, module in self.modules.items():
            def slow_load(*args, _name=name, **kwargs):
                time.sleep(0.005)
                return self.handles[_name]
            module.load_checkpoints.side_effect = slow_load

        def worker(_):
            start.wait(timeout=5)
            return pipeline.load_checkpoints()

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(4)))
        self.assertTrue(all(result is pipeline for result in results))
        for name, module in self.modules.items():
            module.load_checkpoints.assert_called_once()
            self.assertIs(pipeline.models[name], self.handles[name])

    def test_concurrent_first_inference_reuses_handles(self):
        path = self._image("concurrent_spine.png", "lumbar_spine")
        pipeline = self._pipeline()
        start = Barrier(4)
        for name in ("classifier", "spine"):
            def slow_load(*args, _name=name, **kwargs):
                time.sleep(0.005)
                return self.handles[_name]
            self.modules[name].load_checkpoints.side_effect = slow_load

        def worker(_):
            start.wait(timeout=5)
            return pipeline.infer(path)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(4)))
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result["routing"]["target_module"] == "spine" for result in results))
        for name in ("classifier", "spine"):
            self.modules[name].load_checkpoints.assert_called_once()
            self.assertEqual(self.modules[name].infer.call_count, 4)
        self.modules["hip"].load_checkpoints.assert_not_called()

    def test_successive_inference_reuses_classifier_and_profile_handles(self):
        first = self._image("first.png", "lumbar_spine")
        second = self._image("second.dcm", "lumbar_spine")
        pipeline = self._pipeline()
        pipeline.infer(first)
        pipeline.infer(second)
        self.modules["classifier"].load_checkpoints.assert_called_once()
        self.modules["spine"].load_checkpoints.assert_called_once()
        self.modules["hip"].load_checkpoints.assert_not_called()
        for call in self.modules["classifier"].infer.call_args_list:
            self.assertIs(call.kwargs["model"], self.handles["classifier"])
        for call in self.modules["spine"].infer.call_args_list:
            self.assertIs(call.kwargs["model"], self.handles["spine"])

    def test_routing_preserves_complete_child_output(self):
        for region, module in (("hip", "hip"), ("lumbar_spine", "spine")):
            with self.subTest(region=region):
                path = self._image(f"{module}.png", region)
                result = self._pipeline().infer(path)
                for key, value in self.responses[module].items():
                    self.assertEqual(result[key], value)
                self.assertEqual(result["routing"]["region"], region)
                self.assertEqual(result["routing"]["confidence"], 0.99)
                self.assertEqual(result["routing"]["target_module"], module)
        self.assertNotIn("laterality", self.modules["hip"].infer.call_args.kwargs)

    def test_mixed_folder_sorted_and_recursive_only_when_requested(self):
        self._image("c_hip.png", "hip")
        self._image("a_spine.dcm", "lumbar_spine")
        self._image("b_hip.png", "hip")
        self._image("nested/d_spine.png", "lumbar_spine")
        (self.root / "notes.json").write_text("{}")
        pipeline = self._pipeline()
        result = pipeline.infer(self.root)
        self.assertIsInstance(result, list)
        self.assertEqual([Path(x["routing"]["source"]).name for x in result],
                         ["a_spine.dcm", "b_hip.png", "c_hip.png"])
        self.assertEqual([x["routing"]["target_module"] for x in result],
                         ["spine", "hip", "hip"])
        recursive = pipeline.infer(self.root, recursive=True)
        self.assertEqual(len(recursive), 4)
        for module in self.modules.values():
            module.load_checkpoints.assert_called_once()

    def test_uncertain_or_invalid_image_never_runs_quality_control(self):
        pipeline = self._pipeline()
        for name, region in (("uncertain.png", "hip"), ("invalid.png", None)):
            with self.subTest(region=region):
                path = self._image(name, region, needs_review=True, confidence=0.6)
                result = pipeline.infer(path)
                self.assertEqual(result["status"], "invalid_image" if region is None else "needs_review")
                self.assertEqual(result["labels"], {})
                self.assertIsNone(result["annotated_image"])
                self.assertIsNone(result["any_violation"])
                self.assertTrue(result["routing"]["needs_review"])
        for name in ("hip", "spine"):
            self.modules[name].infer.assert_not_called()
            self.modules[name].load_checkpoints.assert_not_called()

    def test_hip_laterality_prediction_is_returned_without_manual_input(self):
        path = self._image("unknown_side.png", "hip")
        pipeline = self._pipeline()
        result = pipeline.infer(path)
        self.assertEqual(result["laterality"], "left_hip")
        self.assertEqual(result["laterality_prediction"], self.responses["hip"]["laterality_prediction"])
        self.assertEqual(result["metadata"]["laterality_source"], "model")
        self.assertNotIn("laterality", self.modules["hip"].infer.call_args.kwargs)
        self.assertEqual(result["routing"]["target_module"], "hip")

    def test_mixed_left_right_and_spine_folder_needs_no_side_argument(self):
        self._image("a_left.png", "hip")
        self._image("b_right.png", "hip")
        self._image("c_spine.png", "lumbar_spine")

        def automatic_side(source, **kwargs):
            self.assertNotIn("laterality", kwargs)
            result = copy.deepcopy(self.responses["hip"])
            side = "left_hip" if Path(source).name == "a_left.png" else "right_hip"
            result["laterality"] = side
            result["metadata"]["laterality"] = side
            result["laterality_prediction"]["laterality"] = side
            return result

        self.modules["hip"].infer.side_effect = automatic_side
        result = self._pipeline().infer(self.root)
        self.assertEqual(len(result), 3)
        self.assertEqual([row.get("laterality") for row in result], ["left_hip", "right_hip", None])
        self.assertEqual(result[2]["labels"], self.responses["spine"]["labels"])
        self.modules["spine"].infer.assert_called_once()

    def test_unrelated_profile_errors_are_not_disguised_as_side_errors(self):
        path = self._image("broken_hip.png", "hip")
        self.modules["hip"].infer.side_effect = ValueError("corrupt model output")
        with self.assertRaisesRegex(ValueError, "corrupt model output"):
            self._pipeline().infer(path)

    def test_inference_flags_override_constructor_defaults(self):
        path = self._image("quiet.png", "lumbar_spine")
        pipeline = self._pipeline(verbose=True, ci=True)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            pipeline.infer(path, verbose=False, ci=False)
        self.assertEqual(output.getvalue(), "")
        for name in ("classifier", "spine"):
            for operation in ("load_checkpoints", "infer"):
                kwargs = getattr(self.modules[name], operation).call_args.kwargs
                self.assertFalse(kwargs["verbose"])
                self.assertFalse(kwargs["ci"])
        with contextlib.redirect_stderr(output):
            pipeline.infer(path)
        self.assertIn("[router]", output.getvalue())
        self.assertTrue(self.modules["classifier"].infer.call_args.kwargs["verbose"])
        self.assertTrue(self.modules["classifier"].infer.call_args.kwargs["ci"])

    def test_train_invalidates_only_completed_runtime_replacements(self):
        pipeline = self._pipeline().load_checkpoints()
        self.modules["hip"].train.side_effect = RuntimeError("injected hip training failure")
        with self.assertRaisesRegex(RuntimeError, "hip training failure"):
            pipeline.train_all()
        self.assertNotIn("classifier", pipeline.models)
        self.assertIs(pipeline.models["hip"], self.handles["hip"])
        self.assertIs(pipeline.models["spine"], self.handles["spine"])

    def test_source_only_training_keeps_previous_runtime_handles(self):
        pipeline = self._pipeline().load_checkpoints()
        pipeline.train_all(convert=False, epochs={"hip": 1})
        for name in self.handles:
            self.assertIs(pipeline.models[name], self.handles[name])
            self.assertFalse(self.modules[name].train.call_args.kwargs["convert"])
        self.assertEqual(self.modules["hip"].train.call_args.kwargs["epochs"], 1)
        self.assertIsNone(self.modules["spine"].train.call_args.kwargs["epochs"])

    def test_convert_invalidates_only_successful_exports(self):
        pipeline = self._pipeline().load_checkpoints()
        self.converters["hip"].convert_all.side_effect = RuntimeError("injected export failure")
        with self.assertRaisesRegex(RuntimeError, "export failure"):
            pipeline.convert_all()
        self.assertNotIn("classifier", pipeline.models)
        self.assertIs(pipeline.models["hip"], self.handles["hip"])
        self.assertIs(pipeline.models["spine"], self.handles["spine"])

    def _publication_then_error(self, pipeline, name, relative="manifest.json"):
        manifest = pipeline.checkpoint_dirs[name] / "runtime" / relative
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"generation": 1}')

        def operation(*args, **kwargs):
            manifest.write_text('{"generation": 2}')
            raise OSError("report write failed")

        return operation

    def test_train_failure_after_publication_discards_changed_handle(self):
        pipeline = self._pipeline().load_checkpoints()
        self.modules["classifier"].train.side_effect = self._publication_then_error(pipeline, "classifier")
        with self.assertRaisesRegex(OSError, "report write failed"):
            pipeline.train_all()
        self.assertNotIn("classifier", pipeline.models)
        self.assertEqual(pipeline.states["classifier"]["last_operation"], "train_published_before_error")
        self.assertIs(pipeline.models["hip"], self.handles["hip"])
        self.assertIs(pipeline.models["spine"], self.handles["spine"])
        self.modules["hip"].train.assert_not_called()

    def test_conversion_failure_after_publication_discards_changed_handle(self):
        pipeline = self._pipeline().load_checkpoints()
        self.converters["classifier"].convert_all.side_effect = self._publication_then_error(pipeline, "classifier")
        with self.assertRaisesRegex(OSError, "report write failed"):
            pipeline.convert_all()
        self.assertNotIn("classifier", pipeline.models)
        self.assertEqual(pipeline.states["classifier"]["last_operation"], "convert_published_before_error")
        self.assertIs(pipeline.models["hip"], self.handles["hip"])
        self.assertIs(pipeline.models["spine"], self.handles["spine"])
        self.converters["hip"].convert_all.assert_not_called()

    def test_nested_spine_manifest_failure_discards_stale_spine(self):
        pipeline = self._pipeline().load_checkpoints()
        self.converters["spine"].convert_all.side_effect = self._publication_then_error(
            pipeline, "spine", "classifiers/model_manifest.json")
        with self.assertRaisesRegex(OSError, "report write failed"):
            pipeline.convert_all()
        self.assertNotIn("spine", pipeline.models)
        self.assertEqual(pipeline.states["spine"]["last_operation"], "convert_published_before_error")


if __name__ == "__main__":
    unittest.main()
