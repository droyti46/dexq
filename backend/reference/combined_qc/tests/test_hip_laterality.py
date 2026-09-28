"""Automatic side detection, positioning mirroring and public hip output."""
import base64
from io import BytesIO
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from combined_qc.hip import pipeline
from combined_qc.hip.preprocessing import letterbox
from combined_qc.hip.runtime import HipONNXPredictor


class RecordedSession:
    def __init__(self):
        self.inputs = []

    def run(self, outputs, inputs):
        self.inputs.append({name: value.copy() for name, value in inputs.items()})
        return [np.asarray([.2], dtype=np.float32)]


class AutomaticHipLateralityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.native = np.full((48, 62), 210, dtype=np.uint8)
        self.native[:, :31] = 30
        self.path = self.root / "unrelated_name.png"
        Image.fromarray(self.native).save(self.path)
        self.side_model = types.SimpleNamespace(
            predict_array=MagicMock(), provenance={"test_side_model": True})
        self.handle = HipONNXPredictor.__new__(HipONNXPredictor)
        self.handle.device = "cpu"
        self.handle.checkpoints_dir = self.root / "checkpoints"
        self.handle.config = {
            "image_size": 320,
            "geometry_feature_names": [f"pixel_{i}" for i in range(8)],
            "acquisition_feature_names": [f"acquisition_{i}" for i in range(7)],
        }
        self.handle.mean = np.zeros(3, dtype=np.float32)
        self.handle.std = np.ones(3, dtype=np.float32)
        self.handle.inventory = {
            "positioning_rotation": [{"fold": 0, "file": "position.onnx", "threshold": .5}],
            "roi": [{"fold": 0, "file": "roi.onnx", "threshold": .5}],
        }
        self.handle.sessions = {name: RecordedSession() for name in ("position.onnx", "roi.onnx")}
        self.handle.laterality_model = self.side_model
        self.handle.provenance = {"test_quality_model": True}
        self.handle.predict_prepared = MagicMock(wraps=self.handle.predict_prepared)
        self.call_order = MagicMock()
        self.call_order.attach_mock(self.side_model.predict_array, "side")
        self.call_order.attach_mock(self.handle.predict_prepared, "quality")
        self.side_model.predict_array.return_value = self._prediction("left_hip")
        patch.object(pipeline, "_known_records", return_value={}).start()
        patch.object(HipONNXPredictor, "_providers", return_value=["CPUExecutionProvider"]).start()
        self.addCleanup(patch.stopall)

    @staticmethod
    def _prediction(side, confidence=.99):
        invalid = side is None
        return {
            "source": "classifier-source", "laterality": side,
            "label": None if invalid else int(side == "right_hip"),
            "confidence": None if invalid else confidence,
            "scores": None if invalid else {
                "left_hip": confidence if side == "left_hip" else 1 - confidence,
                "right_hip": confidence if side == "right_hip" else 1 - confidence,
            },
            "needs_review": invalid or confidence < .9,
            "metadata": {"status": "invalid_image" if invalid else "classified",
                         "label_interpretation": "Provisional visual side labels",
                         "provenance": {"test_side_model": True}},
        }

    def test_native_image_is_classified_before_quality_inference(self):
        prediction = self._prediction("left_hip")
        self.side_model.predict_array.return_value = prediction
        result = pipeline.infer(self.path, model=self.handle)
        self.side_model.predict_array.assert_called_once()
        args, kwargs = self.side_model.predict_array.call_args
        np.testing.assert_array_equal(args[0], self.native)
        self.assertEqual(Path(kwargs["source"]), self.path)
        self.assertEqual(result["laterality"], "left_hip")
        self.assertEqual(result["laterality_prediction"], prediction)
        self.assertEqual(result["metadata"]["laterality_source"], "model")
        self.assertEqual(result["metadata"]["laterality_confidence"], .99)
        self.assertFalse(result["needs_review"])
        self.assertIsNotNone(result["annotated_image"])
        self.assertEqual([call[0] for call in self.call_order.mock_calls], ["side", "quality"])

    def test_automatic_right_side_mirrors_positioning_only(self):
        self.side_model.predict_array.return_value = self._prediction("right_hip")
        result = pipeline.infer(self.path, model=self.handle)
        prepared, _ = letterbox(self.native)
        positioning = self.handle.sessions["position.onnx"].inputs[0]["pixels"][0, 0]
        roi = self.handle.sessions["roi.onnx"].inputs[0]["pixels"][0, 0]
        np.testing.assert_array_equal(positioning, prepared[:, ::-1])
        np.testing.assert_array_equal(roi, prepared)
        self.assertTrue(result["geometry"]["positioning_mirrored"])
        self.assertEqual(self.handle.predict_prepared.call_args.args[2], "right_hip")

    def test_automatic_left_side_keeps_positioning_orientation(self):
        result = pipeline.infer(self.path, model=self.handle)
        prepared, _ = letterbox(self.native)
        positioning = self.handle.sessions["position.onnx"].inputs[0]["pixels"][0, 0]
        np.testing.assert_array_equal(positioning, prepared)
        self.assertFalse(result["geometry"]["positioning_mirrored"])

    def test_known_manifest_side_never_overrides_classifier_prediction(self):
        row = {"anatomical_region": "left_hip", "image_path": "images/known.png",
               "image_id": "known-id", "study_id": "known-study"}
        self.side_model.predict_array.return_value = self._prediction("right_hip")
        with patch.object(pipeline, "_known_records", return_value={str(self.path): row}):
            result = pipeline.infer(self.path, model=self.handle)
        self.assertEqual(result["laterality"], "right_hip")
        self.assertEqual(result["metadata"]["laterality_source"], "model")
        self.assertEqual(result["metadata"]["image_id"], "known-id")
        self.assertTrue(result["geometry"]["positioning_mirrored"])
        self.side_model.predict_array.assert_called_once()

    def test_uncertain_valid_side_preserves_qc_and_reports_review_flag(self):
        self.side_model.predict_array.return_value = self._prediction("right_hip", .6)
        result = pipeline.infer(self.path, model=self.handle)
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["laterality_prediction"]["confidence"], .6)
        self.handle.predict_prepared.assert_called_once()
        self.assertTrue(all(value is not None for value in result["labels"].values()))
        self.assertIsNotNone(result["any_violation"])
        self.assertIsNotNone(result["annotated_image"])

    def test_invalid_side_never_runs_quality_models_and_still_returns_visualization(self):
        Image.fromarray(np.zeros_like(self.native)).save(self.path)
        self.side_model.predict_array.return_value = self._prediction(None)
        result = pipeline.infer(self.path, model=self.handle)
        self.handle.predict_prepared.assert_not_called()
        self.assertTrue(all(not session.inputs for session in self.handle.sessions.values()))
        self.assertEqual(result["status"], "invalid_image")
        self.assertIsNone(result["laterality"])
        self.assertTrue(result["needs_review"])
        self.assertTrue(all(value is None for value in result["labels"].values()))
        self.assertTrue(all(value is None for value in result["scores"].values()))
        self.assertIsNone(result["any_violation"])
        with Image.open(BytesIO(base64.b64decode(result["annotated_image"]["data"], validate=True))) as image:
            self.assertEqual(image.format, "PNG")

    def test_repeated_inference_reuses_quality_and_side_handles(self):
        identities = (id(self.handle.laterality_model),
                      tuple(id(session) for session in self.handle.sessions.values()))
        with patch.object(pipeline, "load_checkpoints") as loader:
            first = pipeline.infer(self.path, model=self.handle)
            second = pipeline.infer(self.path, model=self.handle)
            loader.assert_not_called()
        self.assertEqual(self.side_model.predict_array.call_count, 2)
        self.assertEqual(first["laterality_prediction"], second["laterality_prediction"])
        self.assertEqual(identities, (id(self.handle.laterality_model),
                         tuple(id(session) for session in self.handle.sessions.values())))

    def test_mixed_sides_in_sorted_folder_are_detected_from_each_native_image(self):
        right_path = self.root / "a.png"
        Image.fromarray(self.native[:, ::-1]).save(right_path)

        def detect(pixels, *, source):
            side = "left_hip" if pixels[:, :31].mean() < pixels[:, 31:].mean() else "right_hip"
            return self._prediction(side)

        self.side_model.predict_array.side_effect = detect
        result = pipeline.infer(self.root, model=self.handle)
        self.assertEqual([Path(row["source"]).name for row in result], ["a.png", "unrelated_name.png"])
        self.assertEqual([row["laterality"] for row in result], ["right_hip", "left_hip"])
        self.assertEqual([row["geometry"]["positioning_mirrored"] for row in result], [True, False])

    def test_wrong_checkpoint_handle_stops_before_side_or_quality_prediction(self):
        with self.assertRaises(ValueError):
            pipeline.infer(self.path, model=self.handle, checkpoints_dir=self.root / "other-checkpoints")
        self.side_model.predict_array.assert_not_called()
        self.handle.predict_prepared.assert_not_called()


if __name__ == "__main__":
    unittest.main()
