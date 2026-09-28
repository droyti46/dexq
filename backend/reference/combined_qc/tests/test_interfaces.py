import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from combined_qc import hip, spine
from combined_qc.common import Progress, gather_sources, publish_directory
from combined_qc.hip.preprocessing import letterbox, validate_geometry
from combined_qc.hip.runtime import HipONNXPredictor


class InterfaceTests(unittest.TestCase):
    def test_identical_signatures(self):
        for name in ('train', 'load_checkpoints', 'infer'):
            self.assertEqual(inspect.signature(getattr(hip, name)),
                             inspect.signature(getattr(spine, name)))

    def test_all_central_inference_signatures_have_no_manual_side(self):
        from combined_qc import router
        for infer in (hip.infer, spine.infer, router.infer, router.QCPipeline.infer):
            with self.subTest(function=infer.__qualname__):
                self.assertNotIn('laterality', inspect.signature(infer).parameters)

    def test_command_line_inference_has_no_manual_side_flags(self):
        from combined_qc.router.__main__ import main as router_main
        from combined_qc.common import run_cli
        for region in ('hip', 'spine', 'router'):
            with self.subTest(region=region):
                output = io.StringIO()
                with patch('sys.argv', [region, 'infer', '--help']), contextlib.redirect_stdout(output):
                    with self.assertRaises(SystemExit) as exited:
                        if region == 'router':
                            router_main(['infer', '--help'])
                        else:
                            run_cli(region)
                self.assertEqual(exited.exception.code, 0)
                self.assertNotIn('--laterality', output.getvalue())

    def test_ci_environment_does_not_enable_logs(self):
        output = io.StringIO()
        with patch.dict(os.environ, {'CI': 'true'}), contextlib.redirect_stderr(output):
            Progress('hip').update('load', 'model', 1, 10)
        self.assertEqual(output.getvalue(), '')
        with contextlib.redirect_stderr(output):
            Progress('spine', enabled=True).update('load', 'model', 1, 3)
        self.assertIn('[spine] load 1/3', output.getvalue())

    def test_folder_includes_every_image_and_resolves_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'b.png').touch(); (folder/'a.dcm').touch()
            (folder/'unrelated.json').touch(); (folder/'sub').mkdir()
            (folder/'sub'/'c.png').touch()
            files, is_folder = gather_sources(folder)
            self.assertTrue(is_folder)
            self.assertEqual([p.name for p in files], ['a.dcm', 'b.png'])
            self.assertEqual(len(gather_sources(folder, recursive=True)[0]), 3)

    def test_nan_metadata_rejected_before_model_execution(self):
        pixels, geometry = letterbox(np.ones((261, 280), dtype=np.uint8))
        handle = HipONNXPredictor.__new__(HipONNXPredictor)
        handle.config = {'image_size': 320}
        # No session/mean exists: validation must stop before any model logic.
        for field in geometry:
            for value in (float('nan'), float('inf'), float('-inf')):
                invalid = dict(geometry, **{field: value})
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    handle.predict_prepared(pixels, invalid, 'left_hip')
        for field in ('original_height', 'original_width'):
            with self.assertRaises(ValueError):
                validate_geometry(dict(geometry, **{field: 0}))
        with self.assertRaises(ValueError):
            validate_geometry(dict(geometry, pad_top=geometry['pad_top']+1))

    def test_nonfinite_model_output_rejected(self):
        handle = HipONNXPredictor.__new__(HipONNXPredictor)
        class BadSession:
            def run(self, *args): return [np.array([float('nan')])]
        handle.sessions = {'model.onnx': BadSession()}
        with self.assertRaises(ValueError): handle._run('model.onnx', {})

    def test_publish_failure_preserves_previous_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root/'stage'; dest = root/'runtime'
            stage.mkdir(); dest.mkdir(); (dest/'old').write_text('old')
            original = Path.rename
            def fail_stage(path, target):
                if path == stage: raise OSError('injected publish failure')
                return original(path, target)
            with patch.object(Path, 'rename', fail_stage), self.assertRaises(OSError):
                publish_directory(stage, dest)
            self.assertEqual((dest/'old').read_text(), 'old')

    def test_publish_rejects_symlink_before_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); stage=root/'stage'; actual=root/'actual'; dest=root/'runtime'
            stage.mkdir(); actual.mkdir(); dest.symlink_to(actual, target_is_directory=True)
            with self.assertRaises(ValueError): publish_directory(stage,dest)
            self.assertTrue(stage.is_dir()); self.assertTrue(dest.is_symlink())


if __name__ == '__main__': unittest.main()
