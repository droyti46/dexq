"""Pinned public ResNet18 source and shared hip feature graph preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import warnings

from combined_qc.common import Progress, quiet_output, write_json
from combined_qc.router.bootstrap import OFFICIAL_SHA, SOURCE_NAME, OPSET, inference_threads
from combined_qc.router.features import RESNET_STATE_SHA
from .preprocessing import sha256

ROOT = Path(__file__).resolve().parent
BACKBONE_NAME = 'quality_backbone.onnx'
OUTPUT_NAMES = ['global_512', 'spatial_2048', 'stage3_6400']


def export_recipe_sha256():
    paths = [ROOT / 'bootstrap.py', ROOT / 'onnx_models.py',
             ROOT.parent / 'router/bootstrap.py', ROOT.parent / 'spine/onnx_models.py',
             ROOT.parent / 'spine/engine/features.py']
    return hashlib.sha256(json.dumps({str(path.relative_to(ROOT.parent)): sha256(path)
                                     for path in paths}, sort_keys=True).encode()).hexdigest()


def prepare_source(checkpoints_dir, *, verbose=False, ci=False):
    """Copy/download verified official ImageNet weights into hip/source only."""
    from combined_qc.router.bootstrap import prepare_source as prepare_public
    progress = Progress('hip', verbose or ci)
    progress.update('source', 'Checking pinned public ImageNet ResNet18 weights')
    with quiet_output(False):
        target = prepare_public(checkpoints_dir, verbose=False, ci=False)
    progress.update('source', 'Verified public weights; encoder remains frozen')
    return target


def load_source_graph(checkpoints_dir):
    from combined_qc.router.bootstrap import load_source_graph as load_public
    from .onnx_models import SharedQualityGraph
    public, provenance = load_public(checkpoints_dir)
    graph = SharedQualityGraph(public).cpu().eval().requires_grad_(False)
    provenance = dict(provenance, export_recipe_sha256=export_recipe_sha256(),
                      output_features={'global_512': 512, 'spatial_2048': 2048, 'stage3_6400': 6400})
    return graph, provenance


def export_backbone(checkpoints_dir, target, *, verbose=False, ci=False):
    import onnx
    import torch
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    graph, provenance = load_source_graph(checkpoints_dir)
    Progress('hip', verbose or ci).update('convert', 'Exporting one frozen shared ResNet18 feature encoder')
    with inference_threads(), torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        torch.onnx.export(graph, (torch.zeros((1, 320, 320), dtype=torch.uint8),), target,
            input_names=['gray_u8'], output_names=OUTPUT_NAMES, opset_version=OPSET,
            dynamo=False, external_data=False,
            dynamic_axes={name: {0: 'batch'} for name in ['gray_u8', *OUTPUT_NAMES]})
    onnx.checker.check_model(str(target))
    return {'path': target.name, 'sha256': sha256(target), **provenance}


def prepare_backbone(checkpoints_dir, *, verbose=False, ci=False):
    """Private training graph; live runtime is published only by convert_all."""
    with quiet_output(verbose or ci):
        base = Path(checkpoints_dir).resolve()
        prepare_source(base, verbose=verbose, ci=ci)
        target = base / 'source' / BACKBONE_NAME
        metadata_path = target.with_suffix('.json')
        if target.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if (metadata.get('sha256') == sha256(target)
                    and metadata.get('source_sha256') == OFFICIAL_SHA
                    and metadata.get('feature_state_sha256') == RESNET_STATE_SHA
                    and metadata.get('export_recipe_sha256') == export_recipe_sha256()):
                Progress('hip', verbose or ci).update('features', 'Reusing verified private shared feature graph')
                return target
        with tempfile.TemporaryDirectory(dir=base / 'source', prefix='.backbone-') as temporary:
            staged = Path(temporary) / BACKBONE_NAME
            metadata = export_backbone(base, staged, verbose=verbose, ci=ci)
            staged.replace(target)
        write_json(metadata_path, metadata)
        return target
