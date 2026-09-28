"""Pinned public ImageNet source weights and reproducible ONNX export.

Training uses a graph under source/. Publishing runtime models is exclusively
the conversion module's job, so train(convert=False) never updates inference.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from contextlib import contextmanager
import shutil
import tempfile
from urllib.request import Request, urlopen
import warnings

from .features import PROJECT, RESNET_STATE_SHA, sha256
from combined_qc.common import Progress, quiet_output, write_json

URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
OFFICIAL_SHA = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
SOURCE_NAME = 'resnet18_imagenet.pth'
BACKBONE_NAME = 'features_backbone.onnx'
OPSET = 17


def export_recipe_sha256():
    paths = [Path(__file__), PROJECT / 'combined_qc/spine/onnx_models.py',
             PROJECT / 'combined_qc/spine/engine/features.py']
    return hashlib.sha256(json.dumps([sha256(path) for path in paths]).encode()).hexdigest()


@contextmanager
def inference_threads():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def prepare_source(checkpoints_dir, *, verbose=False, ci=False):
    """Save the actual public PyTorch checkpoint inside this module's source/."""
    base = Path(checkpoints_dir).resolve()
    source = base / 'source'
    source.mkdir(parents=True, exist_ok=True)
    target = source / SOURCE_NAME
    progress = Progress('router', verbose or ci)
    if target.is_file():
        if sha256(target) != OFFICIAL_SHA:
            raise ValueError('Router public ImageNet source checksum mismatch')
        progress.update('source', 'Verified pinned public ImageNet ResNet18 checkpoint')
    else:
        candidates = [Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth',
                      PROJECT / 'combined_qc/experiments/region_classifier/checkpoints/source/resnet18_imagenet.pth']
        cached = next((p for p in candidates if p.is_file() and sha256(p) == OFFICIAL_SHA), None)
        with tempfile.TemporaryDirectory(dir=source, prefix='.download-') as temporary:
            downloaded = Path(temporary) / SOURCE_NAME
            if cached:
                progress.update('source', 'Copying verified cached public ImageNet weights')
                shutil.copy2(cached, downloaded)
            else:
                progress.update('source', 'Downloading pinned public ImageNet ResNet18')
                with urlopen(Request(URL, headers={'User-Agent': 'dxa-router/1.0'}), timeout=120) as response, downloaded.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
            if sha256(downloaded) != OFFICIAL_SHA:
                raise ValueError('Public ImageNet download checksum mismatch')
            downloaded.replace(target)
    write_json(source / 'resnet18.json', {
        'public_url': URL, 'official_sha256': OFFICIAL_SHA,
        'checkpoint': SOURCE_NAME, 'checkpoint_sha256': sha256(target),
        'feature_state_sha256': RESNET_STATE_SHA, 'weights_only_load': True,
        'pretrained_dataset': 'ImageNet', 'fine_tuned_on_dxa': False,
    })
    return target


def load_source_graph(checkpoints_dir):
    """Strictly load public weights; never download or load a serialized object."""
    import torch
    from torch import nn
    from torchvision.models import resnet18
    from combined_qc.spine.engine.features import state_digest
    from combined_qc.spine.onnx_models import ResNet18Features

    source_file = Path(checkpoints_dir).resolve() / 'source' / SOURCE_NAME
    if not source_file.is_file():
        raise FileNotFoundError(f'Missing source checkpoint: {source_file}; run train first')
    if sha256(source_file) != OFFICIAL_SHA:
        raise ValueError('Public ImageNet source checksum mismatch')
    full = resnet18(weights=None)
    full.load_state_dict(torch.load(source_file, map_location='cpu', weights_only=True), strict=True)
    backbone = nn.Sequential(*list(full.children())[:-2]).eval()
    if state_digest(backbone) != RESNET_STATE_SHA:
        raise ValueError('Frozen ImageNet feature state differs from the pin')
    graph = ResNet18Features(backbone).cpu().eval().requires_grad_(False)
    provenance = {'source_checkpoint': SOURCE_NAME, 'source_sha256': OFFICIAL_SHA,
                  'feature_state_sha256': RESNET_STATE_SHA, 'opset': OPSET,
                  'export_recipe_sha256': export_recipe_sha256(),
                  'weights_only_load': True, 'strict_load': True,
                  'fine_tuned_on_dxa': False}
    return graph, provenance


def export_backbone(checkpoints_dir, target, *, verbose=False, ci=False):
    """Regenerate a self-contained dynamic-batch graph from the source weights."""
    import torch
    import onnx

    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    graph, provenance = load_source_graph(checkpoints_dir)
    Progress('router', verbose or ci).update('convert', 'Exporting frozen ResNet18 from public source weights')
    with inference_threads(), torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        torch.onnx.export(graph, (torch.zeros((1, 320, 320), dtype=torch.uint8),), target,
            input_names=['gray_u8'], output_names=['global_512', 'spatial_2048'],
            opset_version=OPSET, dynamo=False, external_data=False,
            dynamic_axes={key: {0: 'batch'} for key in ['gray_u8', 'global_512', 'spatial_2048']})
    onnx.checker.check_model(str(target))
    return {'path': target.name, 'sha256': sha256(target), **provenance}


def prepare_backbone(checkpoints_dir, *, verbose=False, ci=False):
    """Prepare source weights plus a private graph for extracting fit features."""
    with quiet_output(verbose or ci):
        base = Path(checkpoints_dir).resolve()
        prepare_source(base, verbose=verbose, ci=ci)
        target = base / 'source' / BACKBONE_NAME
        metadata_path = base / 'source' / 'features_backbone.json'
        if target.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if (metadata.get('sha256') == sha256(target)
                    and metadata.get('source_sha256') == OFFICIAL_SHA
                    and metadata.get('feature_state_sha256') == RESNET_STATE_SHA
                    and metadata.get('export_recipe_sha256') == export_recipe_sha256()):
                Progress('router', verbose or ci).update('features', 'Using verified private training ONNX graph')
                return target
        with tempfile.TemporaryDirectory(dir=base / 'source', prefix='.backbone-') as temporary:
            staged = Path(temporary) / BACKBONE_NAME
            metadata = export_backbone(base, staged, verbose=verbose, ci=ci)
            staged.replace(target)
        write_json(metadata_path, metadata)
        return target
