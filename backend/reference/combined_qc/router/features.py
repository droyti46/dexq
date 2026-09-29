"""Fixed image-only preprocessing and an ImageNet ONNX feature extractor."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
DATA = PROJECT / 'combined_qc/data/dxa_v1'
RESNET_SHA = '9506741d36b44aae108e6015f6da92d22df89570ed692b395024129a133cdc0a'
RESNET_STATE_SHA = '6125408e356323c4193a8ab1f56bb155aad05af3fc2b542e3c5a8587e9a17505'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def prepare_native(array):
    """Remove only fully black outer rows/columns, then resize to 320 square.

    No width, aspect, acquisition attributes, side or filename enters features.
    Mirror invariance is provided by mean(normal, horizontal mirror) embeddings.
    """
    array=np.asarray(array)
    if array.dtype!=np.uint8 or array.ndim!=2 or min(array.shape)<12:
        raise ValueError('Expected a native uint8 grayscale image at least 12 pixels per side')
    if int(array.max())-int(array.min())<4 or float(array.std())<.8:
        raise ValueError('Image has no usable contrast; classification unavailable')
    y=np.flatnonzero(np.any(array>1,axis=1));x=np.flatnonzero(np.any(array>1,axis=0))
    if not len(y) or not len(x):raise ValueError('Blank image; classification unavailable')
    crop=array[y[0]:y[-1]+1,x[0]:x[-1]+1]
    return np.asarray(Image.fromarray(crop).resize((320,320),Image.Resampling.BILINEAR)).copy()


class FeatureExtractor:
    def __init__(self, path, expected_sha=None):
        import onnxruntime as ort
        path=Path(path).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        if expected_sha is not None and sha256(path)!=expected_sha:raise ValueError('ResNet checksum mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.log_severity_level=3
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        if [(n.name,n.type) for n in self.session.get_inputs()]!=[('gray_u8','tensor(uint8)')]:
            raise ValueError('Unexpected ImageNet ONNX input')
        if [n.name for n in self.session.get_outputs()]!=['global_512','spatial_2048']:
            raise ValueError('Unexpected ImageNet ONNX outputs')

    def extract(self, prepared, batch_size=16):
        frames=np.asarray(prepared)
        if frames.dtype!=np.uint8 or frames.ndim!=3 or frames.shape[1:]!=(320,320) or len(frames)==0:
            raise ValueError('Expected uint8 [N,320,320]')
        output=[]
        for start in range(0,len(frames),batch_size):
            original=frames[start:start+batch_size]
            augmented=np.concatenate([original,original[:,:,::-1]],axis=0)
            global_features,spatial_features=self.session.run(None,{'gray_u8':np.ascontiguousarray(augmented)})
            n=len(original)
            if global_features.shape!=(2*n,512) or spatial_features.shape!=(2*n,2048):
                raise ValueError('Unexpected feature dimensions')
            joined=np.concatenate([global_features,spatial_features],axis=1)
            averaged=(joined[:n]+joined[n:])/2
            if not np.isfinite(averaged).all():raise ValueError('Nonfinite neural features')
            output.append(averaged)
        return np.concatenate(output).astype(np.float32)


def records(data=DATA):
    data=Path(data).resolve()
    rows=[json.loads(line) for line in (data/'manifest.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=255 or len({r['image_id'] for r in rows})!=255:
        raise ValueError('Expected all 255 unique region-labelled images')
    if any(r['anatomical_region'] not in {'left_hip','right_hip','lumbar_spine'} for r in rows):
        raise ValueError('Unexpected region label')
    return rows


def recipe_fingerprint():
    from importlib.metadata import version
    from .bootstrap import export_recipe_sha256
    recipe={'code':{name:sha256(ROOT/name) for name in ['features.py','training.py','bootstrap.py','conversion.py']},
            'export_recipe_sha256':export_recipe_sha256(),
            'libraries':{name:version(name) for name in ['numpy','Pillow','scikit-learn','onnxruntime']}}
    return hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()


def actual_source(row,data=DATA,source_data=None):
    data=Path(data).resolve()
    native=(data/row['native_path']).resolve()
    if not native.is_relative_to(data):raise ValueError('Native path escapes data')
    if native.is_file():
        if sha256(native)!=row['native_sha256']:raise ValueError('Native PNG checksum mismatch')
        return native
    root=Path(source_data).resolve() if source_data else PROJECT/'source_data'
    source=(root/row['canonical_source_path']).resolve()
    if not source.is_relative_to(root):raise ValueError('Source path escapes source_data')
    if not source.is_file():raise FileNotFoundError(f'Native source unavailable: {source}')
    return source


def load_record(row,data=DATA,source_data=None):
    from .preprocessing import load_native
    pixels=load_native(actual_source(row,data,source_data))
    digest=hashlib.sha256(str(pixels.shape).encode()+np.ascontiguousarray(pixels).tobytes()).hexdigest()
    if digest!=row['pixel_hash']:raise ValueError(f"Native pixels differ from manifest: {row['image_id']}")
    return pixels
