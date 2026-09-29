"""Image-only input reader; acquisition metadata is never a model feature."""
from pathlib import Path
import numpy as np
from PIL import Image

SUFFIXES={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.dcm','.dicom'}


def load_native(path):
    path=Path(path).resolve()
    if path.suffix.lower() in {'.dcm','.dicom'}:
        from combined_qc.spine.preprocessing import load_native as read_dicom
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Invalid value for VR UI.*',module='pydicom.valuerep')
            return read_dicom(path)
    with Image.open(path) as image:
        if getattr(image,'n_frames',1)!=1:raise ValueError('Expected a single image frame')
        if image.mode not in {'L','RGB','RGBA'}:raise ValueError('Expected an 8-bit grayscale or RGB raster image')
        pixels=np.asarray(image.convert('L')).copy()
    if pixels.dtype!=np.uint8 or pixels.ndim!=2:raise ValueError('Expected uint8 grayscale pixels')
    return pixels


def gather_sources(source,recursive=False):
    source=Path(source).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in SUFFIXES:raise ValueError('Unsupported image extension')
        return [source],False
    if not source.is_dir():raise FileNotFoundError(source)
    iterator=source.rglob('*') if recursive else source.iterdir()
    files=[p.resolve() for p in sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUFFIXES)]
    if not files:raise ValueError('No supported image files in the directory')
    return files,True
