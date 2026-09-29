"""Native uint8 PNG/DICOM input and the existing spine letterbox preparation."""
from pathlib import Path
import numpy as np
from .engine.utils import letterbox320, read_grayscale


def load_native(path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".png":
        return read_grayscale(path)
    if path.suffix.lower() not in {".dcm", ".dicom"}:
        raise ValueError("Expected native grayscale PNG or DICOM")
    import pydicom
    ds = pydicom.dcmread(path)
    expected = {"PhotometricInterpretation": "MONOCHROME2", "SamplesPerPixel": 1,
                "BitsAllocated": 8, "BitsStored": 8, "HighBit": 7, "PixelRepresentation": 0}
    if any(getattr(ds, key, None) != value for key, value in expected.items()):
        raise ValueError("DICOM must be single-channel uint8 MONOCHROME2")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("Multiframe DICOM is unsupported")
    pixels = ds.pixel_array
    if (pixels.shape != (int(ds.Rows), int(ds.Columns)) or pixels.dtype != np.uint8
            or min(pixels.shape) < 1):
        raise ValueError("Invalid native DICOM pixels")
    return pixels
