"""Fish's 44.1 kHz, 10-codebook DAC codec on MLX (mlx-audio's fish_s1_dac).

Drama 3 ships the S2-Pro codec (byte-identical codec.pth). The checkpoint is read
straight from PyTorch's zip format with a restricted unpickler that knows only the
five globals the file uses, so neither torch nor arbitrary pickle code runs.
"""

import pickle
import zipfile
from collections import OrderedDict
from math import gcd
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.codec.models.fish_s1_dac.fish_s1_dac import build_ae
from scipy.signal import resample_poly

# numpy dtype per torch storage; bf16 is read as raw uint16 and reinterpreted.
STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "BFloat16Storage": np.uint16,
    "BoolStorage": np.bool_,
}


def _rebuild_tensor(storage, offset, size, stride, *_):
    kind, raw = storage
    itemsize = np.dtype(STORAGE_DTYPES[kind]).itemsize
    flat = np.frombuffer(raw, dtype=STORAGE_DTYPES[kind])
    array = np.lib.stride_tricks.as_strided(
        flat[offset:], shape=size, strides=[s * itemsize for s in stride]
    )
    array = mx.array(np.ascontiguousarray(array))
    return array.view(mx.bfloat16) if kind == "BFloat16Storage" else array


class _TorchZipUnpickler(pickle.Unpickler):
    def __init__(self, archive: zipfile.ZipFile, prefix: str, data: bytes):
        import io

        super().__init__(io.BytesIO(data))
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in STORAGE_DTYPES:
            return name
        raise pickle.UnpicklingError(f"codec.pth uses an unexpected global {module}.{name}")

    def persistent_load(self, pid):
        _, kind, key, _location, _numel = pid
        return kind, self.archive.read(f"{self.prefix}/data/{key}")


def read_torch_state_dict(path: Path) -> dict[str, mx.array]:
    with zipfile.ZipFile(path) as archive:
        (pickle_name,) = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        prefix = pickle_name[: -len("/data.pkl")]
        return dict(_TorchZipUnpickler(archive, prefix, archive.read(pickle_name)).load())


class FishCodec:
    def __init__(self, path: str | Path):
        path = Path(path)
        weights = (
            read_torch_state_dict(path) if path.suffix == ".pth" else mx.load(str(path))
        )
        self.dac = build_ae()
        # The causal weight-norm convs rebuild their inner conv weights from
        # weight_g / weight_v on every call, so those have no checkpoint tensors.
        self.dac.load_weights(list(self.dac.sanitize(weights).items()), strict=False)
        mx.eval(self.dac.parameters())
        self.sample_rate = self.dac.sample_rate
        self.frame_length = self.dac.frame_length

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Mono float audio at sample_rate -> codes [num_codebooks, frames]."""
        if sample_rate != self.sample_rate:
            factor = gcd(sample_rate, self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // factor, sample_rate // factor)
        audio = mx.array(audio.astype(np.float32))[None, None, :]
        indices, lengths = self.dac.encode(audio, mx.array([audio.shape[-1]]))
        return np.array(indices[0, :, : int(lengths[0])])

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Codes [num_codebooks, frames] -> mono float32 audio."""
        audio, lengths = self.dac.decode(
            mx.array(codes.astype(np.int32))[None], mx.array([codes.shape[1]])
        )
        return np.array(audio[0, 0, : int(lengths[0])].astype(mx.float32))
