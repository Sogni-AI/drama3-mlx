"""High-level Drama 3 text-to-speech on MLX."""

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from .chunking import CHUNK_MAX_NEW_TOKENS, ChunkHistory, split_text
from .codec import FishCodec
from .generate import generate_frames
from .loading import load_model
from .prompt import PromptTooLongError, build_prompt, check_tokenizer
from .sampling import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P

# Frames per request with single_chunk and no limit given (fish_omni/generate.py).
SINGLE_CHUNK_MAX_NEW_TOKENS = 2048


@dataclass
class Reference:
    """A voice to clone: its exact transcript and codec codes [num_codebooks, frames]."""

    text: str
    codes: np.ndarray


@dataclass
class Segment:
    text: str
    codes: np.ndarray  # [num_codebooks, frames]
    audio: np.ndarray  # float32 mono at Drama3.sample_rate
    finish_reason: str
    prompt_rows: int
    seconds: float


class Drama3:
    def __init__(self, model, tokenizer: Tokenizer, codec: FishCodec):
        self.model, self.tokenizer, self.codec = model, tokenizer, codec
        check_tokenizer(tokenizer, model.config)
        self.sample_rate = codec.sample_rate

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        codec_path: str | Path | None = None,
        dtype: mx.Dtype = mx.bfloat16,
        experts: str = "q8",
    ) -> "Drama3":
        """Fish's release directory or a drama3_mlx.convert output.

        experts sets how the release's experts are held: fp8 as published (exact,
        three times slower), q8 (default: 8-bit, same memory, as close to the fp32
        model as Fish's own bf16 runtime), q4 (25 GB) or bf16 (70 GB). A converted
        checkpoint keeps its own format. The codec is codec.pth (release) or
        codec.safetensors (converted) next to the weights unless codec_path is given.
        """
        path = Path(path)
        if codec_path is None:
            codec_path = next(
                p for p in (path / "codec.safetensors", path / "codec.pth") if p.exists()
            )
        model = load_model(path, dtype=dtype, experts=experts)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        return cls(model, tokenizer, FishCodec(codec_path))

    def reference(self, audio: np.ndarray, sample_rate: int, text: str) -> Reference:
        """A Reference from mono audio and its exact transcript."""
        return Reference(text=text, codes=self.codec.encode(audio, sample_rate))

    def generate(
        self,
        text: str,
        *,
        references: Sequence[Reference] = (),
        previous_takes: Sequence[tuple[str, np.ndarray]] = (),
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = 0,
        single_chunk: bool = False,
        seed: int | None = None,
        accept: Callable[[Segment], bool] | None = None,
        attempts: int = 1,
    ) -> Iterator[Segment]:
        """Speech for text, one Segment per chunk as soon as it is done.

        Long text is split like Fish's hosted server; the first chunk and the
        latest one go back into the next prompt so the voice stays consistent.
        previous_takes are earlier (text, codes) takes, placed before every chunk
        as completed turns: Drama 3's editing-as-continuation, e.g. re-reading a
        line with one word changed. max_new_tokens <= 0 means 600 frames per
        chunk (2048 with single_chunk). With accept, a chunk is sampled again, up
        to attempts times in all, until accept(segment) holds; the last take is
        kept if none does.
        """
        if seed is not None:
            mx.random.seed(seed)
        config = self.model.config
        prompt_texts = [r.text for r in references]
        prompt_codes = [r.codes for r in references]
        chunks = [text] if single_chunk else split_text(text) or [text]
        # Reject bad text in any chunk before generating any audio.
        previous_takes = list(previous_takes)
        for chunk in chunks:
            build_prompt(
                chunk, self.tokenizer, config, prompt_texts, prompt_codes, previous_takes
            )
        budget = max_new_tokens
        if budget <= 0:
            budget = SINGLE_CHUNK_MAX_NEW_TOKENS if single_chunk else CHUNK_MAX_NEW_TOKENS

        history = ChunkHistory()
        for chunk in chunks:
            prompt = self._chunk_prompt(
                chunk, prompt_texts, prompt_codes, previous_takes, history
            )
            started = time.perf_counter()
            segment = None
            for _ in range(max(1, attempts)):
                frames, finish_reason = generate_frames(
                    self.model,
                    prompt,
                    max_new_tokens=budget,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                if not len(frames):
                    continue
                codes = frames[:, 1:].T.copy()
                segment = Segment(
                    text=chunk,
                    codes=codes,
                    audio=self.codec.decode(codes),
                    finish_reason=finish_reason,
                    prompt_rows=len(prompt),
                    seconds=time.perf_counter() - started,
                )
                if accept is None or accept(segment):
                    break
            if segment is None:
                continue
            history.push(chunk, segment.codes)
            yield segment

    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """The whole of generate() as one float32 waveform."""
        segments = [segment.audio for segment in self.generate(text, **kwargs)]
        return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)

    def _chunk_prompt(
        self, text, prompt_texts, prompt_codes, previous_takes, history
    ) -> np.ndarray:
        """The prompt for one chunk with as much chunk history as fits."""
        error = None
        for turns in history.candidates():
            try:
                return build_prompt(
                    text,
                    self.tokenizer,
                    self.model.config,
                    prompt_texts,
                    prompt_codes,
                    previous_takes + turns,
                )
            except PromptTooLongError as too_long:
                error = too_long
        raise error
