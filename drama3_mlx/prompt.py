"""Drama 3 prompt rows, as built by fish_omni/generate.py (fish-speech v3.0.0-beta).

Text rows are [token, 0, ..., 0]; audio rows are [c0 + semantic_start, c0..cN-1].
Every literal is tokenized on its own, because tokenizing the joined string changes
ids at the seams.
"""

import re
from collections.abc import Sequence

import numpy as np
from tokenizers import Tokenizer

from .config import Drama3Config

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
# Only the prompt builder writes these; other special tokens in caller text are content.
STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
# The tokenizer has one token per speaker, <|speaker:0|> to <|speaker:15|>.
MAX_SPEAKER_ID = 15
# The Unicode White_Space property; str.strip() without arguments also strips
# U+001C..U+001F, which would change the empty check and the speaker-tag rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)
MAX_PROMPT_ROWS = 8192


class InvalidInputError(ValueError):
    """The request cannot be served as given: bad text, references or length."""


class PromptTooLongError(InvalidInputError):
    """The prompt has more than MAX_PROMPT_ROWS rows."""


def validate_text(text: str, location: str) -> None:
    """Non-empty, no structural tokens, and only canonical speaker tags 0..15."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise InvalidInputError(f"{location}: text must not be empty")
    for token in STRUCTURAL_TOKENS:
        if token in text:
            raise InvalidInputError(f"{location}: {token} is written by the prompt builder")
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG_PATTERN.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise InvalidInputError(
                f"{location}: speaker tags must be <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>; got {text!r}"
            )


def with_speaker_tag(text: str) -> str:
    """Prefix <|speaker:0|> unless a tag already leads the text after white space."""
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text


def check_tokenizer(tokenizer: Tokenizer, config: Drama3Config) -> None:
    codebook_size = config.audio_decoder.vocab_size
    for token, expected in (
        ("<|semantic:0|>", config.semantic_start_token_id),
        (f"<|semantic:{codebook_size - 1}|>", config.semantic_end_token_id),
        ("<|im_end|>", config.eos_token_id),
    ):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"tokenizer/config mismatch for {token}")


def build_prompt(
    text: str,
    tokenizer: Tokenizer,
    config: Drama3Config,
    prompt_texts: Sequence[str] = (),
    prompt_codes: Sequence[np.ndarray] = (),
    history: Sequence[tuple[str, np.ndarray]] = (),
) -> np.ndarray:
    """int64 rows [rows, 1 + num_codebooks] ending in the open assistant voice turn.

    prompt_texts / prompt_codes are reference transcripts and their codec codes
    [num_codebooks, frames]; history holds earlier (text, codes) chunks, written as
    completed user and assistant voice turns before the current one.
    """
    num_codebooks = config.audio_decoder.num_codebooks
    codebook_size = config.audio_decoder.vocab_size
    if len(prompt_texts) != len(prompt_codes):
        raise InvalidInputError("every reference needs both a transcript and codes")
    validate_text(text, "text")

    def check_codes(codes: np.ndarray, location: str) -> None:
        if (
            not np.issubdtype(codes.dtype, np.integer)
            or codes.ndim != 2
            or codes.shape[0] != num_codebooks
            or codes.shape[1] == 0
        ):
            raise InvalidInputError(
                f"{location}: codes must be integer [{num_codebooks}, frames >= 1]; "
                f"got {codes.dtype} {codes.shape}"
            )
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise InvalidInputError(f"{location}: codes must be in [0, {codebook_size})")

    segments: list[str | np.ndarray] = [
        "<|im_start|>system\n",
        REFERENCE_INSTRUCTION if prompt_codes else TTS_INSTRUCTION,
    ]
    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_codes), 1):
        location = f"reference {index}"
        validate_text(transcript, location)
        check_codes(codes, location)
        # The tag must follow "Text: " directly; leading white space moves behind it.
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_text(turn_text, location)
        check_codes(codes, location)
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            MESSAGE_END,
            GENERATION_START,
            codes,
            MESSAGE_END,
        ]
    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        MESSAGE_END,
        GENERATION_START,
    ]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = np.array(tokenizer.encode(segment).ids, dtype=np.int64)
            text_rows = np.zeros((len(ids), 1 + num_codebooks), dtype=np.int64)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.astype(np.int64)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(np.concatenate([main, codes]).T)
    rows = np.concatenate(rows)
    if len(rows) > MAX_PROMPT_ROWS:
        raise PromptTooLongError(f"prompt has {len(rows)} rows; at most {MAX_PROMPT_ROWS}")
    return rows
