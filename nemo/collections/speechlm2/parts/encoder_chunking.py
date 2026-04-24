# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def encode_audio_with_optional_chunking(
    perception: Callable,
    input_signal: Tensor,
    input_signal_length: Tensor,
    *,
    chunk_size_seconds: float | None,
    sampling_rate: int,
) -> list[Tensor]:
    """Encode audio rows, splitting long rows into time chunks before the perception forward."""
    chunk_size_samples = _get_chunk_size_samples(chunk_size_seconds, sampling_rate)
    if chunk_size_samples is None or input_signal_length.numel() == 0:
        audio_embs, audio_emb_lens = perception(input_signal=input_signal, input_signal_length=input_signal_length)
        return _unpad_audio_embeddings(audio_embs, audio_emb_lens)

    min_chunk_size_samples = _get_min_chunk_size_samples(perception)
    chunk_size_samples = max(chunk_size_samples, min_chunk_size_samples)
    input_signal_lengths = input_signal_length.tolist()
    if max(input_signal_lengths) <= chunk_size_samples:
        audio_embs, audio_emb_lens = perception(input_signal=input_signal, input_signal_length=input_signal_length)
        return _unpad_audio_embeddings(audio_embs, audio_emb_lens)

    chunks, chunk_lens, chunks_per_audio = _split_audio_into_chunks(
        input_signal=input_signal,
        input_signal_lengths=input_signal_lengths,
        chunk_size_samples=chunk_size_samples,
        min_chunk_size_samples=min_chunk_size_samples,
    )
    chunked_signal = pad_sequence(chunks, batch_first=True)
    chunked_lens = torch.as_tensor(chunk_lens, device=input_signal_length.device, dtype=input_signal_length.dtype)
    chunked_embs, chunked_emb_lens = perception(input_signal=chunked_signal, input_signal_length=chunked_lens)
    return _recombine_chunked_audio_embeddings(chunked_embs, chunked_emb_lens, chunks_per_audio)


def _get_chunk_size_samples(chunk_size_seconds: float | None, sampling_rate: int) -> int | None:
    if chunk_size_seconds is None:
        return None
    chunk_size_seconds = float(chunk_size_seconds)
    if chunk_size_seconds <= 0.0:
        raise ValueError("encoder_chunk_size_seconds must be positive when set.")
    return max(1, int(round(chunk_size_seconds * sampling_rate)))


def _get_min_chunk_size_samples(perception: Callable) -> int:
    featurizer = getattr(getattr(perception, "preprocessor", None), "featurizer", None)
    hop_length = getattr(featurizer, "hop_length", None)
    if hop_length is None:
        return 1

    hop_length = max(1, int(hop_length))
    get_seq_len = getattr(featurizer, "get_seq_len", None)
    if get_seq_len is None:
        return 2 * hop_length

    samples = hop_length
    for _ in range(16):
        seq_len = get_seq_len(torch.tensor(samples, dtype=torch.float32, device="cpu"))
        if int(seq_len.item()) >= 2:
            return samples
        samples += hop_length
    return max(samples, 2 * hop_length)


def _split_audio_into_chunks(
    input_signal: Tensor,
    input_signal_lengths: list[int],
    chunk_size_samples: int,
    min_chunk_size_samples: int,
) -> tuple[list[Tensor], list[int], list[int]]:
    chunks, chunk_lens, chunks_per_audio = [], [], []
    for audio, audio_len in zip(input_signal, input_signal_lengths):
        if audio_len == 0:
            chunks.append(audio[:0])
            chunk_lens.append(0)
            chunks_per_audio.append(1)
            continue

        spans = []
        for begin in range(0, audio_len, chunk_size_samples):
            end = min(begin + chunk_size_samples, audio_len)
            spans.append((begin, end))
        # A tiny tail chunk can produce a single feature frame, which breaks
        # per-feature normalization in the audio preprocessor. Fold that tail
        # into the previous chunk; this preserves all samples and only lets the
        # final chunk exceed the requested chunk size by a small remainder.
        if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_chunk_size_samples:
            spans[-2] = (spans[-2][0], spans[-1][1])
            spans.pop()

        for begin, end in spans:
            chunks.append(audio[begin:end])
            chunk_lens.append(end - begin)
        chunks_per_audio.append(len(spans))
    return chunks, chunk_lens, chunks_per_audio


def _unpad_audio_embeddings(audio_embs: Tensor, audio_emb_lens: Tensor) -> list[Tensor]:
    return [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]


def _recombine_chunked_audio_embeddings(
    chunked_embs: Tensor,
    chunked_emb_lens: Tensor,
    chunks_per_audio: list[int],
) -> list[Tensor]:
    audio_embs = []
    chunk_idx = 0
    for num_chunks in chunks_per_audio:
        parts = [chunked_embs[i, : chunked_emb_lens[i]] for i in range(chunk_idx, chunk_idx + num_chunks)]
        audio_embs.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
        chunk_idx += num_chunks
    return audio_embs
