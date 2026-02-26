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
"""
Offline evaluation script for StreamingSALM with **per-utterance** context biasing.

Supports multiple context strategies:
  - ``none``: No context (equivalent to standard eval)
  - ``oracle``: Full reference transcript as context (upper bound)
  - ``prefix_N``: First N words of reference (e.g., ``prefix_3``)
  - ``random_N``: N random words from reference (e.g., ``random_1``, ``random_3``)
  - ``adversarial``: Random words sampled from *other* utterances

Usage::

    python streaming_salm_generate_with_context.py \
        pretrained_name=models/baseline_hf \
        inputs=/data/librispeech/lhotse/librispeech_cuts_lower_test-clean.jsonl.gz \
        context_strategy=oracle \
        latency=5 \
        batch_size=64 \
        output_manifest=results/eval/ctx_oracle_test-clean.jsonl
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2.models import StreamingSALM
from nemo.core.config import hydra_runner
from nemo.utils import logging


class ToAudio(torch.utils.data.Dataset):
    """Minimal dataset that loads audio from a CutSet."""

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


@dataclass
class ContextEvalConfig:
    pretrained_name: str = ""
    inputs: str = ""
    batch_size: int = 64
    latency: int = 5
    context_strategy: str = "none"  # none, oracle, prefix_N, random_N, adversarial
    output_manifest: Optional[str] = "streaming_salm_context_eval.jsonl"
    verbose: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_normalizer: Optional[str] = "english"
    seed: int = 42


def _generate_context(
    strategy: str,
    ref_text: str,
    all_refs: list[str],
    rng: random.Random,
) -> str | None:
    """Generate per-utterance context based on strategy."""
    if strategy == "none":
        return None

    words = ref_text.split()

    if strategy == "oracle":
        return ref_text

    if strategy.startswith("prefix_"):
        n = int(strategy.split("_")[1])
        return " ".join(words[:n]) if words else None

    if strategy.startswith("random_"):
        n = int(strategy.split("_")[1])
        if len(words) <= n:
            return ref_text
        selected = rng.sample(words, min(n, len(words)))
        return " ".join(selected)

    if strategy == "adversarial":
        # Sample 3 random words from other utterances
        other_words = []
        for _ in range(10):
            other_ref = rng.choice(all_refs)
            other_words.extend(other_ref.split())
        if len(other_words) >= 3:
            selected = rng.sample(other_words, 3)
            return " ".join(selected)
        return " ".join(other_words) if other_words else None

    raise ValueError(f"Unknown context strategy: {strategy}")


@hydra_runner(config_name="ContextEvalConfig", schema=ContextEvalConfig)
def main(cfg: ContextEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    model = StreamingSALM.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)

    _normalizer_key = cfg.use_normalizer.lower() if isinstance(cfg.use_normalizer, str) else cfg.use_normalizer
    normalizer = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}.get(
        _normalizer_key, lambda x: x
    )

    # Load all cuts upfront (needed for adversarial context);
    # resample to model's expected sample rate if needed.
    cuts = guess_parse_cutset(cfg.inputs).sort_by_duration()
    cuts_list = list(cuts)
    if cuts_list and cuts_list[0].sampling_rate != model.sample_rate:
        logging.info(
            f"Resampling cuts from {cuts_list[0].sampling_rate} to {model.sample_rate} Hz"
        )
        cuts_list = [c.resample(model.sample_rate) for c in cuts_list]
        cuts = CutSet.from_cuts(cuts_list)

    # Build lookup: cut_id -> normalized reference text
    ref_by_id = {cut.id: normalizer(cut.supervisions[0].text) for cut in cuts_list}
    all_refs = list(ref_by_id.values())

    rng = random.Random(cfg.seed)

    # Pre-generate contexts for all cuts (deterministic order by cut id)
    context_by_id: dict[str, str | None] = {}
    for cut in cuts_list:
        ref_text = ref_by_id[cut.id]
        context_by_id[cut.id] = _generate_context(cfg.context_strategy, ref_text, all_refs, rng)

    # Batched dataloader
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size),
        num_workers=1,
        batch_size=None,
    )

    refs = []
    hyps = []
    contexts_used = []
    cut_ids = []
    input_durations = []
    infer_durations = []

    for batch_idx, batch in enumerate(dloader):
        batch_cuts = batch["cuts"]
        batch_refs = [ref_by_id[c.id] for c in batch_cuts]
        batch_contexts = [context_by_id[c.id] for c in batch_cuts]

        ts = perf_counter()
        batch_hyps_raw = model.generate(
            audio=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            latency=cfg.latency,
            context=batch_contexts,
        )
        batch_infer_duration = perf_counter() - ts

        batch_duration = sum(c.duration for c in batch_cuts)
        batch_hyps = [normalizer(h.strip()) for h in batch_hyps_raw]

        if cfg.verbose:
            batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info(
                f"Batch {batch_idx}: "
                f"WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] "
                f"RTFx={batch_rtfx:.1f}"
            )

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        contexts_used.extend(batch_contexts)
        cut_ids.extend(c.id for c in batch_cuts)
        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)

    # Final metrics
    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = sum(input_durations) / sum(infer_durations) if sum(infer_durations) > 0 else float("inf")

    logging.info(f"\n=== Results (strategy={cfg.context_strategy}, K={cfg.latency}) ===")
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    logging.info(f"RTFx: {rtfx:.1f}")
    logging.info(f"Utterances: {len(refs)}")

    if cfg.output_manifest is not None:
        # Build a lookup for results by cut id (since batched order may differ from input order)
        result_by_id = {}
        for cid, ref, hyp, ctx in zip(cut_ids, refs, hyps, contexts_used):
            result_by_id[cid] = {"text": ref, "pred_text": hyp, "context": ctx}
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for cut in cuts_list:
                res = result_by_id[cut.id]
                writer.write({
                    "id": cut.id,
                    "duration": cut.duration,
                    "text": res["text"],
                    "pred_text": res["pred_text"],
                    "context_strategy": cfg.context_strategy,
                    "context": res["context"],
                })
        logging.info(f"Wrote {len(hyps)} entries to {cfg.output_manifest}")


if __name__ == "__main__":
    main()
