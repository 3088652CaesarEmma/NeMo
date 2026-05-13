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


from queue import Queue
from typing import Any

import torch
from torch import Tensor

from nemo.collections.asr.inference.utils.cache import Cache, QuantizedCache, RawCache
from nemo.collections.asr.inference.utils.turboquant.mse import TurboQuantMSE


class CacheAwareContext:
    """
    Stores the cache state for the Cache-Aware models.
    """

    def __init__(
        self,
        cache_last_channel: Tensor | None = None,
        cache_last_time: Tensor | None = None,
        cache_last_channel_len: Tensor | None = None,
    ):
        """
        Args:
            cache_last_channel (Tensor | None): Last channel of the cache.
            cache_last_time (Tensor | None): Last time of the cache.
            cache_last_channel_len (Tensor | None): Last channel length of the cache.
        """
        self.cache_last_channel = cache_last_channel
        self.cache_last_time = cache_last_time
        self.cache_last_channel_len = cache_last_channel_len


class CacheAwareContextManager:
    """
    Manager class to manipulate the cached states for the Cache-Aware models.
    """

    def __init__(
        self,
        cache_aware_model: Any,
        num_slots: int,
        use_cache: bool = True,
        quantize_cache: bool = False,
        quant_bits: int = 4,
    ):
        """
        Initialize the CacheAwareContextManager.
        Args:
            cache_aware_model (Any): Cache-Aware model object. It should have the get_initial_cache_state method.
            num_slots (int): Number of slots to use for the cache. It should be greater than or equal to the batch size.
            use_cache (bool): Whether to use the cache. Default is True. If False, the cache is disabled.
            quantize_cache (bool): If True, store `cache_last_channel` in a `QuantizedCache` (TurboQuantMSE).
                `cache_last_time` and `cache_last_channel_len` are always stored raw. Default False.
            quant_bits (int): Bits per coordinate for the TurboQuantMSE scalar codebook. Default 4.
        """
        self.cache_aware_model = cache_aware_model
        # Cache aware model should have the following methods:
        if not hasattr(self.cache_aware_model, "get_initial_cache_state"):
            raise ValueError("Cache aware model should have the get_initial_cache_state method")

        self.num_slots = num_slots
        self.cache_disabled = not use_cache
        self.quantize_cache = quantize_cache
        self.quant_bits = quant_bits
        self._quantizer: TurboQuantMSE | None = None
        self.cache_last_channel: Cache | None = None
        self.cache_last_time: Cache | None = None
        self.cache_last_channel_len = None
        self._step_mem_logs_remaining = 3
        self.reset()

    def reset(self) -> None:
        """Resets the context manager"""
        if self.cache_disabled:
            return

        self.streamidx2slotidx = {}
        self.slotidx2streamidx = {}
        self.free_slots = Queue(self.num_slots)
        for i in range(self.num_slots):
            self.free_slots.put(i)
        torch.cuda.reset_peak_memory_stats()
        (
            initial_cache_last_channel,  # [17, B, 70, 512]
            initial_cache_last_time,  # [17, B, 512, 8]
            self.cache_last_channel_len,  # B
        ) = self.cache_aware_model.get_initial_cache_state(self.num_slots)

        if self.quantize_cache:
            if self._quantizer is None:
                # `cache_last_channel` carries the hidden vector at axis 3 (shape [..., D]) and
                # `cache_last_time` carries it at axis 2 (shape [..., D, T]). Both have the same
                # D, so a single TurboQuantMSE instance can serve both.
                self._quantizer = TurboQuantMSE(
                    d=initial_cache_last_channel.shape[-1],
                    bits=self.quant_bits,
                    device=initial_cache_last_channel.device,
                    dtype=initial_cache_last_channel.dtype,
                )
            # Build the quantized caches from shape specs rather than quantizing the float tensors.
            # All-zero norms decode to zero regardless of indices, so this matches the previous
            # zero-initial state without paying the peak transient of `quantizer.quantize()`.
            self.cache_last_channel = QuantizedCache.empty(
                shape=tuple(initial_cache_last_channel.shape),
                dtype=initial_cache_last_channel.dtype,
                device=initial_cache_last_channel.device,
                quantizer=self._quantizer,
                vec_axis=3,
            )
            self.cache_last_time = QuantizedCache.empty(
                shape=tuple(initial_cache_last_time.shape),
                dtype=initial_cache_last_time.dtype,
                device=initial_cache_last_time.device,
                quantizer=self._quantizer,
                vec_axis=2,
            )
            del initial_cache_last_channel
            del initial_cache_last_time
        else:
            self.cache_last_channel = RawCache(initial_cache_last_channel)
            self.cache_last_time = RawCache(initial_cache_last_time)

        torch.cuda.empty_cache()
        tag = "quant" if self.quantize_cache else "raw  "
        cache_mb = self.cache_last_channel.storage_nbytes() / 1024**2
        clt_mb = self.cache_last_time.storage_nbytes() / 1024**2
        cll_mb = (
            self.cache_last_channel_len.element_size() * self.cache_last_channel_len.numel()
        ) / 1024**2
        print(
            f"[cache-mem {tag}] cache size : "
            f"cache_last_channel={cache_mb:.1f} MB, "
            f"cache_last_time={clt_mb:.1f} MB, "
            f"cache_last_channel_len={cll_mb:.3f} MB"
        )
        print(
            f"[cache-mem {tag}] after init : "
            f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
        )
        print(
            f"[cache-mem {tag}] peak init  : "
            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
        )
        torch.cuda.reset_peak_memory_stats()

        self.device = self.cache_last_channel.device

    def _reset_slots(self, slot_ids: list[int]) -> None:
        """
        Resets the slots for the given slot_ids
        Args:
            slot_ids: list of slot indices to reset
        """
        if self.cache_disabled:
            return

        slot_ids_tensor = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
        self.cache_last_channel.reset_slots(slot_ids_tensor)
        self.cache_last_time.reset_slots(slot_ids_tensor)
        self.cache_last_channel_len.index_fill_(0, slot_ids_tensor, 0)

        # free the slot, so that it can be used by other streams
        # remove the stream from the mappings
        for slot_id in slot_ids:
            self.free_slots.put(slot_id)
            stream_id = self.slotidx2streamidx[slot_id]
            del self.slotidx2streamidx[slot_id]
            del self.streamidx2slotidx[stream_id]

    def update_cache(self, stream_ids: list[int], new_context: CacheAwareContext, mapping: dict) -> None:
        """
        Updates the cache for the given stream_ids with the new_context
        Args:
            stream_ids (list[int]): list of stream ids
            new_context (CacheAwareContext): new context to update corresponding to the stream_ids
            mapping (dict): mapping between the old and new slots
        """
        if self.cache_disabled:
            return

        slot_ids_list = [self.streamidx2slotidx[sid] for sid in stream_ids]
        slot_ids = torch.tensor(slot_ids_list, device=self.device, dtype=torch.long)
        tgt_slot_ids = torch.tensor(
            [mapping[sid] for sid in slot_ids_list],
            device=self.device,
            dtype=torch.long,
        )

        # In-place copy along batch/slot dimension
        self.cache_last_channel.update_slots(slot_ids, new_context.cache_last_channel, tgt_slot_ids)
        self.cache_last_time.update_slots(slot_ids, new_context.cache_last_time, tgt_slot_ids)
        self.cache_last_channel_len.index_copy_(
            0, slot_ids, new_context.cache_last_channel_len.index_select(0, tgt_slot_ids)
        )

        if self._step_mem_logs_remaining > 0:
            tag = "quant" if self.quantize_cache else "raw  "
            print(
                f"[cache-mem {tag}] after step: "
                f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
            )
            print(
                f"[cache-mem {tag}] peak step : "
                f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
            )
            self._step_mem_logs_remaining -= 1

    def reset_slots(self, stream_ids: list[int], eos_flags: list[bool]) -> None:
        """
        Resets the slots for the finished streams
        Args:
            stream_ids (list[int]): list of stream ids
            eos_flags (list[bool]): list of eos flags indicating whether the stream has finished
        """
        if self.cache_disabled:
            return

        if len(stream_ids) != len(eos_flags):
            raise ValueError("stream_ids and eos_flags must have the same length")

        if len(stream_ids) == 0:
            return

        # reset the slots for finished streams
        self._reset_slots([self.streamidx2slotidx[sid] for sid, eos in zip(stream_ids, eos_flags) if eos])

    def get_context(self, stream_ids: list[int]) -> tuple[CacheAwareContext, dict]:
        """
        Retrieves the context from the cache for the given stream_ids
        Args:
            stream_ids (list[int]): list of stream ids
        Returns:
            context (CacheAwareContext): context for the given stream_ids
            mapping (dict): mapping between the cache and retrieved context
        """

        if len(stream_ids) == 0 or self.cache_disabled:
            # Create a dummy context with None values
            return CacheAwareContext(), {}

        if self._step_mem_logs_remaining > 0:
            torch.cuda.reset_peak_memory_stats()

        # if the stream_id is new, we need to assign a slot to it
        for stream_id in stream_ids:
            if stream_id not in self.streamidx2slotidx:
                if self.free_slots.empty():
                    raise RuntimeError("No free slots available")
                slot_idx = self.free_slots.get()
                self.streamidx2slotidx[stream_id] = slot_idx
                self.slotidx2streamidx[slot_idx] = stream_id

        # get the cache for the particular stream_ids
        slot_ids = [self.streamidx2slotidx[stream_id] for stream_id in stream_ids]
        cache_last_channel = self.cache_last_channel.gather_slots(slot_ids)
        cache_last_time = self.cache_last_time.gather_slots(slot_ids)
        cache_last_channel_len = self.cache_last_channel_len[slot_ids]

        # create a context object
        context = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        # mapping between cache and context
        mapping = dict(zip(slot_ids, range(len(slot_ids))))
        return context, mapping
