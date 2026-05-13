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


from abc import ABC, abstractmethod

import torch
from torch import Tensor

from nemo.collections.asr.inference.utils.turboquant.mse import TurboQuantMSE


class Cache(ABC):
    """
    Storage for the per-slot `cache_last_channel` tensor used by Cache-Aware models.

    Concrete subclasses choose how the tensor is stored (raw float, or quantized).
    The slot dimension is dim 1 of the underlying [L, B, T, D] tensor.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device on which the underlying storage lives."""

    @abstractmethod
    def reset_slots(self, slot_ids: Tensor) -> None:
        """
        Zero out the given slots along the slot dimension (dim 1), in place.
        Args:
            slot_ids (Tensor): 1-D long tensor of slot indices to zero.
        """

    @abstractmethod
    def update_slots(self, dst_slot_ids: Tensor, src: Tensor, src_slot_ids: Tensor) -> None:
        """
        Copy `src[:, src_slot_ids]` into `self[:, dst_slot_ids]` along the slot dimension, in place.
        Args:
            dst_slot_ids (Tensor): 1-D long tensor of destination slot indices.
            src (Tensor): raw float tensor of shape [L, B, T, D] supplying new values.
            src_slot_ids (Tensor): 1-D long tensor of source slot indices into `src`.
        """

    @abstractmethod
    def gather_slots(self, slot_ids: list[int]) -> Tensor:
        """
        Return the raw float tensor for the requested slots.
        Args:
            slot_ids (list[int]): slot indices to gather.
        Returns:
            Tensor of shape [L, len(slot_ids), T, D] in the original (dequantized) float dtype.
        """

    @abstractmethod
    def storage_nbytes(self) -> int:
        """Total bytes occupied by the underlying storage tensors (excludes Python overhead)."""


class RawCache(Cache):
    """Stores `cache_last_channel` as the original raw float tensor."""

    def __init__(self, tensor: Tensor):
        """
        Args:
            tensor (Tensor): initial cache tensor of shape [L, B, T, D].
        """
        self._data = tensor

    @property
    def device(self) -> torch.device:
        return self._data.device

    def reset_slots(self, slot_ids: Tensor) -> None:
        self._data.index_fill_(1, slot_ids, 0.0)

    def update_slots(self, dst_slot_ids: Tensor, src: Tensor, src_slot_ids: Tensor) -> None:
        self._data.index_copy_(1, dst_slot_ids, src.index_select(1, src_slot_ids))

    def gather_slots(self, slot_ids: list[int]) -> Tensor:
        return self._data[:, slot_ids, :, :]

    def storage_nbytes(self) -> int:
        return self._data.element_size() * self._data.numel()


class QuantizedCache(Cache):
    """
    Stores `cache_last_channel` quantized along its hidden dimension using `TurboQuantMSE`.

    Backing storage is the pair `(indices: uint8 [L, B, T, D], norms: dtype [L, B, T])` returned by
    `TurboQuantMSE.quantize(tensor, vec_axis=vec_axis)`. The slot dimension remains dim 1 on both, so
    `index_fill_` / `index_copy_` / `index_select` operations carry over unchanged from the raw case.
    """

    def __init__(self, tensor: Tensor, quantizer: TurboQuantMSE, vec_axis: int = 3):
        """
        Args:
            tensor (Tensor): initial cache tensor of shape [L, B, T, D].
            quantizer (TurboQuantMSE): rotation + Lloyd-Max scalar quantizer.
            vec_axis (int): axis of `tensor` that holds the hidden-dim vector. Default 3.
        """
        self.quantizer = quantizer
        self.vec_axis = vec_axis
        self._indices, self._norms = quantizer.quantize(tensor, vec_axis=vec_axis)

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        quantizer: TurboQuantMSE,
        vec_axis: int = 3,
    ) -> "QuantizedCache":
        """Construct a zero-valued quantized cache without materialising the float tensor.

        Dequantizing an all-zero `norms` tensor reconstructs to zero regardless of `indices`
        (see `TurboQuantMSE.dequantize`), so this is equivalent to `QuantizedCache(zeros(shape))`
        but avoids the float-cache allocation and the transient buffers inside `quantize()`.

        Args:
            shape (tuple[int, ...]): the would-be float cache shape, e.g. [L, B, T, D].
            dtype (torch.dtype): dtype of the `norms` tensor (the indices are always uint8).
            device (torch.device): device for the underlying storage.
            quantizer (TurboQuantMSE): the quantizer to use for subsequent update/gather calls.
            vec_axis (int): axis of `shape` that holds the hidden-dim vector. Default 3.
        """
        obj = cls.__new__(cls)
        obj.quantizer = quantizer
        obj.vec_axis = vec_axis
        # `_indices` lives in the packed layout: the `vec_axis` extent shrinks by items_per_byte.
        packed_shape = list(shape)
        packed_shape[vec_axis] = shape[vec_axis] // quantizer.items_per_byte
        obj._indices = torch.zeros(packed_shape, dtype=torch.uint8, device=device)
        norms_shape = list(shape)
        norms_shape.pop(vec_axis)
        obj._norms = torch.zeros(norms_shape, dtype=dtype, device=device)
        return obj

    @property
    def device(self) -> torch.device:
        return self._indices.device

    def reset_slots(self, slot_ids: Tensor) -> None:
        # Dequantize multiplies by norms.unsqueeze(-1); zero norms reconstruct to zero regardless of indices.
        self._norms.index_fill_(1, slot_ids, 0.0)

    def update_slots(self, dst_slot_ids: Tensor, src: Tensor, src_slot_ids: Tensor) -> None:
        # Slice the active slots out of `src` before quantizing so the transient float/int32
        # buffers inside `quantize()` are sized to len(src_slot_ids), not the full src batch.
        src_slice = src.index_select(1, src_slot_ids)
        src_indices, src_norms = self.quantizer.quantize(src_slice, vec_axis=self.vec_axis)
        self._indices.index_copy_(1, dst_slot_ids, src_indices)
        self._norms.index_copy_(1, dst_slot_ids, src_norms)

    def gather_slots(self, slot_ids: list[int]) -> Tensor:
        indices = self._indices[:, slot_ids, :, :]
        norms = self._norms[:, slot_ids, :]
        return self.quantizer.dequantize(indices, norms, vec_axis=self.vec_axis)

    def storage_nbytes(self) -> int:
        return (
            self._indices.element_size() * self._indices.numel()
            + self._norms.element_size() * self._norms.numel()
        )
