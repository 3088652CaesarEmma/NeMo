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


class CastCache(Cache):
    """Stores `cache_last_channel` cast to a reduced-precision dtype (e.g. fp16, bf16).

    The original dtype is remembered so `gather_slots` returns tensors in the source dtype
    expected by the encoder. Compared to `QuantizedCache` this has no rotation/codebook
    overhead — just a cast on update and another cast on gather — at the cost of weaker
    compression (2× vs ~4×).
    """

    def __init__(self, tensor: Tensor, storage_dtype: torch.dtype):
        """
        Args:
            tensor (Tensor): initial cache tensor of shape [L, B, T, D].
            storage_dtype (torch.dtype): dtype used for backing storage (e.g. torch.float16).
        """
        self._source_dtype = tensor.dtype
        self._data = tensor.to(storage_dtype)

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        source_dtype: torch.dtype,
        storage_dtype: torch.dtype,
        device: torch.device,
    ) -> "CastCache":
        """Construct a zero-valued cast cache without materialising the full-precision tensor."""
        obj = cls.__new__(cls)
        obj._source_dtype = source_dtype
        obj._data = torch.zeros(shape, dtype=storage_dtype, device=device)
        return obj

    @property
    def device(self) -> torch.device:
        return self._data.device

    def reset_slots(self, slot_ids: Tensor) -> None:
        self._data.index_fill_(1, slot_ids, 0.0)

    def update_slots(self, dst_slot_ids: Tensor, src: Tensor, src_slot_ids: Tensor) -> None:
        src_slice = src.index_select(1, src_slot_ids).to(self._data.dtype)
        self._data.index_copy_(1, dst_slot_ids, src_slice)

    def gather_slots(self, slot_ids: list[int]) -> Tensor:
        return self._data[:, slot_ids, :, :].to(self._source_dtype)

    def storage_nbytes(self) -> int:
        return self._data.element_size() * self._data.numel()


class Int8Cache(Cache):
    """Stores the cache as per-vector absmax-scaled int8 along `vec_axis`.

    For each vector v along `vec_axis`:
        scale = max(|v|) / 127
        v_int8 = clamp(round(v / scale), -128, 127)
    Reconstruction: v ≈ v_int8 * scale, cast back to the source dtype.

    Storage is `(indices: int8, scale: source_dtype)` with the slot dimension on dim 1
    (matching `RawCache`/`CastCache`/`QuantizedCache`), so all `index_fill_` /
    `index_copy_` / `index_select` slot ops carry over unchanged.

    Compared to `QuantizedCache` at bits=8 this has the same storage size (1 byte per
    coordinate) but no rotation/codebook overhead — quantize is one absmax + round,
    dequantize is one multiply. Compared to `CastCache(fp16)` it is 2× smaller.
    """

    _INT8_MAX = 127.0

    def __init__(self, tensor: Tensor, vec_axis: int = 3):
        """
        Args:
            tensor (Tensor): initial cache tensor.
            vec_axis (int): axis that holds the per-vector quantization unit. Default 3.
        """
        self._source_dtype = tensor.dtype
        self.vec_axis = vec_axis
        self._indices, self._scale = self._quantize(tensor, vec_axis)

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        source_dtype: torch.dtype,
        device: torch.device,
        vec_axis: int = 3,
    ) -> "Int8Cache":
        """Construct a zero-valued int8 cache without materialising the float tensor.

        Dequantizing an all-zero `scale` reconstructs to zero regardless of indices, so this
        matches `Int8Cache(zeros(shape))` while avoiding the float-cache allocation.
        """
        obj = cls.__new__(cls)
        obj._source_dtype = source_dtype
        obj.vec_axis = vec_axis
        obj._indices = torch.zeros(shape, dtype=torch.int8, device=device)
        scale_shape = list(shape)
        scale_shape.pop(vec_axis)
        obj._scale = torch.zeros(scale_shape, dtype=source_dtype, device=device)
        return obj

    @classmethod
    def _quantize(cls, x: Tensor, vec_axis: int) -> tuple[Tensor, Tensor]:
        abs_max = x.abs().amax(dim=vec_axis, keepdim=True)
        # clamp_min guards against all-zero vectors (would produce NaN under x / scale).
        scale = (abs_max / cls._INT8_MAX).clamp_min(1e-12)
        q = (x / scale).round().clamp_(-cls._INT8_MAX - 1.0, cls._INT8_MAX).to(torch.int8)
        return q, scale.squeeze(vec_axis)

    @property
    def device(self) -> torch.device:
        return self._indices.device

    def reset_slots(self, slot_ids: Tensor) -> None:
        # Dequantize multiplies indices by scale; zero scale → zero output regardless of indices.
        self._scale.index_fill_(1, slot_ids, 0.0)

    def update_slots(self, dst_slot_ids: Tensor, src: Tensor, src_slot_ids: Tensor) -> None:
        # Slice before quantizing so transient float buffers are sized to len(src_slot_ids).
        src_slice = src.index_select(1, src_slot_ids)
        src_q, src_scale = self._quantize(src_slice, self.vec_axis)
        self._indices.index_copy_(1, dst_slot_ids, src_q)
        self._scale.index_copy_(1, dst_slot_ids, src_scale)

    def gather_slots(self, slot_ids: list[int]) -> Tensor:
        q = self._indices[:, slot_ids]
        scale = self._scale[:, slot_ids]
        return q.to(self._source_dtype) * scale.unsqueeze(self.vec_axis).to(self._source_dtype)

    def storage_nbytes(self) -> int:
        return (
            self._indices.element_size() * self._indices.numel()
            + self._scale.element_size() * self._scale.numel()
        )


class Int4Cache(Cache):
    """Stores the cache as per-vector absmax-scaled int4 with 2-nibble bit packing.

    For each vector v along `vec_axis`:
        scale  = max(|v|) / 7
        nibble = clamp(round(v / scale), -7, 7) + 7   # stored unsigned in [0, 14]
    Two nibbles are packed into one uint8 along `vec_axis`, so storage along that
    axis is half the source extent. Reconstruction: v ≈ (nibble - 7) * scale.

    Compared to `Int8Cache` this is 2× smaller again at the cost of higher quant
    error. Compared to `QuantizedCache` with bits=4 it skips the rotation +
    codebook lookup at the cost of slightly worse MSE on Gaussian coordinates.
    """

    _INT4_MAX = 7.0
    _NIBBLE_ZP = 7  # unsigned offset so signed [-7, 7] maps to [0, 14]
    _PACK_FACTOR = 2

    def __init__(self, tensor: Tensor, vec_axis: int = 3):
        """
        Args:
            tensor (Tensor): initial cache tensor.
            vec_axis (int): axis that holds the per-vector quantization unit. Default 3.
        """
        if tensor.shape[vec_axis] % self._PACK_FACTOR != 0:
            raise ValueError(
                f"size at vec_axis ({tensor.shape[vec_axis]}) must be even for 4-bit packing"
            )
        self._source_dtype = tensor.dtype
        self.vec_axis = vec_axis
        self._indices, self._scale = self._quantize(tensor, vec_axis)

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        source_dtype: torch.dtype,
        device: torch.device,
        vec_axis: int = 3,
    ) -> "Int4Cache":
        """Construct a zero-valued int4 cache without materialising the float tensor.

        Dequantizing an all-zero `scale` reconstructs to zero regardless of indices,
        so this matches `Int4Cache(zeros(shape))` while avoiding the float-cache
        allocation and the transient buffers inside `_quantize`.
        """
        if shape[vec_axis] % cls._PACK_FACTOR != 0:
            raise ValueError(
                f"size at vec_axis ({shape[vec_axis]}) must be even for 4-bit packing"
            )
        obj = cls.__new__(cls)
        obj._source_dtype = source_dtype
        obj.vec_axis = vec_axis
        packed_shape = list(shape)
        packed_shape[vec_axis] = shape[vec_axis] // cls._PACK_FACTOR
        obj._indices = torch.zeros(packed_shape, dtype=torch.uint8, device=device)
        scale_shape = list(shape)
        scale_shape.pop(vec_axis)
        obj._scale = torch.zeros(scale_shape, dtype=source_dtype, device=device)
        return obj

    @classmethod
    def _quantize(cls, x: Tensor, vec_axis: int) -> tuple[Tensor, Tensor]:
        abs_max = x.abs().amax(dim=vec_axis, keepdim=True)
        # clamp_min guards against all-zero vectors (would produce NaN under x / scale).
        scale = (abs_max / cls._INT4_MAX).clamp_min(1e-12)
        # Map to unsigned nibble [0, 14] in a chain of in-place ops on the divided tensor.
        nibbles = (x / scale).round_().clamp_(-cls._INT4_MAX, cls._INT4_MAX).add_(cls._NIBBLE_ZP).to(torch.uint8)
        # Pack along vec_axis: two nibbles per byte. movedim so packing is along the last axis.
        n_last = nibbles.movedim(vec_axis, -1)
        packed_last = n_last[..., 0::2] | (n_last[..., 1::2] << 4)
        packed = packed_last.movedim(-1, vec_axis).contiguous()
        return packed, scale.squeeze(vec_axis)

    @property
    def device(self) -> torch.device:
        return self._indices.device

    def reset_slots(self, slot_ids: Tensor) -> None:
        # Dequantize multiplies by scale; zero scale → zero output regardless of indices.
        self._scale.index_fill_(1, slot_ids, 0.0)

    def update_slots(self, dst_slot_ids: Tensor, src: Tensor, src_slot_ids: Tensor) -> None:
        # Slice before quantizing so transient float buffers are sized to len(src_slot_ids).
        src_slice = src.index_select(1, src_slot_ids)
        src_q, src_scale = self._quantize(src_slice, self.vec_axis)
        self._indices.index_copy_(1, dst_slot_ids, src_q)
        self._scale.index_copy_(1, dst_slot_ids, src_scale)

    def gather_slots(self, slot_ids: list[int]) -> Tensor:
        packed = self._indices[:, slot_ids]
        scale = self._scale[:, slot_ids]
        # Unpack two nibbles per byte along vec_axis.
        packed_last = packed.movedim(self.vec_axis, -1)
        *prefix, half = packed_last.shape
        unpacked = torch.empty(*prefix, half * self._PACK_FACTOR, dtype=torch.uint8, device=packed.device)
        unpacked[..., 0::2] = packed_last & 0x0F
        unpacked[..., 1::2] = (packed_last >> 4) & 0x0F
        unpacked = unpacked.movedim(-1, self.vec_axis)
        return (unpacked.to(self._source_dtype) - self._NIBBLE_ZP) * scale.unsqueeze(self.vec_axis).to(self._source_dtype)

    def storage_nbytes(self) -> int:
        return (
            self._indices.element_size() * self._indices.numel()
            + self._scale.element_size() * self._scale.numel()
        )


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
