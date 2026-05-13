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

import math
import torch

from nemo.collections.asr.inference.utils.turboquant.lloyd_max import lloyd_max_centroids
from nemo.collections.asr.inference.utils.turboquant.rotation import random_rotation


class TurboQuantMSE:
    """TurboQuant Algorithm 1: rotation + Lloyd-Max scalar quantization.

    Designed for a 4D tensor [L, B, D, H] with vec_axis=2 (D = hidden dim),
    but works for any shape whose `vec_axis` dimension equals `d`. A single
    rotation and codebook are shared across every other axis.

    quantize(x) returns (indices, norms):
      - indices: same shape as x, uint8, holding centroid indices per coord
      - norms:   x.shape with vec_axis removed; the per-vector L2 norm of x

    dequantize(indices, norms) returns a tensor of the original shape.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        if bits not in (1, 2, 4, 8):
            raise ValueError(f"bits must be one of 1, 2, 4, 8 (for bit-packing), got {bits}")
        self.d = d
        self.bits = bits
        # Number of indices that fit in a single uint8 byte; the storage axis shrinks by this factor.
        self.items_per_byte = 8 // bits
        if d % self.items_per_byte != 0:
            raise ValueError(
                f"d={d} must be divisible by items_per_byte={self.items_per_byte} for bits={bits} packing"
            )
        self._mask = (1 << bits) - 1
        self.device = device
        self.dtype = dtype
        self.rotation = random_rotation(d, seed=seed, device=device, dtype=dtype)
        centroids_np = lloyd_max_centroids(d, bits)
        centroids = torch.as_tensor(centroids_np, device=device, dtype=dtype)
        # bucketize in quantize() requires ascending boundaries; Lloyd-Max preserves order but sort defensively.
        self.centroids, _ = torch.sort(centroids)
        # Midpoints between adjacent centroids — the decision boundaries for nearest-centroid lookup.
        self.midpoints = 0.5 * (self.centroids[:-1] + self.centroids[1:])

    def _pack_last(self, idx: torch.Tensor) -> torch.Tensor:
        """Pack `items_per_byte` indices into each uint8 along the last axis.

        Args:
            idx: integer tensor of shape [..., D] with values in [0, 2**bits - 1].
        Returns:
            uint8 tensor of shape [..., D // items_per_byte].
        """
        n = self.items_per_byte
        if n == 1:
            return idx.to(torch.uint8)
        *prefix, last = idx.shape
        grouped = idx.view(*prefix, last // n, n)
        packed = grouped[..., 0].to(torch.uint8)
        for i in range(1, n):
            packed = packed | (grouped[..., i].to(torch.uint8) << (self.bits * i))
        return packed

    def _unpack_last(self, packed: torch.Tensor) -> torch.Tensor:
        """Inverse of `_pack_last`: expand each uint8 back into `items_per_byte` indices.

        Args:
            packed: uint8 tensor of shape [..., D // items_per_byte].
        Returns:
            uint8 tensor of shape [..., D] with values in [0, 2**bits - 1].
        """
        n = self.items_per_byte
        if n == 1:
            return packed
        *prefix, half = packed.shape
        out = torch.empty(*prefix, half * n, dtype=torch.uint8, device=packed.device)
        for i in range(n):
            out[..., i::n] = (packed >> (self.bits * i)) & self._mask
        return out

    def quantize(self, x: torch.Tensor, vec_axis: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        x_p = x.movedim(vec_axis, -1)
        if x_p.shape[-1] != self.d:
            raise ValueError(
                f"expected size {self.d} at axis {vec_axis}, got {x_p.shape[-1]}"
            )

        # Rotation is orthogonal, so ||x_p|| == ||x_p @ rotation.T||. Rotate first, then
        # normalize in place — avoids the separate `x_unit = x_p / norms` allocation.
        y = x_p @ self.rotation.T
        norms = y.norm(dim=-1)
        y.div_(norms.unsqueeze(-1).clamp_min(1e-12))

        # Nearest centroid via bucketize on midpoints: yields the final index in one pass,
        # no left/right/dist intermediates. int32 output (256 centroids fit easily).
        idx_last = torch.bucketize(y.contiguous(), self.midpoints, out_int32=True)
        del y

        # Pack along the last axis (bits < 8 → multiple indices per byte) and place vec_axis back.
        packed_last = self._pack_last(idx_last)
        indices = packed_last.movedim(-1, vec_axis).contiguous()
        return indices, norms

    def dequantize(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        vec_axis: int = 2,
    ) -> torch.Tensor:
        packed_last = indices.movedim(vec_axis, -1)
        idx_p = self._unpack_last(packed_last).long()
        y_hat = self.centroids[idx_p]
        x_hat = y_hat @ self.rotation
        x_hat.mul_(norms.unsqueeze(-1).to(x_hat.dtype))
        return x_hat.movedim(-1, vec_axis)

    def upper_bound(self) -> float:
        """Theoretical MSE upper bound (Theorem 1) for unit-norm vectors:
            D_mse <= (sqrt(3) * pi / 2) * 4^(-b)
        """
        return (math.sqrt(3) * math.pi / 2) * (4 ** -self.bits)
