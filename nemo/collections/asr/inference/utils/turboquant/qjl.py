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


class QJL:
    """Quantized Johnson-Lindenstrauss: 1-bit projection of unit-direction vectors.

    Used as the residual stage of TurboQuantProd. quantize() returns sign(S v);
    dequantize() returns (sqrt(pi/2)/d) * ||r|| * (sign(S v) @ S), so that the
    full TurboQuantProd reconstruction is unbiased for inner products.
    """

    def __init__(
        self,
        d: int,
        seed: int = 43,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.d = d
        self.device = device
        self.dtype = dtype
        gen = torch.Generator(device="cpu").manual_seed(seed)
        s = torch.randn(d, d, generator=gen, dtype=torch.float64)
        self.S = s.to(device=device, dtype=dtype)
        self._scale = math.sqrt(math.pi / 2) / d

    def quantize(self, direction: torch.Tensor) -> torch.Tensor:
        """direction: (..., d) unit-norm residual direction. Returns int8 signs."""
        proj = direction @ self.S.T
        return torch.sign(proj).to(torch.int8)

    def dequantize(self, sign: torch.Tensor, residual_norm: torch.Tensor) -> torch.Tensor:
        """Reconstruct the residual contribution (..., d)."""
        contrib = sign.to(self.dtype) @ self.S
        return self._scale * residual_norm.unsqueeze(-1) * contrib
