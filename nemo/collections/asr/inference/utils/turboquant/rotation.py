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

import torch


def random_rotation(
    d: int,
    seed: int = 42,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a (d, d) random orthogonal matrix with det = +1.

    QR-decompose a Gaussian (d, d) matrix and sign-correct by diag(R) so the
    result is a proper rotation. Build in float64 for numerical accuracy, then
    cast to the requested device/dtype.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    signs = torch.sign(torch.diag(r))
    q = q * signs.unsqueeze(0)
    return q.to(device=device, dtype=dtype)
