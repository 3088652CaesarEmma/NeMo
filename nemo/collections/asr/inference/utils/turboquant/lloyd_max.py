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

import numpy as np
from scipy.stats import norm


def lloyd_max_centroids(
    d: int,
    bits: int,
    n_iter: int = 200,
    tol: float = 1e-7,
) -> np.ndarray:
    """Optimal scalar quantizer centroids for N(0, 1/d).

    After a random rotation, each coordinate of a unit-norm vector is
    approximately N(0, 1/d) in high dimensions. We solve the 1D Lloyd-Max
    problem against that Gaussian to produce 2^bits centroids that minimize
    expected squared error.
    """
    n_levels = 1 << bits
    sigma = 1.0 / np.sqrt(d)

    quantiles = (np.arange(n_levels) + 0.5) / n_levels
    centroids = norm.ppf(quantiles, scale=sigma)

    for _ in range(n_iter):
        midpoints = 0.5 * (centroids[:-1] + centroids[1:])
        lefts = np.concatenate([[-np.inf], midpoints])
        rights = np.concatenate([midpoints, [np.inf]])

        # E[X | a < X < b] under N(0, sigma^2):
        #   numerator   = sigma^2 * (phi(a) - phi(b))
        #   denominator = Phi(b) - Phi(a)
        pdf_l = np.where(np.isfinite(lefts), norm.pdf(lefts, scale=sigma), 0.0)
        pdf_r = np.where(np.isfinite(rights), norm.pdf(rights, scale=sigma), 0.0)
        cdf_l = norm.cdf(lefts, scale=sigma)
        cdf_r = norm.cdf(rights, scale=sigma)

        num = sigma * sigma * (pdf_l - pdf_r)
        den = cdf_r - cdf_l
        new_c = np.where(den > 1e-20, num / np.where(den > 0, den, 1.0), centroids)

        if np.max(np.abs(new_c - centroids)) < tol:
            centroids = new_c
            break
        centroids = new_c

    return centroids
