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

"""Stub module for load_distributed_model_weights and related functions.

This module has been moved to the Export-Deploy repository.
"""


def load_distributed_model_weights(*args, **kwargs):
    """Stub function that raises an error directing users to the Export-Deploy repository."""
    raise ImportError(
        "The 'load_distributed_model_weights' function has been moved to a separate repository. "
        "Please use the Export-Deploy repository: https://github.com/NVIDIA-NeMo/Export-Deploy\n"
        "Install with: pip install git+https://github.com/NVIDIA-NeMo/Export-Deploy.git"
    )

