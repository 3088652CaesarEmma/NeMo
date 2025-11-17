# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import abc
from abc import abstractmethod
from typing import cast

import torch
import torch.nn as nn

from nemo.collections.asr.parts.context_biasing import GPUBoostingTreeModel
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel


class FusedGPUBiasingModelBase(abc.ABC, nn.Module):
    @abstractmethod
    def advance(
        self, states: torch.Tensor, model_ids: torch.Tensor, eos_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            model_ids: ids of models for each state
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        pass


class FusedGPUBiasingModelNonBatched(FusedGPUBiasingModelBase):
    """Reference slow implementation"""

    def __init__(self, models: list[NGramGPULanguageModel]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.vocab_size = None
        self.float_dtype = None
        for model in self.models:
            self._check_model_compatibility(model)

    def _check_model_compatibility(self, model):
        if self.vocab_size is None:
            self.vocab_size = model.vocab_size
        else:
            if self.vocab_size != model.vocab_size:
                raise ValueError(f"Inconsistent vocab size: {model.vocab_size}")
        if self.float_dtype is None:
            self.float_dtype = model.arc_weights.dtype

    def advance(
        self, states: torch.Tensor, model_ids: torch.Tensor, eos_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            model_ids: ids of models for each state
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        batch_size = states.shape[0]
        assert model_ids.shape[0] == batch_size
        device = next(iter(self.parameters())).device
        scores = torch.zeros([batch_size, self.vocab_size], device=device, dtype=self.float_dtype)
        new_states = torch.zeros([batch_size, self.vocab_size], dtype=torch.long, device=device)
        model_ids = model_ids.to("cpu").tolist()
        for batch_i, model_id in enumerate(model_ids):
            if model_id < 0:
                continue
            model = cast(NGramGPULanguageModel, self.models[model_id])
            scores_i, new_states_i = model.advance(states[batch_i : batch_i + 1], eos_id=eos_id)
            scores[batch_i : batch_i + 1] = scores_i
            new_states[batch_i : batch_i + 1] = new_states_i
        return scores, new_states
