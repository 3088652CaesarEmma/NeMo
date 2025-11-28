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
from dataclasses import dataclass, field
from typing import Callable, cast

import torch
import torch.nn as nn

from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeModelConfig,
    GPUBoostingTreeModel,
)
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.common.tokenizers import TokenizerSpec


@dataclass
class BiasingRequestItemConfig:
    boosting_model_cfg: BoostingTreeModelConfig = field(default_factory=BoostingTreeModelConfig)
    boosting_model_alpha: float = 1.0
    multi_model_id: int | None = None  # compiled model id
    auto_manage_multi_model: bool = True

    def is_empty(self):
        if self.multi_model_id is not None:
            return False
        if not self.boosting_model_cfg.is_empty(self.boosting_model_cfg):
            return False
        return True

    def get_model(self, tokenizer: TokenizerSpec) -> NGramGPULanguageModel | GPUBoostingTreeModel | None:
        if self.boosting_model_cfg.is_empty(self.boosting_model_cfg):
            return None
        boosting_model = GPUBoostingTreeModel.from_config(self.boosting_model_cfg, tokenizer=tokenizer)
        return boosting_model

    def add_to_multi_model(self, tokenizer: TokenizerSpec, biasing_multi_model: "GPUBiasingMultiModelBase"):
        boosting_model = self.get_model(tokenizer=tokenizer)
        if boosting_model is None:
            raise ValueError("Nothing to add, biasing model is empty")
        self.multi_model_id = biasing_multi_model.add_model(model=boosting_model, alpha=self.boosting_model_alpha)

    def remove_from_multi_model(self, biasing_multi_model: "GPUBiasingMultiModelBase"):
        if self.multi_model_id is None:
            # nothing to remove
            return
        biasing_multi_model.remove_model(self.multi_model_id)
        self.multi_model_id = None


class GPUBiasingMultiModelBase(abc.ABC, nn.Module):
    @abstractmethod
    def add_model(self, model: NGramGPULanguageModel, alpha: float = 1.0) -> int:
        raise NotImplementedError

    @abstractmethod
    def remove_model(self, model_id: int):
        raise NotImplementedError

    @staticmethod
    def compatible_with_cuda_graphs() -> bool:
        """True if model can be compiled as a part of CUDA graph, False otherwise"""
        return False

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

    @abstractmethod
    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        pass


class GPUBiasingMultiModelReference(GPUBiasingMultiModelBase):
    """Reference implementation (incompatible with CUDA graphs)"""

    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([])
        self.alphas: list[float] = []
        self.vocab_size: int | None = None
        self.float_dtype: torch.dtype | None = None
        self.bos_state: int | None = None
        self.start_state: int | None = None
        self._params_defined = False
        self.free_ids = set()
        self._device = torch.device("cpu")

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        self._device = device
        return super().to(*args, **kwargs)

    def _check_model_compatibility(self, model: NGramGPULanguageModel):
        if self.vocab_size != model.vocab_size:
            raise ValueError(f"Inconsistent vocab size: {model.vocab_size}")
        if self.bos_state != model.bos_state:
            raise ValueError(f"Inconsistent bos state: {self.bos_state} vs {model.bos_state}")
        if self.start_state != model.START_STATE:
            raise ValueError(f"Inconsistent start state: {self.start_state} vs {model.START_STATE}")

    def add_model(self, model: NGramGPULanguageModel, alpha: float = 1.0) -> int:
        if not self._params_defined:
            # there were no previous models
            self.vocab_size = model.vocab_size
            self.bos_state = model.bos_state
            self.start_state = model.START_STATE
            self.float_dtype = model.arcs_weights.dtype
            self._params_defined = True
        self._check_model_compatibility(model=model)
        try:
            model_id = self.free_ids.pop()
        except KeyError:
            model_id = None
        if model_id is None:
            model_id = len(self.models)
            self.models.append(model)
            self.alphas.append(alpha)
        else:
            self.models[model_id] = model
            self.alphas[model_id] = alpha
        return model_id

    def remove_model(self, model_id: int):
        self.models[model_id] = nn.Identity()  # dummy nn model
        self.alphas[model_id] = 0.0
        self.free_ids.add(model_id)

    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        if not self._params_defined:
            return torch.zeros([batch_size], device=self._device, dtype=torch.long)
        device = self.models[0].arcs_weights.device
        return torch.full(
            [batch_size], fill_value=self.bos_state if bos else self.start_state, device=device, dtype=torch.long
        )

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
            scores[batch_i : batch_i + 1] = scores_i * self.alphas[model_id]
            new_states[batch_i : batch_i + 1] = new_states_i
        return scores, new_states


class GPUBiasingMultiModel(GPUBiasingMultiModelBase):
    """Efficient multi-model implementation"""

    INIT_NUM_ARCS = 1_000_000
    INIT_NUM_STATES = 1_000_000
    INIT_NUM_MODELS = 128

    def __init__(self, reallocation_callback_fn: Callable | None = None):
        super().__init__()
        self.vocab_size: int | None = None
        self.float_dtype: torch.dtype | None = None
        self.bos_state: int | None = None
        self.start_state: int | None = None
        self._params_defined = False
        self.free_ids = set()

        self.reallocation_callbacks = []
        if reallocation_callback_fn is not None:
            self.reallocation_callbacks.append(reallocation_callback_fn)

        self.num_models = 0

        int_dtype = torch.int64

        self.num_states_all = self.INIT_NUM_STATES
        self.num_arcs_all = self.INIT_NUM_ARCS
        self.num_arcs_extended_all = self.INIT_NUM_ARCS  # + extra padding
        self.num_models = 0

        self.num_states_reserved = self.INIT_NUM_STATES
        self.num_arcs_reserved = self.INIT_NUM_ARCS
        self.num_arcs_extended_reserved = self.INIT_NUM_ARCS  # + extra padding
        self.num_models_reserved = self.INIT_NUM_MODELS

        # store each model properties
        self.alphas = nn.Buffer(torch.zeros([self.num_models_reserved]))
        self.start_states_offsets = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.num_states = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.num_arcs = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))
        self.num_arcs_extended = nn.Buffer(torch.zeros([self.num_models_reserved], dtype=torch.int64))

        # parameters: weights (forward/backoff/final)
        self.arcs_weights = nn.Parameter(torch.zeros([self.num_arcs_extended_all]))
        self.backoff_weights = nn.Parameter(torch.zeros([self.num_states_all]))
        # TODO check final resolved
        self.final_weights = nn.Parameter(torch.zeros([self.num_states_all]))

        # buffers: LM (suffix tree) structure
        # arcs data
        self.from_states = nn.Buffer(torch.zeros([self.num_arcs_extended_all], dtype=int_dtype))
        self.to_states = nn.Buffer(torch.zeros([self.num_arcs_extended_all], dtype=int_dtype))
        self.ilabels = nn.Buffer(torch.zeros([self.num_arcs_extended_all], dtype=int_dtype))

        # states data
        self.backoff_to_states = nn.Buffer(torch.zeros([self.num_states_all], dtype=int_dtype))
        self.start_end_arcs = nn.Buffer(torch.zeros([self.num_states_all, 2], dtype=int_dtype))
        self.state_order = nn.Buffer(torch.zeros([self.num_states_all], dtype=int_dtype))

    def _check_model_compatibility(self, model: NGramGPULanguageModel):
        if self.vocab_size != model.vocab_size:
            raise ValueError(f"Inconsistent vocab size: {model.vocab_size}")
        if self.bos_state != model.bos_state:
            raise ValueError(f"Inconsistent bos state: {self.bos_state} vs {model.bos_state}")
        if self.start_state != model.START_STATE:
            raise ValueError(f"Inconsistent start state: {self.start_state} vs {model.START_STATE}")

    def _maybe_extend(
        self, add_num_models: int, add_num_states: int, add_num_arcs: int, add_num_arcs_extended: int
    ) -> bool:
        """Extend memory, return True if any tensor is reallocated"""
        ...

    def add_model(self, model: GPUBoostingTreeModel, alpha: float = 1.0) -> int:
        if not self._params_defined:
            # there were no previous models
            self.vocab_size = model.vocab_size
            self.bos_state = model.bos_state
            self.start_state = model.START_STATE
            self.float_dtype = model.arcs_weights.dtype
            self._params_defined = True
        self._check_model_compatibility(model=model)

        model_id = self.num_models
        self.num_models += 1

        self.alphas[model_id] = alpha
        # TODO: store model
        return model_id

    def remove_model(self, model_id: int):
        raise NotImplementedError

    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        device = self.arcs_weights.device
        if not self._params_defined:
            return torch.zeros([batch_size], device=device, dtype=torch.long)
        return torch.full(
            [batch_size], fill_value=self.bos_state if bos else self.start_state, device=device, dtype=torch.long
        )

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
        device = self.arcs_weights.device
        scores = torch.zeros([batch_size, self.vocab_size], device=device, dtype=self.float_dtype)
        new_states = torch.zeros([batch_size, self.vocab_size], dtype=torch.long, device=device)
        raise NotImplementedError
        return scores, new_states
