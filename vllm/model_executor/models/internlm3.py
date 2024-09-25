# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (PagedAttention,
                                                  PagedCrossAttention)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear
                                               )
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class InternLM3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.w2 = RowParallelLinear(intermediate_size,
                                    hidden_size,
                                    bias=False,
                                    linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x


class InternLM3SelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.wqkv = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.wo = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   self.scaling,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.wqkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.wo(attn_output)
        return output


class InternLM3CrossAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.wq = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            linear_method=linear_method,
        )
        self.wo = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = PagedCrossAttention(self.num_heads,
                                   self.head_dim,
                                   self.scaling,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        key_states: Optional[torch.Tensor],
        value_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q, _ = self.wq(hidden_states)
        q, _ = self.rotary_emb(positions, q)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, key_states, value_states, k_cache, v_cache, input_metadata)
        output, _ = self.wo(attn_output)
        return output


class InternLM3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
        is_cross_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.is_cross_decoder = is_cross_decoder
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          32768)
        if is_cross_decoder:
            self.attention = InternLM3CrossAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                linear_method=linear_method,
            )
        else:
            self.attention = InternLM3SelfAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                linear_method=linear_method,
            )

        self.feed_forward = InternLM3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.attention_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(
                hidden_states, residual)
        if self.is_cross_decoder:
            hidden_states = self.attention(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                input_metadata=input_metadata,
                key_states=key_states,
                value_states=value_states
            )
        else:
            hidden_states = self.attention(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                input_metadata=input_metadata,
            )
        # Fully Connected
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class InternLM3SelfDecoder(nn.Module):

    def __init__(self,
                config: PretrainedConfig,
                linear_method: Optional[LinearMethodBase] = None,
                ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            InternLM3DecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor] = None):

        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )
        return hidden_states, residual


class InternLM3CrossDecoder(nn.Module):

    def __init__(self,
                config: PretrainedConfig,
                linear_method: Optional[LinearMethodBase] = None,
                ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            InternLM3DecoderLayer(config, linear_method, is_cross_decoder=True)
            for _ in range(config.num_hidden_layers)
        ])

        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.hidden_size = config.hidden_size
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.key_value_groups = int(self.num_heads / self.num_kv_heads)
        self.wk = ColumnParallelLinear(self.hidden_size, 
                                       self.total_num_kv_heads * self.head_dim, 
                                       bias=config.bias,
                                       linear_method=linear_method,
                                       )
        self.wv = ColumnParallelLinear(self.hidden_size,
                                       self.total_num_kv_heads * self.head_dim,
                                       bias=config.bias,
                                       linear_method=linear_method,)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
  
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor] = None):

        hidden_states_norm, residual = self.norm(hidden_states,  residual)

        key_states, _ = self.wk(hidden_states_norm)
        value_states, _ = self.wv(hidden_states_norm)
        key_states, _ = self.rotary_emb(positions, key_states)
        pre_hidden_states = None
        pre_residule = None
        shared_kv_cache = kv_caches[-1]
        # if not in profile_run
        if shared_kv_cache[0] is not None:
            key_states = key_states.view(-1, self.num_kv_heads, self.head_dim)
            value_states = value_states.view(-1, self.num_kv_heads, self.head_dim)
            key_cache, value_cache = shared_kv_cache
            # cache kv outside attn
            cache_ops.reshape_and_cache(
                key_states,
                value_states,
                key_cache,
                value_cache,
                input_metadata.slot_mapping.flatten(),
                input_metadata.kv_cache_dtype,
            )
            
            if input_metadata.is_prompt:
                batch_indices = torch.arange(hidden_states.shape[0], device=key_states.device)
                last_token_indices = input_metadata.prompt_lens - 1
                positions = last_token_indices[:, None]
                pre_hidden_states = hidden_states
                pre_residule = residual
                shared_kv_cache = (None, None)
                hidden_states = hidden_states[batch_indices, last_token_indices, :].unsqueeze(1)
                residual = residual[batch_indices, last_token_indices, :].unsqueeze(1)
            else:
                key_states = None
                value_states = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                shared_kv_cache,
                input_metadata,
                residual,
                key_states=key_states,
                value_states=value_states
            )
        
        if pre_hidden_states is not None:
            batch_indices = torch.arange(hidden_states.shape[0], device=key_states.device)
            last_token_indices = input_metadata.prompt_lens - 1
            pre_hidden_states[batch_indices, last_token_indices, :] = hidden_states.squeeze(1)
            pre_residule[batch_indices, last_token_indices, :] = residual.squeeze(1)
            hidden_states = pre_hidden_states
            residual = pre_residule
        return hidden_states, residual
    

class InternLM3Model(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.self_decoder = InternLM3SelfDecoder(config, linear_method)
        self.cross_decoder = InternLM3CrossDecoder(config, linear_method)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.tok_embeddings(input_ids)
        residual = None
        hidden_states, residual = self.self_decoder(positions, hidden_states, kv_caches, input_metadata, residual)
        hidden_states, residual = self.cross_decoder(positions, hidden_states, kv_caches, input_metadata, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class InternLM3ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = InternLM3Model(config, linear_method)
        self.output = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.output.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                if "wqkv" in name:
                    config = self.config
                    kv_groups = config.num_attention_heads // config.num_key_value_heads
                    head_dim = config.hidden_size // config.num_attention_heads
                    loaded_weight = loaded_weight.view(-1, 2 + kv_groups,
                                                       head_dim,
                                                       loaded_weight.shape[-1])
                    wq, wk, wv = torch.split(loaded_weight, [kv_groups, 1, 1],
                                             dim=1)
                    wq = wq.reshape(-1, wq.shape[-1])
                    wk = wk.reshape(-1, wk.shape[-1])
                    wv = wv.reshape(-1, wv.shape[-1])
                    weight_loader = param.weight_loader
                    weight_loader(param, wq, 'q')
                    weight_loader(param, wk, 'k')
                    weight_loader(param, wv, 'v')
                else:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
