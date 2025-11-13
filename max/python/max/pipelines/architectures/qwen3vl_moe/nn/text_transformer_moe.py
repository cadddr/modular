from collections.abc import Callable

from max import nn;
from max.pipelines.architectures.qwen3.layers.attention import Qwen3Attention
from max.pipelines.architectures.qwen2_5vl.nn.decoder import Qwen25VLDecoder
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.graph import ops;
from max.graph import TensorValue, Weight;
from max.experimental import tensor;
from max.dtype import DType;


class Qwen3TextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size # duplicate assign?
        self.gate_up_proj = Weight(shape=(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = Weight(shape=(self.num_experts, self.expert_dim, self.hidden_size))

        if config.hidden_act == 'silu':
            self.act_fn: Callable[[TensorValue], TensorValue] = ops.silu 
        else:
            raise NotImplementedError()

    def __call__(self, hidden_states: TensorValue, router_weights: TensorValue, router_indices: TensorValue) -> TensorValue: 
        # hidden_states: B, T, D
        # router_weights: B*T, E
        # router_indices: B*T, k
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape((-1, self.hidden_size)) 
        #TODO: implement expert parallelism
        hidden_states = ops.tile(hidden_states, (self.num_experts, 1)) # hf uses torch.repeat - Check
        hidden_states = hidden_states.reshape((self.num_experts, -1, self.hidden_size))
        gate_up = ops.matmul(hidden_states, self.gate_up_proj) # hf uses torch.bmm
        gate, up = ops.chunk(gate_up, chunks=2, axis=-1) # E, B*T, expert_dim each
        next_states = ops.matmul((up * self.act_fn(gate)), self.down_proj) # E, B*T, D
        next_states = next_states.reshape((self.num_experts, batch_size, -1, self.hidden_size))
        router_weights = ops.transpose(router_weights, 0, 1).reshape((self.num_experts, batch_size, -1))
        next_states = next_states * ops.unsqueeze(router_weights, axis=-1) # E, B, T, D
        next_states = ops.sum(next_states, axis=0) # B, T, D
        return next_states


class Qwen3TextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.higgen_size = config.hidden_size
        self.num_experts = config.num_expertss
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(in_dim=self.hidden_size, out_dim=self.num_experts, has_bias=False) # dtype, device
        self.experts = Qwen3TextExperts(config)

    def __call__(self, hidden_states: TensorValue) -> TensorValue: # B, T, D
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape((-1, self.hidden_size)) # TensorValue.reshape accepts ShapeLike
        router_logits = self.gate(hidden_states) # B*T, E
        # hf casts to float
        router_weights = ops.softmax(router_logits, axis=-1) 
        topk_weights, topk_indices = ops.top_k(router_weights, self.top_k, axis=-1)
        topk_weights = topk_weights / ops.sum(topk_weights, axis=-1) # op keeps reduced dim
        # hf casts back to logit dtype
        router_weights = ops.constant(tensor.zeros(router_logits.shape))
        router_weights = ops.scatter(router_weights, topk_weights, topk_indices, axis=1)
        hidden_states = hidden_states.reshape((batch_size, -1, self.hidden_size)) # why reshape if flattened again in experts
        experts_out = self.experts(hidden_states, router_weights, topk_indices)
        return experts_out
        

class Qwen3TextTransformerMoe(nn.Transformer):
    """
        Reusing Qwen3 Text Transformer by replaceing MLP with MoE
        and incorporateing vision embeddings
    """
    def __init__(self, config):                    
        rope = nn.Llama3RotaryEmbedding( # TODO: 3d
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta, 
            max_seq_len=config.max_seq_len, #
            head_dim=config.kv_params.head_dim, #
            interleaved=config.interleaved_rope_weights, #
            scaling_params=config.rope_scaling_params, #
        )

        create_norm: Callable[..., nn.Module] = functools.partial(
            nn.RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,  # disable Gemma3-style scaling
        )

        linear_cls: Callable[..., nn.Linear] = functools.partial(
            Linear, float8_config=config.float8_config #
        )

        attention_cls: Callable[..., Qwen3Attention] = functools.partial(
            Qwen3Attention, # TODO: remove sliding window
            scale=config.attention_multiplier, #
            has_bias=config.attention_bias,
        )
        mlp_cls = Qwen3TextSparseMoeBlock
            
        layers = [
            nn.transformer.TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params, #
                    layer_idx=i,
                    dtype=config.dtype, #
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices, #
                ),
                mlp=mlp_cls(
                    config
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier, #
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype #
        embedding_output_quantization = config.model_quantization_encoding #
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0], #
            quantization_encoding=embedding_output_quantization,
        )
        output = Linear( # why not linear_cls
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0], #
            quantization_encoding=embedding_output_quantization,
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params, #
            rope=rope,
            return_logits=config.return_logits, #
            embedding_multiplier=config.embedding_multiplier, #
        )

    def __call__(self,
        tokens,
        return_n_logits,
        input_row_offsets,
        image_embeddings,
        image_token_indices,
        decoder_position_ids,
        signal_buffers,
        kv_cache_inputs
    ):
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        kv_collection = PagedCacheValues(kv_cache_inputs.blocks, kv_cache_inputs.cache_lengths, kv_cache_inputs.lookup_table, kv_cache_inputs.max_lengths)
        # Create position embeddings shared across the decoder layers.
        freqs_cis = self.rope.freqs_cis
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
            ##################################################################
            # add visual features to the hidden states of first several layers
            if image_embeddings is not None and idx in range(len(image_embeddings)):
                # _deepstack_process
                h = ops.scatter(h, h[image_token_indices] + image_embeddings[idx], image_token_indices, axis=0) # check dims!
            ##################################################################

        # Retrieve a variable number of tokens
        last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = ops.cast(self.lm_head(self.norm(last_h)), DType.float32)
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets, shape=(-1,))
            last_tokens = ops.gather(h, last_indices, axis=0)
            logits = ops.cast(
                self.lm_head(self.norm(last_tokens)), DType.float32
            )
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(self.norm(h)), DType.float32)
            offsets = input_row_offsets

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)


class Qwen3VLDecoderMoe(Qwen25VLDecoder):
    """"
    Reusing Qwen2.5 VL, which already incorporates vision embeddings, also swapping MLP for MoE.
    Differences from Qwen3 VL:
    - vision embeddings are passed as input_embeds rather than concatenated to text hidden states
    DO NOT USE
    """
    mlp_cls = Qwen3TextSparseMoeBlock

# TODO: sharding
