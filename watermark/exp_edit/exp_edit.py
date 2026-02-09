# Copyright 2024 THU-BPM MarkLLM.
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

# ================================================
# exp_edit.py
# Description: Implementation of EXPEdit algorithm
# ================================================

from math import log

import numpy as np
import torch

from exceptions.exceptions import AlgorithmNameMismatchError
from utils.transformers_config import TransformersConfig
from utils.utils import load_config_file
from visualize.data_for_visualization import DataForVisualization

from ..base import BaseConfig, BaseWatermark
from .cython_files.levenshtein import levenshtein
from .mersenne import MersenneRNG


class EXPEditConfig(BaseConfig):
    """Config class for EXPEdit algorithm, load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.pseudo_length = self.config_dict["pseudo_length"]
        self.sequence_length = self.config_dict["sequence_length"]
        self.n_runs = self.config_dict["n_runs"]
        self.p_threshold = self.config_dict["p_threshold"]
        self.key = self.config_dict["key"]
        self.top_k = self.config_dict["top_k"]

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "EXPEdit"


class EXPEditUtils:
    """Utility class for EXPEdit algorithm, contains helper functions."""

    def __init__(self, config: EXPEditConfig, *args, **kwargs) -> None:
        """
        Initialize the EXPEdit utility class.

        Parameters:
            config (EXPEditConfig): Configuration for the EXPEdit algorithm.
        """
        self.config = config
        self.rng = MersenneRNG(self.config.key)
        self.xi = torch.tensor(
            [
                self.rng.rand()
                for _ in range(
                    self.config.pseudo_length * self.config.vocab_size
                )
            ]
        ).view(self.config.pseudo_length, self.config.vocab_size)

    def exp_sampling_pure(
        self, probs: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """使用指数采样公式直接对整个词表进行采样"""
        return torch.argmax(u ** (1 / probs), dim=1).unsqueeze(-1)

    def exp_sampling(
        self, probs: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Sample token using exponential distribution."""

        # If top_k is not specified, use argmax
        if self.config.top_k <= 0:
            return torch.argmax(u ** (1 / probs), dim=1).unsqueeze(-1)

        # Ensure top_k is not greater than the vocabulary size
        top_k = min(self.config.top_k, probs.size(-1))

        # Get the top_k probabilities and their indices
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        # Perform exponential sampling on the top_k probabilities
        sampled_indices = torch.argmax(
            u.gather(-1, top_indices) ** (1 / top_probs), dim=-1
        )

        # Map back the sampled indices to the original vocabulary indices
        return top_indices.gather(-1, sampled_indices.unsqueeze(-1))

    def value_transformation(self, value: float) -> float:
        """Transform value to range [0, 1]."""
        return value / (value + 1)

    def one_run(self, tokens: np.ndarray, xi: np.ndarray) -> tuple:
        """Run one test."""
        k = len(tokens)
        n = len(xi)
        A = np.empty((1, n))
        for i in range(1):
            for j in range(n):
                A[i][j] = levenshtein(
                    tokens[i : i + k], xi[(j + np.arange(k)) % n], 0.0
                )

        return np.min(A), np.argmin(A)


class EXPEdit(BaseWatermark):
    """Top-level class for the EXPEdit algorithm."""

    def __init__(
        self,
        algorithm_config: str | EXPEditConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = EXPEditConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, EXPEditConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be either a path string or a EXPEditConfig instance"
            )

        self.utils = EXPEditUtils(self.config)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)

        # Initialize
        shift = torch.randint(self.config.pseudo_length, (1,))
        inputs = encoded_prompt
        attn = torch.ones_like(inputs)
        past = None

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(
                        inputs[:, -1:],
                        past_key_values=past,
                        attention_mask=attn,
                    )
                else:
                    output = self.config.generation_model(inputs)

            # Get probabilities
            probs = torch.nn.functional.softmax(
                output.logits[:, -1, : self.config.vocab_size], dim=-1
            ).cpu()

            # Sample token to add watermark
            token = self.utils.exp_sampling(
                probs, self.utils.xi[(shift + i) % self.config.pseudo_length, :]
            ).to(self.config.device)

            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update past
            past = output.past_key_values

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(
            watermarked_tokens, skip_special_tokens=True
        )

        return watermarked_text

    @torch.no_grad()
    def generate_watermarked_text_rllm(
        self, prompt: str, *args, **kwargs
    ) -> str:
        """
        为推理型 LLM (如 DeepSeek, QwQ) 生成带水印的文本。
        该方法专为处理这些模型的特殊生成模式设计，即先生成思考过程，然后生成答案。
        仅对 </think> 之后的内容应用水印。

        Args:
            prompt: 原始文本提示，类型为 str

        Returns:
            生成的文本，其中 </think> 后的部分添加了水印
        """
        model_inputs = self.config.generation_tokenizer(
            [prompt], return_tensors="pt"
        ).to(self.config.device)

        # 初始化
        inputs: torch.Tensor = model_inputs["input_ids"]
        attn: torch.Tensor = model_inputs["attention_mask"]
        past = None

        # 获取 EOS token ID
        eos_token_id = getattr(
            self.config.generation_tokenizer, "eos_token_id", None
        )
        if eos_token_id is None:
            eos_token_id = getattr(
                self.config.generation_tokenizer, "pad_token_id", 0
            )

        # 获取 </think> 对应的 token ID
        think_end_tokens: torch.Tensor = (
            self.config.generation_tokenizer.encode(
                "</think>", add_special_tokens=False, return_tensors="pt"
            ).to(self.config.device)[0]
        )

        # 确保 think_end_tokens 是唯一的序列
        assert len(think_end_tokens) == 1, "Failed to encode '</think>' token"
        think_end_token_id = think_end_tokens[
            -1
        ].item()  # 使用最后一个 token 作为标识

        # 标记是否已经遇到 </think>
        passed_think = False
        think_end_pos = -1

        # 初始化随机偏移量
        shift = torch.randint(self.config.pseudo_length, (1,)).item()

        # 生成 tokens
        for i in range(self.config.sequence_length):
            if past:
                output = self.config.generation_model(
                    inputs[:, -1:],
                    past_key_values=past,
                    attention_mask=attn,
                )
            else:
                output = self.config.generation_model(inputs)

            # 获取 logits
            logits = output.logits[:, -1, : self.config.vocab_size]

            # 应用 logits processors 如果有的话
            if hasattr(self.config, "logits_processor_list"):
                logits = self.config.logits_processor_list(inputs, logits)

            probs = torch.softmax(logits, dim=-1)

            # 根据是否已通过 </think> 决定采样方式
            if passed_think:
                # 对于回答部分，使用水印采样
                # 使用 EXP 采样带水印
                token = self.utils.exp_sampling_pure(
                    probs.cpu(),
                    self.utils.xi[(shift + i) % self.config.pseudo_length, :],
                ).to(self.config.device)
            else:
                # 思考部分使用常规采样
                token = torch.multinomial(probs, num_samples=1)

            # 更新输入序列
            inputs = torch.cat([inputs, token], dim=-1)

            # 更新 past_key_values
            past = output.past_key_values

            # 更新注意力掩码
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # 检查是否生成了 EOS 令牌
            if token.item() == eos_token_id:
                break

            # 检查是否生成了 </think> 标记的最后一个 token
            if not passed_think and token.item() == think_end_token_id:
                passed_think = True
                think_end_pos = len(inputs[0]) - 1
                # 重设计数器，从 </think> 之后重新开始计算偏移量
                i = 0

        # 解码生成的令牌序列
        generated_text: str = self.config.generation_tokenizer.decode(
            inputs[0], skip_special_tokens=True
        )
        return generated_text

    def detect_watermark(
        self, text: str, return_dict: bool = True, *args, **kwargs
    ) -> dict | tuple[bool, float]:
        """Detect watermark in the text."""

        encoded_text = self.config.generation_tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=False
        ).numpy()[0]

        test_result, _ = self.utils.one_run(encoded_text, self.utils.xi.numpy())

        p_val = 0

        for i in range(self.config.n_runs):
            xi_alternative = np.random.rand(
                self.config.pseudo_length, self.config.vocab_size
            ).astype(np.float32)
            null_result, _ = self.utils.one_run(encoded_text, xi_alternative)

            # assuming lower test values indicate presence of watermark
            p_val += null_result <= test_result
            print(f"round: {i + 1}, good: {null_result > test_result}")

        p_val = (p_val + 1.0) / (self.config.n_runs + 1.0)

        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = p_val < self.config.p_threshold

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": p_val}
        else:
            return (is_watermarked, p_val)

    def get_data_for_visualization(
        self, text: str, *args, **kwargs
    ) -> DataForVisualization:
        """Get data for visualization."""

        # Encode text
        encoded_text = self.config.generation_tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=False
        ).numpy()[0]

        # Find best match
        _, index = self.utils.one_run(encoded_text, self.utils.xi.numpy())
        random_numbers = self.utils.xi[
            (index + np.arange(len(encoded_text))) % len(self.utils.xi)
        ]

        highlight_values = []

        # Compute highlight values
        for i in range(0, len(encoded_text)):
            r = random_numbers[i][encoded_text[i]]
            v = log(1 / (1 - r))
            v = self.utils.value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)

        return DataForVisualization(decoded_tokens, highlight_values)
