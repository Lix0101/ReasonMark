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

# ============================================
# transformers_config.py
# Description: Configuration for transformers
# ============================================

from transformers import (
    LogitsProcessorList,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class TransformersConfig:
    """Configuration class for transformers."""

    def __init__(
        self, model, tokenizer, vocab_size=None, device="cuda", *args, **kwargs
    ):
        """
        Initialize the transformers configuration.

        Parameters:
            model (object): The model object.
            tokenizer (object): The tokenizer object.
            vocab_size (int): The vocabulary size.
            device (str): The device to use.
            kwargs: Additional keyword arguments.
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)
        self.logits_processor_list = self._create_logits_processors(kwargs)

    def _create_logits_processors(self, kwargs) -> LogitsProcessorList:
        """
        基于配置参数创建 LogitsProcessor 列表

        Args:
            kwargs: 配置参数字典

        Returns:
            LogitsProcessorList: 处理器列表
        """
        processors = []
        # Repetition penalty
        if (
            "repetition_penalty" in kwargs
            and kwargs["repetition_penalty"]
            and kwargs["repetition_penalty"] != 1.0
        ):
            processors.append(
                RepetitionPenaltyLogitsProcessor(
                    penalty=kwargs["repetition_penalty"]
                )
            )

        # No repeat n-gram
        if (
            "no_repeat_ngram_size" in kwargs
            and kwargs["no_repeat_ngram_size"]
            and kwargs["no_repeat_ngram_size"] > 0
        ):
            processors.append(
                NoRepeatNGramLogitsProcessor(
                    ngram_size=kwargs["no_repeat_ngram_size"]
                )
            )

        # Min new tokens
        if (
            "min_new_tokens" in kwargs
            and kwargs["min_new_tokens"]
            and kwargs["min_new_tokens"] > 0
        ):
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is None:
                eos_id = getattr(self.tokenizer, "pad_token_id", 0)

            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    min_new_tokens=kwargs["min_new_tokens"],
                    eos_token_id=eos_id,
                    prompt_length_to_skip=0,  # 默认不跳过任何 prompt tokens
                    device=self.device,
                )
            )

        # Temperature
        if (
            "temperature" in kwargs
            and kwargs["temperature"]
            and kwargs["temperature"] != 1.0
        ):
            processors.append(
                TemperatureLogitsWarper(temperature=kwargs["temperature"])
            )

        # Top-k
        if "top_k" in kwargs and kwargs["top_k"] and kwargs["top_k"] != 0:
            processors.append(TopKLogitsWarper(top_k=kwargs["top_k"]))

        # Top-p (nucleus sampling)
        if "top_p" in kwargs and kwargs["top_p"] and kwargs["top_p"] < 1.0:
            processors.append(TopPLogitsWarper(top_p=kwargs["top_p"]))

        # Min-p
        if "min_p" in kwargs and kwargs["min_p"] and kwargs["min_p"] > 0.0:
            processors.append(MinPLogitsWarper(min_p=kwargs["min_p"]))

        return LogitsProcessorList(processors)
