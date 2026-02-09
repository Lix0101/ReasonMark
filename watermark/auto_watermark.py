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

# =========================================================================
# AutoWatermark.py
# Description: This is a generic watermark class that will be instantiated
#              as one of the watermark classes of the library when created
#              with the [`AutoWatermark.load`] class method.
# =========================================================================

import importlib

import torch
from transformers import LogitsProcessor

from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization
from watermark.auto_config import AutoConfig

WATERMARK_MAPPING_NAMES = {
    "KGW": "watermark.kgw.KGW",
    "Unigram": "watermark.unigram.Unigram",
    "SWEET": "watermark.sweet.SWEET",
    "UPV": "watermark.upv.UPV",
    "SIR": "watermark.sir.SIR",
    "XSIR": "watermark.xsir.XSIR",
    "Unbiased": "watermark.unbiased.UnbiasedWatermark",
    "DIP": "watermark.dip.DIP",
    "EWD": "watermark.ewd.EWD",
    "EXP": "watermark.exp.EXP",
    "EXPGumbel": "watermark.exp_gumbel.EXPGumbel",
    "EXPEdit": "watermark.exp_edit.EXPEdit",
    "ITSEdit": "watermark.its_edit.ITSEdit",
    "SynthID": "watermark.synthid.SynthID",
    "TS": "watermark.ts.TS",
    "PF": "watermark.pf.PF",
    "WatME": "watermark.watme.WatME",
    "OURS": "watermark.ours.OURS",
    'MorphMark':'watermark.morphmark.MorphMark',
    "OURS_NoCPS": "watermark.ours.ours_no_cps.OURS",
    "OURS_NoGCC": "watermark.ours.ours_no_gcc.OURS",
    "OURS_RandomCT": "watermark.ours.ours_no_semantic_guidance.OURS",
    "OURS_NoPhaseSplit": "watermark.ours.ours_phase_separation.OURS",
    "OURS_Decrease": "watermark.ours.ours_decrease.OURS",
}


def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    if name in WATERMARK_MAPPING_NAMES:
        return WATERMARK_MAPPING_NAMES[name]
    else:
        raise ValueError(f"Invalid algorithm name: {name}")


class AutoWatermark:
    """
    This is a generic watermark class that will be instantiated as one of the watermark classes of the library when
    created with the [`AutoWatermark.load`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)` method."
        )

    @staticmethod
    def load(
        algorithm_name,
        algorithm_config=None,
        transformers_config=None,
        *args,
        **kwargs,
    ):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)
        watermark_config = AutoConfig.load(
            algorithm_name,
            transformers_config,
            algorithm_config_path=algorithm_config,
            **kwargs,
        )
        watermark_instance = watermark_class(watermark_config)
        return watermark_instance


vllm_supported_methods = ["UPV", "KGW", "Unigram"]


class AutoWatermarkForVLLM:
    def __init__(self, algorithm_name, algorithm_config, transformers_config):
        if not algorithm_name in vllm_supported_methods:
            raise NotImplementedError(
                f"vllm integrating currently supports {vllm_supported_methods}, but got {algorithm_name}"
            )
        self.watermark = AutoWatermark.load(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            transformers_config=transformers_config,
        )

    def __call__(
        self,
        prompt_tokens: list[int],
        generated_tokens: list[int],
        scores: torch.FloatTensor,
    ) -> torch.Tensor:
        if len(prompt_tokens) == 0:
            return scores

        # concencate prompt_tokens and generated_tokens
        input_ids = torch.LongTensor(prompt_tokens + generated_tokens).to(
            self.watermark.config.device
        )[None, :]
        scores = scores[None, :]
        assert len(input_ids.shape) == 2, input_ids.shape
        assert len(scores.shape) == 2, scores.shape

        scores = self.watermark.logits_processor(input_ids, scores)
        return scores[0, :]

    def get_data_for_visualization(self, text):
        data = self.watermark.get_data_for_visualization(text)
        return data

    def detect_watermark(self, text):
        if type(text) is list:
            return [self.watermark.detect_watermark(_) for _ in text]
        return self.watermark.detect_watermark(text)


class AutoWatermarkForRLLM:
    """
    该类为专门支持推理型 LLM 模型（如 deepseek-ai/DeepSeek-R1-Distill 和 Qwen/QwQ）的水印处理。
    推理型模型生成回复时，可能首先输出思考部分，通常由 <think></think> 或类似标签包裹，
    而水印算法仅需要作用于正式回答（即思考部分之后），但依然以输入 prompt 作为上下文。
    """

    def __init__(
        self,
        algorithm_name: str,
        transformers_config: TransformersConfig,
        algorithm_config: str | None = None,
        watermark_before_think: bool = False,
    ) -> None:
        # 检查当前算法是否支持 vllm（supported methods列表中的方法均支持）
        # if algorithm_name not in vllm_supported_methods:
        #     raise NotImplementedError(
        #         f"vllm integrating currently supports {vllm_supported_methods}, but got {algorithm_name}"
        #     )
        # 加载原有水印实例（内部会调用 AutoConfig.load 获取算法相关配置）
        self.watermark = AutoWatermark.load(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            transformers_config=transformers_config,
        )
        self.watermark_before_think = watermark_before_think
        if not watermark_before_think:
            # 获取生成器所使用的 tokenizer，用于将 "</think>" 编码成 token id
            self.tokenizer = self.watermark.config.generation_tokenizer
            # 这里假定 "</think>" 的 encoding 只产生一个 token
            self.think_end_token_id: int = self.tokenizer.encode(
                "</think>", add_special_tokens=False
            )[0]

    def __call__(
        self,
        prompt_tokens: list[int],
        generated_tokens: list[int],
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        用于 vllm 推理时每步对 logits（scores）的水印处理。

        参数说明：
            prompt_tokens: 输入 prompt 的 token id 列表
            generated_tokens: 模型生成的 token id 列表，其中可能包含思考部分和正式回答
            scores: 对生成 token 的打分，张量形状通常是一维

        处理流程：
            1. 查找 generated_tokens 中结束思考标签对应的 token id 位置。
            2. 取思考结束标签之后的 token 作为正式回答部分；若未发现，则将整个生成文本视为正式回答。
            3. 拼接 prompt_tokens 与正式回答部分，构造输入序列供水印处理器使用。
        """
        think_end_index = -1
        if not self.watermark_before_think:
            try:
                # 查找 "</think>" 对应的 token 在 generated_tokens 列表中的索引
                think_end_index = generated_tokens.index(
                    self.think_end_token_id
                )
            except ValueError:
                # 若没有发现 "</think>"，说明当前回复不符合预期；直接返回原始 scores
                return scores

        # 取 think 部分之后的正式回答 tokens
        answer_tokens = generated_tokens[think_end_index + 1 :]
        # if not answer_tokens:
        #     # 如果正式回答为空，则直接返回原 scores
        #     return scores

        # 构造用于水印处理的输入序列
        if self.watermark_before_think:
            # OURS 等需要思考阶段：使用完整序列（prompt + 已生成）
            seq_tokens = prompt_tokens + generated_tokens
        else:
            # KGW 等仅对回答阶段生效：只使用回答部分（不包含 prompt）
            if len(answer_tokens) == 0:
                # 如果正式回答为空，则直接返回原 scores（等待回答开始后再施加水印）
                return scores
            seq_tokens = answer_tokens

        # input_ids = torch.tensor(
        #     [seq_tokens],
        #     dtype=torch.long,
        #     device=self.watermark.config.device,
        # )

        # scores = scores.unsqueeze(0)
        # assert len(input_ids.shape) == 2, input_ids.shape
        # assert len(scores.shape) == 2, scores.shape

        # scores = self.watermark.logits_processor(input_ids, scores)
        # return scores[0]
        input_ids = torch.tensor(
            [seq_tokens],
            dtype=torch.long,
            device=self.watermark.config.device,
        )
        original_device = scores.device
        scores = scores.to(self.watermark.config.device).unsqueeze(0)
        assert len(input_ids.shape) == 2, input_ids.shape
        assert len(scores.shape) == 2, scores.shape

        scores = self.watermark.logits_processor(input_ids, scores)
        return scores[0].to(original_device)

    def get_data_for_visualization(self, text: str) -> DataForVisualization:
        """
        对于 DeepSeek 模式，先提取正式回答部分（即 "</think>" 标签后的文本）
        然后调用原有 watermark 接口生成用于可视化的数据。
        """
        return self.watermark.get_data_for_visualization(text)

    def detect_watermark(self, text):
        if isinstance(text, list):
            return [self.watermark.detect_watermark(_) for _ in text]
        return self.watermark.detect_watermark(text)


class AutoWatermarkForRLLMHF(LogitsProcessor):
    """
    该类为支持 transformers 推理的 LLM 模型（如 deepseek-ai/DeepSeek-R1-Distill 和 Qwen/QwQ）的水印处理。
    继承自 LogitsProcessor，适配 transformers 的 generate 方法。
    推理型模型生成回复时，可能首先输出思考部分，通常由 <think></think> 或类似标签包裹，
    而水印算法仅需要作用于正式回答（即思考部分之后）。
    """

    def __init__(
        self,
        algorithm_name: str,
        transformers_config: TransformersConfig,
        algorithm_config: str | None = None,
        watermark_before_think: bool = False,
    ) -> None:
        # 检查当前算法是否支持（supported methods列表中的方法均支持）
        # if algorithm_name not in vllm_supported_methods:
        #     raise NotImplementedError(
        #         f"transformers integrating currently supports {vllm_supported_methods}, but got {algorithm_name}"
        #     )
        # 加载原有水印实例
        self.watermark = AutoWatermark.load(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            transformers_config=transformers_config,
        )
        # 获取生成器所使用的 tokenizer，用于将 "</think>" 编码成 token id
        self.tokenizer = self.watermark.config.generation_tokenizer
        # 这里假定 "</think>" 的 encoding 只产生一个 token
        self.think_end_token_id: int = self.tokenizer.encode(
            "</think>", add_special_tokens=False
        )[0]
        self.watermark_before_think = watermark_before_think

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        用于 transformers 推理时每步对 logits 的水印处理。

        参数说明：
            input_ids: 当前生成过程中的完整输入序列，包含原始 prompt 和已生成的 tokens
            scores: 当前步骤的 logits 分数

        处理流程：
            1. 检测是否已找到 </think> 标记
            2. 若已找到，仅使用回答部分计算水印
            3. 若未找到，检查当前序列中是否包含 </think>
        """
        if self.watermark_before_think:
            # Run logits processing on the watermark device, then move back
            original_device = scores.device
            input_ids = input_ids.to(self.watermark.config.device)
            scores = scores.to(self.watermark.config.device)
            out = self.watermark.logits_processor(input_ids, scores)
            return out.to(original_device)
        # 只处理第一个序列（batch 中的第一个）
        input_ids_seq: torch.Tensor = input_ids[0]
        think_end_indices: torch.Tensor = (
            input_ids_seq == self.think_end_token_id
        ).nonzero(as_tuple=True)[0]
        # 如果当前序列中没有 </think> 标记，直接返回原始分数
        if len(think_end_indices) > 0:
            answer_start_idx: int = think_end_indices[-1].item() + 1  # type: ignore
            input_ids = input_ids_seq[answer_start_idx:].unsqueeze(0)
            # Run logits processing on the watermark device, then move back
            original_device = scores.device
            input_ids = input_ids.to(self.watermark.config.device)
            scores = scores.to(self.watermark.config.device)
            out = self.watermark.logits_processor(input_ids, scores)
            return out.to(original_device)
        return scores

    def get_data_for_visualization(self, text: str) -> DataForVisualization:
        return self.watermark.get_data_for_visualization(text)

    def detect_watermark(self, text):
        if isinstance(text, list):
            return [self.watermark.detect_watermark(_) for _ in text]
        return self.watermark.detect_watermark(text)
