from typing import Any

from tqdm import tqdm

from evaluation.dataset import BaseDataset
from evaluation.pipelines.detection import (
    DetectionPipelineReturnType,
    WatermarkDetectionResult,
)
from evaluation.tools.text_editor import TextEditor
from watermark.base import BaseWatermark


class WatermarkDetectionVLLMPipeline:
    """Base Pipeline for watermark detection with VLLM."""

    def __init__(
        self,
        dataset: BaseDataset,
        text_editor_list: list[TextEditor] | None = None,
        show_progress: bool = True,
        return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES,
    ) -> None:
        """
        Initialize the watermark detection pipeline for VLLM.

        Parameters:
            dataset (BaseDataset): The dataset for the pipeline
            text_editor_list (list[TextEditor]): The list of text editors
            show_progress (bool): Whether to show progress bar
            return_type (DetectionPipelineReturnType): The return type of the pipeline
        """
        self.dataset = dataset
        self.text_editor_list = (
            text_editor_list if text_editor_list is not None else []
        )
        self.show_progress = show_progress
        self.return_type = return_type

    def _edit_text(self, text: str, prompt: str | None = None) -> str:
        """Edit text using text editors."""
        for text_editor in self.text_editor_list:
            text = text_editor.edit(text, prompt)
        return text

    def _truncate_or_filter_text(
        self,
        text: str,
        tokenizer,
        min_answer_tokens: int | None = None,
        max_answer_tokens: int | None = None,
    ) -> str | None:
        """
        过滤或截断文本基于token数量

        Parameters:
            text (str): 待处理的文本
            tokenizer: 分词器，用于计算token数量
            min_answer_tokens (int | None): 最小token数，小于此值将被过滤
            max_answer_tokens (int | None): 最大token数，大于此值将被截断

        Returns:
            str | None: 处理后的文本，或者None表示文本被过滤
        """
        if not text:
            return None

        # 计算token数量
        tokens = tokenizer.encode(text)
        token_count = len(tokens)

        # 过滤过短的文本
        if min_answer_tokens is not None and token_count < min_answer_tokens:
            print(
                f"文本被过滤: token数量({token_count})小于最小要求({min_answer_tokens})"
            )
            return None

        # 截断过长的文本
        if max_answer_tokens is not None and token_count > max_answer_tokens:
            print(
                f"文本被截断: 从{token_count}个token截断到{max_answer_tokens}个token"
            )
            truncated_tokens = tokens[:max_answer_tokens]
            return tokenizer.decode(truncated_tokens)

        return text

    def _detect_watermark(
        self, text: str, watermark: BaseWatermark
    ) -> dict[str, Any]:
        """Detect watermark in text."""
        detect_result = watermark.detect_watermark(text)
        return (
            dict(detect_result)
            if not isinstance(detect_result, dict)
            else detect_result
        )

    def _get_iterable(self) -> range:
        """返回数据集中所有 prompt 的索引范围。"""
        return range(self.dataset.prompt_nums)

    def _get_progress_bar(self, iterable) -> Any:
        """Return an iterable possibly wrapped with a progress bar."""
        if self.show_progress:
            return tqdm(iterable, desc="Processing", leave=True)
        return iterable

    def evaluate(
        self,
        watermark: BaseWatermark,
        texts: list[str],
        min_answer_tokens: int | None = None,
        max_answer_tokens: int | None = None,
    ) -> list[dict[str, Any] | float | bool]:
        """
        Conduct evaluation utilizing the pipeline.

        Parameters:
            watermark: The watermark object
            texts (list[str]): List of texts to evaluate
            min_answer_tokens (int | None): 最小token数量，小于此值的文本将被过滤
            max_answer_tokens (int | None): 最大token数量，大于此值的文本将被截断

        Returns:
            List of evaluation results in the specified return type format
        """
        evaluation_result = []
        bar = self._get_progress_bar(self._get_iterable())

        # 获取 tokenizer
        assert hasattr(
            watermark.config, "generation_tokenizer"
        ), "Watermark object must have generation_tokenizer attribute in config."
        tokenizer = watermark.config.generation_tokenizer

        for index in bar:
            if index >= len(texts):
                continue

            text = texts[index]
            prompt = self.dataset.get_prompt(index)
            edited_text = self._edit_text(text, prompt)

            # 过滤或截断文本
            processed_text = edited_text
            if min_answer_tokens is not None or max_answer_tokens is not None:
                processed_text = self._truncate_or_filter_text(
                    edited_text, tokenizer, min_answer_tokens, max_answer_tokens
                )

            # 如果文本被过滤，跳过检测
            if processed_text is None:
                continue

            detect_result = self._detect_watermark(processed_text, watermark)
            evaluation_result.append(
                WatermarkDetectionResult(text, processed_text, detect_result)
            )

        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_result
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [
                result.detect_result["score"] for result in evaluation_result
            ]
        elif self.return_type == DetectionPipelineReturnType.IS_WATERMARKED:
            return [
                result.detect_result["is_watermarked"]
                for result in evaluation_result
            ]
        return []
