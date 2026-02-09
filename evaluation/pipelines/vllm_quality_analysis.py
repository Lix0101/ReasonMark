from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar, cast

from tqdm import tqdm

from evaluation.dataset import BaseDataset
from evaluation.pipelines.quality_analysis import (
    QualityPipelineReturnType,
    TextQualityComparisonResult,
)
from evaluation.tools.text_editor import TextEditor
from evaluation.tools.text_quality_analyzer import (
    DirectTextQualityAnalyzer,
    ExternalDiscriminatorTextQualityAnalyzer,
    ReferencedTextQualityAnalyzer,
    TextQualityAnalyzer,
)
from exceptions.exceptions import (
    InvalidDirectAnalyzerTypeError,
    InvalidReferencedAnalyzerTypeError,
    InvalidTextSourceModeError,
)

# 定义通用返回类型
ResultT = TypeVar(
    "ResultT",
    list[TextQualityComparisonResult],
    list[dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
)


class TextQualityAnalysisVLLMPipeline(ABC):
    """使用 VLLM 进行文本质量分析的流水线。"""

    def __init__(
        self,
        dataset: BaseDataset,
        watermarked_text_editor_list: list[TextEditor] | None = None,
        unwatermarked_text_editor_list: list[TextEditor] | None = None,
        analyzers: list[TextQualityAnalyzer] | None = None,
        unwatermarked_text_source: Literal[
            "natural", "generated"
        ] = "generated",
        show_progress: bool = True,
        return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES,
    ) -> None:
        """
        初始化 VLLM 文本质量分析流水线。

        参数:
            dataset (BaseDataset): 用于评估的数据集。
            watermarked_text_editor_list (list[TextEditor]): 水印文本编辑器列表。
            unwatermarked_text_editor_list (list[TextEditor]): 无水印文本编辑器列表。
            analyzers (list[TextQualityAnalyzer]): 文本质量分析器列表。
            unwatermarked_text_source (Literal["natural", "generated"]): 无水印文本的来源，'natural' 或 'generated'。
            show_progress (bool): 是否显示进度条。
            return_type (QualityPipelineReturnType): 流水线返回类型。
        """
        if unwatermarked_text_source not in ["natural", "generated"]:
            raise InvalidTextSourceModeError(unwatermarked_text_source)

        self.dataset = dataset
        self.watermarked_text_editor_list = (
            watermarked_text_editor_list
            if watermarked_text_editor_list is not None
            else []
        )
        self.unwatermarked_text_editor_list = (
            unwatermarked_text_editor_list
            if unwatermarked_text_editor_list is not None
            else []
        )
        self.analyzers = analyzers if analyzers is not None else []
        self.unwatermarked_text_source = unwatermarked_text_source
        self.show_progress = show_progress
        self.return_type = return_type

    @abstractmethod
    def _get_iterable(self) -> Any:
        """返回数据集的迭代器。"""
        raise NotImplementedError("子类必须实现此方法")

    def _get_progress_bar(self, iterable: Any) -> Any:
        """返回可能带有进度条的迭代器。"""
        if self.show_progress:
            return tqdm(iterable, desc="文本质量分析处理中", leave=True)
        return iterable

    def _edit_watermarked_text(
        self, text: str, prompt: str | None = None
    ) -> str:
        """使用文本编辑器编辑水印文本。"""
        for text_editor in self.watermarked_text_editor_list:
            text = text_editor.edit(text, prompt)
        return text

    def _edit_unwatermarked_text(
        self, text: str, prompt: str | None = None
    ) -> str:
        """使用文本编辑器编辑无水印文本。"""
        for text_editor in self.unwatermarked_text_editor_list:
            text = text_editor.edit(text, prompt)
        return text

    @abstractmethod
    def _prepare_input_for_quality_analyzer(
        self, watermarked_text: str, unwatermarked_text: str, index: int
    ) -> Any:
        """为质量分析器准备输入。"""
        raise NotImplementedError("子类必须实现此方法")

    @abstractmethod
    def analyze_quality(
        self, prepared_data: Any, analyzer: TextQualityAnalyzer
    ) -> tuple[float, float]:
        """分析水印和无水印文本的质量。"""
        raise NotImplementedError("子类必须实现此方法")

    def evaluate(
        self, watermarked_texts: list[str], unwatermarked_texts: list[str]
    ) -> ResultT:
        """
        使用流水线进行评估。

        参数:
            watermarked_texts (list[str]): 水印文本列表。
            unwatermarked_texts (list[str]): 无水印文本列表。

        返回:
            根据 return_type 返回不同格式的评估结果。
        """
        evaluation_result = []
        bar = self._get_progress_bar(self._get_iterable())

        for index in bar:
            # 获取水印和无水印文本
            watermarked_text = (
                watermarked_texts[index]
                if index < len(watermarked_texts)
                else ""
            )
            unwatermarked_text = (
                unwatermarked_texts[index]
                if index < len(unwatermarked_texts)
                else ""
            )

            # 编辑水印和无水印文本
            edited_watermarked_text = self._edit_watermarked_text(
                watermarked_text, self.dataset.get_prompt(index)
            )
            edited_unwatermarked_text = self._edit_unwatermarked_text(
                unwatermarked_text, self.dataset.get_prompt(index)
            )

            # 初始化分数字典
            watermarked_scores: dict[str, float] = {}
            unwatermarked_scores: dict[str, float] = {}

            # 使用每个分析器分析质量
            for analyzer in self.analyzers:
                prepared_data = self._prepare_input_for_quality_analyzer(
                    edited_watermarked_text, edited_unwatermarked_text, index
                )
                w_score, u_score = self.analyze_quality(prepared_data, analyzer)
                analyzer_name = analyzer.__class__.__name__
                watermarked_scores[analyzer_name] = w_score
                unwatermarked_scores[analyzer_name] = u_score

            # 添加结果
            evaluation_result.append(
                TextQualityComparisonResult(
                    edited_watermarked_text,
                    edited_unwatermarked_text,
                    watermarked_scores,
                    unwatermarked_scores,
                )
            )

        # 根据返回类型返回结果
        if self.return_type == QualityPipelineReturnType.FULL:
            return evaluation_result  # type: ignore
        elif self.return_type == QualityPipelineReturnType.SCORES:
            return [
                {
                    "watermarked": result.watermarked_quality_scores,
                    "unwatermarked": result.unwatermarked_quality_scores,
                }
                for result in evaluation_result
            ]  # type: ignore
        elif self.return_type == QualityPipelineReturnType.MEAN_SCORES:
            # 计算每个分析器的平均分数
            mean_watermarked: dict[str, float] = {}
            mean_unwatermarked: dict[str, float] = {}

            if evaluation_result:
                analyzer_names = evaluation_result[
                    0
                ].watermarked_quality_scores.keys()
                for analyzer_name in analyzer_names:
                    mean_watermarked[analyzer_name] = sum(
                        result.watermarked_quality_scores[analyzer_name]
                        for result in evaluation_result
                    ) / len(evaluation_result)

                    mean_unwatermarked[analyzer_name] = sum(
                        result.unwatermarked_quality_scores[analyzer_name]
                        for result in evaluation_result
                    ) / len(evaluation_result)

            return {
                "watermarked": mean_watermarked,
                "unwatermarked": mean_unwatermarked,
            }  # type: ignore

        # 不应该达到这里，但为了类型检查完整性
        raise ValueError(f"未知的返回类型: {self.return_type}")


class DirectTextQualityAnalysisVLLMPipeline(TextQualityAnalysisVLLMPipeline):
    """
    直接 VLLM 文本质量分析流水线。

    该类通过直接比较水印文本与无水印文本的特征来分析文本质量。
    它评估诸如困惑度（PPL）和对数多样性等指标，无需任何外部参考文本。

    使用此流水线直接评估水印对文本质量的影响。
    """

    def __init__(
        self,
        dataset: BaseDataset,
        watermarked_text_editor_list: list[TextEditor] | None = None,
        unwatermarked_text_editor_list: list[TextEditor] | None = None,
        analyzers: list[DirectTextQualityAnalyzer] | None = None,
        unwatermarked_text_source: Literal[
            "natural", "generated"
        ] = "generated",
        show_progress: bool = True,
        return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES,
    ) -> None:

        # 验证分析器
        if analyzers is not None and not all(
            isinstance(analyzer, DirectTextQualityAnalyzer)
            for analyzer in analyzers
        ):
            raise InvalidDirectAnalyzerTypeError()

        super().__init__(
            dataset,
            watermarked_text_editor_list,
            unwatermarked_text_editor_list,
            cast(list[TextQualityAnalyzer] | None, analyzers),
            unwatermarked_text_source,
            show_progress,
            return_type,
        )

    def _get_iterable(self) -> range:
        """返回数据集的迭代器。"""
        return range(self.dataset.prompt_nums)

    def _prepare_input_for_quality_analyzer(
        self, watermarked_text: str, unwatermarked_text: str, index: int
    ) -> tuple[str, str, str]:
        """准备质量分析器的输入数据

        Args:
            watermarked_text: 有水印文本
            unwatermarked_text: 无水印文本
            index: 数据索引

        Returns:
            准备好的输入数据：有水印文本、无水印文本、提示（可用作前缀）
        """
        prompt = (
            self.dataset.get_prompt(index)
            if index < self.dataset.prompt_nums
            else ""
        )
        return watermarked_text, unwatermarked_text, prompt

    def analyze_quality(
        self,
        prepared_data: tuple[str, str, str],
        analyzer: DirectTextQualityAnalyzer,
    ) -> tuple[float, float]:
        """Analyze the quality of the texts.

        Args:
            prepared_data: 准备好的数据（有水印文本、无水印文本、提示）
            analyzer: 质量分析器

        Returns:
            有水印和无水印文本的分析结果
        """
        watermarked_text, unwatermarked_text, prompt = prepared_data

        # 如果分析器是PrefixPPLCalculator，则传入prefix参数
        if analyzer.__class__.__name__ == "PrefixPPLCalculator":
            watermarked_score = analyzer.analyze(
                watermarked_text, prefix=prompt
            )
            unwatermarked_score = analyzer.analyze(
                unwatermarked_text, prefix=prompt
            )
        else:
            watermarked_score = analyzer.analyze(watermarked_text)
            unwatermarked_score = analyzer.analyze(unwatermarked_text)

        # 确保返回值为浮点数
        watermarked_result = (
            0.0 if watermarked_score is None else float(watermarked_score)
        )
        unwatermarked_result = (
            0.0 if unwatermarked_score is None else float(unwatermarked_score)
        )

        return watermarked_result, unwatermarked_result


class ReferencedTextQualityAnalysisVLLMPipeline(
    TextQualityAnalysisVLLMPipeline
):
    """
    参考 VLLM 文本质量分析流水线。

    该流水线通过将水印和无水印文本与共同参考文本进行比较来评估文本质量。
    它测量与参考文本的相似度或偏差程度。

    适用于需要评估水印对文本质量影响的场景，特别是与特定下游任务相关的情况。
    """

    def __init__(
        self,
        dataset: BaseDataset,
        watermarked_text_editor_list: list[TextEditor] | None = None,
        unwatermarked_text_editor_list: list[TextEditor] | None = None,
        analyzers: list[ReferencedTextQualityAnalyzer] | None = None,
        unwatermarked_text_source: Literal[
            "natural", "generated"
        ] = "generated",
        show_progress: bool = True,
        return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES,
    ) -> None:
        """
        初始化参考 VLLM 文本质量分析流水线。

        参数:
            dataset (BaseDataset): 用于评估的数据集。
            watermarked_text_editor_list (list[TextEditor]): 水印文本编辑器列表。
            unwatermarked_text_editor_list (list[TextEditor]): 无水印文本编辑器列表。
            analyzers (list[TextQualityAnalyzer]): 文本质量分析器列表。
            unwatermarked_text_source (Literal["natural", "generated"]): 无水印文本的来源，'natural' 或 'generated'。
            show_progress (bool): 是否显示进度条。
            return_type (QualityPipelineReturnType): 流水线返回类型。
        """
        # 验证分析器
        if analyzers:
            for analyzer in analyzers:
                if not isinstance(analyzer, ReferencedTextQualityAnalyzer):
                    raise InvalidReferencedAnalyzerTypeError

        super().__init__(
            dataset,
            watermarked_text_editor_list,
            unwatermarked_text_editor_list,
            cast(list[TextQualityAnalyzer], analyzers),
            unwatermarked_text_source,
            show_progress,
            return_type,
        )

    def _get_iterable(self) -> range:
        """返回数据集的迭代器。"""
        return range(self.dataset.prompt_nums)

    def _prepare_input_for_quality_analyzer(
        self, watermarked_text: str, unwatermarked_text: str, index: int
    ) -> tuple[str, str, str]:
        """为质量分析器准备输入。"""
        return (
            watermarked_text,
            unwatermarked_text,
            self.dataset.get_reference(index),
        )

    def analyze_quality(
        self,
        prepared_data: tuple[str, str, str],
        analyzer: ReferencedTextQualityAnalyzer,
    ) -> tuple[float, float]:
        """分析水印和无水印文本的质量。"""
        watermarked_text = prepared_data[0]
        unwatermarked_text = prepared_data[1]
        reference = prepared_data[2]
        # 处理可能为 None 的返回值
        w_result = analyzer.analyze(watermarked_text, reference)
        u_result = analyzer.analyze(unwatermarked_text, reference)
        # 确保返回类型为 tuple[float, float]
        w_score = 0.0 if w_result is None else float(w_result)
        u_score = 0.0 if u_result is None else float(u_result)
        return w_score, u_score


class ExternalDiscriminatorTextQualityAnalysisVLLMPipeline(
    TextQualityAnalysisVLLMPipeline
):
    """
    外部判别器 VLLM 文本质量分析流水线。

    该类利用外部判别器（如 GPT-4）比较水印和无水印文本的质量。
    判别器根据用户提供的任务描述评估文本质量，指示由于水印导致的质量降低或保留。

    当需要对水印的微妙影响进行基于 AI 的高级意见时，此分析器特别有用。
    """

    def __init__(
        self,
        dataset: BaseDataset,
        watermarked_text_editor_list: list[TextEditor] | None = None,
        unwatermarked_text_editor_list: list[TextEditor] | None = None,
        analyzers: list[ExternalDiscriminatorTextQualityAnalyzer] | None = None,
        unwatermarked_text_source: Literal[
            "natural", "generated"
        ] = "generated",
        show_progress: bool = True,
        return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES,
    ) -> None:
        """
        初始化外部判别器 VLLM 文本质量分析流水线。

        参数:
            dataset (BaseDataset): 用于评估的数据集。
            watermarked_text_editor_list (list[TextEditor]): 水印文本编辑器列表。
            unwatermarked_text_editor_list (list[TextEditor]): 无水印文本编辑器列表。
            analyzers (list[TextQualityAnalyzer]): 文本质量分析器列表。
            unwatermarked_text_source (Literal["natural", "generated"]): 无水印文本的来源，'natural' 或 'generated'。
            show_progress (bool): 是否显示进度条。
            return_type (QualityPipelineReturnType): 流水线返回类型。
        """
        # 验证分析器
        if analyzers:
            for analyzer in analyzers:
                if not isinstance(
                    analyzer, ExternalDiscriminatorTextQualityAnalyzer
                ):
                    raise InvalidReferencedAnalyzerTypeError

        super().__init__(
            dataset,
            watermarked_text_editor_list,
            unwatermarked_text_editor_list,
            cast(list[TextQualityAnalyzer], analyzers),
            unwatermarked_text_source,
            show_progress,
            return_type,
        )

    def _get_iterable(self) -> range:
        """返回数据集的迭代器。"""
        return range(self.dataset.prompt_nums)

    def _prepare_input_for_quality_analyzer(
        self, watermarked_text: str, unwatermarked_text: str, index: int
    ) -> tuple[str, str, str]:
        """为质量分析器准备输入，返回水印文本、无水印文本和当前提示。"""
        prompt = self.dataset.get_prompt(index)
        return watermarked_text, unwatermarked_text, prompt

    def analyze_quality(
        self,
        prepared_data: tuple[str, str, str],
        analyzer: ExternalDiscriminatorTextQualityAnalyzer,
    ) -> tuple[float, float]:
        """分析水印和无水印文本的质量。"""
        watermarked_text = prepared_data[0]
        unwatermarked_text = prepared_data[1]
        prompt = prepared_data[2]  # 获取提示作为问题

        result = analyzer.analyze(watermarked_text, unwatermarked_text, prompt)

        # 处理可能为 None 的返回值
        score = 0.0 if result is None else float(result)
        return score, score
