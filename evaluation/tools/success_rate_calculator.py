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

# ==========================================================
# success_rate_calculator.py
# Description: Calculate success rate of watermark detection
# ==========================================================

from typing import List, Dict, Union
from exceptions.exceptions import TypeMismatchException, ConfigurationError
import numpy as np
from sklearn.metrics import roc_curve, auc


class DetectionResult:
    """Detection result."""

    def __init__(
        self, gold_label: bool, detect_result: Union[bool, float]
    ) -> None:
        """
        Initialize the detection result.

        Parameters:
            gold_label (bool): The expected watermark presence.
            detect_result (Union[bool, float]): The detection result.
        """
        self.gold_label = gold_label
        self.detect_result = detect_result


class BaseSuccessRateCalculator:
    """Base class for success rate calculator."""

    def __init__(
        self,
        labels: List[str] = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
    ) -> None:
        """
        Initialize the success rate calculator.

        Parameters:
            labels (List[str]): The list of metric labels to include in the output.
        """
        self.labels = labels

    def _check_instance(
        self, data: List[Union[bool, float]], expected_type: type
    ):
        """Check if the data is an instance of the expected type."""
        for d in data:
            if not isinstance(d, expected_type):
                raise TypeMismatchException(expected_type, type(d))

    def _filter_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Filter metrics based on the provided labels."""
        return {
            label: metrics[label] for label in self.labels if label in metrics
        }

    def calculate(
        self,
        watermarked_result: List[Union[bool, float]],
        non_watermarked_result: List[Union[bool, float]],
    ) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        pass


class FundamentalSuccessRateCalculator(BaseSuccessRateCalculator):
    """
    Calculator for fundamental success rates of watermark detection.

    This class specifically handles the calculation of success rates for scenarios involving
    watermark detection after fixed thresholding. It provides metrics based on comparisons
    between expected watermarked results and actual detection outputs.

    Use this class when you need to evaluate the effectiveness of watermark detection algorithms
    under fixed thresholding conditions.
    """

    def __init__(
        self,
        labels: List[str] = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
    ) -> None:
        """
        Initialize the fundamental success rate calculator.

        Parameters:
            labels (List[str]): The list of metric labels to include in the output.
        """
        super().__init__(labels)

    def _compute_metrics(
        self, inputs: List[DetectionResult]
    ) -> Dict[str, float]:
        """Compute metrics based on the provided inputs."""
        TP = sum(1 for d in inputs if d.detect_result and d.gold_label)
        TN = sum(1 for d in inputs if not d.detect_result and not d.gold_label)
        FP = sum(1 for d in inputs if d.detect_result and not d.gold_label)
        FN = sum(1 for d in inputs if not d.detect_result and d.gold_label)

        TPR = TP / (TP + FN) if TP + FN else 0.0
        FPR = FP / (FP + TN) if FP + TN else 0.0
        TNR = TN / (TN + FP) if TN + FP else 0.0
        FNR = FN / (FN + TP) if FN + TP else 0.0
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * (P * R) / (P + R) if P + R else 0.0
        ACC = (TP + TN) / (len(inputs)) if inputs else 0.0

        return {
            "TPR": TPR,
            "TNR": TNR,
            "FPR": FPR,
            "FNR": FNR,
            "P": P,
            "R": R,
            "F1": F1,
            "ACC": ACC,
        }

    def calculate(
        self, watermarked_result: List[bool], non_watermarked_result: List[bool]
    ) -> Dict[str, float]:
        """calculate success rates of watermark detection based on provided results."""
        self._check_instance(watermarked_result, bool)
        self._check_instance(non_watermarked_result, bool)

        inputs = [DetectionResult(True, x) for x in watermarked_result] + [
            DetectionResult(False, x) for x in non_watermarked_result
        ]
        metrics = self._compute_metrics(inputs)
        return self._filter_metrics(metrics)


class DynamicThresholdSuccessRateCalculator(BaseSuccessRateCalculator):
    """
    Calculator for success rates of watermark detection with dynamic thresholding.

    This class calculates success rates for watermark detection scenarios where the detection
    thresholds can dynamically change based on varying conditions. It supports evaluating the
    effectiveness of watermark detection algorithms that adapt to different signal or noise conditions.

    Use this class to evaluate detection systems where the threshold for detecting a watermark
    is not fixed and can vary.
    """

    def __init__(
        self,
        labels: List[str] = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"],
        rule="best",
        target_fpr=None,
        reverse=False,
    ) -> None:
        """
        Initialize the dynamic threshold success rate calculator.

        Parameters:
            labels (List[str]): The list of metric labels to include in the output.
            rule (str): The rule for determining the threshold. Choose from 'best' or 'target_fpr'.
            target_fpr (float): The target false positive rate to achieve.
            reverse (bool): Whether to reverse the sorting order of the detection results.
                            True: higher values are considered positive.
                            False: lower values are considered positive.
        """
        super().__init__(labels)
        self.rule = rule
        self.target_fpr = target_fpr
        self.reverse = reverse

        # Validate rule configuration
        if self.rule not in ["best", "target_fpr"]:
            raise ConfigurationError(
                f"Invalid rule specified: {self.rule}. Choose from 'best' or 'target_fpr'."
            )

        # Validate target_fpr configuration based on the rule
        if self.rule == "target_fpr":
            if self.target_fpr is None:
                raise ConfigurationError(
                    "target_fpr must be set when rule is 'target_fpr'."
                )
            if not isinstance(self.target_fpr, (float, int)) or not (
                0 <= self.target_fpr <= 1
            ):
                raise ConfigurationError(
                    "target_fpr must be a float or int within the range [0, 1]."
                )

    def _find_best_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the best threshold that maximizes F1."""
        best_threshold = 0
        best_metrics = None
        for i in range(len(inputs) - 1):
            threshold = (
                inputs[i].detect_result + inputs[i + 1].detect_result
            ) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if best_metrics is None or metrics["F1"] > best_metrics["F1"]:
                best_threshold = threshold
                best_metrics = metrics
        return best_threshold

    def _find_threshold_by_fpr(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold that achieves the target FPR."""
        threshold = 0
        for i in range(len(inputs) - 1):
            threshold = (
                inputs[i].detect_result + inputs[i + 1].detect_result
            ) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if metrics["FPR"] <= self.target_fpr:
                break
        return threshold

    def _find_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold based on the specified rule."""
        sorted_inputs = sorted(
            inputs, key=lambda x: x.detect_result, reverse=self.reverse
        )

        # If the rule is to find the best threshold by maximizing accuracy
        if self.rule == "best":
            return self._find_best_threshold(sorted_inputs)
        else:
            # If the rule is to find the threshold that achieves the target FPR
            return self._find_threshold_by_fpr(sorted_inputs)

    def _compute_metrics(
        self, inputs: List[DetectionResult], threshold: float
    ) -> Dict[str, float]:
        """Compute metrics based on the provided inputs and threshold."""
        if not self.reverse:
            TP = sum(
                1
                for x in inputs
                if x.detect_result >= threshold and x.gold_label
            )
            FP = sum(
                1
                for x in inputs
                if x.detect_result >= threshold and not x.gold_label
            )
            TN = sum(
                1
                for x in inputs
                if x.detect_result < threshold and not x.gold_label
            )
            FN = sum(
                1
                for x in inputs
                if x.detect_result < threshold and x.gold_label
            )
        else:
            TP = sum(
                1
                for x in inputs
                if x.detect_result <= threshold and x.gold_label
            )
            FP = sum(
                1
                for x in inputs
                if x.detect_result <= threshold and not x.gold_label
            )
            TN = sum(
                1
                for x in inputs
                if x.detect_result > threshold and not x.gold_label
            )
            FN = sum(
                1
                for x in inputs
                if x.detect_result > threshold and x.gold_label
            )

        metrics = {
            "TPR": TP / (TP + FN) if TP + FN else 0,
            "FPR": FP / (FP + TN) if FP + TN else 0,
            "TNR": TN / (TN + FP) if TN + FP else 0,
            "FNR": FN / (FN + TP) if FN + TP else 0,
            "P": TP / (TP + FP) if TP + FP else 0,
            "R": TP / (TP + FN) if TP + FN else 0,
            "F1": 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0,
            "ACC": (TP + TN) / (len(inputs)) if inputs else 0,
        }
        return metrics

    def calculate(
        self,
        watermarked_result: List[float],
        non_watermarked_result: List[float],
    ) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        self._check_instance(watermarked_result + non_watermarked_result, float)

        inputs = [DetectionResult(True, x) for x in watermarked_result] + [
            DetectionResult(False, x) for x in non_watermarked_result
        ]
        threshold = self._find_threshold(inputs)
        metrics = self._compute_metrics(inputs, threshold)
        return self._filter_metrics(metrics)


class ROCSuccessRateCalculator(BaseSuccessRateCalculator):
    """
    使用 ROC 曲线分析计算水印检测评估指标。

    该类先计算 ROC 曲线和 AUROC，然后根据指定规则选择阈值：
    - 'best'：最大化 F1 分数的阈值
    - 'target_fpr'：找到最接近目标 FPR 的阈值

    然后计算在该阈值下的所有评估指标。
    """

    def __init__(
        self,
        labels: list[str] = [
            "TPR",
            "TNR",
            "FPR",
            "FNR",
            "P",
            "R",
            "F1",
            "ACC",
            "AUROC",
        ],
        rule: str = "best",
        target_fpr: float = 0.01,
        reverse: bool = False,
    ) -> None:
        """
        初始化 ROC 成功率计算器。

        参数:
            labels (list[str]): 要包含在输出中的指标标签列表
            rule (str): 阈值确定规则，可选 'best'(最大化F1) 或 'target_fpr'(目标假阳性率)
            target_fpr (float): 目标假阳性率，默认 0.01，仅当 rule='target_fpr' 时使用
            reverse (bool): 是否反转得分排序（True：较低得分视为正例）
        """
        super().__init__(labels)
        self.rule = rule
        self.target_fpr = target_fpr
        self.reverse = reverse

        # 验证规则
        if self.rule not in ["best", "target_fpr"]:
            raise ConfigurationError(
                f"无效的规则：{self.rule}。请从 'best' 或 'target_fpr' 中选择。"
            )

        # 验证 target_fpr
        if self.rule == "target_fpr":
            if self.target_fpr is None:
                raise ConfigurationError(
                    "当规则为 'target_fpr' 时，必须设置 target_fpr。"
                )
            if not isinstance(self.target_fpr, (float, int)) or not (
                0 <= self.target_fpr <= 1
            ):
                raise ConfigurationError(
                    "target_fpr 必须是 [0, 1] 范围内的浮点数"
                )

    def calculate(
        self,
        watermarked_result: list[float],
        non_watermarked_result: list[float],
    ) -> dict[str, float]:
        """
        基于 ROC 曲线计算评估指标。

        参数:
            watermarked_result (list[float]): 含水印文本的检测分数
            non_watermarked_result (list[float]): 无水印文本的检测分数

        返回:
            dict[str, float]: 计算出的各项指标
        """
        # 检查输入
        self._check_instance(watermarked_result + non_watermarked_result, float)

        # 准备数据
        y_true = [1] * len(watermarked_result) + [0] * len(
            non_watermarked_result
        )
        y_scores = watermarked_result + non_watermarked_result

        # 如果需要反转分数
        if self.reverse:
            y_scores = [-score for score in y_scores]

        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # 计算 AUROC
        auroc = auc(fpr, tpr)

        # 根据规则选择阈值
        if self.rule == "target_fpr":
            # 找到最接近目标 FPR 的阈值
            target_idx = np.argmin(np.abs(fpr - self.target_fpr))
        else:  # rule == "best"
            # 计算每个阈值的 F1 分数，选择最大的
            f1_scores = []
            for threshold in thresholds:
                y_pred = [score >= threshold for score in y_scores]
                TP = sum(1 for p, t in zip(y_pred, y_true) if p and t)
                FP = sum(1 for p, t in zip(y_pred, y_true) if p and not t)
                FN = sum(1 for p, t in zip(y_pred, y_true) if not p and t)
                P = TP / (TP + FP) if TP + FP else 0
                R = TP / (TP + FN) if TP + FN else 0
                F1 = 2 * P * R / (P + R) if P + R else 0
                f1_scores.append(F1)

            # 找到最大 F1 分数对应的索引
            target_idx = np.argmax(f1_scores)

        # 获取选中的阈值
        target_threshold = thresholds[target_idx]
        actual_fpr = fpr[target_idx]

        # 使用阈值进行分类
        y_pred = [score >= target_threshold for score in y_scores]

        # 计算各项指标
        TP = sum(1 for p, t in zip(y_pred, y_true) if p and t)
        FP = sum(1 for p, t in zip(y_pred, y_true) if p and not t)
        TN = sum(1 for p, t in zip(y_pred, y_true) if not p and not t)
        FN = sum(1 for p, t in zip(y_pred, y_true) if not p and t)

        # 避免除零
        TPR = TP / (TP + FN) if TP + FN else 0
        TNR = TN / (TN + FP) if TN + FP else 0
        FPR = FP / (FP + TN) if FP + TN else 0
        FNR = FN / (FN + TP) if FN + TP else 0
        P = TP / (TP + FP) if TP + FP else 0
        R = TPR  # 召回率等同于 TPR
        F1 = 2 * P * R / (P + R) if P + R else 0
        ACC = (TP + TN) / len(y_true)

        metrics: dict[str, float] = {
            "threshold": target_threshold,
            "AUROC": auroc,
            "TPR": TPR,
            "TNR": TNR,
            "FPR": FPR,
            "FNR": FNR,
            "P": P,
            "R": R,
            "F1": F1,
            "ACC": ACC,
        }

        return self._filter_metrics(metrics)
