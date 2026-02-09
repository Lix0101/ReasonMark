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

# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json
import random


class BaseDataset:
    """Base class for dataset."""

    def __init__(self, max_samples: int = 200, seed: int | None = None):
        """
        Initialize the dataset.

        Parameters:
            max_samples (int): Maximum number of samples to load. Default is 200.
            seed (int | None): Random seed for sampling data. If None, sequential sampling is used.
        """
        self.max_samples = max_samples
        self.seed = seed
        self.prompts = []
        self.natural_texts = []
        self.references = []

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def _get_samples(self, data, max_samples):
        """Get samples using either random or sequential sampling.

        Args:
            data: The full dataset
            max_samples: Maximum number of samples to return

        Returns:
            List of selected samples
        """
        if len(data) <= max_samples:
            return data

        if self.seed is not None:
            # 使用随机抽样
            random.seed(self.seed)
            indices = random.sample(range(len(data)), max_samples)
            return [data[i] for i in sorted(indices)]
        else:
            # 顺序抽样
            return data[:max_samples]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass


class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(
        self, data_source: str, max_samples: int = 200, seed: int | None = None
    ):
        """
        Initialize the C4 dataset.

        Parameters:
            data_source (str): The path to the C4 dataset file.
            max_samples (int): Maximum number of samples to load.
            seed (int | None): Random seed for sampling data. If None, sequential sampling is used.
        """
        super().__init__(max_samples, seed)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, "r") as f:
            lines = f.readlines()

        selected_lines = self._get_samples(lines, self.max_samples)

        for line in selected_lines:
            item = json.loads(line)
            self.prompts.append(item["prompt"])
            self.natural_texts.append(item["natural_text"])


class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""

    def __init__(
        self, data_source: str, max_samples: int = 200, seed: int | None = None
    ) -> None:
        """
        Initialize the WMT16 DE-EN dataset.

        Parameters:
            data_source (str): The path to the WMT16 DE-EN dataset file.
            max_samples (int): Maximum number of samples to load.
            seed (int | None): Random seed for sampling data. If None, sequential sampling is used.
        """
        super().__init__(max_samples, seed)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, "r") as f:
            lines = f.readlines()

        selected_lines = self._get_samples(lines, self.max_samples)

        for line in selected_lines:
            item = json.loads(line)
            self.prompts.append(item["de"])
            self.references.append(item["en"])


class WMT19ZH_ENDataset(BaseDataset):
    """Dataset class for WMT19 ZH-EN dataset."""

    def __init__(
        self, data_source: str, max_samples: int = 200, seed: int | None = None
    ) -> None:
        """
        Initialize the WMT19 ZH-EN dataset.

        Parameters:
            data_source (str): The path to the WMT19 ZH-EN dataset file.
            max_samples (int): Maximum number of samples to load.
            seed (int | None): Random seed for sampling data. If None, sequential sampling is used.
        """
        super().__init__(max_samples, seed)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """Load data from the WMT19 ZH-EN dataset file."""
        with open(self.data_source, "r") as f:
            lines = f.readlines()

        selected_lines = self._get_samples(lines, self.max_samples)

        for line in selected_lines:
            item = json.loads(line)
            self.prompts.append(item["zh"])
            self.references.append(item["en"])


class CNN_DailyMailDataset(BaseDataset):
    """Dataset class for CNN/DailyMail dataset."""

    def __init__(
        self,
        data_source: str,
        max_samples: int = 200,
        global_prompt="Please summarize the following article: ",
        seed: int | None = None,
    ) -> None:
        """
        Initialize the CNN/DailyMail dataset.

        Parameters:
            data_source (str): The path to the CNN/DailyMail dataset file.
            max_samples (int): Maximum number of samples to load.
            global_prompt (str): Global prompt to prepend to each article.
            seed (int | None): Random seed for sampling data. If None, sequential sampling is used.
        """
        super().__init__(max_samples, seed)
        self.data_source = data_source
        self.global_prompt = global_prompt
        self.load_data()

    def load_data(self):
        """Load data from the CNN/DailyMail dataset file."""
        with open(self.data_source, "r") as f:
            lines = f.readlines()

        selected_lines = self._get_samples(lines, self.max_samples)

        for line in selected_lines:
            item = json.loads(line)
            self.prompts.append(f"{self.global_prompt}{item['article']}")
            self.references.append(item["highlights"])


class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""

    def __init__(
        self, data_source: str, max_samples: int = 200, seed: int | None = None
    ) -> None:
        """
        Initialize the HumanEval dataset.

        Parameters:
            data_source (str): The path to the HumanEval dataset file.
            max_samples (int): Maximum number of samples to load.
            seed (int | None): Random seed for sampling data. If None, sequential sampling is used.
        """
        super().__init__(max_samples, seed)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, "r") as f:
            lines = f.readlines()

        selected_lines = self._get_samples(lines, self.max_samples)

        for line in selected_lines:
            item = json.loads(line)
            # process prompt
            prompt = item["prompt"]
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '"""'

            self.prompts.append(prompt)
            self.references.append(
                {
                    "task": prompt,
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                }
            )


class GSM8KDataset(BaseDataset):
    """Dataset class for GSM8K dataset."""

    def __init__(
        self, data_source: str, max_samples: int = 200, seed: int | None = None
    ) -> None:
        """
        Initialize the GSM8K dataset.

        Parameters:
            data_source (str): The path to the GSM8K dataset file.
            max_samples (int): Maximum number of samples to load.
            seed (int | None): Random seed for sampling data. If None, sequential sampling is used.
        """
        super().__init__(max_samples, seed)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """从 GSM8K 数据集文件加载数据并处理答案"""
        with open(self.data_source, "r") as f:
            lines = f.readlines()

        selected_lines = self._get_samples(lines, self.max_samples)

        for line in selected_lines:
            item = json.loads(line)
            self.prompts.append(item["question"])

            # 处理答案，提取 #### 后的内容
            answer = item["answer"]
            if "#### " in answer:
                # 分割答案，获取 #### 后的部分
                answer = answer.split("#### ")[-1].strip()

            self.references.append(answer)


class MMLU_ProDataset(BaseDataset):
    """数据集类用于 MMLU-Pro 数据集。"""

    def __init__(
        self, data_source: str, max_samples: int = 200, seed: int | None = None
    ) -> None:
        """
        初始化 MMLU-Pro 数据集。

        参数:
            data_source (str): MMLU-Pro 数据集文件的路径。
            max_samples (int): 要加载的最大样本数。
            seed (int | None): 采样数据的随机种子。如果为 None，则使用顺序采样。
        """
        super().__init__(max_samples, seed)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """从 MMLU-Pro 数据集文件加载数据。"""
        with open(self.data_source, "r") as f:
            lines = f.readlines()

        selected_lines = self._get_samples(lines, self.max_samples)

        for line in selected_lines:
            item = json.loads(line)
            # 将问题和选项存储为字典
            self.prompts.append(
                {"question": item["question"], "options": item["options"]}
            )
            # 存储答案用于评估
            self.references.append(item["answer"])


class AIME2025Dataset(BaseDataset):
    """数据集类用于 AIME 2025 数据集。"""

    def __init__(
        self, data_source: str, max_samples: int = 200, seed: int | None = None
    ) -> None:
        """
        初始化 AIME 2025 数据集。

        参数:
            data_source (str): AIME 2025 数据集文件的路径。
            max_samples (int): 要加载的最大样本数。
            seed (int | None): 采样数据的随机种子。如果为 None，则使用顺序采样。
        """
        super().__init__(max_samples, seed)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """从 AIME 2025 数据集文件加载数据。"""
        with open(self.data_source, "r") as f:
            lines = f.readlines()

        selected_lines = self._get_samples(lines, self.max_samples)

        for line in selected_lines:
            item = json.loads(line)
            # 将问题存储到 prompts 中
            self.prompts.append(item["problem"])
            # 存储答案用于评估
            self.references.append(item["answer"])


if __name__ == "__main__":
    d1 = C4Dataset("dataset/c4/processed_c4.json", max_samples=100)
    d2 = WMT16DE_ENDataset(
        "dataset/wmt16_de_en/validation.jsonl", max_samples=100
    )
    d3 = HumanEvalDataset("dataset/HumanEval/test.jsonl", max_samples=100)
    d4 = WMT19ZH_ENDataset(
        "dataset/wmt19_zh_en/validation.jsonl", max_samples=100
    )
