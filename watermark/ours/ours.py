from functools import partial
from math import sqrt
import time
import math
import string
import torch
from torch.nn import functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization

from ..base import BaseConfig, BaseWatermark


class OURSConfig(BaseConfig):
    """Config class for OURS algorithm，针对推理型 LLMs 的Critical Tokens水印算法配置"""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.gamma = self.config_dict["gamma"]
        self.delta = self.config_dict["delta"]
        self.hash_key = self.config_dict["hash_key"]
        self.z_threshold = self.config_dict["z_threshold"]
        self.prefix_length = self.config_dict["prefix_length"]
        self.f_scheme = self.config_dict["f_scheme"]
        self.window_scheme = self.config_dict["window_scheme"]      
        
        
        self.beta = self.config_dict.get("beta", 1.0)  
        self.top_k_candidates = self.config_dict.get("top_k_candidates", 10)  
        
        self.delta0 = self.config_dict.get("delta0", 1.5)
        self.delta_lambda = self.config_dict.get("delta_lambda", 3.0)
        self.beta0 = self.config_dict.get("beta0", 0.1)

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "OURS"


class OURSUtils:
    """Utility class for OURS algorithm, contains helper functions."""

    def __init__(self, config: OURSConfig, *args, **kwargs) -> None:
        """
        Initialize the OURS utility class.

        Parameters:
            config (OURSConfig): Configuration for the OURS algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)
        
        self.prf = torch.randperm(
            self.config.vocab_size,
            device=self.config.device,
            generator=self.rng,
        )
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min,
        }
        self.window_scheme_map = {
            "left": self._get_greenlist_ids_left,
            "self": self._get_greenlist_ids_self,
        }

    def _f(self, input_ids: torch.Tensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))

    def _f_time(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]  # type: ignore

    def _f_additive(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the previous token additive."""
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        return self.prf[additive_result % self.config.vocab_size]  # type: ignore

    def _f_skip(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the previous token skip."""
        return self.prf[input_ids[-self.config.prefix_length]]

    def _f_min(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the previous token min."""
        
        min_value = self.prf[input_ids[-1]]
        for i in range(1, self.config.prefix_length):
            current = self.prf[input_ids[-1 - i]]
            min_value = torch.min(min_value, current)
        return min_value


    def compute_gcc(self, think_tokens: torch.Tensor, think_logits: torch.Tensor) -> torch.Tensor:
        """
        计算全局因果贡献度 (Global Causal Contribution) 
        """
        think_logits = think_logits.float()
        dtype = think_logits.dtype
        probs = torch.softmax(think_logits, dim=-1)  # [N, V]
        gcc_scores = self._compute_gcc_streaming(probs)
        if gcc_scores.dtype != dtype:
            gcc_scores = gcc_scores.to(dtype)
        return gcc_scores

    def _compute_gcc_streaming(self, probs: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """分块(流式) GCC 计算，避免构造 N×V 大矩阵。"""
        N, vocab_size = probs.shape
        device = self.config.device
        dtype = probs.dtype  

        
        lambda_weights = torch.zeros(N, device=device, dtype=dtype)
        if N > 1:
            for i in range(1, N):
                kl_div = torch.sum(
                    probs[i] * (torch.log(probs[i] + 1e-10) - torch.log(probs[i - 1] + 1e-10))
                )
                lambda_weights[i] = kl_div

        
        probs_norm = F.normalize(probs, dim=-1, eps=1e-10)
        similarities = torch.mm(probs_norm, probs_norm.T)  # [N, N]
        upper_tri_mask = torch.triu(torch.ones(N, N, device=device, dtype=dtype), diagonal=1)
        sim_upper = similarities * upper_tri_mask
        row_sum = sim_upper.sum(dim=-1, keepdim=True) + 1e-10
        alpha_matrix = sim_upper / row_sum  # [N, N]

        
        B = lambda_weights.unsqueeze(1) * alpha_matrix  # [N, N]

        
        gcc_scores = torch.zeros(vocab_size, device=device, dtype=dtype)

        
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            idx = torch.arange(start, end, device=device)
            P_chunk = probs[:, idx]  
            
            BP_chunk = torch.mm(B, P_chunk)
            gcc_chunk = (P_chunk * BP_chunk).sum(dim=0)
            gcc_scores[idx] = gcc_chunk

        return gcc_scores

    def compute_cps(self, think_tokens: torch.Tensor, think_logits: torch.Tensor) -> torch.Tensor:
        """
        计算竞争持续性评分 (Competitive Persistence Scoring) - 始终使用流式算法
        """
        think_logits = think_logits.float()
        dtype = think_logits.dtype
        probs = torch.softmax(think_logits, dim=-1)  # [N, V]

        _, top_k_indices = torch.topk(
            think_logits, self.config.top_k_candidates, dim=-1
        )  # [N, k_top]

        cps_scores = self._compute_cps_streaming(
            probs, think_logits, think_tokens, top_k_indices
        )
        if cps_scores.dtype != dtype:
            cps_scores = cps_scores.to(dtype)
        return cps_scores

    def _compute_cps_streaming(
        self,
        probs: torch.Tensor,
        think_logits: torch.Tensor,
        think_tokens: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        
        device = self.config.device
        N, vocab_size = probs.shape
        k_top = self.config.top_k_candidates

        
        generated_probs = probs[torch.arange(N, device=device), think_tokens]
        # surprise_rewards = 1.0 / (-torch.log(generated_probs + 1e-10))  # [N]
        denom = -torch.log(generated_probs.clamp_min(1e-12))  # [N], >= 0
        surprise_rewards = 1.0 / torch.clamp(denom, min=1e-6)  

        
        top2_vals, _ = torch.topk(think_logits, 2, dim=1)             # [N, 2]
        delta_selected = torch.abs(top2_vals[:, 0] - top2_vals[:, 1]) # [N]
        comp_selected = torch.clamp(1.0 - delta_selected, min=0.0)    # [N]

        future_counts = torch.zeros(vocab_size, dtype=torch.int32, device=device)
        cps_scores = torch.zeros(vocab_size, dtype=torch.float32, device=device)

        for i in range(N - 1, -1, -1):
            S_inv = surprise_rewards[i]
            t_i = think_tokens[i]

            step_probs = probs[i]                          
            tk = top_k_indices[i]                          
            mask_other = tk != t_i
            tk_other = tk[mask_other]

            
            top2_prob_vals, _ = torch.topk(step_probs, 2)
            delta_selected = torch.abs(top2_prob_vals[0] - top2_prob_vals[1])  # ≤1
            comp_selected = 1.0 - delta_selected
            contrib_ti = S_inv * comp_selected * future_counts[t_i].float()
            cps_scores[t_i] += contrib_ti

            
            if tk_other.numel() > 0:
                probs_candidate = step_probs[tk_other]
                prob_selected = step_probs[t_i]
                delta_other = torch.abs(probs_candidate - prob_selected)
                delta_other = torch.clamp(delta_other, max=1.0)  
                comp_other = 1.0 - delta_other
                contrib_other = S_inv * comp_other * future_counts[tk_other].float()
                cps_scores[tk_other] += contrib_other
                
            future_counts[tk] += 1

        return cps_scores

    def compute_criticality_score(self, think_tokens: torch.Tensor, think_logits: torch.Tensor) -> torch.Tensor:
        """
        计算综合关键得分 (Criticality Score)
        
        参数:
            think_tokens (torch.Tensor): 思考段tokens [N]
            think_logits (torch.Tensor): 思考段logits [N, vocab_size]
            
        返回:
            torch.Tensor: 每个词的CS得分 [vocab_size]
        """
        think_logits = think_logits.float()
        gcc_scores = self.compute_gcc(think_tokens, think_logits)
        cps_scores = self.compute_cps(think_tokens, think_logits)
        
        # CS(w) = GCC(w) * log(1 + CPS(w))
        cs_scores = gcc_scores * torch.log(1 + cps_scores)
        
        return cs_scores


    def get_greenlist_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    def _get_greenlist_ids_left(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        self.rng.manual_seed(
            (self.config.hash_key * self._f(input_ids)) % self.config.vocab_size
        )
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        
        vocab_permutation = torch.randperm(
            self.config.vocab_size, device=self.config.device, generator=self.rng
        )
        
        greenlist_ids = vocab_permutation[:greenlist_size].to(input_ids.device)
        return greenlist_ids

    def _get_greenlist_ids_self(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get greenlist ids for the input_ids via selfHash scheme."""
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        greenlist_ids = []
        f_x = self._f(input_ids)
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])
            self.rng.manual_seed(h_k % self.config.vocab_size)
            
            vocab_permutation = torch.randperm(
                self.config.vocab_size,
                device=self.config.device,
                generator=self.rng,
            )
            
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        
        return torch.tensor(greenlist_ids, device=input_ids.device)

    def _compute_z_score(self, observed_count: int, T: int) -> float:
        """
        Compute calibrated z-score considering the critical tokens.

        参数:
            observed_count (int): 观察到的绿名单token数量
            T (int): 总token数量

        返回:
            float: 校准后的z-score
            
        """
        mu = self.config.gamma
        numer = observed_count - mu * T
        denom = sqrt(T * mu * (1 - mu))
        z = numer / denom
        return z

    def score_sequence(
        self, input_ids: torch.Tensor
    ) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token_id = int(input_ids[idx].item())
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            is_green = (greenlist_ids == curr_token_id).any().item()
            if is_green:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        ratio = green_token_count / max(1, num_tokens_scored)
        #print(f"[detect] green_ratio={ratio:.3f}, gamma={self.config.gamma}")
        return z_score, green_token_flags


class OURSLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for OURS algorithm, process logits to add watermark."""

    def __init__(
        self, config: OURSConfig, utils: OURSUtils, *args, **kwargs
    ) -> None:
        """
        Initialize the OURS logits processor.

        参数:
            config (OURSConfig): OURS 算法的配置
            utils (OURSUtils): OURS 算法的工具类
        """
        self.config = config
        self.utils = utils

        
        self.think_end_token_id: int = self.config.generation_tokenizer.encode(
            "</think>", add_special_tokens=False
        )[-1]
    
        self.embed_weight = self.config.generation_model.get_input_embeddings().weight  # [V, D]
        
        self.delta0 = getattr(self.config, "delta0", 1.5)
        self.delta_lambda = getattr(self.config, "delta_lambda", 3.0)
        self.beta0 = getattr(self.config, "beta0", 0.1)

        
        self._spacy_nlp = None
        self.spacy_excluded_pos = set(["ADP", "CCONJ", "SCONJ", "DET", "AUX", "PUNCT", "PART"])
        self._manual_stopwords = set([
            "the", "and", "a", "an", "to", "of", "in", "on", "for", "with",
            "as", "at", "by", "from", "or", "but", "if", "is", "are", "be",
            "was", "were", "this", "that", "these", "those", "it", "i", "no",
            "her", "his", "he", "she", "they", "we", "you", "me", "him", "them",
            "us", "my", "your", "their", "our", "its", " "
        ])
        self.reset_state()

    def reset_state(self):
        """重置处理器状态，处理新的生成会话"""
        
        self.passed_think = False
        
        self.think_tokens_list: list[int] = []
        self.think_logits_list: list[torch.Tensor] = []
        
        self.R: torch.Tensor | None = None
        
        self.critical_tokens: torch.Tensor | None = None
        
        self.think_start_time: float | None = None
        self.total_think_steps = 0
        
        self.prev_logits: torch.Tensor | None = None
        
        self.last_update_pos: int = 0

    def _ensure_spacy(self):
        if self._spacy_nlp is not None:
            return
        try:
            import spacy  # type: ignore
            self._spacy_nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"[OURS][spaCy] Load failed: {e}")
            self._spacy_nlp = None

    def _should_filter_by_pos(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        
        if all(ch in string.punctuation for ch in stripped):
            return True
        norm = stripped.lower()
        
        if norm in self._manual_stopwords:
            return True
        
        try:
            from spacy.lang.en import STOP_WORDS  # type: ignore
            if norm in STOP_WORDS:
                return True
        except Exception:
            pass
        
        self._ensure_spacy()
        if self._spacy_nlp is None:
            return False
        doc = self._spacy_nlp(stripped)
        if len(doc) != 1:
            return False
        return doc[0].pos_ in self.spacy_excluded_pos

    def _calc_greenlist_mask(
        self, scores: torch.Tensor, greenlist_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate greenlist mask for the given scores and greenlist token ids.

        参数:
            scores (torch.Tensor): 分数 [batch_size, vocab_size]
            greenlist_token_ids (torch.Tensor): 绿名单 token ids [batch_size, greenlist_size]

        返回:
            torch.Tensor: 绿名单掩码 [batch_size, vocab_size]
        """
        
        green_tokens_mask = torch.zeros_like(
            scores, dtype=torch.bool
        )  # [batch_size, vocab_size]

        
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx].index_fill_(
                0, greenlist_token_ids[b_idx], True
            )

        return green_tokens_mask

    def _bias_logits(
        self, scores: torch.Tensor, tokens_mask: torch.Tensor, bias: float
    ) -> torch.Tensor:
        """
        Bias the scores for the tokens.

        参数:
            scores (torch.Tensor): 分数 [batch_size, vocab_size]
            tokens_mask (torch.Tensor): 要添加偏置的掩码 [batch_size, vocab_size]
            bias (float): 偏置值

        返回:
            torch.Tensor: 添加偏置后的分数 [batch_size, vocab_size]
        """
        scores = scores + bias * tokens_mask.float()
        return scores

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Process logits to add watermark.

        参数:
            input_ids (torch.Tensor): 输入 token ids [batch_size, seq_len]
            scores (torch.Tensor): logits 分数 [batch_size, vocab_size]

        返回:
            torch.Tensor: 添加水印后的 logits [batch_size, vocab_size]
        """
        batch_size = input_ids.shape[0]

        
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        
        input_ids_seq = input_ids[0]  # [seq_len]

        
        think_end_indices = (input_ids_seq == self.think_end_token_id).nonzero(
            as_tuple=True
        )[0]

        if len(think_end_indices) == 0 and self.passed_think:
            self.reset_state()

        if len(think_end_indices) > 0 and not self.passed_think:
            self.passed_think = True

            
            if self.think_start_time is not None:
                think_end_time = time.time()
                think_duration = think_end_time - self.think_start_time
                
            
            if self.think_tokens_list and self.think_logits_list:
                extraction_start = time.time()

                embed_device = self.embed_weight.device

                think_tokens = torch.tensor(
                    self.think_tokens_list,
                    dtype=torch.long,
                    device=self.config.device,
                )
                think_logits = torch.stack(self.think_logits_list)  # [N, vocab_size]

                
                cs_scores = self.utils.compute_criticality_score(
                    think_tokens.to(think_logits.device), think_logits
                )

                
                k_target = min(10, cs_scores.numel())
                kept_ids = []
                kept_scores = []
                if k_target > 0:
                    vals_all, idx_all = torch.topk(cs_scores, cs_scores.numel())
                    for score, tid in zip(vals_all.tolist(), idx_all.tolist()):
                        tok_text = self.config.generation_tokenizer.decode(int(tid))
                        if not self._should_filter_by_pos(tok_text):
                            kept_ids.append(tid)
                            kept_scores.append(score)
                            if len(kept_ids) >= k_target:
                                break
                
                if len(kept_ids) == 0 and cs_scores.numel() > 0:
                    top1_val, top1_idx = torch.topk(cs_scores, 1)
                    kept_ids = [int(top1_idx[0].item())]
                    kept_scores = [float(top1_val[0].item())]

                cs_top_idx = torch.tensor(kept_ids, device=think_logits.device, dtype=torch.long)
                cs_top_vals = torch.tensor(kept_scores, device=think_logits.device, dtype=cs_scores.dtype)

                
                
                a = torch.softmax(cs_top_vals, dim=0)                      # [m] on think_logits.device
                idx_on_embed = cs_top_idx.to(embed_device)
                E = self.embed_weight[idx_on_embed].float()                # [m, d] on embed_device
                E = F.normalize(E, dim=1)

                
                a_embed = a.to(embed_device)
                denom = a_embed.sum() + 1e-12
                mu = (a_embed.unsqueeze(1) * E).sum(dim=0) / denom         # [d]
                X = E - mu                                                 # [m, d]
                Xw = X * a_embed.sqrt().unsqueeze(1)                       # [m, d]

                
                try:
                    if Xw.shape[0] >= 2:
                        U, S, V = torch.pca_lowrank(Xw, q=1, center=False) # V: [d, 1]
                        R0 = V[:, 0]
                        R0 = F.normalize(R0, dim=0)
                        if torch.isnan(R0).any() or torch.isinf(R0).any():
                            R0 = F.normalize(mu, dim=0)
                    else:
                        R0 = F.normalize(mu, dim=0)
                except Exception:
                    R0 = F.normalize(mu, dim=0)

                self.R = R0  
                self.last_update_pos = len(input_ids_seq)

                extraction_end = time.time()
                
            
            self.think_tokens_list = []
            self.think_logits_list = []
            
            self.prev_logits = None

        
        if not self.passed_think:
            
            if self.think_start_time is None:
                self.think_start_time = time.time()
            
            self.total_think_steps += 1
            
            if self.prev_logits is not None and len(input_ids_seq) > 0:
                last_token = input_ids_seq[-1].item()  
                self.think_tokens_list.append(last_token)
                self.think_logits_list.append(self.prev_logits)
            
            self.prev_logits = scores[0].clone().detach()
            
            return scores



        
        curr_len = len(input_ids_seq)
        if self.R is not None and curr_len > self.last_update_pos:
            last_token_id = input_ids_seq[-1].item()
            if last_token_id != self.think_end_token_id:
                e_sel = self.embed_weight[last_token_id].float()
                e_sel = F.normalize(e_sel.unsqueeze(0), dim=1)[0]
                s_selected = torch.dot(e_sel, self.R).clamp(0.0, 1.0).item()
                beta_t = self.beta0 * s_selected
                self.R = F.normalize(((1.0 - beta_t) * self.R + beta_t * e_sel).unsqueeze(0), dim=1)[0]
            self.last_update_pos = curr_len

        
        batched_greenlist_ids = []
        for b_idx in range(batch_size):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids.append(greenlist_ids)

        
        if self.R is None:
            print("未提取到think tokens，退化为KGW")
            green_tokens_mask = self._calc_greenlist_mask(
                scores, torch.stack(batched_greenlist_ids)
            )
            return self._bias_logits(scores, green_tokens_mask, self.config.delta)

        
        embed_device = self.embed_weight.device
        for b_idx in range(batch_size):
            green_ids = batched_greenlist_ids[b_idx]  # [G]
            if green_ids.numel() == 0:
                continue

            
            green_ids_embed = green_ids.to(embed_device)
            E = self.embed_weight[green_ids_embed].float()      # [G, D] on embed_device
            E = F.normalize(E, dim=1)
            s = torch.matmul(E, self.R).clamp(0.0, 1.0)         # [G] on embed_device
            delta_vec = (self.delta0 + self.delta_lambda * s).to(scores.dtype)  # [G] on embed_device

            
            green_ids_scores = green_ids.to(scores.device)
            delta_vec_scores = delta_vec.to(scores.device)
            scores[b_idx, green_ids_scores] = scores[b_idx, green_ids_scores] + delta_vec_scores

        
        mask_inf = torch.isinf(scores)
        has_nan = torch.isnan(scores).any()
        has_pos_inf = (mask_inf & (scores > 0)).any()
        if has_nan or has_pos_inf:
            print("[OURS] scores contains NaN or +Inf")
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
        return scores


class OURS(BaseWatermark):
    """针对推理型 LLMs 的Critical Tokens水印算法顶层类"""

    def __init__(
        self,
        algorithm_config: str | OURSConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        初始化 OURS 算法

        参数:
            algorithm_config (str | OURSConfig): 算法配置文件路径或 OURSConfig 实例
            transformers_config (TransformersConfig | None): transformers 模型的配置
        """
        if isinstance(algorithm_config, str):
            self.config = OURSConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, OURSConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be either a path string or a OURSConfig instance"
            )

        self.utils = OURSUtils(self.config)
        self.logits_processor = None
        if self.config.generation_model is not None:
            self.logits_processor = OURSLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        
        # Configure generate_with_watermark
        if self.logits_processor is None:
            self.logits_processor = OURSLogitsProcessor(self.config, self.utils)
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(
            encoded_watermarked_text, skip_special_tokens=True
        )[0]
        return watermarked_text

    def detect_watermark(
        self, text: str, return_dict: bool = True, *args, **kwargs
    ) -> tuple[bool, float] | dict[str, bool | float]:
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)

        # Compute z_score using a utility method
        z_score, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def get_data_for_visualization(
        self, text: str, *args, **kwargs
    ) -> DataForVisualization:
        """Get data for visualization."""

        # Encode text
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)

        # Compute z-score and highlight values
        z_score, highlight_values = self.utils.score_sequence(encoded_text)

        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)

        return DataForVisualization(decoded_tokens, highlight_values)