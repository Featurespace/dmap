from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from dmap.utils import (
    format_dmap_plot,
    get_aligned_logits,
    get_rank_interval,
    logger,
    logits_to_probabilities,
)

sns.set_style("white")


class DMAP:
    def __init__(self, evaluator_model: str | Path) -> None:
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(evaluator_model)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            evaluator_model
        )
        self.samples: list = []

    def fit(self, texts: str | list[str], num_bins: int = 40) -> None:
        if self.samples:
            logger.warning("Overwriting existing fitted DMAP samples.")

        if isinstance(texts, str):
            texts = [texts]

        for idx, text in enumerate(texts):
            logger.info(f"Processing text {idx} / {len(texts)}")
            self.samples.append(self.process_text_distribution(text))

        self.averaged_histogram = self.process_weighted_histogram_data(bins=num_bins)[0]

    def plot(
        self,
        num_bins: int = 40,
        save_path: Optional[str] = None,
        figsize: Optional[tuple[int, int]] = (5, 5),
        color: Optional[str] = "#1B9E77",
    ):
        if not self.samples:
            raise RuntimeError(
                "Warning: you must first fit DMAP before generating a histogram"
            )

        fig, ax = plt.subplots(figsize=figsize)
        bin_width = 1.0 / num_bins
        ax.bar(
            np.arange(num_bins) * bin_width + (bin_width / 2),
            self.averaged_histogram,
            color=color,
            alpha=0.6,
            edgecolor="white",
            linewidth=1.0,
            width=bin_width,
        )
        format_dmap_plot(ax, self.averaged_histogram)

        if save_path:
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

    def process_text_distribution(self, text, final_cutoff=3000):
        logits, input_ids = get_aligned_logits(text, self.model, self.tokenizer)
        probabilities = logits_to_probabilities(logits)
        text_length = int(probabilities.size(1))

        int_start_list, int_end_list = [], []
        chosen_token_log_rank_list, chosen_token_prob_list = [], []
        node_entropy_list = []

        for i in range(0, min(text_length, final_cutoff)):
            all_token_probs = probabilities[
                0, i, :
            ]  # List of probabilities of possible next token.
            node_entropy_list.append(float(stats.entropy(all_token_probs.tolist())))

            token_id = input_ids[0, i]
            chosen_token_prob = all_token_probs[
                token_id
            ].item()  # List of probabilities of the token present in the text.
            chosen_token_rank, int_start, int_end = get_rank_interval(
                all_token_probs, token_id
            )

            int_start_list.append(float(int_start))
            int_end_list.append(float(int_end))
            chosen_token_log_rank_list.append(float(np.log(chosen_token_rank)))
            chosen_token_prob_list.append(float(chosen_token_prob))

        return {
            "chosen_token_log_rank_list": chosen_token_log_rank_list,
            "chosen_token_prob_list": chosen_token_prob_list,
            "int_start_list": int_start_list,
            "int_end_list": int_end_list,
            "text_length": text_length,
            "node_entropy_list": node_entropy_list,
        }

    def process_weighted_histogram_data(
        self, bins=40, initial_cutoff=0, final_cutoff=3000
    ):
        entropy_sums = [0.0] * len(self.samples)
        individual_histograms = []
        for i in range(len(self.samples)):
            individual_histogram = [0.0] * bins
            for j in range(
                initial_cutoff, min(self.samples[i]["text_length"], final_cutoff)
            ):
                entropy_sums[i] += min(self.samples[i]["node_entropy_list"][j], 2)
                for y in range(bins):
                    proportion_in_bin = (
                        min(self.samples[i]["int_end_list"][j], (y + 1) / bins)
                        - max(self.samples[i]["int_start_list"][j], y / bins)
                    ) / (
                        self.samples[i]["int_end_list"][j]
                        - self.samples[i]["int_start_list"][j]
                        + 10**-20
                    )
                    if (
                        self.samples[i]["int_start_list"][j] < (y + 1) / bins
                        and self.samples[i]["int_end_list"][j] > y / bins
                    ):
                        individual_histogram[y] += min(
                            self.samples[i]["node_entropy_list"][j], 2
                        ) * min(proportion_in_bin, 1)
            for y in range(bins):
                if entropy_sums[i] > 1e-9:
                    individual_histogram[y] *= bins / entropy_sums[i]
            individual_histograms.append(individual_histogram)
        averaged_histogram = [0.0] * bins
        entropy_sum = np.sum(entropy_sums)
        for i in range(len(self.samples)):
            for y in range(bins):
                averaged_histogram[y] += individual_histograms[i][y] * entropy_sums[i]

        for y in range(bins):
            averaged_histogram[y] /= entropy_sum

        return averaged_histogram, entropy_sum, individual_histograms, entropy_sums

    def chi_squared_uniformity_test(self, bins=4, alpha=0.05):
        logger.info(f"Running Chi-squared test at significance level alpha={alpha}")
        logger.info("Null hypothesis: the DMAP sample density is uniform")

        np.random.seed(42)
        points = []
        text_index = 0
        generated_points = 0
        for text_index in range(len(self.samples)):
            for i in range(len(self.samples[text_index]["int_start_list"])):
                points.append(
                    np.random.uniform(
                        self.samples[text_index]["int_start_list"][i],
                        self.samples[text_index]["int_end_list"][i],
                    )
                )
                generated_points += 1
            text_index += 1

        logger.info(f"Generated {generated_points} points for Chi-squared test")

        observed_counts = [0] * bins
        i = 0
        while i < len(points):
            bin_index = min(int(np.floor(points[i] * bins)), bins - 1)
            observed_counts[bin_index] += 1
            i += 1

        observed = np.array(observed_counts)
        n_bins = len(observed)
        total_count = np.sum(observed)
        expected_count_per_bin = total_count / n_bins
        expected = np.full(n_bins, expected_count_per_bin)
        chi_square_stat = np.sum((observed - expected) ** 2 / expected)
        logger.info(f"Chi-squared test statistic: {chi_square_stat}")

        df = n_bins - 1  # Degrees of freedom = number of bins - 1.
        p_value = 1 - stats.chi2.cdf(
            chi_square_stat, df
        )  # Calculate p-value (right-tail test).

        logger.info(f"Chi-squared test p-value: {p_value}")
        if p_value < alpha:
            logger.info(
                f"A p-value of {p_value} is significant at level {alpha} and gives evidence against the null hypothesis"
            )
        else:
            logger.info(f"A p-value of {p_value} is not significant at level {alpha}")
        return float(p_value)
