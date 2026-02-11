import sys

import numpy as np
import torch
from loguru import logger

logger.remove()

logger.add(
    sys.stdout,
    colorize=False,
    format="{time:HH:mm:ss} | {level} | {message}",
    diagnose=False,
    backtrace=False,
)


def get_aligned_logits(text, model, tokenizer, max_tokens=2000):
    """
    Returns lists of logits and tokenIDs, where the kth item in each list relates to
    the probability distribution over token k + 1 and the chosen token at position k + 1.
    """

    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_tokens]

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits
    logits_t0_to_k_minus_2 = logits[:, :-1, :]
    tokens_t1_to_k_minus_1 = input_ids[:, 1:]
    return logits_t0_to_k_minus_2, tokens_t1_to_k_minus_1


def logits_to_probabilities(logits):
    """
    Convert raw logits to probabilities.
    """

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities


def get_rank_interval(probs, token_index):
    """
    Returns the rank of the token, and the start and end of the interval
    in [a,b] in [0,1] corresponding to the token choice.
    """

    rank = 1
    int_start = 0
    token_prob = probs[token_index]
    big_probs = probs[probs > token_prob].tolist()
    rank = len(big_probs) + 1
    int_start = np.sum(big_probs)
    int_end = int_start + token_prob
    return rank, int_start, int_end


def format_dmap_plot(ax, histogram):
    ax.axhline(y=1, color="dimgrey", linestyle="-")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("dimgrey")
        ax.spines["left"].set_linewidth(1.4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(histogram) * 1.1)

    ax.set_xlabel("DMAP Samples", fontsize=12, fontweight="medium", color="#2C3E50")
    ax.set_ylabel("Density", fontsize=12, fontweight="medium", color="#2C3E50")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
        colors="#34495E",
        length=6,
        width=1.2,
    )
