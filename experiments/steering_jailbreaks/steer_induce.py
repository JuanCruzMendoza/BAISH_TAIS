"""Spec 5.5: induce compliance on jailbreaks that were refused. GPU.

    python steer_induce.py <model> --direction story_v2 --sweep-layers 15,17,18 --alpha 0.5
    python steer_induce.py <model> --direction harm --layers band

The mirror image of 5.4: add at +alpha for story/persona, ablate for harm/eval. Same
flags, same cell contract; only the prompt set and the mode mapping differ.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.steering_jailbreaks import steer_single

SCRIPT = "steer_induce"
PROMPT_SET = "refusal"


if __name__ == "__main__":
    args = steer_single.add_cell_args(argparse.ArgumentParser()).parse_args()
    steer_single.run(args, SCRIPT, PROMPT_SET, "refusal")
