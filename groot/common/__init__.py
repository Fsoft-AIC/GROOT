import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = os.path.join(ROOT_DIR, "data")
CKPT_DIR = os.path.join(ROOT_DIR, "ckpts")


def get_gt_csv(task):
    return os.path.join(DATA_DIR, f"{task}/ground_truth.csv")


def get_oracle_dir(task):
    return os.path.join(CKPT_DIR, f"{task}/oracles")
