import argparse
import yaml
import importlib
import pandas as pd
import os
import torch
import json
from datetime import datetime
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from groot.common import DATA_DIR
from groot.common.utils import parse_module_name_from_path
from groot.eval import EvalRunner
from groot.models import CNNVAE
from groot.models.modules.predictor import DropoutPredictor
from groot.optimizers.optimizer import OptimizerInterface


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize sequences.")
    parser.add_argument("config_file", type=str, help="Path to config module")
    parser.add_argument("--model_ckpt_path",
                        type=str,
                        required=True,
                        help="Checkpoint of model.")
    parser.add_argument("--dataset", type=str, choices=["AAV", "GFP"], required=True)
    parser.add_argument("--level",
                        type=str,
                        choices=["easy", "medium", "hard", "harder1", "harder2", "harder3"],
                        required=True)
    parser.add_argument("--optim_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--devices",
                        type=str,
                        default="-1",
                        help="Training devices separated by comma.")
    parser.add_argument("--changes",
                        type=str,
                        nargs="+",
                        help="Specify any changes to the yaml file (optim only).")
    args = parser.parse_args()
    return args


def initialize_models(dec_type: str, model_ckpt: str, device: torch.device):
    module = CNNVAE.load_from_checkpoint(model_ckpt,
                                         map_location=device,
                                         device=device)
    module.eval()

    return module


def sample_pool_sequences(
    csv_file: str,
    frac: float = None,
    strategy: str = "random",
    seed: int = 0
):
    df = pd.read_csv(csv_file)
    if strategy == "random":
        pool_df = df.sample(frac=frac, random_state=seed)
    elif strategy == "bottom":
        n = int(frac * len(df))
        pool_df = df.nsmallest(n, columns="target")
    elif strategy == "quantile":
        quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        dfs = []
        for i in range(len(quantiles) - 1):
            lower_val = df.target.quantile(quantiles[i])
            upper_val = df.target.quantile(quantiles[i + 1])
            filtered_df = df[df.target.between(lower_val, upper_val)]
            dfs.append(filtered_df)
        pool_df = pd.concat(dfs, ignore_index=True)

    return pool_df, df


def alter_argument(changes, config: DictConfig):
    for change in changes:
        conf, val = change.split("=")
        subconf = conf.split(".")
        if len(subconf) == 1:
            if isinstance(config[subconf[0]], bool):
                val = int(val)
            config[subconf[0]] = type(config[subconf[0]])(val)
        elif len(subconf) == 2:
            if isinstance(config[subconf[0]][subconf[1]], bool):
                val = int(val)
            config[subconf[0]][subconf[1]] = type(config[subconf[0]][subconf[1]])(val)
        else:
            raise ValueError("Not valid alternatives.")

    return config


def get_csv_files(rootdir: str, dataset: str, difficulty: str):
    if difficulty == "easy":
        csv_file = f"{rootdir}/{dataset}/mutations_0/percentile_0.5_0.6/base_seqs.csv"
    elif difficulty == "medium":
        csv_file = f"{rootdir}/{dataset}/mutations_6/percentile_0.2_0.4/base_seqs.csv"
    elif difficulty == "hard":
        csv_file = f"{rootdir}/{dataset}/mutations_7/percentile_0.0_0.3/base_seqs.csv"
    else:
        no_mut = 13 if dataset == "AAV" else "8"
        if difficulty == "harder1":
            csv_file = f"{rootdir}/{dataset}/mutations_{no_mut}/percentile_0.0_0.3/base_seqs.csv"
        elif difficulty == "harder2":
            csv_file = f"{rootdir}/{dataset}/mutations_{no_mut}/percentile_0.0_0.2/base_seqs.csv"
        elif difficulty == "harder3":
            csv_file = f"{rootdir}/{dataset}/mutations_{no_mut}/percentile_0.0_0.1/base_seqs.csv"

    return csv_file


def init_everything(args):
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load configs
    cfg = importlib.import_module(parse_module_name_from_path(
        args.config_file))

    with open(args.optim_config_path, "r") as file_:
        optim_config = yaml.safe_load(file_)
        optim_config = DictConfig(optim_config)

    if len(args.changes) != 0:
        optim_config = alter_argument(args.changes, optim_config)

    with open(f"scripts/configs/evaluate/{args.dataset}.yaml", "r") as file_:
        eval_config = yaml.safe_load(file_)
        eval_config = DictConfig(eval_config)

    print("Finish loading configurations.")

    # Get data file
    csv_file = get_csv_files(DATA_DIR, args.dataset, args.level)

    torch.set_float32_matmul_precision(cfg.precision)
    device = torch.device("cpu" if args.devices == "-1" else f"cuda:{args.devices}")
    module = initialize_models(cfg.decoder_type, args.model_ckpt_path, device)

    return curr_time, device, module, optim_config, eval_config, csv_file


def main(args, seed, curr_time, device, module, optim_config, eval_config, csv_file):
    seed_everything(seed)

    pool_df, df = sample_pool_sequences(csv_file,
                                        optim_config.pool_frac,
                                        optim_config.sample_strategy,
                                        seed)
    save_path = f"{args.output_dir}/optim_log/{args.dataset}/{args.level}/base_pool_{optim_config.pool_frac}_{seed}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pool_df.to_csv(save_path, index=False)
    pool_seqs = pool_df.sequence.tolist()

    predictor = DropoutPredictor(module.hparams.latent_dim, optim_config.pred_hidden_dim)
    optimizer = OptimizerInterface(module, predictor, optim_config)

    if optim_config.smooth:
        optimizer.smooth_func(pool_df, args.batch_size, device)

    optimized_seqs, pred_scores = optimizer.optimize(pool_seqs)

    eval_config.base_pool_path = save_path
    evaluator = EvalRunner(eval_config)

    prefix = "smoothed" if optim_config.smooth else "unsmoothed"
    results_df, metrics_df = evaluator.evaluate_sequences(pred_scores, optimized_seqs)
    save_path = f"{args.output_dir}/log_results/{args.dataset}/{args.level}/{prefix}_{optim_config.algo_name}/run_{curr_time}"

    os.makedirs(save_path, exist_ok=True)
    results_df.to_csv(os.path.join(save_path, f"sequences_{seed}.csv"), index=False)
    metrics_df.to_csv(os.path.join(save_path, f"metrics_{seed}.csv"), index=False)

    print(results_df.head(5))
    print(metrics_df)

    return optim_config, eval_config, save_path


def merge_write_dictionaries(args, optim_config, eval_config, save_dir):
    conf = vars(args)
    conf.update(OmegaConf.to_container(optim_config, resolve=True))
    conf.update(OmegaConf.to_container(eval_config, resolve=True))
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(dict(conf), f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    curr_time, device, module, optim_config, eval_config, csv_file = init_everything(args)

    for seed in range(5):
        optim_config, eval_config, save_dir = main(
            args, seed, curr_time, device, module, optim_config, eval_config, csv_file
        )
        merge_write_dictionaries(args, optim_config, eval_config, save_dir)
