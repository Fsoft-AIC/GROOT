import os
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from polyleven import levenshtein
from omegaconf import OmegaConf
from .common import get_gt_csv, get_oracle_dir
from .utils.tokenize import Encoder
from .models.predictors import BaseCNN


def to_np(x: torch.Tensor):
    return x.cpu().detach().numpy()


def to_list(x: torch.Tensor):
    return to_np(x).tolist()


def diversity(seqs):
    num_seqs = len(seqs)
    total_dist = 0
    for i in range(num_seqs):
        for j in range(num_seqs):
            x = seqs[i]
            y = seqs[j]
            if x == y:
                continue
            total_dist += levenshtein(x, y)
    return total_dist / (num_seqs * (num_seqs - 1))


class EvalRunner:

    def __init__(self, runner_cfg):
        self._cfg = runner_cfg
        self._log = logging.getLogger(__name__)
        self.predictor_tokenizer = Encoder()
        self._cfg.gt_csv = get_gt_csv(self._cfg.task)
        self._cfg.oracle_dir = get_oracle_dir(self._cfg.task)
        gt_csv = pd.read_csv(self._cfg.gt_csv)
        gt_csv = gt_csv[gt_csv.augmented == 0]
        oracle_dir = self._cfg.oracle_dir
        self.use_normalization = self._cfg.use_normalization
        # Read in known sequences and their fitnesses
        self._max_known_score = np.max(gt_csv.target)
        self._min_known_score = np.min(gt_csv.target)
        self.normalize = lambda x: to_np(
            (x - self._min_known_score) /
            (self._max_known_score - self._min_known_score)).item()

        # Read in base pool used to generate sequences.
        base_pool_seqs = pd.read_csv(self._cfg.base_pool_path)
        self._base_pool_seqs = base_pool_seqs.sequence.tolist()

        self.device = torch.device('cuda')
        oracle_path = os.path.join(oracle_dir, 'cnn_oracle.ckpt')
        oracle_state_dict = torch.load(oracle_path, map_location=self.device)
        cfg_path = os.path.join(oracle_dir, 'config.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)

        self._cnn_oracle = BaseCNN(
            **ckpt_cfg.model.predictor
        )  # oracle has same architecture as predictor
        self._cnn_oracle.load_state_dict({
            k.replace('predictor.', ''): v
            for k, v in oracle_state_dict['state_dict'].items()
        })
        self._cnn_oracle = self._cnn_oracle.to(self.device)
        self._cnn_oracle.eval()
        if self._cfg.predictor_dir is not None:
            predictor_path = os.path.join(self._cfg.predictor_dir, 'last.ckpt')
            predictor_state_dict = torch.load(predictor_path,
                                              map_location=self.device)
            self._predictor = BaseCNN(
                **ckpt_cfg.model.predictor
            )  # oracle has same architecture as predictor
            self._predictor.load_state_dict({
                k.replace('predictor.', ''): v
                for k, v in predictor_state_dict['state_dict'].items()
            })
            self._predictor = self._predictor.to(self.device)
        self.run_oracle = self._run_cnn_oracle
        self.run_predictor = self._run_predictor if self._cfg.predictor_dir is not None else None

    def novelty(self, sampled_seqs):
        # sampled_seqs: top k
        # existing_seqs: range dataset
        all_novelty = []
        for src in tqdm(sampled_seqs):
            min_dist = 1e9
            for known in self._base_pool_seqs:
                dist = levenshtein(src, known)
                if dist < min_dist:
                    min_dist = dist
            all_novelty.append(min_dist)
        return all_novelty

    def tokenize(self, seqs):
        return self.predictor_tokenizer.encode(seqs).to(self.device)

    def _run_cnn_oracle(self, seqs):
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self._cfg.batch_size, 0)
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self._cnn_oracle(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0)

    def _run_predictor(self, seqs):
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self._cfg.batch_size, 0)
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self._predictor(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0)

    def filter_duplicate(self, scores, seqs):
        unq_seqs, ids = np.unique(seqs, axis=0, return_index=True)
        unq_seqs = unq_seqs.tolist()
        unq_scores = scores[ids]
        return unq_seqs, unq_scores

    def evaluate_sequences(self, pred_scores, topk_seqs, use_oracle=True):
        pred_scores = to_np(pred_scores)
        topk_seqs, topk_pred_scores = self.filter_duplicate(pred_scores, topk_seqs)
        topk_seqs = topk_seqs[:128]
        topk_pred_scores = topk_pred_scores[:128]
        # pred_scores = to_list(pred_scores)
        # topk_seqs = list(set(topk_seqs))
        num_unique_seqs = len(topk_seqs)
        topk_scores = self.run_oracle(
            topk_seqs) if use_oracle else self.run_predictor(topk_seqs)
        normalized_scores = [self.normalize(x) for x in topk_scores]
        seq_novelty = self.novelty(topk_seqs)
        results_df = pd.DataFrame({
            'sequence': topk_seqs,
            'predictor_score': topk_pred_scores.tolist(),
            'oracle_score': to_list(topk_scores),
            'normalized_score': normalized_scores,
            'novelty': seq_novelty,
        }) if use_oracle else pd.DataFrame(
            {
                'sequence': topk_seqs,
                'predictor_score': to_list(topk_pred_scores),
                'normalized_score': normalized_scores,
                'novelty': seq_novelty,
            })
        if use_oracle:
            results_df.sort_values(
                by="oracle_score", ascending=False, inplace=True, ignore_index=True
            )

        if num_unique_seqs == 1:
            seq_diversity = 0
        else:
            seq_diversity = diversity(topk_seqs)

        metrics_scores = normalized_scores if self.use_normalization \
            else topk_scores.detach().cpu().numpy()
        metrics_df = pd.DataFrame({
            'num_unique': [num_unique_seqs],
            'mean_fitness': [np.mean(metrics_scores)],
            'median_fitness': [np.median(metrics_scores)],
            'std_fitness': [np.std(metrics_scores)],
            'max_fitness': [np.max(metrics_scores)],
            'mean_diversity': [seq_diversity],
            'mean_novelty': [np.mean(seq_novelty)],
            'median_novelty': [np.median(seq_novelty)],
        })
        return results_df, metrics_df
