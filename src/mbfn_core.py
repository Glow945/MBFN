from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

REQUIRED_COLUMNS = ["sample_id", "dataset_id", "detect_label", "type_label", "stage_label"]


def load_feature_list(path: str | Path) -> List[str]:
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@dataclass
class PreprocessStats:
    common_median: np.ndarray
    common_mean: np.ndarray
    common_std: np.ndarray
    specific_median: np.ndarray
    specific_mean: np.ndarray
    specific_std: np.ndarray


def _ensure_columns(df: pd.DataFrame, cols: List[str], fill_value=np.nan) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def load_and_merge_data(data1: str | Path, data2: str | Path, common_features: List[str], specific_features: List[str]) -> pd.DataFrame:
    df1, df2 = pd.read_csv(data1), pd.read_csv(data2)
    if "dataset_id" not in df1.columns:
        df1["dataset_id"] = 0
    if "dataset_id" not in df2.columns:
        df2["dataset_id"] = 1
    all_features = common_features + specific_features
    df1 = _ensure_columns(df1, REQUIRED_COLUMNS + all_features)
    df2 = _ensure_columns(df2, REQUIRED_COLUMNS + all_features)
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    for c in all_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["dataset_id", "detect_label", "type_label", "stage_label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    df.loc[df["dataset_id"] == 0, specific_features] = df.loc[df["dataset_id"] == 0, specific_features].fillna(0.0)
    return df


def _log1p_nonnegative(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def fit_preprocess_stats(df: pd.DataFrame, common_features: List[str], specific_features: List[str]) -> PreprocessStats:
    common = _log1p_nonnegative(df[common_features].to_numpy(dtype=np.float32))
    specific = _log1p_nonnegative(df[specific_features].to_numpy(dtype=np.float32))
    cm = np.nanmedian(common, axis=0)
    sm = np.nanmedian(specific, axis=0)
    common = np.where(np.isnan(common), cm[None, :], common)
    specific = np.where(np.isnan(specific), sm[None, :], specific)
    return PreprocessStats(
        cm, common.mean(axis=0), common.std(axis=0) + 1e-6,
        sm, specific.mean(axis=0), specific.std(axis=0) + 1e-6,
    )


def apply_preprocess(df: pd.DataFrame, common_features: List[str], specific_features: List[str], stats: PreprocessStats) -> Tuple[np.ndarray, np.ndarray]:
    common = _log1p_nonnegative(df[common_features].to_numpy(dtype=np.float32))
    specific = _log1p_nonnegative(df[specific_features].to_numpy(dtype=np.float32))
    common = np.where(np.isnan(common), stats.common_median[None, :], common)
    specific = np.where(np.isnan(specific), stats.specific_median[None, :], specific)
    common = (common - stats.common_mean[None, :]) / stats.common_std[None, :]
    specific = (specific - stats.specific_mean[None, :]) / stats.specific_std[None, :]
    return common.astype(np.float32), specific.astype(np.float32)


class BiomarkerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, common_features: List[str], specific_features: List[str], stats: PreprocessStats):
        self.df = df.reset_index(drop=True).copy()
        self.common, self.specific = apply_preprocess(self.df, common_features, specific_features, stats)
        self.dataset_id = self.df["dataset_id"].astype(int).to_numpy()
        self.detect_label = self.df["detect_label"].astype(int).to_numpy()
        self.type_label = self.df["type_label"].astype(int).to_numpy()
        self.stage_label = self.df["stage_label"].astype(int).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "common": torch.tensor(self.common[idx], dtype=torch.float32),
            "specific": torch.tensor(self.specific[idx], dtype=torch.float32),
            "dataset_id": torch.tensor(self.dataset_id[idx], dtype=torch.long),
            "detect_label": torch.tensor(self.detect_label[idx], dtype=torch.long),
            "type_label": torch.tensor(self.type_label[idx], dtype=torch.long),
            "stage_label": torch.tensor(self.stage_label[idx], dtype=torch.long),
        }


def split_dataframe(df: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    strat = (df["dataset_id"].astype(str) + "_" + df["detect_label"].astype(str)).to_numpy()
    try:
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=strat)
        strat_temp = (temp_df["dataset_id"].astype(str) + "_" + temp_df["detect_label"].astype(str)).to_numpy()
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=strat_temp)
    except ValueError:
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)
    return {"train": train_df.reset_index(drop=True), "val": val_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


def make_loaders(data1: str | Path, data2: str | Path, common_features: List[str], specific_features: List[str], batch_size: int = 64, seed: int = 42):
    df = load_and_merge_data(data1, data2, common_features, specific_features)
    splits = split_dataframe(df, seed)
    stats = fit_preprocess_stats(splits["train"], common_features, specific_features)
    datasets = {k: BiomarkerDataset(v, common_features, specific_features, stats) for k, v in splits.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
    return loaders, stats, splits


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor):
        return GradientReversalFunction.apply(x, self.lambd)


def moment_alignment_loss(z: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
    z0, z1 = z[dataset_id == 0], z[dataset_id == 1]
    if z0.size(0) < 2 or z1.size(0) < 2:
        return z.new_tensor(0.0)
    mean_loss = torch.mean((z0.mean(0) - z1.mean(0)) ** 2)
    def cov(x):
        x = x - x.mean(0, keepdim=True)
        return (x.T @ x) / max(x.size(0) - 1, 1)
    cov_loss = torch.mean((cov(z0) - cov(z1)) ** 2)
    return mean_loss + cov_loss


class BiomarkerTokenizer(nn.Module):
    def __init__(self, num_features: int, token_dim: int = 64):
        super().__init__()
        self.scalar_proj = nn.Linear(1, token_dim)
        self.feature_embed = nn.Embedding(max(num_features, 1), token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, nfeat = x.shape
        ids = torch.arange(nfeat, device=x.device).unsqueeze(0).expand(bsz, -1)
        return self.norm(self.scalar_proj(x.unsqueeze(-1)) + self.feature_embed(ids))


class AttentionFeatureSelector(nn.Module):
    def __init__(self, token_dim: int = 64, num_heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.attn = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor):
        attn_out, attn_weights = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
        attn_out = self.norm(attn_out + tokens)
        gates = self.gate(attn_out)
        pooled = (attn_out * gates).mean(dim=1)
        return pooled, gates.squeeze(-1), attn_weights


class BiomarkerAlignmentNetwork(nn.Module):
    def __init__(self, n_common: int, n_specific: int, token_dim: int = 64, latent_dim: int = 128, heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.common_tok = BiomarkerTokenizer(n_common, token_dim)
        self.specific_tok = BiomarkerTokenizer(max(n_specific, 1), token_dim)
        self.common_selector = AttentionFeatureSelector(token_dim, heads, dropout)
        self.specific_selector = AttentionFeatureSelector(token_dim, heads, dropout)
        self.common_proj = nn.Sequential(nn.Linear(token_dim, latent_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(latent_dim))
        self.specific_proj = nn.Sequential(nn.Linear(token_dim, latent_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(latent_dim))
        self.fusion = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(latent_dim))

    def forward(self, common: torch.Tensor, specific: torch.Tensor, dataset_id: torch.Tensor):
        cp, cg, ca = self.common_selector(self.common_tok(common))
        zp = self.common_proj(cp)
        if specific.size(1) == 0:
            specific = torch.zeros(common.size(0), 1, device=common.device, dtype=common.dtype)
        sp, sg, sa = self.specific_selector(self.specific_tok(specific))
        zs = self.specific_proj(sp) * (dataset_id == 1).float().unsqueeze(-1)
        z = self.fusion(torch.cat([zp, zs], dim=-1))
        return {"z": z, "common_gates": cg, "specific_gates": sg, "common_attn": ca, "specific_attn": sa}


class MBFN(nn.Module):
    def __init__(self, n_common: int, n_specific: int, num_types: int, num_stages: int, token_dim: int = 64, latent_dim: int = 128, heads: int = 4, dropout: float = 0.10, grl_lambda: float = 1.0):
        super().__init__()
        self.alignment = BiomarkerAlignmentNetwork(n_common, n_specific, token_dim, latent_dim, heads, dropout)
        self.det_head = nn.Sequential(nn.Linear(latent_dim, latent_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim // 2, 2))
        self.type_head = nn.Sequential(nn.Linear(latent_dim + 2, latent_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim // 2, num_types))
        self.stage_head = nn.Sequential(nn.Linear(latent_dim + 2 + num_types, latent_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim // 2, num_stages))
        self.grl = GradientReversal(grl_lambda)
        self.domain_head = nn.Sequential(nn.Linear(latent_dim, latent_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim // 2, 2))

    def forward(self, common: torch.Tensor, specific: torch.Tensor, dataset_id: torch.Tensor):
        a = self.alignment(common, specific, dataset_id)
        z = a["z"]
        det_logits = self.det_head(z)
        det_prob = F.softmax(det_logits, dim=-1)
        type_logits = self.type_head(torch.cat([z, det_prob], dim=-1))
        type_prob = F.softmax(type_logits, dim=-1)
        stage_logits = self.stage_head(torch.cat([z, det_prob, type_prob], dim=-1))
        domain_logits = self.domain_head(self.grl(z))
        return {"z": z, "det_logits": det_logits, "type_logits": type_logits, "stage_logits": stage_logits, "domain_logits": domain_logits, **a}


@dataclass
class LossWeights:
    detect: float = 1.0
    type: float = 0.8
    stage: float = 0.6
    domain: float = 0.05
    align: float = 0.10


class MBFNLoss(nn.Module):
    def __init__(self, weights: LossWeights):
        super().__init__()
        self.w = weights
        self.ce = nn.CrossEntropyLoss()
        self.ce_ignore = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        ld = self.ce(outputs["det_logits"], batch["detect_label"])
        lt = self.ce_ignore(outputs["type_logits"], batch["type_label"])
        ls = self.ce_ignore(outputs["stage_logits"], batch["stage_label"])
        ldom = self.ce(outputs["domain_logits"], batch["dataset_id"])
        lalign = moment_alignment_loss(outputs["z"], batch["dataset_id"])
        total = self.w.detect * ld + self.w.type * lt + self.w.stage * ls + self.w.domain * ldom + self.w.align * lalign
        return {"loss": total, "loss_detect": ld.detach(), "loss_type": lt.detach(), "loss_stage": ls.detach(), "loss_domain": ldom.detach(), "loss_align": lalign.detach()}


def compute_metrics(y_det, p_det, y_type, pred_type, y_stage, pred_stage):
    y_det, p_det = np.asarray(y_det), np.asarray(p_det)
    pred_det = (p_det >= 0.5).astype(int)
    out = {"det_acc": float(accuracy_score(y_det, pred_det)), "det_f1_macro": float(f1_score(y_det, pred_det, average="macro", zero_division=0))}
    try:
        out["det_auc"] = float(roc_auc_score(y_det, p_det))
    except ValueError:
        out["det_auc"] = float("nan")
    y_type, pred_type = np.asarray(y_type), np.asarray(pred_type)
    mask = y_type >= 0
    out["type_acc"] = float(accuracy_score(y_type[mask], pred_type[mask])) if mask.any() else float("nan")
    out["type_f1_macro"] = float(f1_score(y_type[mask], pred_type[mask], average="macro", zero_division=0)) if mask.any() else float("nan")
    y_stage, pred_stage = np.asarray(y_stage), np.asarray(pred_stage)
    mask = y_stage >= 0
    out["stage_acc"] = float(accuracy_score(y_stage[mask], pred_stage[mask])) if mask.any() else float("nan")
    out["stage_f1_macro"] = float(f1_score(y_stage[mask], pred_stage[mask], average="macro", zero_division=0)) if mask.any() else float("nan")
    return out


def save_preprocess_stats(stats: PreprocessStats, path: str | Path):
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_preprocess_stats(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
