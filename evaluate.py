from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.mbfn_core import (
    BiomarkerDataset, MBFN, compute_metrics, load_and_merge_data, load_feature_list,
    load_preprocess_stats, split_dataframe, to_device, save_json,
)


@torch.no_grad()
def collect_metrics(model, loader, device):
    model.eval()
    y_det, p_det, y_type, pred_type, y_stage, pred_stage = [], [], [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["common"], batch["specific"], batch["dataset_id"])
        y_det.extend(batch["detect_label"].cpu().numpy().tolist())
        p_det.extend(torch.softmax(out["det_logits"], dim=-1)[:, 1].cpu().numpy().tolist())
        y_type.extend(batch["type_label"].cpu().numpy().tolist())
        pred_type.extend(out["type_logits"].argmax(dim=-1).cpu().numpy().tolist())
        y_stage.extend(batch["stage_label"].cpu().numpy().tolist())
        pred_stage.extend(out["stage_logits"].argmax(dim=-1).cpu().numpy().tolist())
    return compute_metrics(y_det, p_det, y_type, pred_type, y_stage, pred_stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data1", required=True)
    parser.add_argument("--data2", required=True)
    parser.add_argument("--common_features", required=True)
    parser.add_argument("--specific_features", required=True)
    parser.add_argument("--num_types", type=int, required=True)
    parser.add_argument("--num_stages", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get("args", {})
    common_features = load_feature_list(args.common_features)
    specific_features = load_feature_list(args.specific_features)
    model = MBFN(
        len(common_features), len(specific_features), args.num_types, args.num_stages,
        token_dim=model_args.get("token_dim", 64), latent_dim=model_args.get("latent_dim", 128),
        heads=model_args.get("heads", 4), dropout=model_args.get("dropout", 0.10),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    stats = load_preprocess_stats(Path(args.checkpoint).parent / "preprocess_stats.pkl")
    df = load_and_merge_data(args.data1, args.data2, common_features, specific_features)
    splits = split_dataframe(df, seed=model_args.get("seed", 42))
    use_df = df if args.split == "all" else splits[args.split]
    dataset = BiomarkerDataset(use_df, common_features, specific_features, stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = collect_metrics(model, loader, device)
    print(metrics)
    if args.out:
        save_json(metrics, args.out)


if __name__ == "__main__":
    main()
