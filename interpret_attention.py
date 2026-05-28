from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.mbfn_core import (
    BiomarkerDataset, MBFN, load_and_merge_data, load_feature_list,
    load_preprocess_stats, split_dataframe, to_device,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data1", required=True)
    parser.add_argument("--data2", required=True)
    parser.add_argument("--common_features", required=True)
    parser.add_argument("--specific_features", required=True)
    parser.add_argument("--num_types", type=int, required=True)
    parser.add_argument("--num_stages", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    model.eval()
    stats = load_preprocess_stats(Path(args.checkpoint).parent / "preprocess_stats.pkl")
    df = load_and_merge_data(args.data1, args.data2, common_features, specific_features)
    splits = split_dataframe(df, seed=model_args.get("seed", 42))
    dataset = BiomarkerDataset(splits["test"], common_features, specific_features, stats)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    cg, sg, n = None, None, 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["common"], batch["specific"], batch["dataset_id"])
        c = out["common_gates"].mean(dim=0).detach().cpu()
        s = out["specific_gates"].mean(dim=0).detach().cpu()
        cg = c if cg is None else cg + c
        sg = s if sg is None else sg + s
        n += 1
    rows = []
    for name, val in zip(common_features, (cg / max(n, 1)).numpy()):
        rows.append({"feature": name, "group": "common", "mean_gate_weight": float(val)})
    for name, val in zip(specific_features, (sg / max(n, 1)).numpy()):
        rows.append({"feature": name, "group": "specific", "mean_gate_weight": float(val)})
    pd.DataFrame(rows).sort_values("mean_gate_weight", ascending=False).to_csv(out_dir / "attention_gate_weights.csv", index=False)
    print(f"Saved {out_dir / 'attention_gate_weights.csv'}")


if __name__ == "__main__":
    main()
