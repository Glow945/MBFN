from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.mbfn_core import (
    LossWeights, MBFN, MBFNLoss, compute_metrics, load_feature_list,
    make_loaders, save_json, save_preprocess_stats, set_seed, to_device,
)


@torch.no_grad()
def evaluate_model(model, loader, device):
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
    parser.add_argument("--data1", required=True)
    parser.add_argument("--data2", required=True)
    parser.add_argument("--common_features", required=True)
    parser.add_argument("--specific_features", required=True)
    parser.add_argument("--num_types", type=int, required=True)
    parser.add_argument("--num_stages", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lambda_align", type=float, default=0.10)
    parser.add_argument("--lambda_domain", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_features = load_feature_list(args.common_features)
    specific_features = load_feature_list(args.specific_features)
    loaders, stats, _ = make_loaders(args.data1, args.data2, common_features, specific_features, args.batch_size, args.seed)
    save_preprocess_stats(stats, out_dir / "preprocess_stats.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MBFN(
        len(common_features), len(specific_features), args.num_types, args.num_stages,
        token_dim=args.token_dim, latent_dim=args.latent_dim, heads=args.heads, dropout=args.dropout,
    ).to(device)
    criterion = MBFNLoss(LossWeights(domain=args.lambda_domain, align=args.lambda_align))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_score, wait, history = -np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}", leave=False):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["common"], batch["specific"], batch["dataset_id"])
            loss_dict = criterion(out, batch)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss_dict["loss"].detach().cpu()))
        scheduler.step()
        val = evaluate_model(model, loaders["val"], device)
        score = np.nanmean([val.get("det_auc", np.nan), val.get("type_f1_macro", np.nan), val.get("stage_f1_macro", np.nan)])
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        print(rec)
        if score > best_score:
            best_score, wait = score, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args), "common_features": common_features, "specific_features": specific_features}, out_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    save_json({"history": history}, out_dir / "history.json")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate_model(model, loaders["test"], device)
    save_json(test, out_dir / "test_metrics.json")
    print("Test metrics:", test)


if __name__ == "__main__":
    main()
