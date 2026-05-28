from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COMMON = ["CA125", "CEA", "CA19_9", "Prolactin", "HGF", "OPN", "Myeloperoxidase", "TIMP_1", "OmegaScore", "Age", "Sex"]
SPECIFIC = ["AFP", "Angiopoietin_2", "AXL", "CA15_3", "CD44", "CYFRA21_1", "DKK1", "Endoglin", "FGF2", "Follistatin", "Galectin_3", "G_CSF", "GDF15", "HE4", "IL_6", "IL_8", "Kallikrein_6", "Leptin", "Mesothelin", "Midkine", "NSE", "OPG", "PAR", "sEGFR", "sFas", "SHBG", "sHER2", "sPECAM_1", "TGFa", "Thrombospondin_2", "TIMP_2"]


def make_dataset(n: int, dataset_id: int, has_specific: bool, seed: int):
    rng = np.random.default_rng(seed)
    detect = rng.binomial(1, 0.55 if dataset_id == 0 else 1.0, n)
    type_label = np.where(detect == 1, rng.integers(0, 8, n), -1)
    stage_label = np.where(detect == 1, rng.integers(0, 3, n), -1)
    data = {
        "sample_id": [f"DS{dataset_id}_{i:05d}" for i in range(n)],
        "dataset_id": np.full(n, dataset_id),
        "detect_label": detect,
        "type_label": type_label,
        "stage_label": stage_label,
    }
    for f in COMMON:
        base = rng.lognormal(mean=1.0, sigma=0.6, size=n)
        shift = detect * rng.lognormal(mean=0.4, sigma=0.3, size=n)
        data[f] = base + shift
    if has_specific:
        for f in SPECIFIC:
            base = rng.lognormal(mean=0.8, sigma=0.7, size=n)
            shift = detect * rng.lognormal(mean=0.25, sigma=0.2, size=n)
            data[f] = base + shift
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make_dataset(300, 0, False, 7).to_csv(out / "dataset1.csv", index=False)
    make_dataset(180, 1, True, 9).to_csv(out / "dataset2.csv", index=False)
    print(f"Saved toy data to {out}")


if __name__ == "__main__":
    main()
