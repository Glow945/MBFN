# MBFN Core Code

Core implementation for **MBFN: Multi-modal Biomarker Fusion Network for Cross-Dataset Cancer Detection and Classification**.

Implemented modules:

- heterogeneous biomarker CSV loading and preprocessing;
- common/specific biomarker feature handling;
- Biomarker Alignment Network with moment-matching alignment loss;
- adversarial domain adaptation with gradient reversal;
- attention-guided feature selection over biomarker tokens;
- hierarchical multi-task heads for detection, type classification, and stage prediction;
- evaluation metrics and optional attention/SHAP-style interpretation hooks.

## Environment

```bash
conda create -n mbfn python=3.10 -y
conda activate mbfn
pip install -r requirements.txt
```

## Data format

Two CSV files are expected:

```text
sample_id,dataset_id,detect_label,type_label,stage_label,<biomarker columns...>
```

Label conventions:

- `dataset_id`: 0 for Dataset 1, 1 for Dataset 2.
- `detect_label`: 0 normal, 1 cancer.
- `type_label`: integer cancer-type label; use -1 for normal or unknown.
- `stage_label`: integer stage label; use -1 for normal or unknown.
- Biomarker names should match `config/common_biomarkers.txt` and `config/specific_biomarkers.txt`.
- Missing feature columns are filled automatically.

## Quick test with toy data

```bash
python scripts/prepare_dummy_data.py --out data/demo
python train.py --data1 data/demo/dataset1.csv --data2 data/demo/dataset2.csv \
  --common_features config/common_biomarkers.txt \
  --specific_features config/specific_biomarkers.txt \
  --num_types 8 --num_stages 3 --epochs 5 --out outputs/demo_run
python evaluate.py --checkpoint outputs/demo_run/best_model.pt \
  --data1 data/demo/dataset1.csv --data2 data/demo/dataset2.csv \
  --common_features config/common_biomarkers.txt \
  --specific_features config/specific_biomarkers.txt \
  --num_types 8 --num_stages 3 --split test
```

`prepare_dummy_data.py` is only for code sanity checking. Replace it with real biomarker data for actual experiments.

## Main files

```text
src/mbfn_core.py              MBFN model, preprocessing, loss, metrics
train.py                      training entry point
evaluate.py                   evaluation entry point
interpret_attention.py         exports attention/gate weights
scripts/prepare_dummy_data.py  small toy dataset generator
```
