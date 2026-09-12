# SCA Gas and Chemical-Family Classification

Experimental Python code accompanying a master's thesis on classifying gases and
their chemical families from Smart Cable Air (SCA) sensor measurements.

The project evaluates gas recognition and gas-family consistency with several
supervised-learning configurations:

- multi-task MLPs for simultaneous gas and family prediction;
- separated and joint cascades with hard or soft family representations;
- oracle configurations that use the true family as an upper-bound reference;
- Random Forest baselines;
- raw, rolling-window and statistical time-series features.

## Repository contents

- `multi_class.py`: baseline multi-task MLP, utilities and evaluation pipeline.
- `multi_class_window_stats.py`: MLP and Random Forest experiments using
  statistical features.
- `multi_class_sep_cas.py` and `multi_class_joint_cas.py`: separated and joint
  cascade models.
- `multi_class_hard_oracle.py` and `multi_class_oracle_scan.py`: oracle models.
- `multi_label_mlp.py`: multi-label MLP experiment.
- `feature_engineering.py`: self-contained SCA time-series feature extraction.
- `cv10_splits_train_valid.json`: predefined cross-validation partitions.

## Setup

Use Python 3.9 or newer:

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

## Data

The raw SCA measurements are intentionally not included. Place the dataset in a
separate directory and pass it with `--data-root`; expected CSV paths are
configured by `--path-template` (default: `{class_name}/{file}.csv`).

## Example

Run the multi-task MLP with the provided ten-fold partitions:

```bash
python multi_class.py --splits cv10_splits_train_valid.json --data-root ../data1 --sep ";" --epochs 100 --batch-size 64 --lr 0.001 --alpha-family 1.0 --save-report results/report.csv --save-heatmap --heatmap-dir results/heatmaps
```

The output directories are ignored by Git. Reported performance depends on the
dataset version, split configuration and training environment.

## Notes

The thesis compares raw sensor signals with statistical descriptors extracted
from temporal windows. It reports that statistical features and Random Forest
models are particularly effective within the evaluated configurations, while
external-test performance highlights a distribution shift between acquisitions.
