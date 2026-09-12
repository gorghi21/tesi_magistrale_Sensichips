# HARD CASCADE
# python .\multi_class_sep_cas.py --splits .\cv10_splits_train_valid.json --data-root ..\data1 --sep ';' --epochs 100 --batch-size 64 --lr 0.001 --cascade-input hard --loss-mode standard --oof-folds 5 --save-report .\hard_standard_report.csv
# python .\multi_class_sep_cas.py --splits .\cv10_splits_train_valid.json --data-root ..\data1 --sep ';' --epochs 100 --batch-size 64 --lr 0.001 --cascade-input hard --loss-mode hierarchical --lambda-hier 0.5 --hier-beta 0.3 --hier-gamma 3.0 --oof-folds 5 --save-report .\hard_hierarchical_report.csv
# python .\multi_class_sep_cas.py --splits .\cv10_splits_train_valid.json --data-root ..\data1 --sep ';' --epochs 100 --batch-size 64 --lr 0.001 --cascade-input hard --loss-mode focal --focal-gamma 2.0 --oof-folds 5 --save-report .\hard_focal_report.csv

# SOFT CASCADE
# python .\multi_class_sep_cas.py --splits .\cv10_splits_train_valid.json --data-root ..\data1 --sep ';' --epochs 100 --batch-size 64 --lr 0.001 --cascade-input soft --loss-mode standard --oof-folds 5 --save-report .\soft_standard_report.csv
# python .\multi_class_sep_cas.py --splits .\cv10_splits_train_valid.json --data-root ..\data1 --sep ';' --epochs 100 --batch-size 64 --lr 0.001 --cascade-input soft --loss-mode hierarchical --lambda-hier 0.5 --hier-beta 0.3 --hier-gamma 3.0 --oof-folds 5 --save-report .\soft_hierarchical_report.csv
# python .\multi_class_sep_cas.py --splits .\cv10_splits_train_valid.json --data-root ..\data1 --sep ';' --epochs 100 --batch-size 64 --lr 0.001 --cascade-input soft --loss-mode focal --focal-gamma 2.0 --oof-folds 5 --save-report .\soft_focal_report.csv

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from multi_class import (
    GAS_TO_FAMILY,
    FocalLoss,
    TabularGasDataset,
    align_columns,
    build_gas_to_family_indices,
    build_listing_from_root,
    consistency_accuracy,
    hierarchical_consistency_penalty,
    map_gas_to_family,
    normalized_confusion_matrix,
    prepare_xy,
    save_heatmap,
    scale_features,
    set_seed,
)
from multi_class_window_stats import (
    limit_listing,
    prepare_xy_window_stats,
    sanitize_feature_frame,
)


class FamilyMLP(nn.Module):
    """
    Primo stadio della cascata:
    X -> famiglia chimica.
    """

    def __init__(self, input_dim: int, num_family: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_family),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GasCascadeMLP(nn.Module):
    """
    Secondo stadio della cascata:
    [X, famiglia_predetta] -> gas.
    """

    def __init__(self, input_dim: int, num_family: int, num_gas: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_family, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_gas),
        )

    def forward(self, x: torch.Tensor, family_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, family_features], dim=1))


class CascadeGasDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        family_features: np.ndarray,
        y_gas: np.ndarray,
        y_family: np.ndarray,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.family_features = torch.tensor(family_features, dtype=torch.float32)
        self.y_gas = torch.tensor(y_gas, dtype=torch.long)
        self.y_family = torch.tensor(y_family, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.family_features[idx],
            self.y_gas[idx],
            self.y_family[idx],
        )


def family_features_from_logits(
    family_logits: torch.Tensor,
    cascade_input: str,
    num_family: int,
) -> torch.Tensor:
    if cascade_input == "soft":
        return torch.softmax(family_logits, dim=1)

    pred_family = torch.argmax(family_logits, dim=1)
    return nn.functional.one_hot(pred_family, num_classes=num_family).float()


def train_family_one_epoch(
    family_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    family_model.train()
    total_loss = 0.0
    total_samples = 0

    for X, _, y_family in loader:
        X = X.to(device)
        y_family = y_family.to(device)

        optimizer.zero_grad()
        family_logits = family_model(X)
        loss = criterion(family_logits, y_family)
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_family_model(
    X: np.ndarray,
    y_gas: np.ndarray,
    y_family: np.ndarray,
    input_dim: int,
    num_family: int,
    args: argparse.Namespace,
    criterion: nn.Module,
    device: torch.device,
    stage_name: str,
) -> Tuple[FamilyMLP, int]:
    dataset = TabularGasDataset(X, y_gas, y_family)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    family_model = FamilyMLP(
        input_dim=input_dim,
        num_family=num_family,
    ).to(device)
    optimizer = torch.optim.Adam(
        family_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_family_one_epoch(
            family_model=family_model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        epochs_run = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"{stage_name} epoch {epoch:03d}/{args.epochs} - loss={train_loss:.6f}")

        if args.n_iter_no_change > 0:
            if train_loss < best_loss - args.tol:
                best_loss = train_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.n_iter_no_change:
                print(
                    f"Stopping {stage_name} at epoch {epoch}: training loss did not improve "
                    f"by at least tol={args.tol} for "
                    f"{args.n_iter_no_change} consecutive epochs."
                )
                break

    return family_model, epochs_run


@torch.no_grad()
def predict_family_features(
    family_model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    cascade_input: str,
    num_family: int,
) -> Tuple[np.ndarray, np.ndarray]:
    family_model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
    features = []
    preds = []

    for X_batch in loader:
        X_batch = X_batch.to(device)
        family_logits = family_model(X_batch)
        family_features = family_features_from_logits(
            family_logits,
            cascade_input=cascade_input,
            num_family=num_family,
        )
        features.append(family_features.cpu().numpy())
        preds.append(torch.argmax(family_logits, dim=1).cpu().numpy())

    return np.concatenate(features), np.concatenate(preds)


def rf_family_features(
    family_model: RandomForestClassifier,
    X: np.ndarray,
    num_family: int,
    cascade_input: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if cascade_input == "soft":
        proba = family_model.predict_proba(X)
        features = np.zeros((X.shape[0], num_family), dtype=np.float32)
        for column_idx, class_idx in enumerate(family_model.classes_):
            features[:, int(class_idx)] = proba[:, column_idx]
        pred_family = np.argmax(features, axis=1)
        return features, pred_family

    pred_family = family_model.predict(X).astype(int)
    features = np.eye(num_family, dtype=np.float32)[pred_family]
    return features, pred_family


def oof_family_features(
    X_train: np.ndarray,
    y_train_gas: np.ndarray,
    y_train_family: np.ndarray,
    input_dim: int,
    num_family: int,
    args: argparse.Namespace,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int]:
    counts = np.bincount(y_train_family, minlength=num_family)
    min_count = int(counts[counts > 0].min())
    effective_folds = min(args.oof_folds, min_count)

    if effective_folds < 2:
        print("Warning: not enough samples per family for OOF. Falling back to in-sample family predictions.")
        family_model, epochs_run = train_family_model(
            X=X_train,
            y_gas=y_train_gas,
            y_family=y_train_family,
            input_dim=input_dim,
            num_family=num_family,
            args=args,
            criterion=criterion,
            device=device,
            stage_name="Family fallback",
        )
        features, preds = predict_family_features(
            family_model,
            X_train,
            batch_size=args.batch_size,
            device=device,
            cascade_input=args.cascade_input,
            num_family=num_family,
        )
        return features, preds, epochs_run

    print(f"Stage 1/3: generating OOF family predictions ({effective_folds} folds)")
    oof_features = np.zeros((len(X_train), num_family), dtype=np.float32)
    oof_preds = np.zeros(len(X_train), dtype=np.int64)
    total_epochs = 0

    skf = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=args.seed,
    )

    for fold_idx, (train_idx, valid_idx) in enumerate(
        skf.split(X_train, y_train_family),
        start=1,
    ):
        print(f"OOF family fold {fold_idx}/{effective_folds}")
        family_model, epochs_run = train_family_model(
            X=X_train[train_idx],
            y_gas=y_train_gas[train_idx],
            y_family=y_train_family[train_idx],
            input_dim=input_dim,
            num_family=num_family,
            args=args,
            criterion=criterion,
            device=device,
            stage_name=f"OOF family fold {fold_idx}",
        )
        total_epochs += epochs_run

        fold_features, fold_preds = predict_family_features(
            family_model,
            X_train[valid_idx],
            batch_size=args.batch_size,
            device=device,
            cascade_input=args.cascade_input,
            num_family=num_family,
        )
        oof_features[valid_idx] = fold_features
        oof_preds[valid_idx] = fold_preds

    return oof_features, oof_preds, total_epochs


def train_gas_one_epoch(
    gas_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_hier: float,
    hier_beta: float,
    hier_gamma: float,
    gas_to_family_idx: torch.Tensor,
) -> float:
    gas_model.train()
    total_loss = 0.0
    total_samples = 0

    for X, family_features, y_gas, _ in loader:
        X = X.to(device)
        family_features = family_features.to(device)
        y_gas = y_gas.to(device)

        optimizer.zero_grad()

        gas_logits = gas_model(X, family_features)
        loss_gas = criterion(gas_logits, y_gas)
        loss = loss_gas

        if lambda_hier > 0.0:
            loss_hier = hierarchical_consistency_penalty(
                gas_logits=gas_logits,
                family_logits=family_features,
                y_gas=y_gas,
                gas_to_family_idx=gas_to_family_idx,
                beta=hier_beta,
                gamma=hier_gamma,
            )
            loss = loss + lambda_hier * loss_hier

        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    family_model: nn.Module,
    gas_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cascade_input: str,
    num_family: int,
) -> Tuple[np.ndarray, np.ndarray]:
    family_model.eval()
    gas_model.eval()
    gas_preds = []
    family_preds = []

    for X, _, _ in loader:
        X = X.to(device)
        family_logits = family_model(X)
        family_features = family_features_from_logits(
            family_logits,
            cascade_input=cascade_input,
            num_family=num_family,
        )
        gas_logits = gas_model(X, family_features)

        gas_preds.append(torch.argmax(gas_logits, dim=1).cpu().numpy())
        family_preds.append(torch.argmax(family_logits, dim=1).cpu().numpy())

    return np.concatenate(gas_preds), np.concatenate(family_preds)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--splits", default=None)
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--path-template", default="{class_name}/{file}.csv")
    ap.add_argument("--sep", default=None)
    ap.add_argument("--decimal", default=".")
    ap.add_argument("--index-col", default=None, type=str)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--drop-unseen-test-labels", action="store_true")
    ap.add_argument("--max-files-per-class", type=int, default=None)
    ap.add_argument("--rolling-window", type=int, default=1)
    ap.add_argument("--rolling-step", type=int, default=1)
    ap.add_argument(
        "--use-stats-features",
        action="store_true",
        help="Use code_2 statistical features instead of raw/rolling numeric rows.",
    )
    ap.add_argument("--stats-window-size", type=int, default=20)
    ap.add_argument("--stats-step", type=int, default=20)
    ap.add_argument("--stats-n-windows", type=int, default=4)
    ap.add_argument(
        "--aggregate-per-file",
        action="store_true",
        help="With --use-stats-features, compute one statistical vector per CSV.",
    )
    ap.add_argument(
        "--verbose-files",
        action="store_true",
        help="Print every CSV file processed during statistical feature extraction.",
    )
    ap.add_argument(
        "--scaler",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Feature scaling: standard z-score, minmax 0-1, or none.",
    )

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument(
        "--model",
        choices=["mlp", "rf"],
        default="mlp",
        help="mlp: PyTorch separated cascade; rf: RandomForest separated cascade on the same inputs.",
    )
    ap.add_argument("--rf-estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--rf-balanced", action="store_true")
    ap.add_argument(
        "--alpha-family",
        type=float,
        default=1.0,
        help="Kept for CLI compatibility; ignored because this is a true separated cascade.",
    )
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--n-iter-no-change", type=int, default=10)
    ap.add_argument(
        "--loss-mode",
        choices=["standard", "hierarchical", "focal"],
        default="standard",
    )
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--lambda-hier", type=float, default=1.0)
    ap.add_argument("--hier-beta", type=float, default=1.0)
    ap.add_argument("--hier-gamma", type=float, default=2.0)
    ap.add_argument(
        "--cascade-input",
        choices=["hard", "soft"],
        default="hard",
        help=(
            "hard: famiglia predetta con argmax e one-hot; "
            "soft: probabilita softmax della famiglia."
        ),
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--oof-folds",
        type=int,
        default=5,
        help="Number of inner folds used to generate OOF family predictions for gas training.",
    )
    ap.add_argument(
        "--use-oof",
        action="store_true",
        help="Enable OOF family predictions for gas training.",
    )
    ap.add_argument(
        "--no-oof",
        action="store_true",
        help=(
            "Deprecated compatibility flag. OOF is disabled by default."
        ),
    )
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--save-report", default=None)
    ap.add_argument("--save-heatmap", action="store_true")
    ap.add_argument("--heatmap-dir", default=".")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    use_oof = args.use_oof and not args.no_oof

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    index_col = None
    if args.index_col is not None and str(args.index_col).lower() != "none":
        try:
            index_col = int(args.index_col)
        except ValueError:
            index_col = args.index_col

    data_root = Path(args.data_root).resolve()
    test_root = Path(args.test_root).resolve() if args.test_root else None

    if test_root is not None:
        splits = {
            "external_test": {
                "train": build_listing_from_root(data_root, args.max_files_per_class),
                "test": build_listing_from_root(test_root, args.max_files_per_class),
            }
        }
    else:
        if args.splits is None:
            raise ValueError("Provide --splits, or use --test-root for external testing.")
        with open(args.splits, "r", encoding="utf-8") as f:
            splits = json.load(f)

    fold_results = []
    gas_cms = []
    family_cms = []
    gas_cm_labels = None
    family_cm_labels = None
    heatmap_dir = Path(args.heatmap_dir)

    for fold_name, fold in splits.items():
        print(f"\n=== {fold_name} ===")

        train_listing = fold.get("train", {})
        test_listing = fold.get("valid", {}) or fold.get("test", {})

        current_test_root = test_root if test_root is not None else data_root
        if args.use_stats_features:
            train_listing = limit_listing(train_listing, args.max_files_per_class)
            test_listing = limit_listing(test_listing, args.max_files_per_class)
            X_train, y_train_gas = prepare_xy_window_stats(
                train_listing,
                data_root,
                args.path_template,
                sep=args.sep,
                decimal=args.decimal,
                index_col=index_col,
                label_col=args.label_col,
                stats_window_size=args.stats_window_size,
                stats_step=args.stats_step,
                stats_n_windows=args.stats_n_windows,
                aggregate_per_file=args.aggregate_per_file,
                split_name="train",
                verbose_files=args.verbose_files,
            )
            X_test, y_test_gas = prepare_xy_window_stats(
                test_listing,
                current_test_root,
                args.path_template,
                sep=args.sep,
                decimal=args.decimal,
                index_col=index_col,
                label_col=args.label_col,
                stats_window_size=args.stats_window_size,
                stats_step=args.stats_step,
                stats_n_windows=args.stats_n_windows,
                aggregate_per_file=args.aggregate_per_file,
                split_name="test",
                verbose_files=args.verbose_files,
            )
        else:
            X_train, y_train_gas = prepare_xy(
                train_listing,
                data_root,
                args.path_template,
                sep=args.sep,
                decimal=args.decimal,
                index_col=index_col,
                label_col=args.label_col,
                rolling_window=args.rolling_window,
                rolling_step=args.rolling_step,
            )
            X_test, y_test_gas = prepare_xy(
                test_listing,
                current_test_root,
                args.path_template,
                sep=args.sep,
                decimal=args.decimal,
                index_col=index_col,
                label_col=args.label_col,
                rolling_window=args.rolling_window,
                rolling_step=args.rolling_step,
            )

        X_train, X_test = align_columns(X_train, X_test)
        if args.use_stats_features:
            X_train = sanitize_feature_frame(X_train, "train")
            X_test = sanitize_feature_frame(X_test, "test")

        unseen_gas = sorted(set(y_test_gas) - set(y_train_gas))
        if unseen_gas:
            message = f"Test gas labels absent from training: {unseen_gas}"
            if args.drop_unseen_test_labels:
                keep_mask = np.isin(y_test_gas, y_train_gas)
                dropped = int((~keep_mask).sum())
                X_test = X_test.loc[keep_mask].reset_index(drop=True)
                y_test_gas = y_test_gas[keep_mask]
                print(f"Warning: {message}. Dropped {dropped} test rows.")
                if y_test_gas.size == 0:
                    raise ValueError("No test rows left after dropping unseen gas labels.")
            else:
                raise ValueError(
                    f"{message}. Use --drop-unseen-test-labels for closed-set evaluation."
                )

        y_train_family = map_gas_to_family(y_train_gas)
        y_test_family = map_gas_to_family(y_test_gas)

        unseen_family = sorted(set(y_test_family) - set(y_train_family))
        if unseen_family:
            raise ValueError(f"Test family labels absent from training: {unseen_family}")

        gas_encoder = LabelEncoder()
        family_encoder = LabelEncoder()
        gas_encoder.fit(y_train_gas)
        family_encoder.fit(y_train_family)

        y_train_gas_enc = gas_encoder.transform(y_train_gas)
        y_test_gas_enc = gas_encoder.transform(y_test_gas)
        y_train_family_enc = family_encoder.transform(y_train_family)
        y_test_family_enc = family_encoder.transform(y_test_family)

        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train,
            X_test,
            args.scaler,
        )

        train_dataset = TabularGasDataset(
            X_train_scaled,
            y_train_gas_enc,
            y_train_family_enc,
        )
        test_dataset = TabularGasDataset(
            X_test_scaled,
            y_test_gas_enc,
            y_test_family_enc,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        input_dim = X_train_scaled.shape[1]
        num_gas = len(gas_encoder.classes_)
        num_family = len(family_encoder.classes_)

        print(f"Input dim: {input_dim}")
        print(f"Gas classes: {num_gas}")
        print(f"Family classes: {num_family}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        if args.cascade_input == "hard":
            print("Cascade input: hard ([X, predicted family one-hot] -> gas)")
        else:
            print("Cascade input: soft ([X, predicted family probabilities] -> gas)")
        if args.rolling_window > 1:
            print(f"Rolling mean: window={args.rolling_window}, step={args.rolling_step}")
        if args.use_stats_features:
            if args.aggregate_per_file:
                print(f"Stats features: aggregate-per-file, n_windows={args.stats_n_windows}")
            else:
                print(
                    f"Stats features: window_size={args.stats_window_size}, "
                    f"step={args.stats_step}, n_windows={args.stats_n_windows}"
                )
        effective_lambda_hier = (
            args.lambda_hier if args.loss_mode == "hierarchical" else 0.0
        )

        if args.loss_mode == "hierarchical":
            print(
                f"Hierarchical loss: lambda={args.lambda_hier}, "
                f"beta={args.hier_beta}, gamma={args.hier_gamma}"
            )
        elif args.loss_mode == "focal":
            print(f"Loss mode: focal loss (gamma={args.focal_gamma})")
        else:
            print("Loss mode: standard cross-entropy")
        print("Separated training: --alpha-family is ignored.")
        if use_oof:
            print(f"OOF: enabled with {args.oof_folds} inner folds.")
        else:
            print("OOF: disabled. Gas training uses in-sample family predictions.")

        gas_to_family_idx = build_gas_to_family_indices(
            gas_encoder.classes_,
            family_encoder.classes_,
        ).to(device)

        criterion = (
            FocalLoss(gamma=args.focal_gamma)
            if args.loss_mode == "focal"
            else nn.CrossEntropyLoss()
        )

        if args.model == "rf":
            if args.loss_mode != "standard":
                print("Warning: --loss-mode is ignored with --model rf.")
            if use_oof:
                print("Warning: --use-oof is ignored with --model rf.")

            class_weight = "balanced" if args.rf_balanced else None
            print(
                f"RandomForest separated cascade: n_estimators={args.rf_estimators}, "
                f"max_depth={args.rf_max_depth}, min_samples_leaf={args.rf_min_samples_leaf}, "
                f"class_weight={class_weight}"
            )

            family_model = RandomForestClassifier(
                n_estimators=args.rf_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                class_weight=class_weight,
                random_state=args.seed,
                n_jobs=-1,
            )
            family_model.fit(X_train_scaled, y_train_family_enc)

            train_family_features, _ = rf_family_features(
                family_model,
                X_train_scaled,
                num_family,
                args.cascade_input,
            )
            test_family_features, y_pred_family_enc = rf_family_features(
                family_model,
                X_test_scaled,
                num_family,
                args.cascade_input,
            )

            X_train_gas = np.concatenate([X_train_scaled, train_family_features], axis=1)
            X_test_gas = np.concatenate([X_test_scaled, test_family_features], axis=1)

            gas_model = RandomForestClassifier(
                n_estimators=args.rf_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                class_weight=class_weight,
                random_state=args.seed,
                n_jobs=-1,
            )
            gas_model.fit(X_train_gas, y_train_gas_enc)
            y_pred_gas_enc = gas_model.predict(X_test_gas)

            y_pred_gas = gas_encoder.inverse_transform(y_pred_gas_enc)
            y_pred_family = family_encoder.inverse_transform(y_pred_family_enc)

            gas_acc = accuracy_score(y_test_gas, y_pred_gas)
            family_acc = accuracy_score(y_test_family, y_pred_family)
            cascade_consistency_acc = consistency_accuracy(y_pred_gas, y_pred_family)

            print(f"GAS accuracy: {gas_acc:.6f}")
            print(f"FAMILY accuracy: {family_acc:.6f}")
            print(f"CONSISTENCY accuracy: {cascade_consistency_acc:.6f}")

            print("\nGas report")
            print(classification_report(y_test_gas, y_pred_gas, zero_division=0))

            print("\nFamily report")
            print(classification_report(y_test_family, y_pred_family, zero_division=0))

            if args.save_heatmap:
                gas_labels = gas_encoder.classes_.tolist()
                family_labels = family_encoder.classes_.tolist()

                gas_cm = normalized_confusion_matrix(y_test_gas, y_pred_gas, gas_labels)
                family_cm = normalized_confusion_matrix(
                    y_test_family,
                    y_pred_family,
                    family_labels,
                )

                if gas_cm_labels is None:
                    gas_cm_labels = gas_labels
                if family_cm_labels is None:
                    family_cm_labels = family_labels

                if gas_labels == gas_cm_labels:
                    gas_cms.append(gas_cm)
                else:
                    print("Warning: GAS labels differ across folds. Skipping GAS mean heatmap for this fold.")

                if family_labels == family_cm_labels:
                    family_cms.append(family_cm)
                else:
                    print("Warning: FAMILY labels differ across folds. Skipping FAMILY mean heatmap for this fold.")

                gas_heatmap_path = heatmap_dir / f"gas_confusion_heatmap_{fold_name}.png"
                family_heatmap_path = heatmap_dir / f"family_confusion_heatmap_{fold_name}.png"

                save_heatmap(gas_cm, gas_labels, gas_heatmap_path, f"GAS confusion matrix (%) - {fold_name}")
                save_heatmap(
                    family_cm,
                    family_labels,
                    family_heatmap_path,
                    f"FAMILY confusion matrix (%) - {fold_name}",
                )
                print(f"Saved GAS heatmap to: {gas_heatmap_path}")
                print(f"Saved FAMILY heatmap to: {family_heatmap_path}")

            fold_results.append(
                {
                    "fold": fold_name,
                    "gas_accuracy": gas_acc,
                    "family_accuracy": family_acc,
                    "consistency_accuracy": cascade_consistency_acc,
                    "n_train_samples": len(train_dataset),
                    "n_test_samples": len(test_dataset),
                    "input_dim": input_dim,
                    "num_gas_classes": num_gas,
                    "num_family_classes": num_family,
                    "epochs_run": 0,
                    "oof_family_epochs_run": 0,
                    "family_epochs_run": 0,
                    "gas_epochs_run": 0,
                    "rolling_window": args.rolling_window,
                    "rolling_step": args.rolling_step,
                    "use_stats_features": args.use_stats_features,
                    "stats_window_size": args.stats_window_size if args.use_stats_features else None,
                    "stats_step": args.stats_step if args.use_stats_features else None,
                    "stats_n_windows": args.stats_n_windows if args.use_stats_features else None,
                    "aggregate_per_file": args.aggregate_per_file if args.use_stats_features else False,
                    "cascade_input": args.cascade_input,
                    "oof_enabled": False,
                    "oof_folds": 0,
                    "model": args.model,
                }
            )

            if args.save_model:
                out_path = Path(args.save_model)
                if len(splits) > 1:
                    out_path = out_path.with_name(f"{out_path.stem}_{fold_name}{out_path.suffix}")

                torch.save(
                    {
                        "family_model": family_model,
                        "gas_model": gas_model,
                        "input_dim": input_dim,
                        "num_gas": num_gas,
                        "num_family": num_family,
                        "gas_classes": gas_encoder.classes_.tolist(),
                        "family_classes": family_encoder.classes_.tolist(),
                        "feature_columns": list(X_train.columns),
                        "scaler": args.scaler,
                        "scaler_mean": scaler.mean_.tolist() if scaler is not None and hasattr(scaler, "mean_") else None,
                        "scaler_scale": scaler.scale_.tolist() if scaler is not None and hasattr(scaler, "scale_") else None,
                        "scaler_min": scaler.min_.tolist() if scaler is not None and hasattr(scaler, "min_") else None,
                        "scaler_data_min": scaler.data_min_.tolist() if scaler is not None and hasattr(scaler, "data_min_") else None,
                        "scaler_data_max": scaler.data_max_.tolist() if scaler is not None and hasattr(scaler, "data_max_") else None,
                        "gas_to_family": GAS_TO_FAMILY,
                        "architecture": (
                            "RandomForest separated cascade: train family RF, generate family features, "
                            f"then train gas RF on [X, predicted family {args.cascade_input}]"
                        ),
                        "cascade_input": args.cascade_input,
                        "model": args.model,
                    },
                    out_path,
                )
                print(f"Saved model to: {out_path}")

            continue

        if use_oof:
            train_family_features, _, oof_family_epochs_run = oof_family_features(
                X_train=X_train_scaled,
                y_train_gas=y_train_gas_enc,
                y_train_family=y_train_family_enc,
                input_dim=input_dim,
                num_family=num_family,
                args=args,
                criterion=criterion,
                device=device,
            )
            print("Stage 2/3: training final family model for test-time predictions")
        else:
            oof_family_epochs_run = 0
            print("Stage 1/2: training family model")

        family_model, family_epochs_run = train_family_model(
            X=X_train_scaled,
            y_gas=y_train_gas_enc,
            y_family=y_train_family_enc,
            input_dim=input_dim,
            num_family=num_family,
            args=args,
            criterion=criterion,
            device=device,
            stage_name="Final family",
        )
        for param in family_model.parameters():
            param.requires_grad = False

        if not use_oof:
            train_family_features, _ = predict_family_features(
                family_model,
                X_train_scaled,
                batch_size=args.batch_size,
                device=device,
                cascade_input=args.cascade_input,
                num_family=num_family,
            )

        gas_model = GasCascadeMLP(
            input_dim=input_dim,
            num_family=num_family,
            num_gas=num_gas,
        ).to(device)
        gas_optimizer = torch.optim.Adam(
            gas_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        gas_train_dataset = CascadeGasDataset(
            X_train_scaled,
            train_family_features,
            y_train_gas_enc,
            y_train_family_enc,
        )
        gas_train_loader = DataLoader(
            gas_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        best_gas_loss = float("inf")
        gas_epochs_no_improve = 0
        gas_epochs_run = 0

        if use_oof:
            print("Stage 3/3: training gas model from OOF family predictions")
        else:
            print("Stage 2/2: training gas model from in-sample family predictions")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_gas_one_epoch(
                gas_model=gas_model,
                loader=gas_train_loader,
                optimizer=gas_optimizer,
                criterion=criterion,
                device=device,
                lambda_hier=effective_lambda_hier,
                hier_beta=args.hier_beta,
                hier_gamma=args.hier_gamma,
                gas_to_family_idx=gas_to_family_idx,
            )
            gas_epochs_run = epoch

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"Gas epoch {epoch:03d}/{args.epochs} - loss={train_loss:.6f}")

            if args.n_iter_no_change > 0:
                if train_loss < best_gas_loss - args.tol:
                    best_gas_loss = train_loss
                    gas_epochs_no_improve = 0
                else:
                    gas_epochs_no_improve += 1

                if gas_epochs_no_improve >= args.n_iter_no_change:
                    print(
                        f"Stopping gas model at epoch {epoch}: training loss did not improve "
                        f"by at least tol={args.tol} for "
                        f"{args.n_iter_no_change} consecutive epochs."
                    )
                    break

        y_pred_gas_enc, y_pred_family_enc = evaluate(
            family_model,
            gas_model,
            test_loader,
            device,
            cascade_input=args.cascade_input,
            num_family=num_family,
        )

        y_pred_gas = gas_encoder.inverse_transform(y_pred_gas_enc)
        y_pred_family = family_encoder.inverse_transform(y_pred_family_enc)

        gas_acc = accuracy_score(y_test_gas, y_pred_gas)
        family_acc = accuracy_score(y_test_family, y_pred_family)
        cascade_consistency_acc = consistency_accuracy(y_pred_gas, y_pred_family)

        print(f"GAS accuracy: {gas_acc:.6f}")
        print(f"FAMILY accuracy: {family_acc:.6f}")
        print(f"CONSISTENCY accuracy: {cascade_consistency_acc:.6f}")

        print("\nGas report")
        print(classification_report(y_test_gas, y_pred_gas, zero_division=0))

        print("\nFamily report")
        print(classification_report(y_test_family, y_pred_family, zero_division=0))

        if args.save_heatmap:
            gas_labels = gas_encoder.classes_.tolist()
            family_labels = family_encoder.classes_.tolist()

            gas_cm = normalized_confusion_matrix(y_test_gas, y_pred_gas, gas_labels)
            family_cm = normalized_confusion_matrix(
                y_test_family,
                y_pred_family,
                family_labels,
            )

            if gas_cm_labels is None:
                gas_cm_labels = gas_labels
            if family_cm_labels is None:
                family_cm_labels = family_labels

            if gas_labels == gas_cm_labels:
                gas_cms.append(gas_cm)
            else:
                print("Warning: GAS labels differ across folds. Skipping GAS mean heatmap for this fold.")

            if family_labels == family_cm_labels:
                family_cms.append(family_cm)
            else:
                print("Warning: FAMILY labels differ across folds. Skipping FAMILY mean heatmap for this fold.")

            gas_heatmap_path = heatmap_dir / f"gas_confusion_heatmap_{fold_name}.png"
            family_heatmap_path = heatmap_dir / f"family_confusion_heatmap_{fold_name}.png"

            save_heatmap(gas_cm, gas_labels, gas_heatmap_path, f"GAS confusion matrix (%) - {fold_name}")
            save_heatmap(
                family_cm,
                family_labels,
                family_heatmap_path,
                f"FAMILY confusion matrix (%) - {fold_name}",
            )
            print(f"Saved GAS heatmap to: {gas_heatmap_path}")
            print(f"Saved FAMILY heatmap to: {family_heatmap_path}")

        fold_results.append(
            {
                "fold": fold_name,
                "gas_accuracy": gas_acc,
                "family_accuracy": family_acc,
                "consistency_accuracy": cascade_consistency_acc,
                "n_train_samples": len(train_dataset),
                "n_test_samples": len(test_dataset),
                "input_dim": input_dim,
                "num_gas_classes": num_gas,
                "num_family_classes": num_family,
                "epochs_run": oof_family_epochs_run + family_epochs_run + gas_epochs_run,
                "oof_family_epochs_run": oof_family_epochs_run,
                "family_epochs_run": family_epochs_run,
                "gas_epochs_run": gas_epochs_run,
                "rolling_window": args.rolling_window,
                "rolling_step": args.rolling_step,
                "use_stats_features": args.use_stats_features,
                "stats_window_size": args.stats_window_size if args.use_stats_features else None,
                "stats_step": args.stats_step if args.use_stats_features else None,
                "stats_n_windows": args.stats_n_windows if args.use_stats_features else None,
                "aggregate_per_file": args.aggregate_per_file if args.use_stats_features else False,
                "cascade_input": args.cascade_input,
                "oof_enabled": use_oof,
                "oof_folds": args.oof_folds if use_oof else 0,
                "model": args.model,
            }
        )

        if args.save_model:
            out_path = Path(args.save_model)
            if len(splits) > 1:
                out_path = out_path.with_name(f"{out_path.stem}_{fold_name}{out_path.suffix}")

            payload = {
                "family_model_state_dict": family_model.state_dict(),
                "gas_model_state_dict": gas_model.state_dict(),
                "input_dim": input_dim,
                "num_gas": num_gas,
                "num_family": num_family,
                "gas_classes": gas_encoder.classes_.tolist(),
                "family_classes": family_encoder.classes_.tolist(),
                "feature_columns": list(X_train.columns),
                "scaler": args.scaler,
                "scaler_mean": scaler.mean_.tolist() if scaler is not None and hasattr(scaler, "mean_") else None,
                "scaler_scale": scaler.scale_.tolist() if scaler is not None and hasattr(scaler, "scale_") else None,
                "scaler_min": scaler.min_.tolist() if scaler is not None and hasattr(scaler, "min_") else None,
                "scaler_data_min": scaler.data_min_.tolist() if scaler is not None and hasattr(scaler, "data_min_") else None,
                "scaler_data_max": scaler.data_max_.tolist() if scaler is not None and hasattr(scaler, "data_max_") else None,
                "gas_to_family": GAS_TO_FAMILY,
                "architecture": (
                    "separated cascade: train family MLP, generate family features, then train gas MLP; "
                    f"[X, predicted family {args.cascade_input}] -> gas MLP"
                ),
                "cascade_input": args.cascade_input,
                "oof_enabled": use_oof,
                "oof_folds": args.oof_folds if use_oof else 0,
                "epochs_run": oof_family_epochs_run + family_epochs_run + gas_epochs_run,
                "oof_family_epochs_run": oof_family_epochs_run,
                "family_epochs_run": family_epochs_run,
                "gas_epochs_run": gas_epochs_run,
                "oof_folds": args.oof_folds,
                "max_epochs": args.epochs,
                "tol": args.tol,
                "n_iter_no_change": args.n_iter_no_change,
                "rolling_window": args.rolling_window,
                "rolling_step": args.rolling_step,
                "loss": {
                    "gas": "FocalLoss" if args.loss_mode == "focal" else "CrossEntropyLoss",
                    "family": "FocalLoss" if args.loss_mode == "focal" else "CrossEntropyLoss",
                    "focal_gamma": args.focal_gamma,
                    "alpha_family": None,
                    "mode": args.loss_mode,
                    "lambda_hier": effective_lambda_hier,
                    "hier_beta": args.hier_beta,
                    "hier_gamma": args.hier_gamma,
                },
            }

            torch.save(payload, out_path)
            print(f"Saved model to: {out_path}")

    if fold_results:
        df_results = pd.DataFrame(fold_results)

        print("\n=== Summary ===")
        print(f"Mean GAS accuracy: {df_results['gas_accuracy'].mean():.6f} +/- {df_results['gas_accuracy'].std(ddof=1):.6f}")
        print(f"Mean FAMILY accuracy: {df_results['family_accuracy'].mean():.6f} +/- {df_results['family_accuracy'].std(ddof=1):.6f}")
        print(f"Mean CONSISTENCY accuracy: {df_results['consistency_accuracy'].mean():.6f} +/- {df_results['consistency_accuracy'].std(ddof=1):.6f}")

        if args.save_report:
            report_df = df_results.copy()
            summary_rows = []

            for summary_name, summary_func in [
                ("mean", pd.Series.mean),
                ("std", pd.Series.std),
            ]:
                row = {col: np.nan for col in report_df.columns}
                row["fold"] = summary_name
                for col in [
                    "gas_accuracy",
                    "family_accuracy",
                    "consistency_accuracy",
                ]:
                    row[col] = summary_func(report_df[col])
                summary_rows.append(row)

            report_df = pd.concat(
                [report_df, pd.DataFrame(summary_rows)],
                axis=0,
                ignore_index=True,
            )
            report_df.to_csv(args.save_report, index=False)
            print(f"Saved report to: {args.save_report}")

        if args.save_heatmap:
            if gas_cms and gas_cm_labels is not None:
                mean_gas_cm = np.mean(gas_cms, axis=0)
                save_heatmap(
                    mean_gas_cm,
                    gas_cm_labels,
                    heatmap_dir / "gas_confusion_heatmap_mean.png",
                    "Mean GAS confusion matrix (%)",
                )
                print(f"Saved mean GAS heatmap to: {heatmap_dir / 'gas_confusion_heatmap_mean.png'}")

            if family_cms and family_cm_labels is not None:
                mean_family_cm = np.mean(family_cms, axis=0)
                save_heatmap(
                    mean_family_cm,
                    family_cm_labels,
                    heatmap_dir / "family_confusion_heatmap_mean.png",
                    "Mean FAMILY confusion matrix (%)",
                )
                print(f"Saved mean FAMILY heatmap to: {heatmap_dir / 'family_confusion_heatmap_mean.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
