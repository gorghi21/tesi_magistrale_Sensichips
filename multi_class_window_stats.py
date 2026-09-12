import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from feature_engineering import aggregate_numeric_timeseries

from multi_class import (  # noqa: E402
    FocalLoss,
    GAS_TO_FAMILY,
    GasFamilyMLP,
    TabularGasDataset,
    align_columns,
    build_gas_to_family_indices,
    build_listing_from_root,
    consistency_accuracy,
    evaluate,
    hierarchical_consistency_penalty,
    load_table,
    map_gas_to_family,
    normalized_confusion_matrix,
    resolve_path,
    save_heatmap,
    scale_features,
    set_seed,
    split_xy,
)


def apply_window_stats(
    X: pd.DataFrame,
    y: np.ndarray,
    window_size: int,
    step: int,
    n_windows: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if window_size < 1:
        raise ValueError("--stats-window-size must be >= 1.")
    if step < 1:
        raise ValueError("--stats-step must be >= 1.")

    frames = []
    labels = []

    if len(X) == 0:
        raise ValueError("Empty CSV after numeric feature extraction.")

    if len(X) < window_size:
        X_stats = aggregate_numeric_timeseries(X.reset_index(drop=True), n_windows=n_windows)
        return X_stats.reset_index(drop=True), np.asarray([y[-1]], dtype=str)

    starts = list(range(0, len(X) - window_size + 1, step))
    if not starts:
        starts = [0]

    last_end = 0
    for start in starts:
        end = min(start + window_size, len(X))
        X_window = X.iloc[start:end].reset_index(drop=True)
        frames.append(aggregate_numeric_timeseries(X_window, n_windows=n_windows))
        labels.append(y[end - 1])
        last_end = end

    if last_end < len(X):
        start = last_end
        end = len(X)
        X_window = X.iloc[start:end].reset_index(drop=True)
        frames.append(aggregate_numeric_timeseries(X_window, n_windows=n_windows))
        labels.append(y[end - 1])

    if not frames:
        X_stats = aggregate_numeric_timeseries(X.reset_index(drop=True), n_windows=n_windows)
        return X_stats.reset_index(drop=True), np.asarray([y[-1]], dtype=str)

    return (
        pd.concat(frames, axis=0, ignore_index=True),
        np.asarray(labels, dtype=str),
    )


def prepare_xy_window_stats(
    listing: Dict[str, List[str]],
    data_root: Path,
    path_template: str,
    sep: Optional[str],
    decimal: str,
    index_col,
    label_col: Optional[str],
    stats_window_size: int,
    stats_step: int,
    stats_n_windows: int,
    aggregate_per_file: bool,
    split_name: str,
    verbose_files: bool,
) -> Tuple[pd.DataFrame, np.ndarray]:
    frames = []
    labels = []
    total_files = sum(len(files) for files in listing.values())
    processed_files = 0

    for class_name, files in listing.items():
        for file_base in files:
            processed_files += 1
            if verbose_files:
                print(
                    f"  [{split_name}] processing {processed_files}/{total_files}: "
                    f"{class_name}/{file_base}"
                )
            csv_path = resolve_path(data_root, path_template, class_name, file_base)
            df = load_table(csv_path, sep=sep, decimal=decimal, index_col=index_col)
            X_part, y_part = split_xy(df, label_col)
            if aggregate_per_file:
                X_part = aggregate_numeric_timeseries(
                    X_part.reset_index(drop=True),
                    n_windows=stats_n_windows,
                ).reset_index(drop=True)
                y_part = np.asarray([y_part[-1]], dtype=str)
            else:
                X_part, y_part = apply_window_stats(
                    X_part,
                    y_part,
                    window_size=stats_window_size,
                    step=stats_step,
                    n_windows=stats_n_windows,
                )
            frames.append(X_part)
            labels.append(y_part)

    if not frames:
        raise ValueError("Empty dataset after loading split.")

    return pd.concat(frames, axis=0, ignore_index=True), np.concatenate(labels)


def limit_listing(
    listing: Dict[str, List[str]],
    max_files_per_class: Optional[int],
) -> Dict[str, List[str]]:
    if max_files_per_class is None:
        return listing
    return {
        class_name: files[:max_files_per_class]
        for class_name, files in listing.items()
    }


def sanitize_feature_frame(X: pd.DataFrame, name: str) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    bad_cells = int(X.isna().sum().sum())
    if bad_cells:
        print(f"Warning: {name} contains {bad_cells} NaN/inf feature values. Replacing with 0.0.")
    return X.fillna(0.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    alpha_family: float,
    lambda_hier: float,
    hier_beta: float,
    hier_gamma: float,
    gas_to_family_idx: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y_gas, y_family in loader:
        X = X.to(device)
        y_gas = y_gas.to(device)
        y_family = y_family.to(device)

        optimizer.zero_grad()
        gas_logits, family_logits = model(X)

        loss_gas = criterion(gas_logits, y_gas)
        loss_family = criterion(family_logits, y_family)
        loss = loss_gas + alpha_family * loss_family

        if lambda_hier > 0.0:
            loss_hier = hierarchical_consistency_penalty(
                gas_logits=gas_logits,
                family_logits=family_logits,
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--splits", default=None)
    ap.add_argument(
        "--only-fold",
        default=None,
        help="Run only one fold from the splits JSON, e.g. fold_1.",
    )
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--path-template", default="{class_name}/{file}.csv")
    ap.add_argument("--sep", default=None)
    ap.add_argument("--decimal", default=".")
    ap.add_argument("--index-col", default=None, type=str)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--drop-unseen-test-labels", action="store_true")
    ap.add_argument("--max-files-per-class", type=int, default=None)
    ap.add_argument(
        "--verbose-files",
        action="store_true",
        help="Print every CSV file processed during feature extraction.",
    )
    ap.add_argument(
        "--stats-window-size",
        type=int,
        default=20,
        help="Number of consecutive rows used to compute one statistical feature vector.",
    )
    ap.add_argument(
        "--stats-step",
        type=int,
        default=20,
        help="Stride between statistical windows.",
    )
    ap.add_argument(
        "--stats-n-windows",
        type=int,
        default=4,
        help="Internal sub-windows used by code_2 aggregate_numeric_timeseries.",
    )
    ap.add_argument(
        "--aggregate-per-file",
        action="store_true",
        help="Compute one statistical feature vector for each whole CSV, like code_2/script.py.",
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
        help="Classifier backend. mlp uses the PyTorch multi-task MLP; rf uses two RandomForest classifiers.",
    )
    ap.add_argument("--rf-estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--rf-balanced", action="store_true")
    ap.add_argument("--alpha-family", type=float, default=1.0)
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

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--save-report", default=None)
    ap.add_argument("--save-heatmap", action="store_true")
    ap.add_argument("--heatmap-dir", default=".")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.aggregate_per_file:
        print("Feature mode: code_2 statistical features aggregated per CSV file")
    else:
        print("Feature mode: code_2 statistical features on temporal windows")

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

        if args.only_fold is not None:
            if args.only_fold not in splits:
                raise ValueError(
                    f"Fold '{args.only_fold}' not found. Available folds: {list(splits.keys())}"
                )
            splits = {args.only_fold: splits[args.only_fold]}

    fold_results = []
    gas_cms = []
    family_cms = []
    gas_cm_labels = None
    family_cm_labels = None
    heatmap_dir = Path(args.heatmap_dir)

    for fold_name, fold in splits.items():
        print(f"\n=== {fold_name} ===")

        train_listing = limit_listing(
            fold.get("train", {}),
            args.max_files_per_class,
        )
        test_listing = limit_listing(
            fold.get("valid", {}) or fold.get("test", {}),
            args.max_files_per_class,
        )

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

        current_test_root = test_root if test_root is not None else data_root
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

        X_train, X_test = align_columns(X_train, X_test)
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
        if not np.isfinite(X_train_scaled).all() or not np.isfinite(X_test_scaled).all():
            print("Warning: scaler produced NaN/inf values. Replacing with 0.0.")
            X_train_scaled = np.nan_to_num(
                X_train_scaled,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            X_test_scaled = np.nan_to_num(
                X_test_scaled,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
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
        if args.aggregate_per_file:
            print(f"Stats aggregation: per file, internal_n_windows={args.stats_n_windows}")
        else:
            print(
                f"Stats windows: size={args.stats_window_size}, "
                f"step={args.stats_step}, internal_n_windows={args.stats_n_windows}"
            )
        print(f"Scaler: {args.scaler}")

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
        print(f"Model: {args.model}")

        actual_epochs = 0

        if args.model == "rf":
            if args.loss_mode != "standard":
                print("Warning: --loss-mode is ignored with --model rf. RandomForest uses its native criterion.")
            print(
                f"RandomForest: n_estimators={args.rf_estimators}, "
                f"max_depth={args.rf_max_depth}, "
                f"min_samples_leaf={args.rf_min_samples_leaf}, "
                f"balanced={args.rf_balanced}"
            )
            gas_model = RandomForestClassifier(
                n_estimators=args.rf_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                random_state=args.seed,
                n_jobs=-1,
                class_weight="balanced_subsample" if args.rf_balanced else None,
            )
            family_model = RandomForestClassifier(
                n_estimators=args.rf_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                random_state=args.seed,
                n_jobs=-1,
                class_weight="balanced_subsample" if args.rf_balanced else None,
            )
            gas_model.fit(X_train_scaled, y_train_gas_enc)
            family_model.fit(X_train_scaled, y_train_family_enc)
            y_pred_gas_enc = gas_model.predict(X_test_scaled)
            y_pred_family_enc = family_model.predict(X_test_scaled)
        else:
            model = GasFamilyMLP(
                input_dim=input_dim,
                num_gas=num_gas,
                num_family=num_family,
            ).to(device)

            gas_to_family_idx = build_gas_to_family_indices(
                gas_encoder.classes_,
                family_encoder.classes_,
            ).to(device)

            criterion = (
                FocalLoss(gamma=args.focal_gamma)
                if args.loss_mode == "focal"
                else nn.CrossEntropyLoss()
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            best_train_loss = float("inf")
            epochs_no_improve = 0

            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    alpha_family=args.alpha_family,
                    lambda_hier=effective_lambda_hier,
                    hier_beta=args.hier_beta,
                    hier_gamma=args.hier_gamma,
                    gas_to_family_idx=gas_to_family_idx,
                )
                actual_epochs = epoch

                if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                    print(f"Epoch {epoch:03d}/{args.epochs} - loss={train_loss:.6f}")

                if args.n_iter_no_change > 0:
                    if train_loss < best_train_loss - args.tol:
                        best_train_loss = train_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= args.n_iter_no_change:
                        print(
                            f"Stopping at epoch {epoch}: training loss did not improve "
                            f"by at least tol={args.tol} for "
                            f"{args.n_iter_no_change} consecutive epochs."
                        )
                        break

            y_pred_gas_enc, y_pred_family_enc = evaluate(model, test_loader, device)

        y_pred_gas = gas_encoder.inverse_transform(y_pred_gas_enc)
        y_pred_family = family_encoder.inverse_transform(y_pred_family_enc)

        gas_acc = accuracy_score(y_test_gas, y_pred_gas)
        family_acc = accuracy_score(y_test_family, y_pred_family)
        consistency_acc = consistency_accuracy(y_pred_gas, y_pred_family)

        print(f"GAS accuracy: {gas_acc:.6f}")
        print(f"FAMILY accuracy: {family_acc:.6f}")
        print(f"CONSISTENCY accuracy: {consistency_acc:.6f}")

        print("\nGas report")
        print(classification_report(y_test_gas, y_pred_gas, zero_division=0))

        print("\nFamily report")
        print(classification_report(y_test_family, y_pred_family, zero_division=0))

        if args.save_heatmap:
            gas_labels = gas_encoder.classes_.tolist()
            family_labels = family_encoder.classes_.tolist()

            gas_cm = normalized_confusion_matrix(y_test_gas, y_pred_gas, gas_labels)
            family_cm = normalized_confusion_matrix(y_test_family, y_pred_family, family_labels)

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
            save_heatmap(family_cm, family_labels, family_heatmap_path, f"FAMILY confusion matrix (%) - {fold_name}")
            print(f"Saved GAS heatmap to: {gas_heatmap_path}")
            print(f"Saved FAMILY heatmap to: {family_heatmap_path}")

        fold_results.append(
            {
                "fold": fold_name,
                "gas_accuracy": gas_acc,
                "family_accuracy": family_acc,
                "consistency_accuracy": consistency_acc,
                "n_train_samples": len(train_dataset),
                "n_test_samples": len(test_dataset),
                "input_dim": input_dim,
                "num_gas_classes": num_gas,
                "num_family_classes": num_family,
                "epochs_run": actual_epochs,
                "stats_window_size": args.stats_window_size,
                "stats_step": args.stats_step,
                "stats_n_windows": args.stats_n_windows,
                "aggregate_per_file": args.aggregate_per_file,
                "scaler": args.scaler,
                "model": args.model,
            }
        )

        if args.save_model:
            out_path = Path(args.save_model)
            if len(splits) > 1:
                out_path = out_path.with_name(f"{out_path.stem}_{fold_name}{out_path.suffix}")

            payload = {
                "input_dim": input_dim,
                "num_gas": num_gas,
                "num_family": num_family,
                "gas_classes": gas_encoder.classes_.tolist(),
                "family_classes": family_encoder.classes_.tolist(),
                "feature_columns": list(X_train.columns),
                "model": args.model,
                "architecture": (
                    "window statistical features + RandomForest gas/family classifiers"
                    if args.model == "rf"
                    else "window statistical features + MLP 128 -> 64 -> gas_head + family_head"
                ),
                "scaler": args.scaler,
                "scaler_mean": scaler.mean_.tolist() if scaler is not None and hasattr(scaler, "mean_") else None,
                "scaler_scale": scaler.scale_.tolist() if scaler is not None and hasattr(scaler, "scale_") else None,
                "scaler_min": scaler.min_.tolist() if scaler is not None and hasattr(scaler, "min_") else None,
                "scaler_data_min": scaler.data_min_.tolist() if scaler is not None and hasattr(scaler, "data_min_") else None,
                "scaler_data_max": scaler.data_max_.tolist() if scaler is not None and hasattr(scaler, "data_max_") else None,
                "gas_to_family": GAS_TO_FAMILY,
                "epochs_run": actual_epochs,
                "max_epochs": args.epochs,
                "tol": args.tol,
                "n_iter_no_change": args.n_iter_no_change,
                "stats_window_size": args.stats_window_size,
                "stats_step": args.stats_step,
                "stats_n_windows": args.stats_n_windows,
                "aggregate_per_file": args.aggregate_per_file,
                "loss": {
                    "gas": "FocalLoss" if args.loss_mode == "focal" else "CrossEntropyLoss",
                    "family": "FocalLoss" if args.loss_mode == "focal" else "CrossEntropyLoss",
                    "focal_gamma": args.focal_gamma,
                    "alpha_family": args.alpha_family,
                    "mode": args.loss_mode,
                    "lambda_hier": effective_lambda_hier,
                    "hier_beta": args.hier_beta,
                    "hier_gamma": args.hier_gamma,
                },
            }

            if args.model == "rf":
                payload["gas_model"] = gas_model
                payload["family_model"] = family_model
                payload["rf"] = {
                    "n_estimators": args.rf_estimators,
                    "max_depth": args.rf_max_depth,
                    "min_samples_leaf": args.rf_min_samples_leaf,
                    "balanced": args.rf_balanced,
                }
            else:
                payload["model_state_dict"] = model.state_dict()

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
