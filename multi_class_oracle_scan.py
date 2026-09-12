import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from multi_class import (
    FocalLoss,
    GAS_TO_FAMILY,
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
from multi_class_hard_oracle import (
    HardOracleGasMLP,
    OracleGasDataset,
    one_hot,
)


class OracleScanDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y_family: np.ndarray,
        y_gas: np.ndarray,
        y_match: np.ndarray,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_family = torch.tensor(y_family, dtype=torch.float32)
        self.y_gas = torch.tensor(y_gas, dtype=torch.long)
        self.y_match = torch.tensor(y_match, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_family[idx], self.y_gas[idx], self.y_match[idx]


class OracleScanRejectMLP(nn.Module):
    """
    Conditional oracle with rejection:
    [X, family_one_hot] -> gas logits + family compatibility logit.
    """

    def __init__(self, input_dim: int, num_family: int, num_gas: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim + num_family, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.gas_head = nn.Linear(64, num_gas)
        self.match_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, family_one_hot: torch.Tensor):
        z = self.shared(torch.cat([x, family_one_hot], dim=1))
        return self.gas_head(z), self.match_head(z).squeeze(1)


def build_family_to_gas_mask(
    gas_classes: np.ndarray,
    family_classes: np.ndarray,
) -> torch.Tensor:
    gas_to_family_idx = build_gas_to_family_indices(gas_classes, family_classes)
    masks = []
    for family_idx in range(len(family_classes)):
        masks.append(gas_to_family_idx.eq(family_idx))
    return torch.stack(masks, dim=0)


def build_negative_scan_dataset(
    X: np.ndarray,
    y_family_enc: np.ndarray,
    y_gas_enc: np.ndarray,
    num_family: int,
    negative_families: int,
    seed: int,
) -> OracleScanDataset:
    rng = np.random.default_rng(seed)
    X_rows = []
    family_rows = []
    gas_rows = []
    match_rows = []
    eye = np.eye(num_family, dtype=np.float32)
    max_negatives = max(num_family - 1, 0)
    negative_families = min(max(negative_families, 0), max_negatives)

    for x_row, true_family, true_gas in zip(X, y_family_enc, y_gas_enc):
        X_rows.append(x_row)
        family_rows.append(eye[true_family])
        gas_rows.append(true_gas)
        match_rows.append(1.0)

        if negative_families > 0:
            candidates = np.asarray(
                [idx for idx in range(num_family) if idx != true_family],
                dtype=np.int64,
            )
            sampled = rng.choice(candidates, size=negative_families, replace=False)
            for wrong_family in sampled:
                X_rows.append(x_row)
                family_rows.append(eye[wrong_family])
                gas_rows.append(true_gas)
                match_rows.append(0.0)

    return OracleScanDataset(
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(family_rows, dtype=np.float32),
        np.asarray(gas_rows, dtype=np.int64),
        np.asarray(match_rows, dtype=np.float32),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_hier: float,
    hier_beta: float,
    hier_gamma: float,
    gas_to_family_idx: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y_family, y_gas in loader:
        X = X.to(device)
        y_family = y_family.to(device)
        y_gas = y_gas.to(device)

        optimizer.zero_grad()
        gas_logits = model(X, y_family)
        loss = criterion(gas_logits, y_gas)

        if lambda_hier > 0.0:
            loss_hier = hierarchical_consistency_penalty(
                gas_logits=gas_logits,
                family_logits=y_family,
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


def train_scan_reject_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    match_criterion: nn.Module,
    device: torch.device,
    match_loss_weight: float,
    lambda_hier: float,
    hier_beta: float,
    hier_gamma: float,
    gas_to_family_idx: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y_family, y_gas, y_match in loader:
        X = X.to(device)
        y_family = y_family.to(device)
        y_gas = y_gas.to(device)
        y_match = y_match.to(device)

        optimizer.zero_grad()
        gas_logits, match_logits = model(X, y_family)

        positive_mask = y_match.gt(0.5)
        if positive_mask.any():
            loss_gas = criterion(gas_logits[positive_mask], y_gas[positive_mask])
        else:
            loss_gas = torch.zeros((), device=device)

        loss_match = match_criterion(match_logits, y_match)
        loss = loss_gas + match_loss_weight * loss_match

        if lambda_hier > 0.0 and positive_mask.any():
            loss_hier = hierarchical_consistency_penalty(
                gas_logits=gas_logits[positive_mask],
                family_logits=y_family[positive_mask],
                y_gas=y_gas[positive_mask],
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
def evaluate_scan(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_family: int,
    family_to_gas_mask: torch.Tensor,
    scan_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
    family_eye = torch.eye(num_family, dtype=torch.float32, device=device)
    family_to_gas_mask = family_to_gas_mask.to(device)

    gas_preds = []
    family_preds = []

    for X_batch in loader:
        X_batch = X_batch.to(device)
        batch_size_current = X_batch.size(0)
        best_scores = None
        best_gas = None
        best_family = None

        for family_idx in range(num_family):
            family_one_hot = family_eye[family_idx].unsqueeze(0).repeat(batch_size_current, 1)
            model_out = model(X_batch, family_one_hot)
            if isinstance(model_out, tuple):
                gas_logits, match_logits = model_out
                match_log_probs = torch.nn.functional.logsigmoid(match_logits)
            else:
                gas_logits = model_out
                match_log_probs = None

            gas_log_probs = torch.log_softmax(gas_logits, dim=1)

            if scan_mode == "masked":
                mask = family_to_gas_mask[family_idx].unsqueeze(0)
                gas_scores = gas_log_probs.masked_fill(~mask, -torch.inf)
            else:
                gas_scores = gas_log_probs

            family_scores, family_gas = torch.max(gas_scores, dim=1)
            if match_log_probs is not None:
                family_scores = family_scores + match_log_probs

            if best_scores is None:
                best_scores = family_scores
                best_gas = family_gas
                best_family = torch.full_like(family_gas, family_idx)
            else:
                improve = family_scores > best_scores
                best_scores = torch.where(improve, family_scores, best_scores)
                best_gas = torch.where(improve, family_gas, best_gas)
                best_family = torch.where(
                    improve,
                    torch.full_like(family_gas, family_idx),
                    best_family,
                )

        gas_preds.append(best_gas.cpu().numpy())
        family_preds.append(best_family.cpu().numpy())

    return np.concatenate(gas_preds), np.concatenate(family_preds)


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
    ap.add_argument("--rolling-window", type=int, default=1)
    ap.add_argument("--rolling-step", type=int, default=1)
    ap.add_argument(
        "--scaler",
        choices=["standard", "minmax", "none"],
        default="standard",
    )

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
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
        "--negative-families",
        type=int,
        default=0,
        help=(
            "Number of wrong family one-hot vectors sampled per training row. "
            "0 keeps the original oracle-scan training."
        ),
    )
    ap.add_argument(
        "--match-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the compatibility loss used when --negative-families > 0.",
    )
    ap.add_argument(
        "--scan-mode",
        choices=["unmasked", "masked"],
        default="unmasked",
        help=(
            "unmasked: scan all families and all gas outputs; "
            "masked: only gas compatible with each candidate family are considered."
        ),
    )

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
    print("Oracle scan mode: train with true family, test by scanning all families")
    print(f"Scan compatibility mode: {args.scan_mode}")
    if args.negative_families > 0:
        print(
            f"Negative family training: {args.negative_families} negatives per row, "
            f"match_loss_weight={args.match_loss_weight}"
        )

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

        train_listing = fold.get("train", {})
        test_listing = fold.get("valid", {}) or fold.get("test", {})

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
        current_test_root = test_root if test_root is not None else data_root
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

        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, args.scaler)
        train_family_one_hot = one_hot(y_train_family_enc, len(family_encoder.classes_))

        if args.negative_families > 0:
            train_dataset = build_negative_scan_dataset(
                X=X_train_scaled,
                y_family_enc=y_train_family_enc,
                y_gas_enc=y_train_gas_enc,
                num_family=len(family_encoder.classes_),
                negative_families=args.negative_families,
                seed=args.seed,
            )
        else:
            train_dataset = OracleGasDataset(
                X_train_scaled,
                train_family_one_hot,
                y_train_gas_enc,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        input_dim = X_train_scaled.shape[1]
        num_gas = len(gas_encoder.classes_)
        num_family = len(family_encoder.classes_)

        print(f"Input dim: {input_dim}")
        print(f"Gas classes: {num_gas}")
        print(f"Family classes: {num_family}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(X_test_scaled)}")
        print(f"Scaler: {args.scaler}")
        if args.rolling_window > 1:
            print(f"Rolling mean: window={args.rolling_window}, step={args.rolling_step}")

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

        if args.negative_families > 0:
            model = OracleScanRejectMLP(
                input_dim=input_dim,
                num_family=num_family,
                num_gas=num_gas,
            ).to(device)
        else:
            model = HardOracleGasMLP(
                input_dim=input_dim,
                num_family=num_family,
                num_gas=num_gas,
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
        match_criterion = nn.BCEWithLogitsLoss()
        gas_to_family_idx = build_gas_to_family_indices(
            gas_encoder.classes_,
            family_encoder.classes_,
        ).to(device)
        family_to_gas_mask = build_family_to_gas_mask(
            gas_encoder.classes_,
            family_encoder.classes_,
        )

        best_train_loss = float("inf")
        epochs_no_improve = 0
        actual_epochs = 0

        for epoch in range(1, args.epochs + 1):
            if args.negative_families > 0:
                train_loss = train_scan_reject_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    match_criterion=match_criterion,
                    device=device,
                    match_loss_weight=args.match_loss_weight,
                    lambda_hier=effective_lambda_hier,
                    hier_beta=args.hier_beta,
                    hier_gamma=args.hier_gamma,
                    gas_to_family_idx=gas_to_family_idx,
                )
            else:
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
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

        y_pred_gas_enc, y_pred_family_enc = evaluate_scan(
            model,
            X_test_scaled,
            batch_size=args.batch_size,
            device=device,
            num_family=num_family,
            family_to_gas_mask=family_to_gas_mask,
            scan_mode=args.scan_mode,
        )

        y_pred_gas = gas_encoder.inverse_transform(y_pred_gas_enc)
        y_pred_family = family_encoder.inverse_transform(y_pred_family_enc)

        gas_acc = accuracy_score(y_test_gas, y_pred_gas)
        family_acc = accuracy_score(y_test_family, y_pred_family)
        scan_consistency_acc = consistency_accuracy(y_pred_gas, y_pred_family)

        print(f"GAS accuracy: {gas_acc:.6f}")
        print(f"FAMILY scan accuracy: {family_acc:.6f}")
        print(f"CONSISTENCY accuracy: {scan_consistency_acc:.6f}")

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
            if family_labels == family_cm_labels:
                family_cms.append(family_cm)

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
                "consistency_accuracy": scan_consistency_acc,
                "n_train_samples": len(train_dataset),
                "n_test_samples": len(X_test_scaled),
                "input_dim": input_dim,
                "num_gas_classes": num_gas,
                "num_family_classes": num_family,
                "epochs_run": actual_epochs,
                "rolling_window": args.rolling_window,
                "rolling_step": args.rolling_step,
                "scaler": args.scaler,
                "scan_mode": args.scan_mode,
                "negative_families": args.negative_families,
                "match_loss_weight": args.match_loss_weight if args.negative_families > 0 else 0.0,
            }
        )

        if args.save_model:
            out_path = Path(args.save_model)
            if len(splits) > 1:
                out_path = out_path.with_name(f"{out_path.stem}_{fold_name}{out_path.suffix}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_gas": num_gas,
                    "num_family": num_family,
                    "gas_classes": gas_encoder.classes_.tolist(),
                    "family_classes": family_encoder.classes_.tolist(),
                    "feature_columns": list(X_train.columns),
                    "architecture": (
                        "oracle scan with rejection"
                        if args.negative_families > 0
                        else "oracle scan: train [X, true family one-hot] -> gas; test scans all family one-hot vectors"
                    ),
                    "scan_mode": args.scan_mode,
                    "negative_families": args.negative_families,
                    "match_loss_weight": args.match_loss_weight if args.negative_families > 0 else 0.0,
                    "scaler": args.scaler,
                    "scaler_mean": scaler.mean_.tolist() if scaler is not None and hasattr(scaler, "mean_") else None,
                    "scaler_scale": scaler.scale_.tolist() if scaler is not None and hasattr(scaler, "scale_") else None,
                    "scaler_min": scaler.min_.tolist() if scaler is not None and hasattr(scaler, "min_") else None,
                    "scaler_data_min": scaler.data_min_.tolist() if scaler is not None and hasattr(scaler, "data_min_") else None,
                    "scaler_data_max": scaler.data_max_.tolist() if scaler is not None and hasattr(scaler, "data_max_") else None,
                },
                out_path,
            )
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
                save_heatmap(
                    np.mean(gas_cms, axis=0),
                    gas_cm_labels,
                    heatmap_dir / "gas_confusion_heatmap_mean.png",
                    "Mean GAS confusion matrix (%)",
                )
            if family_cms and family_cm_labels is not None:
                save_heatmap(
                    np.mean(family_cms, axis=0),
                    family_cm_labels,
                    heatmap_dir / "family_confusion_heatmap_mean.png",
                    "Mean FAMILY confusion matrix (%)",
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
