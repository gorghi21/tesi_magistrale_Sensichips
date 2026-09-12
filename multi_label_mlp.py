import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from multi_class import (
    build_listing_from_root,
    map_gas_to_family,
    prepare_xy,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultilabelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MultilabelMLP(nn.Module):
    """
    MLP coerente con multi_class.py:
    input -> 128 -> 64 -> label multilabel.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_split_data(
    listing,
    data_root: Path,
    args: argparse.Namespace,
    index_col,
) -> Tuple[pd.DataFrame, np.ndarray]:
    return prepare_xy(
        listing,
        data_root,
        args.path_template,
        sep=args.sep,
        decimal=args.decimal,
        index_col=index_col,
        label_col=args.label_col,
        rolling_window=args.rolling_window,
        rolling_step=args.rolling_step,
    )


def make_multilabel_targets(
    y_gas: np.ndarray,
    gas_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
) -> np.ndarray:
    y_family = map_gas_to_family(y_gas)
    y_gas_enc = gas_encoder.transform(y_gas)
    y_family_enc = family_encoder.transform(y_family)

    num_gas = len(gas_encoder.classes_)
    num_family = len(family_encoder.classes_)
    targets = np.zeros((len(y_gas), num_gas + num_family), dtype=np.float32)

    rows = np.arange(len(y_gas))
    targets[rows, y_gas_enc] = 1.0
    targets[rows, num_gas + y_family_enc] = 1.0

    return targets


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs = []

    for X, _ in loader:
        X = X.to(device)
        logits = model(X)
        probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs)


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
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for generic multilabel metrics.",
    )

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--n-iter-no-change", type=int, default=10)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--save-report", default=None)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    for fold_name, fold in splits.items():
        print(f"\n=== {fold_name} ===")

        train_listing = fold.get("train", {})
        test_listing = fold.get("valid", {}) or fold.get("test", {})
        current_test_root = test_root if test_root is not None else data_root

        X_train, y_train_gas = load_split_data(
            train_listing,
            data_root,
            args,
            index_col,
        )
        X_test, y_test_gas = load_split_data(
            test_listing,
            current_test_root,
            args,
            index_col,
        )

        train_cols = X_train.columns
        X_test = X_test.reindex(columns=train_cols, fill_value=0.0)

        y_train_family = map_gas_to_family(y_train_gas)
        y_test_family = map_gas_to_family(y_test_gas)

        unseen_gas = sorted(set(y_test_gas) - set(y_train_gas))
        unseen_family = sorted(set(y_test_family) - set(y_train_family))
        if unseen_gas or unseen_family:
            message = (
                f"Test gas absent from training: {unseen_gas}; "
                f"test family absent from training: {unseen_family}"
            )
            if args.drop_unseen_test_labels:
                keep_mask = np.isin(y_test_gas, y_train_gas) & np.isin(
                    y_test_family,
                    y_train_family,
                )
                dropped = int((~keep_mask).sum())
                X_test = X_test.loc[keep_mask].reset_index(drop=True)
                y_test_gas = y_test_gas[keep_mask]
                y_test_family = y_test_family[keep_mask]
                print(f"Warning: {message}. Dropped {dropped} test rows.")
                if y_test_gas.size == 0:
                    raise ValueError("No test rows left after dropping unseen labels.")
            else:
                raise ValueError(
                    f"{message}. Use --drop-unseen-test-labels for closed-set evaluation."
                )

        gas_encoder = LabelEncoder()
        family_encoder = LabelEncoder()
        gas_encoder.fit(y_train_gas)
        family_encoder.fit(y_train_family)

        y_train_multi = make_multilabel_targets(
            y_train_gas,
            gas_encoder,
            family_encoder,
        )
        y_test_multi = make_multilabel_targets(
            y_test_gas,
            gas_encoder,
            family_encoder,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values).astype(np.float32)
        X_test_scaled = scaler.transform(X_test.values).astype(np.float32)

        train_dataset = MultilabelDataset(X_train_scaled, y_train_multi)
        test_dataset = MultilabelDataset(X_test_scaled, y_test_multi)

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
        output_dim = num_gas + num_family

        print("Target: multilabel gas + family")
        print(f"Input dim: {input_dim}")
        print(f"Gas labels: {num_gas}")
        print(f"Family labels: {num_family}")
        print(f"Output labels: {output_dim}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        if args.rolling_window > 1:
            print(
                f"Rolling mean: window={args.rolling_window}, "
                f"step={args.rolling_step}"
            )

        model = MultilabelMLP(input_dim=input_dim, output_dim=output_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_train_loss = float("inf")
        epochs_no_improve = 0
        actual_epochs = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
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

        y_prob = predict_probabilities(model, test_loader, device)
        y_pred_multi = (y_prob >= args.threshold).astype(np.float32)

        y_pred_gas_enc = np.argmax(y_prob[:, :num_gas], axis=1)
        y_pred_family_enc = np.argmax(y_prob[:, num_gas:], axis=1)
        y_pred_gas = gas_encoder.inverse_transform(y_pred_gas_enc)
        y_pred_family = family_encoder.inverse_transform(y_pred_family_enc)

        gas_acc = accuracy_score(y_test_gas, y_pred_gas)
        family_acc = accuracy_score(y_test_family, y_pred_family)
        exact_match = np.mean(np.all(y_pred_multi == y_test_multi, axis=1))
        micro_f1 = f1_score(y_test_multi, y_pred_multi, average="micro", zero_division=0)
        macro_f1 = f1_score(y_test_multi, y_pred_multi, average="macro", zero_division=0)

        print(f"Exact multilabel match @ {args.threshold}: {exact_match:.6f}")
        print(f"Multilabel micro-F1 @ {args.threshold}: {micro_f1:.6f}")
        print(f"Multilabel macro-F1 @ {args.threshold}: {macro_f1:.6f}")
        print(f"GAS accuracy from gas label group: {gas_acc:.6f}")
        print(f"FAMILY accuracy from family label group: {family_acc:.6f}")

        print("\nGas report")
        print(classification_report(y_test_gas, y_pred_gas, zero_division=0))

        print("\nFamily report")
        print(classification_report(y_test_family, y_pred_family, zero_division=0))

        fold_results.append(
            {
                "fold": fold_name,
                "exact_match": exact_match,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "gas_accuracy": gas_acc,
                "family_accuracy": family_acc,
                "n_train_samples": len(train_dataset),
                "n_test_samples": len(test_dataset),
                "input_dim": input_dim,
                "num_gas_labels": num_gas,
                "num_family_labels": num_family,
                "output_dim": output_dim,
                "epochs_run": actual_epochs,
                "rolling_window": args.rolling_window,
                "rolling_step": args.rolling_step,
                "threshold": args.threshold,
            }
        )

        if args.save_model:
            out_path = Path(args.save_model)
            if len(splits) > 1:
                out_path = out_path.with_name(
                    f"{out_path.stem}_{fold_name}{out_path.suffix}"
                )

            payload = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "gas_classes": gas_encoder.classes_.tolist(),
                "family_classes": family_encoder.classes_.tolist(),
                "feature_columns": list(X_train.columns),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "architecture": "MLP input -> 128 -> 64 -> multilabel gas+family",
                "epochs_run": actual_epochs,
                "max_epochs": args.epochs,
                "rolling_window": args.rolling_window,
                "rolling_step": args.rolling_step,
                "threshold": args.threshold,
            }
            torch.save(payload, out_path)
            print(f"Saved model to: {out_path}")

    if fold_results:
        df_results = pd.DataFrame(fold_results)
        gas_std = df_results["gas_accuracy"].std(ddof=1) if len(df_results) > 1 else 0.0
        family_std = (
            df_results["family_accuracy"].std(ddof=1) if len(df_results) > 1 else 0.0
        )

        print("\n=== Summary ===")
        print(
            f"Mean GAS accuracy: {df_results['gas_accuracy'].mean():.6f} +/- "
            f"{gas_std:.6f}"
        )
        print(
            f"Mean FAMILY accuracy: {df_results['family_accuracy'].mean():.6f} +/- "
            f"{family_std:.6f}"
        )
        print(f"Mean micro-F1: {df_results['micro_f1'].mean():.6f}")
        print(f"Mean macro-F1: {df_results['macro_f1'].mean():.6f}")

        if args.save_report:
            df_results.to_csv(args.save_report, index=False)
            print(f"Saved report to: {args.save_report}")


if __name__ == "__main__":
    main()
