import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader


# Mapping gas -> famiglia chimica.
GAS_TO_FAMILY = {
    "acetic_acid": "acid",
    "formic_acid": "acid",
    "phosphoric_acid": "acid",
    "apple_vinegar": "acid_mixture",
    "balsamic_vinegar": "acid_mixture",

    "ammonia": "base",
    "sodium_hydroxide": "base",

    "ammonium_chloride": "salt",
    "calcium_nitrate": "salt_oxidizer",

    "ethanol": "alcohol",
    "bioethanol": "alcohol",
    "isopropanol": "alcohol",

    "red_wine": "alcohol_mixture",

    "acetone": "ketone",

    "methane": "hydrocarbon",
    "butane": "hydrocarbon",

    "gasoline": "hydrocarbon_mixture",
    "diesel": "hydrocarbon_mixture",
    "kerosene": "hydrocarbon_mixture",
    "lighter_fluid": "hydrocarbon_mixture",

    "hydrogen_peroxide": "oxidizer",

    "urea": "amide",

    "air": "background",
    "vapor_water": "background",
    "water_vapor": "background",

    "nitromethane": "nitro_compound",
}

WARNED_FEATURE_COUNTS = set()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_label(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def resolve_path(
    data_root: Path,
    path_template: str,
    class_name: str,
    file_base: str,
) -> Path:
    rel = path_template.format(class_name=class_name, file=file_base)
    p = Path(rel)
    if not p.is_absolute():
        p = data_root / p
    return p.with_suffix(".csv") if p.suffix.lower() != ".csv" else p


def sniff_sep(sample: str) -> Optional[str]:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return None


def load_table(
    csv_path: Path,
    sep: Optional[str] = None,
    decimal: str = ".",
    index_col=None,
) -> pd.DataFrame:
    if sep is None:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(65536)
        autod = sniff_sep(sample)
    else:
        autod = sep

    try:
        df = pd.read_csv(
            csv_path,
            sep=autod if autod else ",",
            decimal=decimal,
            index_col=index_col,
        )

        if df.shape[1] == 1 and sep is None:
            for alt in [";", "\t", "|"]:
                df_alt = pd.read_csv(
                    csv_path,
                    sep=alt,
                    decimal=decimal,
                    index_col=index_col,
                )
                if df_alt.shape[1] > 1:
                    df = df_alt
                    break

        df = df.dropna(axis=1, how="all")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"Missing file: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Failed reading {csv_path}: {e}")


def split_xy(
    df: pd.DataFrame,
    label_col: Optional[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    if df.shape[1] < 2:
        raise ValueError(
            "CSV must contain at least 2 columns: features + label. "
            "Check separator with --sep ';'."
        )

    if label_col is None:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype(str).to_numpy()
    else:
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. Columns: {list(df.columns)}"
            )
        y = df[label_col].astype(str).to_numpy()
        X = df.drop(columns=[label_col])

    timestamp_cols = [
        col for col in X.columns
        if "timestamp" in str(col).strip().lower()
    ]
    if timestamp_cols:
        X = X.drop(columns=timestamp_cols)

    X_num = X.select_dtypes(include=[np.number])

    if X_num.shape[1] != 16 and X_num.shape[1] not in WARNED_FEATURE_COUNTS:
        WARNED_FEATURE_COUNTS.add(X_num.shape[1])
        print(
            f"Warning: expected 16 numeric features, got {X_num.shape[1]}. "
            "Using numeric columns only."
        )

    if X_num.empty:
        raise ValueError("No numeric features available.")

    X_num = X_num.fillna(0.0)

    return X_num.reset_index(drop=True), y


def apply_rolling_mean(
    X: pd.DataFrame,
    y: np.ndarray,
    window: int,
    step: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if window <= 1:
        return X, y

    if step < 1:
        raise ValueError("--rolling-step must be >= 1.")

    if len(X) < window:
        X_avg = X.mean(axis=0).to_frame().T
        return X_avg.reset_index(drop=True), np.asarray([y[-1]], dtype=str)

    X_avg = X.rolling(window=window, min_periods=window).mean().iloc[window - 1::step]
    y_avg = y[window - 1::step]

    return X_avg.reset_index(drop=True), y_avg


def prepare_xy(
    listing: Dict[str, List[str]],
    data_root: Path,
    path_template: str,
    sep: Optional[str],
    decimal: str,
    index_col,
    label_col: Optional[str],
    rolling_window: int,
    rolling_step: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    frames = []
    labels = []

    for class_name, files in listing.items():
        for file_base in files:
            csv_path = resolve_path(data_root, path_template, class_name, file_base)
            df = load_table(csv_path, sep=sep, decimal=decimal, index_col=index_col)
            X_part, y_part = split_xy(df, label_col)
            X_part, y_part = apply_rolling_mean(
                X_part,
                y_part,
                window=rolling_window,
                step=rolling_step,
            )
            frames.append(X_part)
            labels.append(y_part)

    if not frames:
        raise ValueError("Empty dataset after loading split.")

    X = pd.concat(frames, axis=0, ignore_index=True)
    y = np.concatenate(labels)

    return X, y


def build_listing_from_root(
    data_root: Path,
    max_files_per_class: Optional[int] = None,
) -> Dict[str, List[str]]:
    listing = {}

    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        files = sorted(p.stem for p in class_dir.glob("*.csv"))
        if max_files_per_class is not None:
            files = files[:max_files_per_class]
        if files:
            listing[class_dir.name] = files

    if not listing:
        raise ValueError(f"No CSV files found under: {data_root}")

    return listing


def map_gas_to_family(y_gas: np.ndarray) -> np.ndarray:
    families = []
    missing = set()

    for gas in y_gas:
        gas_norm = norm_label(gas)
        family = GAS_TO_FAMILY.get(gas_norm)

        if family is None:
            missing.add(gas_norm)
            family = "unknown"

        families.append(family)

    if missing:
        print("Warning: gas labels missing in GAS_TO_FAMILY:")
        for gas in sorted(missing):
            print(f"  - {gas}")

    return np.asarray(families, dtype=str)


def build_gas_to_family_indices(
    gas_classes: np.ndarray,
    family_classes: np.ndarray,
) -> torch.Tensor:
    family_to_idx = {family: idx for idx, family in enumerate(family_classes)}
    indices = []

    for gas in gas_classes:
        family = GAS_TO_FAMILY.get(norm_label(gas), "unknown")
        if family not in family_to_idx:
            raise ValueError(
                f"Family '{family}' for gas '{gas}' is not present in family encoder classes: "
                f"{list(family_classes)}"
            )
        indices.append(family_to_idx[family])

    return torch.tensor(indices, dtype=torch.long)


def hierarchical_consistency_penalty(
    gas_logits: torch.Tensor,
    family_logits: torch.Tensor,
    y_gas: torch.Tensor,
    gas_to_family_idx: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    pred_gas = torch.argmax(gas_logits, dim=1)
    pred_family = torch.argmax(family_logits, dim=1)
    pred_gas_family = gas_to_family_idx[pred_gas]

    log_probs = torch.log_softmax(gas_logits, dim=1)
    pred_log_probs = log_probs.gather(1, pred_gas.unsqueeze(1)).squeeze(1)
    alpha = -beta * pred_log_probs

    hierarchy_violation = pred_gas_family.ne(pred_family)
    wrong_but_consistent = pred_gas_family.eq(pred_family) & pred_gas.ne(y_gas)

    penalty = torch.zeros_like(alpha) #predizione giusta
    penalty = torch.where(hierarchy_violation, gamma * alpha, penalty) #gas e famiglia incoerenti
    penalty = torch.where(wrong_but_consistent, alpha, penalty) #famiglia coerente, gas errato

    return penalty.mean()


def consistency_accuracy(
    pred_gas: np.ndarray,
    pred_family: np.ndarray,
) -> float:
    """
    Percentuale di coerenza tra:
    gas predetto -> famiglia attesa
    e famiglia predetta dal modello.
    """
    consistent = 0

    for gas, family in zip(pred_gas, pred_family):
        expected_family = GAS_TO_FAMILY.get(norm_label(gas), "unknown")

        if expected_family == norm_label(family):
            consistent += 1

    return consistent / max(len(pred_gas), 1)


def align_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_cols = X_train.columns
    X_test_aligned = X_test.reindex(columns=train_cols, fill_value=0.0)
    return X_train, X_test_aligned


def build_scaler(scaler_name: str):
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "minmax":
        return MinMaxScaler(feature_range=(0.0, 1.0))
    if scaler_name == "none":
        return None
    raise ValueError(f"Unsupported scaler: {scaler_name}")


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_name: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[object]]:
    scaler = build_scaler(scaler_name)
    if scaler is None:
        return (
            X_train.values.astype(np.float32),
            X_test.values.astype(np.float32),
            None,
        )

    X_train_scaled = scaler.fit_transform(X_train.values).astype(np.float32)
    X_test_scaled = scaler.transform(X_test.values).astype(np.float32)
    return X_train_scaled, X_test_scaled, scaler


class TabularGasDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y_gas: np.ndarray,
        y_family: np.ndarray,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_gas = torch.tensor(y_gas, dtype=torch.long)
        self.y_family = torch.tensor(y_family, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_gas[idx], self.y_family[idx]


class FocalLoss(nn.Module):
    """
    Focal Loss per classi sbilanciate o difficili.
    Riduce il peso dei campioni facili e aumenta quello dei campioni difficili.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()

        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()

        if self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss


class GasFamilyMLP(nn.Module):
    """
    MLP aziendale adattato a due task:
    - shared backbone: 16 -> 128 -> 64
    - gas head: 64 -> 25
    - family head: 64 -> 14
    """

    def __init__(self, input_dim: int, num_gas: int, num_family: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.gas_head = nn.Linear(64, num_gas)
        self.family_head = nn.Linear(64, num_family)

    def forward(self, x):
        z = self.shared(x)
        gas_logits = self.gas_head(z)
        family_logits = self.family_head(z)
        return gas_logits, family_logits


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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    gas_preds = []
    family_preds = []

    for X, _, _ in loader:
        X = X.to(device)

        gas_logits, family_logits = model(X)

        pred_gas = torch.argmax(gas_logits, dim=1)
        pred_family = torch.argmax(family_logits, dim=1)

        gas_preds.append(pred_gas.cpu().numpy())
        family_preds.append(pred_family.cpu().numpy())

    return np.concatenate(gas_preds), np.concatenate(family_preds)


def normalized_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)

    return np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    ) * 100.0


def save_heatmap(cm: np.ndarray, labels: List[str], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(max(10, len(labels) * 0.7), max(7, len(labels) * 0.6)))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--splits",
        default=None,
        help="Path to splits JSON file. Not needed when --test-root is used.",
    )
    ap.add_argument("--data-root", default=".", help="Root folder containing CSV data")
    ap.add_argument(
        "--test-root",
        default=None,
        help="External test folder. If set, train uses all CSVs in --data-root.",
    )
    ap.add_argument(
        "--path-template",
        default="{class_name}/{file}.csv",
        help="Relative path template from data-root to CSV",
    )
    ap.add_argument("--sep", default=None, help="CSV separator. Example: --sep ';'")
    ap.add_argument("--decimal", default=".", help="CSV decimal point")
    ap.add_argument("--index-col", default=None, type=str)
    ap.add_argument("--label-col", default=None)
    ap.add_argument(
        "--drop-unseen-test-labels",
        action="store_true",
        help="Drop external test rows whose gas label is absent from training.",
    )
    ap.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Optional debug limit when building listings from --data-root/--test-root.",
    )
    ap.add_argument(
        "--rolling-window",
        type=int,
        default=1,
        help="Number of consecutive rows averaged inside each CSV before training/testing.",
    )
    ap.add_argument(
        "--rolling-step",
        type=int,
        default=1,
        help="Stride between rolling windows. Use the same value as --rolling-window for non-overlapping windows.",
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
    ap.add_argument("--heatmap-dir", default=".", help="Folder where heatmap PNG files are saved")

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
                    f"{message}. Use --drop-unseen-test-labels for closed-set evaluation, "
                    "or add those classes to the training set."
                )

        y_train_family = map_gas_to_family(y_train_gas)
        y_test_family = map_gas_to_family(y_test_gas)

        gas_encoder = LabelEncoder()
        family_encoder = LabelEncoder()

        unseen_family = sorted(set(y_test_family) - set(y_train_family))
        if unseen_family:
            raise ValueError(
                "Test family labels absent from training: "
                f"{unseen_family}. Add training examples or drop those labels."
            )

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
        if args.rolling_window > 1:
            print(
                f"Rolling mean: window={args.rolling_window}, "
                f"step={args.rolling_step}"
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
        if args.n_iter_no_change > 0:
            print(
                f"Convergence stop: tol={args.tol}, "
                f"n_iter_no_change={args.n_iter_no_change}"
            )
        else:
            print("Convergence stop: disabled")

        model = GasFamilyMLP(
            input_dim=input_dim,
            num_gas=num_gas,
            num_family=num_family,
        ).to(device)

        gas_to_family_idx = build_gas_to_family_indices(
            gas_encoder.classes_,
            family_encoder.classes_,
        ).to(device)

        if args.loss_mode == "focal":
            criterion = FocalLoss(gamma=args.focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss()

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

        consistency_acc = consistency_accuracy(
            y_pred_gas,
            y_pred_family,
        )

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

            save_heatmap(
                gas_cm,
                gas_labels,
                gas_heatmap_path,
                f"GAS confusion matrix (%) - {fold_name}",
            )
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
                "consistency_accuracy": consistency_acc,
                "n_train_samples": len(train_dataset),
                "n_test_samples": len(test_dataset),
                "input_dim": input_dim,
                "num_gas_classes": num_gas,
                "num_family_classes": num_family,
                "epochs_run": actual_epochs,
                "rolling_window": args.rolling_window,
                "rolling_step": args.rolling_step,
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
                "architecture": "MLP 16/num_features -> 128 -> 64 -> gas_head + family_head",
                "epochs_run": actual_epochs,
                "max_epochs": args.epochs,
                "tol": args.tol,
                "n_iter_no_change": args.n_iter_no_change,
                "rolling_window": args.rolling_window,
                "rolling_step": args.rolling_step,
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

            torch.save(payload, out_path)
            print(f"Saved model to: {out_path}")

    if fold_results:
        df_results = pd.DataFrame(fold_results)

        print("\n=== Summary ===")
        print(f"Mean GAS accuracy: {df_results['gas_accuracy'].mean():.6f} ± {df_results['gas_accuracy'].std(ddof=1):.6f}")
        print(f"Mean FAMILY accuracy: {df_results['family_accuracy'].mean():.6f} ± {df_results['family_accuracy'].std(ddof=1):.6f}")
        print(f"Mean CONSISTENCY accuracy: {df_results['consistency_accuracy'].mean():.6f} ± {df_results['consistency_accuracy'].std(ddof=1):.6f}")

        if args.save_report:
            df_results.to_csv(args.save_report, index=False)
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
    main()
