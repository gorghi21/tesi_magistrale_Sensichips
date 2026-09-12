"""Statistical feature extraction for SCA sensor time series."""

import re

import numpy as np
import pandas as pd


SNAU_PATTERN = re.compile(
    r"^OFFCHIP_SENSIMOX_SnAu_(?P<temp>\d+)_(?P<freq>\d+(?:\.\d+)?)_IN-PHASE$",
    re.IGNORECASE,
)


def safe_scalar(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _window_stats(values: np.ndarray, prefix: str) -> dict:
    if values.size == 0:
        return {}

    first = safe_scalar(values[0])
    last = safe_scalar(values[-1])
    minimum = safe_scalar(np.min(values))
    maximum = safe_scalar(np.max(values))
    q25 = safe_scalar(np.quantile(values, 0.25))
    q75 = safe_scalar(np.quantile(values, 0.75))
    if values.size > 1:
        index = np.arange(values.size, dtype=np.float32)
        slope = safe_scalar(np.polyfit(index, values, 1)[0])
        differences = np.diff(values)
        diff_mean = safe_scalar(np.mean(differences))
        diff_std = safe_scalar(np.std(differences))
        diff_abs_mean = safe_scalar(np.mean(np.abs(differences)))
        max_jump = safe_scalar(np.max(np.abs(differences)))
    else:
        slope = diff_mean = diff_std = diff_abs_mean = max_jump = 0.0
    if values.size > 2:
        skew = safe_scalar(pd.Series(values).skew())
        kurtosis = safe_scalar(pd.Series(values).kurt())
    else:
        skew = kurtosis = 0.0

    statistics = {
        "mean": safe_scalar(np.mean(values)), "std": safe_scalar(np.std(values)),
        "min": minimum, "max": maximum, "median": safe_scalar(np.median(values)),
        "q25": q25, "q75": q75, "iqr": q75 - q25, "first": first, "last": last,
        "delta": last - first, "range": maximum - minimum,
        "mean_abs": safe_scalar(np.mean(np.abs(values))),
        "rms": safe_scalar(np.sqrt(np.mean(np.square(values)))),
        "auc": safe_scalar(np.trapezoid(values)), "slope": slope,
        "peak_idx_rel": safe_scalar(np.argmax(values) / max(values.size - 1, 1)),
        "min_idx_rel": safe_scalar(np.argmin(values) / max(values.size - 1, 1)),
        "diff_mean": diff_mean, "diff_std": diff_std, "diff_abs_mean": diff_abs_mean,
        "max_jump": max_jump, "skew": skew, "kurtosis": kurtosis,
    }
    return {f"{prefix}__{name}": safe_scalar(value) for name, value in statistics.items()}


def _profile_stats(x_axis: np.ndarray, values: np.ndarray, prefix: str) -> dict:
    features = _window_stats(values.astype(np.float32), prefix)
    if values.size == 0:
        return features
    if x_axis.size == values.size and values.size > 1:
        differences = np.diff(values)
        features.update({
            f"{prefix}__x_slope": safe_scalar(np.polyfit(x_axis.astype(np.float32), values.astype(np.float32), 1)[0]),
            f"{prefix}__peak_x": safe_scalar(x_axis[int(np.argmax(values))]),
            f"{prefix}__min_x": safe_scalar(x_axis[int(np.argmin(values))]),
            f"{prefix}__x_auc": safe_scalar(np.trapezoid(values.astype(np.float32), x_axis.astype(np.float32))),
            f"{prefix}__x_diff_mean": safe_scalar(np.mean(differences)) if differences.size else 0.0,
            f"{prefix}__x_diff_abs_mean": safe_scalar(np.mean(np.abs(differences))) if differences.size else 0.0,
            f"{prefix}__x_max_jump": safe_scalar(np.max(np.abs(differences))) if differences.size else 0.0,
        })
    else:
        features.update({f"{prefix}__{name}": 0.0 for name in ("x_slope", "peak_x", "min_x", "x_auc", "x_diff_mean", "x_diff_abs_mean", "x_max_jump")})
    return features


def _parse_sensor_groups(columns: list[str]) -> tuple[dict[float, list[tuple[int, str]]], list[str], list[str]]:
    snau_groups: dict[float, list[tuple[int, str]]] = {}
    onchip_columns, other_columns = [], []
    for column in columns:
        match = SNAU_PATTERN.match(column)
        if match:
            snau_groups.setdefault(float(match.group("freq")), []).append((int(match.group("temp")), column))
        elif "ONCHIP_ALUMINUM_OXIDE" in column.upper():
            onchip_columns.append(column)
        else:
            other_columns.append(column)
    for frequency in snau_groups:
        snau_groups[frequency] = sorted(snau_groups[frequency], key=lambda pair: pair[0])
    return snau_groups, onchip_columns, other_columns


def aggregate_numeric_timeseries(data: pd.DataFrame, n_windows: int = 4) -> pd.DataFrame:
    """Build the statistical representation used by the thesis experiments."""
    if data.empty:
        return pd.DataFrame([{}], dtype=np.float32)
    n_windows = max(1, int(n_windows))
    features = {"__n_rows": float(len(data)), "__n_windows": float(n_windows)}
    snau_groups, onchip_columns, other_columns = _parse_sensor_groups(list(data.columns))
    segments = [("full", data)]
    for index, segment in enumerate(np.array_split(data, n_windows), start=1):
        segments.append((f"w{index}", segment.reset_index(drop=True)))

    for segment_name, segment in segments:
        for column in onchip_columns + other_columns:
            values = pd.to_numeric(segment[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            features.update(_window_stats(values, f"{column}__{segment_name}"))

    for segment_name, segment in segments:
        for frequency, pairs in snau_groups.items():
            temperatures = np.asarray([temperature for temperature, _ in pairs], dtype=np.float32)
            columns = [column for _, column in pairs]
            matrix = segment[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if matrix.size == 0:
                continue
            tag = str(int(frequency)) if frequency.is_integer() else str(frequency).replace(".", "p")
            base = f"SnAu__freq_{tag}__{segment_name}"
            for name, profile in (("mean", matrix.mean(axis=0)), ("std", matrix.std(axis=0)), ("min", matrix.min(axis=0)), ("max", matrix.max(axis=0))):
                features.update(_profile_stats(temperatures, profile, f"{base}__temp_profile_{name}"))
            for temperature, value, spread in zip(temperatures.astype(int), matrix.mean(axis=0), matrix.std(axis=0)):
                features[f"{base}__temp_{temperature}__mean"] = safe_scalar(value)
                features[f"{base}__temp_{temperature}__std"] = safe_scalar(spread)

        if len(snau_groups) >= 2:
            frequencies = sorted(snau_groups)
            common_temperatures = sorted(set(temp for temp, _ in snau_groups[frequencies[0]]).intersection(*[set(temp for temp, _ in snau_groups[frequency]) for frequency in frequencies[1:]]))
            if common_temperatures:
                means = {}
                for frequency, pairs in snau_groups.items():
                    columns = [column for _, column in pairs]
                    temperatures = [temperature for temperature, _ in pairs]
                    profile = segment[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=0).to_numpy(dtype=np.float32)
                    means[frequency] = {temperature: safe_scalar(value) for temperature, value in zip(temperatures, profile)}
                low, high = frequencies[:2]
                differences = np.asarray([means[high][temp] - means[low][temp] for temp in common_temperatures], dtype=np.float32)
                low_tag = str(int(low)) if low.is_integer() else str(low).replace(".", "p")
                high_tag = str(int(high)) if high.is_integer() else str(high).replace(".", "p")
                prefix = f"SnAu__freqdiff_{high_tag}_minus_{low_tag}__{segment_name}"
                features.update(_profile_stats(np.asarray(common_temperatures, dtype=np.float32), differences, prefix))
                for temperature, value in zip(common_temperatures, differences):
                    features[f"{prefix}__temp_{temperature}"] = safe_scalar(value)
    return pd.DataFrame([features], dtype=np.float32)
