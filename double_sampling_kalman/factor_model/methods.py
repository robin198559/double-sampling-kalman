from typing import List, Tuple

import numpy as np


def moving_std(a, window_size=3) -> np.array:
    assert window_size > 1
    return np.array([np.std(a[i - window_size : i]) for i in range(window_size, len(a) + 1)])


def calculate_filter_convergence(error_cov_list: List[float], window_size: int) -> List[float]:
    if len(error_cov_list) > window_size:
        return moving_std(error_cov_list, window_size=window_size).tolist()
    else:
        return []


def stop_filter_scan(convergence_series: List[float], window_size: int, convergence_log10_tol: float):
    if len(convergence_series) > window_size:
        within_range = (np.array(convergence_series) < convergence_log10_tol).tolist()
        if np.sum(within_range[-window_size:]) >= window_size:
            return True
    return False


def construct_multiplier_list(multiplier_granularity: int, multiplier_log10_width: float) -> List[float]:
    assert multiplier_granularity > 1
    assert multiplier_log10_width > 0
    log_filter_multipliers = np.arange(
        -multiplier_log10_width,
        multiplier_log10_width,
        2 * multiplier_log10_width / int(multiplier_granularity),
    ).tolist() + [multiplier_log10_width]
    return np.pow(10, log_filter_multipliers)


def calculate_filter_signal(
    forward: np.ndarray,
    backward: np.ndarray,
) -> float:
    filter_displacement = np.mean((forward - backward)[:, :, 0], axis=0)
    model_displacement = np.std((forward - backward)[:, :, 0], axis=0)

    np.seterr(invalid="raise")
    try:
        sig = np.sqrt(np.sum(np.pow(filter_displacement / model_displacement, 2)))
    except FloatingPointError as err:
        raise FloatingPointError(f"{err}. Last filter values: Most likely the filter exploded.")
    return float(np.log10(sig))


def calculate_filter_diff_squared_log10(
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    filter_displacement = np.sqrt(np.sum(np.pow((x1 - x2)[:, :, 0], 2), axis=0) / x1.shape[0])
    sig = np.sqrt(np.sum(np.pow(filter_displacement, 2)))
    return float(np.log10(sig))


def zoom_in_signal_multipliers(
    current_multipliers: List[float],
    current_signal: List[float],
    filter_tuning_multiplier_granularity: int,
) -> Tuple[float, float]:
    convex = np.diff(np.diff(current_signal))
    positive_convex_index_list = [i for i, v in enumerate(convex) if v > 0 and v > max(convex) * 0.8]
    left_convex_index = max(positive_convex_index_list) if positive_convex_index_list else 0
    negative_convex_index_list = [
        i for i, v in enumerate(convex) if v < 0 and v < min(convex) * 0.8 and i > left_convex_index
    ]
    large_signal_convexity_index = negative_convex_index_list + positive_convex_index_list
    left_wing = current_multipliers[min(large_signal_convexity_index)]
    right_wing = current_multipliers[max(large_signal_convexity_index)]
    multiplier_log10_width = max(
        abs(np.log10(left_wing)),
        abs(np.log10(right_wing)),
    )
    multiplier_log10_width = min(multiplier_log10_width + 0.5, max(np.log10(current_multipliers)))
    return filter_tuning_multiplier_granularity, multiplier_log10_width
