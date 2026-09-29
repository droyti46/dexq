"""Training-free image rules for lumbar coverage.

The lower branch measures the amount of bone-like signal in fixed lower-side
regions.  The upper branch traces an anonymous spinal centerline and measures
how far a regular sequence of dark intervertebral gaps extends towards the top
of the image.  Neither branch assigns a vertebral level.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_closing,
    gaussian_filter,
    gaussian_filter1d,
    label,
    map_coordinates,
    uniform_filter1d,
)
from scipy.signal import find_peaks


# Scalar development calibration retained for every image in the one-model
# inference pipeline. These pixel-rule constants are not fitted ML models.
LOWER_THRESHOLD, LOWER_SCALE = (-0.01941480441018939, 0.3362742066383362)
UPPER_THRESHOLD, UPPER_SCALE = (-4.228571428571429, 0.994741598607578)


def _roi(a: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    h, w = a.shape
    return a[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]


def lower_lateral_score(image: np.ndarray) -> tuple[float, float]:
    """Return ``(feature, violation_score)`` for the iliac-crest proxy."""
    a = np.asarray(image, dtype=np.float32) / 255.0
    left = _roi(a, 0.00, 0.34, 0.68, 1.00)
    right = _roi(a, 0.66, 1.00, 0.68, 1.00)
    side = np.concatenate([left.ravel(), right.ravel()])
    whole = _roi(a, 0.05, 0.95, 0.05, 0.95)
    feature = float(np.mean(side) / (float(np.mean(whole)) + 1e-6))
    return feature, -feature


def _row_normalize(image: np.ndarray) -> np.ndarray:
    a = image.astype(np.float32) / 255.0
    central = a[:, round(.12 * a.shape[1]):round(.88 * a.shape[1])]
    lo, hi = np.quantile(central, [.05, .97], axis=1)
    a = np.clip((a - lo[:, None]) / np.maximum(.08, hi - lo)[:, None], 0, 1)
    return gaussian_filter(a, sigma=(1.2, 1.4))


def _response_maps(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, w = a.shape
    d = max(3, round(.025 * w))
    left = np.roll(a, -d, axis=1) - np.roll(a, d, axis=1)
    right = np.roll(a, d, axis=1) - np.roll(a, -d, axis=1)
    d2 = max(6, round(.05 * w))
    left += .55 * (np.roll(a, -d2, axis=1) - np.roll(a, d2, axis=1))
    right += .55 * (np.roll(a, d2, axis=1) - np.roll(a, -d2, axis=1))
    left[:, :d2 + 2] = left[:, -d2 - 2:] = -3
    right[:, :d2 + 2] = right[:, -d2 - 2:] = -3
    return left, right


def _centerline(image: np.ndarray, *, return_edges: bool = False):
    """Dynamic-programming center; optionally return its selected edge pair.

    The edge pair is a pixel-response proxy, not an anatomical segmentation.
    ``return_edges`` records widths for visualization without changing scores
    or the default centerline used by inference.
    """
    a = _row_normalize(image)
    h, w = a.shape
    left_response, right_response = _response_maps(a)
    centers = np.arange(round(.27 * w), round(.73 * w) + 1)
    widths = np.arange(max(10, round(.085 * w)), max(12, round(.215 * w)) + 1, 2)
    scores = np.full((h, len(centers)), -1e9, np.float32)
    selected_widths = np.zeros(scores.shape, np.int16) if return_edges else None
    preferred_width = .14 * w
    for width in widths:
        left, right = centers - width, centers + width
        valid = (left >= 2) & (right < w - 2)
        c, l, r = centers[valid], left[valid], right[valid]
        boundary = left_response[:, l] + right_response[:, r]
        inside = uniform_filter1d(a, size=max(3, 2 * width - 4), axis=1, mode="nearest")[:, c]
        outside = .5 * (a[:, np.maximum(0, l - 4)] + a[:, np.minimum(w - 1, r + 4)])
        value = boundary + .35 * (inside - outside) - .16 * ((width - preferred_width) / (.07 * w)) ** 2
        if return_edges:
            selected_widths[:, valid] = np.where(value > scores[:, valid], width,
                                                 selected_widths[:, valid])
        scores[:, valid] = np.maximum(scores[:, valid], value)
    median = np.median(scores, axis=1, keepdims=True)
    z = (scores - median) / (np.median(np.abs(scores - median), axis=1, keepdims=True) + .03)
    count = len(centers)
    back = np.zeros((h, count), np.int16)
    dynamic = z[0].copy()
    delta = centers[:, None] - centers[None, :]
    penalty = .22 * np.abs(delta) + .035 * delta ** 2
    for y in range(1, h):
        candidates = dynamic[None, :] - penalty
        back[y] = np.argmax(candidates, axis=1)
        dynamic = (z[y] + candidates[np.arange(count), back[y]]).astype(np.float32)
    indices = np.zeros(h, np.int16)
    indices[-1] = int(np.argmax(dynamic))
    for y in range(h - 1, 0, -1):
        indices[y - 1] = back[y, indices[y]]
    centerline = centers[indices].astype(float)
    if not return_edges:
        return centerline
    half_width = selected_widths[np.arange(h), indices].astype(float)
    if np.any(half_width <= 0):
        raise ValueError("No paired edges found for a selected center")
    return centerline, centerline - half_width, centerline + half_width


def _normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 64 or not np.isfinite(image).all():
        raise ValueError("Expected a finite grayscale image with both dimensions >= 64")
    lo, hi = np.percentile(image, [2.0, 98.0])
    if hi - lo < 1e-6:
        return np.zeros(image.shape, np.float32)
    return np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _estimate_period(a: np.ndarray, center: np.ndarray):
    h, w = a.shape
    offsets = np.linspace(-.12 * w, .12 * w, 41)
    offsets = offsets[np.abs(offsets) > .025 * w]
    ys = np.arange(h)
    strip = map_coordinates(
        a,
        [np.broadcast_to(ys[:, None], (h, len(offsets))), center[:, None] + offsets[None, :]],
        order=1,
        mode="nearest",
    )
    profile = np.mean(strip, axis=1)
    smoothed = gaussian_filter1d(profile, max(.8, .006 * w))
    detrended = smoothed - gaussian_filter1d(smoothed, max(2.0, .09 * w))
    lo, hi = max(0, round(h - .85 * w)), max(2, round(h - .13 * w))
    usable = detrended[lo:hi] - np.mean(detrended[lo:hi])
    low, high = max(6, round(.08 * w)), min(round(.24 * w), len(usable) // 2)
    if high <= low or np.std(usable) < .005:
        return .15 * w, 0.0
    correlations = np.array([
        float(np.dot(usable[:-lag], usable[lag:]) /
              (np.linalg.norm(usable[:-lag]) * np.linalg.norm(usable[lag:]) + 1e-9))
        for lag in range(low, high + 1)
    ])
    lags = np.arange(low, high + 1)
    ranking = correlations - .10 * np.abs(np.log(lags / (.15 * w)))
    peaks = find_peaks(ranking)[0]
    index = int(peaks[np.argmax(ranking[peaks])]) if len(peaks) else int(np.argmax(ranking))
    return float(lags[index]), float(correlations[index])


def _gap_response(a: np.ndarray, center: np.ndarray, period: float):
    h, w = a.shape
    dx = np.linspace(-.115 * w, .115 * w, 31)
    dx = dx[np.abs(dx) > .02 * w]
    ys = np.arange(h, dtype=float)[:, None]
    xs = center[:, None] + dx[None, :]
    slopes_to_try = np.asarray([-.35, -.175, 0.0, .175, .35])
    contrasts = []
    for slope in slopes_to_try:
        yy = ys + slope * dx[None, :]
        middle = map_coordinates(a, [yy, xs], order=1, mode="nearest")
        above = map_coordinates(a, [yy - .22 * period, xs], order=1, mode="nearest")
        below = map_coordinates(a, [yy + .22 * period, xs], order=1, mode="nearest")
        difference = .5 * (above + below) - middle
        left, right = np.median(difference[:, dx < 0], axis=1), np.median(difference[:, dx > 0], axis=1)
        value = np.maximum(0.0, .5 * (left + right) - .30 * np.abs(left - right))
        value *= np.mean(difference > 0, axis=1)
        valid = (yy.min(axis=1) - .22 * period >= 0) & (yy.max(axis=1) + .22 * period <= h - 1)
        valid &= (xs.min(axis=1) >= 0) & (xs.max(axis=1) <= w - 1)
        value[~valid] = 0
        contrasts.append(value)
    contrasts = np.asarray(contrasts)
    selected = contrasts.argmax(axis=0)
    response = gaussian_filter1d(contrasts[selected, np.arange(h)], max(.6, .003 * w))
    response[:max(1, round(.12 * period))] = 0
    response[h - max(1, round(.15 * period)):] = 0
    return response, slopes_to_try[selected]


def _otsu(a: np.ndarray) -> float:
    histogram = np.bincount(np.clip(a, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(float)
    probability = histogram / max(histogram.sum(), 1)
    cumulative = np.cumsum(probability)
    mean = np.cumsum(probability * np.arange(256))
    between = (mean[-1] * cumulative - mean) ** 2 / np.maximum(cumulative * (1 - cumulative), 1e-12)
    return float(np.argmax(between))


def _pelvic_components(image: np.ndarray) -> list[dict | None]:
    h, w = image.shape
    output = []
    for x0, x1 in [(round(.02 * w), round(.34 * w)), (round(.66 * w), round(.98 * w))]:
        y0 = round(.55 * h)
        roi = gaussian_filter(image[y0:, x0:x1].astype(float), 1.0)
        threshold = max(20.0, _otsu(roi))
        mask = binary_closing(np.pad(roi > threshold, 2, mode="edge"), iterations=1)[2:-2, 2:-2]
        labels, count = label(mask)
        options = []
        for component in range(1, count + 1):
            yy, xx = np.where(labels == component)
            if len(yy) < max(30, .001 * h * w) or yy.max() < roi.shape[0] - max(4, round(.04 * h)):
                continue
            if np.ptp(yy) < .04 * h or np.ptp(xx) < .04 * w:
                continue
            options.append((len(yy), yy, xx))
        if not options:
            output.append(None)
            continue
        _, yy, xx = max(options, key=lambda item: item[0])
        top = float(np.quantile(yy, .01))
        near = yy <= top + 2
        output.append({
            "x": float(np.median(xx[near])) + x0,
            "y": top + y0,
            "threshold": threshold,
            "size": int(len(yy)),
            "at_search_border": bool(yy.min() <= 1),
        })
    return output


def _anchored_chain(peaks, response, period, anchor):
    peaks = np.asarray(peaks, int)
    if not len(peaks):
        return [], None
    reward = 1 + np.clip(response[peaks] / .10, 0, 2)
    total, previous = reward.copy(), np.full(len(peaks), -1, int)
    for j in range(len(peaks)):
        for i in range(j):
            spacing = (peaks[j] - peaks[i]) / period
            if .55 <= spacing <= 1.55:
                candidate = total[i] + reward[j] - 2 * ((spacing - 1) / .5) ** 2
                if candidate > total[j]:
                    total[j], previous[j] = candidate, i
    target = anchor + .45 * period
    allowed = np.flatnonzero(np.abs(peaks - target) <= .80 * period)
    if not len(allowed):
        return [], None
    j = int(max(allowed, key=lambda k: total[k] - .5 * abs(peaks[k] - target) / period))
    end, chain = int(peaks[j]), []
    while j >= 0:
        chain.append(int(peaks[j]))
        j = int(previous[j])
    return chain[::-1], end


def upper_gap_sequence(image: np.ndarray) -> dict:
    """Measure anonymous vertical gap coverage; low confidence abstains."""
    normalized = _normalize(image)
    h, w = normalized.shape
    normalized = gaussian_filter(normalized, (max(.6, .002 * w), max(.6, .003 * w)))
    center = _centerline(np.uint8(np.rint(normalized * 255)))
    period, period_confidence = _estimate_period(normalized, center)
    response, slopes = _gap_response(normalized, center, period)
    height = max(.025, .15 * float(np.max(response)))
    peaks = find_peaks(response, distance=max(3, round(.55 * period)), prominence=.015, height=height)[0]
    pelvis = _pelvic_components(np.uint8(np.rint(normalized * 255)))
    reliable = [point for point in pelvis if point is not None and not point["at_search_border"]]
    reasons, anchor, chain, last = [], None, [], None
    if len(reliable) != 2:
        reasons.append("bilateral_lower_anchor_not_found")
    else:
        anchor = float(np.mean([point["y"] for point in reliable]))
        if abs(reliable[0]["y"] - reliable[1]["y"]) > 1.25 * period:
            reasons.append("bilateral_anchor_disagreement")
        chain, last = _anchored_chain(peaks, response, period, anchor)
    if len(chain) < 3:
        reasons.append("fewer_than_three_observed_gaps")
    spacing = np.diff(chain)
    spacing_cv = float(np.std(spacing) / (np.mean(spacing) + 1e-9)) if len(spacing) else None
    if spacing_cv is not None and spacing_cv > .32:
        reasons.append("irregular_gap_spacing")
    if period_confidence < .05:
        reasons.append("weak_periodicity")
    if chain and chain[0] > 1.6 * period:
        reasons.append("upper_sequence_not_observed")
    local_period = float(np.median(spacing)) if len(spacing) >= 2 else period
    visible_units = float(len(chain) - 1 + chain[0] / local_period) if chain else None
    gaps = []
    for y in chain:
        slope, x, half = float(slopes[y]), float(center[y]), .115 * w
        gaps.append({
            "x": x,
            "y": float(y),
            "slope": slope,
            "contrast": float(response[y]),
            "endpoints": [[x - half, float(y - slope * half)], [x + half, float(y + slope * half)]],
        })
    confident = not reasons
    return {
        "confident": confident,
        "unknown_reasons": reasons,
        "period_px": period,
        "period_confidence": period_confidence,
        "local_period_px": local_period,
        "spacing_cv": spacing_cv,
        "anchor_y": anchor,
        "anchor_gap_y": last,
        "visible_units": visible_units,
        "upper_missing_score": -visible_units if confident else None,
        "observed_gaps": gaps,
        "pelvic_proxies": pelvis,
        "inferred_anatomical_level": None,
        "th12_visible": None,
    }


def measure_coverage(image: np.ndarray) -> dict:
    """Combine lower pixel sum and upper gap search with a Boolean OR."""
    lower_threshold, lower_scale = LOWER_THRESHOLD, LOWER_SCALE
    upper_threshold, upper_scale = UPPER_THRESHOLD, UPPER_SCALE
    feature, lower_score = lower_lateral_score(image)
    lower_margin = float((lower_score - lower_threshold) / lower_scale)
    lower_violation = bool(lower_score >= lower_threshold)
    sequence = upper_gap_sequence(image)
    if sequence["upper_missing_score"] is None:
        upper_margin, upper_violation, upper_status = None, False, "unknown_branch_abstained"
    else:
        upper_margin = float((sequence["upper_missing_score"] - upper_threshold) / upper_scale)
        upper_violation, upper_status = upper_margin > 0, "measured"
    combined_score = max(lower_margin, upper_margin if upper_margin is not None else -1e6)
    return {
        "method": "normalized_lower_lateral_sum_or_anonymous_gap_sequence",
        "score": float(combined_score),
        "threshold": 0.0,
        "violation": bool(lower_violation or upper_violation),
        "status": "research_prediction",
        "score_semantics": "maximum standardized violation margin; positive triggers a branch",
        "calibration": "full_development_set",
        "model_training": False,
        "components": {
            "lower_lateral_sum": {
                "feature_value": feature,
                "violation_score": lower_score,
                "threshold": lower_threshold,
                "scale": lower_scale,
                "margin": lower_margin,
                "violation": lower_violation,
            },
            "upper_gap_sequence": {
                **sequence,
                "threshold": upper_threshold,
                "scale": upper_scale,
                "margin": upper_margin,
                "violation": bool(upper_violation),
                "status": upper_status,
            },
        },
        "anatomical_evidence": {
            "status": "anonymous_geometric_proxies_only",
            "t12_level_assigned": False,
            "iliac_crests_anatomically_validated": False,
        },
    }
