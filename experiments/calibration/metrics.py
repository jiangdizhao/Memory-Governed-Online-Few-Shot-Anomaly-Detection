"""Offline-only evaluator. No controller receives this module or its labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass
class PrivateOutcomeLedger:
    _labels: tuple[int, ...]
    def __init__(self, labels: Iterable[int]):
        values = tuple(int(y) for y in labels)
        if any(y not in (0, 1) for y in values):
            raise ValueError("Evaluator labels must be binary")
        self._labels = values
    def __len__(self) -> int: return len(self._labels)
    def _disclose_to_engine(self, t: int, queried: bool) -> int | None:
        if not 0 <= t < len(self._labels):
            raise IndexError("Outcome index outside private ledger")
        return self._labels[t] if queried else None
    def evaluate(self, scores: Sequence[float],
                 predictions: dict[str, Sequence[int | None]], *,
                 prediction_availability: dict[str, Sequence[bool]] | None = None) -> dict:
        if len(scores) != len(self._labels):
            raise ValueError("Score/outcome stream lengths differ")
        if any(not math.isfinite(float(score)) for score in scores):
            raise ValueError("Nonfinite score in offline evaluation")
        methods = {}
        prediction_availability = prediction_availability or {}
        for name, labels in predictions.items():
            if len(labels) != len(self._labels):
                raise ValueError(f"Prediction stream length differs for {name}")
            available = list(prediction_availability.get(name, [True] * len(labels)))
            if len(available) != len(labels):
                raise ValueError(f"Prediction-availability stream length differs for {name}")
            if any((not is_available) and pred is not None
                   for pred, is_available in zip(labels, available)):
                raise ValueError(f"Unavailable prediction carries a binary label for {name}")
            tn = fp = fn = tp = abstain = abstain_normal = abstain_defect = 0
            missing = missing_normal = missing_defect = 0
            current_error_run = longest_error_run = 0
            for y, pred, is_available in zip(self._labels, labels, available):
                if pred is None:
                    if is_available:
                        abstain += 1
                        if y == 0: abstain_normal += 1
                        else: abstain_defect += 1
                    else:
                        missing += 1
                        if y == 0: missing_normal += 1
                        else: missing_defect += 1
                    current_error_run = 0
                elif y == 0 and pred == 0: tn += 1
                elif y == 0 and pred == 1: fp += 1
                elif y == 1 and pred == 0: fn += 1
                elif y == 1 and pred == 1: tp += 1
                else: raise ValueError("Nonbinary prediction")
                if pred is not None:
                    if pred != y:
                        current_error_run += 1
                        longest_error_run = max(longest_error_run, current_error_run)
                    else:
                        current_error_run = 0
            n_normal, n_defect = self._labels.count(0), self._labels.count(1)
            fpr_answered = fp / (tn + fp) if tn + fp else None
            fnr_answered = fn / (fn + tp) if fn + tp else None
            fpr = (fp / n_normal if n_normal and not (abstain_normal or missing_normal) else None)
            fnr = (fn / n_defect if n_defect and not (abstain_defect or missing_defect) else None)
            ba = (None if abstain or missing or fpr is None or fnr is None
                  else 1 - (fpr + fnr) / 2)
            methods[name] = {
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "abstentions": abstain, "normal_abstentions": abstain_normal,
                "defect_abstentions": abstain_defect,
                "missing_predictions": missing,
                "normal_missing_predictions": missing_normal,
                "defect_missing_predictions": missing_defect,
                "withheld_predictions": abstain + missing,
                "prediction_availability_fraction": (
                    (len(labels) - missing) / len(labels) if labels else None),
                "coverage": (len(labels) - abstain - missing) / len(labels) if labels else None,
                "normal_denominator": n_normal, "defect_denominator": n_defect,
                "fpr": fpr, "fnr": fnr, "balanced_accuracy": ba,
                "fpr_answered_only": fpr_answered,
                "fnr_answered_only": fnr_answered,
                "longest_consecutive_errors": longest_error_run,
                "balanced_accuracy_status": "UNAVAILABLE_MISSING_PREDICTIONS" if missing else
                    "UNAVAILABLE_ABSTENTION" if abstain else
                    "UNAVAILABLE_CLASS_ABSENT" if fpr is None or fnr is None else "DEFINED",
                "windows": _window_metrics(self._labels, labels, window_size=256,
                                           prediction_availability=available),
            }
        auc, auprc, auc_status = _ranking_metrics(self._labels, scores)
        return {"n": len(self._labels), "n_normal": self._labels.count(0),
                "n_defect": self._labels.count(1), "methods": methods,
                "shared_score_auroc": auc, "shared_score_auprc": auprc,
                "ranking_status": auc_status}


def _window_metrics(labels: Sequence[int], predictions: Sequence[int | None],
                    *, window_size: int,
                    prediction_availability: Sequence[bool] | None = None) -> dict:
    prediction_availability = (list(prediction_availability) if
                               prediction_availability is not None else
                               [True] * len(predictions))
    if len(prediction_availability) != len(predictions):
        raise ValueError("Window prediction-availability stream length differs")
    windows = []
    for begin in range(0, len(labels), window_size):
        end = min(len(labels), begin + window_size)
        ys, ps = labels[begin:end], predictions[begin:end]
        av = prediction_availability[begin:end]
        tn = fp = fn = tp = abstain_n = abstain_d = 0
        missing_n = missing_d = 0
        for y, pred, is_available in zip(ys, ps, av):
            if pred is None:
                if is_available:
                    if y == 0: abstain_n += 1
                    else: abstain_d += 1
                elif y == 0: missing_n += 1
                else: missing_d += 1
            elif y == 0 and pred == 0: tn += 1
            elif y == 0 and pred == 1: fp += 1
            elif y == 1 and pred == 0: fn += 1
            elif y == 1 and pred == 1: tp += 1
        n_normal = sum(y == 0 for y in ys)
        n_defect = sum(y == 1 for y in ys)
        fpr = fp / n_normal if n_normal and not (abstain_n or missing_n) else None
        fnr = fn / n_defect if n_defect and not (abstain_d or missing_d) else None
        ba = 1 - (fpr + fnr) / 2 if fpr is not None and fnr is not None else None
        status = ("DEFINED" if ba is not None else
                  "UNAVAILABLE_CLASS_ABSENT" if not n_normal or not n_defect else
                  "UNAVAILABLE_MISSING_PREDICTIONS" if missing_n or missing_d else
                  "UNAVAILABLE_ABSTENTION")
        windows.append({"start": begin, "stop_exclusive": end, "n": end - begin,
                        "n_normal": n_normal, "n_defect": n_defect,
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        "normal_abstentions": abstain_n,
                        "defect_abstentions": abstain_d,
                        "normal_missing_predictions": missing_n,
                        "defect_missing_predictions": missing_d,
                        "fpr": fpr, "fnr": fnr,
                        "balanced_accuracy": ba, "status": status})
    return {
        "window_size": window_size,
        "windows": windows,
        "worst_defined_fpr": max((x["fpr"] for x in windows if x["fpr"] is not None),
                                 default=None),
        "worst_defined_fnr": max((x["fnr"] for x in windows if x["fnr"] is not None),
                                 default=None),
        "worst_defined_balanced_accuracy": min(
            (x["balanced_accuracy"] for x in windows
             if x["balanced_accuracy"] is not None), default=None),
    }


def _ranking_metrics(labels: Sequence[int], scores: Sequence[float]):
    positives = sum(y == 1 for y in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None, None, "UNAVAILABLE_CLASS_ABSENT"
    pairs = sorted(zip(map(float, scores), labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        mean_rank = ((i + 1) + j) / 2
        rank_sum += mean_rank * sum(y == 1 for _, y in pairs[i:j])
        i = j
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    descending = sorted(zip(map(float, scores), labels), key=lambda x: -x[0])
    tp = fp = area = 0
    last_recall = 0.0
    i = 0
    while i < len(descending):
        j = i + 1
        while j < len(descending) and descending[j][0] == descending[i][0]: j += 1
        tp += sum(y == 1 for _, y in descending[i:j])
        fp += sum(y == 0 for _, y in descending[i:j])
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - last_recall) * precision
        last_recall = recall
        i = j
    return float(auc), float(area), "DEFINED"
