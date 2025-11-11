"""Utility functions using pure Python numeric operations."""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def normalize_rows(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    result: List[List[float]] = []
    for row in matrix:
        norm = l2_norm(row)
        if norm == 0:
            result.append([float(x) for x in row])
        else:
            result.append([float(x) / norm for x in row])
    return result


def relu_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, dot(a, b))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def zeros(length: int) -> List[float]:
    return [0.0 for _ in range(length)]


def ints(length: int, value: int = -1) -> List[int]:
    return [value for _ in range(length)]


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def abs_list(values: Sequence[float]) -> List[float]:
    return [abs(v) for v in values]


def subtract(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]
