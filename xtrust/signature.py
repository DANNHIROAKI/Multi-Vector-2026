"""Document signature data structures implemented without third-party deps."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Set


@dataclass
class BloomSignature:
    num_bits: int
    num_hashes: int
    bits: bytearray
    universe: Set[int]

    @classmethod
    def create(cls, size: int, false_positive_rate: float = 0.05) -> "BloomSignature":
        if size <= 0:
            return cls(num_bits=8, num_hashes=2, bits=bytearray(1), universe=set())
        n_bits = max(8, int(math.ceil(-(size * math.log(false_positive_rate)) / (math.log(2) ** 2))))
        n_hashes = max(2, int(round((n_bits / size) * math.log(2))))
        byte_len = (n_bits + 7) // 8
        return cls(num_bits=n_bits, num_hashes=n_hashes, bits=bytearray(byte_len), universe=set())

    def add(self, value: int) -> None:
        for seed in range(self.num_hashes):
            idx = self._hash(value, seed)
            self._set_bit(idx)
        self.universe.add(value)

    def update(self, values: Iterable[int]) -> None:
        for value in values:
            self.add(value)

    def contains(self, value: int) -> bool:
        if value in self.universe:
            return True
        for seed in range(self.num_hashes):
            if not self._get_bit(self._hash(value, seed)):
                return False
        return True

    def superset(self) -> Set[int]:
        return set(self.universe)

    def _hash(self, value: int, seed: int) -> int:
        return hash((value, seed)) % self.num_bits

    def _set_bit(self, idx: int) -> None:
        byte_idx = idx // 8
        bit_idx = idx % 8
        self.bits[byte_idx] |= 1 << bit_idx

    def _get_bit(self, idx: int) -> bool:
        byte_idx = idx // 8
        bit_idx = idx % 8
        return bool(self.bits[byte_idx] & (1 << bit_idx))
