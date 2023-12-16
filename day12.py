from dataclasses import dataclass
from functools import cache
import itertools

from pathlib import Path
from typing import List, Tuple

FILE = Path() / "docs" / "day12.txt"


@dataclass
class SpringRow:
    arrangement: str
    counts: List[int]

    def fold(self, num_folds: int):
        self.arrangement = "?".join([self.arrangement] * num_folds)
        self.counts = self.counts * num_folds


def read_file(fn) -> List[SpringRow]:
    ret = []
    with open(fn) as f:
        for line in f:
            arrangement, counts = line.rstrip("\n").split()
            ret.append(
                SpringRow(
                    arrangement=arrangement, counts=list(map(int, counts.split(",")))
                )
            )
    return ret


def row_can_start_broken(row, count):
    if count > len(row):
        return False
    if not all(x in "#?" for x in row[:count]):
        return False
    if count == len(row) or row[count] in ".?":
        return True
    return False


@cache
def count_arrangements(row: str, counts: Tuple[int]):
    if len(counts) == 0:
        return 1 if all(x in ".?" for x in row) else 0

    if sum(counts) + len(counts) - 1 > len(row):
        return 0

    if row[0] == ".":
        return count_arrangements(row[1:], counts)

    if row[0] == "#":
        if row_can_start_broken(row, counts[0]):
            rest = row[counts[0] + 1 :]
            if not rest:
                return 1
            return count_arrangements(rest, counts[1:])
        else:
            return 0

    # Question marks are the only place we branch we consider
    # Both cases where we have a . or #
    unbroken_count = count_arrangements(row[1:], counts)
    broken_count = count_arrangements("#" + row[1:], counts)
    return unbroken_count + broken_count


spring_rows = read_file(FILE)

total = 0
for sr in spring_rows:
    arr = count_arrangements(sr.arrangement, tuple(sr.counts))
    print(sr.arrangement, sr.counts, arr)
    total += arr

print(f"1: {total}")
print("-" * 80)
total = 0
for sr in spring_rows:
    sr.fold(5)
    arr = count_arrangements(sr.arrangement, tuple(sr.counts))
    print(sr.arrangement, sr.counts, arr)
    total += arr

print(f"2: {total}")
