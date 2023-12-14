from dataclasses import dataclass
from pathlib import Path
from typing import List


FILE = Path() / "docs" / "day09.txt"


def parse_file(file_name: Path) -> List[List[int]]:
    all_sequences = []
    with open(file_name) as f:
        for line in f:
            all_sequences.append(list(map(int, line.split())))
    return all_sequences


all_sequences = parse_file(FILE)


def get_below(seq: List[int]) -> List[int]:
    return [b - a for (a, b) in zip(seq, seq[1:])]


def expand_sequence(seq: List[int]) -> int:
    if len(set(seq)) == 1:
        return seq[0]
    return seq[-1] + expand_sequence(get_below(seq))


total = 0
for seq in all_sequences:
    total += expand_sequence(seq)

print(f"1: {total}")


def expand_sequence_backwards(seq: List[int]) -> int:
    """
    b - a = x
    a = b - x
    """
    if len(set(seq)) == 1:
        return seq[0]
    return seq[0] - expand_sequence_backwards(get_below(seq))


total = 0
for seq in all_sequences:
    total += expand_sequence_backwards(seq)

print(f"2: {total}")
