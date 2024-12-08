import copy
from dataclasses import dataclass
from functools import cache
import itertools

from pathlib import Path
from typing import List, Tuple

from util import strip_new_line

FILE = Path() / "docs" / "day15.txt"


def load_file(file_name) -> List[str]:
    with open(file_name) as f:
        return f.read().rstrip("\n").split(",")


def hashish(s: str) -> int:
    total = 0
    for c in s:
        total += ord(c)
        total = (17 * total) % 256
    return total


def parse_command(s: str):
    if s.endswith("-"):
        return s[:-1], s[-1]
    seq, val = s.split("=")
    return (seq, int(val)), "="


def find_seq(s: str, box: List[Tuple[str, int]]):
    for idx, val in enumerate(box):
        if val[0] == s:
            return idx
    return None


sequences = load_file(FILE)
print(f"1: {sum(map(hashish, sequences))}")

boxes = [list() for _ in range(256)]

for seq in sequences:
    # print(f"{seq=}")
    data, op = parse_command(seq)
    if op == "-":
        box_num = hashish(data)
        idx = find_seq(data, boxes[box_num])
        if idx is not None:
            del boxes[box_num][idx]
    else:
        val, focal_len = data
        box_num = hashish(val)
        idx = find_seq(val, boxes[box_num])
        if idx is None:
            boxes[box_num].append((val, focal_len))
        else:
            boxes[box_num][idx] = (val, focal_len)

    # for idx, box in enumerate(boxes):
    #     if box:
    #         print(f"Box {idx}: {box}")

total = 0
for box_num, box in enumerate(boxes):
    for slot_num, item in enumerate(box):
        total += (box_num + 1) * (slot_num + 1) * item[1]
print(f"2: {total}")
