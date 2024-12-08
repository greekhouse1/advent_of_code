from collections import defaultdict
import itertools
from pathlib import Path
import string

FILE = Path() / "docs" / "day03.txt"

DIGITS = set(string.digits)


def is_digit(x: str) -> bool:
    return x in DIGITS


def is_symbol(x: str) -> bool:
    return x not in DIGITS and x != "."


def neighbors(x: int, y: int, x_max: int, y_max: int):
    for i in (x - 1, x, x + 1):
        if i < 0 or i >= x_max:
            continue
        for j in (y - 1, y, y + 1):
            if j < 0 or j >= y_max:
                continue
            yield (i, j)


def numbers(s: str):
    i = 0
    while i < len(s):
        if is_digit(s[i]):
            this_str = ""
            this_idx = tuple()
            while i < len(s) and is_digit(s[i]):
                this_str += s[i]
                this_idx += (i,)
                i += 1
            yield int(this_str), this_idx
        else:
            i += 1


with open(FILE) as f:
    grid = [x.rstrip("\n") for x in f]

GRID_WIDTH, GRID_HEIGHT = len(grid[0]), len(grid)
# print(f"{GRID_WIDTH} {GRID_HEIGHT}")

total1 = 0

for y, row in enumerate(grid):
    for number, idx in numbers(row):
        all_neighbors = set(
            itertools.chain(*(neighbors(y, x, GRID_WIDTH, GRID_HEIGHT) for x in idx))
        )
        if any(is_symbol(grid[i][j]) for (i, j) in all_neighbors):
            total1 += number

print(f"{total1 = }")

star_neighbors = defaultdict(list)

for y, row in enumerate(grid):
    for number, idx in numbers(row):
        all_neighbors = set(
            itertools.chain(*(neighbors(y, x, GRID_WIDTH, GRID_HEIGHT) for x in idx))
        )
        for i, j in all_neighbors:
            if grid[i][j] == "*":
                star_neighbors[(i, j)].append(number)

total2 = 0
for nums in star_neighbors.values():
    if len(nums) == 2:
        total2 += nums[0] * nums[1]

print(f"{total2 = }")
