import copy
from dataclasses import dataclass
from functools import cache
import itertools

from pathlib import Path
from typing import Generator, List, Tuple

from util import strip_new_line

FILE = Path() / "docs" / "day13.txt"


@dataclass
class Grid:
    grid: List[str]

    def shape(self) -> Tuple[int, int]:
        return len(self.grid), len(self.grid[0])

    def row(self, idx: int) -> str:
        return self.grid[idx]

    def col(self, idx: int) -> int:
        return "".join(row[idx] for row in self.grid)

    def display(self):
        for row in self.grid:
            print(row)

    def check_fold(self, idx: int, type_="row") -> bool:
        num_rows, num_cols = self.shape()
        max_idx = num_rows if type_ == "row" else num_cols
        if type_ == "row":
            func = self.row
        else:
            func = self.col
        for i in range(max_idx):
            top_index = idx - i - 1
            bottom_index = idx + i
            if top_index < 0 or bottom_index >= max_idx:
                return True
            if func(top_index) != func(bottom_index):
                return False
        return True

    def find_fold(self, type_="row") -> List[int]:
        ret = []
        num_rows, num_cols = self.shape()
        max_idx = num_rows if type_ == "row" else num_cols
        for i in range(1, max_idx):
            if self.check_fold(i, type_):
                ret.append(i)

        return ret

    def check_smudge(self) -> int:
        num_rows, num_cols = self.shape()
        cur_row_folds = self.find_fold()
        cur_col_folds = self.find_fold("col")
        for i in range(num_rows):
            for j in range(num_cols):
                g = copy.deepcopy(self.grid)
                new_char = "#" if g[i][j] == "." else "."
                g[i] = g[i][:j] + new_char + g[i][j + 1 :]
                grid = Grid(g)
                row_fold = [x for x in grid.find_fold() if x not in cur_row_folds]
                col_fold = [x for x in grid.find_fold("col") if x not in cur_col_folds]
                if len(row_fold) + len(col_fold) == 1:
                    total = 0
                    for x in row_fold:
                        total += 100 * x
                    for x in col_fold:
                        total += x
                    return total


def read_file(fn) -> Generator[List[str], None, None]:
    grid = []
    with open(fn) as f:
        for line in strip_new_line(f):
            if not line:
                yield Grid(grid)
                grid = []
            else:
                grid.append(line)
    if grid:
        yield Grid(grid)


total = 0
for g in read_file(FILE):
    row_fold = g.find_fold()
    if row_fold:
        total += 100 * row_fold[0]
    col_fold = g.find_fold("col")
    if col_fold:
        total += col_fold[0]

print(f"1: {total}")

total = 0
for g in read_file(FILE):
    total += g.check_smudge()

print(f"2: {total}")
