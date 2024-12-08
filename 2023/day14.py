import copy
from dataclasses import dataclass
from functools import cache
import itertools

from pathlib import Path
from typing import List, Tuple

from util import strip_new_line

FILE = Path() / "docs" / "day14.txt"


@dataclass
class Grid:
    grid: List[List[str]]

    def shape(self) -> Tuple[int, int]:
        return len(self.grid), len(self.grid[0])

    def row(self, idx: int) -> str:
        return self.grid[idx]

    def col(self, idx: int) -> int:
        return "".join(row[idx] for row in self.grid)

    def display(self):
        for row in self.grid:
            print("".join(row))

    def tilt_north(self):
        rows, cols = self.shape()
        for i in range(rows):
            for j in range(cols):
                if self.grid[i][j] != "O":
                    continue
                idx = i
                while idx >= 1:
                    if self.grid[idx - 1][j] == ".":
                        idx -= 1
                    else:
                        break
                self.grid[idx][j], self.grid[i][j] = self.grid[i][j], self.grid[idx][j]

    def tilt_south(self):
        rows, cols = self.shape()
        for i in range(rows)[::-1]:
            for j in range(cols):
                if self.grid[i][j] != "O":
                    continue
                idx = i
                while idx <= rows - 2:
                    if self.grid[idx + 1][j] == ".":
                        idx += 1
                    else:
                        break
                self.grid[idx][j], self.grid[i][j] = self.grid[i][j], self.grid[idx][j]

    def tilt_west(self):
        rows, cols = self.shape()
        for i in range(rows):
            for j in range(cols):
                if self.grid[i][j] != "O":
                    continue
                jdx = j
                while jdx >= 1:
                    if self.grid[i][jdx - 1] == ".":
                        jdx -= 1
                    else:
                        break
                self.grid[i][j], self.grid[i][jdx] = self.grid[i][jdx], self.grid[i][j]

    def tilt_east(self):
        rows, cols = self.shape()
        for i in range(rows):
            for j in range(cols)[::-1]:
                if self.grid[i][j] != "O":
                    continue
                jdx = j
                while jdx <= cols - 2:
                    if self.grid[i][jdx + 1] == ".":
                        jdx += 1
                    else:
                        break
                self.grid[i][j], self.grid[i][jdx] = self.grid[i][jdx], self.grid[i][j]

    def cycle(self):
        self.tilt_north()
        self.tilt_west()
        self.tilt_south()
        self.tilt_east()

    def tuplify(self):
        return tuple(tuple(x) for x in self.grid)

    def super_cycle(self, count=1_000_000_000):
        previous = []
        current = self.tuplify()
        while current not in previous:
            previous.append(current)
            self.cycle()
            current = self.tuplify()

        last_seen = previous.index(current)
        num_to_go = count - last_seen

        previous = previous[last_seen:]
        desired_grid = previous[num_to_go % len(previous)]

        self.grid = list(list(x) for x in desired_grid)
        return self.score_north()

    def score_north(self) -> int:
        total = 0
        rows, _ = self.shape()
        for i, row in enumerate(self.grid):
            total += (rows - i) * row.count("O")
        return total


def read_file(fn) -> Grid:
    grid = []
    with open(fn) as f:
        for line in strip_new_line(f):
            grid.append(list(line))
    return Grid(grid)


g = read_file(FILE)
g.tilt_north()
print(f"1: {g.score_north()}")

g = read_file(FILE)
print(f"2: {g.super_cycle()}")
