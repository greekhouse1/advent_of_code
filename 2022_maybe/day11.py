from dataclasses import dataclass
import itertools

from pathlib import Path
from typing import List, Tuple

FILE = Path() / "docs" / "day11.txt"


@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other: "Point"):
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other: "Point"):
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash(self.x + 12345 * self.y)


@dataclass
class Node:
    location: Point
    is_galaxy: bool


@dataclass
class Grid:
    grid: List[List[bool]]

    def __getitem__(self, value: Point) -> Node:
        return self.grid[value.x][value.y]

    def display(self):
        for row in self.grid:
            print("".join("#" if x else "." for x in row))

    def shape(self) -> Tuple[int, int]:
        return len(self.grid), len(self.grid[0])

    def get_empty_rows(self):
        rows, _ = self.shape()
        expandable = []
        for idx in range(rows):
            if not any(self.grid[idx]):
                expandable.append(idx)
        return expandable

    def get_empty_columns(self):
        rows, cols = self.shape()
        expandable = []
        for idx in range(cols):
            if not any(self.grid[row][idx] for row in range(rows)):
                expandable.append(idx)
        return expandable

    def expand_rows(self):
        _, cols = self.shape()
        expandable = self.get_empty_rows()
        for idx in reversed(expandable):
            self.grid.insert(idx, [False] * cols)

    def expand_cols(self):
        expandable = self.get_empty_columns()
        for idx in reversed(expandable):
            for row in self.grid:
                row.insert(idx, False)

    def expand(self):
        self.expand_cols()
        self.expand_rows()

    def galaxy_locations(self):
        galaxies = []
        for i, j in itertools.product(*map(range, self.shape())):
            if self.grid[i][j]:
                galaxies.append(Point(i, j))
        return galaxies


def load_file(file_name: Path) -> Grid:
    grid = []
    with open(file_name) as f:
        for line in f:
            line = line.rstrip("\n")
            grid.append([x == "#" for x in line])
    return Grid(grid)


g = load_file(FILE)
g.expand()

galaxies = g.galaxy_locations()
print(f"Num galaxies: {len(galaxies)}")

total_distance = 0
for g1, g2 in itertools.combinations(galaxies, 2):
    total_distance += abs(g1.x - g2.x) + abs(g1.y - g2.y)

print(f"1: {total_distance}")

g = load_file(FILE)
galaxies = g.galaxy_locations()
EXPANSION_FACTOR = 1_000_000
expandable_rows = g.get_empty_rows()
expandable_cols = g.get_empty_columns()

total_distance = 0
for g1, g2 in itertools.combinations(galaxies, 2):
    total_distance += abs(g1.x - g2.x) + abs(g1.y - g2.y)
    exp_rows = len(
        [x for x in expandable_rows if min(g1.x, g2.x) < x < max(g1.x, g2.x)]
    )
    total_distance += exp_rows * (EXPANSION_FACTOR - 1)
    exp_cols = len(
        [y for y in expandable_cols if min(g1.y, g2.y) < y < max(g1.y, g2.y)]
    )
    total_distance += exp_cols * (EXPANSION_FACTOR - 1)


print(f"2: {total_distance}")
