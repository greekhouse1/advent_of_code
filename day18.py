from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterator, List, Set, Tuple

from util import DIRECTIONS, Point, NORTH, SOUTH, EAST, WEST, opposite, strip_new_line

FILE = Path() / "docs" / "day18_test.txt"

DIR_MAP = {"U": NORTH, "D": SOUTH, "R": EAST, "L": WEST}
DIR_MAP_INT = {"3": NORTH, "1": SOUTH, "0": EAST, "2": WEST}


@dataclass
class Instruction:
    dir: str
    num: int
    color: str

    def expand(self):
        self.dir = DIR_MAP_INT[self.color[-1]]
        self.num = int(self.color[:-1], 16)


@dataclass(init=False)
class Grid:
    grid: List[List[str]]
    instructions: List[Instruction]

    def __init__(self, instructions: List[Instruction]):
        self.instructions = instructions
        self._make_grid()

    def __getitem__(self, value: Point) -> str:
        return self.grid[value.x][value.y]

    def __setitem__(self, idx: Point, value: str) -> str:
        self.grid[idx.x][idx.y] = value

    def shape(self) -> Tuple[int, int]:
        return len(self.grid), len(self.grid[0])

    def display(self):
        for row in self.grid:
            print("".join(row))

    def _outside(self, p: Point):
        rows, cols = self.shape()
        if p.x < 0 or p.y < 0:
            return True
        if p.x >= rows or p.y >= cols:
            return True
        return False

    def _get_boundary(self) -> Set[Point]:
        path = [Point(0, 0)]
        for inst in self.instructions:
            direction = DIR_MAP[inst.dir]
            for _ in range(inst.num):
                path.append(path[-1] + direction)

        return set(path)

    def _shift_boundary(self, boundary: Set[Point]) -> Set[Point]:
        min_x = min(p.x for p in boundary)
        min_y = min(p.y for p in boundary)

        return set(Point(p.x - min_x, p.y - min_y) for p in boundary)

    def _make_grid(self):
        boundary = self._shift_boundary(self._get_boundary())
        rows = max(p.x for p in boundary)
        cols = max(p.y for p in boundary)
        grid = []
        for i in range(rows + 1):
            new_row = []
            for j in range(cols + 1):
                if Point(i, j) in boundary:
                    new_row.append("#")
                else:
                    new_row.append(".")
            grid.append(new_row)
        self.grid = grid

    def _get_outer(self):
        to_expand = set()
        rows, cols = self.shape()

        for i in range(cols):
            p = Point(0, i)
            if self[p] == ".":
                to_expand.add(p)
            p = Point(rows - 1, i)
            if self[p] == ".":
                to_expand.add(p)

        for i in range(rows):
            p = Point(i, 0)
            if self[p] == ".":
                to_expand.add(p)
            p = Point(i, cols - 1)
            if self[p] == ".":
                to_expand.add(p)

        outside = set()

        while to_expand:
            point = to_expand.pop()
            outside.add(point)
            for direction in DIRECTIONS:
                new_p = point + direction
                if self._outside(new_p):
                    continue
                if new_p in outside or new_p in to_expand:
                    continue
                if self[new_p] == ".":
                    to_expand.add(new_p)

        return outside

    def fill(self):
        outer = self._get_outer()
        rows, cols = self.shape()
        for i in range(rows):
            for j in range(cols):
                p = Point(i, j)
                if p not in outer:
                    self[p] = "#"

    def dug(self) -> int:
        return sum(row.count("#") for row in self.grid)


def load_file(file_name: Path) -> Grid:
    instructions = []
    with open(file_name) as f:
        for line in strip_new_line(f):
            dir, num, color = line.split()
            instructions.append(Instruction(dir, int(num), color[2:-1]))
    return Grid(instructions)


g = load_file(FILE)
g.display()
g.fill()
print()
g.display()
print(f"1: {g.dug()}")

g = load_file(FILE)
for inst in g.instructions:
    inst.expand()
    print(inst)
