from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

from util import DIRECTIONS, Point, NORTH, SOUTH, EAST, WEST, opposite, strip_new_line

FILE = Path() / "docs" / "day18.txt"

DIR_MAP = {"U": NORTH, "D": SOUTH, "R": EAST, "L": WEST}
DIR_MAP_INT = {"0": "R", "1": "D", "2": "L", "3": "U"}


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


HORZ = 0
UP = 1
DOWN = 2


@dataclass
class Line:
    start: Point
    end: Point

    def shift(self, p: Point):
        self.start -= p
        self.end -= p

    def min_x(self):
        return min(self.start.x, self.end.x)

    def max_x(self):
        return max(self.start.x, self.end.x)

    def min_y(self):
        return min(self.start.y, self.end.y)

    def max_y(self):
        return max(self.start.y, self.end.y)

    def orientation(self):
        if self.start.x == self.end.x:
            return HORZ
        if self.start.x > self.end.x:
            return UP
        return DOWN

    def intersects_row(self, row: int) -> bool:
        if self.orientation() == HORZ:
            if self.start.x == row:
                return True
            return False
        min_x = min(self.start.x, self.end.x)
        max_x = max(self.start.x, self.end.x)
        if min_x <= row <= max_x:
            return True
        return False


@dataclass(init=False)
class ExpandGrid:
    instructions: List[Instruction]
    lines: List[Line]

    def __init__(self, instructions: List[Instruction], expand: bool = False):
        self.instructions = instructions
        if expand:
            for inst in self.instructions:
                inst.expand()
        self._make_lines()

    def _make_lines(self):
        min_x, max_x, min_y, max_y = (0, 0, 0, 0)

        lines = []
        start_point = Point(0, 0)
        cur_point = start_point
        for inst in self.instructions:
            new_point = cur_point + DIR_MAP[inst.dir] * inst.num
            lines.append(Line(cur_point, new_point))
            cur_point = new_point
            min_x = min(min_x, cur_point.x)
            max_x = max(max_x, cur_point.x)
            min_y = min(min_y, cur_point.y)
            max_y = max(max_y, cur_point.y)

        assert cur_point == start_point
        mins = Point(min_x, min_y)
        for line in lines:
            line.shift(mins)

        self.lines = lines
        self._shape = (max_x + 1, max_y + 1)

    def shape(self):
        return self._shape

    def intersect_lines(self, row: int):
        lines = []
        for line in self.lines:
            if line.intersects_row(row):
                lines.append(line)

        return sorted(
            lines,
            key=lambda p: (
                min(p.start.tuple[::-1], p.end.tuple[::-1]),
                max(p.start.tuple[::-1], p.end.tuple[::-1]),
            ),
        )

    def score_row(self, row: int) -> int:
        intersections = self.intersect_lines(row)
        score = 0
        inside = False
        for idx, line in enumerate(intersections):
            if line.orientation() != HORZ:
                score += 1
                if inside and idx != 0:
                    prev = intersections[idx - 1]
                    if prev.orientation() != HORZ:
                        score += line.min_y() - prev.max_y() - 1
                inside = not inside
            else:
                prev = intersections[idx - 1]
                next = intersections[idx + 1]
                if prev.orientation() == next.orientation():
                    inside = not inside
                score += line.max_y() - line.min_y() - 1  # Don't count endpoints

        return score

    def dug(self):
        num_rows, _ = self.shape()
        score = 0
        for i in range(num_rows):
            score += self.score_row(i)
        return score


def expand_load_file(file_name: Path, expand=False) -> Grid:
    instructions = []
    with open(file_name) as f:
        for line in strip_new_line(f):
            dir, num, color = line.split()
            instructions.append(Instruction(dir, int(num), color[2:-1]))
    return ExpandGrid(instructions, expand=expand)


g = expand_load_file(FILE, expand=True)

num_rows, _ = g.shape()
print(f"{g.shape() = }")

# for i in range(num_rows):
#     print(f"***ROW {i}***")
#     for line in g.intersect_lines(i):
#         print(line, line.orientation())
#     print(f"Row score: {g.score_row(i)}")
#     print()

print(f"2: {g.dug()}")
