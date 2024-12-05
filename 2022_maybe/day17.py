from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterator, List, Set, Tuple

from util import DIRECTIONS, Point, NORTH, SOUTH, EAST, WEST, opposite, strip_new_line

FILE = Path() / "docs" / "day17.txt"


@dataclass
class Grid:
    grid: List[List[int]]

    def __getitem__(self, value: Point) -> str:
        return self.grid[value.x][value.y]

    def shape(self) -> Tuple[int, int]:
        return len(self.grid), len(self.grid[0])

    def display(self):
        for row in self.grid:
            print(row)

    def _outside(self, p: Point):
        rows, cols = self.shape()
        if p.x < 0 or p.y < 0:
            return True
        if p.x >= rows or p.y >= cols:
            return True
        return False

    def get_min_heat(self, min_steps: int = 0, max_steps: int = 3):
        rows, cols = self.shape()
        heap = [(0, Point(0, 0), tuple())]
        used = set()

        while heap:
            score, point, illegal = heappop(heap)
            if point.x == rows - 1 and point.y == cols - 1:
                return score

            if (point, illegal) in used:
                continue
            used.add((point, illegal))

            for direction in DIRECTIONS:
                if direction in illegal:
                    continue

                new_score = score
                new_point = point
                new_illegal = (direction, opposite(direction))

                for step in range(max_steps):
                    new_point += direction
                    if self._outside(new_point):
                        break
                    new_score += self[new_point]

                    if step >= min_steps - 1:
                        heappush(heap, (new_score, new_point, new_illegal))

        assert False, "Unreachable"


def load_file(file_name: Path) -> Grid:
    grid = []
    with open(file_name) as f:
        for line in strip_new_line(f):
            grid.append(list(map(int, line)))
    return Grid(grid)


g = load_file(FILE)
# g.display()
print(f"1: {g.get_min_heat()}")
print(f"2: {g.get_min_heat(min_steps=4, max_steps=10)}")
