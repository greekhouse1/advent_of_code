from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Set, Tuple

from util import Point, NORTH, SOUTH, EAST, WEST, strip_new_line

FILE = Path() / "docs" / "day16.txt"


@dataclass
class Grid:
    grid: List[List[str]]

    def __getitem__(self, value: Point) -> str:
        return self.grid[value.x][value.y]

    def shape(self) -> Tuple[int, int]:
        return len(self.grid), len(self.grid[0])

    def display(self):
        for row in self.grid:
            print(row)

    def display_visited(self, start: Point = Point(0, 0), dir: Point = EAST):
        visited = self.get_visited(start, dir)
        rows, cols = self.shape()
        for x in range(rows):
            row = ""
            for y in range(cols):
                p = Point(x, y)
                if p in visited:
                    row += "#"
                else:
                    row += "."

            print(row)

    def get_visited(self, start=Point(0, 0), dir=EAST) -> Set[Point]:
        stack = [(start, dir)]
        visited = set()

        while stack:
            loc, dir = stack.pop()
            if (loc, dir) in visited:
                continue

            rows, cols = self.shape()
            if not (0 <= loc.x < rows):
                continue
            if not (0 <= loc.y < cols):
                continue

            visited.add((loc, dir))

            for new_dir in get_new_direction(self[loc], dir):
                stack.append((loc + new_dir, new_dir))

        return set(x[0] for x in visited)

    def get_best(self):
        best = 0
        rows, cols = self.shape()

        for x in range(cols):
            visited = self.get_visited(Point(0, x), SOUTH)
            best = max(best, len(visited))
            visited = self.get_visited(Point(rows - 1, x), NORTH)
            best = max(best, len(visited))

        for x in range(cols):
            visited = self.get_visited(Point(x, 0), EAST)
            best = max(best, len(visited))
            visited = self.get_visited(Point(x, cols - 1), WEST)
            best = max(best, len(visited))

        return best


def get_new_direction(symbol: str, dir: Point) -> Iterator[Point]:
    match symbol:
        case ".":
            yield dir
        case "-":
            if dir in (EAST, WEST):
                yield dir
            else:
                yield EAST
                yield WEST
        case "|":
            if dir in (NORTH, SOUTH):
                yield dir
            else:
                yield NORTH
                yield SOUTH
        case "/":
            if dir == EAST:
                yield NORTH
            elif dir == WEST:
                yield SOUTH
            elif dir == NORTH:
                yield EAST
            else:
                yield WEST
        case "\\":
            if dir == EAST:
                yield SOUTH
            elif dir == WEST:
                yield NORTH
            elif dir == NORTH:
                yield WEST
            else:
                yield EAST


def load_file(file_name: Path) -> Grid:
    grid = []
    with open(file_name) as f:
        for line in strip_new_line(f):
            grid.append(line)
    return Grid(grid)


g = load_file(FILE)
g.display()
g.display_visited()
print(f"1: {len(g.get_visited())}")
print(f"2: {g.get_best()}")
