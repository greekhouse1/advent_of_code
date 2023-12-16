from dataclasses import dataclass
from heapq import heappop, heappush
import itertools
from pathlib import Path
from typing import List, Tuple

FILE = Path() / "docs" / "day10.txt"


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
    neighbors: Tuple[Point, ...]
    char: str

    def adjacent(self, node: "Node"):
        return node.location in self.neighbors and self.location in node.neighbors

    def __lt__(self, other):
        return True


box_lu = {
    "|": "│",
    "-": "─",
    "L": "└",
    "J": "┘",
    "7": "┐",
    "F": "┌",
    ".": " ",
    "I": "\033[92mI\033[0m",
    "O": " ",
    "S": "S",
}


@dataclass
class Grid:
    grid: List[List[Node]]
    start: Node

    def __getitem__(self, value: Point) -> Node:
        return self.grid[value.x][value.y]

    def display(self):
        for row in self.grid:
            print("".join(box_lu[x.char] for x in row))

    def _get_used(self):
        used = set()
        stack = [grid.start.location]
        while stack:
            location = stack.pop()
            used.add(location)

            for neighbor in self[location].neighbors:
                if neighbor in used:
                    continue
                if self[location].adjacent(self[neighbor]):
                    stack.append(self[neighbor].location)
        return used

    def dejunk(self):
        used = self._get_used()
        for x, row in enumerate(self.grid):
            for y, _ in enumerate(row):
                p = Point(x, y)
                if p in used:
                    continue
                self.grid[x][y] = Node(location=p, neighbors=tuple(), char=".")

        start_node = self[self.start.location]
        adj_north = start_node.adjacent(self[self.start.location + NORTH])
        adj_south = start_node.adjacent(self[self.start.location + SOUTH])
        adj_east = start_node.adjacent(self[self.start.location + EAST])
        adj_west = start_node.adjacent(self[self.start.location + WEST])

        if adj_north and adj_south:
            char = "|"
        if adj_north and adj_east:
            char = "L"
        if adj_north and adj_west:
            char = "J"
        if adj_south and adj_east:
            char = "F"
        if adj_south and adj_west:
            char = "7"
        if adj_east and adj_west:
            char = "-"

        self.grid[self.start.location.x][self.start.location.y] = Node(
            self.start.location, self.start.neighbors, char
        )

    def count_inside(self) -> int:
        self.dejunk()
        count = 0

        for row in self.grid:
            outside = True
            entry = None
            for node in row:
                if node.char == "|":
                    outside = not outside
                if node.char in "FL":
                    entry = node.char
                if node.char == "J" and entry == "F":
                    outside = not outside
                if node.char == "7" and entry == "L":
                    outside = not outside
                if node.char == ".":
                    if outside:
                        node.char = "O"
                    else:
                        node.char = "I"
                        count += 1

            # assert outside is True

        return count


NORTH = Point(-1, 0)
SOUTH = Point(1, 0)
WEST = Point(0, -1)
EAST = Point(0, 1)

neighbor_delta_map = {
    "|": (NORTH, SOUTH),
    "-": (WEST, EAST),
    "L": (NORTH, EAST),
    "J": (NORTH, WEST),
    "7": (SOUTH, WEST),
    "F": (SOUTH, EAST),
    ".": tuple(),
    "S": (NORTH, EAST, SOUTH, WEST),
}


def load_file(file_name: Path) -> Grid:
    grid = []
    with open(file_name) as f:
        for x, line in enumerate(f):
            row = []
            for y, char in enumerate(line.rstrip("\n")):
                if char in "IO":
                    char = "."
                location = Point(x, y)
                neighbors = tuple()
                for delta in neighbor_delta_map[char]:
                    neighbors += (location + delta,)
                this_node = Node(location, neighbors, char)
                row.append(this_node)
                if char == "S":
                    start = this_node
            grid.append(row)

    return Grid(grid, start)


def farthest_distance(grid: Grid) -> int:
    used = set()
    heap = [(0, grid.start)]
    while heap:
        score, node = heappop(heap)
        used.add(node.location)

        for neighbor in node.neighbors:
            if neighbor in used:
                continue
            if node.adjacent(grid[neighbor]):
                heappush(heap, (score + 1, grid[neighbor]))

    return score


grid = load_file(FILE)
score = farthest_distance(grid)
grid.display()
print(f"1: {score}")

grid.dejunk()
print("DEJUNKED")
grid.display()
print()

size = grid.count_inside()
grid.display()

print(f"2: {size}")
