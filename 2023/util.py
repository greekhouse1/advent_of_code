from dataclasses import dataclass
from functools import total_ordering
from typing import Generator, Iterable


def strip_new_line(it: Iterable[str]) -> Generator[str, None, None]:
    for x in it:
        yield x.rstrip("\n")


@total_ordering
@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other: "Point"):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point"):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: int):
        return Point(other * self.x, other * self.y)

    def __rmul__(self, other: int):
        return Point(other * self.x, other * self.y)

    def __eq__(self, other: "Point"):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other: "Point"):
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self) -> int:
        return hash(self.x + 12345 * self.y)

    @property
    def tuple(self):
        return (self.x, self.y)


NORTH = Point(-1, 0)
SOUTH = Point(1, 0)
WEST = Point(0, -1)
EAST = Point(0, 1)

DIRECTIONS = (NORTH, EAST, SOUTH, WEST)


def opposite(p: Point):
    return Point(-p.x, -p.y)
