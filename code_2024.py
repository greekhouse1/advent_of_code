import collections
import copy
from dataclasses import dataclass
from functools import cache, total_ordering
from heapq import heappop, heappush
import itertools
import re

from collections import Counter, defaultdict
import time
from typing import List, Tuple

import numpy as np


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

    def in_grid(self, rows, cols):
        return 0 <= self.x < rows and 0 <= self.y < cols

    def orthogonal(self, rows, cols):
        for i in (self.x - 1, self.x + 1):
            p = Point(i, self.y)
            if p.in_grid(rows, cols):
                yield p

        for j in (self.y - 1, self.y + 1):
            p = Point(self.x, j)
            if p.in_grid(rows, cols):
                yield p

    def taxicab(self, count, rows, cols):
        for i in range(-count, count + 1):
            for j in range(-count + abs(i), count - abs(i) + 1):
                p = Point(self.x + i, self.y + j)
                if p.in_grid(rows, cols):
                    yield p

    def abs(self):
        return abs(self.x) + abs(self.y)


###############################################################################
#################################### Day 1 ####################################
###############################################################################


def read1(fn) -> Tuple[List, List]:
    with open(fn) as f:
        raw_text = [line for line in f.readlines()]

    left, right = [], []
    for line in raw_text:
        x, y = map(int, line.rstrip("\n").split())
        left.append(x)
        right.append(y)
    return left, right


def p1a(fn):
    left, right = read1(fn)
    left.sort(), right.sort()

    print(sum(abs(x - y) for x, y in zip(left, right)))


def p1b(fn):
    left, right = read1(fn)
    c = Counter(right)

    total = 0
    for val in left:
        total += val * c[val]
    print(total)


###############################################################################
#################################### Day 2 ####################################
###############################################################################


def read2(fn) -> List[List]:
    with open(fn) as f:
        raw_text = [line for line in f.readlines()]

    input_ = []
    for line in raw_text:
        input_.append(list(map(int, line.rstrip("\n").split())))

    return input_


def checks_out(line: List[int]) -> bool:
    return all(0 < x - y < 4 for x, y in zip(line, line[1:]))


def checks_out_b(line: List[int]) -> bool:
    if checks_out(line):
        return True

    for idx in range(len(line)):
        new_line = line[:idx] + line[idx + 1 :]
        if checks_out(new_line):
            return True

    return False


def is_safe(line: List[int]) -> bool:
    return checks_out_b(line) or checks_out_b(line[::-1])


def p2a(fn: str):
    input_ = read2(fn)
    total = 0
    for line in input_:
        if is_safe(line):
            total += 1

    print(total)


###############################################################################
#################################### Day 3 ####################################
###############################################################################


def p3a(fn: str):
    with open(fn) as f:
        text = f.read()

    mul = re.compile(r"mul\((\d+),(\d+)\)")
    total = 0
    for x in mul.findall(text):
        print(x)
        total += int(x[0]) * int(x[1])
    print(total)


def p3b(fn: str):
    with open(fn) as f:
        text = f.read()

    mul = re.compile(r"mul\((\d+),(\d+)\)|(do\(\))|(don't\(\))")

    total = 0
    active = True
    for x in mul.findall(text):
        if x[2]:
            active = True
        elif x[3]:
            active = False
        else:
            if not active:
                continue
            total += int(x[0]) * int(x[1])
    print(total)


###############################################################################
#################################### Day 4 ####################################
###############################################################################

DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]


def p4a(fn):
    with open(fn) as f:
        grid = [x.rstrip() for x in f.readlines()]

    rows, cols = len(grid), len(grid[0])

    count = 0
    for x in range(rows):
        for y in range(cols):
            for xi, yi in DIRECTIONS:
                word = ""
                for i in range(4):
                    x_idx = x + i * xi
                    y_idx = y + i * yi
                    if x_idx < 0 or y_idx < 0 or x_idx >= rows or y_idx >= cols:
                        break
                    word += grid[x_idx][y_idx]
                if word == "XMAS":
                    count += 1

    print(count)


def p4b(fn):
    with open(fn) as f:
        grid = [x.rstrip() for x in f.readlines()]

    rows, cols = len(grid), len(grid[0])

    count = 0
    for x in range(rows - 2):
        for y in range(cols - 2):
            w1 = grid[x][y] + grid[x + 1][y + 1] + grid[x + 2][y + 2]
            w2 = grid[x][y + 2] + grid[x + 1][y + 1] + grid[x + 2][y]

            if w1 in ("MAS", "SAM") and w2 in ("MAS", "SAM"):
                count += 1

    print(count)


###############################################################################
#################################### Day 5 ####################################
###############################################################################


def read_p5(fn):
    rules = defaultdict(set)
    updates = []

    with open(fn) as f:
        in_rules = True
        for line in f.readlines():
            line = line.rstrip()
            if not line:
                in_rules = False
                continue
            if in_rules:
                x, y = line.split("|")
                rules[int(x)].add(int(y))
            else:
                updates.append(list(map(int, line.split(","))))
    return rules, updates


def check_update(update, rules):
    if len(update) < 2:
        return True
    *rest, last = update
    for x in rest:
        if x in rules[last]:
            return False
    return check_update(rest, rules)


def reorder(update, rules):
    if len(update) < 2:
        return update

    for idx, candidate in enumerate(update):
        if not any(y in rules[candidate] for y in update):
            return [candidate] + reorder(update[:idx] + update[idx + 1 :], rules)


def p5a(fn):
    rules, updates = read_p5(fn)

    total = 0
    for x in updates:
        if check_update(x, rules):
            total += x[len(x) // 2]
    print(total)


def p5b(fn):
    rules, updates = read_p5(fn)

    total = 0
    for x in updates:
        if not check_update(x, rules):
            x = reorder(x, rules)
            total += x[len(x) // 2]
    print(total)


###############################################################################
#################################### Day 6 ####################################
###############################################################################

DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]


def get_start(grid):
    for idx, row in enumerate(grid):
        jdx = row.find("^")
        if jdx >= 0:
            return idx, jdx


TURN_RIGHT = {
    (1, 0): (0, -1),
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (0, -1): (-1, 0),
}


def add_tuple(a, b):
    return (a[0] + b[0], a[1] + b[1])


def subtract_tuple(a, b):
    return (a[0] - b[0], a[1] - b[1])


def on_grid(x, y, rows, cols):
    return 0 <= x < rows and 0 <= y < cols


def p6a(fn):
    with open(fn) as f:
        grid = [x.rstrip() for x in f.readlines()]

    rows, cols = len(grid), len(grid[0])

    position = get_start(grid)
    orientation = (-1, 0)
    visited = set()
    while True:
        if not on_grid(*position, rows, cols):
            break
        if grid[position[0]][position[1]] == "#":
            position = subtract_tuple(position, orientation)
            orientation = TURN_RIGHT[orientation]
            continue
        visited.add(position)
        position = add_tuple(position, orientation)

    # print(len(visited))
    return grid, visited


def expand_path(grid, visited):
    expanded = set()
    for v in visited:
        if grid[v[0]][v[1]] != "^":
            expanded.add(v)
        for dir in TURN_RIGHT:
            v2 = add_tuple(v, dir)
            if on_grid(*v2, len(grid), len(grid[0])) and grid[v2[0]][v2[1]] not in "#^":
                expanded.add(v2)
    return expanded


def has_loop(grid):
    rows, cols = len(grid), len(grid[0])

    position = get_start(grid)
    orientation = (-1, 0)
    visited = set()
    while True:
        if not on_grid(*position, rows, cols):
            return False
        if grid[position[0]][position[1]] == "#":
            position = subtract_tuple(position, orientation)
            orientation = TURN_RIGHT[orientation]
            continue
        if (position, orientation) in visited:
            return True
        visited.add((position, orientation))
        position = add_tuple(position, orientation)

    assert False, "Unreachable"


def p6b(fn):
    grid, visited = p6a(fn)
    start = get_start(grid)

    looped = 0
    for p in visited:
        if p == start:
            continue
        new_grid = copy.deepcopy(grid)
        new_grid[p[0]] = "".join(
            "#" if i == p[1] else grid[p[0]][i] for i in range(len(grid[0]))
        )
        if has_loop(new_grid):
            looped += 1
    print(looped)


###############################################################################
#################################### Day 7 ####################################
###############################################################################


def read7(fn):
    data = []

    with open(fn) as f:
        for line in f:
            line = line.rstrip().replace(":", "")
            data.append(list(map(int, line.split())))
    return data


def check7(values):
    x, y, *rest = values
    a = x + y
    b = x * y
    if not rest:
        yield a
        yield b
        return

    yield from check7([a] + rest)
    yield from check7([b] + rest)


def p7a(fn):
    data = read7(fn)

    total = 0
    for calibration in data:
        target, *values = calibration
        if any(x == target for x in check7(values)):
            total += target
    print(total)


def check7b(values):
    x, y, *rest = values
    methods = [x + y, x * y, int(str(x) + str(y))]
    if not rest:
        yield from methods
        return
    for v in methods:
        yield from check7b([v] + rest)


def p7b(fn):
    data = read7(fn)

    total = 0
    for calibration in data:
        target, *values = calibration
        if any(x == target for x in check7b(values)):
            total += target
    print(total)


###############################################################################
#################################### Day 8 ####################################
###############################################################################


def in_grid(p: Point, rows: int, cols: int):
    return 0 <= p.x < rows and 0 <= p.y < cols


def p8a(fn):
    with open(fn) as f:
        grid = [x.rstrip() for x in f.readlines()]
        rows = len(grid)
        cols = len(grid[0])

    frequencies = defaultdict(list)

    for idx, row in enumerate(grid):
        for jdx, c in enumerate(row):
            if c == ".":
                continue
            frequencies[c].append(Point(idx, jdx))

    antinodes = set()

    for freq in frequencies.values():
        for p1, p2 in itertools.combinations(freq, 2):
            for x in 2 * p1 - p2, 2 * p2 - p1:
                if in_grid(x, rows, cols):
                    antinodes.add(x)

    print(len(antinodes))


def p8b(fn):
    with open(fn) as f:
        grid = [x.rstrip() for x in f.readlines()]
        rows = len(grid)
        cols = len(grid[0])

    frequencies = defaultdict(list)

    for idx, row in enumerate(grid):
        for jdx, c in enumerate(row):
            if c == ".":
                continue
            frequencies[c].append(Point(idx, jdx))

    antinodes = set()

    for freq in frequencies.values():
        for p1, p2 in itertools.combinations(freq, 2):
            for point, dir in (
                (p1, p1 - p2),
                (p2, p2 - p1),
            ):
                for k in range(500):
                    x = point + k * dir
                    if not in_grid(x, rows, cols):
                        break
                    antinodes.add(x)

    print(len(antinodes))


###############################################################################
#################################### Day 9 ####################################
###############################################################################


def read9(fn):
    with open(fn) as f:
        disk = f.read().strip()

    unpacked = []
    cur_idx = 0
    blank = True

    for c in disk:
        blank = not blank
        if blank:
            for _ in range(int(c)):
                unpacked.append(".")
        else:
            for _ in range(int(c)):
                unpacked.append(cur_idx)
            cur_idx += 1

    return unpacked


def p9a(fn):

    unpacked = read9(fn)
    # print("".join(map(str, unpacked)))
    front_idx = 0
    back_idx = len(unpacked) - 1

    while front_idx < back_idx:
        if unpacked[front_idx] != ".":
            front_idx += 1
            continue
        if unpacked[back_idx] == ".":
            back_idx -= 1
            continue

        unpacked[front_idx], unpacked[back_idx] = (
            unpacked[back_idx],
            unpacked[front_idx],
        )
        front_idx += 1
        back_idx -= 1

    # print("".join(map(str, unpacked)))
    score = 0
    for idx, val in enumerate(unpacked):
        try:
            score += idx * val
        except TypeError:
            break
    print(score)


def read9b(fn):
    with open(fn) as f:
        disk = f.read().strip()

    packed = []
    cur_idx = 0
    blank = True

    for c in disk:
        blank = not blank
        if blank:
            packed.append((".", int(c)))
        else:
            packed.append((cur_idx, int(c)))
            cur_idx += 1

    return packed


def unpack(packed):
    s = list()
    for val, count in packed:
        s += [val] * count

    return s


def find_insertable(marker, packed):
    count = packed[marker][1]

    idx = len(packed) - 1
    while idx > marker:
        id, idx_count = packed[idx]
        if id == ".":
            idx -= 1
            continue
        if idx_count <= count:
            return idx
        idx -= 1
    return None


def p9b(fn):

    packed = read9b(fn)
    # print(packed)

    marker = 0
    while marker < len(packed):
        id, count = packed[marker]
        if id != ".":
            marker += 1
            continue

        idx = find_insertable(marker, packed)
        if idx is None:
            marker += 1
            continue

        new_id, new_count = packed[idx]
        packed[idx] = (".", new_count)
        packed[marker] = (new_id, new_count)
        if new_count != count:
            packed.insert(marker + 1, (".", count - new_count))
        marker += 1

    unpacked = unpack(packed)
    # print(unpacked)
    score = 0
    for idx, val in enumerate(unpacked):
        try:
            score += idx * val
        except TypeError:
            continue
    print(score)


###############################################################################
#################################### Day 10 ###################################
###############################################################################


def read10(fn):
    mat = []
    with open(fn) as f:
        for row in f:
            mat.append(list(map(int, row.strip())))

    return mat


def get_zeros(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                yield Point(i, j)


def p10a(fn):
    mat = read10(fn)
    rows = len(mat)
    cols = len(mat[0])

    starting_points = list(get_zeros(mat))

    reachable = {x: {x} for x in starting_points}
    print(reachable)
    for target in range(1, 10):
        new_reachable = {x: set() for x in starting_points}
        for start, current_list in reachable.items():
            for p in current_list:
                for neighbor in p.orthogonal(rows, cols):
                    if mat[neighbor.x][neighbor.y] == target:
                        new_reachable[start].add(neighbor)

        reachable = new_reachable

    print(sum(len(x) for x in reachable.values()))


def p10b(fn):
    mat = read10(fn)
    rows = len(mat)
    cols = len(mat[0])

    starting_points = list(get_zeros(mat))

    reachable = {x: Counter([x]) for x in starting_points}
    for target in range(1, 10):
        new_reachable = {x: Counter() for x in starting_points}
        for start, current_counter in reachable.items():
            for p, count in current_counter.items():
                for neighbor in p.orthogonal(rows, cols):
                    if mat[neighbor.x][neighbor.y] == target:
                        new_reachable[start][neighbor] += count

        reachable = new_reachable

    total = 0
    for x in reachable.values():
        total += sum(x.values())
    print(total)


###############################################################################
#################################### Day 11 ###################################
###############################################################################


def p11a(fn):
    with open(fn) as f:
        stones = list(map(int, f.read().split()))

    for _ in range(25):
        new_stones = []
        for x in stones:
            if x == 0:
                new_stones.append(1)
            elif len(str(x)) % 2 == 0:
                a = str(x)
                y = [int(a[: len(a) // 2]), int(a[len(a) // 2 :])]
                new_stones += y
            else:
                new_stones.append(x * 2024)
        stones = new_stones

    print(len(stones))


def p11b(fn, num=25):
    with open(fn) as f:
        stones = list(map(int, f.read().split()))

    stones = Counter(stones)

    for _ in range(num):
        new_stones = Counter()
        for x, count in stones.items():
            if x == 0:
                new_stones[1] += count
            elif len(str(x)) % 2 == 0:
                a = str(x)
                new_rocks = [int(a[: len(a) // 2]), int(a[len(a) // 2 :])]
                for new_stone in new_rocks:
                    new_stones[new_stone] += count
            else:
                new_stones[x * 2024] += count
        stones = new_stones

    count = sum(x for x in stones.values())
    print(count)


###############################################################################
#################################### Day 12 ###################################
###############################################################################


class Union:
    id: Point
    parent: Point

    def __init__(self, p: Point, val: str):
        self.id = p
        self.parent = self
        self.val = val

    def __repr__(self):
        return f"Union({repr(self.id)}, {repr(self.val)})"


def find(p: Union):
    if p.parent is p:
        return p
    return find(p.parent)


def union(p1, p2):
    p1_rep = find(p1)
    p2_rep = find(p2)
    p1_rep.parent = p2_rep


def perimeter(group, rows, cols):
    shared_edges = 0
    for p in group:
        for neighbor in p.orthogonal(rows, cols):
            if neighbor in group:
                shared_edges += 1
    return 4 * len(group) - shared_edges


def p12a(fn):
    with open(fn) as f:
        garden = [x.rstrip() for x in f]
        rows = len(garden)
        cols = len(garden[0])

    unions = {
        Point(i, j): Union(Point(i, j), garden[i][j])
        for i in range(rows)
        for j in range(cols)
    }

    for this_point in unions.values():

        for p in this_point.id.orthogonal(rows, cols):
            if unions[p].val == this_point.val:
                union(unions[p], this_point)

    groups = defaultdict(set)
    for point in unions.values():
        groups[find(point)].add(point.id)

    total = 0
    for v in groups.values():
        total += len(v) * perimeter(v, rows, cols)
    print(total)


def sides(group, rows, cols):
    max_row = max(p.x for p in group)
    min_row = min(p.x for p in group)

    max_col = max(p.y for p in group)
    min_col = min(p.y for p in group)

    edges = 0
    # horizontal test
    for i in range(min_row - 1, max_row + 1):
        pattern = (False, False)
        for j in range(min_col, max_col + 1):
            p1 = Point(i, j)
            p2 = Point(i + 1, j)
            this_pattern = (p1 in group, p2 in group)

            if this_pattern == pattern:
                continue

            if this_pattern in {(True, False), (False, True)}:
                edges += 1

            pattern = this_pattern

        # horizontal test
    for j in range(min_col - 1, max_col + 1):
        pattern = (False, False)
        for i in range(min_row, max_row + 1):
            p1 = Point(i, j)
            p2 = Point(i, j + 1)
            this_pattern = (p1 in group, p2 in group)

            if this_pattern == pattern:
                continue

            if this_pattern in {(True, False), (False, True)}:
                edges += 1

            pattern = this_pattern
    return edges


def p12b(fn):
    with open(fn) as f:
        garden = [x.rstrip() for x in f]
        rows = len(garden)
        cols = len(garden[0])

    unions = {
        Point(i, j): Union(Point(i, j), garden[i][j])
        for i in range(rows)
        for j in range(cols)
    }

    for this_point in unions.values():

        for p in this_point.id.orthogonal(rows, cols):
            if unions[p].val == this_point.val:
                union(unions[p], this_point)

    groups = defaultdict(set)
    for point in unions.values():
        groups[find(point)].add(point.id)

    total = 0
    for v in groups.values():
        # print(len(v), sides(v, rows, cols))
        total += len(v) * sides(v, rows, cols)
    print(total)


###############################################################################
#################################### Day 13 ###################################
###############################################################################


def read13(fn):
    systems = []
    with open(fn) as f:
        for line in f:
            if "A" in line or "B" in line:
                s1 = line.split(":")[1]
                x, y = s1.strip().split(",")
                x = int(x[2:])
                y = int(y[2:])
                if "A" in line:
                    a = (x, y)
                else:
                    b = (x, y)
            if "Prize" in line:
                mat = np.array([[a[0], b[0]], [a[1], b[1]]])

                s1 = line.split(":")[1]
                x, y = s1.strip().split(",")
                x = int(x.strip()[2:])
                y = int(y.strip()[2:])
                target = np.array([x, y])
                systems.append((mat, target))

    return systems


def read13b(fn):
    systems = []
    with open(fn) as f:
        for line in f:
            if "A" in line or "B" in line:
                s1 = line.split(":")[1]
                x, y = s1.strip().split(",")
                x = int(x[2:])
                y = int(y[2:])
                if "A" in line:
                    a = (x, y)
                else:
                    b = (x, y)
            if "Prize" in line:
                mat = np.array([[a[0], b[0]], [a[1], b[1]]])

                s1 = line.split(":")[1]
                x, y = s1.strip().split(",")
                x = int(x.strip()[2:]) + 10000000000000
                y = int(y.strip()[2:]) + 10000000000000
                target = np.array([x, y])
                systems.append((mat, target))

    return systems


def p13a(fn):
    systems = read13(fn)

    val = 0

    for mat, target in systems:
        sol = np.linalg.solve(mat, target)
        if any(x > 100 for x in sol):
            continue
        int_vec = np.array([int(sol[0] + 0.5), int(sol[1] + 0.5)])
        approx_target = np.matmul(mat, int_vec.T)

        if (approx_target == target).all():
            val += 3 * int_vec[0] + int_vec[1]

    print(val)


def p13b(fn):
    systems = read13b(fn)

    val = 0

    for mat, target in systems:
        sol = np.linalg.solve(mat, target)

        int_vec = np.array([int(sol[0] + 0.5), int(sol[1] + 0.5)])
        approx_target = np.matmul(mat, int_vec.T)

        if (approx_target == target).all():
            val += 3 * int_vec[0] + int_vec[1]

    print(val)


###############################################################################
#################################### Day 14 ###################################
###############################################################################


class Robot:
    def __init__(self, line):
        p, v = line.split()
        p, v = p.split("=")[-1], v.split("=")[-1]
        p, v = list(map(int, p.split(","))), list(map(int, v.split(",")))

        self.p = p
        self.v = v

    def __repr__(self):
        return f"Robot(p={self.p}, v={self.v})"

    def move(self, steps, rows, cols):
        self.p = (
            (self.p[0] + steps * self.v[0]) % rows,
            (self.p[1] + steps * self.v[1]) % cols,
        )

    def quadrant(self, rows, cols):
        mid_row = rows // 2
        mid_col = cols // 2

        if self.p[0] < mid_row:
            if self.p[1] < mid_col:
                return 0
            elif self.p[1] > mid_col:
                return 1
        elif self.p[0] > mid_row:
            if self.p[1] < mid_col:
                return 2
            elif self.p[1] > mid_col:
                return 3

        return None


def p14a(fn, rows=11, cols=7):
    robots = []
    with open(fn) as f:
        for line in f:
            robots.append(Robot(line))

    quad_counts = [0, 0, 0, 0]

    for r in robots:
        r.move(100, rows, cols)
        q = r.quadrant(rows, cols)
        print(r.p, q)
        if q is not None:
            quad_counts[q] += 1

    print(quad_counts)
    print(quad_counts[0] * quad_counts[1] * quad_counts[2] * quad_counts[3])


def print_grid(robots, rows, cols):
    grid = [[False for _ in range(rows)] for _ in range(cols)]
    for r in robots:
        grid[r.p[1]][r.p[0]] = True

    for row in grid:
        print("".join("#" if x else "." for x in row))


def p14b(fn, rows=11, cols=7):
    robots = []
    with open(fn) as f:
        for line in f:
            robots.append(Robot(line))

    for it in range(10**7):
        if it % 103 == 33:
            print_grid(robots, rows, cols)
            print(it)
            print("-" * 80)
            input()
        for r in robots:
            r.move(1, rows, cols)


###############################################################################
#################################### Day 15 ###################################
###############################################################################


def read15(fn):
    with open(fn) as f:
        in_moves = False
        move_string = ""
        grid = []
        for line in f:
            line = line.strip()
            if not line:
                in_moves = True
                continue
            if in_moves:
                move_string += line
            else:
                grid.append(list(line))

        return grid, move_string


def get_robot_position(grid):
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == "@":
                return i, j


class CantMove(Exception): ...


DIR_MAP = {
    "^": (-1, 0),
    "v": (1, 0),
    "<": (0, -1),
    ">": (0, 1),
}


def move(position, direction, grid, thing="@"):
    new_pos = (position[0] + direction[0], position[1] + direction[1])
    if grid[new_pos[0]][new_pos[1]] == ".":
        grid[new_pos[0]][new_pos[1]] = thing
        grid[position[0]][position[1]] = "."
        return
    elif grid[new_pos[0]][new_pos[1]] == "#":
        raise CantMove
    else:
        new_thing = grid[new_pos[0]][new_pos[1]]
        try:
            move(new_pos, direction, grid, thing=new_thing)
        except CantMove as e:
            raise e
        grid[new_pos[0]][new_pos[1]] = thing
        grid[position[0]][position[1]] = "."


def p15a(fn):
    grid, moves = read15(fn)
    # for line in grid:
    #     print("".join(line))
    # print()
    # print(moves)
    # print()
    robot = get_robot_position(grid)

    for d in moves:
        dir = DIR_MAP[d]
        try:
            move(robot, dir, grid)
        except CantMove:
            pass
        else:
            robot = (robot[0] + dir[0], robot[1] + dir[1])
        # finally:
        #     print(f"Move {d}:")
        #     for line in grid:
        #         print("".join(line))
        #     print()

    total = 0
    for i, row in enumerate(grid):
        for j, symbol in enumerate(row):
            if symbol == "O":
                total += 100 * i + j
    print(total)


def double_grid(grid):
    new_grid = []
    for row in grid:
        new_row = []
        for x in row:
            if x in "#.":
                new_row += [x, x]
            elif x == "O":
                new_row += ["[", "]"]
            else:
                new_row += ["@", "."]
        new_grid.append(new_row)
    return new_grid


def move2(position, direction, grid, thing=("@",)):
    if direction[0] == 0:
        return move(position[0], direction, grid, thing[0])
    pos_to_push = [(x[0] + direction[0], x[1]) for x in position]
    grid_markers = [grid[x][y] for (x, y) in pos_to_push]
    if "#" in grid_markers:
        raise CantMove
    if all(x == "." for x in grid_markers):
        for x, y, t in zip(pos_to_push, position, thing):
            grid[x[0]][x[1]] = t
            grid[y[0]][y[1]] = "."
        return

    pos_for_next_push = []
    new_thing = []
    for newx, t in zip(pos_to_push, thing):
        t_new = grid[newx[0]][newx[1]]
        if t_new == ".":
            continue
        pos_for_next_push.append(newx)
        new_thing.append(t_new)
        if t == t_new:
            continue
        if t_new == "[":
            p_right = (newx[0], newx[1] + 1)
            pos_for_next_push.append(p_right)
            new_thing.append("]")
        if t_new == "]":
            p_left = (newx[0], newx[1] - 1)
            pos_for_next_push.append(p_left)
            new_thing.append("[")

    # Will raise exception if can't move
    move2(pos_for_next_push, direction, grid, new_thing)

    for p in position:
        grid[p[0]][p[1]] = "."
    for p, t in zip(pos_to_push, thing):
        grid[p[0]][p[1]] = t


def p15b(fn):
    grid, moves = read15(fn)
    grid = double_grid(grid)
    for line in grid:
        print("".join(line))
    print()
    print(moves)
    print()
    robot = get_robot_position(grid)
    for d in moves:
        dir = DIR_MAP[d]
        try:
            move2((robot,), dir, grid)
        except CantMove:
            pass
        else:
            robot = (robot[0] + dir[0], robot[1] + dir[1])
        # finally:
        #     print(f"Move {d}:")
        #     for line in grid:
        #         print("".join(line))
        #     print()

    total = 0
    for i, row in enumerate(grid):
        for j, symbol in enumerate(row):
            if symbol == "[":
                total += 100 * i + j
    print(total)


###############################################################################
#################################### Day 16 ###################################
###############################################################################


def read16(fn):
    with open(fn) as f:
        grid = []
        for line in f:
            line = line.strip()
            grid.append(list(line))

        return grid


def get_start16(grid):
    for idx, row in enumerate(grid):
        for jdx, char in enumerate(row):
            if char == "S":
                return Point(idx, jdx)


def get_rotations(p):
    if p.x == 0:
        return Point(1, 0), Point(-1, 0)
    return Point(0, 1), Point(0, -1)


def p16a(fn):
    grid = read16(fn)
    start = get_start16(grid)

    heap = [(0, start, Point(0, 1))]
    used = set()
    while heap:
        score, position, orientation = heappop(heap)
        if (position, orientation) in used:
            continue
        else:
            used.add((position, orientation))

        if grid[position.x][position.y] == "E":
            print(score)
            return score
        if grid[position.x][position.y] == "#":
            continue

        heappush(heap, (score + 1, position + orientation, orientation))
        for new_ori in get_rotations(orientation):
            heappush(heap, (score + 1001, position + new_ori, new_ori))


def p16b(fn):
    best_score = p16a(fn)
    grid = read16(fn)
    start = get_start16(grid)

    heap = [(0, [start], [Point(0, 1)])]
    used = defaultdict(int)
    visitations = defaultdict(set)
    best_tiles = set()
    while heap:
        score, position_list, orientation_list = heappop(heap)
        if score > best_score:
            break
        position = position_list[-1]
        orientation = orientation_list[-1]
        if (position, orientation) not in used:
            used[(position, orientation)] = score
            visitations[(position, orientation)].update(position_list)
        elif used[((position, orientation))] == score:
            visitations[(position, orientation)].update(position_list)
            continue
        else:
            continue

        if grid[position.x][position.y] == "E":
            # print(score)
            # print(position_list)
            for p, ori in zip(position_list, orientation_list):
                best_tiles.update(visitations[p, ori])

        if grid[position.x][position.y] == "#":
            continue

        heappush(
            heap,
            (
                score + 1,
                position_list + [position + orientation],
                orientation_list + [orientation],
            ),
        )
        for new_ori in get_rotations(orientation):
            heappush(
                heap,
                (
                    score + 1001,
                    position_list + [position + new_ori],
                    orientation_list + [new_ori],
                ),
            )

    print(len(best_tiles))


###############################################################################
#################################### Day 17 ###################################
###############################################################################


@dataclass
class Register:
    A: int = 0
    B: int = 0
    C: int = 0


test17 = [Register(A=729), [0, 1, 5, 4, 3, 0]]
real17 = [Register(A=51342988), [2, 4, 1, 3, 7, 5, 4, 0, 1, 3, 0, 3, 5, 5, 3, 0]]


def combo(x, register):
    if x < 4:
        return x
    if x == 4:
        return register.A
    if x == 5:
        return register.B
    if x == 6:
        return register.C


def adv(x, register):
    register.A = register.A >> combo(x, register)


def bxl(x, register):
    register.B ^= x


def bst(x, register):
    register.B = combo(x, register) % 8


def jnz(x, register):
    if register.A == 0:
        return
    return x


def bxc(_, register):
    register.B ^= register.C


def out(x, register):
    return str(combo(x, register) % 8)


def bdv(x, register):
    register.B = register.A >> combo(x, register)


def cdv(x, register):
    register.C = register.A >> combo(x, register)


instructions = [adv, bxl, bst, jnz, bxc, out, bdv, cdv]


def execute(register, program):
    idx = 0
    final = []
    while idx < len(program):
        op = instructions[program[idx]]
        out = op(program[idx + 1], register)
        if isinstance(out, str):
            final.append(out)

        if isinstance(out, int):
            idx = out
        else:
            idx += 2
    return ",".join(final)


examples = [
    # [Register(C=9), [2, 6]],
    # [Register(A=10), [5, 0, 5, 1, 5, 4]],
    # [Register(A=2024), [0, 1, 5, 4, 3, 0]],
    [Register(B=29), [1, 7]],
    [Register(B=2024, C=43690), [4, 0]],
]


def p17a(register, program):
    # for register, program in examples:
    #     print(register)
    #     out = execute(register, program)
    #     print(register)
    #     print(out)
    #     print("-" * 80)
    print(execute(register, program))


def match(output, program):
    target, *rest = output
    if not rest:
        for a in range(8):
            out = int(execute(Register(A=a), program))
            if out == target:
                yield a
    else:
        for a in match(rest, program):
            for i in range(8):
                new_a = (8 * a) + i
                out = int(execute(Register(A=new_a), program))
                if out == target:
                    yield new_a


def p17b(_, program):
    for x in sorted(match(program, program[:-2])):
        e = list(map(int, execute(Register(A=x), program).split(",")))
        if e == program:
            print(x, execute(Register(A=x), program))


test17b = [Register(2024), [0, 3, 5, 4, 3, 0]]

###############################################################################
#################################### Day 18 ###################################
###############################################################################


def read18(fn):
    ret = []
    with open(fn) as f:
        for line in f:
            x, y = map(int, line.split(","))
            ret.append(Point(x, y))
    return ret


def p18a(fn, bound=6, stop=12):
    corrupted = read18(fn)[:stop]
    start = Point(0, 0)
    target = Point(bound, bound)
    visited = {start}
    visited.update(corrupted)
    heap = [(0, start)]
    while heap:
        # print(heap)
        steps, p = heappop(heap)
        if p == target:
            print(steps)
            return steps
        for x in p.orthogonal(bound + 1, bound + 1):
            if x not in visited:
                heappush(heap, (steps + 1, x))
                visited.add(x)


def p18b(fn, bound=6):
    corrupted = read18(fn)
    for idx in range(len(corrupted)):
        steps = p18a(fn, bound=bound, stop=idx)
        if steps is None:
            p = corrupted[idx - 1]
            print(f"{p.x},{p.y}")
            break


###############################################################################
#################################### Day 18 ###################################
###############################################################################


def read19(fn):
    with open(fn) as f:
        line = next(f)
        towels = frozenset({x.strip() for x in line.split(",")})
        next(f)
        patterns = [x.strip() for x in f]
    return towels, patterns


@cache
def get_patterns(pattern, towels, max_len):
    if not pattern:
        return 1

    vals = 0
    for i in range(1, min(len(pattern), max_len) + 1):
        t1 = pattern[:i]
        if t1 in towels:
            vals += get_patterns(pattern[i:], towels, max_len)
    return vals


def p19a(fn):
    towels, patterns = read19(fn)
    max_towel_length = max(len(x) for x in towels)
    print(max_towel_length)

    count = 0
    for p in patterns:
        gp = get_patterns(p, towels, max_len=max_towel_length)
        print(p, gp)
        if gp:
            count += 1
    print(count)


def p19a(fn):
    towels, patterns = read19(fn)
    max_towel_length = max(len(x) for x in towels)
    print(max_towel_length)

    count = 0
    for p in patterns:
        gp = get_patterns(p, towels, max_len=max_towel_length)
        print(p, gp)
        count += gp
    print(count)


###############################################################################
#################################### Day 20 ###################################
###############################################################################


def read20(fn):
    ret = []
    with open(fn) as f:
        for line in f:
            row = [c for c in line.strip()]
            ret.append(row)
    return ret


def find_element(elt, grid):
    for idx, row in enumerate(grid):
        for jdx, col in enumerate(row):
            if col == elt:
                return Point(idx, jdx)


def make_dict(start, grid):
    rows = len(grid)
    cols = len(grid[0])
    fastest = {start: 0}
    heap = [(0, start)]
    while heap:
        score, p = heappop(heap)
        for neighbor in p.orthogonal(rows, cols):
            if grid[neighbor.x][neighbor.y] != "#":
                if neighbor not in fastest:
                    fastest[neighbor] = score + 1
                    heappush(heap, (score + 1, neighbor))
    return fastest


def improve(grid, cheat, fastest):
    rows = len(grid)
    cols = len(grid[0])
    neighbors = [p for p in cheat.orthogonal(rows, cols) if grid[p.x][p.y] != "#"]
    if len(neighbors) != 2:
        return False

    n0, n1 = neighbors
    for x, y in [(n0, n1), (n1, n0)]:
        if fastest[x] - fastest[y] >= 100 + 2:
            return True
    return False


def improve2(grid, cheat, fastest):
    rows = len(grid)
    cols = len(grid[0])

    position = fastest[cheat]
    count = 0

    for y in cheat.taxicab(20, rows, cols):
        if y in fastest:
            dist = abs(cheat.x - y.x) + abs(cheat.y - y.y)
            if fastest[y] - position >= 100 + dist:
                # print(cheat, y, dist, position, fastest[y])
                count += 1
    return count


def p20a(fn):
    grid = read20(fn)

    start = find_element("S", grid)
    d = make_dict(start, grid)

    count = 0
    for idx, row in enumerate(grid):
        for jdx, col in enumerate(row):
            if improve2(grid, Point(idx, jdx), d):
                count += 1
    print(count)


def p20b(fn):
    grid = read20(fn)
    start = find_element("S", grid)
    d = make_dict(start, grid)

    count = 0
    for idx, row in enumerate(grid):
        for jdx, col in enumerate(row):
            if grid[idx][jdx] == "#":
                continue
            count += improve2(grid, Point(idx, jdx), d)
    print(count)


###############################################################################
#################################### Day 21 ###################################
###############################################################################

test21 = """029A
980A
179A
456A
379A""".split(
    "\n"
)

input21 = """413A
480A
682A
879A
083A""".split(
    "\n"
)

dir_to_char = {
    Point(1, 0): "v",
    Point(-1, 0): "^",
    Point(0, 1): ">",
    Point(0, -1): "<",
}

char_to_dir = {y: x for (x, y) in dir_to_char.items()}

KEYPAD = {
    "7": Point(0, 0),
    "8": Point(1, 0),
    "9": Point(2, 0),
    "4": Point(0, 1),
    "5": Point(1, 1),
    "6": Point(2, 1),
    "1": Point(0, 2),
    "2": Point(1, 2),
    "3": Point(2, 2),
    "0": Point(1, 3),
    "A": Point(2, 3),
}

KEYPAD_R = {y: x for (x, y) in KEYPAD.items()}

DIRPAD = {
    "^": Point(1, 0),
    "A": Point(2, 0),
    "<": Point(0, 1),
    "v": Point(1, 1),
    ">": Point(2, 1),
}

DIRPAD_R = {y: x for (x, y) in DIRPAD.items()}


def get_paths_on_keyboard(k1, k2, keyboard=None, keyboard_r=None):
    if keyboard is None:
        keyboard = DIRPAD
    if keyboard_r is None:
        keyboard_r = DIRPAD_R

    if k1 not in keyboard_r:
        return

    if k1 == k2:
        yield "A"
        return

    if k1.x > k2.x:
        new_k1 = Point(k1.x - 1, k1.y)
        for x in get_paths_on_keyboard(new_k1, k2, keyboard, keyboard_r):
            yield "<" + x

    if k1.x < k2.x:
        new_k1 = Point(k1.x + 1, k1.y)
        for x in get_paths_on_keyboard(new_k1, k2, keyboard, keyboard_r):
            yield ">" + x

    if k1.y > k2.y:
        new_k1 = Point(k1.x, k1.y - 1)
        for x in get_paths_on_keyboard(new_k1, k2, keyboard, keyboard_r):
            yield "^" + x

    if k1.y < k2.y:
        new_k1 = Point(k1.x, k1.y + 1)
        for x in get_paths_on_keyboard(new_k1, k2, keyboard, keyboard_r):
            yield "v" + x


def generate(code, keyboard=None, keyboard_r=None):
    if keyboard is None:
        keyboard = DIRPAD
    if keyboard_r is None:
        keyboard_r = DIRPAD_R

    if len(code) == 2:
        yield from get_paths_on_keyboard(
            keyboard[code[0]],
            keyboard[code[1]],
            keyboard=keyboard,
            keyboard_r=keyboard_r,
        )
        return

    paths = get_paths_on_keyboard(
        keyboard[code[0]],
        keyboard[code[1]],
        keyboard=keyboard,
        keyboard_r=keyboard_r,
    )
    gen = generate(code[1:], keyboard=keyboard, keyboard_r=keyboard_r)
    yield from ("".join(x) for x in itertools.product(paths, gen))


def get_length(k1, k2, keyboard=None):
    if keyboard is None:
        keyboard = DIRPAD

    return (keyboard[k1] - keyboard[k2]).abs() + 1


def p21a():
    score = 0
    for x in input21:
        min_len = 10**100
        for p1 in generate("A" + x, keyboard=KEYPAD, keyboard_r=KEYPAD_R):
            for p2 in generate("A" + p1):
                # for p3 in generate("A" + p2):
                length = sum(get_length(x, y) for (x, y) in zip("A" + p2, p2))
                min_len = min(min_len, length)
        val = int(x[:-1])
        print(x, min_len, val)
        score += min_len * val
    print(score)


def make_pair_lu():
    ret = dict()
    for x in itertools.product("A^v<>", repeat=2):
        key = "".join(x)
        v = ["A" + x for x in generate(key)]
        ret[key] = [tuple("".join(x) for x in zip(y, y[1:])) for y in v]
    return ret


PAIR_TRANSITIONS = make_pair_lu()


@cache
def path_cost(k1, k2, depth=1):
    if depth == 0:
        return 1

    best = 10**100

    for path in PAIR_TRANSITIONS[k1 + k2]:
        # print(k1, k2, depth, path)
        length = sum(path_cost(*x, depth=depth - 1) for x in path)
        best = min(best, length)
    return best


def p21b():
    print(PAIR_TRANSITIONS)
    score = 0
    for x in input21:
        print(x)
        min_len = 10**100

        for p1 in generate("A" + x, keyboard=KEYPAD, keyboard_r=KEYPAD_R):
            p1 = "A" + p1
            cost = sum(path_cost(x, y, 25) for (x, y) in zip(p1, p1[1:]))
            min_len = min(min_len, cost)
            print(x, p1, cost)

        val = int(x[:-1])
        print(x, min_len, val, min_len * val)
        score += min_len * val
    print(score)


###############################################################################
#################################### Day 21 ###################################
###############################################################################


def mix(x, y):
    return x ^ y


def prune(x):
    return x % 16777216


def evolve(secret):
    s = secret << 6  # 64*secret
    secret = prune(mix(secret, s))

    s = secret >> 5  # int(secret/32)
    secret = prune(mix(secret, s))

    s = secret << 11  # 2048*secret
    return prune(mix(secret, s))


def p22a(fn):
    with open(fn) as f:
        secrets = [int(x) for x in f]

    total = 0
    for secret in secrets:
        s = secret
        for _ in range(2000):
            s = evolve(s)
        total += s
    print(total)

def make_seq_lu(secret):
    lu = dict()
    seq = tuple()
    for _ in range(2000):
        s = evolve(secret)
        p0, p1 = secret % 10, s % 10
        change = p1 - p0
        secret = s
        if len(seq) < 4:
            seq += (change,)
        else:
            seq = seq[1:] + (change,)
        if len(seq) == 4 and seq not in lu:
            lu[seq] = p1
    return lu

def p22b(fn):
    with open(fn) as f:
        secrets = [int(x) for x in f]

    keys = set()
    lookups = []
    for secret in secrets:
        lu = make_seq_lu(secret)
        lookups.append(lu)
        keys.update(lu.keys())

    max_key = None
    max_bananas = 0
    for key in keys:
        bananas = 0
        for lu in lookups:
            bananas += lu.get(key, 0)
        if bananas > max_bananas:
            max_bananas = bananas
            max_key = key
    print(max_key, max_bananas)



if __name__ == "__main__":
    t0 = time.time()
    p22b("data/day22.txt")
    print(f"Time: {time.time() - t0}")
