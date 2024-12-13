import copy
from dataclasses import dataclass
from functools import total_ordering
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
#################################### Day 12 ###################################
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


if __name__ == "__main__":
    t0 = time.time()
    p13b("data/day13.txt")
    print(f"Time: {time.time() - t0}")
