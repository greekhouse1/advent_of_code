import itertools
import re

from collections import Counter, defaultdict
from typing import List, Tuple

###############################################################################
#################################### Day 2 ####################################
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
#################################### Day 4 ####################################
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


if __name__ == "__main__":
    p5b("data/day5.txt")
