from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, TypeVar


FILE = Path() / "docs" / "day08.txt"

Instructions = TypeVar("Instructions", bound=str)
NodeMap = Dict[str, Tuple[str, str]]


def parse_file(file_name: Path) -> Tuple[Instructions, NodeMap]:
    instructions: Instructions = ""
    node_map: NodeMap = {}
    with open(file_name) as f:
        for line in f:
            line = line.rstrip("\n")
            if not instructions:
                instructions = line
                continue
            if not line:
                continue
            key, rest = line.split(" = ")
            left, right = rest[1:-1].split(", ")
            node_map[key] = (left, right)
    return instructions, node_map


instructions, node_map = parse_file(FILE)


def num_steps(
    instructions: Instructions, node_map: NodeMap, start="AAA", end="ZZZ"
) -> int:
    cur_node = start
    for idx, direction in enumerate(itertools.cycle(instructions)):
        if cur_node == end:
            return idx
        if direction == "L":
            cur_node = node_map[cur_node][0]
        else:
            cur_node = node_map[cur_node][1]


# print(f"1: {num_steps(instructions, node_map)}")


def ghost_step(
    instructions: Instructions, node_map: NodeMap, start="A", end="Z"
) -> int:
    cur_node = [x for x in node_map if x.endswith(start)]
    for idx, direction in enumerate(itertools.cycle(instructions)):
        if all(x.endswith(end) for x in cur_node):
            return idx
        if direction == "L":
            cur_node = [node_map[x][0] for x in cur_node]
        else:
            cur_node = [node_map[x][1] for x in cur_node]


@dataclass
class Cycle:
    offset: int
    length: int
    indices: Set[int]

    def __contains__(self, value: int):
        if value < self.offset:
            return False
        return (value - self.offset) % self.length in self.indices

    @classmethod
    def from_map(
        cls, instructions: Instructions, node_map: NodeMap, start: str, end="Z"
    ):
        visited: List[str] = []
        cur_node = start
        for idx, direction in itertools.cycle(enumerate(instructions)):
            if (idx, cur_node) in visited:
                offset = visited.index((idx, cur_node))
                cycle = visited[offset:]
                indices = [idx for idx, x in enumerate(cycle) if x[1].endswith(end)]
                return Cycle(offset=offset, length=len(cycle), indices=set(indices))
            else:
                visited.append((idx, cur_node))
                if direction == "L":
                    cur_node = node_map[cur_node][0]
                else:
                    cur_node = node_map[cur_node][1]


def ghost_step2(
    instructions: Instructions, node_map: NodeMap, start="A", end="Z"
) -> int:
    cur_node = [x for x in node_map if x.endswith(start)]
    cycles = [Cycle.from_map(instructions, node_map, x, end=end) for x in cur_node]
    for x in cycles:
        print(x)
    return math.lcm(*(c.length for c in cycles))
    value = 0
    primary, *cycles = cycles
    while True:
        for modulus in primary.indices:
            check = primary.offset + modulus + value * primary.length
            if all(check in c for c in cycles):
                return check
        value += 1
        if value % 100_000 == 0:
            print(check)


print(f"2: {ghost_step2(instructions, node_map)}")
