from collections import defaultdict
from dataclasses import dataclass
from functools import total_ordering
import itertools
from pathlib import Path
from typing import Dict, List, Tuple


FILE = Path() / "docs" / "day05.txt"


@dataclass
class SingleMap:
    source: range
    destination: range

    def __lt__(self, other):
        return self.source.start < other.source.start


class Map:
    def __init__(self):
        self.maps = []

    def add_ranges(self, source: range, destination: range):
        self.maps.append(SingleMap(source, destination))
        self.maps.sort()

    def _getitem_int(self, item: int) -> int:
        for single_map in self.maps:
            if item in single_map.source:
                idx = single_map.source.index(item)
                return single_map.destination[idx]

        return item

    def _getitem_range_single(self, item: range, maps: List[SingleMap]) -> List[range]:
        if not maps:
            return [item]
        if item.start == item.stop:
            return []

        this_map, *rest = maps

        if item.stop <= this_map.source.start:
            return [item]

        if item.start >= this_map.source.stop:
            return self._getitem_range_single(item, rest)

        if item.start < this_map.source.start:
            return [
                range(item.start, this_map.source.start)
            ] + self._getitem_range_single(
                range(this_map.source.start, item.stop), maps
            )

        # At this point, we have this_map.source.start <= item.start < this_map.source.end
        start_idx = this_map.source.index(item.start)
        if item.stop < this_map.source.stop:
            stop_idx = this_map.source.index(item.stop)
            return [this_map.destination[start_idx:stop_idx]]
        return [this_map.destination[start_idx:]] + self._getitem_range_single(
            range(this_map.source.stop, item.stop), rest
        )

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._getitem_int(item)

        return sum((self._getitem_range_single(x, self.maps) for x in item), start=[])


def parse_seeds(line: str) -> List[int]:
    "No error checking here!"
    return list(map(int, line[7:].split()))


def parse_key(line: str) -> Tuple[str, str]:
    x = line.split()[0]
    k1, _, k2 = x.split("-")
    return k1, k2


def parse_ranges(line: str) -> Tuple[range, range]:
    s1, s2, inc = map(int, line.split())
    return range(s1, s1 + inc), range(s2, s2 + inc)


def parse_file(
    file_name: str,
) -> Tuple[List[int], Tuple[str], Dict[Tuple[str, str], Map]]:
    seeds = None
    full_map = {}
    current_key = None
    key_order = None
    with open(file_name) as f:
        for line in f.read().split("\n"):
            if not line:
                continue
            elif not seeds:
                seeds = parse_seeds(line)
            elif "map" in line:
                current_key = parse_key(line)
                full_map[current_key] = Map()
                if key_order is None:
                    key_order = current_key
                else:
                    key_order += (current_key[1],)
            else:
                destination_range, source_range = parse_ranges(line)
                full_map[current_key].add_ranges(source_range, destination_range)

    return seeds, key_order, full_map


def get_location(
    seed_location: int, key_order: Tuple[str], full_map: Dict[Tuple[str, str], Map]
) -> int:
    current = seed_location
    for key in zip(key_order, key_order[1:]):
        print(f"{key[0]}: {current}, ", end="")
        current = full_map[key][current]
    print()
    return current


seeds, key_order, full_map = parse_file(FILE)

locations = []
for seed in seeds:
    location = get_location(seed, key_order, full_map)
    locations.append(location)

print(f"1: {min(locations) = }")


def parse_seeds(line: str) -> List[range]:
    "No error checking here!"
    all_ints = map(int, line[7:].split())
    ret = []
    while x := itertools.islice(all_ints, 2):
        x = tuple(x)
        if not x:
            break
        ret.append(range(x[0], x[0] + x[1]))
    return ret


def get_location(
    seed_location: List[range],
    key_order: Tuple[str],
    full_map: Dict[Tuple[str, str], Map],
) -> List[range]:
    current = seed_location
    for key in zip(key_order, key_order[1:]):
        current = full_map[key][current]
    return current


seeds, key_order, full_map = parse_file(FILE)
print(seeds)
locations = get_location(seeds, key_order, full_map)
print(locations)

print(f"2: {min(x.start for x in locations)}")
