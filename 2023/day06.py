from typing import List


TEST_TIMES = [7, 15, 30]
TEST_DISTANCES = [9, 40, 200]

REAL_TIMES = [47, 98, 66, 98]
REAL_DISTANCES = [400, 1213, 1011, 1540]

TIMES = REAL_TIMES
DISTANCES = REAL_DISTANCES

# ht + st = tt
# d = ht*st = ht*(tt - ht)


def race_distances(total_time: int) -> List[int]:
    return [t * (total_time - t) for t in range(total_time + 1)]


product = 1

for total_time, record_distance in zip(TIMES, DISTANCES):
    distances = race_distances(total_time)
    num_ways_to_win = len([x for x in distances if x > record_distance])
    product *= num_ways_to_win

print(f"1: {product}")

TIMES = [int("".join(map(str, TIMES)))]
DISTANCES = [int("".join(map(str, DISTANCES)))]


for total_time, record_distance in zip(TIMES, DISTANCES):
    distances = race_distances(total_time)
    num_ways_to_win = len([x for x in distances if x > record_distance])

print(f"2: {num_ways_to_win}")
