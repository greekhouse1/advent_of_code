from typing import Generator, Iterable


def strip_new_line(it: Iterable[str]) -> Generator[str, None, None]:
    for x in it:
        yield x.rstrip("\n")
