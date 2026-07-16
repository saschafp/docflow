from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def maybe_tqdm(
    items: Iterable[T],
    enabled: bool = False,
    description: str | None = None,
) -> Iterator[T]:
    """Wrap an iterable in tqdm if progress display is enabled."""
    if not enabled:
        yield from items
        return

    try:
        from tqdm import tqdm
    except ImportError as error:
        raise ImportError(
            "Progress bars require tqdm. Install it with `pip install tqdm`."
        ) from error

    yield from tqdm(items, desc=description)
