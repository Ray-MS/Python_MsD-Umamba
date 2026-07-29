from typing import Any, Callable, Dict

_dataset_entrypoints: Dict[str, Callable[..., Any]] = {}


def register_dataset(fn: Callable) -> Callable:
    dataset_name = fn.__name__
    _dataset_entrypoints[dataset_name] = fn
    return fn


def dataset_entrypoints(dataset_name: str):
    return _dataset_entrypoints[dataset_name]


def create_dataset(
    dataset_name: str,
    root,
    **kwargs,
):
    dataset_name = dataset_name.lower()
    return dataset_entrypoints(dataset_name)(root, **kwargs)
