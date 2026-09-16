
def flatten_dict(d):
    """Recursively flatten a nested dict of dicts/lists into a single list.

    Args:
        d: A dict whose values may be dicts (recursed into) or lists (extended).
           Non-dict, non-list values are ignored.

    Returns:
        Flat list of all list items found at any depth.

    Example:
        >>> flatten_dict({'a': {'b': [1, 2]}, 'c': [3, 4]})
        [1, 2, 3, 4]
    """
    result = []
    for v in d.values():
        if isinstance(v, dict):
            result.extend(flatten_dict(v))
        elif isinstance(v, list):
            result.extend(v)
    return result