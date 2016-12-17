def moving_average(a: int, w: int = 10)\
        -> [float]:
    """Computes the moving average.
    Ugly implementation, no need to understand."""
    if len(a) < w:
        return a[:]

    # for idx, val in enumerate(a):
    #     if idx < w:
    #         return val
    #     else:
    #         return (sum(a[(idx - w) : idx]) / w)
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]
