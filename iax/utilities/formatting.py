from math import ceil


def format_seconds(seconds: int, granularity: int = 2):
    if seconds < 1:
        return '<1s'
    else:
        seconds = ceil(seconds)

    results = []

    intervals = (
        ('w', 604800),
        ('d', 86400),
        ('h', 3600),
        ('m', 60),
        ('s', 1)
    )

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            results.append("{}{}".format(value, name))

    return ''.join(results[:granularity])


def format_ordinal(x: int):
    d = {1: 'st', 2: 'nd', 3: 'rd'}
    last = x % 10
    if last in d.keys() and x not in (11, 12, 13):
        return "%s%s" % (x, d[last])
    else:
        return "%s%s" % (x, 'th')
