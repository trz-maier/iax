from math import floor


def progress_bar(current, maximum, length: int = 50):
    progress = floor(current/maximum * 100)
    l1 = floor(progress/(100/length))
    l2 = length - l1
    return "[%s%s] %s%%" % ('#'*l1, ' '*l2, str(progress))


