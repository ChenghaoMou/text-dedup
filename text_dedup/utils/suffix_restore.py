from collections import deque


def restore(offsets, seg_file: str):

    indices = deque([])
    with open(seg_file) as f:
        for line in f:
            try:
                x, y = line.strip().split(" ", 1)
                if x.isdigit() and y.isdigit():
                    indices.append((int(x), int(y)))
            except:
                pass

    for i, (start, end) in enumerate(offsets):
        while indices:
            x, y = indices.popleft()
            while y <= start and indices:
                x, y = indices.popleft()

            if y <= start:
                break

            if x >= end:
                indices.appendleft((x, y))
                break

            if start <= x < end <= y:
                yield i, (x - start, end - start)
                if y > end:
                    indices.appendleft((end, y))
                break
            elif start <= x < y <= end:
                yield i, (x - start, y - start)
                continue
            elif x < start < y <= end:
                yield i, (0, y - start)
                continue
            elif x < start < end <= y:
                yield i, (0, end - start)
                if y > end:
                    indices.appendleft((end, y))
                break
