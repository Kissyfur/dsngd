import time


def duration(seconds):
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f'{hours} h {minutes:02d} min'
    if minutes:
        return f'{minutes} min {seconds:02d} s'
    return f'{seconds} s'


class Progress:
    """Elapsed time and expected remaining time of a sequence of tasks.

    Each task has a key, such as (algorithm, manifold), and a relative cost. A remaining task is expected to take the
    mean time of the finished tasks with its key or, if there are none, its cost times the mean time per unit of cost
    of the finished tasks with the same first element of the key, or of all of them.
    """

    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.times = []
        self.start = self.last = time.time()

    def _expected(self, key, cost):
        finished = list(zip(self.tasks, self.times))
        for match in (lambda k: k == key, lambda k: k[0] == key[0], lambda k: True):
            selected = [(c, t) for (k, c), t in finished if match(k)]
            if selected:
                return cost * sum(t for _, t in selected) / sum(c for c, _ in selected)

    def step(self):
        # Marks the next task as finished and describes the progress
        now = time.time()
        self.times.append(now - self.last)
        self.last = now
        remaining = sum(self._expected(key, cost) for key, cost in self.tasks[len(self.times):])
        return (f'{len(self.times)}/{len(self.tasks)} done, elapsed {duration(now - self.start)}, '
                f'expected remaining {duration(remaining)}')

    def elapsed(self):
        return duration(time.time() - self.start)
