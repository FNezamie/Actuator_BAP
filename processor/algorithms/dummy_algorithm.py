class DummyMeasurements(object):
    def __init__(self):
        self.INDEX_STEP = 30  # Max FPS
        self.MAX_ITERATIONS = 20

        self.sim_measurements = [640 if i % self.INDEX_STEP == 0 else None for i in range(1, self.INDEX_STEP * self.MAX_ITERATIONS)]
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.index >= self.INDEX_STEP * self.MAX_ITERATIONS - 1:
            raise StopIteration()
        self.index += 1

        return self.sim_measurements[self.index]
