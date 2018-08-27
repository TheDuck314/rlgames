import time

class TimeTracker:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None

        self.part_start_times = {}
        self.part_seconds = {}

    def start(self, part):
        self.part_start_times[part] = time.time()

    def end(self, part=None):
        now = time.time()
        if part is not None:
            self.part_seconds[part] = self.part_seconds.get(part, 0.0) + now - self.part_start_times[part]
        else:
            self.end_time = now

    def get_total_seconds(self):
        return (self.end_time or time.time()) - self.start_time

    def __repr__(self):
        return "TimeTracker(total_seconds={}, part_seconds={})".format(self.total_seconds, self.part_seconds)


