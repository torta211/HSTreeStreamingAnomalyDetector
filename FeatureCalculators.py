from queue import Queue


class FeatureCalculator:
    """
    Derive from this class to provide a method for feature calculation
    """
    def __init__(self, name, initialization_steps=0):
        self.name = name
        self.initialization_steps = initialization_steps

    def calculate_new(self, input_value):
        return input_value["value"]


class RawValue(FeatureCalculator):
    def __init__(self):
        super().__init__("RawValue", 0)


class Diff(FeatureCalculator):
    def __init__(self):
        super().__init__("Diff", 1)
        self.last_value = 0

    def calculate_new(self, input_value):
        diff = input_value["value"] - self.last_value
        self.last_value = input_value["value"]
        return diff


class DiffFromRollingMean(FeatureCalculator):
    def __init__(self, rolling_mean_lag):
        super().__init__(f"DiffFromMean{rolling_mean_lag}", rolling_mean_lag - 1)
        self.buffer = Queue(rolling_mean_lag)
        while not self.buffer.full():
            self.buffer.put(0)
        self.sum = 0
        self.rolling_mean_lag = rolling_mean_lag

    def calculate_new(self, input_value):
        val = input_value["value"]
        self.sum += val - self.buffer.get()
        self.buffer.put(val)
        return val - self.sum / self.rolling_mean_lag


class HourOfTheDay(FeatureCalculator):
    def __init__(self):
        super().__init__("HourOfTheDay", 0)

    def calculate_new(self, input_value):
        return input_value["timestamp"].dt.hour


class DayOfTheWeek(FeatureCalculator):
    def __init__(self):
        super().__init__("DayOfTheWeek", 0)

    def calculate_new(self, input_value):
        return input_value["timestamp"].dt.dayofweek


class DiffDiffFromDiffRollingMean(FeatureCalculator):
    def __init__(self, rolling_mean_lag):
        super().__init__(f"DiffDiffFromDiffRollingMean{rolling_mean_lag}", rolling_mean_lag)
        self.buffer = Queue(rolling_mean_lag)
        while not self.buffer.full():
            self.buffer.put(0)
        self.sum = 0
        self.rolling_mean_lag = rolling_mean_lag
        self.last_value = 0

    def calculate_new(self, input_value):
        val = input_value["value"]
        diff = val - self.last_value
        self.last_value = val
        self.sum += diff - self.buffer.get()
        self.buffer.put(diff)
        return diff - self.sum / self.rolling_mean_lag


class DiffFromPastWeeksAvg(FeatureCalculator):
    def __init__(self, same_day_last_week_weight, samples_per_day=1440):
        super().__init__(f"DiffFromPastWeeksAvg{same_day_last_week_weight}", 7 * samples_per_day)
        self.buffer = Queue(7 * samples_per_day)
        while not self.buffer.full():
            self.buffer.put(0)
        self.samples_per_day = samples_per_day
        self.same_day_of_week_weight = same_day_last_week_weight
        self.weight_sum = 6 + same_day_last_week_weight

    def calculate_new(self, input_value):
        values_past_days = []
        for i in range(6):
            values_past_days.append(self.buffer.queue[i * self.samples_per_day])
        for i in range(self.same_day_of_week_weight):
            values_past_days.append(self.buffer.queue[6 * self.samples_per_day])
        val = input_value["value"]
        self.buffer.get()
        self.buffer.put(val)
        return val - sum(values_past_days) / self.weight_sum



