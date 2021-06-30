import pandas as pd

from HSTreeDetector import HSTreeDetector
from FeatureCalculators import (
    RawValue,
    Diff,
    DiffFromRollingMean,
    DiffDiffFromDiffRollingMean,
    DiffFromPastWeeksAvg
)

stream = pd.read_csv("example_input_3.csv")
stream["timestamp"] = pd.to_datetime(stream["timestamp"])

features = [RawValue(), Diff()]

detector = HSTreeDetector(features, steps_recorded_in_model=200, model_update_period=50)

for i in range(stream.shape[0]):
    detector.step(stream.iloc[i])

detector.create_plot().write_html("out.html", auto_open=False)
detector.plot_features(normalized=False).write_html("feats.html", auto_open=False)
detector.plot_features(normalized=True).write_html("feats_norm.html", auto_open=False)




