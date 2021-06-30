import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List

from HSTreeModel import (
    build_trees,
    update_model,
    update_mass_next_model,
    score_tree
)
from FeatureCalculators import FeatureCalculator


class HSTreeDetector:
    """
    How to interpret model weights:
    Let
    -> steps_recorded_in_model = 10080 (1 week)
    -> model_update_period = 1440 (1 day)
    -> relative_weight_old_model = 0.9
    This results
    -> self.weight_old_model = 0.9
    -> self.weight_new_model = 0.7
    Let's suppose
    -> we have a model with a mass=21 in some node, this means 3 records / day
    -> the new model has mass=2 in this node, this means 2 records / day
    -> specifying relative_weight_old_model=0.9, we expect the new model to have
       mass=(0.9*3+0.1*2)=2.9/day=20.3 in this node
    -> the result is indeed 20.3(=0.9*21+0.7*2)
    """
    def __init__(self,
                 features: List[FeatureCalculator],
                 steps_recorded_in_model=10080,
                 model_update_period=1440,
                 relative_weight_old_model=0.9,
                 num_trees=60,
                 depth_trees=12):
        self.trees = build_trees(len(features), num_trees, depth_trees)
        self.features = features
        self.model_update_period = model_update_period
        self.features_initialization_steps = max([f.initialization_steps for f in features])
        self.steps_recorded_in_model = steps_recorded_in_model
        self.initialization_steps = steps_recorded_in_model + self.features_initialization_steps
        self.weight_old_model = relative_weight_old_model
        self.weight_new_model = (1 - relative_weight_old_model) * steps_recorded_in_model / model_update_period
        self.features_past_values = pd.DataFrame(
            {feature.name: np.zeros((steps_recorded_in_model)) for feature in features})
        self.current_normalizers = {feature.name: {"min": 0, "max": 0} for feature in features}
        self.steps_done = 0
        self.steps_since_update = 0
        self.score_divisor = num_trees + sum([2**k for k in range(depth_trees)])

        self.scores = []
        self.scored_inputs = []
        self.scored_times = []
        self.features_to_plot_orig = {feature.name: [] for feature in features}
        self.features_to_plot_normalized = {feature.name: [] for feature in features}

    def _recalculate_normalizer_extrema(self):
        for feature in self.features:
            self.current_normalizers[feature.name]["min"] = np.percentile(self.features_past_values[feature.name], 1)
            self.current_normalizers[feature.name]["max"] = np.percentile(self.features_past_values[feature.name], 99)

    def _shift_past_values(self):
        self.features_past_values.values[: self.steps_recorded_in_model - self.model_update_period, :] = \
            self.features_past_values.values[self.model_update_period:, :]

    def _perform_model_update(self, weight_old_model, weight_new_model):
        for tree in self.trees:
            update_model(tree, weight_old=weight_old_model, weight_new=weight_new_model)
        print(f"{self.steps_done} steps done, model updated")

    def _record_feature_row(self, feature_row):
        for tree in self.trees:
            update_mass_next_model(feature_row, tree)

    def _score_feature_row(self, feature_row):
        score = 0
        for tree in self.trees:
            score += score_tree(feature_row, tree)
        return score / self.score_divisor

    def _step_features_uninitialized(self, value):
        for feature in self.features:
            feature.calculate_new(input_value=value)

    def _step_model_uninitialized(self, value):
        row = self.steps_done - self.features_initialization_steps
        for feature in self.features:
            self.features_past_values[feature.name][row] = feature.calculate_new(input_value=value)

    def _initialize_model(self):
        self._recalculate_normalizer_extrema()
        data = self.features_past_values.copy()
        for feature in self.features:
            feat_min = self.current_normalizers[feature.name]["min"]
            feat_max = self.current_normalizers[feature.name]["max"]
            data[feature.name] = (data[feature.name] - feat_min) / (feat_max - feat_min)
        data = data.values
        for row in range(data.shape[0]):
            self._record_feature_row(data[row, :])
        self._perform_model_update(weight_old_model=0, weight_new_model=1)
        print(f"HSTreeDetector reached initial state, starting scoring now steps done = ({self.steps_done})")

    def _step_model_initialized(self, value):
        row = self.steps_recorded_in_model - self.model_update_period + self.steps_since_update
        feats = []
        for feature in self.features:
            feature_value = feature.calculate_new(input_value=value)
            self.features_to_plot_orig[feature.name].append(feature_value)
            self.features_past_values[feature.name][row] = feature_value
            feature_min = self.current_normalizers[feature.name]["min"]
            feature_max = self.current_normalizers[feature.name]["max"]
            normalized_feature_value = (feature_value - feature_min) / (feature_max - feature_min)
            self.features_to_plot_normalized[feature.name].append(normalized_feature_value)
            feats.append(normalized_feature_value)

        feat_row = np.array(feats)
        self._record_feature_row(feat_row)
        self.scored_inputs.append(value["value"])
        self.scored_times.append(value["timestamp"])
        self.scores.append(self._score_feature_row(feat_row))

    def step(self, value):
        if self.steps_done < self.features_initialization_steps:
            self._step_features_uninitialized(value)
        elif self.steps_done < self.initialization_steps:
            self._step_model_uninitialized(value)
        else:
            if self.steps_done == self.initialization_steps:
                self._initialize_model()
                self._shift_past_values()

            self._step_model_initialized(value)

            self.steps_since_update += 1
            if self.steps_since_update == self.model_update_period:
                self._perform_model_update(weight_old_model=self.weight_old_model,
                                           weight_new_model=self.weight_new_model)
                self._recalculate_normalizer_extrema()
                self._shift_past_values()
                self.steps_since_update = 0
        self.steps_done += 1

    def create_plot(self):
        fig = make_subplots(rows=2, cols=1,
                            specs=[[{"rowspan": 1, "secondary_y": False}],
                                   [{"rowspan": 1, "secondary_y": False}]],
                            vertical_spacing=0.02, shared_xaxes=True)
        fig.append_trace(go.Scattergl(x=self.scored_times,
                                      y=self.scored_inputs, name="original data",
                                      line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=self.scored_times,
                                   y=self.scores,
                                   name=f"Raw anomaly score",
                                   line=dict(color='black', width=2)), row=2, col=1)
        print("created output plot")
        return fig

    def plot_features(self, normalized=False):
        fig = make_subplots(rows=len(self.features), cols=1,
                            specs=[[{"rowspan": 1, "secondary_y": False}] for _ in range(len(self.features))],
                            vertical_spacing=0.02, shared_xaxes=True)
        for i in range(len(self.features)):
            feat = self.features_to_plot_normalized[self.features[i].name] if normalized else \
                self.features_to_plot_orig[self.features[i].name]
            fig.append_trace(go.Scattergl(x=self.scored_times,
                                          y=feat, name=self.features[i].name,
                                          line=dict(color='blue', width=2)), row=i+1, col=1)
        print("created features plot")
        return fig






