import numpy as np
import pandas as pd

import backend.modeling.residual_tracker as residual_tracker


def test_predict_remaining_rise_uses_model_feature_names(monkeypatch):
    class FakeModel:
        feature_names_in_ = np.array(["hour_local", "temp_f"])

        def predict(self, X):
            assert isinstance(X, pd.DataFrame)
            assert list(X.columns) == ["hour_local", "temp_f"]
            assert X.iloc[0]["hour_local"] == 11
            assert X.iloc[0]["temp_f"] == 80.0
            return [3.25]

    monkeypatch.setattr(residual_tracker, "_model", FakeModel())
    monkeypatch.setattr(residual_tracker, "_features", [])
    monkeypatch.setattr(residual_tracker, "_loaded", True)

    assert residual_tracker.predict_remaining_rise(11, 80.0) == 3.25
