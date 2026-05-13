from backend.ingestion.forecasts import _apply_open_meteo_outlier_gate


def test_open_meteo_ai_outlier_is_stored_as_parse_error():
    high, parse_error, gate = _apply_open_meteo_outlier_gate(
        source_key="ecmwf_aifs",
        high_f=48.0,
        reference_median=80.0,
        unit="F",
    )

    assert high is None
    assert parse_error == "outlier_vs_reference"
    assert gate == {
        "reference_median": 80.0,
        "raw_high_f": 48.0,
        "delta": 32.0,
        "threshold": 12.0,
    }


def test_open_meteo_operational_model_is_not_outlier_gated():
    high, parse_error, gate = _apply_open_meteo_outlier_gate(
        source_key="ecmwf_ifs",
        high_f=48.0,
        reference_median=80.0,
        unit="F",
    )

    assert high == 48.0
    assert parse_error is None
    assert gate is None
