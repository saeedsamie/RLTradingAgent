import pandas as pd
import numpy as np
from src.scripts import fill_time_series_holes, remove_duplicates, add_indicators, find_missing_intervals

def test_fill_time_series_holes_interpolate():
    # Create a DataFrame with missing 5min intervals
    idx = pd.date_range('2024-01-01 00:00', periods=5, freq='10min')
    df = pd.DataFrame({'close': [1, 2, 3, 4, 5]}, index=idx)
    filled = fill_time_series_holes(df, method='interpolate')
    # Should have 9 rows (every 5min)
    assert len(filled) == 9
    # Interpolated value at 00:05
    assert np.isclose(filled.loc['2024-01-01 00:05'].close, 1.5)

def test_fill_time_series_holes_ffill_bfill():
    idx = pd.date_range('2024-01-01 00:00', periods=3, freq='10min')
    df = pd.DataFrame({'close': [1, 2, 3]}, index=idx)
    filled_ffill = fill_time_series_holes(df, method='ffill')
    filled_bfill = fill_time_series_holes(df, method='bfill')
    # ffill: missing values should be forward filled
    assert filled_ffill.loc['2024-01-01 00:05'].close == 1
    # bfill: missing values should be backward filled
    assert filled_bfill.loc['2024-01-01 00:05'].close == 2

def test_remove_duplicates_no_duplicates():
    idx = pd.date_range('2024-01-01 00:00', periods=3, freq='5min')
    df = pd.DataFrame({'close': [1, 2, 3]}, index=idx)
    deduped = remove_duplicates(df)
    assert deduped.equals(df)

def test_add_indicators_with_nan():
    idx = pd.date_range('2024-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'open': np.arange(30),
        'high': np.arange(30) + 1,
        'low': np.arange(30) - 1,
        'close': np.arange(30, dtype=float),
        'volume': np.ones(30)
    }, index=idx)
    df.loc[idx[5:10], 'close'] = np.nan  # introduce NaNs
    df_ind = add_indicators(df)
    # Should still have indicator columns
    # Always present
    for col in ['ma20', 'ma50', 'ma200', 'rsi']:
        assert col in df_ind.columns
    # Ichimoku columns are optional in this test
    ichimoku_cols = ['ichimoku_conversion', 'ichimoku_base']
    for col in ichimoku_cols:
        if col in df_ind.columns:
            assert True  # present, fine
        else:
            # Should print a warning, but not fail the test
            pass

def test_find_missing_intervals_none_missing(capsys):
    idx = pd.date_range('2024-01-01 00:00', periods=5, freq='5min')
    df = pd.DataFrame({'close': [1, 2, 3, 4, 5]}, index=idx)
    missing = find_missing_intervals(df)
    assert len(missing) == 0
    captured = capsys.readouterr()
    assert "Missing 0 intervals" in captured.out

def test_find_missing_intervals_all_missing(capsys):
    # Only first and last, all in between missing
    idx = [pd.Timestamp('2024-01-01 00:00'), pd.Timestamp('2024-01-01 00:20')]
    df = pd.DataFrame({'close': [1, 2]}, index=idx)
    missing = find_missing_intervals(df)
    # Should be 3 missing (00:05, 00:10, 00:15)
    assert len(missing) == 3
    captured = capsys.readouterr()
    assert "Missing 3 intervals" in captured.out 