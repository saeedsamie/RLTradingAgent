import pandas as pd


def load_data(csv_path):
    """Load the OHLCV+indicators dataset from CSV."""
    df = pd.read_csv(csv_path, parse_dates=[0])
    return df


def check_missing_intervals(df, time_col='time', freq='5T'):
    """Check and report missing 5-min intervals in the DataFrame."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    all_times = pd.date_range(df[time_col].iloc[0], df[time_col].iloc[-1], freq=freq)
    missing = all_times.difference(df[time_col])
    if len(missing) == 0:
        print('No missing intervals.')
    else:
        print(f'Missing intervals: {len(missing)}')
        print(missing)
    return missing
