"""
dataset_generator.py

Pipeline for downloading, processing, and augmenting XAUUSD (Gold/USD) 5-minute candlestick data with technical indicators.
- Downloads tick data from Alpari
- Aggregates to 5-minute OHLCV
- Fills missing intervals
- Adds technical indicators (MA, RSI, Ichimoku)
- Saves each step to a separate file
"""

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pandas_ta as ta
import requests

def download_and_aggregate_month(year: int, month: int) -> pd.DataFrame:
    """
    Download and aggregate tick data for a given year and month from Alpari.
    Returns a DataFrame of 5-minute OHLCV bars.
    """
    ym = f"{year:04d}{month:02d}"
    url = f"https://ticks.alpari.org/ecn1/XAUUSD/{year:04d}/{ym}_ecn1_XAUUSD.zip"
    print(f"Downloading {url}...")
    r = requests.get(url)
    r.raise_for_status()
    print(f"response {r}")

    monthly_frames = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for fname in z.namelist():
            if not fname.endswith('.txt'):
                continue
            print(f"  Parsing day file: {fname}")
            with z.open(fname) as f:
                df = pd.read_csv(
                    f,
                    sep='\t',
                    header=0,
                    usecols=['RateDateTime', 'RateBid', 'RateAsk'],
                    engine='python',
                    on_bad_lines='skip'
                )
            if df.empty:
                continue
            df['ts'] = pd.to_datetime(df['RateDateTime'], format="%Y-%m-%d\t%H:%M:%S.%f")
            df.set_index('ts', inplace=True)
            df['price'] = (df['RateBid'] + df['RateAsk']) / 2
            monthly_frames.append(df[['price']])

    if not monthly_frames:
        print("No data found for this month.")
        return pd.DataFrame()

    month_df = pd.concat(monthly_frames).sort_index()
    ohlc = month_df['price'].resample('5min').agg(['first', 'max', 'min', 'last'])
    ohlc.columns = ['open', 'high', 'low', 'close']
    ohlc['volume'] = month_df['price'].resample('5min').count()
    return ohlc.dropna()

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate index rows from the DataFrame, keeping the first occurrence.
    Prints duplicates if found.
    """
    df = df.copy()
    duplicates = df.index[df.index.duplicated(keep=False)]
    if not duplicates.empty:
        print("Duplicated index rows:")
        print(df.loc[duplicates])
    df = df[~df.index.duplicated(keep='first')]
    return df

def fetch_range(start_year: int, start_month: int, end_year: int, end_month: int) -> pd.DataFrame:
    """
    Download and aggregate data for a range of months (sequential).
    Returns a concatenated DataFrame of all months.
    """
    dates = pd.date_range(
        f"{start_year}-{start_month:02d}-01",
        f"{end_year}-{end_month:02d}-01",
        freq='MS'
    )
    frames = []
    for dt in dates:
        try:
            dfm = download_and_aggregate_month(dt.year, dt.month)
            if not dfm.empty:
                frames.append(dfm)
        except requests.HTTPError:
            print(f"Failed download {dt.strftime('%Y-%m')}")
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators (MA20, MA50, MA200, RSI, Ichimoku) to a DataFrame with OHLCV columns.
    Returns a new DataFrame with indicator columns.
    """
    df = df.copy()
    df['ma20'] = ta.sma(df['close'], length=20)
    df['ma50'] = ta.sma(df['close'], length=50)
    df['ma200'] = ta.sma(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'], length=14)
    ich = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52, include_chikou=True)
    if ich is not None and isinstance(ich, (list, tuple)) and ich[0] is not None:
        visible, _ = ich
        visible = visible.rename(columns={
            'ISA_9': 'ichimoku_conversion',
            'ISB_26': 'ichimoku_base',
            visible.columns[2]: 'ichimoku_leading_a',
            visible.columns[3]: 'ichimoku_leading_b',
            visible.columns[4]: 'ichimoku_chikou'
        })
        df = df.join(visible)
    else:
        print("Ichimoku could not be calculated for this input; skipping Ichimoku columns.")
    return df.dropna(how='all')

def fill_time_series_holes(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Reindexes the DataFrame to a complete 5-minute range and fills missing values.
    method: 'ffill', 'bfill', or 'interpolate'.
    Returns a DataFrame with no missing intervals.
    """
    df = df.copy()
    df = df.sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min')
    df_full = df.reindex(full_idx)
    if method == 'ffill':
        return df_full.ffill()
    elif method == 'bfill':
        return df_full.bfill()
    elif method == 'interpolate':
        return df_full.interpolate(method='time')
    else:
        raise ValueError("Method must be 'ffill', 'bfill', or 'interpolate'")

def find_missing_intervals(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Prints and returns missing 5-minute intervals in the DataFrame's index.
    """
    df = df.sort_index()
    print("Loaded data rows:", len(df))
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
    missing = full_idx.difference(df.index)
    print(f"Missing {len(missing)} intervals:")
    print(missing[:10])
    return missing

def download_and_aggregate_month_thread(year: int, month: int) -> pd.DataFrame:
    """
    Download and aggregate tick data for a given year and month (for parallel use).
    Returns a DataFrame of 5-minute OHLCV bars.
    """
    ym = f"{year:04d}{month:02d}"
    url = f"https://ticks.alpari.org/ecn1/XAUUSD/{year}/{ym}_ecn1_XAUUSD.zip"
    print(f"Downloading {url}…")
    r = requests.get(url)
    r.raise_for_status()
    daily_frames = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for fname in z.namelist():
            if not fname.endswith('.txt'):
                continue
            with z.open(fname) as f:
                df = pd.read_csv(
                    f,
                    sep='\t',
                    header=0,
                    usecols=['RateDateTime', 'RateBid', 'RateAsk'],
                    engine='python',
                    on_bad_lines='skip'
                )
            if df.empty:
                continue
            df['ts'] = pd.to_datetime(df['RateDateTime'], format="%Y-%m-%d %H:%M:%S.%f")
            df.set_index('ts', inplace=True)
            df['price'] = (df['RateBid'] + df['RateAsk']) / 2
            daily_frames.append(df[['price']])
    if not daily_frames:
        return pd.DataFrame()
    month_df = pd.concat(daily_frames).sort_index()
    ohlc = month_df['price'].resample('5min').agg(['first', 'max', 'min', 'last'])
    ohlc.columns = ['open', 'high', 'low', 'close']
    ohlc['volume'] = month_df['price'].resample('5min').count()
    return ohlc.dropna()

def fetch_range_parallel(start_year: int, start_month: int, end_year: int, end_month: int) -> pd.DataFrame:
    """
    Download and aggregate data for a range of months in parallel.
    Returns a concatenated DataFrame of all months.
    """
    dates = pd.date_range(f"{start_year}-{start_month:02d}-01", f"{end_year}-{end_month:02d}-01", freq='MS')
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_and_aggregate_month_thread, dt.year, dt.month): dt for dt in dates}
        frames = []
        for future in as_completed(futures):
            dt = futures[future]
            try:
                dfm = future.result()
                if not dfm.empty:
                    frames.append(dfm)
            except Exception as e:
                print(f"❗ {dt.strftime('%Y-%m')} failed:", e)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()

if __name__ == "__main__":
    """
    Main pipeline:
    1. Download and aggregate raw data (parallel)
    2. Fill missing intervals
    3. Add indicators
    4. Save each step to a separate file
    5. Print info and check for missing intervals
    """
    # 1. Download and aggregate (parallel)
    df5m = fetch_range_parallel(2015, 1, 2025, 7)
    df5m.to_csv("xauusd_5m_alpari_raw.csv")
    print(f"Raw 5m data saved: {len(df5m)} rows")

    # 2. Fill missing intervals
    df_filled = fill_time_series_holes(df5m, method='interpolate')
    df_filled.to_csv("xauusd_5m_alpari_filled.csv")
    print(f"Filled 5m data saved: {len(df_filled)} rows")

    # 3. Add indicators
    df_ind = add_indicators(df_filled)
    df_ind.to_csv("xauusd_5m_alpari_filled_indicated.csv")
    df_ind.to_parquet("xauusd_5m_alpari.parquet")
    print(f"Indicated data saved: {len(df_ind)} rows")

    # 4. Optionally, check for missing intervals in the final file
    find_missing_intervals(df_ind)
    print(df_ind.info())
    print(df_ind.head())
