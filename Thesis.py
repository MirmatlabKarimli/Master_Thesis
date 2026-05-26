from typing import NoReturn
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def rmse(y_true, y_pred) -> float:
    """Compute RMSE compatible with older sklearn versions."""
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Compute MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Compute MAPE and raise if all target values are zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    non_zero = y_true != 0
    if not np.any(non_zero):
        raise ValueError('MAPE cannot be computed because all target values are zero')
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)


def smape(y_true, y_pred) -> float:
    """Compute sMAPE and raise if all true/predicted denominators are zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    non_zero = denom != 0
    if not np.any(non_zero):
        raise ValueError('sMAPE cannot be computed because all denominators are zero')
    return float(np.mean(2.0 * np.abs(y_pred[non_zero] - y_true[non_zero]) / denom[non_zero]) * 100.0)


def empty_metrics() -> NoReturn:
    """Raise instead of returning placeholder metrics for invalid model inputs."""
    raise ValueError('Cannot compute metrics because the model evaluation set is empty or too short')


def _time_series_folds(
    n_samples: int,
    folds: int,
    min_train_size: int,
    test_size: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chronological expanding-window train/test index folds."""
    if folds < 1:
        raise ValueError('folds must be at least 1')
    if test_size < 1:
        raise ValueError('test_size must be at least 1')
    if n_samples <= min_train_size:
        raise ValueError('not enough samples for the requested time-series folds')

    max_folds = n_samples - min_train_size - test_size + 1
    effective_folds = min(folds, max_folds)
    if effective_folds < 1:
        raise ValueError('not enough samples for at least one time-series fold')

    test_starts = np.linspace(min_train_size, n_samples - test_size, effective_folds, dtype=int)
    splits = []
    for test_start in sorted(set(test_starts)):
        train_idx = np.arange(test_start)
        test_idx = np.arange(test_start, test_start + test_size)
        if len(train_idx) < min_train_size or len(test_idx) != test_size:
            raise ValueError('invalid time-series fold generated')
        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError('no time-series folds generated')
    return splits


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute forecast metrics for aligned actual and predicted values."""
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }


def _bootstrap_metric_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Estimate approximate metric CIs by resampling prediction pairs."""
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be at least 1')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be between 0 and 1')
    if len(y_true) < 2:
        raise ValueError('At least two prediction pairs are required for bootstrap CIs')

    rng = np.random.default_rng(random_state)
    metric_samples = {'rmse': [], 'mae': [], 'mape': [], 'smape': []}
    sample_size = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, sample_size, sample_size)
        sample_metrics = _point_metrics(y_true[idx], y_pred[idx])
        for metric_name, metric_value in sample_metrics.items():
            metric_samples[metric_name].append(metric_value)

    alpha = 1.0 - confidence
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    return {
        metric_name: (
            float(np.quantile(samples, lower_q)),
            float(np.quantile(samples, upper_q)),
        )
        for metric_name, samples in metric_samples.items()
    }


def _metrics_from_predictions(
    y_true_values: list,
    y_pred_values: list,
    n_bootstrap: int = 2000,
) -> dict:
    """Compute pooled metrics and approximate bootstrap CIs."""
    if not y_true_values or not y_pred_values:
        return empty_metrics()
    y_true = np.concatenate([np.asarray(values, dtype=float).reshape(-1) for values in y_true_values])
    y_pred = np.concatenate([np.asarray(values, dtype=float).reshape(-1) for values in y_pred_values])
    if len(y_true) != len(y_pred):
        raise ValueError('Cannot compute metrics because prediction and target lengths differ')

    metrics = _point_metrics(y_true, y_pred)
    cis = _bootstrap_metric_cis(y_true, y_pred, n_bootstrap=n_bootstrap)
    for metric_name, ci in cis.items():
        metrics[f'{metric_name}_ci'] = ci
    metrics['n_predictions'] = int(len(y_true))
    return metrics


def _format_metric(
    value: float,
    ci: tuple[float, float],
    suffix: str = '',
    decimals: int = 3,
) -> str:
    """Format a metric with its approximate 95% bootstrap CI."""
    return (
        f'{value:.{decimals}f}{suffix} '
        f'[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]{suffix}'
    )


def plot_seasonal_boxplots(df: pd.DataFrame, dataset_name: str, save_path: str | None = None) -> None:
    """Plot seasonal drug-level sale quantities by year.

    The input must contain Year, Season of the year, Drug ID, and Sale Quantity.
    Missing required columns or an empty seasonal aggregation raise ValueError.
    """
    required = {'Year', 'Season of the year', 'Drug ID', 'Sale Quantity'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{dataset_name}: missing required columns for seasonal plot: {sorted(missing)}')

    agg = (
        df.groupby(['Year', 'Season of the year', 'Drug ID'])['Sale Quantity']
        .sum()
        .reset_index()
    )
    if agg.empty:
        raise ValueError(f'{dataset_name}: no rows available for seasonal plot')

    years = sorted(pd.to_numeric(agg['Year']).unique())
    seasons = sorted(pd.to_numeric(agg['Season of the year']).unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = []
    positions = []
    box_years = []
    width = 0.8 / max(len(years), 1)

    cmap = plt.get_cmap('tab20', len(years))
    year_colors = {year: cmap(i) for i, year in enumerate(years)}

    for season_idx, season in enumerate(seasons):
        base = season_idx + 1
        for year_idx, year in enumerate(years):
            vals = agg[(agg['Season of the year'] == season) & (agg['Year'] == year)][
                'Sale Quantity'
            ]
            if vals.empty:
                continue
            box_data.append(vals)
            offset = (year_idx - (len(years) - 1) / 2) * width
            positions.append(base + offset)
            box_years.append(year)

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=False,
        whis=3,  # extend whiskers to 3*IQR to reduce flagged outliers
    )

    for patch, year in zip(bp['boxes'], box_years):
        patch.set_facecolor(year_colors[year])
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(seasons) + 1))
    ax.set_xticklabels([str(season) for season in seasons])
    ax.set_xlabel('Season')
    ax.set_ylabel('Total quantity sold per drug')
    ax.set_title(f'Seasonal total quantities by year – {dataset_name}')

    legend_handles = [plt.Line2D([0], [0], color=year_colors[year], lw=4, label=str(year)) for year in years]
    ax.legend(handles=legend_handles, title='Year')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)


def sanitize_sales(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Convert negative Sale Quantity values to positive magnitudes and return a copy."""
    if 'Sale Quantity' not in df.columns:
        raise ValueError(f'{dataset_name}: missing Sale Quantity column')
    df = df.copy()
    neg_count = (df['Sale Quantity'] < 0).sum()
    if neg_count:
        print(f'{dataset_name}: converting {neg_count} negative Sale Quantity rows to positive values')
        df['Sale Quantity'] = df['Sale Quantity'].abs()
    return df


def evaluate_arima(
    series: pd.Series,
    order=(1, 1, 1),
    test_size: int = 1,
    folds: int = 5,
    use_log: bool = False,
    print_params: bool = False,
    label: str | None = None,
) -> dict:
    """Fit ARIMA over chronological folds and return pooled holdout metrics.

    Each fold trains on earlier months and forecasts the next test window.
    Metrics are computed once across all fold predictions rather than averaged
    from per-fold scores.
    Raises ValueError when the series is too short for the requested folds.
    """
    from statsmodels.tsa.arima.model import ARIMA

    min_train_size = max(sum(order) + 1, 3)
    if len(series) <= min_train_size:
        return empty_metrics()

    series_use = np.log1p(series) if use_log else series
    y_true_values = []
    y_pred_values = []
    split_indices = _time_series_folds(len(series_use), folds, min_train_size, test_size)
    for fold_idx, (train_idx, test_idx) in enumerate(split_indices, start=1):
        train = series_use.iloc[train_idx]
        test = series_use.iloc[test_idx]
        model = ARIMA(train, order=order).fit()
        if print_params and fold_idx == 1:
            name = label or 'ARIMA'
            params = model.params.to_dict() if hasattr(model.params, 'to_dict') else model.params
            print(f'\n{name} parameters:')
            print(f'  order: {order}')
            print(f'  folds requested: {folds}')
            print(f'  folds used: {len(split_indices)}')
            print(f'  test_size: {test_size}')
            print(f'  params first fold: {params}')
        forecast = model.forecast(steps=len(test))

        y_true = np.expm1(test) if use_log else test
        y_pred = np.expm1(forecast) if use_log else forecast
        y_true_values.append(y_true)
        y_pred_values.append(y_pred)

    return _metrics_from_predictions(y_true_values, y_pred_values)


def arima_forecast_intervals(
    series: pd.Series,
    steps: int = 12,
    order=(1, 1, 1),
    use_log: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit ARIMA on the full series and return forecast intervals.

    When use_log is True, forecasts and confidence bounds are back-transformed
    to the original sales scale with expm1.
    """
    from statsmodels.tsa.arima.model import ARIMA

    if steps < 1:
        raise ValueError('steps must be at least 1')
    if len(series) < sum(order) + 1:
        raise ValueError('series is too short for the requested ARIMA order')

    series_use = np.log1p(series) if use_log else series
    model = ARIMA(series_use, order=order).fit()
    forecast_result = model.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=alpha)

    lower = forecast_ci.iloc[:, 0] if hasattr(forecast_ci, 'iloc') else forecast_ci[:, 0]
    upper = forecast_ci.iloc[:, 1] if hasattr(forecast_ci, 'iloc') else forecast_ci[:, 1]

    if use_log:
        forecast_mean = np.expm1(forecast_mean)
        lower = np.expm1(lower)
        upper = np.expm1(upper)

    if isinstance(series.index, pd.DatetimeIndex):
        start = series.index.max() + pd.offsets.MonthBegin(1)
        forecast_dates = pd.date_range(start=start, periods=steps, freq='MS')
    else:
        forecast_dates = pd.RangeIndex(start=1, stop=steps + 1)

    return pd.DataFrame(
        {
            'Horizon (months ahead)': range(1, steps + 1),
            'Forecast Date': forecast_dates,
            'Forecast': np.asarray(forecast_mean, dtype=float),
            'Lower 95% CI': np.asarray(lower, dtype=float),
            'Upper 95% CI': np.asarray(upper, dtype=float),
        }
    )


def _lstm_sequences(values: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i : i + look_back])
        y.append(values[i + look_back])
    X = np.array(X)
    y = np.array(y)
    return X, y


def evaluate_lstm(
    series: pd.Series,
    look_back: int = 6,
    units: int = 32,
    test_size: int = 1,
    epochs: int = 30,
    use_log: bool = False,
    folds: int = 5,
    print_params: bool = False,
    label: str | None = None,
) -> dict:
    """Fit a simple LSTM over chronological folds and return pooled metrics.

    Metrics are computed once across all fold predictions rather than averaged
    from per-fold scores.
    Raises ValueError when the series cannot support sequence construction or
    the requested time split.
    """
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    np.random.seed(42)
    tf.random.set_seed(42)

    if len(series) <= look_back + 1:
        return empty_metrics()

    series_use = np.log1p(series) if use_log else series

    values = series_use.values.reshape(-1, 1).astype(float)
    total = len(values) - look_back
    if total < 3:
        return empty_metrics()

    y_true_values = []
    y_pred_values = []
    split_indices = _time_series_folds(total, folds, min_train_size=2, test_size=test_size)
    for fold_idx, (train_idx, test_idx) in enumerate(split_indices, start=1):
        val_size = max(1, int(len(train_idx) * 0.2))
        if len(train_idx) <= val_size:
            raise ValueError('LSTM fold does not have enough training rows after validation split')

        fit_idx = train_idx[:-val_size]
        val_idx = train_idx[-val_size:]
        train_scaler_end = int(fit_idx[-1] + look_back + 1)

        scaler = MinMaxScaler()
        scaler.fit(values[:train_scaler_end])
        scaled = scaler.transform(values)
        X, y = _lstm_sequences(scaled, look_back)

        X_train, y_train = X[fit_idx], y[fit_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential(
            [
                LSTM(units, input_shape=(look_back, 1)),
                Dropout(0.2),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        if print_params and fold_idx == 1:
            name = label or 'LSTM'
            print(f'\n{name} parameters:')
            print(f'  units: {units}')
            print('  dropout: 0.2')
            print(f'  look_back: {look_back}')
            print(f'  epochs: {epochs}')
            print('  batch_size: 32')
            print('  learning_rate: 0.001')
            print(f'  folds requested: {folds}')
            print(f'  folds used: {len(split_indices)}')
            print(f'  test_size: {test_size}')
            print(f'  total_params: {model.count_params()}')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
        )

        preds = model.predict(X_test, verbose=0)
        preds_inv_log = scaler.inverse_transform(preds)
        y_test_inv_log = scaler.inverse_transform(y_test.reshape(-1, 1))

        preds_final = np.expm1(preds_inv_log) if use_log else preds_inv_log
        y_test_final = np.expm1(y_test_inv_log) if use_log else y_test_inv_log

        y_true_values.append(y_test_final)
        y_pred_values.append(preds_final)

    return _metrics_from_predictions(y_true_values, y_pred_values)


def _lag_features(series: pd.Series, lags: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame({'y': series})
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df = df.dropna()
    X = df.drop(columns=['y'])
    y = df['y']
    return X, y


def evaluate_xgboost(
    series: pd.Series,
    lags: int = 12,
    test_size: int = 1,
    folds: int = 5,
    n_estimators: int = 400,
    max_depth: int = 3,
    use_log: bool = False,
    print_params: bool = False,
    label: str | None = None,
) -> dict:
    """Fit a lag-feature XGBoost regressor over chronological folds.

    The lag count is reduced for short series so that chronological folds can
    still be built. Metrics are computed once across all fold predictions.
    The model uses a fixed number of estimators to avoid using test folds for
    early stopping.
    Raises ValueError when even one lag leaves too little data.
    """
    from xgboost import XGBRegressor

    min_train_size = 2
    min_supervised_rows = min_train_size + test_size
    if len(series) <= min_supervised_rows + 1:
        return empty_metrics()

    max_foldable_lags = len(series) - min_supervised_rows
    if max_foldable_lags < 1:
        return empty_metrics()
    max_requested_lags = len(series) - (min_train_size + test_size + folds - 1)
    if max_requested_lags >= 1 and lags > max_requested_lags:
        lags = max_requested_lags
    elif lags > max_foldable_lags:
        lags = max_foldable_lags

    series_use = np.log1p(series) if use_log else series

    X, y = _lag_features(series_use, lags)
    y_true_values = []
    y_pred_values = []
    split_indices = _time_series_folds(len(X), folds, min_train_size, test_size)

    for fold_idx, (train_idx, test_idx) in enumerate(split_indices, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse',
        )
        if print_params and fold_idx == 1:
            name = label or 'XGBoost'
            print(f'\n{name} parameters:')
            print(f'  lags: {lags}')
            print(f'  folds requested: {folds}')
            print(f'  folds used: {len(split_indices)}')
            print(f'  test_size: {test_size}')
            print(f'  params: {model.get_params()}')
        model.fit(X_train, y_train, verbose=False)

        preds_log = model.predict(X_test)

        y_true = np.expm1(y_test) if use_log else y_test
        y_pred = np.expm1(preds_log) if use_log else preds_log
        y_true_values.append(y_true)
        y_pred_values.append(y_pred)

    return _metrics_from_predictions(y_true_values, y_pred_values)


def load_azerbaijan_sales(path: str) -> pd.DataFrame:
    """Load the IMS Azerbaijan report and reshape monthly unit/value columns.

    The workbook is a formatted report, not a flat table. Rows 3 and 4 contain
    the month/metric headers, row 5 is a total-market summary, and row 6 onward
    contains product rows. The returned data has one row per product-month with
    Sale Quantity holding monthly units and Sale Value USD holding monthly value.
    """
    raw = pd.read_excel(path, sheet_name='Report', header=None)
    if raw.shape[0] < 6 or raw.shape[1] < 12:
        raise ValueError(f'{path}: IMS report is too small to contain product-month data')

    header_row = raw.iloc[2]
    metric_row = raw.iloc[3]
    data = raw.iloc[5:].copy()

    product_no = pd.to_numeric(data.iloc[:, 0], errors='coerce')
    data = data[product_no.notna()].copy()
    if data.empty:
        raise ValueError(f'{path}: no product rows found after removing report header/summary rows')

    monthly_columns = {}
    for col_idx in range(10, raw.shape[1]):
        period_label = header_row.iloc[col_idx]
        metric_label = metric_row.iloc[col_idx]
        if not isinstance(period_label, str):
            continue

        period = pd.to_datetime(period_label.split()[0], format='%Y/%m', errors='coerce')
        if pd.isna(period):
            continue

        if metric_label == 'Sum Units':
            monthly_columns.setdefault(period, {})['quantity'] = col_idx
        elif metric_label == 'Sum TRD Price in USD':
            monthly_columns.setdefault(period, {})['value'] = col_idx

    incomplete = [period.strftime('%Y-%m') for period, cols in monthly_columns.items() if {'quantity', 'value'} - set(cols)]
    if incomplete:
        raise ValueError(f'{path}: monthly IMS columns are missing units or value for {incomplete}')
    if not monthly_columns:
        raise ValueError(f'{path}: no monthly IMS unit/value columns found')

    base = pd.DataFrame(
        {
            'Drug ID': 'az_' + pd.to_numeric(data.iloc[:, 0], errors='raise').astype('int64').astype(str),
            'ATC3': data.iloc[:, 1],
            'ATC4': data.iloc[:, 2],
            'Molecule': data.iloc[:, 3],
            'Trade Name': data.iloc[:, 4],
            'Drug Form Description': data.iloc[:, 5],
            'Dosage': data.iloc[:, 6],
            'Drug Form': data.iloc[:, 7],
            'Pack Size': data.iloc[:, 8],
            'Corporation': data.iloc[:, 9],
        },
        index=data.index,
    )

    long_frames = []
    for period, cols in sorted(monthly_columns.items()):
        frame = base.copy()
        frame['Date of Sale'] = period
        frame['Year'] = period.year
        frame['Month'] = period.month
        frame['Season of the year'] = ((period.month - 1) // 3) + 1
        frame['Sale Quantity'] = pd.to_numeric(data.iloc[:, cols['quantity']], errors='coerce')
        frame['Sale Value USD'] = pd.to_numeric(data.iloc[:, cols['value']], errors='coerce')
        long_frames.append(frame)

    aze = pd.concat(long_frames, ignore_index=True)
    if aze[['Sale Quantity', 'Sale Value USD']].isna().any().any():
        raise ValueError(f'{path}: found non-numeric monthly quantity or value cells')

    return aze


def report_linear_correlations(df: pd.DataFrame, dataset_name: str, target: str = 'Sale Quantity') -> None:
    """Print Pearson correlations between numeric features and the target.

    Raises ValueError when the target or numeric predictors are unavailable.
    """
    numeric = df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        raise ValueError(f'{dataset_name}: target {target} not numeric or missing')
    corr = numeric.corr()[target].drop(labels=[target], errors='ignore').dropna()
    if corr.empty:
        raise ValueError(f'{dataset_name}: no numeric predictors to correlate with {target}')
    print(f'\n{dataset_name} – Pearson correlation with {target}:')
    for name, val in corr.sort_values(key=lambda x: x.abs(), ascending=False).items():
        print(f'  {name}: {val:.3f}')


def report_series_quality(df: pd.DataFrame, dataset_name: str) -> None:
    """Print data quality and volatility checks for monthly totals.

    Raises ValueError when Sale Quantity or Date of Sale is missing, invalid, or
    cannot form a monthly aggregate.
    """
    required = {'Sale Quantity', 'Date of Sale'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{dataset_name}: missing required columns for quality report: {sorted(missing)}')

    print(f'\n{dataset_name} – data quality checks:')
    print(f"  missing Sale Quantity: {df['Sale Quantity'].isna().sum()}")
    print(f"  negative Sale Quantity: {(df['Sale Quantity'] < 0).sum()}")
    print(f"  zero Sale Quantity: {(df['Sale Quantity'] == 0).sum()}")

    date_series = pd.to_datetime(df['Date of Sale'], errors='coerce')
    if date_series.isna().any():
        raise ValueError(f'{dataset_name}: found {date_series.isna().sum()} invalid Date of Sale values')

    print(f'  missing Date of Sale: {date_series.isna().sum()}')
    print(f'  date range: {date_series.min()} -> {date_series.max()}')
    monthly = (
        df.assign(date=date_series)
        .groupby(pd.Grouper(key='date', freq='MS'))['Sale Quantity']
        .sum()
        .sort_index()
    )
    if monthly.empty:
        raise ValueError(f'{dataset_name}: monthly aggregation is empty after parsing dates')

    full_range = pd.date_range(monthly.index.min(), monthly.index.max(), freq='MS')
    missing_months = full_range.difference(monthly.index)
    print(f'  monthly points: {len(monthly)}')
    print(f'  missing months: {len(missing_months)}')
    print(f'  months with zero total: {(monthly == 0).sum()}')
    print(f'  monthly mean: {monthly.mean():.3f}')
    print(f'  monthly median: {monthly.median():.3f}')
    print(f'  monthly std: {monthly.std():.3f}')
    print(f'  monthly min: {monthly.min():.3f}')
    print(f'  monthly max: {monthly.max():.3f}')
    min_month = monthly.idxmin()
    print(f'  month with min total: {min_month.strftime("%Y-%m")}')
    if monthly.mean() != 0:
        print(f'  coef of variation: {monthly.std() / monthly.mean():.3f}')


def main() -> None:
    """Run the sales plotting, forecasting, and diagnostic reports."""
    turkish = pd.read_excel(r'C:\Users\HP\Desktop\Thesis\Code\Drug_Sale_Data.xlsx')
    ger_pl = pd.read_excel(r'C:\Users\HP\Desktop\Thesis\Code\Medicine.xlsx')
    aze = load_azerbaijan_sales(r'C:\Users\HP\Desktop\Thesis\Code\ims 2021.xlsx')

    ger_pl['Drug ID'] = 'gp_' + ger_pl['Drug ID'].astype(str)
    turkish['Drug ID'] = 'tr_' + turkish['Drug ID'].astype(str)

    turkish['Date of Sale'] = pd.to_datetime(turkish['Date of Sale'], format='%m/%d/%Y')
    turkish['Year'] = turkish['Date of Sale'].dt.year
    turkish['Season of the year'] = turkish['Season of the year'].astype(int)

    ger_pl['Year'] = ger_pl['Year'].astype(int)
    ger_pl['Month'] = ger_pl['Month'].astype(int)
    ger_pl['Season of the year'] = ger_pl['Season of the year'].astype(int)

    ger_pl = sanitize_sales(ger_pl, 'German sales')
    turkish = sanitize_sales(turkish, 'Turkish sales')
    aze = sanitize_sales(aze, 'Azerbaijan sales')

    plot_seasonal_boxplots(ger_pl, 'German sales', save_path='german_sales.png')
    plot_seasonal_boxplots(turkish, 'Turkish sales', save_path='turkish_sales.png')
    plot_seasonal_boxplots(aze, 'Azerbaijan sales', save_path='azerbaijan_sales.png')
    plt.show()

    turkish_monthly = (
        turkish.groupby(pd.Grouper(key='Date of Sale', freq='MS'))['Sale Quantity']
        .sum()
        .sort_index()
    )
    ger_monthly = (
        ger_pl.assign(date=pd.to_datetime(dict(year=ger_pl['Year'], month=ger_pl['Month'], day=1)))
        .groupby('date')['Sale Quantity']
        .sum()
        .sort_index()
    )
    aze_monthly = (
        aze.groupby(pd.Grouper(key='Date of Sale', freq='MS'))['Sale Quantity']
        .sum()
        .sort_index()
    )
    cv_folds = 5
    arima_orders = [(2, 1, 1), (1, 1, 2)]
    lstm_units = [16, 64]
    xgboost_max_depths = [2, 6]

    monthly_series = {
        'German': ger_monthly,
        'Turkish': turkish_monthly,
        'Azerbaijan': aze_monthly,
    }

    results = {}
    for region, monthly in monthly_series.items():
        for order in arima_orders:
            label = f'{region} ARIMA order {order} (log)'
            results[label] = evaluate_arima(
                monthly,
                order=order,
                folds=cv_folds,
                use_log=True,
                print_params=True,
                label=label,
            )

        for units in lstm_units:
            label = f'{region} LSTM {units} units (log)'
            results[label] = evaluate_lstm(
                monthly,
                units=units,
                folds=cv_folds,
                use_log=True,
                print_params=True,
                label=label,
            )

        for max_depth in xgboost_max_depths:
            label = f'{region} XGBoost max_depth {max_depth} (log)'
            results[label] = evaluate_xgboost(
                monthly,
                max_depth=max_depth,
                folds=cv_folds,
                use_log=True,
                print_params=True,
                label=label,
            )

    print('Forecasting metrics with approximate 95% bootstrap CIs:')
    for k, v in results.items():
        print(
            f"{k} "
            f"n: {v['n_predictions']} | "
            f"RMSE: {_format_metric(v['rmse'], v['rmse_ci'])} | "
            f"MAE: {_format_metric(v['mae'], v['mae_ci'])} | "
            f"MAPE: {_format_metric(v['mape'], v['mape_ci'], suffix='%', decimals=2)} | "
            f"sMAPE: {_format_metric(v['smape'], v['smape_ci'], suffix='%', decimals=2)}"
        )

    mean_by_region = {
        'German': ger_monthly.mean(),
        'Turkish': turkish_monthly.mean(),
        'Azerbaijan': aze_monthly.mean(),
    }

    print('\nRelative RMSE vs mean monthly sales (with approximate 95% bootstrap CIs):')
    for k, v in results.items():
        region = k.split()[0]
        mean_val = mean_by_region.get(region, 0)
        if not mean_val:
            raise ValueError(f'{region}: mean monthly sales is zero, so relative RMSE cannot be computed')
        rel_rmse = v['rmse'] / mean_val
        rel_rmse_ci = (v['rmse_ci'][0] / mean_val, v['rmse_ci'][1] / mean_val)
        print(
            f"{k} "
            f"RMSE/mean: {_format_metric(rel_rmse, rel_rmse_ci)} | "
            f"sMAPE: {_format_metric(v['smape'], v['smape_ci'], suffix='%', decimals=2)}"
        )

    report_linear_correlations(ger_pl, 'German sales (raw rows)')
    report_linear_correlations(turkish, 'Turkish sales (raw rows)')
    report_linear_correlations(aze, 'Azerbaijan sales (product-month rows)')

    report_series_quality(turkish, 'Turkish sales (raw rows)')
    report_series_quality(aze, 'Azerbaijan sales (product-month rows)')

    print('\nGerman monthly target summary:')
    print(ger_monthly.describe())
    print('\nAzerbaijan monthly target summary:')
    print(aze_monthly.describe())

    german_forecast_intervals = arima_forecast_intervals(ger_monthly, steps=12, use_log=True)
    print('\nGerman ARIMA 12-month forecast with 95% confidence intervals:')
    print(
        german_forecast_intervals.to_string(
            index=False,
            formatters={
                'Forecast Date': lambda x: x.strftime('%Y-%m') if hasattr(x, 'strftime') else str(x),
                'Forecast': lambda x: f'{x:,.0f}',
                'Lower 95% CI': lambda x: f'{x:,.0f}',
                'Upper 95% CI': lambda x: f'{x:,.0f}',
            },
        )
    )


if __name__ == '__main__':
    main()
