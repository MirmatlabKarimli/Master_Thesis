import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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
    """Compute MAPE, ignoring zero targets to avoid division blowups."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return float('nan')
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)


def smape(y_true, y_pred) -> float:
    """Compute sMAPE for more stability when values are near zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    non_zero = denom != 0
    if not np.any(non_zero):
        return float('nan')
    return float(np.mean(2.0 * np.abs(y_pred[non_zero] - y_true[non_zero]) / denom[non_zero]) * 100.0)
def plot_seasonal_boxplots(df: pd.DataFrame, dataset_name: str, save_path: str | None = None) -> None:
    # Sum quantity per drug per season/year, then plot grouped boxplots
    agg = (
        df.groupby(['Year', 'Season of the year', 'Drug ID'])['Sale Quantity']
        .sum()
        .reset_index()
    )

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
    neg_count = (df['Sale Quantity'] < 0).sum()
    if neg_count:
        print(f'{dataset_name}: dropping {neg_count} rows with negative Sale Quantity')
    return df[df['Sale Quantity'] >= 0].copy()


def _train_test_split_series(series: pd.Series, test_size: float = 0.2):
    n_test = max(1, int(len(series) * test_size))
    return series.iloc[:-n_test], series.iloc[-n_test:]


def evaluate_arima(
    series: pd.Series,
    order=(1, 1, 1),
    test_size: float = 0.2,
    use_log: bool = False,
    print_params: bool = False,
    label: str | None = None,
) -> dict:
    """Fit ARIMA on the training portion and return metrics on the holdout."""
    from statsmodels.tsa.arima.model import ARIMA

    if len(series) < sum(order) + 1:
        return {'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan'), 'smape': float('nan')}

    series_use = np.log1p(series) if use_log else series

    train, test = _train_test_split_series(series_use, test_size)
    model = ARIMA(train, order=order).fit()
    if print_params:
        name = label or 'ARIMA'
        params = model.params.to_dict() if hasattr(model.params, 'to_dict') else model.params
        print(f'\n{name} parameters:')
        print(f'  order: {order}')
        print(f'  params: {params}')
    forecast = model.forecast(steps=len(test))

    y_true = np.expm1(test) if use_log else test
    y_pred = np.expm1(forecast) if use_log else forecast
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }


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
    test_size: float = 0.2,
    epochs: int = 30,
    use_log: bool = False,
    folds: int = 5,
    print_params: bool = False,
    label: str | None = None,
) -> dict:
    """Fit a simple LSTM and return mean metrics across 3/1/1 time splits."""
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    if len(series) <= look_back + 1:
        return {'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan'), 'smape': float('nan')}

    series_use = np.log1p(series) if use_log else series

    values = series_use.values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = _lstm_sequences(scaled, look_back)
    total = X.shape[0]
    if total < 5:
        return {'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan'), 'smape': float('nan')}

    effective_blocks = min(folds, total)
    if effective_blocks < 5:
        # Fallback to a single 60/20/20 split when there isn't enough data for 5 blocks.
        train_end = max(1, int(total * 0.6))
        val_end = max(train_end + 1, int(total * 0.8))
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential(
            [
                LSTM(32, input_shape=(look_back, 1)),
                Dropout(0.2),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        if print_params:
            name = label or 'LSTM'
            print(f'\n{name} parameters:')
            print('  units: 32')
            print('  dropout: 0.2')
            print(f'  look_back: {look_back}')
            print(f'  epochs: {epochs}')
            print('  batch_size: 32')
            print('  learning_rate: 0.001')
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

        return {
            'rmse': rmse(y_test_final, preds_final),
            'mae': mae(y_test_final, preds_final),
            'mape': mape(y_test_final, preds_final),
            'smape': smape(y_test_final, preds_final),
        }

    rmse_scores = []
    mae_scores = []
    mape_scores = []
    smape_scores = []
    blocks = np.array_split(np.arange(total), effective_blocks)

    # Use contiguous 3/1/1 block splits; with 5 blocks there is one split.
    for block_start in range(0, len(blocks) - 4):
        train_idx = np.concatenate(blocks[block_start : block_start + 3])
        val_idx = blocks[block_start + 3]
        test_idx = blocks[block_start + 4]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential(
            [
                LSTM(32, input_shape=(look_back, 1)),
                Dropout(0.2),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        if print_params:
            name = label or 'LSTM'
            print(f'\n{name} parameters:')
            print('  units: 32')
            print('  dropout: 0.2')
            print(f'  look_back: {look_back}')
            print(f'  epochs: {epochs}')
            print('  batch_size: 32')
            print('  learning_rate: 0.001')
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

        rmse_scores.append(rmse(y_test_final, preds_final))
        mae_scores.append(mae(y_test_final, preds_final))
        mape_scores.append(mape(y_test_final, preds_final))
        smape_scores.append(smape(y_test_final, preds_final))

    if not rmse_scores:
        return {'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan'), 'smape': float('nan')}

    return {
        'rmse': float(np.mean(rmse_scores)),
        'mae': float(np.mean(mae_scores)),
        'mape': float(np.mean(mape_scores)),
        'smape': float(np.mean(smape_scores)),
    }


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
    test_size: float = 0.2,
    n_estimators: int = 400,
    use_log: bool = False,
    print_params: bool = False,
    label: str | None = None,
) -> dict:
    """Fit a gradient boosted tree regressor on lag features and return metrics."""
    from xgboost import XGBRegressor

    if len(series) <= 2:
        return {'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan'), 'smape': float('nan')}

    if len(series) <= lags + 1:
        lags = max(1, len(series) - 2)

    series_use = np.log1p(series) if use_log else series

    X, y = _lag_features(series_use, lags)
    split_idx = len(X) - max(1, int(len(X) * test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse',
    )
    if print_params:
        name = label or 'XGBoost'
        print(f'\n{name} parameters:')
        print(f'  lags: {lags}')
        print(f'  params: {model.get_params()}')
    # Older xgboost versions may not support early_stopping_rounds in fit; fallback if needed
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=30,
            verbose=False,
        )
    except TypeError:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds_log = model.predict(X_test)

    y_true = np.expm1(y_test) if use_log else y_test
    y_pred = np.expm1(preds_log) if use_log else preds_log

    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }
    #TODO MAPE


def report_linear_correlations(df: pd.DataFrame, dataset_name: str, target: str = 'Sale Quantity') -> None:
    """Print Pearson correlations between numeric features and the target."""
    numeric = df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        print(f'{dataset_name}: target {target} not numeric or missing; skipping correlation')
        return
    corr = numeric.corr()[target].drop(labels=[target], errors='ignore').dropna()
    if corr.empty:
        print(f'{dataset_name}: no numeric predictors to correlate with {target}')
        return
    print(f'\n{dataset_name} – Pearson correlation with {target}:')
    for name, val in corr.sort_values(key=lambda x: x.abs(), ascending=False).items():
        print(f'  {name}: {val:.3f}')


def report_series_quality(df: pd.DataFrame, dataset_name: str) -> None:
    """Print data quality and volatility checks for monthly totals."""
    print(f'\n{dataset_name} – data quality checks:')
    print(f"  missing Sale Quantity: {df['Sale Quantity'].isna().sum()}")
    print(f"  negative Sale Quantity: {(df['Sale Quantity'] < 0).sum()}")
    print(f"  zero Sale Quantity: {(df['Sale Quantity'] == 0).sum()}")

    if 'Date of Sale' in df.columns:
        date_series = pd.to_datetime(df['Date of Sale'], errors='coerce')
        print(f'  missing Date of Sale: {date_series.isna().sum()}')
        print(f'  date range: {date_series.min()} -> {date_series.max()}')
        monthly = (
            df.assign(date=date_series)
            .groupby(pd.Grouper(key='date', freq='MS'))['Sale Quantity']
            .sum()
            .sort_index()
        )
        if not monthly.empty:
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
        else:
            print('  monthly aggregation is empty after parsing dates')

turkish = pd.read_excel('./Drug Sale Data.xlsx')
ger_pl = pd.read_excel('./Medicine.xlsx')

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

plot_seasonal_boxplots(ger_pl, 'German sales', save_path='german_sales.png')
plot_seasonal_boxplots(turkish, 'Turkish sales', save_path='turkish_sales.png')
plt.show()

# Build monthly series for forecastingturkish_monthly
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
# Evaluate forecasting models
results = {
    'German ARIMA (log)': evaluate_arima(ger_monthly, use_log=True, print_params=True, label='German ARIMA (log)'),
    'German LSTM (log)': evaluate_lstm(ger_monthly, use_log=True, print_params=True, label='German LSTM (log)'),
    'German XGBoost (log)': evaluate_xgboost(ger_monthly, use_log=True, print_params=True, label='German XGBoost (log)'),
    'Turkish ARIMA (log)': evaluate_arima(turkish_monthly, use_log=True, print_params=True, label='Turkish ARIMA (log)'),
    'Turkish LSTM (log)': evaluate_lstm(turkish_monthly, use_log=True, print_params=True, label='Turkish LSTM (log)'),
    'Turkish XGBoost (log)': evaluate_xgboost(turkish_monthly, use_log=True, print_params=True, label='Turkish XGBoost (log)'),
}

print('Forecasting metrics:')
for k, v in results.items():
    print(
        f"{k} RMSE: {v['rmse']:.3f} | MAE: {v['mae']:.3f} | "
        f"MAPE: {v['mape']:.2f}% | sMAPE: {v['smape']:.2f}%"
    )

# Relative scale of errors
ger_mean = ger_monthly.mean()
turkish_mean = turkish_monthly.mean()
mean_by_region = {'German': ger_mean, 'Turkish': turkish_mean}

print('\nRelative RMSE vs mean monthly sales (with sMAPE):')
for k, v in results.items():
    region = k.split()[0]
    mean_val = mean_by_region.get(region, 0)
    rel_rmse = v['rmse'] / mean_val if mean_val else float('nan')
    print(f"{k} RMSE/mean: {rel_rmse:.3f} | sMAPE: {v['smape']:.2f}%")

# Linearity checks via correlations
report_linear_correlations(ger_pl, 'German sales (raw rows)')
report_linear_correlations(turkish, 'Turkish sales (raw rows)')

# Data quality diagnostics for Turkish series
report_series_quality(turkish, 'Turkish sales (raw rows)')

print('\nGerman monthly target summary:')
print(ger_monthly.describe())
