# ETH-USD Multi-Timeframe ML Trading Strategies

A collection of machine learning-based trading strategies for Ethereum (ETH-USD) across multiple timeframes, using ensemble models with optimized stop-loss mechanisms.

## Overview

This project implements algorithmic trading strategies that leverage ensemble machine learning models to predict price movements and generate buy/sell signals. Each strategy is tailored for a specific timeframe, allowing traders to analyze and backtest across different trading horizons.

## Timeframes Covered

| Notebook | Timeframe | Description |
|----------|-----------|-------------|
| `1min_based_strategy.ipynb` | 1 Minute | High-frequency scalping strategy |
| `5MIN_based_strategy.ipynb` | 5 Minutes | Short-term scalping strategy |
| `15MIN_based_strategy_v2.ipynb` | 15 Minutes | Intraday swing strategy |
| `30MIN_based_strategy.ipynb` | 30 Minutes | Intraday position strategy |
| `1H_based_strategy.ipynb` | 1 Hour | Medium-term swing strategy |
| `4H_based_strategy.ipynb` | 4 Hours | Position trading strategy |
| `multi_model_predection.ipynb` | Multiple | Combined multi-model prediction framework |

## Features

- **Ensemble ML Models**: Utilizes top-performing models including:
  - LightGBM
  - Random Forest
  - AdaBoost
  - CNN (Convolutional Neural Networks)
  - LSTM (Long Short-Term Memory)

- **Technical Indicators** (31 features):
  - Trend: EMA, SMA (10, 20, 50), Parabolic SAR
  - Momentum: RSI, ROC, CCI, MACD (with signal & histogram)
  - Volatility: ATR, Bollinger Bands, Standard Deviation
  - Volume: CMF (Chaikin Money Flow), Volume lags
  - Price Action: Log returns, Lag features (1-5)
  - Higher Timeframe: Daily close, return, and log return

- **Risk Management**:
  - Configurable stop-loss percentages
  - Trading cost modeling (0.1% default)
  - Grid search optimization for buy/sell thresholds
  - Stop-loss priority over sell signals

- **Backtesting Framework**:
  - Balance-based position sizing
  - Detailed trade history tracking
  - Performance metrics and optimization

## Requirements

```
pandas
numpy
yfinance
joblib
tensorflow>=2.0
scikit-learn
ta-lib
```

## Usage

1. **Load Pre-trained Models**: Each notebook loads models from a `results/` folder containing:
   - `models_registry.json` - Model leaderboard and metadata
   - `features.json` - Feature configurations
   - `config.json` - Scaler statistics for normalization
   - `models/` - Saved model files (.pkl, .joblib, .keras)

2. **Fetch Live Data**: Uses `yfinance` to download the latest ETH-USD data

3. **Generate Predictions**: Ensemble averaging of top 3 models by forward accuracy

4. **Backtest with Optimization**:
   ```python
   final_balance, trades = run_strategy_with_stop_loss(
       predictions_df,
       buy_threshold=0.60,
       sell_threshold=0.20,
       stop_loss_pct=0.01,
       initial_balance=1000,
       trading_cost=0.001
   )
   ```

## Strategy Logic

1. **Entry**: Buy when probability of upward movement > buy_threshold
2. **Exit (Signal)**: Sell when probability < sell_threshold
3. **Exit (Stop Loss)**: Triggered when price drops below entry * (1 - stop_loss_pct)
4. **Priority**: Stop-loss always takes precedence over sell signals

## Grid Search Optimization

Each notebook includes parameter optimization:
- Buy thresholds: 0.50 to 0.80 (16 values)
- Sell thresholds: 0.20 to 0.49 (15 values)
- Stop losses: 1% to 5% (9 values)
- Total combinations tested: 2,160

## Disclaimer

This project is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consider your financial situation before trading.

## License

MIT License
