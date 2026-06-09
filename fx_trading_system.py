"""
COMPLETE FOREX TRADING SIGNAL GENERATION SYSTEM
================================================
All-in-one module for technical analysis, ML signals, backtesting, and risk metrics.
Production-ready code - deadline delivery version.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Compute all technical indicators required by the project"""
    
    @staticmethod
    def sma(series, period):
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series, period):
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series, period=14):
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        """MACD with signal line and histogram"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high, low, close, period=14, smooth=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=smooth).mean()
        return k_percent, d_percent
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index"""
        plus_dm = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        minus_dm = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([pd.Series(tr1), pd.Series(tr2), pd.Series(tr3)], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di


# ============================================================================
# PART 2: COMPOSITE SIGNAL ENGINE
# ============================================================================

class CompositeSignalEngine:
    """
    Weighted voting system combining multiple technical indicators.
    Each indicator votes on BUY (+1), SELL (-1), or HOLD (0).
    Final signal = weighted average of all votes.
    """
    
    def __init__(self, weights=None):
        """Initialize with weights (default = equal weight)"""
        self.weights = weights or {
            'ma_cross': 0.20,
            'rsi': 0.20,
            'macd': 0.20,
            'bollinger': 0.15,
            'stochastic': 0.15,
            'atr_trend': 0.10
        }
        self.ti = TechnicalIndicators()
    
    def generate_signals(self, df):
        """
        Generate composite signal for each candle.
        Returns: DataFrame with individual signal components and composite score.
        """
        df = df.copy()
        
        # ---- MA CROSSOVER SIGNAL ----
        df['SMA20'] = self.ti.sma(df['Close'], 20)
        df['SMA50'] = self.ti.sma(df['Close'], 50)
        df['ma_signal'] = np.where(df['SMA20'] > df['SMA50'], 1, 
                                   np.where(df['SMA20'] < df['SMA50'], -1, 0))
        
        # ---- RSI SIGNAL ----
        df['RSI'] = self.ti.rsi(df['Close'], 14)
        df['rsi_signal'] = np.where(df['RSI'] < 30, 1,           # Oversold -> BUY
                                    np.where(df['RSI'] > 70, -1,   # Overbought -> SELL
                                    0))
        
        # ---- MACD SIGNAL ----
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.ti.macd(df['Close'])
        df['macd_signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1,
                                     np.where(df['MACD'] < df['MACD_Signal'], -1, 0))
        
        # ---- BOLLINGER BANDS SIGNAL ----
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.ti.bollinger_bands(df['Close'], 20, 2)
        df['bollinger_signal'] = np.where(df['Close'] < df['BB_Lower'], 1,      # Price touches lower band -> BUY
                                          np.where(df['Close'] > df['BB_Upper'], -1,  # Price touches upper band -> SELL
                                          0))
        
        # ---- STOCHASTIC SIGNAL ----
        df['Stoch_K'], df['Stoch_D'] = self.ti.stochastic(df['High'], df['Low'], df['Close'], 14, 3)
        df['stochastic_signal'] = np.where(df['Stoch_K'] < 20, 1,
                                           np.where(df['Stoch_K'] > 80, -1, 0))
        
        # ---- ATR TREND SIGNAL ----
        df['ATR'] = self.ti.atr(df['High'], df['Low'], df['Close'], 14)
        df['atr_trend'] = np.where(df['Close'] > df['Close'].shift(20), 1,
                                   np.where(df['Close'] < df['Close'].shift(20), -1, 0))
        
        # ---- COMPOSITE SCORE ----
        signal_cols = ['ma_signal', 'rsi_signal', 'macd_signal', 'bollinger_signal', 'stochastic_signal', 'atr_trend']
        weights_list = [self.weights[col.replace('_signal', '').replace('_trend', '_trend')] for col in signal_cols]
        
        df['composite_score'] = (
            df['ma_signal'] * self.weights['ma_cross'] +
            df['rsi_signal'] * self.weights['rsi'] +
            df['macd_signal'] * self.weights['macd'] +
            df['bollinger_signal'] * self.weights['bollinger'] +
            df['stochastic_signal'] * self.weights['stochastic'] +
            df['atr_trend'] * self.weights['atr_trend']
        )
        
        # ---- FINAL SIGNAL ----
        df['signal'] = np.where(df['composite_score'] > 0.3, 'BUY',
                                np.where(df['composite_score'] < -0.3, 'SELL', 'HOLD'))
        df['signal_strength'] = abs(df['composite_score'])
        
        return df


# ============================================================================
# PART 3: MACHINE LEARNING SIGNALS
# ============================================================================

class MLSignalGenerator:
    """Machine Learning approach to signal generation"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.ti = TechnicalIndicators()
    
    def create_features(self, df):
        """Engineer features for ML model"""
        df = df.copy()
        
        # Price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_change'] = df['Close'] - df['Close'].shift(1)
        
        # Technical indicators
        df['RSI_14'] = self.ti.rsi(df['Close'], 14)
        df['RSI_7'] = self.ti.rsi(df['Close'], 7)
        df['SMA_20'] = self.ti.sma(df['Close'], 20)
        df['SMA_50'] = self.ti.sma(df['Close'], 50)
        df['EMA_12'] = self.ti.ema(df['Close'], 12)
        df['EMA_26'] = self.ti.ema(df['Close'], 26)
        
        # MACD
        macd, signal, hist = self.ti.macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = self.ti.bollinger_bands(df['Close'], 20, 2)
        df['BB_Position'] = (df['Close'] - lower) / (upper - lower)
        
        # ATR
        df['ATR'] = self.ti.atr(df['High'], df['Low'], df['Close'], 14)
        df['ATR_pct'] = df['ATR'] / df['Close'] * 100
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Momentum
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['ROC_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
        
        return df
    
    def create_target(self, df, lookahead=5):
        """Create target variable (next 5 candles direction)"""
        df = df.copy()
        future_return = df['Close'].shift(-lookahead) - df['Close']
        df['target'] = (future_return > 0).astype(int)  # 1=UP, 0=DOWN
        return df
    
    def train(self, df):
        """Train ML model"""
        df = self.create_features(df)
        df = self.create_target(df, lookahead=5)
        
        feature_cols = [col for col in df.columns if col not in 
                       ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'target']]
        
        # Remove NaN rows
        df_clean = df.dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        self.model.fit(X_scaled, y)
        print(f"✅ {self.model_type.upper()} model trained. Accuracy: {self.model.score(X_scaled, y):.3f}")
        
        return df_clean, feature_cols
    
    def predict(self, df, feature_cols):
        """Generate ML predictions"""
        df = self.create_features(df)
        
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        df['ml_probability'] = self.model.predict_proba(X_scaled)[:, 1]
        df['ml_signal'] = np.where(df['ml_probability'] > 0.6, 'BUY',
                                   np.where(df['ml_probability'] < 0.4, 'SELL', 'HOLD'))
        
        return df


# ============================================================================
# PART 4: BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """Complete backtesting with position management and P&L calculation"""
    
    def __init__(self, initial_balance=100000, position_size=100000, 
                 commission=0.0007, slippage=0.0002):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage
    
    def backtest(self, df, signal_column='signal'):
        """Run backtest with given signal column"""
        df = df.copy().dropna()
        
        df['position'] = 0  # -1=SHORT, 0=FLAT, 1=LONG
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['entry_time'] = pd.NaT
        df['exit_time'] = pd.NaT
        df['pnl'] = 0.0
        df['equity'] = self.initial_balance
        
        position = 0
        entry_price = 0
        entry_idx = 0
        
        for i in range(len(df)):
            signal = df.iloc[i][signal_column]
            close = df.iloc[i]['Close']
            datetime = df.index[i]
            
            # Exit logic
            if position != 0 and (signal == 'HOLD' or (position == 1 and signal == 'SELL') or (position == -1 and signal == 'BUY')):
                exit_price = close * (1 - self.slippage if position == 1 else 1 + self.slippage)
                pnl = (exit_price - entry_price) * self.position_size if position == 1 else (entry_price - exit_price) * self.position_size
                pnl -= self.commission * self.position_size
                
                df.loc[df.index[i], 'exit_price'] = exit_price
                df.loc[df.index[i], 'exit_time'] = datetime
                df.loc[df.index[i], 'pnl'] = pnl
                
                position = 0
            
            # Entry logic
            if position == 0 and signal != 'HOLD':
                entry_price = close * (1 + self.slippage if signal == 'BUY' else 1 - self.slippage)
                entry_idx = i
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'entry_time'] = datetime
                position = 1 if signal == 'BUY' else -1
            
            df.loc[df.index[i], 'position'] = position
            
            # Update equity
            if position != 0:
                mark_to_market = (close - entry_price) * self.position_size if position == 1 else (entry_price - close) * self.position_size
                current_equity = self.initial_balance + mark_to_market
            else:
                current_equity = self.initial_balance + df.loc[:df.index[i], 'pnl'].sum()
            
            df.loc[df.index[i], 'equity'] = current_equity
        
        return df
    
    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        trades = df[df['pnl'] != 0].copy()
        
        if len(trades) == 0:
            print("⚠️ No completed trades")
            return {}
        
        total_return = (df['equity'].iloc[-1] - self.initial_balance) / self.initial_balance
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['pnl'].sum()) / abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        
        drawdown = (df['equity'].cummax() - df['equity']) / df['equity'].cummax()
        max_drawdown = drawdown.max()
        
        returns = df['equity'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        metrics = {
            'Total Return': f"{total_return*100:.2f}%",
            'Win Rate': f"{win_rate*100:.2f}%",
            'Trades': len(trades),
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Avg Win': f"${avg_win:.2f}",
            'Avg Loss': f"${avg_loss:.2f}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Max Drawdown': f"{max_drawdown*100:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Final Equity': f"${df['equity'].iloc[-1]:,.2f}"
        }
        
        return metrics


# ============================================================================
# PART 5: RISK ANALYTICS
# ============================================================================

class RiskAnalytics:
    """Value at Risk, Monte Carlo, and risk metrics"""
    
    @staticmethod
    def calculate_var(returns, confidence=0.95):
        """Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns, confidence=0.95):
        """Conditional Value at Risk (Expected Shortfall)"""
        var = np.percentile(returns, (1 - confidence) * 100)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_metrics(equity_curve):
        """Comprehensive risk metrics"""
        returns = equity_curve.pct_change().dropna()
        
        metrics = {
            'Daily Return Mean': returns.mean(),
            'Daily Return Std': returns.std(),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),
            'VaR 95%': RiskAnalytics.calculate_var(returns, 0.95),
            'CVaR 95%': RiskAnalytics.calculate_cvar(returns, 0.95),
            'Max Daily Return': returns.max(),
            'Min Daily Return': returns.min(),
            'Best Day': f"${equity_curve.max():.2f}",
            'Worst Day': f"${equity_curve.min():.2f}"
        }
        
        return metrics


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(filepath):
    """Load and preprocess forex data"""
    df = pd.read_csv(filepath)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    return df

def prepare_powerbi_export(df):
    """Prepare data for Power BI import"""
    export_df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Spread',
                   'RSI', 'MACD', 'MACD_Signal', 'SMA20', 'SMA50',
                   'signal', 'signal_strength', 'composite_score']].copy()
    export_df['DateTime'] = export_df.index
    return export_df.reset_index(drop=True)

