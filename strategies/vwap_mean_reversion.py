"""
VWAP Mean Reversion Trading Strategy

This strategy trades when price deviates significantly from VWAP,
betting on mean reversion back to VWAP.

Entry Signals:
- Long: Price drops below VWAP by threshold% with volume confirmation
- Short: Price rises above VWAP by threshold% with volume confirmation

Exit Signals:
- Price returns to VWAP (mean reversion achieved)
- Take profit or stop loss hit
- Maximum holding period exceeded
"""

from typing import Optional, Dict, List
import logging
from datetime import datetime

from strategies.base_strategy import BaseStrategy
from analysis.technical_analysis import TechnicalAnalysis
from config.vwap_parameters import VWAPParameters, DEFAULT_VWAP_PARAMS

logger = logging.getLogger(__name__)


class VWAPMeanReversionStrategy(BaseStrategy):
    """VWAP Mean Reversion Strategy with dynamic parameters"""

    def __init__(self, config, binance_api, risk_manager, params: VWAPParameters = None):
        super().__init__(config, binance_api, risk_manager)
        self.params = params if params else DEFAULT_VWAP_PARAMS
        self.entry_times = {}  # Track entry times for max holding period

    @property
    def strategy_name(self) -> str:
        return "vwap_mean_reversion"

    @property
    def expected_win_rate(self) -> float:
        # Mean reversion strategies typically have 60-70% win rate
        return 0.65

    def update_parameters(self, params: VWAPParameters):
        """Update strategy parameters (called by weekly optimizer)"""
        if params.validate():
            self.params = params
            logger.info(f"✅ VWAP strategy parameters updated: {params.to_dict()}")
        else:
            logger.error("❌ Invalid parameters, keeping current parameters")

    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate VWAP mean reversion signal"""

        try:
            # Fetch klines data
            klines = self.api.get_klines(
                symbol,
                self.params.timeframe,
                self.params.lookback_candles
            )

            if len(klines) < self.params.vwap_period:
                logger.debug(f"Not enough data for {symbol}: {len(klines)} candles")
                return None

            # Extract OHLCV data
            opens = [float(k[1]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            current_price = closes[-1]

            # Calculate indicators
            vwap = TechnicalAnalysis.calculate_vwap(
                highs, lows, closes, volumes,
                period=self.params.vwap_period
            )

            upper_band, vwap_mid, lower_band = TechnicalAnalysis.calculate_vwap_bands(
                highs, lows, closes, volumes,
                period=self.params.vwap_period,
                std_multiplier=1.0
            )

            atr = TechnicalAnalysis.calculate_atr(
                highs, lows, closes,
                period=self.params.atr_period
            )

            # Calculate VWAP deviation
            vwap_deviation = TechnicalAnalysis.calculate_vwap_deviation(
                current_price, vwap
            )

            # Calculate volume metrics
            avg_volume = TechnicalAnalysis.calculate_sma(
                volumes[-20:], period=20
            )
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            # Volatility filter
            atr_percent = atr / current_price
            if atr_percent < self.params.min_atr_filter:
                logger.debug(f"{symbol}: ATR too low ({atr_percent:.4f}), skipping")
                return None

            if atr_percent > self.params.max_atr_filter:
                logger.debug(f"{symbol}: ATR too high ({atr_percent:.4f}), skipping")
                return None

            # Volume filter
            if volume_ratio < self.params.min_volume_multiplier:
                logger.debug(f"{symbol}: Volume too low ({volume_ratio:.2f}x), skipping")
                return None

            # Check for mean reversion opportunity
            signal = self._check_entry_conditions(
                symbol=symbol,
                current_price=current_price,
                vwap=vwap,
                vwap_deviation=vwap_deviation,
                upper_band=upper_band,
                lower_band=lower_band,
                atr=atr,
                volume_ratio=volume_ratio
            )

            if signal:
                logger.info(
                    f"🎯 {signal['action'].upper()} signal for {symbol}: "
                    f"Price={current_price:.2f}, VWAP={vwap:.2f}, "
                    f"Deviation={vwap_deviation*100:.2f}%, "
                    f"Volume={volume_ratio:.2f}x, ATR%={atr_percent*100:.2f}%"
                )

            return signal

        except Exception as e:
            logger.error(f"Error generating VWAP signal for {symbol}: {e}")
            return None

    def _check_entry_conditions(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
        vwap_deviation: float,
        upper_band: float,
        lower_band: float,
        atr: float,
        volume_ratio: float
    ) -> Optional[Dict]:
        """Check if entry conditions are met"""

        # Calculate confidence based on deviation magnitude and volume
        deviation_score = min(abs(vwap_deviation) / self.params.entry_threshold, 2.0)
        volume_score = min(volume_ratio / self.params.min_volume_multiplier, 2.0)
        confidence = self.params.base_confidence * (
            0.6 * deviation_score + 0.4 * volume_score
        ) / 2.0
        confidence = min(confidence, 0.95)  # Cap at 95%

        # LONG SIGNAL: Price below VWAP by threshold
        if vwap_deviation <= -self.params.entry_threshold:
            # Additional confirmation: price near or below lower band
            if current_price <= lower_band * 1.02:  # Within 2% of lower band

                entry_price = current_price
                stop_loss = entry_price - (self.params.stop_loss_atr_mult * atr)

                # Target is mean reversion back to VWAP
                take_profit = vwap + (self.params.profit_target_atr_mult * atr * 0.5)

                # Track entry time for max holding period
                self.entry_times[symbol] = datetime.now()

                return {
                    'action': 'buy',
                    'symbol': symbol,
                    'price': entry_price,
                    'entry_price': entry_price,
                    'stop_loss': max(stop_loss, 0),  # Ensure non-negative
                    'take_profit': take_profit,
                    'strategy': self.strategy_name,
                    'confidence': confidence,
                    'vwap': vwap,
                    'vwap_deviation': vwap_deviation,
                    'volume_ratio': volume_ratio,
                    'atr': atr,
                    'reason': f'Price {abs(vwap_deviation)*100:.2f}% below VWAP'
                }

        # SHORT SIGNAL: Price above VWAP by threshold
        elif vwap_deviation >= self.params.entry_threshold:
            # Additional confirmation: price near or above upper band
            if current_price >= upper_band * 0.98:  # Within 2% of upper band

                entry_price = current_price
                stop_loss = entry_price + (self.params.stop_loss_atr_mult * atr)

                # Target is mean reversion back to VWAP
                take_profit = vwap - (self.params.profit_target_atr_mult * atr * 0.5)

                # Track entry time
                self.entry_times[symbol] = datetime.now()

                return {
                    'action': 'sell',
                    'symbol': symbol,
                    'price': entry_price,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': max(take_profit, 0),  # Ensure non-negative
                    'strategy': self.strategy_name,
                    'confidence': confidence,
                    'vwap': vwap,
                    'vwap_deviation': vwap_deviation,
                    'volume_ratio': volume_ratio,
                    'atr': atr,
                    'reason': f'Price {abs(vwap_deviation)*100:.2f}% above VWAP'
                }

        return None

    def check_exit_conditions(self, symbol: str, entry_price: float,
                             action: str, current_price: float) -> Optional[str]:
        """
        Check if position should be exited based on mean reversion or time

        Returns: exit_reason string or None
        """
        try:
            # Get current VWAP
            klines = self.api.get_klines(symbol, self.params.timeframe, self.params.vwap_period + 10)
            if not klines:
                return None

            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            vwap = TechnicalAnalysis.calculate_vwap(
                highs, lows, closes, volumes,
                period=self.params.vwap_period
            )

            vwap_deviation = TechnicalAnalysis.calculate_vwap_deviation(current_price, vwap)

            # Exit if price has reverted to VWAP
            if abs(vwap_deviation) <= self.params.exit_threshold:
                return f"mean_reversion_complete"

            # Exit after max holding period
            if symbol in self.entry_times:
                entry_time = self.entry_times[symbol]
                holding_periods = (datetime.now() - entry_time).seconds / 60  # minutes
                if holding_periods >= self.params.max_holding_periods:
                    del self.entry_times[symbol]
                    return f"max_holding_period_exceeded"

            return None

        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}")
            return None

    def get_current_parameters(self) -> Dict:
        """Get current strategy parameters"""
        return self.params.to_dict()

    def get_parameter_version(self) -> str:
        """Get parameter version identifier for tracking"""
        # Use hash of parameters as version
        import hashlib
        param_str = self.params.to_json()
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
