"""
Модуль технических индикаторов.
Реализация популярных технических индикаторов для анализа акций.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


class TechnicalIndicators:
    """
    Класс для расчёта технических индикаторов.
    """
    
    def __init__(self, logger):
        """
        Инициализация класса технических индикаторов.
        
        Args:
            logger: Объект логгера
        """
        self.logger = logger
    
    def sma(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Simple Moving Average - простая скользящая средняя.
        
        Args:
            data (pd.Series): Временной ряд цен
            period (int): Период для расчёта
        
        Returns:
            pd.Series: SMA
        """
        return data.rolling(window=period).mean()
    
    def ema(self, data: pd.Series, period: int = 20) -> pd.Series:
        """
        Exponential Moving Average - экспоненциальная скользящая средняя.
        
        Args:
            data (pd.Series): Временной ряд цен
            period (int): Период для расчёта
        
        Returns:
            pd.Series: EMA
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index - индекс относительной силы.
        
        Args:
            data (pd.Series): Временной ряд цен
            period (int): Период для расчёта
        
        Returns:
            pd.Series: RSI (0-100)
        """
        # Вычисляем изменения
        delta = data.diff()
        
        # Разделяем на прибыли и убытки
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Средние прибыли и убытки
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # RS и RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD - схождение/расхождение скользящих средних.
        
        Args:
            data (pd.Series): Временной ряд цен
            fast (int): Период быстрой EMA
            slow (int): Период медленной EMA
            signal (int): Период сигнальной линии
        
        Returns:
            Dict[str, pd.Series]: Словарь с MACD, сигнальной линией и гистограммой
        """
        # EMA
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        
        # MACD линия
        macd_line = ema_fast - ema_slow
        
        # Сигнальная линия
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Гистограмма
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """
        Bollinger Bands - полосы Боллинджера.
        
        Args:
            data (pd.Series): Временной ряд цен
            period (int): Период для расчёта
            std_dev (int): Количество стандартных отклонений
        
        Returns:
            Dict[str, pd.Series]: Словарь с верхней, средней и нижней полосами
        """
        # Средняя линия (SMA)
        middle_band = self.sma(data, period)
        
        # Стандартное отклонение
        std = data.rolling(window=period).std()
        
        # Верхняя и нижняя полосы
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator - стохастический осциллятор.
        
        Args:
            high (pd.Series): Максимальные цены
            low (pd.Series): Минимальные цены
            close (pd.Series): Цены закрытия
            period (int): Период для расчёта
        
        Returns:
            Tuple[pd.Series, pd.Series]: %K и %D линии
        """
        # Минимум и максимум за период
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        # %K линия
        k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # %D линия (SMA от %K)
        d_line = k_line.rolling(window=3).mean()
        
        return k_line, d_line
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range - средний истинный диапазон (мера волатильности).
        
        Args:
            high (pd.Series): Максимальные цены
            low (pd.Series): Минимальные цены
            close (pd.Series): Цены закрытия
            period (int): Период для расчёта
        
        Returns:
            pd.Series: ATR
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume - балансовый объём.
        
        Args:
            close (pd.Series): Цены закрытия
            volume (pd.Series): Объёмы торгов
        
        Returns:
            pd.Series: OBV
        """
        # Направление движения цены
        direction = np.sign(close.diff())
        
        # OBV
        obv = (direction * volume).cumsum()
        
        return obv
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index - индекс товарного канала.
        
        Args:
            high (pd.Series): Максимальные цены
            low (pd.Series): Минимальные цены
            close (pd.Series): Цены закрытия
            period (int): Период для расчёта
        
        Returns:
            pd.Series: CCI
        """
        # Типичная цена
        typical_price = (high + low + close) / 3
        
        # SMA типичной цены
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Среднее отклонение
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        # CCI
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci









