"""
Модуль моделей временных рядов.
Использует методы анализа временных рядов для прогнозирования.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class TimeSeriesModel:
    """
    Класс для прогнозирования методами временных рядов.
    """
    
    def __init__(self, logger):
        """
        Инициализация модели временных рядов.
        
        Args:
            logger: Объект логгера
        """
        self.logger = logger
    
    def predict(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """
        Прогнозирование методом экспоненциального сглаживания.
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            horizon (int): Горизонт прогнозирования в днях
        
        Returns:
            Dict[str, Any]: Результаты прогнозирования
        """
        self.logger.info("Прогнозирование методом временных рядов")
        
        try:
            # Используем простое экспоненциальное сглаживание
            prices = data['close'].values
            
            # Параметр сглаживания (alpha)
            alpha = 0.3
            
            # Прогноз
            predicted_price = self._exponential_smoothing(prices, alpha, horizon)
            
            current_price = float(prices[-1])
            
            # Направление
            if predicted_price > current_price * 1.02:
                direction = 'up'
            elif predicted_price < current_price * 0.98:
                direction = 'down'
            else:
                direction = 'sideways'
            
            result = {
                'predicted_price': float(predicted_price),
                'current_price': current_price,
                'direction': direction,
                'method': 'exponential_smoothing',
                'alpha': alpha,
                'change_percent': float((predicted_price - current_price) / current_price * 100)
            }
            
            self.logger.info(f"Прогноз временных рядов: {predicted_price:.2f} ({direction})")
            return result
        
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании: {str(e)}")
            return {}
    
    def _exponential_smoothing(self, data: np.ndarray, alpha: float, horizon: int) -> float:
        """
        Простое экспоненциальное сглаживание.
        
        Args:
            data (np.ndarray): Массив исторических значений
            alpha (float): Параметр сглаживания (0-1)
            horizon (int): Горизонт прогнозирования
        
        Returns:
            float: Прогнозное значение
        """
        # Инициализация
        smoothed = data[0]
        
        # Сглаживание
        for value in data[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Прогноз на horizon шагов (в простом ES прогноз константный)
        return smoothed
    
    def predict_with_trend(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """
        Прогнозирование с учётом тренда (метод Холта).
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            horizon (int): Горизонт прогнозирования в днях
        
        Returns:
            Dict[str, Any]: Результаты прогнозирования
        """
        self.logger.info("Прогнозирование методом Холта (с трендом)")
        
        try:
            prices = data['close'].values
            
            # Параметры
            alpha = 0.3  # сглаживание уровня
            beta = 0.1   # сглаживание тренда
            
            # Инициализация
            level = prices[0]
            trend = prices[1] - prices[0]
            
            # Обновление уровня и тренда
            for value in prices[1:]:
                last_level = level
                level = alpha * value + (1 - alpha) * (level + trend)
                trend = beta * (level - last_level) + (1 - beta) * trend
            
            # Прогноз
            predicted_price = level + horizon * trend
            
            current_price = float(prices[-1])
            
            result = {
                'predicted_price': float(predicted_price),
                'current_price': current_price,
                'method': 'holt',
                'level': float(level),
                'trend': float(trend),
                'change_percent': float((predicted_price - current_price) / current_price * 100)
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании с трендом: {str(e)}")
            return {}
    
    def predict_moving_average(self, data: pd.DataFrame, window: int = 10, horizon: int = 30) -> Dict[str, Any]:
        """
        Прогнозирование на основе скользящего среднего.
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            window (int): Окно для скользящего среднего
            horizon (int): Горизонт прогнозирования (не используется, т.к. MA даёт константный прогноз)
        
        Returns:
            Dict[str, Any]: Результаты прогнозирования
        """
        self.logger.info(f"Прогнозирование на основе MA({window})")
        
        try:
            prices = data['close'].values
            
            # Скользящее среднее за последние window значений
            if len(prices) < window:
                window = len(prices)
            
            ma_value = np.mean(prices[-window:])
            
            current_price = float(prices[-1])
            
            result = {
                'predicted_price': float(ma_value),
                'current_price': current_price,
                'method': f'moving_average_{window}',
                'change_percent': float((ma_value - current_price) / current_price * 100)
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании MA: {str(e)}")
            return {}









