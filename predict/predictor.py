"""
Модуль главного предсказателя.
Координирует различные модели прогнозирования.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .linear_model import LinearModel
from .time_series_model import TimeSeriesModel


class Predictor:
    """
    Главный класс прогнозирования.
    Управляет различными моделями и выбирает лучшую.
    """
    
    def __init__(self, config, logger):
        """
        Инициализация предсказателя.
        
        Args:
            config: Объект конфигурации системы
            logger: Объект логгера
        """
        self.config = config
        self.logger = logger
        
        # Инициализация моделей
        self.linear_model = LinearModel(logger)
        self.ts_model = TimeSeriesModel(logger)
        
        self.logger.info("Predictor инициализирован")
    
    def predict(self, data: pd.DataFrame, horizon: int = None) -> Dict[str, Any]:
        """
        Прогнозирование цен на основе исторических данных.
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            horizon (int): Горизонт прогнозирования в днях
        
        Returns:
            Dict[str, Any]: Результаты прогнозирования
        """
        if horizon is None:
            horizon = self.config.get("predict.prediction_horizon", 30)
        
        self.logger.info(f"Запуск прогнозирования на {horizon} дней")
        
        if data.empty or len(data) < 30:
            self.logger.warning("Недостаточно данных для прогнозирования")
            return {}
        
        results = {}
        
        try:
            # Линейная регрессия
            linear_pred = self.linear_model.predict(data, horizon)
            results['linear_regression'] = linear_pred
            
            # Модель временных рядов
            ts_pred = self.ts_model.predict(data, horizon)
            results['time_series'] = ts_pred
            
            # Ансамбль (средневзвешенное)
            ensemble_pred = self._ensemble_prediction(linear_pred, ts_pred)
            results['ensemble'] = ensemble_pred
            
            # Рекомендация
            recommendation = self._generate_recommendation(data, results)
            results['recommendation'] = recommendation
            
            self.logger.info("Прогнозирование завершено успешно")
            return results
        
        except Exception as e:
            self.logger.exception("Ошибка при прогнозировании")
            raise
    
    def _ensemble_prediction(self, linear_pred: Dict[str, Any], ts_pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ансамблевое прогнозирование (комбинирование моделей).
        
        Args:
            linear_pred (Dict[str, Any]): Прогноз линейной модели
            ts_pred (Dict[str, Any]): Прогноз модели временных рядов
        
        Returns:
            Dict[str, Any]: Ансамблевый прогноз
        """
        self.logger.info("Создание ансамблевого прогноза")
        
        # Веса для каждой модели (можно настраивать)
        weights = {
            'linear': 0.4,
            'time_series': 0.6
        }
        
        # Комбинируем прогнозы
        if 'predicted_price' in linear_pred and 'predicted_price' in ts_pred:
            ensemble_price = (
                weights['linear'] * linear_pred['predicted_price'] +
                weights['time_series'] * ts_pred['predicted_price']
            )
        else:
            ensemble_price = None
        
        # Комбинируем направление
        linear_direction = linear_pred.get('direction', 'neutral')
        ts_direction = ts_pred.get('direction', 'neutral')
        
        if linear_direction == ts_direction:
            ensemble_direction = linear_direction
            confidence = 'high'
        else:
            # Выбираем направление модели с большим весом
            ensemble_direction = ts_direction
            confidence = 'low'
        
        result = {
            'predicted_price': ensemble_price,
            'direction': ensemble_direction,
            'confidence': confidence,
            'weights': weights
        }
        
        return result
    
    def _generate_recommendation(self, data: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерация инвестиционной рекомендации.
        
        Args:
            data (pd.DataFrame): Исторические данные
            predictions (Dict[str, Any]): Результаты прогнозирования
        
        Returns:
            Dict[str, Any]: Рекомендация
        """
        self.logger.info("Генерация рекомендации")
        
        current_price = float(data['close'].iloc[-1])
        ensemble = predictions.get('ensemble', {})
        predicted_price = ensemble.get('predicted_price')
        
        if predicted_price is None:
            return {
                'action': 'hold',
                'reason': 'Недостаточно данных для прогноза',
                'confidence': 'low'
            }
        
        # Расчёт ожидаемого изменения
        expected_change = (predicted_price - current_price) / current_price * 100
        
        # Определение действия
        if expected_change > 5:
            action = 'buy'
            reason = f'Ожидается рост на {expected_change:.1f}%'
        elif expected_change < -5:
            action = 'sell'
            reason = f'Ожидается падение на {abs(expected_change):.1f}%'
        else:
            action = 'hold'
            reason = f'Ожидается изменение на {expected_change:.1f}% (незначительное)'
        
        # Уровень уверенности
        confidence = ensemble.get('confidence', 'medium')
        
        recommendation = {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'expected_change_percent': expected_change
        }
        
        self.logger.info(f"Рекомендация: {action} (изменение {expected_change:.1f}%)")
        return recommendation
    
    def backtest(self, data: pd.DataFrame, test_period: int = 30) -> Dict[str, Any]:
        """
        Бэктестинг моделей на исторических данных.
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            test_period (int): Период для тестирования в днях
        
        Returns:
            Dict[str, Any]: Результаты бэктестинга
        """
        self.logger.info(f"Запуск бэктестинга на {test_period} днях")
        
        if len(data) < test_period + 30:
            self.logger.warning("Недостаточно данных для бэктестинга")
            return {}
        
        # Разделяем данные на обучающую и тестовую выборки
        train_data = data.iloc[:-test_period].copy()
        test_data = data.iloc[-test_period:].copy()
        
        # Прогнозируем
        predictions = self.predict(train_data, horizon=test_period)
        
        # Оцениваем точность
        actual_prices = test_data['close'].values
        predicted_price = predictions.get('ensemble', {}).get('predicted_price')
        
        if predicted_price is None:
            return {}
        
        # Метрики точности
        mae = abs(predicted_price - actual_prices[-1])
        mape = (mae / actual_prices[-1]) * 100
        
        results = {
            'test_period': test_period,
            'actual_final_price': float(actual_prices[-1]),
            'predicted_final_price': predicted_price,
            'mae': float(mae),
            'mape': float(mape),
            'accuracy': float(100 - mape)
        }
        
        self.logger.info(f"Бэктестинг завершён. Точность: {results['accuracy']:.1f}%")
        return results









