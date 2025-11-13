"""
Модуль линейных моделей прогнозирования.
Использует линейную регрессию для предсказания цен.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class LinearModel:
    """
    Класс для прогнозирования с помощью линейной регрессии.
    """
    
    def __init__(self, logger):
        """
        Инициализация линейной модели.
        
        Args:
            logger: Объект логгера
        """
        self.logger = logger
        self.model = None
        self.scaler = StandardScaler()
    
    def predict(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """
        Прогнозирование с помощью линейной регрессии.
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            horizon (int): Горизонт прогнозирования в днях
        
        Returns:
            Dict[str, Any]: Результаты прогнозирования
        """
        self.logger.info("Прогнозирование методом линейной регрессии")
        
        try:
            # Подготовка признаков
            X, y = self._prepare_features(data)
            
            if len(X) < 10:
                self.logger.warning("Недостаточно данных для обучения модели")
                return {}
            
            # Обучение модели
            self.model = LinearRegression()
            self.model.fit(X, y)
            
            # Прогноз на horizon дней вперёд
            last_index = len(data) - 1
            future_X = np.array([[last_index + horizon]])
            predicted_price = self.model.predict(future_X)[0]
            
            # Текущая цена
            current_price = float(data['close'].iloc[-1])
            
            # Направление тренда
            if predicted_price > current_price * 1.02:
                direction = 'up'
            elif predicted_price < current_price * 0.98:
                direction = 'down'
            else:
                direction = 'sideways'
            
            # Оценка качества модели
            score = self.model.score(X, y)
            
            result = {
                'predicted_price': float(predicted_price),
                'current_price': current_price,
                'direction': direction,
                'model_score': float(score),
                'change_percent': float((predicted_price - current_price) / current_price * 100)
            }
            
            self.logger.info(f"Прогноз линейной регрессии: {predicted_price:.2f} ({direction})")
            return result
        
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании: {str(e)}")
            return {}
    
    def _prepare_features(self, data: pd.DataFrame) -> tuple:
        """
        Подготовка признаков для модели.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
        
        Returns:
            tuple: (X, y) - признаки и целевая переменная
        """
        # Простой подход: используем индекс времени как признак
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['close'].values
        
        return X, y
    
    def predict_with_features(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """
        Прогнозирование с дополнительными признаками (SMA, EMA, Volume).
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            horizon (int): Горизонт прогнозирования
        
        Returns:
            Dict[str, Any]: Результаты прогнозирования
        """
        self.logger.info("Прогнозирование с дополнительными признаками")
        
        try:
            # Добавляем технические индикаторы как признаки
            data = data.copy()
            
            # SMA
            data['sma_10'] = data['close'].rolling(window=10).mean()
            data['sma_30'] = data['close'].rolling(window=30).mean()
            
            # EMA
            data['ema_10'] = data['close'].ewm(span=10, adjust=False).mean()
            
            # Volume change
            if 'volume' in data.columns:
                data['volume_change'] = data['volume'].pct_change()
            
            # Удаляем NaN
            data = data.dropna()
            
            if len(data) < 10:
                return self.predict(data, horizon)
            
            # Подготовка признаков
            feature_columns = ['sma_10', 'sma_30', 'ema_10']
            if 'volume_change' in data.columns:
                feature_columns.append('volume_change')
            
            X = data[feature_columns].values
            y = data['close'].values
            
            # Нормализация
            X = self.scaler.fit_transform(X)
            
            # Обучение модели
            self.model = LinearRegression()
            self.model.fit(X, y)
            
            # Прогноз (используем последние известные значения признаков)
            last_features = X[-1:].reshape(1, -1)
            predicted_price = self.model.predict(last_features)[0]
            
            current_price = float(data['close'].iloc[-1])
            
            result = {
                'predicted_price': float(predicted_price),
                'current_price': current_price,
                'model_score': float(self.model.score(X, y)),
                'features_used': feature_columns
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании с признаками: {str(e)}")
            return self.predict(data, horizon)









