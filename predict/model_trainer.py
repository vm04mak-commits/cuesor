"""
Модуль для обучения моделей прогнозирования.
Обучает модели на исторических данных и сохраняет их для использования.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.database import Database


class ModelTrainer:
    """
    Класс для обучения моделей прогнозирования на исторических данных.
    """
    
    def __init__(self, config, logger):
        """
        Инициализация тренера моделей.
        
        Args:
            config: Объект конфигурации системы
            logger: Объект логгера
        """
        self.config = config
        self.logger = logger
        
        # Директория для моделей
        self.models_dir = config.base_path / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # База данных
        db_path = config.base_path / "data" / "market_data.db"
        self.database = Database(db_path, logger)
        
        # CSV директория
        self.csv_dir = config.base_path / "data" / "csv"
        
        # Доступные модели
        self.available_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        self.logger.info("ModelTrainer инициализирован")
    
    def load_training_data(self, ticker: str, from_db: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загрузка данных для обучения (котировки + индикаторы).
        
        Args:
            ticker (str): Тикер акции
            from_db (bool): Загружать из БД (True) или CSV (False)
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (котировки, индикаторы)
        """
        self.logger.info(f"Загрузка данных для обучения: {ticker}")
        
        if from_db:
            # Из базы данных
            quotes = self.database.load_quotes(ticker)
            indicators = self.database.load_indicators(ticker)
        else:
            # Из CSV
            quotes_file = self.csv_dir / ticker / f"{ticker}.csv"
            indicators_file = self.csv_dir / ticker / f"{ticker}_indicators.csv"
            
            quotes = pd.read_csv(quotes_file, parse_dates=['date']) if quotes_file.exists() else pd.DataFrame()
            indicators = pd.read_csv(indicators_file, parse_dates=['date'], index_col='date') if indicators_file.exists() else pd.DataFrame()
        
        self.logger.info(f"Загружено {len(quotes)} котировок и {len(indicators)} записей индикаторов")
        return quotes, indicators
    
    def prepare_features(self, quotes: pd.DataFrame, indicators: pd.DataFrame, 
                        target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовка признаков и целевой переменной для обучения.
        
        Args:
            quotes (pd.DataFrame): Котировки
            indicators (pd.DataFrame): Индикаторы
            target_horizon (int): Горизонт прогноза в днях
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (признаки X, целевая переменная y)
        """
        self.logger.info(f"Подготовка признаков (horizon={target_horizon})")
        
        # Объединяем котировки и индикаторы
        if 'date' in quotes.columns:
            quotes = quotes.set_index('date')
        
        # Объединение
        data = quotes.join(indicators, how='left')
        
        # Удаляем NaN
        data = data.dropna()
        
        if len(data) < target_horizon + 10:
            raise ValueError("Недостаточно данных для обучения")
        
        # Создаём целевую переменную (цена через N дней)
        data['target'] = data['close'].shift(-target_horizon)
        
        # Удаляем последние N строк (где нет целевой переменной)
        data = data[:-target_horizon]
        
        # Признаки
        feature_columns = [col for col in data.columns if col not in ['target', 'date']]
        X = data[feature_columns]
        y = data['target']
        
        self.logger.info(f"Подготовлено {len(X)} примеров с {len(feature_columns)} признаками")
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'linear',
                   test_size: float = 0.2) -> Dict[str, Any]:
        """
        Обучение модели.
        
        Args:
            X (pd.DataFrame): Признаки
            y (pd.Series): Целевая переменная
            model_type (str): Тип модели
            test_size (float): Размер тестовой выборки
        
        Returns:
            Dict[str, Any]: Результаты обучения
        """
        self.logger.info(f"Обучение модели: {model_type}")
        
        if model_type not in self.available_models:
            raise ValueError(f"Неизвестная модель: {model_type}")
        
        # Разделение на train/test (с сохранением порядка для временных рядов)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение
        model = self.available_models[model_type]
        model.fit(X_train_scaled, y_train)
        
        # Предсказания
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Метрики
        results = {
            'model_type': model_type,
            'train_metrics': {
                'mae': float(mean_absolute_error(y_train, y_train_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                'r2': float(r2_score(y_train, y_train_pred))
            },
            'test_metrics': {
                'mae': float(mean_absolute_error(y_test, y_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                'r2': float(r2_score(y_test, y_test_pred))
            },
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': list(X.columns),
            'trained_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Обучение завершено. Test R²: {results['test_metrics']['r2']:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'results': results
        }
    
    def save_model(self, ticker: str, model_data: Dict[str, Any], model_type: str) -> str:
        """
        Сохранение обученной модели.
        
        Args:
            ticker (str): Тикер акции
            model_data (Dict[str, Any]): Данные модели (model, scaler, results)
            model_type (str): Тип модели
        
        Returns:
            str: Путь к сохранённой модели
        """
        # Директория для тикера
        ticker_dir = self.models_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Имя файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = ticker_dir / f"{ticker}_{model_type}_{timestamp}.pkl"
        
        # Сохранение
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'results': model_data['results']
            }, f)
        
        # Сохранение метрик в JSON
        metrics_file = ticker_dir / f"{ticker}_{model_type}_{timestamp}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(model_data['results'], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Модель сохранена: {model_file}")
        return str(model_file)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Загрузка сохранённой модели.
        
        Args:
            model_path (str): Путь к файлу модели
        
        Returns:
            Dict[str, Any]: Данные модели
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.logger.info(f"Модель загружена: {model_path}")
        return model_data
    
    def train_multiple_models(self, ticker: str, models: List[str] = None,
                            from_db: bool = True, target_horizon: int = 1) -> Dict[str, Dict]:
        """
        Обучение нескольких моделей и сравнение результатов.
        
        Args:
            ticker (str): Тикер акции
            models (List[str]): Список типов моделей для обучения
            from_db (bool): Загружать данные из БД
            target_horizon (int): Горизонт прогноза
        
        Returns:
            Dict[str, Dict]: Результаты для каждой модели
        """
        if models is None:
            models = list(self.available_models.keys())
        
        self.logger.info(f"Обучение {len(models)} моделей для {ticker}")
        
        # Загрузка данных
        quotes, indicators = self.load_training_data(ticker, from_db)
        
        # Подготовка признаков
        X, y = self.prepare_features(quotes, indicators, target_horizon)
        
        # Обучение моделей
        results = {}
        for model_type in models:
            try:
                self.logger.info(f"Обучение {model_type}...")
                model_data = self.train_model(X, y, model_type)
                
                # Сохранение
                model_path = self.save_model(ticker, model_data, model_type)
                
                results[model_type] = {
                    'metrics': model_data['results'],
                    'model_path': model_path
                }
            
            except Exception as e:
                self.logger.error(f"Ошибка обучения {model_type}: {str(e)}")
                results[model_type] = {'error': str(e)}
        
        # Сравнение моделей
        self._compare_models(results)
        
        return results
    
    def _compare_models(self, results: Dict[str, Dict]) -> None:
        """
        Вывод сравнения моделей.
        
        Args:
            results (Dict[str, Dict]): Результаты обучения моделей
        
        Returns:
            None
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("СРАВНЕНИЕ МОДЕЛЕЙ")
        self.logger.info("="*80)
        
        # Сортировка по Test R²
        sorted_models = sorted(
            [(name, data) for name, data in results.items() if 'error' not in data],
            key=lambda x: x[1]['metrics']['test_metrics']['r2'],
            reverse=True
        )
        
        for name, data in sorted_models:
            metrics = data['metrics']['test_metrics']
            self.logger.info(f"\n{name.upper()}:")
            self.logger.info(f"  R²: {metrics['r2']:.4f}")
            self.logger.info(f"  MAE: {metrics['mae']:.2f}")
            self.logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        
        if sorted_models:
            best_model = sorted_models[0][0]
            self.logger.info(f"\n🏆 Лучшая модель: {best_model.upper()}")
        
        self.logger.info("="*80 + "\n")
    
    def get_best_model(self, ticker: str) -> str:
        """
        Получение пути к лучшей модели для тикера.
        
        Args:
            ticker (str): Тикер акции
        
        Returns:
            str: Путь к лучшей модели
        """
        ticker_dir = self.models_dir / ticker
        
        if not ticker_dir.exists():
            return ""
        
        # Ищем все модели
        model_files = list(ticker_dir.glob("*_metrics.json"))
        
        if not model_files:
            return ""
        
        # Находим модель с лучшим R²
        best_r2 = -float('inf')
        best_model_path = ""
        
        for metrics_file in model_files:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            r2 = metrics['test_metrics']['r2']
            if r2 > best_r2:
                best_r2 = r2
                # Получаем путь к pkl файлу
                best_model_path = str(metrics_file).replace('_metrics.json', '.pkl')
        
        return best_model_path









