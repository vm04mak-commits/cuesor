"""
Модуль универсальной модели для всех акций.
Одна модель обучается на данных всех акций сразу.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.database import Database

# TensorFlow для Deep Learning моделей
try:
    import tensorflow as tf
    # Настройка GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pass
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class UniversalModelTrainer:
    """
    Тренер универсальной модели для всех акций.
    """
    
    def __init__(self, config, logger):
        """
        Инициализация.
        
        Args:
            config: Объект конфигурации
            logger: Объект логгера
        """
        self.config = config
        self.logger = logger
        
        # Директория для универсальной модели
        self.models_dir = config.base_path / "models" / "universal"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # База данных
        db_path = config.base_path / "data" / "market_data.db"
        self.database = Database(db_path, logger)
        
        # Энкодер для тикеров
        self.ticker_encoder = LabelEncoder()
        
        self.logger.info("UniversalModelTrainer инициализирован")
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загрузка данных всех акций из БД.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (котировки всех акций, индикаторы всех акций)
        """
        self.logger.info("Загрузка данных всех акций...")
        
        tickers = self.database.get_available_tickers()
        all_quotes = []
        all_indicators = []
        
        for ticker in tickers:
            quotes = self.database.load_quotes(ticker)
            indicators = self.database.load_indicators(ticker)
            
            if not quotes.empty and not indicators.empty:
                quotes['ticker'] = ticker
                all_quotes.append(quotes)
                
                # Добавляем тикер к индикаторам
                indicators_reset = indicators.reset_index()
                indicators_reset['ticker'] = ticker
                all_indicators.append(indicators_reset)
        
        # Объединяем все данные
        combined_quotes = pd.concat(all_quotes, ignore_index=True) if all_quotes else pd.DataFrame()
        combined_indicators = pd.concat(all_indicators, ignore_index=True) if all_indicators else pd.DataFrame()
        
        self.logger.info(f"Загружено {len(tickers)} акций, {len(combined_quotes)} записей котировок")
        
        return combined_quotes, combined_indicators
    
    def prepare_dataset(self, quotes: pd.DataFrame, indicators: pd.DataFrame, 
                       target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Подготовка датасета для обучения универсальной модели.
        
        Args:
            quotes (pd.DataFrame): Котировки всех акций
            indicators (pd.DataFrame): Индикаторы всех акций
            target_horizon (int): Горизонт прогноза
        
        Returns:
            Tuple[pd.DataFrame, pd.Series, list]: (X, y, list_of_tickers)
        """
        self.logger.info("Подготовка датасета...")
        
        # Объединяем котировки и индикаторы
        data = pd.merge(quotes, indicators, on=['ticker', 'date'], how='left')
        
        # Сортируем по тикеру и дате
        data = data.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Удаляем NaN
        data = data.dropna()
        
        if len(data) < 100:
            raise ValueError("Недостаточно данных для обучения")
        
        # Создаём целевую переменную (цена через N дней для каждого тикера)
        data['target'] = data.groupby('ticker')['close'].shift(-target_horizon)
        
        # Удаляем строки без целевой переменной
        data = data.dropna(subset=['target'])
        
        # Энкодим тикер
        data['ticker_encoded'] = self.ticker_encoder.fit_transform(data['ticker'])
        
        # Дополнительные признаки
        data['price_change_1d'] = data.groupby('ticker')['close'].pct_change(1)
        data['price_change_5d'] = data.groupby('ticker')['close'].pct_change(5)
        data['volume_ma_ratio'] = data['volume'] / data.groupby('ticker')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        
        # Удаляем NaN после создания признаков
        data = data.dropna()
        
        # Признаки для модели
        feature_columns = [col for col in data.columns if col not in [
            'ticker', 'date', 'target', 'value'
        ]]
        
        X = data[feature_columns]
        y = data['target']
        tickers_list = data['ticker'].tolist()
        
        self.logger.info(f"Подготовлено {len(X)} примеров с {len(feature_columns)} признаками")
        self.logger.info(f"Признаки: {', '.join(feature_columns[:10])}{'...' if len(feature_columns) > 10 else ''}")
        
        return X, y, tickers_list
    
    def train(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'gradient_boosting',
             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Обучение универсальной модели.
        
        Args:
            X (pd.DataFrame): Признаки
            y (pd.Series): Целевая переменная
            model_type (str): Тип модели
            test_size (float): Размер тестовой выборки
        
        Returns:
            Dict[str, Any]: Результаты обучения
        """
        self.logger.info(f"Обучение универсальной модели: {model_type}")
        
        # Разделение на train/test (хронологически)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Deep Learning модели
        if model_type in ['lstm', 'gru']:
            return self._train_deep_learning(
                X_train_scaled, X_test_scaled, 
                y_train, y_test, 
                scaler, model_type,
                features=list(X.columns)
            )
        
        # Классические модели
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        else:  # gradient_boosting
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                min_samples_split=10,
                random_state=42,
                verbose=1
            )
        
        # Обучение
        self.logger.info("Начинаем обучение...")
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
            'results': results,
            'ticker_encoder': self.ticker_encoder
        }
    
    def _train_deep_learning(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: pd.Series, y_test: pd.Series,
                            scaler: StandardScaler, model_type: str,
                            features: list) -> Dict[str, Any]:
        """
        Обучение Deep Learning модели (LSTM или GRU).
        
        Args:
            X_train: Обучающие признаки
            X_test: Тестовые признаки
            y_train: Обучающие метки
            y_test: Тестовые метки
            scaler: Обученный StandardScaler
            model_type: Тип модели (lstm или gru)
            features: Список названий признаков
        
        Returns:
            Dict[str, Any]: Результаты обучения
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            raise ImportError("TensorFlow не установлен. Установите: pip install tensorflow>=2.12.0")
        
        self.logger.info(f"Обучение {model_type.upper()} модели...")
        
        # Подготовка данных для RNN (reshape для временных рядов)
        timesteps = 1
        n_features = X_train.shape[1]
        
        X_train_reshaped = X_train.reshape((X_train.shape[0], timesteps, n_features))
        X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, n_features))
        
        # Создание модели
        model = Sequential()
        
        if model_type == 'lstm':
            model.add(LSTM(128, return_sequences=True, input_shape=(timesteps, n_features)))
            model.add(Dropout(0.2))
            model.add(LSTM(64))
            model.add(Dropout(0.2))
        else:  # GRU
            model.add(GRU(128, return_sequences=True, input_shape=(timesteps, n_features)))
            model.add(Dropout(0.2))
            model.add(GRU(64))
            model.add(Dropout(0.2))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        # Компиляция
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Обучение
        self.logger.info("Начинаем обучение...")
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_test_reshaped, y_test),
            epochs=100,
            batch_size=256,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Предсказания
        y_train_pred = model.predict(X_train_reshaped, verbose=0).flatten()
        y_test_pred = model.predict(X_test_reshaped, verbose=0).flatten()
        
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
            'features': features,
            'history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            },
            'trained_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Обучение завершено. Test R²: {results['test_metrics']['r2']:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'results': results,
            'ticker_encoder': self.ticker_encoder,
            'model_type': model_type  # Важно для правильного предсказания
        }
    
    def save_model(self, model_data: Dict[str, Any], model_name: str = "universal_model") -> str:
        """
        Сохранение универсальной модели.
        
        Args:
            model_data (Dict[str, Any]): Данные модели
            model_name (str): Имя модели
        
        Returns:
            str: Путь к сохранённой модели
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Проверяем тип модели (Deep Learning или классическая)
        is_deep_learning = model_data.get('model_type') in ['lstm', 'gru']
        
        if is_deep_learning:
            # Для TensorFlow модели сохраняем отдельно
            model_dir = self.models_dir / f"{model_name}_{timestamp}"
            model_dir.mkdir(exist_ok=True)
            
            # Сохранение TensorFlow модели
            model_data['model'].save(str(model_dir / "model.keras"))
            
            # Сохранение остальных данных
            with open(model_dir / "metadata.pkl", 'wb') as f:
                pickle.dump({
                    'scaler': model_data['scaler'],
                    'ticker_encoder': model_data['ticker_encoder'],
                    'results': model_data['results'],
                    'model_type': model_data['model_type']
                }, f)
            
            model_path = str(model_dir)
        else:
            # Для классических моделей используем pickle
            model_file = self.models_dir / f"{model_name}_{timestamp}.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model_data['model'],
                    'scaler': model_data['scaler'],
                    'ticker_encoder': model_data['ticker_encoder'],
                    'results': model_data['results']
                }, f)
            
            model_path = str(model_file)
        
        # Сохранение метрик
        metrics_file = self.models_dir / f"{model_name}_{timestamp}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(model_data['results'], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Модель сохранена: {model_path}")
        return model_path
    
    def load_model(self, model_path: str = None) -> Dict[str, Any]:
        """
        Загрузка универсальной модели.
        
        Args:
            model_path (str): Путь к модели (если None, загружается последняя)
        
        Returns:
            Dict[str, Any]: Данные модели
        """
        from pathlib import Path
        
        if model_path is None:
            # Загрузить последнюю модель
            model_files = list(self.models_dir.glob("universal_model_*.pkl"))
            model_dirs = [d for d in self.models_dir.glob("universal_model_*") if d.is_dir()]
            
            all_models = [(str(f), f.stat().st_mtime) for f in model_files]
            all_models += [(str(d), d.stat().st_mtime) for d in model_dirs]
            
            if not all_models:
                raise FileNotFoundError("Нет сохранённых моделей")
            
            # Сортируем по времени создания и берём последнюю
            model_path = sorted(all_models, key=lambda x: x[1])[-1][0]
        
        model_path = Path(model_path)
        
        # Проверяем тип модели (директория = Deep Learning, файл = классическая)
        if model_path.is_dir():
            # Deep Learning модель
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError("TensorFlow не установлен для загрузки Deep Learning модели")
            
            # Загружаем TensorFlow модель
            model = tf.keras.models.load_model(str(model_path / "model.keras"))
            
            # Загружаем метаданные
            with open(model_path / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            model_data = {
                'model': model,
                'scaler': metadata['scaler'],
                'ticker_encoder': metadata['ticker_encoder'],
                'results': metadata['results'],
                'model_type': metadata.get('model_type', 'lstm')
            }
        else:
            # Классическая модель
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        
        self.logger.info(f"Модель загружена: {model_path}")
        return model_data




