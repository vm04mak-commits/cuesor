"""
Deep Learning Models Module

LSTM/GRU модели для прогнозирования временных рядов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
    
    # Настройка GPU для оптимальной работы
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Разрешаем динамическое выделение памяти GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 GPU доступен: {len(gpus)} устройств")
        except RuntimeError as e:
            print(f"⚠️  Ошибка настройки GPU: {e}")
    else:
        print("💻 GPU не найден, используется CPU")
        
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow не установлен. Установите: pip install tensorflow>=2.12.0")


class DeepLearningPredictor:
    """
    Deep Learning модели для прогнозирования цен акций.
    
    Поддерживаемые архитектуры:
    - LSTM (Long Short-Term Memory)
    - GRU (Gated Recurrent Unit)
    - 1D CNN
    - Hybrid (CNN + LSTM)
    """
    
    def __init__(
        self,
        model_type: str = 'LSTM',
        sequence_length: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация Deep Learning Predictor.
        
        Args:
            model_type: Тип модели ('LSTM', 'GRU', 'CNN', 'Hybrid')
            sequence_length: Длина последовательности (дней)
            logger: Логгер
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен. Установите: pip install tensorflow>=2.12.0")
        
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.logger = logger
        self.model = None
        self.history = None
        self.scaler_X = None
        self.scaler_y = None
    
    # ========== СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ==========
    
    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание последовательностей для RNN моделей.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Tuple[X_sequences, y_sequences]
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:i+self.sequence_length])
            y_sequences.append(y[i+self.sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    # ========== ПОСТРОЕНИЕ МОДЕЛЕЙ ==========
    
    def build_lstm_model(
        self,
        input_shape: Tuple,
        units: List[int] = [128, 64],
        dropout: float = 0.2
    ) -> keras.Model:
        """
        Построение LSTM модели.
        
        Args:
            input_shape: Форма входа (sequence_length, n_features)
            units: Список количества нейронов в слоях
            dropout: Dropout rate
            
        Returns:
            Keras модель
        """
        model = models.Sequential()
        
        # Первый LSTM слой
        model.add(layers.LSTM(
            units[0],
            return_sequences=len(units) > 1,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(dropout))
        
        # Дополнительные LSTM слои
        for i in range(1, len(units)):
            return_seq = i < len(units) - 1
            model.add(layers.LSTM(units[i], return_sequences=return_seq))
            model.add(layers.Dropout(dropout))
        
        # Выходной слой
        model.add(layers.Dense(1))
        
        return model
    
    def build_gru_model(
        self,
        input_shape: Tuple,
        units: List[int] = [128, 64],
        dropout: float = 0.2
    ) -> keras.Model:
        """
        Построение GRU модели.
        
        Args:
            input_shape: Форма входа
            units: Список количества нейронов
            dropout: Dropout rate
            
        Returns:
            Keras модель
        """
        model = models.Sequential()
        
        # Первый GRU слой
        model.add(layers.GRU(
            units[0],
            return_sequences=len(units) > 1,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(dropout))
        
        # Дополнительные GRU слои
        for i in range(1, len(units)):
            return_seq = i < len(units) - 1
            model.add(layers.GRU(units[i], return_sequences=return_seq))
            model.add(layers.Dropout(dropout))
        
        # Выходной слой
        model.add(layers.Dense(1))
        
        return model
    
    def build_cnn_model(
        self,
        input_shape: Tuple,
        filters: List[int] = [64, 32],
        kernel_size: int = 3,
        dropout: float = 0.2
    ) -> keras.Model:
        """
        Построение 1D CNN модели.
        
        Args:
            input_shape: Форма входа
            filters: Список количества фильтров
            kernel_size: Размер kernel
            dropout: Dropout rate
            
        Returns:
            Keras модель
        """
        model = models.Sequential()
        
        # Conv1D слои
        for i, f in enumerate(filters):
            if i == 0:
                model.add(layers.Conv1D(f, kernel_size, activation='relu', input_shape=input_shape))
            else:
                model.add(layers.Conv1D(f, kernel_size, activation='relu'))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Dropout(dropout))
        
        # Flatten и Dense
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))
        
        return model
    
    def build_hybrid_model(
        self,
        input_shape: Tuple,
        cnn_filters: List[int] = [64, 32],
        lstm_units: List[int] = [64],
        dropout: float = 0.2
    ) -> keras.Model:
        """
        Построение Hybrid модели (CNN + LSTM).
        
        Args:
            input_shape: Форма входа
            cnn_filters: Фильтры CNN
            lstm_units: Нейроны LSTM
            dropout: Dropout rate
            
        Returns:
            Keras модель
        """
        model = models.Sequential()
        
        # CNN слои
        for i, f in enumerate(cnn_filters):
            if i == 0:
                model.add(layers.Conv1D(f, 3, activation='relu', input_shape=input_shape))
            else:
                model.add(layers.Conv1D(f, 3, activation='relu'))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Dropout(dropout))
        
        # LSTM слои
        for i, units in enumerate(lstm_units):
            return_seq = i < len(lstm_units) - 1
            model.add(layers.LSTM(units, return_sequences=return_seq))
            model.add(layers.Dropout(dropout))
        
        # Выходной слой
        model.add(layers.Dense(1))
        
        return model
    
    # ========== ОБУЧЕНИЕ ==========
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        **model_params
    ) -> Dict:
        """
        Обучение Deep Learning модели.
        
        Args:
            X_train: Обучающая выборка (признаки)
            y_train: Обучающая выборка (целевая)
            X_val: Валидационная выборка (признаки)
            y_val: Валидационная выборка (целевая)
            epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Learning rate
            early_stopping_patience: Терпение для early stopping
            **model_params: Дополнительные параметры модели
            
        Returns:
            Dict: История обучения и метрики
        """
        print("\n" + "="*80)
        print(f"🤖 ОБУЧЕНИЕ {self.model_type} МОДЕЛИ")
        print("="*80)
        print()
        
        # Создаём последовательности
        print(f"Создание последовательностей (length={self.sequence_length})...")
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        print(f"   Train sequences: {X_train_seq.shape}")
        if validation_data:
            print(f"   Val sequences:   {X_val_seq.shape}")
        print()
        
        # Построение модели
        print(f"Построение {self.model_type} архитектуры...")
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        
        if self.model_type == 'LSTM':
            self.model = self.build_lstm_model(input_shape, **model_params)
        elif self.model_type == 'GRU':
            self.model = self.build_gru_model(input_shape, **model_params)
        elif self.model_type == 'CNN':
            self.model = self.build_cnn_model(input_shape, **model_params)
        elif self.model_type == 'Hybrid':
            self.model = self.build_hybrid_model(input_shape, **model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Компиляция
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(f"✅ Модель построена")
        print()
        print("Архитектура:")
        self.model.summary(print_fn=lambda x: print(f"   {x}"))
        print()
        
        # Callbacks
        callback_list = []
        
        # Early Stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # Reduce LR on Plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Обучение
        print(f"Начало обучения ({epochs} эпох, batch_size={batch_size})...")
        print()
        
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        print()
        print("✅ Обучение завершено")
        print()
        
        # Метрики
        train_loss = self.history.history['loss'][-1]
        train_mae = self.history.history['mae'][-1]
        
        results = {
            'train_loss': train_loss,
            'train_mae': train_mae,
            'epochs_trained': len(self.history.history['loss']),
            'history': self.history.history
        }
        
        if validation_data:
            val_loss = self.history.history['val_loss'][-1]
            val_mae = self.history.history['val_mae'][-1]
            results['val_loss'] = val_loss
            results['val_mae'] = val_mae
        
        # Выводим метрики
        print("📊 Финальные метрики:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Train MAE:  {train_mae:.4f}")
        if validation_data:
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val MAE:    {val_mae:.4f}")
        print()
        
        return results
    
    # ========== ПРЕДСКАЗАНИЕ ==========
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание на новых данных.
        
        Args:
            X: Матрица признаков
            
        Returns:
            Предсказания
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        # Создаём последовательности (без y, так как предсказываем)
        X_sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            X_sequences.append(X[i:i+self.sequence_length])
        
        X_sequences = np.array(X_sequences)
        
        # Предсказание
        predictions = self.model.predict(X_sequences, verbose=0)
        
        return predictions.flatten()
    
    # ========== СОХРАНЕНИЕ/ЗАГРУЗКА ==========
    
    def save_model(self, path: str):
        """
        Сохранение модели.
        
        Args:
            path: Путь для сохранения
        """
        if self.model is None:
            raise ValueError("Нет модели для сохранения")
        
        self.model.save(path)
        print(f"💾 Модель сохранена: {path}")
    
    def load_model(self, path: str):
        """
        Загрузка модели.
        
        Args:
            path: Путь к модели
        """
        self.model = keras.models.load_model(path)
        print(f"📦 Модель загружена: {path}")
    
    # ========== ВИЗУАЛИЗАЦИЯ ==========
    
    def plot_training_history(self):
        """
        Визуализация истории обучения.
        """
        if self.history is None:
            print("Нет истории обучения")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class EnsemblePredictor:
    """
    Ensemble предсказаний из нескольких моделей.
    """
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        """
        Инициализация Ensemble.
        
        Args:
            models: Список моделей
            weights: Веса моделей (опционально)
        """
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Количество весов должно совпадать с количеством моделей")
            # Нормализуем веса
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble предсказание.
        
        Args:
            X: Матрица признаков
            
        Returns:
            Взвешенное среднее предсказаний
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Взвешенное среднее
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание с оценкой неопределённости.
        
        Args:
            X: Матрица признаков
            
        Returns:
            Tuple[predictions, std]: Предсказания и стандартное отклонение
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Взвешенное среднее и std
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        ensemble_std = np.std(predictions, axis=0)
        
        return ensemble_pred, ensemble_std






