"""
Модуль Predict - прогнозирование.

Содержит:
- Predictor (главный класс прогнозирования)
- LinearModel (линейные модели)
- TimeSeriesModel (временные ряды)
- UniversalModelTrainer (универсальная модель)
- FeatureEngineer (автоматическое создание признаков)
- FeatureSelector (отбор признаков)
- HyperparameterTuner (оптимизация гиперпараметров)
- WalkForwardAnalyzer (Walk-Forward бэктестинг)
- DeepLearningPredictor (LSTM/GRU модели)
- EnsemblePredictor (ensemble моделей)
- ModelVersioning (версионирование моделей)
"""

from .predictor import Predictor
from .linear_model import LinearModel
from .time_series_model import TimeSeriesModel
from .universal_model import UniversalModelTrainer
from .model_versioning import ModelVersioning

# Новые модули для улучшения точности
from .feature_engineering import FeatureEngineer
from .feature_selection import FeatureSelector
from .hyperparameter_tuning import HyperparameterTuner
from .walk_forward import WalkForwardAnalyzer

# Deep Learning
try:
    from .deep_learning import DeepLearningPredictor, EnsemblePredictor
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

__all__ = [
    'Predictor', 
    'LinearModel', 
    'TimeSeriesModel', 
    'UniversalModelTrainer',
    'FeatureEngineer',
    'FeatureSelector',
    'HyperparameterTuner',
    'WalkForwardAnalyzer',
    'ModelVersioning'
]

if DEEP_LEARNING_AVAILABLE:
    __all__.extend(['DeepLearningPredictor', 'EnsemblePredictor'])




