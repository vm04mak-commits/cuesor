"""
Hyperparameter Tuning для ML моделей

Автоматическая оптимизация гиперпараметров:
- Grid Search (полный перебор)
- Random Search (случайный поиск)
- Optuna (Bayesian optimization)
- Cross-validation для надёжности
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Optuna (опционально)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """Автоматическая настройка гиперпараметров."""
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        tuning_method: str = 'random_search',
        cv_folds: int = 5,
        n_iter: int = 50,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Инициализация Tuner.
        
        Args:
            model_type: Тип модели ('random_forest', 'gradient_boosting', 'ridge', 'lasso')
            tuning_method: Метод ('grid_search', 'random_search', 'optuna')
            cv_folds: Количество folds для cross-validation
            n_iter: Количество итераций для random_search/optuna
            random_state: Random seed
            n_jobs: Количество параллельных процессов
        """
        self.model_type = model_type
        self.tuning_method = tuning_method
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
        self.best_model: Any = None
        self.tuning_history: List[Dict] = []
    
    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Настроить гиперпараметры.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающая целевая переменная
            param_grid: Сетка параметров (если None, используется дефолтная)
            
        Returns:
            (Лучшая модель, Лучшие параметры)
        """
        print(f"\n⚙️  Hyperparameter Tuning:")
        print(f"   Модель: {self.model_type}")
        print(f"   Метод: {self.tuning_method}")
        print(f"   CV folds: {self.cv_folds}")
        
        # Получаем сетку параметров
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        # Выбираем метод настройки
        if self.tuning_method == 'grid_search':
            self.best_model, self.best_params = self._grid_search(
                X_train, y_train, param_grid
            )
        elif self.tuning_method == 'random_search':
            self.best_model, self.best_params = self._random_search(
                X_train, y_train, param_grid
            )
        elif self.tuning_method == 'optuna':
            if not OPTUNA_AVAILABLE:
                print("   ⚠️  Optuna не установлен, используем Random Search")
                self.best_model, self.best_params = self._random_search(
                    X_train, y_train, param_grid
                )
            else:
                self.best_model, self.best_params = self._optuna_search(
                    X_train, y_train, param_grid
                )
        else:
            raise ValueError(f"Unknown tuning method: {self.tuning_method}")
        
        print(f"\n✅ Настройка завершена!")
        print(f"   Лучший score (R²): {self.best_score:.4f}")
        print(f"   Лучшие параметры:")
        for param, value in self.best_params.items():
            print(f"      {param}: {value}")
        
        return self.best_model, self.best_params
    
    def _get_default_param_grid(self) -> Dict:
        """Получить дефолтную сетку параметров."""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        elif self.model_type == 'gradient_boosting':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        elif self.model_type == 'ridge':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            }
        
        elif self.model_type == 'lasso':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000, 5000]
            }
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_base_model(self):
        """Получить базовую модель."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(random_state=self.random_state)
        elif self.model_type == 'ridge':
            return Ridge(random_state=self.random_state)
        elif self.model_type == 'lasso':
            return Lasso(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _grid_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict
    ) -> Tuple[Any, Dict]:
        """Grid Search."""
        base_model = self._get_base_model()
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring='r2',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_score = grid_search.best_score_
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _random_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict
    ) -> Tuple[Any, Dict]:
        """Random Search."""
        base_model = self._get_base_model()
        
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=self.n_iter,
            cv=self.cv_folds,
            scoring='r2',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_score = random_search.best_score_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def _optuna_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict
    ) -> Tuple[Any, Dict]:
        """Optuna (Bayesian Optimization)."""
        
        def objective(trial):
            # Предлагаем параметры
            params = {}
            
            for param_name, param_values in param_grid.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, type(None))) for v in param_values):
                        # Integer или None
                        non_none = [v for v in param_values if v is not None]
                        if non_none:
                            params[param_name] = trial.suggest_int(
                                param_name,
                                min(non_none),
                                max(non_none)
                            )
                        else:
                            params[param_name] = None
                    elif all(isinstance(v, (float, int)) for v in param_values):
                        # Float/int
                        params[param_name] = trial.suggest_float(
                            param_name,
                            min(param_values),
                            max(param_values)
                        )
                    else:
                        # Categorical
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_values
                        )
            
            # Создаём и обучаем модель
            if self.model_type == 'random_forest':
                model = RandomForestRegressor(
                    **params,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            elif self.model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    **params,
                    random_state=self.random_state
                )
            elif self.model_type == 'ridge':
                model = Ridge(**params, random_state=self.random_state)
            elif self.model_type == 'lasso':
                model = Lasso(**params, random_state=self.random_state)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Cross-validation
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=self.cv_folds,
                scoring='r2',
                n_jobs=self.n_jobs
            )
            
            return scores.mean()
        
        # Создаём study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(objective, n_trials=self.n_iter, show_progress_bar=True)
        
        # Лучшие параметры
        best_params = study.best_params
        self.best_score = study.best_value
        
        # Обучаем финальную модель с лучшими параметрами
        if self.model_type == 'random_forest':
            best_model = RandomForestRegressor(
                **best_params,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif self.model_type == 'gradient_boosting':
            best_model = GradientBoostingRegressor(
                **best_params,
                random_state=self.random_state
            )
        elif self.model_type == 'ridge':
            best_model = Ridge(**best_params, random_state=self.random_state)
        elif self.model_type == 'lasso':
            best_model = Lasso(**best_params, random_state=self.random_state)
        
        best_model.fit(X_train, y_train)
        
        return best_model, best_params
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Оценить лучшую модель на тестовых данных.
        
        Args:
            X_test: Тестовые признаки
            y_test: Тестовая целевая переменная
            
        Returns:
            Словарь с метриками
        """
        if self.best_model is None:
            raise ValueError("Model not tuned yet. Call tune() first.")
        
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    method: str = 'random_search',
    n_iter: int = 50
) -> Tuple[Any, Dict[str, Any]]:
    """
    Удобная функция для быстрой настройки.
    
    Args:
        X_train: Обучающие признаки
        y_train: Обучающая целевая переменная
        model_type: Тип модели
        method: Метод настройки
        n_iter: Количество итераций
        
    Returns:
        (Лучшая модель, Лучшие параметры)
    """
    tuner = HyperparameterTuner(
        model_type=model_type,
        tuning_method=method,
        n_iter=n_iter
    )
    
    best_model, best_params = tuner.tune(X_train, y_train)
    
    return best_model, best_params


if __name__ == "__main__":
    # Пример использования
    print("=" * 80)
    print("🧪 ТЕСТ HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Создаём тестовые данные
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nДанные: {X_train.shape}")
    
    # Настраиваем Random Forest
    tuner = HyperparameterTuner(
        model_type='random_forest',
        tuning_method='random_search',
        n_iter=20
    )
    
    best_model, best_params = tuner.tune(X_train, y_train)
    
    # Оцениваем
    metrics = tuner.evaluate(X_test, y_test)
    
    print(f"\n📊 Метрики на тестовых данных:")
    for metric, value in metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")
    
    print("\n✅ Тест завершён!")
