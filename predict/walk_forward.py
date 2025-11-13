"""
Walk-Forward Backtesting

Более реалистичное тестирование моделей:
- Скользящее окно обучения/тестирования
- Симуляция реального использования
- Оценка стабильности модели во времени
- Anchored и Rolling режимы
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


class WalkForwardAnalyzer:
    """Walk-Forward анализ для ML моделей."""
    
    def __init__(
        self,
        train_size: int = 252,  # ~1 год торговых дней
        test_size: int = 21,    # ~1 месяц
        mode: str = 'rolling',   # 'rolling' или 'anchored'
        retrain_frequency: int = 21  # Переобучать каждые N дней
    ):
        """
        Инициализация Walk-Forward Analyzer.
        
        Args:
            train_size: Размер обучающего окна (дни)
            test_size: Размер тестового окна (дни)
            mode: 'rolling' (двигается окно) или 'anchored' (растёт обучение)
            retrain_frequency: Как часто переобучать модель
        """
        self.train_size = train_size
        self.test_size = test_size
        self.mode = mode
        self.retrain_frequency = retrain_frequency
        
        self.results: List[Dict] = []
        self.predictions: List[Dict] = []
    
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Запустить Walk-Forward анализ.
        
        Args:
            X: Признаки (должен быть DatetimeIndex)
            y: Целевая переменная
            model_factory: Функция, возвращающая новую модель
            verbose: Выводить прогресс
            
        Returns:
            Словарь с результатами
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        
        if verbose:
            print(f"\n📈 Walk-Forward Backtesting:")
            print(f"   Режим: {self.mode}")
            print(f"   Обучающее окно: {self.train_size} дней")
            print(f"   Тестовое окно: {self.test_size} дней")
            print(f"   Переобучение: каждые {self.retrain_frequency} дней")
        
        # Разбиваем на окна
        windows = self._create_windows(X, y)
        
        if verbose:
            print(f"   Всего окон: {len(windows)}")
        
        # Проходим по окнам
        for i, window in enumerate(windows):
            train_start, train_end, test_start, test_end = window
            
            # Обучающие данные
            X_train = X.loc[train_start:train_end]
            y_train = y.loc[train_start:train_end]
            
            # Тестовые данные
            X_test = X.loc[test_start:test_end]
            y_test = y.loc[test_start:test_end]
            
            if len(X_train) < 10 or len(X_test) < 1:
                continue
            
            # Обучаем модель (только если нужно переобучение)
            if i == 0 or i % (self.retrain_frequency // self.test_size) == 0:
                model = model_factory()
                model.fit(X_train, y_train)
            
            # Прогнозируем
            y_pred = model.predict(X_test)
            
            # Сохраняем результаты
            for date, actual, pred in zip(X_test.index, y_test.values, y_pred):
                self.predictions.append({
                    'date': date,
                    'actual': actual,
                    'predicted': pred,
                    'window': i
                })
            
            # Метрики по окну
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            window_result = {
                'window': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
            }
            
            self.results.append(window_result)
            
            if verbose and i % 10 == 0:
                print(f"   Окно {i+1}/{len(windows)}: R²={window_result['r2']:.3f}, RMSE={window_result['rmse']:.3f}")
        
        # Агрегированные результаты
        aggregate = self._aggregate_results()
        
        if verbose:
            print(f"\n✅ Анализ завершён!")
            print(f"   Средний R²: {aggregate['mean_r2']:.4f}")
            print(f"   Средний RMSE: {aggregate['mean_rmse']:.4f}")
            print(f"   Стабильность R²: {aggregate['std_r2']:.4f}")
        
        return aggregate
    
    def _create_windows(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Создать окна для анализа."""
        windows = []
        
        dates = X.index
        n_dates = len(dates)
        
        if self.mode == 'rolling':
            # Скользящее окно
            for i in range(self.train_size, n_dates, self.test_size):
                train_start = dates[i - self.train_size]
                train_end = dates[i - 1]
                
                test_start = dates[i]
                test_end_idx = min(i + self.test_size - 1, n_dates - 1)
                test_end = dates[test_end_idx]
                
                windows.append((train_start, train_end, test_start, test_end))
                
                if test_end_idx >= n_dates - 1:
                    break
        
        elif self.mode == 'anchored':
            # Якорное окно (обучение растёт)
            train_start = dates[0]
            
            for i in range(self.train_size, n_dates, self.test_size):
                train_end = dates[i - 1]
                
                test_start = dates[i]
                test_end_idx = min(i + self.test_size - 1, n_dates - 1)
                test_end = dates[test_end_idx]
                
                windows.append((train_start, train_end, test_start, test_end))
                
                if test_end_idx >= n_dates - 1:
                    break
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return windows
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Агрегировать результаты по окнам."""
        if not self.results:
            return {}
        
        df_results = pd.DataFrame(self.results)
        
        aggregate = {
            'n_windows': len(self.results),
            'mean_r2': df_results['r2'].mean(),
            'std_r2': df_results['r2'].std(),
            'min_r2': df_results['r2'].min(),
            'max_r2': df_results['r2'].max(),
            'mean_rmse': df_results['rmse'].mean(),
            'std_rmse': df_results['rmse'].std(),
            'mean_mae': df_results['mae'].mean(),
            'mean_mape': df_results['mape'].mean(),
            'results_by_window': self.results,
            'all_predictions': self.predictions
        }
        
        return aggregate
    
    def get_results_df(self) -> pd.DataFrame:
        """Получить результаты в виде DataFrame."""
        return pd.DataFrame(self.results)
    
    def get_predictions_df(self) -> pd.DataFrame:
        """Получить все прогнозы в виде DataFrame."""
        return pd.DataFrame(self.predictions)
    
    def plot_results(self, figsize: Tuple[int, int] = (14, 10)):
        """Визуализировать результаты."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib не установлен")
            return
        
        df_results = self.get_results_df()
        df_predictions = self.get_predictions_df()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. R² по окнам
        ax = axes[0, 0]
        ax.plot(df_results['window'], df_results['r2'], marker='o')
        ax.axhline(df_results['r2'].mean(), color='r', linestyle='--', label='Среднее')
        ax.set_xlabel('Окно')
        ax.set_ylabel('R²')
        ax.set_title('R² Score по окнам')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. RMSE по окнам
        ax = axes[0, 1]
        ax.plot(df_results['window'], df_results['rmse'], marker='o', color='orange')
        ax.axhline(df_results['rmse'].mean(), color='r', linestyle='--', label='Среднее')
        ax.set_xlabel('Окно')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE по окнам')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Прогнозы vs Факт
        ax = axes[1, 0]
        ax.scatter(df_predictions['actual'], df_predictions['predicted'], alpha=0.5)
        
        # Линия идеального прогноза
        min_val = min(df_predictions['actual'].min(), df_predictions['predicted'].min())
        max_val = max(df_predictions['actual'].max(), df_predictions['predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Идеальный')
        
        ax.set_xlabel('Фактическое значение')
        ax.set_ylabel('Прогноз')
        ax.set_title('Прогнозы vs Факт')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Временной ряд прогнозов
        ax = axes[1, 1]
        ax.plot(df_predictions['date'], df_predictions['actual'], label='Факт', alpha=0.7)
        ax.plot(df_predictions['date'], df_predictions['predicted'], label='Прогноз', alpha=0.7)
        ax.set_xlabel('Дата')
        ax.set_ylabel('Значение')
        ax.set_title('Временной ряд')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable,
    train_size: int = 252,
    test_size: int = 21,
    mode: str = 'rolling'
) -> Dict[str, Any]:
    """
    Удобная функция для Walk-Forward анализа.
    
    Args:
        X: Признаки
        y: Целевая переменная
        model_factory: Функция, создающая модель
        train_size: Размер обучающего окна
        test_size: Размер тестового окна
        mode: Режим ('rolling' или 'anchored')
        
    Returns:
        Словарь с результатами
    """
    analyzer = WalkForwardAnalyzer(
        train_size=train_size,
        test_size=test_size,
        mode=mode
    )
    
    results = analyzer.run(X, y, model_factory)
    
    return results


if __name__ == "__main__":
    # Пример использования
    print("=" * 80)
    print("🧪 ТЕСТ WALK-FORWARD BACKTESTING")
    print("=" * 80)
    
    # Создаём тестовые данные с временным индексом
    from sklearn.ensemble import RandomForestRegressor
    
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(500, 10),
        columns=[f"feature_{i}" for i in range(10)],
        index=dates
    )
    
    # Целевая переменная с трендом
    y = pd.Series(
        np.cumsum(np.random.randn(500)) + X['feature_0'] * 2,
        index=dates
    )
    
    print(f"\nДанные: {X.shape}")
    print(f"Период: {X.index[0]} - {X.index[-1]}")
    
    # Model factory
    def create_model():
        return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Walk-Forward анализ
    analyzer = WalkForwardAnalyzer(
        train_size=100,
        test_size=20,
        mode='rolling',
        retrain_frequency=40
    )
    
    results = analyzer.run(X, y, create_model)
    
    print(f"\n📊 Результаты:")
    print(f"   Окон: {results['n_windows']}")
    print(f"   Средний R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    print(f"   Диапазон R²: [{results['min_r2']:.4f}, {results['max_r2']:.4f}]")
    print(f"   Средний RMSE: {results['mean_rmse']:.4f}")
    
    # Визуализация
    # analyzer.plot_results()
    
    print("\n✅ Тест завершён!")

