"""
Скрипт 8: Продвинутое обучение ML моделей

Использует:
- Feature Engineering (создание признаков)
- Feature Selection (отбор лучших)
- Hyperparameter Tuning (оптимизация)
- Walk-Forward Backtesting (реалистичная оценка)
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

from core.database import Database
from core.logger import Logger
from core.config import Config
from predict.feature_engineering import FeatureEngineer
from predict.feature_selection import FeatureSelector
from predict.hyperparameter_tuning import HyperparameterTuner
from predict.walk_forward import WalkForwardAnalyzer


def print_header(text: str):
    """Красивый заголовок."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def main():
    """Главная функция."""
    print_header("🚀 ПРОДВИНУТОЕ ОБУЧЕНИЕ ML МОДЕЛЕЙ")
    
    # Инициализация
    config = Config()
    logger = Logger("AdvancedTraining")
    db_path = config.base_path / "data" / "market_data.db"
    database = Database(db_path, logger)
    
    # Меню
    print("\nВыберите режим:")
    print("  1. Быстрая оптимизация (Feature Engineering + Selection)")
    print("  2. Полная оптимизация (+ Hyperparameter Tuning)")
    print("  3. Walk-Forward тест (реалистичная оценка)")
    print("  4. Всё сразу (максимальная точность)")
    
    mode_choice = input("\nРежим (1-4, по умолчанию 1): ").strip() or "1"
    
    # Параметры
    use_feature_engineering = mode_choice in ['1', '2', '3', '4']
    use_feature_selection = mode_choice in ['1', '2', '3', '4']
    use_hyperparameter_tuning = mode_choice in ['2', '4']
    use_walk_forward = mode_choice in ['3', '4']
    
    # Выбор модели
    print("\nВыберите тип модели:")
    print("  1. Random Forest (рекомендуется)")
    print("  2. Gradient Boosting")
    print("  3. Ridge Regression")
    
    model_choice = input("\nМодель (1-3, по умолчанию 1): ").strip() or "1"
    
    model_type_map = {
        '1': 'random_forest',
        '2': 'gradient_boosting',
        '3': 'ridge'
    }
    model_type = model_type_map.get(model_choice, 'random_forest')
    
    print(f"\n✅ Выбрана модель: {model_type}")
    print(f"✅ Feature Engineering: {'Да' if use_feature_engineering else 'Нет'}")
    print(f"✅ Feature Selection: {'Да' if use_feature_selection else 'Нет'}")
    print(f"✅ Hyperparameter Tuning: {'Да' if use_hyperparameter_tuning else 'Нет'}")
    print(f"✅ Walk-Forward Test: {'Да' if use_walk_forward else 'Нет'}")
    
    # Загрузка данных
    print_header("📊 ЗАГРУЗКА ДАННЫХ")
    
    tickers = database.get_available_tickers()
    print(f"Доступно тикеров: {len(tickers)}")
    
    # Собираем данные
    print("\nСбор данных по всем тикерам...")
    all_data = []
    
    for ticker in tickers[:100]:  # Ограничиваем для скорости
        try:
            quotes = database.load_quotes(ticker)
            indicators = database.load_indicators(ticker)
            
            if quotes.empty or indicators.empty:
                continue
            
            # Объединяем
            quotes = quotes.set_index('date')
            data = quotes.join(indicators, how='inner')
            
            # Добавляем тикер
            data['ticker'] = ticker
            
            # Целевая переменная (цена через 5 дней)
            data['target'] = data['close'].shift(-5)
            
            # Удаляем NaN
            data = data.dropna()
            
            if len(data) > 50:
                all_data.append(data)
        
        except Exception as e:
            logger.warning(f"Error loading {ticker}: {e}")
            continue
    
    if not all_data:
        print("❌ Нет данных для обучения")
        return
    
    # Объединяем всё
    df = pd.concat(all_data, axis=0)
    print(f"\n✅ Загружено записей: {len(df)}")
    print(f"   Период: {df.index.min()} - {df.index.max()}")
    
    # Подготовка данных
    print_header("🔧 ПОДГОТОВКА ДАННЫХ")
    
    # Выделяем признаки и целевую переменную
    target_col = 'target'
    exclude_cols = ['target', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Очистка исходных данных
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"Исходных признаков: {len(feature_cols)}")
    
    # Feature Engineering
    if use_feature_engineering:
        print_header("🔧 FEATURE ENGINEERING")
        
        engineer = FeatureEngineer(
            create_lags=True,
            create_rolling=True,
            create_technical=True,
            create_interactions=True,
            create_temporal=True,
            lag_periods=[1, 2, 3, 5],
            rolling_windows=[3, 5, 10]
        )
        
        # Применяем по группам (по тикерам)
        X_engineered_list = []
        y_engineered_list = []
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            X_ticker = X[mask].copy()
            y_ticker = y[mask].copy()
            
            # Индекс должен быть DatetimeIndex
            if not isinstance(X_ticker.index, pd.DatetimeIndex):
                X_ticker.index = pd.to_datetime(X_ticker.index)
            
            X_ticker_eng = engineer.fit_transform(X_ticker, target_col='target')
            
            X_engineered_list.append(X_ticker_eng)
            y_engineered_list.append(y_ticker)
        
        X = pd.concat(X_engineered_list, axis=0)
        y = pd.concat(y_engineered_list, axis=0)
        
        # Удаляем NaN и infinity
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Финальная проверка на outliers
        for col in X.columns:
            if X[col].std() > 0:
                upper = X[col].quantile(0.999)
                lower = X[col].quantile(0.001)
                X[col] = X[col].clip(lower, upper)
        
        print(f"После Feature Engineering: {X.shape[1]} признаков")
        print(f"Записей после очистки: {len(X)}")
    
    # Feature Selection
    if use_feature_selection:
        print_header("🔍 FEATURE SELECTION")
        
        # Ограничиваем выборку для Feature Selection (ускорение)
        sample_size = min(10000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        selector = FeatureSelector(
            methods=['correlation', 'variance', 'importance'],
            n_features_to_select=50,
            correlation_threshold=0.95
        )
        
        selector.fit(X_sample, y_sample)
        
        X = selector.transform(X)
        
        print(f"\nВыбрано признаков: {len(selector.selected_features)}")
        print("\nТоп-10 признаков:")
        print(selector.get_feature_scores(top_n=10))
    
    # Разделение на train/test
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\nОбучающая выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")
    
    # Hyperparameter Tuning
    if use_hyperparameter_tuning:
        print_header("⚙️  HYPERPARAMETER TUNING")
        
        tuner = HyperparameterTuner(
            model_type=model_type,
            tuning_method='random_search',
            cv_folds=5,
            n_iter=30,
            n_jobs=-1
        )
        
        best_model, best_params = tuner.tune(X_train, y_train)
        
        # Оценка на тесте
        metrics = tuner.evaluate(X_test, y_test)
        
        print(f"\n📊 Метрики на тестовых данных:")
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
    
    else:
        # Обучаем с дефолтными параметрами
        print_header("🎓 ОБУЧЕНИЕ МОДЕЛИ")
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'ridge':
            model = Ridge(random_state=42)
        
        model.fit(X_train, y_train)
        
        # Оценка
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        }
        
        print(f"\n📊 Метрики на тестовых данных:")
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
        
        best_model = model
        best_params = model.get_params()
    
    # Walk-Forward Backtesting
    if use_walk_forward:
        print_header("📈 WALK-FORWARD BACKTESTING")
        
        # Берём только данные с DatetimeIndex
        X_wf = X.copy()
        y_wf = y.copy()
        
        if not isinstance(X_wf.index, pd.DatetimeIndex):
            print("⚠️  Индекс не DatetimeIndex, пропускаем Walk-Forward")
        else:
            # Model factory
            def create_model():
                if model_type == 'random_forest':
                    return RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                elif model_type == 'gradient_boosting':
                    return GradientBoostingRegressor(**best_params, random_state=42)
                elif model_type == 'ridge':
                    return Ridge(**best_params, random_state=42)
            
            analyzer = WalkForwardAnalyzer(
                train_size=100,
                test_size=20,
                mode='rolling',
                retrain_frequency=40
            )
            
            wf_results = analyzer.run(X_wf, y_wf, create_model)
            
            print(f"\n📊 Walk-Forward результаты:")
            print(f"   Окон: {wf_results['n_windows']}")
            print(f"   Средний R²: {wf_results['mean_r2']:.4f} ± {wf_results['std_r2']:.4f}")
            print(f"   Диапазон R²: [{wf_results['min_r2']:.4f}, {wf_results['max_r2']:.4f}]")
            print(f"   Средний RMSE: {wf_results['mean_rmse']:.4f}")
    
    # Сохранение модели
    print_header("💾 СОХРАНЕНИЕ МОДЕЛИ")
    
    models_dir = Path("models/advanced")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"advanced_{model_type}_{timestamp}"
    model_path = models_dir / f"{model_name}.pkl"
    
    # Сохраняем
    model_data = {
        'model': best_model,
        'model_type': model_type,
        'params': best_params,
        'features': X.columns.tolist(),
        'metrics': metrics,
        'feature_engineering': use_feature_engineering,
        'feature_selection': use_feature_selection,
        'hyperparameter_tuning': use_hyperparameter_tuning,
        'timestamp': timestamp
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Модель сохранена: {model_path}")
    
    # Итоги
    print_header("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    
    print(f"\nМодель: {model_type}")
    print(f"Признаков: {len(X.columns)}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    
    print(f"\n💡 Используйте эту модель в:")
    print(f"   - python 4_predict_stocks.py")
    print(f"   - python 7_portfolio_trading.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

