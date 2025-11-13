"""
Feature Engineering для ML моделей

Автоматическое создание новых признаков для улучшения точности прогнозов:
- Лаги (прошлые значения)
- Скользящие статистики (mean, std, min, max)
- Технические трансформации (returns, momentum, rate of change)
- Взаимодействия между признаками
- Временные признаки (день недели, месяц, квартал)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Автоматическое создание признаков для ML."""
    
    def __init__(
        self,
        create_lags: bool = True,
        create_rolling: bool = True,
        create_technical: bool = True,
        create_interactions: bool = True,
        create_temporal: bool = True,
        lag_periods: List[int] = None,
        rolling_windows: List[int] = None
    ):
        """
        Инициализация Feature Engineer.
        
        Args:
            create_lags: Создавать лаги
            create_rolling: Создавать скользящие статистики
            create_technical: Создавать технические трансформации
            create_interactions: Создавать взаимодействия признаков
            create_temporal: Создавать временные признаки
            lag_periods: Периоды для лагов (по умолчанию [1, 2, 3, 5, 10])
            rolling_windows: Окна для скользящих статистик (по умолчанию [3, 5, 10, 20])
        """
        self.create_lags = create_lags
        self.create_rolling = create_rolling
        self.create_technical = create_technical
        self.create_interactions = create_interactions
        self.create_temporal = create_temporal
        
        self.lag_periods = lag_periods or [1, 2, 3, 5, 10]
        self.rolling_windows = rolling_windows or [3, 5, 10, 20]
        
        self.created_features: List[str] = []
        self.feature_descriptions: Dict[str, str] = {}
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """
        Создать все признаки.
        
        Args:
            df: Исходный DataFrame
            target_col: Название целевой переменной (не трансформировать)
            
        Returns:
            DataFrame с новыми признаками
        """
        df_engineered = df.copy()
        
        # Получаем числовые колонки (кроме target)
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        print(f"\n🔧 Feature Engineering:")
        print(f"   Исходных признаков: {len(numeric_cols)}")
        
        # 1. Лаги
        if self.create_lags and numeric_cols:
            df_engineered = self._create_lags(df_engineered, numeric_cols)
        
        # 2. Скользящие статистики
        if self.create_rolling and numeric_cols:
            df_engineered = self._create_rolling_features(df_engineered, numeric_cols)
        
        # 3. Технические трансформации
        if self.create_technical and numeric_cols:
            df_engineered = self._create_technical_features(df_engineered, numeric_cols)
        
        # 4. Временные признаки
        if self.create_temporal:
            df_engineered = self._create_temporal_features(df_engineered)
        
        # 5. Взаимодействия (на ограниченном наборе)
        if self.create_interactions:
            # Выбираем только важные признаки для взаимодействий
            important_cols = [col for col in numeric_cols if any(
                keyword in col.lower() 
                for keyword in ['close', 'volume', 'rsi', 'macd', 'bb']
            )][:5]  # Максимум 5 признаков
            
            if len(important_cols) >= 2:
                df_engineered = self._create_interactions(df_engineered, important_cols)
        
        # Конвертируем все числовые колонки в float64 для безопасной обработки
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_engineered[col] = df_engineered[col].astype('float64')
        
        # Удаляем NaN (возникают от лагов и rolling)
        df_engineered = df_engineered.fillna(method='bfill').fillna(0)
        
        # Заменяем infinity на NaN, затем на 0
        df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)
        df_engineered = df_engineered.fillna(0)
        
        # Ограничиваем очень большие значения (защита от outliers)
        for col in numeric_cols:
            try:
                if df_engineered[col].std() > 0:
                    upper_limit = df_engineered[col].quantile(0.999)
                    lower_limit = df_engineered[col].quantile(0.001)
                    df_engineered[col] = df_engineered[col].clip(lower_limit, upper_limit)
            except Exception as e:
                # Если не можем clip - просто пропускаем
                print(f"   Warning: Could not clip {col}: {e}")
                continue
        
        total_features = len(df_engineered.columns) - len(df.columns)
        print(f"   Создано новых признаков: {total_features}")
        print(f"   Всего признаков: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def _create_lags(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Создать лаги для указанных колонок."""
        for col in columns:
            for lag in self.lag_periods:
                new_col = f"{col}_lag_{lag}"
                df[new_col] = df[col].shift(lag)
                self.created_features.append(new_col)
                self.feature_descriptions[new_col] = f"Lag {lag} of {col}"
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Создать скользящие статистики."""
        for col in columns:
            for window in self.rolling_windows:
                # Mean
                new_col = f"{col}_rolling_mean_{window}"
                df[new_col] = df[col].rolling(window=window, min_periods=1).mean()
                self.created_features.append(new_col)
                
                # Std
                new_col = f"{col}_rolling_std_{window}"
                df[new_col] = df[col].rolling(window=window, min_periods=1).std()
                self.created_features.append(new_col)
                
                # Min
                new_col = f"{col}_rolling_min_{window}"
                df[new_col] = df[col].rolling(window=window, min_periods=1).min()
                self.created_features.append(new_col)
                
                # Max
                new_col = f"{col}_rolling_max_{window}"
                df[new_col] = df[col].rolling(window=window, min_periods=1).max()
                self.created_features.append(new_col)
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Создать технические трансформации."""
        for col in columns:
            # Returns (процентное изменение)
            new_col = f"{col}_returns"
            df[new_col] = df[col].pct_change()
            self.created_features.append(new_col)
            
            # Momentum (разница с N периодов назад)
            for period in [3, 5, 10]:
                new_col = f"{col}_momentum_{period}"
                df[new_col] = df[col] - df[col].shift(period)
                self.created_features.append(new_col)
            
            # Rate of Change (ROC)
            new_col = f"{col}_roc_5"
            df[new_col] = ((df[col] - df[col].shift(5)) / df[col].shift(5)) * 100
            self.created_features.append(new_col)
            
            # Z-score (нормализация)
            new_col = f"{col}_zscore"
            mean = df[col].rolling(window=20, min_periods=1).mean()
            std = df[col].rolling(window=20, min_periods=1).std()
            df[new_col] = (df[col] - mean) / (std + 1e-8)
            self.created_features.append(new_col)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создать временные признаки из индекса."""
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['week_of_year'] = df.index.isocalendar().week
            
            self.created_features.extend([
                'day_of_week', 'day_of_month', 'month', 'quarter', 'week_of_year'
            ])
        
        return df
    
    def _create_interactions(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Создать взаимодействия между признаками (только для важных)."""
        # Polynomial features (2-я степень) только для важных признаков
        if len(columns) >= 2:
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            
            # Создаём взаимодействия
            interactions = poly.fit_transform(df[columns])
            
            # Получаем названия
            feature_names = poly.get_feature_names_out(columns)
            
            # Добавляем только новые (взаимодействия, не исходные)
            for i, name in enumerate(feature_names):
                if ' ' in name:  # Это взаимодействие (содержит пробел)
                    col_name = name.replace(' ', '_x_')
                    df[col_name] = interactions[:, i]
                    self.created_features.append(col_name)
        
        return df
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Получить список созданных признаков."""
        summary = pd.DataFrame({
            'feature': self.created_features,
            'description': [self.feature_descriptions.get(f, 'N/A') for f in self.created_features]
        })
        
        return summary
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применить те же трансформации к новым данным.
        
        Note: fit_transform должен быть вызван первым на обучающих данных.
        """
        return self.fit_transform(df)


def create_advanced_features(
    df: pd.DataFrame,
    target_col: str = 'target',
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Удобная функция для быстрого создания признаков.
    
    Args:
        df: Исходный DataFrame
        target_col: Название целевой переменной
        config: Конфигурация (опционально)
        
    Returns:
        DataFrame с новыми признаками
    """
    if config is None:
        config = {
            'create_lags': True,
            'create_rolling': True,
            'create_technical': True,
            'create_interactions': True,
            'create_temporal': True
        }
    
    engineer = FeatureEngineer(**config)
    df_engineered = engineer.fit_transform(df, target_col)
    
    return df_engineered


if __name__ == "__main__":
    # Пример использования
    print("=" * 80)
    print("🧪 ТЕСТ FEATURE ENGINEERING")
    print("=" * 80)
    
    # Создаём тестовые данные
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df_test = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'rsi': np.random.uniform(30, 70, 100),
        'target': np.random.randn(100)
    }, index=dates)
    
    print(f"\nИсходные данные: {df_test.shape}")
    print(df_test.head())
    
    # Создаём признаки
    engineer = FeatureEngineer(
        lag_periods=[1, 2, 3],
        rolling_windows=[3, 5]
    )
    
    df_engineered = engineer.fit_transform(df_test, target_col='target')
    
    print(f"\nПосле feature engineering: {df_engineered.shape}")
    print(f"Новых признаков: {df_engineered.shape[1] - df_test.shape[1]}")
    
    print("\nПримеры новых признаков:")
    new_cols = [col for col in df_engineered.columns if col not in df_test.columns]
    print(new_cols[:20])
    
    print("\n✅ Тест завершён!")

