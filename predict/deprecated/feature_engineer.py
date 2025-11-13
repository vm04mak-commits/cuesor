"""
Feature Engineering Module

Автоматическое создание признаков для ML моделей:
- Lag features (цены за N дней назад)
- Rolling statistics (MA, std, min, max)
- Technical indicators as features
- Feature selection (RFE)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import logging


class FeatureEngineer:
    """
    Автоматическое создание и отбор признаков для ML моделей.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Инициализация Feature Engineer.
        
        Args:
            logger: Логгер (опционально)
        """
        self.logger = logger
        self.feature_names = []
    
    # ========== LAG FEATURES ==========
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 7, 14, 21, 30]
    ) -> pd.DataFrame:
        """
        Создание lag features (значения за N дней назад).
        
        Args:
            df: DataFrame с данными
            columns: Список колонок для создания lag features
            lags: Список лагов (дней назад)
            
        Returns:
            DataFrame с новыми признаками
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                feature_name = f"{col}_lag_{lag}"
                df_result[feature_name] = df[col].shift(lag)
                self.feature_names.append(feature_name)
        
        if self.logger:
            self.logger.info(f"Created {len(columns) * len(lags)} lag features")
        
        return df_result
    
    # ========== ROLLING STATISTICS ==========
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20, 50],
        statistics: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Создание rolling statistics.
        
        Args:
            df: DataFrame с данными
            columns: Список колонок для rolling features
            windows: Размеры окон (дней)
            statistics: Список статистик ('mean', 'std', 'min', 'max', 'median')
            
        Returns:
            DataFrame с новыми признаками
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                for stat in statistics:
                    feature_name = f"{col}_roll_{window}_{stat}"
                    
                    if stat == 'mean':
                        df_result[feature_name] = df[col].rolling(window=window).mean()
                    elif stat == 'std':
                        df_result[feature_name] = df[col].rolling(window=window).std()
                    elif stat == 'min':
                        df_result[feature_name] = df[col].rolling(window=window).min()
                    elif stat == 'max':
                        df_result[feature_name] = df[col].rolling(window=window).max()
                    elif stat == 'median':
                        df_result[feature_name] = df[col].rolling(window=window).median()
                    
                    self.feature_names.append(feature_name)
        
        if self.logger:
            self.logger.info(f"Created {len(columns) * len(windows) * len(statistics)} rolling features")
        
        return df_result
    
    # ========== PRICE CHANGE FEATURES ==========
    
    def create_price_change_features(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        periods: List[int] = [1, 2, 3, 5, 7, 14, 21, 30]
    ) -> pd.DataFrame:
        """
        Создание признаков изменения цены.
        
        Args:
            df: DataFrame с данными
            price_column: Колонка с ценой
            periods: Периоды для расчёта изменения
            
        Returns:
            DataFrame с новыми признаками
        """
        df_result = df.copy()
        
        if price_column not in df.columns:
            return df_result
        
        for period in periods:
            # Процентное изменение
            feature_name = f"price_change_{period}d"
            df_result[feature_name] = df[price_column].pct_change(period)
            self.feature_names.append(feature_name)
            
            # Абсолютное изменение
            feature_name = f"price_diff_{period}d"
            df_result[feature_name] = df[price_column].diff(period)
            self.feature_names.append(feature_name)
        
        if self.logger:
            self.logger.info(f"Created {len(periods) * 2} price change features")
        
        return df_result
    
    # ========== VOLUME FEATURES ==========
    
    def create_volume_features(
        self,
        df: pd.DataFrame,
        volume_column: str = 'volume',
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Создание признаков объёма.
        
        Args:
            df: DataFrame с данными
            volume_column: Колонка с объёмом
            price_column: Колонка с ценой
            
        Returns:
            DataFrame с новыми признаками
        """
        df_result = df.copy()
        
        if volume_column not in df.columns:
            return df_result
        
        # Volume moving averages
        for window in [5, 10, 20]:
            feature_name = f"volume_ma_{window}"
            df_result[feature_name] = df[volume_column].rolling(window=window).mean()
            self.feature_names.append(feature_name)
        
        # Volume ratio
        feature_name = "volume_ratio_20"
        df_result[feature_name] = df[volume_column] / df[volume_column].rolling(window=20).mean()
        self.feature_names.append(feature_name)
        
        # Volume change
        for period in [1, 5]:
            feature_name = f"volume_change_{period}d"
            df_result[feature_name] = df[volume_column].pct_change(period)
            self.feature_names.append(feature_name)
        
        # Price * Volume (денежный объём)
        if price_column in df.columns:
            feature_name = "money_volume"
            df_result[feature_name] = df[price_column] * df[volume_column]
            self.feature_names.append(feature_name)
            
            # Money volume MA
            for window in [5, 20]:
                feature_name = f"money_volume_ma_{window}"
                df_result[feature_name] = df_result['money_volume'].rolling(window=window).mean()
                self.feature_names.append(feature_name)
        
        if self.logger:
            self.logger.info(f"Created volume features")
        
        return df_result
    
    # ========== VOLATILITY FEATURES ==========
    
    def create_volatility_features(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Создание признаков волатильности.
        
        Args:
            df: DataFrame с данными
            price_column: Колонка с ценой
            windows: Размеры окон
            
        Returns:
            DataFrame с новыми признаками
        """
        df_result = df.copy()
        
        if price_column not in df.columns:
            return df_result
        
        # Рассчитываем доходности
        returns = df[price_column].pct_change()
        
        for window in windows:
            # Volatility (std of returns)
            feature_name = f"volatility_{window}"
            df_result[feature_name] = returns.rolling(window=window).std()
            self.feature_names.append(feature_name)
            
            # Parkinson's volatility (using high/low)
            if 'high' in df.columns and 'low' in df.columns:
                feature_name = f"parkinson_vol_{window}"
                hl = np.log(df['high'] / df['low'])
                df_result[feature_name] = np.sqrt((hl ** 2) / (4 * np.log(2)))
                df_result[feature_name] = df_result[feature_name].rolling(window=window).mean()
                self.feature_names.append(feature_name)
        
        if self.logger:
            self.logger.info(f"Created volatility features")
        
        return df_result
    
    # ========== MOMENTUM FEATURES ==========
    
    def create_momentum_features(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        periods: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Создание признаков momentum.
        
        Args:
            df: DataFrame с данными
            price_column: Колонка с ценой
            periods: Периоды для расчёта
            
        Returns:
            DataFrame с новыми признаками
        """
        df_result = df.copy()
        
        if price_column not in df.columns:
            return df_result
        
        for period in periods:
            # ROC (Rate of Change)
            feature_name = f"roc_{period}"
            df_result[feature_name] = ((df[price_column] - df[price_column].shift(period)) / 
                                       df[price_column].shift(period) * 100)
            self.feature_names.append(feature_name)
            
            # Momentum
            feature_name = f"momentum_{period}"
            df_result[feature_name] = df[price_column] - df[price_column].shift(period)
            self.feature_names.append(feature_name)
        
        if self.logger:
            self.logger.info(f"Created momentum features")
        
        return df_result
    
    # ========== ALL FEATURES ==========
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        volume_column: str = 'volume'
    ) -> pd.DataFrame:
        """
        Создание всех признаков автоматически.
        
        Args:
            df: DataFrame с данными
            price_column: Колонка с ценой
            volume_column: Колонка с объёмом
            
        Returns:
            DataFrame со всеми признаками
        """
        print("\n" + "="*80)
        print("🔧 FEATURE ENGINEERING")
        print("="*80)
        print()
        
        self.feature_names = []
        df_result = df.copy()
        
        # Lag features
        print("1️⃣  Создание lag features...")
        df_result = self.create_lag_features(
            df_result,
            columns=[price_column, volume_column],
            lags=[1, 2, 3, 5, 7]
        )
        
        # Rolling features
        print("2️⃣  Создание rolling features...")
        df_result = self.create_rolling_features(
            df_result,
            columns=[price_column, volume_column],
            windows=[5, 10, 20],
            statistics=['mean', 'std']
        )
        
        # Price change features
        print("3️⃣  Создание price change features...")
        df_result = self.create_price_change_features(
            df_result,
            price_column=price_column,
            periods=[1, 2, 3, 5, 7, 14]
        )
        
        # Volume features
        print("4️⃣  Создание volume features...")
        df_result = self.create_volume_features(
            df_result,
            volume_column=volume_column,
            price_column=price_column
        )
        
        # Volatility features
        print("5️⃣  Создание volatility features...")
        df_result = self.create_volatility_features(
            df_result,
            price_column=price_column,
            windows=[5, 10, 20]
        )
        
        # Momentum features
        print("6️⃣  Создание momentum features...")
        df_result = self.create_momentum_features(
            df_result,
            price_column=price_column,
            periods=[5, 10, 20]
        )
        
        print()
        print(f"✅ Создано {len(self.feature_names)} новых признаков")
        print()
        
        return df_result
    
    # ========== FEATURE SELECTION ==========
    
    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 50,
        step: int = 1
    ) -> Tuple[List[str], np.ndarray]:
        """
        Отбор признаков с помощью Recursive Feature Elimination.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            n_features: Количество признаков для отбора
            step: Шаг удаления признаков
            
        Returns:
            Tuple[List[str], np.ndarray]: Список выбранных признаков и маска
        """
        print("\n" + "="*80)
        print("🎯 FEATURE SELECTION (RFE)")
        print("="*80)
        print()
        print(f"Исходных признаков: {X.shape[1]}")
        print(f"Целевое количество: {n_features}")
        print()
        
        # Базовая модель для RFE
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # RFE
        print("Запуск Recursive Feature Elimination...")
        selector = RFE(estimator, n_features_to_select=n_features, step=step)
        selector.fit(X, y)
        
        # Получаем выбранные признаки
        selected_features = X.columns[selector.support_].tolist()
        
        print(f"✅ Отобрано {len(selected_features)} признаков")
        print()
        
        # Показываем топ-20
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_,
            'selected': selector.support_
        }).sort_values('ranking')
        
        print("Топ-20 признаков:")
        for idx, row in feature_ranking.head(20).iterrows():
            status = "✅" if row['selected'] else "❌"
            print(f"   {status} {row['feature']:<40} (rank: {row['ranking']})")
        
        print()
        
        return selected_features, selector.support_
    
    def select_features_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Отбор признаков по важности (feature importance).
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            threshold: Порог важности
            
        Returns:
            List[str]: Список выбранных признаков
        """
        print("\n" + "="*80)
        print("🎯 FEATURE SELECTION (Importance)")
        print("="*80)
        print()
        print(f"Исходных признаков: {X.shape[1]}")
        print(f"Порог важности: {threshold}")
        print()
        
        # Обучаем модель
        print("Обучение модели для оценки важности...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Получаем важности
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Отбираем признаки
        selected_features = importances[importances['importance'] >= threshold]['feature'].tolist()
        
        print(f"✅ Отобрано {len(selected_features)} признаков (важность >= {threshold})")
        print()
        
        # Показываем топ-20
        print("Топ-20 самых важных признаков:")
        for idx, row in importances.head(20).iterrows():
            status = "✅" if row['importance'] >= threshold else "❌"
            print(f"   {status} {row['feature']:<40} {row['importance']:.4f}")
        
        print()
        
        return selected_features
    
    # ========== УТИЛИТЫ ==========
    
    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Получить статистику по признакам.
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            DataFrame со статистикой
        """
        stats = pd.DataFrame({
            'feature': df.columns,
            'dtype': df.dtypes,
            'missing': df.isnull().sum(),
            'missing_pct': df.isnull().sum() / len(df) * 100,
            'unique': df.nunique(),
            'mean': df.mean(numeric_only=True),
            'std': df.std(numeric_only=True),
            'min': df.min(numeric_only=True),
            'max': df.max(numeric_only=True)
        })
        
        return stats
    
    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Удалить сильно коррелированные признаки.
        
        Args:
            df: DataFrame с признаками
            threshold: Порог корреляции
            
        Returns:
            Tuple[DataFrame, List[str]]: Очищенный DataFrame и список удалённых признаков
        """
        print("\n" + "="*80)
        print("🔍 УДАЛЕНИЕ КОРРЕЛИРОВАННЫХ ПРИЗНАКОВ")
        print("="*80)
        print()
        print(f"Порог корреляции: {threshold}")
        print()
        
        # Вычисляем корреляцию
        corr_matrix = df.corr().abs()
        
        # Находим пары с высокой корреляцией
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Признаки для удаления
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        print(f"Найдено {len(to_drop)} коррелированных признаков для удаления")
        
        if to_drop:
            print("\nПримеры коррелированных пар:")
            count = 0
            for column in to_drop[:5]:
                corr_with = upper[column][upper[column] > threshold].index.tolist()
                for corr_col in corr_with[:1]:
                    corr_val = upper.loc[corr_col, column]
                    print(f"   {column} ↔ {corr_col}: {corr_val:.3f}")
                    count += 1
                    if count >= 5:
                        break
        
        # Удаляем
        df_result = df.drop(columns=to_drop)
        
        print()
        print(f"✅ Осталось {df_result.shape[1]} признаков (было {df.shape[1]})")
        print()
        
        return df_result, to_drop






