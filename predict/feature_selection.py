"""
Feature Selection для ML моделей

Автоматический выбор лучших признаков:
- Корреляционный анализ (удаление высококоррелированных)
- Feature Importance (из tree-based моделей)
- Рекурсивное исключение признаков (RFE)
- Statistical tests (ANOVA, mutual information)
- Variance threshold (удаление константных)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """Автоматический выбор лучших признаков."""
    
    def __init__(
        self,
        methods: List[str] = None,
        n_features_to_select: Optional[int] = None,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01
    ):
        """
        Инициализация Feature Selector.
        
        Args:
            methods: Методы отбора ['correlation', 'variance', 'statistical', 'importance', 'rfe']
            n_features_to_select: Количество признаков для выбора (None = автоматически)
            correlation_threshold: Порог корреляции для удаления
            variance_threshold: Минимальная дисперсия
        """
        self.methods = methods or ['correlation', 'variance', 'importance']
        self.n_features_to_select = n_features_to_select
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self.removed_features: Dict[str, str] = {}  # feature -> reason
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> 'FeatureSelector':
        """
        Определить лучшие признаки.
        
        Args:
            X: Признаки
            y: Целевая переменная
            feature_names: Названия признаков (если X не DataFrame)
            
        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        print(f"\n🔍 Feature Selection:")
        print(f"   Исходных признаков: {X.shape[1]}")
        
        # Очистка данных перед отбором
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Проверка на очень большие значения
        for col in X.columns:
            if X[col].abs().max() > 1e15:
                X[col] = X[col].clip(-1e15, 1e15)
        
        print(f"   Данные очищены (inf, nan, outliers)")
        
        # Начинаем со всех признаков
        selected = X.columns.tolist()
        
        # 1. Удаляем признаки с низкой дисперсией
        if 'variance' in self.methods:
            selected = self._remove_low_variance(X[selected], selected)
        
        # 2. Удаляем высококоррелированные
        if 'correlation' in self.methods:
            selected = self._remove_correlated(X[selected], selected)
        
        # 3. Statistical tests (ANOVA, mutual information)
        if 'statistical' in self.methods:
            selected = self._statistical_selection(X[selected], y, selected)
        
        # 4. Feature Importance из Random Forest
        if 'importance' in self.methods:
            selected = self._importance_selection(X[selected], y, selected)
        
        # 5. Recursive Feature Elimination
        if 'rfe' in self.methods:
            selected = self._rfe_selection(X[selected], y, selected)
        
        self.selected_features = selected
        
        print(f"   Выбрано признаков: {len(self.selected_features)}")
        print(f"   Удалено признаков: {len(self.removed_features)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применить отбор к данным.
        
        Args:
            X: Признаки
            
        Returns:
            DataFrame с выбранными признаками
        """
        if not self.selected_features:
            raise ValueError("fit() must be called before transform()")
        
        return X[self.selected_features]
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Определить и применить отбор.
        
        Args:
            X: Признаки
            y: Целевая переменная
            feature_names: Названия признаков
            
        Returns:
            DataFrame с выбранными признаками
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def _remove_low_variance(
        self,
        X: pd.DataFrame,
        features: List[str]
    ) -> List[str]:
        """Удалить признаки с низкой дисперсией."""
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(X)
        
        selected = []
        for i, feature in enumerate(features):
            if selector.get_support()[i]:
                selected.append(feature)
            else:
                self.removed_features[feature] = "low_variance"
        
        return selected
    
    def _remove_correlated(
        self,
        X: pd.DataFrame,
        features: List[str]
    ) -> List[str]:
        """Удалить высококоррелированные признаки."""
        # Вычисляем корреляционную матрицу
        corr_matrix = X.corr().abs()
        
        # Верхний треугольник
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Находим признаки с корреляцией > threshold
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]
        
        for feature in to_drop:
            if feature in features:
                self.removed_features[feature] = f"high_correlation(>{self.correlation_threshold})"
        
        selected = [f for f in features if f not in to_drop]
        
        return selected
    
    def _statistical_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str]
    ) -> List[str]:
        """Статистический отбор (ANOVA F-test и Mutual Information)."""
        n_features = self.n_features_to_select or max(10, len(features) // 2)
        n_features = min(n_features, len(features))
        
        # F-score
        selector_f = SelectKBest(score_func=f_regression, k=n_features)
        selector_f.fit(X, y)
        
        # Mutual Information
        selector_mi = SelectKBest(score_func=mutual_info_regression, k=n_features)
        selector_mi.fit(X, y)
        
        # Комбинируем результаты
        scores_f = dict(zip(features, selector_f.scores_))
        scores_mi = dict(zip(features, selector_mi.scores_))
        
        # Нормализуем и усредняем
        max_f = max(scores_f.values())
        max_mi = max(scores_mi.values())
        
        combined_scores = {}
        for feature in features:
            score = (scores_f[feature] / max_f + scores_mi[feature] / max_mi) / 2
            combined_scores[feature] = score
            self.feature_scores[feature] = score
        
        # Выбираем топ N
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [f for f, _ in sorted_features[:n_features]]
        
        return selected
    
    def _importance_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str]
    ) -> List[str]:
        """Отбор по важности из Random Forest."""
        n_features = self.n_features_to_select or max(10, len(features) // 2)
        n_features = min(n_features, len(features))
        
        # Обучаем Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Получаем важность
        importances = dict(zip(features, rf.feature_importances_))
        
        # Обновляем scores
        for feature, importance in importances.items():
            if feature in self.feature_scores:
                # Усредняем со статистическими scores
                self.feature_scores[feature] = (
                    self.feature_scores[feature] + importance
                ) / 2
            else:
                self.feature_scores[feature] = importance
        
        # Выбираем топ N
        sorted_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        selected = [f for f, _ in sorted_features[:n_features]]
        
        return selected
    
    def _rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str]
    ) -> List[str]:
        """Рекурсивное исключение признаков."""
        n_features = self.n_features_to_select or max(10, len(features) // 3)
        n_features = min(n_features, len(features))
        
        # RFE с Random Forest
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        
        selector.fit(X, y)
        
        # Получаем выбранные признаки
        selected = [
            features[i] for i in range(len(features))
            if selector.support_[i]
        ]
        
        return selected
    
    def get_feature_scores(self, top_n: int = None) -> pd.DataFrame:
        """
        Получить scores признаков.
        
        Args:
            top_n: Показать топ N (None = все)
            
        Returns:
            DataFrame со scores
        """
        df = pd.DataFrame({
            'feature': list(self.feature_scores.keys()),
            'score': list(self.feature_scores.values())
        })
        
        df = df.sort_values('score', ascending=False)
        
        if top_n:
            df = df.head(top_n)
        
        return df
    
    def get_removed_features(self) -> pd.DataFrame:
        """Получить список удалённых признаков с причинами."""
        if not self.removed_features:
            return pd.DataFrame(columns=['feature', 'reason'])
        
        df = pd.DataFrame({
            'feature': list(self.removed_features.keys()),
            'reason': list(self.removed_features.values())
        })
        
        return df


def select_best_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: Optional[int] = None,
    methods: List[str] = None
) -> Tuple[pd.DataFrame, FeatureSelector]:
    """
    Удобная функция для быстрого отбора признаков.
    
    Args:
        X: Признаки
        y: Целевая переменная
        n_features: Количество признаков для выбора
        methods: Методы отбора
        
    Returns:
        (Отобранные признаки, Selector объект)
    """
    selector = FeatureSelector(
        methods=methods or ['correlation', 'variance', 'importance'],
        n_features_to_select=n_features
    )
    
    X_selected = selector.fit_transform(X, y)
    
    return X_selected, selector


if __name__ == "__main__":
    # Пример использования
    print("=" * 80)
    print("🧪 ТЕСТ FEATURE SELECTION")
    print("=" * 80)
    
    # Создаём тестовые данные
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Делаем некоторые признаки коррелированными
    X['feature_1'] = X['feature_0'] * 0.95 + np.random.randn(n_samples) * 0.1
    X['feature_2'] = X['feature_0'] * 0.98 + np.random.randn(n_samples) * 0.05
    
    # Создаём целевую переменную (зависит от первых 5 признаков)
    y = (
        X['feature_0'] * 2 +
        X['feature_3'] * 1.5 +
        X['feature_7'] * 1.2 +
        np.random.randn(n_samples) * 0.5
    )
    
    print(f"\nИсходные данные: {X.shape}")
    print(f"Целевая переменная: {y.shape}")
    
    # Отбираем признаки
    selector = FeatureSelector(
        methods=['correlation', 'variance', 'statistical', 'importance'],
        n_features_to_select=15
    )
    
    X_selected = selector.fit_transform(X, y)
    
    print(f"\nВыбрано признаков: {X_selected.shape[1]}")
    
    print("\n📊 Топ-10 признаков по score:")
    print(selector.get_feature_scores(top_n=10))
    
    print("\n❌ Удалённые признаки:")
    print(selector.get_removed_features().head())
    
    print("\n✅ Тест завершён!")

