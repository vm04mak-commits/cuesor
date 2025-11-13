"""
Модуль корреляционного анализа.
Анализ взаимосвязей между различными акциями.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class CorrelationAnalyzer:
    """
    Класс для корреляционного анализа акций.
    """
    
    def __init__(self, logger):
        """
        Инициализация анализатора корреляций.
        
        Args:
            logger: Объект логгера
        """
        self.logger = logger
    
    def calculate_correlation_matrix(self, stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Расчёт корреляционной матрицы для нескольких акций.
        
        Args:
            stocks_data (Dict[str, pd.DataFrame]): Словарь {тикер: данные}
        
        Returns:
            pd.DataFrame: Корреляционная матрица
        """
        self.logger.info(f"Расчёт корреляционной матрицы для {len(stocks_data)} акций")
        
        # Создаём DataFrame с ценами закрытия
        prices = pd.DataFrame()
        for ticker, data in stocks_data.items():
            if not data.empty and 'close' in data.columns:
                prices[ticker] = data['close'].values
        
        if prices.empty:
            self.logger.warning("Нет данных для расчёта корреляций")
            return pd.DataFrame()
        
        # Расчёт корреляционной матрицы
        correlation_matrix = prices.corr()
        
        return correlation_matrix
    
    def calculate_returns_correlation(self, stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Расчёт корреляции доходностей акций.
        
        Args:
            stocks_data (Dict[str, pd.DataFrame]): Словарь {тикер: данные}
        
        Returns:
            pd.DataFrame: Корреляционная матрица доходностей
        """
        self.logger.info(f"Расчёт корреляции доходностей для {len(stocks_data)} акций")
        
        # Создаём DataFrame с доходностями
        returns = pd.DataFrame()
        for ticker, data in stocks_data.items():
            if not data.empty and 'close' in data.columns:
                returns[ticker] = data['close'].pct_change().values
        
        if returns.empty:
            self.logger.warning("Нет данных для расчёта корреляций доходностей")
            return pd.DataFrame()
        
        # Расчёт корреляционной матрицы
        correlation_matrix = returns.corr()
        
        return correlation_matrix
    
    def find_highly_correlated_pairs(self, correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Поиск пар акций с высокой корреляцией.
        
        Args:
            correlation_matrix (pd.DataFrame): Корреляционная матрица
            threshold (float): Порог корреляции (0-1)
        
        Returns:
            List[Tuple[str, str, float]]: Список пар (тикер1, тикер2, корреляция)
        """
        self.logger.info(f"Поиск пар с корреляцией > {threshold}")
        
        high_corr_pairs = []
        
        # Перебираем все пары
        tickers = correlation_matrix.columns
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker1 = tickers[i]
                ticker2 = tickers[j]
                corr_value = correlation_matrix.loc[ticker1, ticker2]
                
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((ticker1, ticker2, float(corr_value)))
        
        # Сортируем по убыванию корреляции
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        self.logger.info(f"Найдено {len(high_corr_pairs)} пар с высокой корреляцией")
        return high_corr_pairs
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Расчёт бета-коэффициента (чувствительность акции к рынку).
        
        Args:
            stock_returns (pd.Series): Доходности акции
            market_returns (pd.Series): Доходности рынка (индекса)
        
        Returns:
            float: Бета-коэффициент
        """
        # Убираем NaN
        valid_data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(valid_data) < 2:
            self.logger.warning("Недостаточно данных для расчёта бета")
            return 0.0
        
        # Ковариация и дисперсия
        covariance = valid_data['stock'].cov(valid_data['market'])
        market_variance = valid_data['market'].var()
        
        if market_variance == 0:
            return 0.0
        
        beta = covariance / market_variance
        
        self.logger.info(f"Бета-коэффициент: {beta:.3f}")
        return float(beta)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Расчёт коэффициента Шарпа (доходность с учётом риска).
        
        Args:
            returns (pd.Series): Доходности акции
            risk_free_rate (float): Безрисковая ставка (годовая)
        
        Returns:
            float: Коэффициент Шарпа
        """
        # Убираем NaN
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0 or clean_returns.std() == 0:
            return 0.0
        
        # Средняя доходность
        mean_return = clean_returns.mean()
        
        # Приводим безрисковую ставку к периоду данных (обычно дневную)
        daily_risk_free_rate = risk_free_rate / 252
        
        # Избыточная доходность
        excess_return = mean_return - daily_risk_free_rate
        
        # Стандартное отклонение
        std_dev = clean_returns.std()
        
        # Коэффициент Шарпа
        sharpe_ratio = excess_return / std_dev
        
        # Аннуализируем (умножаем на корень из 252 торговых дней)
        annual_sharpe = sharpe_ratio * np.sqrt(252)
        
        self.logger.info(f"Коэффициент Шарпа: {annual_sharpe:.3f}")
        return float(annual_sharpe)









