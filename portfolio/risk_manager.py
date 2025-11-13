"""
Risk Manager

Класс для управления рисками портфеля.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class RiskManager:
    """
    Менеджер рисков для портфеля.
    
    Метрики:
    - Value at Risk (VaR)
    - Conditional VaR (CVaR) / Expected Shortfall
    - Maximum Drawdown
    - Beta и корреляция с рынком
    - Sharpe Ratio, Sortino Ratio
    """
    
    def __init__(self, returns: pd.DataFrame, benchmark_returns: pd.Series = None):
        """
        Инициализация менеджера рисков.
        
        Args:
            returns (pd.DataFrame): Доходности активов
            benchmark_returns (pd.Series): Доходности бенчмарка (опционально)
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        
    def value_at_risk(
        self,
        weights: Dict[str, float],
        confidence_level: float = 0.95,
        days: int = 1
    ) -> float:
        """
        Рассчитать Value at Risk (VaR).
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            confidence_level (float): Уровень доверия (0.95 = 95%)
            days (int): Горизонт (дней)
            
        Returns:
            float: VaR (отрицательное значение = потенциальный убыток)
        """
        # Преобразуем веса в массив
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.returns.columns])
        
        # Доходности портфеля
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Проверка на пустые данные
        if len(portfolio_returns) == 0 or portfolio_returns.isna().all():
            return 0.0
        
        # Удаляем NaN
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) == 0:
            return 0.0
        
        # VaR как квантиль распределения
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Масштабируем на горизонт
        var_scaled = var * np.sqrt(days)
        
        return var_scaled
    
    def conditional_var(
        self,
        weights: Dict[str, float],
        confidence_level: float = 0.95,
        days: int = 1
    ) -> float:
        """
        Рассчитать Conditional VaR (CVaR) / Expected Shortfall.
        
        CVaR - это средний убыток в худших случаях (хуже VaR).
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            confidence_level (float): Уровень доверия
            days (int): Горизонт (дней)
            
        Returns:
            float: CVaR
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.returns.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Проверка на пустые данные
        if len(portfolio_returns) == 0 or portfolio_returns.isna().all():
            return 0.0
        
        # Удаляем NaN
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) == 0:
            return 0.0
        
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # CVaR - среднее значение убытков хуже VaR
        tail_returns = portfolio_returns[portfolio_returns <= var]
        
        if len(tail_returns) == 0:
            return var  # Если нет данных в хвосте, возвращаем VaR
        
        cvar = tail_returns.mean()
        
        # Масштабируем на горизонт
        cvar_scaled = cvar * np.sqrt(days)
        
        return cvar_scaled
    
    def maximum_drawdown(self, weights: Dict[str, float]) -> Dict:
        """
        Рассчитать Maximum Drawdown.
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            
        Returns:
            Dict: Maximum Drawdown, дата начала, дата конца
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.returns.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Проверка на пустые данные
        if len(portfolio_returns) == 0 or portfolio_returns.isna().all():
            return {
                'max_drawdown': 0.0,
                'peak_date': None,
                'trough_date': None,
                'duration_days': 0
            }
        
        # Удаляем NaN
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) == 0:
            return {
                'max_drawdown': 0.0,
                'peak_date': None,
                'trough_date': None,
                'duration_days': 0
            }
        
        # Кумулятивная доходность
        cumulative = (1 + portfolio_returns).cumprod()
        
        # Running maximum
        running_max = cumulative.expanding().max()
        
        # Drawdown
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Находим начало просадки (последний максимум перед max_dd_date)
        # Используем .loc для правильного слайсинга по DatetimeIndex
        try:
            before_trough = cumulative.loc[:max_dd_date]
            if len(before_trough) > 0:
                peak_date = before_trough.idxmax()
                duration_days = (max_dd_date - peak_date).days
            else:
                peak_date = cumulative.index[0]
                duration_days = 0
        except Exception:
            peak_date = cumulative.index[0] if len(cumulative) > 0 else None
            duration_days = 0
        
        return {
            'max_drawdown': max_dd,
            'peak_date': peak_date,
            'trough_date': max_dd_date,
            'duration_days': duration_days
        }
    
    def sharpe_ratio(
        self,
        weights: Dict[str, float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Рассчитать Sharpe Ratio.
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            risk_free_rate (float): Безрисковая ставка (годовая)
            periods_per_year (int): Периодов в году
            
        Returns:
            float: Sharpe Ratio
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.returns.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        excess_returns = portfolio_returns - risk_free_rate / periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        
        return sharpe
    
    def sortino_ratio(
        self,
        weights: Dict[str, float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Рассчитать Sortino Ratio.
        
        Sortino Ratio учитывает только downside риск (отрицательные доходности).
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            risk_free_rate (float): Безрисковая ставка
            periods_per_year (int): Периодов в году
            
        Returns:
            float: Sortino Ratio
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.returns.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        excess_returns = portfolio_returns - risk_free_rate / periods_per_year
        
        # Downside deviation (только отрицательные доходности)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
        
        return sortino
    
    def calmar_ratio(
        self,
        weights: Dict[str, float],
        periods_per_year: int = 252
    ) -> float:
        """
        Рассчитать Calmar Ratio.
        
        Calmar Ratio = Годовая доходность / Maximum Drawdown
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            periods_per_year (int): Периодов в году
            
        Returns:
            float: Calmar Ratio
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.returns.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        annual_return = portfolio_returns.mean() * periods_per_year
        
        max_dd_info = self.maximum_drawdown(weights)
        max_dd = abs(max_dd_info['max_drawdown'])
        
        if max_dd == 0:
            return 0.0
        
        calmar = annual_return / max_dd
        
        return calmar
    
    def beta(self, weights: Dict[str, float]) -> float:
        """
        Рассчитать Beta портфеля относительно бенчмарка.
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            
        Returns:
            float: Beta
        """
        if self.benchmark_returns is None:
            return np.nan
        
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.returns.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Совмещаем данные
        combined = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        if len(combined) < 2:
            return np.nan
        
        # Ковариация / Дисперсия бенчмарка
        covariance = combined['portfolio'].cov(combined['benchmark'])
        benchmark_variance = combined['benchmark'].var()
        
        if benchmark_variance == 0:
            return np.nan
        
        beta = covariance / benchmark_variance
        
        return beta
    
    def correlation_with_benchmark(self, weights: Dict[str, float]) -> float:
        """
        Рассчитать корреляцию с бенчмарком.
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            
        Returns:
            float: Correlation
        """
        if self.benchmark_returns is None:
            return np.nan
        
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.returns.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Совмещаем данные
        combined = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        if len(combined) < 2:
            return np.nan
        
        correlation = combined['portfolio'].corr(combined['benchmark'])
        
        return correlation
    
    def calculate_all_metrics(
        self,
        weights: Dict[str, float],
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Рассчитать все метрики риска.
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            risk_free_rate (float): Безрисковая ставка
            confidence_level (float): Уровень доверия для VaR
            
        Returns:
            Dict: Все метрики
        """
        metrics = {
            'var_95': self.value_at_risk(weights, confidence_level=confidence_level),
            'cvar_95': self.conditional_var(weights, confidence_level=confidence_level),
            'sharpe_ratio': self.sharpe_ratio(weights, risk_free_rate),
            'sortino_ratio': self.sortino_ratio(weights, risk_free_rate),
            'calmar_ratio': self.calmar_ratio(weights),
        }
        
        # Maximum Drawdown
        max_dd_info = self.maximum_drawdown(weights)
        metrics.update({
            'max_drawdown': max_dd_info['max_drawdown'],
            'max_dd_duration': max_dd_info['duration_days']
        })
        
        # Beta и корреляция (если есть бенчмарк)
        if self.benchmark_returns is not None:
            metrics['beta'] = self.beta(weights)
            metrics['correlation'] = self.correlation_with_benchmark(weights)
        
        return metrics
    
    def print_risk_report(
        self,
        weights: Dict[str, float],
        risk_free_rate: float = 0.0,
        portfolio_value: float = 1000000
    ):
        """
        Вывести отчёт по рискам.
        
        Args:
            weights (Dict[str, float]): Веса портфеля
            risk_free_rate (float): Безрисковая ставка
            portfolio_value (float): Стоимость портфеля
        """
        metrics = self.calculate_all_metrics(weights, risk_free_rate)
        
        print(f"\n{'='*80}")
        print(f"🛡️  Отчёт по рискам портфеля")
        print(f"{'='*80}")
        print(f"Стоимость портфеля: {portfolio_value:,.2f} ₽")
        print(f"\n{'Метрика':<30} {'Значение':>20} {'В рублях':>25}")
        print(f"{'-'*80}")
        
        print(f"{'Value at Risk (95%)':<30} {metrics['var_95']:>19.2%} {metrics['var_95']*portfolio_value:>24,.2f} ₽")
        print(f"{'Conditional VaR (95%)':<30} {metrics['cvar_95']:>19.2%} {metrics['cvar_95']*portfolio_value:>24,.2f} ₽")
        print(f"{'Maximum Drawdown':<30} {metrics['max_drawdown']:>19.2%} {metrics['max_drawdown']*portfolio_value:>24,.2f} ₽")
        
        if 'max_dd_duration' in metrics:
            print(f"{'Max DD Duration':<30} {metrics['max_dd_duration']:>16} дней")
        
        print(f"\n{'Коэффициенты:'}")
        print(f"{'-'*80}")
        print(f"{'Sharpe Ratio':<30} {metrics['sharpe_ratio']:>20.3f}")
        print(f"{'Sortino Ratio':<30} {metrics['sortino_ratio']:>20.3f}")
        print(f"{'Calmar Ratio':<30} {metrics['calmar_ratio']:>20.3f}")
        
        if 'beta' in metrics and not np.isnan(metrics['beta']):
            print(f"\n{'Относительно рынка:'}")
            print(f"{'-'*80}")
            print(f"{'Beta':<30} {metrics['beta']:>20.3f}")
            print(f"{'Correlation':<30} {metrics['correlation']:>20.3f}")
        
        print(f"{'='*80}\n")

