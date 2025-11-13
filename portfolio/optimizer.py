"""
Portfolio Optimizer

Класс для оптимизации инвестиционного портфеля по Modern Portfolio Theory.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Оптимизатор портфеля на основе Modern Portfolio Theory (Марковиц).
    
    Методы оптимизации:
    - Max Sharpe Ratio
    - Min Variance
    - Efficient Frontier
    - Equal Weight
    - Risk Parity
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0):
        """
        Инициализация оптимизатора.
        
        Args:
            returns (pd.DataFrame): DataFrame с доходностями акций (тикеры в колонках)
            risk_free_rate (float): Безрисковая ставка (годовая)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean() * 252  # Годовая доходность
        self.cov_matrix = returns.cov() * 252  # Годовая ковариация
        self.tickers = list(returns.columns)
        self.num_assets = len(self.tickers)
        
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Рассчитать производительность портфеля.
        
        Args:
            weights (np.ndarray): Веса активов
            
        Returns:
            Tuple[float, float, float]: (доходность, волатильность, Sharpe Ratio)
        """
        returns = np.sum(self.mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / std if std > 0 else 0
        
        return returns, std, sharpe
    
    def negative_sharpe(self, weights: np.ndarray) -> float:
        """
        Отрицательный Sharpe Ratio (для минимизации).
        
        Args:
            weights (np.ndarray): Веса активов
            
        Returns:
            float: -Sharpe Ratio
        """
        _, _, sharpe = self.portfolio_performance(weights)
        return -sharpe
    
    def portfolio_variance(self, weights: np.ndarray) -> float:
        """
        Дисперсия портфеля.
        
        Args:
            weights (np.ndarray): Веса активов
            
        Returns:
            float: Дисперсия
        """
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def max_sharpe_portfolio(self, target_return: float = None) -> Dict:
        """
        Найти портфель с максимальным Sharpe Ratio.
        
        Args:
            target_return (float): Целевая доходность (опционально)
            
        Returns:
            Dict: Веса, доходность, риск, Sharpe Ratio
        """
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(self.mean_returns * x) - target_return
            })
        
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            self.negative_sharpe,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError(f"Оптимизация не удалась: {result.message}")
        
        weights = result.x
        returns, std, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': dict(zip(self.tickers, weights)),
            'expected_return': returns,
            'volatility': std,
            'sharpe_ratio': sharpe,
            'optimization': 'max_sharpe'
        }
    
    def min_variance_portfolio(self, target_return: float = None) -> Dict:
        """
        Найти портфель с минимальной волатильностью.
        
        Args:
            target_return (float): Целевая доходность (опционально)
            
        Returns:
            Dict: Веса, доходность, риск, Sharpe Ratio
        """
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(self.mean_returns * x) - target_return
            })
        
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            self.portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError(f"Оптимизация не удалась: {result.message}")
        
        weights = result.x
        returns, std, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': dict(zip(self.tickers, weights)),
            'expected_return': returns,
            'volatility': std,
            'sharpe_ratio': sharpe,
            'optimization': 'min_variance'
        }
    
    def efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Построить Efficient Frontier.
        
        Args:
            num_portfolios (int): Количество точек на границе
            
        Returns:
            pd.DataFrame: Портфели на эффективной границе
        """
        # Находим диапазон доходностей
        min_var_portfolio = self.min_variance_portfolio()
        max_sharpe_portfolio = self.max_sharpe_portfolio()
        
        min_return = min_var_portfolio['expected_return']
        max_return = self.mean_returns.max()
        
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        frontier_portfolios = []
        
        for target_return in target_returns:
            try:
                portfolio = self.min_variance_portfolio(target_return=target_return)
                frontier_portfolios.append({
                    'return': portfolio['expected_return'],
                    'volatility': portfolio['volatility'],
                    'sharpe_ratio': portfolio['sharpe_ratio'],
                    **portfolio['weights']
                })
            except:
                continue
        
        return pd.DataFrame(frontier_portfolios)
    
    def equal_weight_portfolio(self) -> Dict:
        """
        Создать портфель с равными весами.
        
        Returns:
            Dict: Веса, доходность, риск, Sharpe Ratio
        """
        weights = np.array([1/self.num_assets] * self.num_assets)
        returns, std, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': dict(zip(self.tickers, weights)),
            'expected_return': returns,
            'volatility': std,
            'sharpe_ratio': sharpe,
            'optimization': 'equal_weight'
        }
    
    def risk_parity_portfolio(self) -> Dict:
        """
        Создать портфель на основе Risk Parity (равный вклад в риск).
        
        Returns:
            Dict: Веса, доходность, риск, Sharpe Ratio
        """
        def risk_parity_objective(weights):
            # Вклад каждого актива в риск портфеля
            portfolio_vol = np.sqrt(self.portfolio_variance(weights))
            marginal_contrib = np.dot(self.cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_vol
            
            # Минимизируем разброс вкладов в риск
            return np.sum((contrib - contrib.mean()) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            risk_parity_objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            # Если не удалось, возвращаем равные веса
            return self.equal_weight_portfolio()
        
        weights = result.x
        returns, std, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': dict(zip(self.tickers, weights)),
            'expected_return': returns,
            'volatility': std,
            'sharpe_ratio': sharpe,
            'optimization': 'risk_parity'
        }
    
    def monte_carlo_portfolios(self, num_portfolios: int = 10000) -> pd.DataFrame:
        """
        Генерация случайных портфелей (Monte Carlo симуляция).
        
        Args:
            num_portfolios (int): Количество портфелей
            
        Returns:
            pd.DataFrame: Случайные портфели
        """
        results = []
        
        for _ in range(num_portfolios):
            # Генерируем случайные веса
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            
            returns, std, sharpe = self.portfolio_performance(weights)
            
            results.append({
                'return': returns,
                'volatility': std,
                'sharpe_ratio': sharpe,
                **dict(zip(self.tickers, weights))
            })
        
        return pd.DataFrame(results)
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Сравнить различные стратегии оптимизации.
        
        Returns:
            pd.DataFrame: Сравнение стратегий
        """
        strategies = []
        
        try:
            max_sharpe = self.max_sharpe_portfolio()
            strategies.append({
                'strategy': 'Max Sharpe',
                'return': max_sharpe['expected_return'],
                'volatility': max_sharpe['volatility'],
                'sharpe_ratio': max_sharpe['sharpe_ratio']
            })
        except Exception as e:
            print(f"⚠️  Max Sharpe не удалось: {e}")
        
        try:
            min_var = self.min_variance_portfolio()
            strategies.append({
                'strategy': 'Min Variance',
                'return': min_var['expected_return'],
                'volatility': min_var['volatility'],
                'sharpe_ratio': min_var['sharpe_ratio']
            })
        except Exception as e:
            print(f"⚠️  Min Variance не удалось: {e}")
        
        try:
            equal_weight = self.equal_weight_portfolio()
            strategies.append({
                'strategy': 'Equal Weight',
                'return': equal_weight['expected_return'],
                'volatility': equal_weight['volatility'],
                'sharpe_ratio': equal_weight['sharpe_ratio']
            })
        except Exception as e:
            print(f"⚠️  Equal Weight не удалось: {e}")
        
        try:
            risk_parity = self.risk_parity_portfolio()
            strategies.append({
                'strategy': 'Risk Parity',
                'return': risk_parity['expected_return'],
                'volatility': risk_parity['volatility'],
                'sharpe_ratio': risk_parity['sharpe_ratio']
            })
        except Exception as e:
            print(f"⚠️  Risk Parity не удалось: {e}")
        
        if not strategies:
            raise ValueError("Ни одна стратегия не сработала")
        
        df = pd.DataFrame(strategies)
        df = df.sort_values('sharpe_ratio', ascending=False)
        
        return df
    
    def optimize_with_constraints(
        self,
        method: str = 'max_sharpe',
        max_weight: float = 0.3,
        min_weight: float = 0.05
    ) -> Dict:
        """
        Оптимизация с ограничениями на веса.
        
        Args:
            method (str): Метод оптимизации ('max_sharpe' или 'min_variance')
            max_weight (float): Максимальный вес одного актива
            min_weight (float): Минимальный вес одного актива
            
        Returns:
            Dict: Оптимизированный портфель
        """
        if method == 'max_sharpe':
            objective = self.negative_sharpe
        elif method == 'min_variance':
            objective = self.portfolio_variance
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((min_weight, max_weight) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError(f"Оптимизация не удалась: {result.message}")
        
        weights = result.x
        returns, std, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': dict(zip(self.tickers, weights)),
            'expected_return': returns,
            'volatility': std,
            'sharpe_ratio': sharpe,
            'optimization': f'{method}_constrained',
            'constraints': {'max_weight': max_weight, 'min_weight': min_weight}
        }









