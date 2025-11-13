"""
Portfolio Management Module

Модуль для управления инвестиционным портфелем:
- Создание и управление портфелем
- Портфельная оптимизация (Modern Portfolio Theory)
- Risk management (VaR, Sharpe Ratio, etc.)
- Автоматическая ребалансировка
- Автоматическая торговля с ML моделями
"""

from .portfolio import Portfolio
from .optimizer import PortfolioOptimizer
from .risk_manager import RiskManager
from .rebalancer import PortfolioRebalancer
from .trader import AutoTrader, TradingSignal, Trade, Position

__all__ = [
    'Portfolio', 
    'PortfolioOptimizer', 
    'RiskManager', 
    'PortfolioRebalancer',
    'AutoTrader',
    'TradingSignal',
    'Trade',
    'Position'
]

