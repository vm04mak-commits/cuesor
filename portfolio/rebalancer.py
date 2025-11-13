"""
Portfolio Rebalancer

Модуль для автоматической ребалансировки портфеля.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class PortfolioRebalancer:
    """
    Класс для ребалансировки портфеля.
    
    Методы:
    - Threshold-based rebalancing
    - Time-based rebalancing
    - Учёт транзакционных издержек
    - Tax-aware rebalancing
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0005,  # 0.05% комиссия
        tax_rate: float = 0.13,           # 13% налог на прибыль
        min_trade_value: float = 1000.0   # Минимальная сумма сделки
    ):
        """
        Инициализация ребалансировщика.
        
        Args:
            commission_rate (float): Комиссия брокера (доля от суммы сделки)
            tax_rate (float): Налог на прибыль (доля от прибыли)
            min_trade_value (float): Минимальная сумма сделки в рублях
        """
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.min_trade_value = min_trade_value
    
    def check_threshold_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Проверка необходимости ребалансировки по порогу отклонения.
        
        Args:
            current_weights (Dict[str, float]): Текущие веса
            target_weights (Dict[str, float]): Целевые веса
            threshold (float): Порог отклонения (0.05 = 5%)
            
        Returns:
            Tuple[bool, Dict[str, float]]: (нужна_ребалансировка, отклонения)
        """
        deviations = {}
        needs_rebalance = False
        
        # Все уникальные тикеры
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            deviation = abs(current - target)
            deviations[ticker] = deviation
            
            if deviation > threshold:
                needs_rebalance = True
        
        return needs_rebalance, deviations
    
    def check_time_rebalance(
        self,
        last_rebalance_date: datetime,
        rebalance_period_days: int = 90
    ) -> bool:
        """
        Проверка необходимости ребалансировки по времени.
        
        Args:
            last_rebalance_date (datetime): Дата последней ребалансировки
            rebalance_period_days (int): Период ребалансировки (дней)
            
        Returns:
            bool: Нужна ли ребалансировка
        """
        days_since_rebalance = (datetime.now() - last_rebalance_date).days
        return days_since_rebalance >= rebalance_period_days
    
    def calculate_rebalance_trades(
        self,
        current_positions: Dict[str, Dict],  # {ticker: {shares, price, value}}
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, Dict]:
        """
        Рассчитать необходимые сделки для ребалансировки.
        
        Args:
            current_positions: Текущие позиции
            target_weights: Целевые веса
            current_prices: Текущие цены
            portfolio_value: Общая стоимость портфеля
            
        Returns:
            Dict: Рекомендуемые сделки {ticker: {action, shares, value, commission}}
        """
        trades = {}
        
        # Все уникальные тикеры
        all_tickers = set(current_positions.keys()) | set(target_weights.keys())
        
        for ticker in all_tickers:
            if ticker not in current_prices:
                continue
            
            # Текущее состояние
            current_shares = current_positions.get(ticker, {}).get('shares', 0)
            current_value = current_shares * current_prices[ticker]
            
            # Целевое состояние
            target_weight = target_weights.get(ticker, 0.0)
            target_value = portfolio_value * target_weight
            
            # Разница
            value_diff = target_value - current_value
            
            # Пропускаем малые сделки
            if abs(value_diff) < self.min_trade_value:
                continue
            
            # Рассчитываем количество акций
            if value_diff > 0:
                # Покупка
                shares_to_buy = int(value_diff / current_prices[ticker])
                if shares_to_buy > 0:
                    actual_value = shares_to_buy * current_prices[ticker]
                    commission = actual_value * self.commission_rate
                    
                    trades[ticker] = {
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_prices[ticker],
                        'value': actual_value,
                        'commission': commission,
                        'total_cost': actual_value + commission
                    }
            else:
                # Продажа
                shares_to_sell = int(abs(value_diff) / current_prices[ticker])
                shares_to_sell = min(shares_to_sell, current_shares)  # Не больше, чем есть
                
                if shares_to_sell > 0:
                    actual_value = shares_to_sell * current_prices[ticker]
                    commission = actual_value * self.commission_rate
                    
                    # Налог на прибыль
                    avg_price = current_positions[ticker].get('avg_price', current_prices[ticker])
                    profit = (current_prices[ticker] - avg_price) * shares_to_sell
                    tax = max(0, profit * self.tax_rate)  # Налог только с прибыли
                    
                    trades[ticker] = {
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_prices[ticker],
                        'value': actual_value,
                        'commission': commission,
                        'tax': tax,
                        'net_proceeds': actual_value - commission - tax
                    }
        
        return trades
    
    def calculate_rebalance_cost(self, trades: Dict[str, Dict]) -> Dict[str, float]:
        """
        Рассчитать общую стоимость ребалансировки.
        
        Args:
            trades: Словарь сделок
            
        Returns:
            Dict: Стоимости {total_commission, total_tax, total_cost}
        """
        total_commission = sum(trade['commission'] for trade in trades.values())
        total_tax = sum(trade.get('tax', 0) for trade in trades.values())
        
        buy_value = sum(
            trade['total_cost'] 
            for trade in trades.values() 
            if trade['action'] == 'BUY'
        )
        
        sell_proceeds = sum(
            trade['net_proceeds'] 
            for trade in trades.values() 
            if trade['action'] == 'SELL'
        )
        
        net_cost = buy_value - sell_proceeds
        
        return {
            'total_commission': total_commission,
            'total_tax': total_tax,
            'buy_value': buy_value,
            'sell_proceeds': sell_proceeds,
            'net_cost': net_cost,
            'total_cost': total_commission + total_tax
        }
    
    def optimize_rebalance_with_costs(
        self,
        trades: Dict[str, Dict],
        expected_return: float,
        time_horizon_days: int = 365
    ) -> Tuple[bool, str]:
        """
        Определить, стоит ли ребалансировать с учётом издержек.
        
        Args:
            trades: Словарь сделок
            expected_return: Ожидаемая годовая доходность (доля)
            time_horizon_days: Горизонт инвестирования (дней)
            
        Returns:
            Tuple[bool, str]: (стоит_ребалансировать, причина)
        """
        costs = self.calculate_rebalance_cost(trades)
        
        # Оцениваем стоимость портфеля из сделок
        total_value = costs['buy_value'] + costs['sell_proceeds']
        
        if total_value == 0:
            return False, "Недостаточно данных для оценки"
        
        # Относительная стоимость ребалансировки
        relative_cost = costs['total_cost'] / total_value
        
        # Ожидаемая доходность за период
        period_return = expected_return * (time_horizon_days / 365)
        
        # Если издержки < 10% от ожидаемой доходности, стоит ребалансировать
        if relative_cost < period_return * 0.1:
            return True, f"Издержки ({relative_cost:.2%}) < 10% от ожидаемой доходности ({period_return:.2%})"
        else:
            return False, f"Издержки ({relative_cost:.2%}) слишком высоки относительно ожидаемой доходности ({period_return:.2%})"
    
    def generate_rebalance_report(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        trades: Dict[str, Dict],
        deviations: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Создать отчёт по ребалансировке.
        
        Args:
            current_weights: Текущие веса
            target_weights: Целевые веса
            trades: Сделки
            deviations: Отклонения
            
        Returns:
            pd.DataFrame: Отчёт
        """
        report_data = []
        
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        for ticker in sorted(all_tickers):
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            deviation = deviations.get(ticker, 0.0)
            
            trade = trades.get(ticker, {})
            action = trade.get('action', '-')
            shares = trade.get('shares', 0)
            trade_value = trade.get('value', 0)
            
            report_data.append({
                'Тикер': ticker,
                'Текущий вес': current,
                'Целевой вес': target,
                'Отклонение': deviation,
                'Действие': action,
                'Акций': shares,
                'Сумма': trade_value
            })
        
        df = pd.DataFrame(report_data)
        df = df.sort_values('Отклонение', ascending=False)
        
        return df
    
    def tax_loss_harvesting(
        self,
        current_positions: Dict[str, Dict],
        current_prices: Dict[str, float],
        min_loss_percent: float = 0.10
    ) -> List[Dict]:
        """
        Определить позиции для tax loss harvesting (продажа убыточных позиций для снижения налогов).
        
        Args:
            current_positions: Текущие позиции
            current_prices: Текущие цены
            min_loss_percent: Минимальный убыток для рассмотрения (10%)
            
        Returns:
            List[Dict]: Рекомендации по tax loss harvesting
        """
        recommendations = []
        
        for ticker, position in current_positions.items():
            if ticker not in current_prices:
                continue
            
            avg_price = position.get('avg_price', 0)
            current_price = current_prices[ticker]
            shares = position.get('shares', 0)
            
            if avg_price == 0:
                continue
            
            # Рассчитываем убыток
            loss_percent = (current_price - avg_price) / avg_price
            
            if loss_percent < -min_loss_percent:
                loss_value = (current_price - avg_price) * shares
                tax_benefit = abs(loss_value) * self.tax_rate
                
                recommendations.append({
                    'ticker': ticker,
                    'shares': shares,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'loss_percent': loss_percent,
                    'loss_value': loss_value,
                    'tax_benefit': tax_benefit
                })
        
        # Сортируем по убытку
        recommendations.sort(key=lambda x: x['loss_value'])
        
        return recommendations









