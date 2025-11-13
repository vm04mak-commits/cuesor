"""
Automatic Trading System

Автоматическая система принятия торговых решений на основе ML прогнозов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pickle

from core.logger import Logger


class TradingSignal:
    """Торговый сигнал."""
    
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Trade:
    """Запись о сделке."""
    
    def __init__(self, ticker: str, action: str, price: float, quantity: int, 
                 timestamp: datetime, commission: float = 0.0):
        self.ticker = ticker
        self.action = action  # BUY или SELL
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
        self.commission = commission
        self.value = price * quantity + commission
    
    def to_dict(self) -> dict:
        """Конвертировать в словарь."""
        return {
            'ticker': self.ticker,
            'action': self.action,
            'price': self.price,
            'quantity': self.quantity,
            'timestamp': self.timestamp,
            'commission': self.commission,
            'value': self.value
        }


class Position:
    """Открытая позиция."""
    
    def __init__(self, ticker: str, quantity: int, entry_price: float, 
                 entry_date: datetime):
        self.ticker = ticker
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_price = entry_price
        self.current_value = entry_price * quantity
        self.pnl = 0.0
        self.pnl_percent = 0.0
    
    def update(self, current_price: float):
        """Обновить текущую цену."""
        self.current_price = current_price
        self.current_value = current_price * self.quantity
        self.pnl = self.current_value - (self.entry_price * self.quantity)
        self.pnl_percent = (self.pnl / (self.entry_price * self.quantity)) * 100
    
    def to_dict(self) -> dict:
        """Конвертировать в словарь."""
        return {
            'ticker': self.ticker,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date,
            'current_price': self.current_price,
            'current_value': self.current_value,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent
        }


class AutoTrader:
    """
    Автоматическая торговая система.
    
    Принимает решения на основе прогнозов ML модели:
    - Покупка при прогнозе роста > порога
    - Продажа при прогнозе падения > порога
    - Удержание в остальных случаях
    
    Учитывает:
    - Баланс и риски
    - Комиссии
    - Максимальное количество позиций
    - Stop-loss и Take-profit
    """
    
    def __init__(
        self, 
        initial_balance: float = 100000.0,
        commission_rate: float = 0.003,  # 0.3%
        max_positions: int = 10,
        max_position_size: float = 0.15,  # 15% капитала на позицию
        buy_threshold: float = 0.02,  # Покупка при прогнозе роста > 2%
        sell_threshold: float = -0.01,  # Продажа при прогнозе падения > 1%
        stop_loss: float = -0.10,  # Stop-loss -10%
        take_profit: float = 0.25,  # Take-profit +25%
        logger: Optional[Logger] = None
    ):
        """
        Инициализация торговой системы.
        
        Args:
            initial_balance: Начальный капитал
            commission_rate: Комиссия брокера (доля)
            max_positions: Макс. количество позиций
            max_position_size: Макс. размер позиции (доля капитала)
            buy_threshold: Порог для покупки (прогноз роста)
            sell_threshold: Порог для продажи (прогноз падения)
            stop_loss: Stop-loss (доля)
            take_profit: Take-profit (доля)
            logger: Логгер
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission_rate
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.logger = logger or Logger("AutoTrader")
        
        # Состояние портфеля
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
    
    def get_signal(
        self, 
        ticker: str, 
        current_price: float, 
        predicted_price: float
    ) -> str:
        """
        Получить торговый сигнал на основе прогноза.
        
        Args:
            ticker: Тикер
            current_price: Текущая цена
            predicted_price: Прогнозная цена
            
        Returns:
            str: Сигнал (BUY/SELL/HOLD)
        """
        # Прогнозируемое изменение цены
        predicted_change = (predicted_price - current_price) / current_price
        
        # Проверяем существующую позицию
        if ticker in self.positions:
            position = self.positions[ticker]
            position.update(current_price)
            
            # Stop-loss
            if position.pnl_percent <= self.stop_loss * 100:
                self.logger.info(
                    f"{ticker}: Stop-loss triggered at {position.pnl_percent:.2f}%"
                )
                return TradingSignal.SELL
            
            # Take-profit
            if position.pnl_percent >= self.take_profit * 100:
                self.logger.info(
                    f"{ticker}: Take-profit triggered at {position.pnl_percent:.2f}%"
                )
                return TradingSignal.SELL
            
            # Продажа по прогнозу
            if predicted_change < self.sell_threshold:
                self.logger.info(
                    f"{ticker}: Sell signal (predicted change: {predicted_change*100:.2f}%)"
                )
                return TradingSignal.SELL
            
            return TradingSignal.HOLD
        
        else:
            # Нет позиции - проверяем покупку
            if predicted_change > self.buy_threshold:
                # Проверяем лимиты
                if len(self.positions) >= self.max_positions:
                    self.logger.debug(
                        f"{ticker}: Max positions reached ({self.max_positions})"
                    )
                    return TradingSignal.HOLD
                
                # Проверяем баланс
                portfolio_value = self.get_portfolio_value()
                max_investment = portfolio_value * self.max_position_size
                
                if self.balance < max_investment * 0.1:  # Минимум 10% от лимита
                    self.logger.debug(f"{ticker}: Insufficient balance")
                    return TradingSignal.HOLD
                
                self.logger.info(
                    f"{ticker}: Buy signal (predicted change: {predicted_change*100:.2f}%)"
                )
                return TradingSignal.BUY
            
            return TradingSignal.HOLD
    
    def execute_trade(
        self, 
        ticker: str, 
        signal: str, 
        price: float, 
        timestamp: datetime
    ) -> Optional[Trade]:
        """
        Исполнить сделку.
        
        Args:
            ticker: Тикер
            signal: Сигнал (BUY/SELL)
            price: Цена
            timestamp: Время
            
        Returns:
            Trade или None
        """
        if signal == TradingSignal.HOLD:
            return None
        
        if signal == TradingSignal.BUY:
            return self._execute_buy(ticker, price, timestamp)
        
        elif signal == TradingSignal.SELL:
            return self._execute_sell(ticker, price, timestamp)
        
        return None
    
    def _execute_buy(
        self, 
        ticker: str, 
        price: float, 
        timestamp: datetime
    ) -> Optional[Trade]:
        """Купить актив."""
        # Рассчитываем размер позиции
        portfolio_value = self.get_portfolio_value()
        max_investment = portfolio_value * self.max_position_size
        available = min(self.balance, max_investment)
        
        if available < price:
            self.logger.warning(f"{ticker}: Insufficient funds for purchase")
            return None
        
        # Количество акций
        quantity = int(available / price)
        if quantity == 0:
            return None
        
        # Стоимость и комиссия
        cost = price * quantity
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        if total_cost > self.balance:
            # Корректируем количество
            quantity = int(self.balance / (price * (1 + self.commission_rate)))
            if quantity == 0:
                return None
            cost = price * quantity
            commission = cost * self.commission_rate
            total_cost = cost + commission
        
        # Списываем с баланса
        self.balance -= total_cost
        self.total_commission += commission
        
        # Открываем позицию
        self.positions[ticker] = Position(ticker, quantity, price, timestamp)
        
        # Записываем сделку
        trade = Trade(ticker, "BUY", price, quantity, timestamp, commission)
        self.trades.append(trade)
        self.total_trades += 1
        
        self.logger.info(
            f"BUY {ticker}: {quantity} @ {price:.2f} = {cost:.2f} "
            f"(commission: {commission:.2f}, balance: {self.balance:.2f})"
        )
        
        return trade
    
    def _execute_sell(
        self, 
        ticker: str, 
        price: float, 
        timestamp: datetime
    ) -> Optional[Trade]:
        """Продать актив."""
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        position.update(price)
        
        quantity = position.quantity
        revenue = price * quantity
        commission = revenue * self.commission_rate
        net_revenue = revenue - commission
        
        # Пополняем баланс
        self.balance += net_revenue
        self.total_commission += commission
        
        # Статистика
        if position.pnl > 0:
            self.winning_trades += 1
        elif position.pnl < 0:
            self.losing_trades += 1
        
        # Удаляем позицию
        del self.positions[ticker]
        
        # Записываем сделку
        trade = Trade(ticker, "SELL", price, quantity, timestamp, commission)
        self.trades.append(trade)
        self.total_trades += 1
        
        self.logger.info(
            f"SELL {ticker}: {quantity} @ {price:.2f} = {revenue:.2f} "
            f"(P&L: {position.pnl:.2f} / {position.pnl_percent:.2f}%, "
            f"commission: {commission:.2f}, balance: {self.balance:.2f})"
        )
        
        return trade
    
    def update_equity_curve(self, timestamp: datetime):
        """Обновить кривую капитала."""
        equity = self.get_portfolio_value()
        self.equity_curve.append((timestamp, equity))
    
    def get_portfolio_value(self) -> float:
        """Получить текущую стоимость портфеля."""
        positions_value = sum(pos.current_value for pos in self.positions.values())
        return self.balance + positions_value
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику торговли.
        
        Returns:
            dict: Статистика
        """
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Доходности из equity curve
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([eq[1] for eq in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            
            # Метрики
            annual_return = total_return  # Упрощённо
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            
            # Sharpe Ratio
            risk_free_rate = 0.08
            if volatility > 0:
                sharpe = (annual_return - risk_free_rate) / volatility
            else:
                sharpe = 0
            
            # Max Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sortino Ratio (только отрицательные доходности)
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            if downside_std > 0:
                sortino = (annual_return - risk_free_rate) / downside_std
            else:
                sortino = 0
        
        else:
            annual_return = total_return
            volatility = 0
            sharpe = 0
            max_drawdown = 0
            sortino = 0
        
        # Win Rate
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
        else:
            win_rate = 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'win_rate_percent': win_rate * 100,
            'total_commission': self.total_commission,
            'open_positions': len(self.positions),
            'equity_curve_points': len(self.equity_curve)
        }
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Получить DataFrame со всеми сделками."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = [trade.to_dict() for trade in self.trades]
        return pd.DataFrame(trades_data)
    
    def get_positions_dataframe(self) -> pd.DataFrame:
        """Получить DataFrame с открытыми позициями."""
        if not self.positions:
            return pd.DataFrame()
        
        positions_data = [pos.to_dict() for pos in self.positions.values()]
        return pd.DataFrame(positions_data)
    
    def close_all_positions(self, prices: Dict[str, float], timestamp: datetime):
        """
        Закрыть все открытые позиции (в конце симуляции).
        
        Args:
            prices: Текущие цены {ticker: price}
            timestamp: Время закрытия
        """
        tickers_to_close = list(self.positions.keys())
        
        for ticker in tickers_to_close:
            if ticker in prices:
                self._execute_sell(ticker, prices[ticker], timestamp)
            else:
                self.logger.warning(f"No price data for {ticker}, cannot close position")
        
        self.logger.info(f"Closed all positions at {timestamp}")
    
    def reset(self):
        """Сбросить состояние для новой симуляции."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        
        self.logger.info("Trader state reset")


