"""
Portfolio Class

Класс для управления инвестиционным портфелем.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
from pathlib import Path


class Portfolio:
    """
    Класс для управления инвестиционным портфелем.
    
    Основные функции:
    - Хранение состава портфеля
    - Расчёт стоимости портфеля
    - История изменений
    - Метрики производительности
    """
    
    def __init__(self, name: str = "My Portfolio", initial_cash: float = 0.0):
        """
        Инициализация портфеля.
        
        Args:
            name (str): Название портфеля
            initial_cash (float): Начальный капитал
        """
        self.name = name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict] = {}  # {ticker: {shares, avg_price, current_price}}
        self.history: List[Dict] = []  # История операций
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def add_position(self, ticker: str, shares: float, price: float, date: str = None):
        """
        Добавить позицию в портфель (покупка акций).
        
        Args:
            ticker (str): Тикер акции
            shares (float): Количество акций
            price (float): Цена покупки
            date (str): Дата операции
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        cost = shares * price
        
        if cost > self.cash:
            raise ValueError(f"Недостаточно средств. Нужно {cost:.2f}, доступно {self.cash:.2f}")
        
        if ticker in self.positions:
            # Усредняем цену покупки
            old_shares = self.positions[ticker]['shares']
            old_price = self.positions[ticker]['avg_price']
            new_shares = old_shares + shares
            new_avg_price = (old_shares * old_price + shares * price) / new_shares
            
            self.positions[ticker]['shares'] = new_shares
            self.positions[ticker]['avg_price'] = new_avg_price
        else:
            self.positions[ticker] = {
                'shares': shares,
                'avg_price': price,
                'current_price': price
            }
        
        self.cash -= cost
        
        # Добавляем в историю
        self.history.append({
            'date': date,
            'action': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'cost': cost,
            'cash_after': self.cash
        })
        
        self.updated_at = datetime.now()
        
    def remove_position(self, ticker: str, shares: float, price: float, date: str = None):
        """
        Удалить позицию из портфеля (продажа акций).
        
        Args:
            ticker (str): Тикер акции
            shares (float): Количество акций
            price (float): Цена продажи
            date (str): Дата операции
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        if ticker not in self.positions:
            raise ValueError(f"Акция {ticker} не найдена в портфеле")
        
        if self.positions[ticker]['shares'] < shares:
            raise ValueError(f"Недостаточно акций {ticker}. Есть {self.positions[ticker]['shares']}, пытаетесь продать {shares}")
        
        proceeds = shares * price
        
        self.positions[ticker]['shares'] -= shares
        
        if self.positions[ticker]['shares'] == 0:
            del self.positions[ticker]
        
        self.cash += proceeds
        
        # Добавляем в историю
        self.history.append({
            'date': date,
            'action': 'SELL',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'proceeds': proceeds,
            'cash_after': self.cash
        })
        
        self.updated_at = datetime.now()
        
    def update_prices(self, prices: Dict[str, float]):
        """
        Обновить текущие цены акций в портфеле.
        
        Args:
            prices (Dict[str, float]): Словарь {ticker: current_price}
        """
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker]['current_price'] = price
        
        self.updated_at = datetime.now()
        
    def get_total_value(self) -> float:
        """
        Получить общую стоимость портфеля.
        
        Returns:
            float: Общая стоимость (акции + кэш)
        """
        stocks_value = sum(
            pos['shares'] * pos['current_price']
            for pos in self.positions.values()
        )
        return stocks_value + self.cash
    
    def get_positions_value(self) -> float:
        """
        Получить стоимость всех позиций (без кэша).
        
        Returns:
            float: Стоимость позиций
        """
        return sum(
            pos['shares'] * pos['current_price']
            for pos in self.positions.values()
        )
    
    def get_position_value(self, ticker: str) -> float:
        """
        Получить стоимость конкретной позиции.
        
        Args:
            ticker (str): Тикер акции
            
        Returns:
            float: Стоимость позиции
        """
        if ticker not in self.positions:
            return 0.0
        
        pos = self.positions[ticker]
        return pos['shares'] * pos['current_price']
    
    def get_weights(self) -> Dict[str, float]:
        """
        Получить веса позиций в портфеле.
        
        Returns:
            Dict[str, float]: Словарь {ticker: weight}
        """
        total_value = self.get_positions_value()
        
        if total_value == 0:
            return {}
        
        weights = {}
        for ticker, pos in self.positions.items():
            position_value = pos['shares'] * pos['current_price']
            weights[ticker] = position_value / total_value
        
        return weights
    
    def get_returns(self) -> Dict[str, float]:
        """
        Получить доходность по каждой позиции.
        
        Returns:
            Dict[str, float]: Словарь {ticker: return_pct}
        """
        returns = {}
        for ticker, pos in self.positions.items():
            avg_price = pos['avg_price']
            current_price = pos['current_price']
            returns[ticker] = (current_price - avg_price) / avg_price * 100
        
        return returns
    
    def get_total_return(self) -> float:
        """
        Получить общую доходность портфеля.
        
        Returns:
            float: Доходность в процентах
        """
        if self.initial_cash == 0:
            return 0.0
        
        current_value = self.get_total_value()
        return (current_value - self.initial_cash) / self.initial_cash * 100
    
    def get_profit_loss(self) -> float:
        """
        Получить абсолютную прибыль/убыток.
        
        Returns:
            float: Прибыль/убыток
        """
        return self.get_total_value() - self.initial_cash
    
    def get_summary(self) -> Dict:
        """
        Получить сводку по портфелю.
        
        Returns:
            Dict: Сводка с метриками
        """
        total_value = self.get_total_value()
        positions_value = self.get_positions_value()
        
        summary = {
            'name': self.name,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S'),
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value,
            'profit_loss': self.get_profit_loss(),
            'total_return': self.get_total_return(),
            'num_positions': len(self.positions),
            'positions': []
        }
        
        for ticker, pos in self.positions.items():
            position_value = pos['shares'] * pos['current_price']
            weight = position_value / positions_value if positions_value > 0 else 0
            return_pct = (pos['current_price'] - pos['avg_price']) / pos['avg_price'] * 100
            
            summary['positions'].append({
                'ticker': ticker,
                'shares': pos['shares'],
                'avg_price': pos['avg_price'],
                'current_price': pos['current_price'],
                'value': position_value,
                'weight': weight * 100,
                'return': return_pct,
                'profit_loss': (pos['current_price'] - pos['avg_price']) * pos['shares']
            })
        
        # Сортируем позиции по стоимости
        summary['positions'].sort(key=lambda x: x['value'], reverse=True)
        
        return summary
    
    def print_summary(self):
        """
        Вывести сводку по портфелю в консоль.
        """
        summary = self.get_summary()
        
        print(f"\n{'='*80}")
        print(f"📊 Портфель: {summary['name']}")
        print(f"{'='*80}")
        print(f"Создан:    {summary['created_at']}")
        print(f"Обновлён:  {summary['updated_at']}")
        print(f"\n{'Капитал':<30} {'Значение':>15} {'Доля':>10}")
        print(f"{'-'*80}")
        print(f"{'Начальный капитал':<30} {summary['initial_cash']:>15,.2f}")
        print(f"{'Текущий кэш':<30} {summary['current_cash']:>15,.2f} {(summary['current_cash']/summary['total_value']*100):>9.1f}%")
        print(f"{'Стоимость позиций':<30} {summary['positions_value']:>15,.2f} {(summary['positions_value']/summary['total_value']*100):>9.1f}%")
        print(f"{'Общая стоимость':<30} {summary['total_value']:>15,.2f}")
        
        profit_symbol = "📈" if summary['profit_loss'] >= 0 else "📉"
        print(f"\n{profit_symbol} {'Прибыль/Убыток':<27} {summary['profit_loss']:>15,.2f} ({summary['total_return']:>+6.2f}%)")
        
        if summary['positions']:
            print(f"\n{'Позиции:':<30} {summary['num_positions']} шт.")
            print(f"\n{'Тикер':<10} {'Акций':>10} {'Ср.цена':>12} {'Тек.цена':>12} {'Стоимость':>15} {'Вес':>8} {'Доход':>10}")
            print(f"{'-'*80}")
            
            for pos in summary['positions']:
                return_symbol = "+" if pos['return'] >= 0 else ""
                print(f"{pos['ticker']:<10} {pos['shares']:>10.0f} "
                      f"{pos['avg_price']:>12,.2f} {pos['current_price']:>12,.2f} "
                      f"{pos['value']:>15,.2f} {pos['weight']:>7.1f}% "
                      f"{return_symbol}{pos['return']:>9.2f}%")
        
        print(f"{'='*80}\n")
    
    def save(self, filepath: str = None):
        """
        Сохранить портфель в JSON файл.
        
        Args:
            filepath (str): Путь к файлу. Если None, сохраняется в portfolios/
        """
        if filepath is None:
            os.makedirs('portfolios', exist_ok=True)
            filename = f"{self.name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join('portfolios', filename)
        
        data = {
            'name': self.name,
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'positions': self.positions,
            'history': self.history,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Портфель сохранён: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'Portfolio':
        """
        Загрузить портфель из JSON файла.
        
        Args:
            filepath (str): Путь к файлу
            
        Returns:
            Portfolio: Загруженный портфель
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        portfolio = cls(name=data['name'], initial_cash=data['initial_cash'])
        portfolio.cash = data['cash']
        portfolio.positions = data['positions']
        portfolio.history = data['history']
        portfolio.created_at = datetime.fromisoformat(data['created_at'])
        portfolio.updated_at = datetime.fromisoformat(data['updated_at'])
        
        print(f"✅ Портфель загружен: {filepath}")
        return portfolio
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Получить историю операций в виде DataFrame.
        
        Returns:
            pd.DataFrame: История операций
        """
        if not self.history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.history)









