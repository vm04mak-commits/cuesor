"""
Скрипт 7: Автоматическая торговля с ML моделями

Автоматическая торговая система:
- Выбор ML модели для прогнозов
- Симуляция торговли на исторических данных
- Автоматическое принятие решений (покупка/продажа/удержание)
- Учёт баланса, комиссий, рисков
- Детальная статистика и метрики
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional

from core.database import Database
from core.logger import Logger
from core.config import Config
from portfolio import AutoTrader

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class TradingSimulator:
    """Симулятор автоматической торговли."""
    
    def __init__(
        self,
        model_path: str = "models/universal_model.pkl",
        initial_balance: float = 100000.0,
        commission_rate: float = 0.003,
        logger: Optional[Logger] = None
    ):
        """
        Инициализация симулятора.
        
        Args:
            model_path: Путь к ML модели
            initial_balance: Начальный капитал
            commission_rate: Комиссия брокера
            logger: Логгер
        """
        self.logger = logger or Logger("TradingSimulator")
        self.config = Config()
        
        # База данных
        db_path = self.config.base_path / "data" / "market_data.db"
        self.database = Database(db_path, self.logger)
        
        # Загрузка модели
        self.model_path = Path(model_path)
        self.model_data = None
        self.model = None
        self.scaler = None
        self.features = None
        self.model_type = None
        self.ticker_encoder = LabelEncoder()
        
        self._load_model()
        
        # Инициализация трейдера
        self.trader = AutoTrader(
            initial_balance=initial_balance,
            commission_rate=commission_rate,
            logger=self.logger
        )
    
    def _load_model(self):
        """Загрузить ML модель."""
        try:
            if self.model_path.is_dir():
                # Deep Learning модель (TensorFlow)
                import tensorflow as tf
                
                model_file = self.model_path / "model.keras"
                metadata_file = self.model_path / "metadata.pkl"
                
                if not model_file.exists() or not metadata_file.exists():
                    raise FileNotFoundError(
                        f"Model files not found in {self.model_path}"
                    )
                
                self.model = tf.keras.models.load_model(model_file)
                
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.scaler = metadata['scaler']
                self.features = metadata['features']
                self.model_type = metadata.get('model_type', 'lstm')
                self.ticker_encoder = metadata.get('ticker_encoder', LabelEncoder())
                
                self.logger.info(
                    f"Loaded Deep Learning model ({self.model_type}) from {self.model_path}"
                )
            
            else:
                # Классическая модель (pickle)
                with open(self.model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                
                self.model = self.model_data['model']
                self.scaler = self.model_data['scaler']
                
                # Получаем features
                if 'results' in self.model_data and 'features' in self.model_data['results']:
                    self.features = self.model_data['results']['features']
                elif 'features' in self.model_data:
                    self.features = self.model_data['features']
                else:
                    raise ValueError("Features not found in model data")
                
                self.model_type = self.model_data.get('model_type', 'random_forest')
                self.ticker_encoder = self.model_data.get('ticker_encoder', LabelEncoder())
                
                self.logger.info(
                    f"Loaded classical model ({self.model_type}) from {self.model_path}"
                )
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_price(
        self, 
        ticker: str, 
        current_data: pd.DataFrame,
        indicators: pd.DataFrame
    ) -> Optional[float]:
        """
        Спрогнозировать цену актива.
        
        Args:
            ticker: Тикер
            current_data: Текущие данные котировок
            indicators: Индикаторы
            
        Returns:
            float: Прогнозная цена или None
        """
        try:
            # Подготовка данных
            pred_data = current_data.copy()
            pred_data['ticker'] = ticker
            
            # Encode ticker
            try:
                pred_data['ticker_encoded'] = self.ticker_encoder.transform([ticker])[0]
            except:
                # Тикер не в encoder - используем среднее значение
                pred_data['ticker_encoded'] = 0
            
            # Добавляем индикаторы
            for col in indicators.columns:
                if col in self.features:
                    pred_data[col] = indicators[col].iloc[0]
            
            # Выбираем признаки
            X_pred = pd.DataFrame()
            for f in self.features:
                if f in pred_data.columns:
                    X_pred[f] = pred_data[f]
                else:
                    X_pred[f] = 0
            
            # Заменяем NaN на 0
            X_pred = X_pred.fillna(0)
            
            # Масштабирование
            X_scaled = self.scaler.transform(X_pred)
            
            # Прогноз
            if self.model_type in ['lstm', 'gru']:
                # Deep Learning модель
                X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                pred = self.model.predict(X_reshaped, verbose=0)
                predicted_price = float(pred[0][0])
            else:
                # Классическая модель
                predicted_price = float(self.model.predict(X_scaled)[0])
            
            return predicted_price
        
        except Exception as e:
            self.logger.error(f"Prediction error for {ticker}: {e}")
            return None
    
    def run_simulation(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval_days: int = 1
    ) -> Dict:
        """
        Запустить симуляцию торговли.
        
        Args:
            tickers: Список тикеров
            start_date: Дата начала (YYYY-MM-DD)
            end_date: Дата окончания (YYYY-MM-DD)
            interval_days: Интервал прогнозов (дней)
            
        Returns:
            dict: Результаты симуляции
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting trading simulation")
        self.logger.info(f"Tickers: {len(tickers)}")
        self.logger.info(f"Period: {start_date} - {end_date}")
        self.logger.info(f"Interval: {interval_days} days")
        self.logger.info(f"Initial balance: {self.trader.initial_balance:,.2f} ₽")
        self.logger.info("=" * 80)
        
        # Сброс трейдера
        self.trader.reset()
        
        # Загрузка данных
        quotes_data = {}
        indicators_data = {}
        
        print("\n📊 Загрузка данных...")
        for ticker in tickers:
            try:
                quotes = self.database.load_quotes(ticker)
                indicators = self.database.load_indicators(ticker)
                
                if quotes.empty or indicators.empty:
                    self.logger.warning(f"No data for {ticker}")
                    continue
                
                # Фильтрация по датам
                if start_date:
                    quotes = quotes[quotes['date'] >= start_date]
                    indicators = indicators[indicators.index >= start_date]
                
                if end_date:
                    quotes = quotes[quotes['date'] <= end_date]
                    indicators = indicators[indicators.index <= end_date]
                
                if len(quotes) < 30:
                    self.logger.warning(f"Insufficient data for {ticker}")
                    continue
                
                quotes_data[ticker] = quotes.reset_index(drop=True)
                indicators_data[ticker] = indicators
                
            except Exception as e:
                self.logger.error(f"Error loading {ticker}: {e}")
        
        if not quotes_data:
            raise ValueError("No data loaded for any ticker")
        
        print(f"✅ Загружено данных для {len(quotes_data)} тикеров")
        
        # Определяем торговые дни
        all_dates = set()
        for quotes in quotes_data.values():
            all_dates.update(quotes['date'].tolist())
        
        trading_days = sorted(list(all_dates))
        
        if not trading_days:
            raise ValueError("No trading days found")
        
        print(f"📅 Торговых дней: {len(trading_days)}")
        print(f"🚀 Запуск симуляции...\n")
        
        # Симуляция по дням
        for day_idx, current_date in enumerate(trading_days[::interval_days]):
            
            # Логируем прогресс
            if day_idx % 10 == 0:
                progress = (day_idx / len(trading_days)) * 100
                print(f"📈 День {day_idx + 1}/{len(trading_days)} ({progress:.1f}%)", end='\r')
            
            # Текущие цены для обновления equity curve
            current_prices = {}
            
            # Проходим по всем тикерам
            for ticker in quotes_data.keys():
                quotes = quotes_data[ticker]
                indicators = indicators_data[ticker]
                
                # Находим данные на текущую дату
                current_row = quotes[quotes['date'] == current_date]
                
                if current_row.empty:
                    continue
                
                current_price = current_row['close'].iloc[0]
                current_prices[ticker] = current_price
                
                # Находим соответствующий индикатор
                indicator_row = indicators[indicators.index == current_date]
                
                if indicator_row.empty:
                    continue
                
                # Делаем прогноз
                predicted_price = self.predict_price(
                    ticker, 
                    current_row, 
                    indicator_row
                )
                
                if predicted_price is None:
                    continue
                
                # Получаем сигнал
                signal = self.trader.get_signal(ticker, current_price, predicted_price)
                
                # Исполняем сделку
                trade = self.trader.execute_trade(ticker, signal, current_price, current_date)
            
            # Обновляем позиции текущими ценами
            for ticker, position in self.trader.positions.items():
                if ticker in current_prices:
                    position.update(current_prices[ticker])
            
            # Обновляем equity curve
            self.trader.update_equity_curve(current_date)
        
        print("\n")
        
        # Закрываем все позиции в конце симуляции
        final_prices = {}
        for ticker in quotes_data.keys():
            quotes = quotes_data[ticker]
            if not quotes.empty:
                final_prices[ticker] = quotes['close'].iloc[-1]
        
        self.trader.close_all_positions(final_prices, trading_days[-1])
        
        # Финальная equity
        self.trader.update_equity_curve(trading_days[-1])
        
        # Получаем статистику
        stats = self.trader.get_statistics()
        
        self.logger.info("=" * 80)
        self.logger.info("Simulation completed")
        self.logger.info("=" * 80)
        
        return {
            'statistics': stats,
            'trades': self.trader.get_trades_dataframe(),
            'equity_curve': self.trader.equity_curve,
            'simulation_days': len(trading_days),
            'quotes_data': quotes_data  # Добавляем данные котировок
        }


def print_statistics(stats: Dict):
    """Вывести статистику."""
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
    print("=" * 80)
    print()
    
    print("💰 Финансовые результаты:")
    print(f"   Начальный капитал:     {stats['initial_balance']:>15,.2f} ₽")
    print(f"   Конечный капитал:      {stats['portfolio_value']:>15,.2f} ₽")
    print(f"   Общая доходность:      {stats['total_return_percent']:>15,.2f}%")
    print(f"   Годовая доходность:    {stats['annual_return']*100:>15,.2f}%")
    print()
    
    print("📈 Метрики производительности:")
    print(f"   Sharpe Ratio:          {stats['sharpe_ratio']:>15,.2f}")
    print(f"   Sortino Ratio:         {stats['sortino_ratio']:>15,.2f}")
    print(f"   Волатильность:         {stats['volatility']*100:>15,.2f}%")
    print(f"   Max Drawdown:          {stats['max_drawdown_percent']:>15,.2f}%")
    print()
    
    print("🎯 Торговая статистика:")
    print(f"   Всего сделок:          {stats['total_trades']:>15,.0f}")
    print(f"   Прибыльных:            {stats['winning_trades']:>15,.0f}")
    print(f"   Убыточных:             {stats['losing_trades']:>15,.0f}")
    print(f"   Win Rate:              {stats['win_rate_percent']:>15,.2f}%")
    print(f"   Комиссии:              {stats['total_commission']:>15,.2f} ₽")
    print(f"   Открытых позиций:      {stats['open_positions']:>15,.0f}")
    print()


def plot_results(equity_curve: List, trades_df: pd.DataFrame):
    """Визуализация результатов."""
    if not equity_curve:
        print("⚠️  Нет данных для визуализации")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Equity Curve
    ax = axes[0, 0]
    dates = [eq[0] for eq in equity_curve]
    values = [eq[1] for eq in equity_curve]
    
    ax.plot(dates, values, linewidth=2, color='steelblue')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Капитал (₽)')
    ax.set_title('Кривая капитала', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # 2. Drawdown
    ax = axes[0, 1]
    equity_series = pd.Series(values)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    
    ax.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown, color='darkred', linewidth=1.5)
    ax.set_xlabel('Дни')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Просадка', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # 3. Trades Timeline
    if not trades_df.empty:
        ax = axes[1, 0]
        
        buys = trades_df[trades_df['action'] == 'BUY']
        sells = trades_df[trades_df['action'] == 'SELL']
        
        if not buys.empty:
            ax.scatter(buys['timestamp'], buys['price'], 
                      color='green', marker='^', s=100, label='Покупка', alpha=0.6)
        
        if not sells.empty:
            ax.scatter(sells['timestamp'], sells['price'], 
                      color='red', marker='v', s=100, label='Продажа', alpha=0.6)
        
        ax.set_xlabel('Дата')
        ax.set_ylabel('Цена (₽)')
        ax.set_title('История сделок', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Returns Distribution
    ax = axes[1, 1]
    returns = equity_series.pct_change().dropna()
    
    if len(returns) > 0:
        ax.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(returns.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Среднее: {returns.mean():.4f}')
        ax.set_xlabel('Доходность')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение доходностей', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('trading_reports', exist_ok=True)
    filename = f"trading_reports/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ График сохранён: {filename}")
    
    plt.show()


def plot_ticker_charts(results: Dict, quotes_data: Dict, simulation_date: str):
    """
    Построить графики котировок для каждого тикера с точками покупки/продажи.
    
    Args:
        results: Результаты симуляции
        quotes_data: Данные котировок
        simulation_date: Дата симуляции для имени папки
    """
    trades_df = results['trades']
    
    if trades_df.empty:
        print("⚠️  Нет сделок для отображения")
        return
    
    # Создаём папку для графиков
    charts_dir = Path(f"trading_charts/{simulation_date}")
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем уникальные тикеры из сделок
    traded_tickers = trades_df['ticker'].unique()
    
    print(f"\n📊 Построение графиков котировок для {len(traded_tickers)} тикеров...")
    
    for idx, ticker in enumerate(traded_tickers, 1):
        try:
            print(f"   {idx}/{len(traded_tickers)}: {ticker}...", end='\r')
            
            # Получаем котировки
            if ticker not in quotes_data:
                continue
            
            quotes = quotes_data[ticker].copy()
            
            # Сделки по этому тикеру
            ticker_trades = trades_df[trades_df['ticker'] == ticker].copy()
            
            if ticker_trades.empty:
                continue
            
            # Создаём график
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                            gridspec_kw={'height_ratios': [3, 1]})
            
            # График 1: Цена закрытия
            ax1.plot(quotes['date'], quotes['close'], 
                    linewidth=1.5, color='steelblue', label='Close', alpha=0.8)
            
            # Точки покупки
            buys = ticker_trades[ticker_trades['action'] == 'BUY']
            if not buys.empty:
                ax1.scatter(buys['timestamp'], buys['price'], 
                           color='green', marker='^', s=150, 
                           label='Покупка', alpha=0.8, edgecolors='darkgreen', linewidths=2)
                
                # Подписи для покупок
                for _, trade in buys.iterrows():
                    ax1.annotate(f"{trade['quantity']}шт", 
                               xy=(trade['timestamp'], trade['price']),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', fontsize=8, color='darkgreen',
                               bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7))
            
            # Точки продажи
            sells = ticker_trades[ticker_trades['action'] == 'SELL']
            if not sells.empty:
                ax1.scatter(sells['timestamp'], sells['price'], 
                           color='red', marker='v', s=150, 
                           label='Продажа', alpha=0.8, edgecolors='darkred', linewidths=2)
                
                # Подписи для продаж
                for _, trade in sells.iterrows():
                    ax1.annotate(f"{trade['quantity']}шт", 
                               xy=(trade['timestamp'], trade['price']),
                               xytext=(0, -15), textcoords='offset points',
                               ha='center', fontsize=8, color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.7))
            
            ax1.set_xlabel('Дата', fontsize=11)
            ax1.set_ylabel('Цена (₽)', fontsize=11)
            ax1.set_title(f'{ticker} — Котировки и сделки', 
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Статистика по тикеру
            total_buys = len(buys)
            total_sells = len(sells)
            
            # Рассчитываем P&L
            total_pnl = 0
            if total_sells > 0:
                for _, sell in sells.iterrows():
                    # Находим соответствующую покупку
                    matching_buys = buys[buys['timestamp'] < sell['timestamp']]
                    if not matching_buys.empty:
                        buy_price = matching_buys.iloc[-1]['price']
                        pnl = (sell['price'] - buy_price) * sell['quantity']
                        total_pnl += pnl
            
            stats_text = f"Покупок: {total_buys} | Продаж: {total_sells}"
            if total_pnl != 0:
                pnl_color = 'green' if total_pnl > 0 else 'red'
                stats_text += f" | P&L: {total_pnl:+,.0f} ₽"
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=pnl_color, alpha=0.2))
            else:
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # График 2: Объём
            colors = ['green' if quotes['close'].iloc[i] >= quotes['open'].iloc[i] 
                     else 'red' for i in range(len(quotes))]
            
            ax2.bar(quotes['date'], quotes['volume'], color=colors, alpha=0.6)
            ax2.set_xlabel('Дата', fontsize=11)
            ax2.set_ylabel('Объём', fontsize=11)
            ax2.set_title('Объём торгов', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Сохранение
            chart_file = charts_dir / f"{ticker}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Сохранение CSV с сделками по тикеру
            csv_file = charts_dir / f"{ticker}_trades.csv"
            ticker_trades.to_csv(csv_file, index=False, encoding='utf-8')
        
        except Exception as e:
            print(f"\n⚠️  Ошибка при построении графика {ticker}: {e}")
            continue
    
    print(f"\n✅ Графики сохранены в: {charts_dir}/")
    print(f"   - {len(traded_tickers)} PNG файлов с графиками")
    print(f"   - {len(traded_tickers)} CSV файлов со сделками")


def save_detailed_report(results: Dict, filename: str):
    """Сохранить детальный отчёт."""
    os.makedirs('trading_reports', exist_ok=True)
    
    report_path = Path('trading_reports') / filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ДЕТАЛЬНЫЙ ОТЧЁТ ТОРГОВОЙ СИМУЛЯЦИИ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Дней симуляции: {results['simulation_days']}\n\n")
        
        # Статистика
        stats = results['statistics']
        
        f.write("=" * 80 + "\n")
        f.write("ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Начальный капитал:     {stats['initial_balance']:,.2f} ₽\n")
        f.write(f"Конечный капитал:      {stats['portfolio_value']:,.2f} ₽\n")
        f.write(f"Общая доходность:      {stats['total_return_percent']:.2f}%\n")
        f.write(f"Годовая доходность:    {stats['annual_return']*100:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Sharpe Ratio:          {stats['sharpe_ratio']:.2f}\n")
        f.write(f"Sortino Ratio:         {stats['sortino_ratio']:.2f}\n")
        f.write(f"Волатильность:         {stats['volatility']*100:.2f}%\n")
        f.write(f"Max Drawdown:          {stats['max_drawdown_percent']:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ТОРГОВАЯ СТАТИСТИКА\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Всего сделок:          {stats['total_trades']:.0f}\n")
        f.write(f"Прибыльных:            {stats['winning_trades']:.0f}\n")
        f.write(f"Убыточных:             {stats['losing_trades']:.0f}\n")
        f.write(f"Win Rate:              {stats['win_rate_percent']:.2f}%\n")
        f.write(f"Комиссии:              {stats['total_commission']:.2f} ₽\n\n")
        
        # Сделки
        trades_df = results['trades']
        if not trades_df.empty:
            f.write("=" * 80 + "\n")
            f.write("ИСТОРИЯ СДЕЛОК\n")
            f.write("=" * 80 + "\n\n")
            
            for _, trade in trades_df.iterrows():
                f.write(
                    f"{trade['timestamp']} | {trade['action']:<4} | "
                    f"{trade['ticker']:<6} | {trade['quantity']:>4} @ {trade['price']:>8.2f} ₽ | "
                    f"Комиссия: {trade['commission']:>6.2f} ₽\n"
                )
    
    print(f"✅ Отчёт сохранён: {report_path}")


def find_available_models():
    """Найти все доступные модели."""
    models_dir = Path("models")
    available_models = []
    
    if not models_dir.exists():
        return available_models
    
    # Ищем .pkl файлы рекурсивно
    for pkl_file in models_dir.rglob("*.pkl"):
        # Пропускаем metadata.pkl
        if pkl_file.name == "metadata.pkl":
            continue
            
        # Проверяем, что это модель (содержит "model" в имени)
        if "model" in pkl_file.name.lower():
            available_models.append({
                'path': str(pkl_file),
                'name': f"{pkl_file.parent.name}/{pkl_file.stem}" if pkl_file.parent.name != "models" else pkl_file.stem,
                'type': 'classical',
                'size': pkl_file.stat().st_size / (1024 * 1024),  # MB
                'modified': datetime.fromtimestamp(pkl_file.stat().st_mtime)
            })
    
    # Ищем директории с Deep Learning моделями рекурсивно
    for keras_file in models_dir.rglob("model.keras"):
        model_dir = keras_file.parent
        available_models.append({
            'path': str(model_dir),
            'name': f"{model_dir.parent.name}/{model_dir.name}" if model_dir.parent.name != "models" else model_dir.name,
            'type': 'deep_learning',
            'size': keras_file.stat().st_size / (1024 * 1024),  # MB
            'modified': datetime.fromtimestamp(keras_file.stat().st_mtime)
        })
    
    # Сортируем по дате изменения (новые первые)
    available_models.sort(key=lambda x: x['modified'], reverse=True)
    
    return available_models


def select_model():
    """Выбрать модель из списка."""
    print("\n" + "=" * 80)
    print("📦 ВЫБОР МОДЕЛИ")
    print("=" * 80)
    
    models = find_available_models()
    
    if not models:
        print("\n❌ Модели не найдены!")
        print("   Сначала обучите модель: python 3_train_universal_model.py")
        return None
    
    print(f"\nНайдено моделей: {len(models)}\n")
    
    for idx, model in enumerate(models, 1):
        model_type_str = "🧠 Deep Learning" if model['type'] == 'deep_learning' else "🌲 Classical ML"
        print(f"  {idx}. {model_type_str}")
        print(f"     Имя: {model['name']}")
        print(f"     Размер: {model['size']:.1f} MB")
        print(f"     Обновлено: {model['modified'].strftime('%Y-%m-%d %H:%M')}")
        print()
    
    while True:
        try:
            choice = input(f"Выберите модель (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"\n✅ Выбрана модель: {selected['name']}")
                return selected['path']
            else:
                print(f"❌ Введите число от 1 до {len(models)}")
        except ValueError:
            print("❌ Введите корректное число")
        except KeyboardInterrupt:
            return None


def main():
    """Главное меню."""
    print("\n" + "=" * 80)
    print("🤖 АВТОМАТИЧЕСКАЯ ТОРГОВАЯ СИСТЕМА")
    print("=" * 80)
    
    # Выбор модели
    model_to_use = select_model()
    
    if not model_to_use:
        return
    
    # Параметры симуляции
    print("\n" + "=" * 80)
    print("⚙️  ПАРАМЕТРЫ СИМУЛЯЦИИ")
    print("=" * 80)
    
    # Начальный капитал
    print("\n💰 Начальный капитал:")
    print("  1. 100,000 ₽ (по умолчанию)")
    print("  2. 500,000 ₽")
    print("  3. 1,000,000 ₽")
    print("  4. Свой вариант")
    
    balance_choice = input("\nВыбор (1-4): ").strip()
    
    if balance_choice == '2':
        initial_balance = 500000
    elif balance_choice == '3':
        initial_balance = 1000000
    elif balance_choice == '4':
        try:
            initial_balance = float(input("Введите сумму (₽): ").strip())
        except:
            initial_balance = 100000
    else:
        initial_balance = 100000
    
    # Период симуляции
    print("\n📅 Период симуляции:")
    print("  1. Последние 6 месяцев")
    print("  2. Последний год")
    print("  3. Последние 2 года")
    print("  4. Свой период")
    
    period_choice = input("\nВыбор (1-4): ").strip()
    
    if period_choice == '1':
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
    elif period_choice == '2':
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
    elif period_choice == '3':
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
    elif period_choice == '4':
        start_date = input("Дата начала (YYYY-MM-DD): ").strip()
        end_date = input("Дата окончания (YYYY-MM-DD): ").strip()
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Интервал прогнозов
    print("\n⏱️  Интервал прогнозов:")
    print("  1. Каждый день")
    print("  2. Каждые 3 дня")
    print("  3. Каждую неделю")
    
    interval_choice = input("\nВыбор (1-3): ").strip()
    
    if interval_choice == '2':
        interval_days = 3
    elif interval_choice == '3':
        interval_days = 7
    else:
        interval_days = 1
    
    # Получаем список тикеров
    print("\n📊 Загрузка списка тикеров...")
    config = Config()
    db_path = config.base_path / "data" / "market_data.db"
    db = Database(db_path, Logger("Main"))
    
    tickers = db.get_available_tickers()
    
    if not tickers:
        print("❌ Нет доступных тикеров в базе данных")
        return
    
    print(f"✅ Найдено {len(tickers)} тикеров")
    
    # Создаём симулятор
    print("\n🚀 Инициализация торговой системы...")
    
    try:
        simulator = TradingSimulator(
            model_path=model_to_use,
            initial_balance=initial_balance,
            commission_rate=0.003
        )
        
        # Запускаем симуляцию
        results = simulator.run_simulation(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            interval_days=interval_days
        )
        
        # Выводим статистику
        print_statistics(results['statistics'])
        
        # Timestamp для папок и файлов
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Сохраняем отчёт
        print("\n💾 Сохранение отчётов...")
        
        report_filename = f"report_{timestamp}.txt"
        save_detailed_report(results, report_filename)
        
        # Визуализация общих результатов
        print("\n📊 Построение общих графиков...")
        plot_results(results['equity_curve'], results['trades'])
        
        # Визуализация котировок по каждому тикеру
        print("\n📈 Построение графиков котировок с точками сделок...")
        plot_ticker_charts(results, results['quotes_data'], timestamp)
        
        print("\n" + "=" * 80)
        print("✅ СИМУЛЯЦИЯ ЗАВЕРШЕНА")
        print("=" * 80)
        print(f"\n📂 Результаты сохранены:")
        print(f"   - trading_reports/report_{timestamp}.txt")
        print(f"   - trading_reports/simulation_{timestamp}.png")
        print(f"   - trading_charts/{timestamp}/ (графики и CSV по тикерам)")
    
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

