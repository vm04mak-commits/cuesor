"""
Скрипт 6: Оптимизация портфеля

Создание оптимального инвестиционного портфеля на основе:
- Прогнозов универсальной модели
- Modern Portfolio Theory (Марковиц)
- Risk management метрик

Возможности:
1. Создать оптимизированный портфель
2. Анализ существующего портфеля
3. Рекомендации по ребалансировке
4. Efficient Frontier
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from core.database import Database
from core.logger import Logger
from core.config import Config
from portfolio import Portfolio, PortfolioOptimizer, RiskManager, PortfolioRebalancer

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class PortfolioManager:
    """Менеджер портфеля."""
    
    def __init__(self):
        self.logger = Logger("PortfolioManager")
        self.config = Config()
        db_path = self.config.base_path / "data" / "market_data.db"
        self.database = Database(db_path, self.logger)
        
    def get_returns_data(self, tickers: list, days: int = 365) -> pd.DataFrame:
        """
        Получить данные доходностей для списка тикеров.
        
        Args:
            tickers (list): Список тикеров
            days (int): Период (дней)
            
        Returns:
            pd.DataFrame: DataFrame с доходностями
        """
        returns_dict = {}
        
        for ticker in tickers:
            try:
                df = self.database.load_quotes(ticker)
                
                if df.empty or len(df) < 30:
                    self.logger.warning(f"Недостаточно данных для {ticker}")
                    continue
                
                # Берём последние N дней
                df = df.tail(days)
                
                # Устанавливаем дату как индекс
                if 'date' in df.columns:
                    df = df.set_index('date')
                
                # Рассчитываем доходности
                df['return'] = df['close'].pct_change()
                
                returns_dict[ticker] = df['return']
                
            except Exception as e:
                self.logger.error(f"Ошибка загрузки {ticker}: {e}")
        
        if not returns_dict:
            raise ValueError("Не удалось загрузить данные ни для одного тикера")
        
        # Создаём DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Удаляем строки где ВСЕ значения NaN
        returns_df = returns_df.dropna(how='all')
        
        # Заполняем пропуски нулями (акция не торговалась = доходность 0)
        returns_df = returns_df.fillna(0)
        
        # Удаляем первую строку (NaN после pct_change)
        if len(returns_df) > 0:
            returns_df = returns_df.iloc[1:]
        
        self.logger.info(f"Подготовлено {len(returns_df)} периодов доходностей для {len(returns_df.columns)} акций")
        
        return returns_df
    
    def get_latest_prices(self, tickers: list) -> dict:
        """
        Получить последние цены для тикеров.
        
        Args:
            tickers (list): Список тикеров
            
        Returns:
            dict: {ticker: price}
        """
        prices = {}
        
        for ticker in tickers:
            try:
                df = self.database.load_quotes(ticker)
                if not df.empty:
                    prices[ticker] = df['close'].iloc[-1]
            except:
                pass
        
        return prices
    
    def load_predictions(self, prediction_file: str = None) -> pd.DataFrame:
        """
        Загрузить прогнозы модели.
        
        Args:
            prediction_file (str): Путь к файлу с прогнозами
            
        Returns:
            pd.DataFrame: Прогнозы
        """
        if prediction_file is None:
            # Ищем последний файл прогнозов
            predictions_dir = 'predictions'
            if not os.path.exists(predictions_dir):
                raise FileNotFoundError("Папка predictions не найдена. Сначала запустите 4_predict_stocks.py")
            
            files = [f for f in os.listdir(predictions_dir) if f.startswith('predictions_') and f.endswith('.csv')]
            
            if not files:
                raise FileNotFoundError("Файлы прогнозов не найдены. Сначала запустите 4_predict_stocks.py")
            
            # Берём последний файл
            files.sort(reverse=True)
            prediction_file = os.path.join(predictions_dir, files[0])
        
        df = pd.read_csv(prediction_file)
        self.logger.info(f"Загружены прогнозы из {prediction_file}")
        
        return df


def create_optimal_portfolio():
    """Создать оптимальный портфель на основе прогнозов."""
    print("\n" + "="*80)
    print("🎯 Создание оптимального портфеля")
    print("="*80)
    
    manager = PortfolioManager()
    
    # 1. Загружаем прогнозы
    print("\n📊 Шаг 1: Загрузка прогнозов...")
    predictions = manager.load_predictions()
    print(f"✅ Загружено прогнозов: {len(predictions)}")
    
    # 2. Выбираем топ акций по прогнозу
    print("\n📈 Шаг 2: Выбор акций...")
    print("\nВыберите количество акций для портфеля:")
    print("  1. Топ-10 акций")
    print("  2. Топ-20 акций")
    print("  3. Топ-30 акций")
    print("  4. Все акции с положительным прогнозом")
    
    choice = input("\nВаш выбор (1-4): ").strip()
    
    if choice == '1':
        top_n = 10
    elif choice == '2':
        top_n = 20
    elif choice == '3':
        top_n = 30
    elif choice == '4':
        top_n = len(predictions[predictions['change_percent'] > 0])
    else:
        print("❌ Неверный выбор")
        return
    
    top_stocks = predictions.nlargest(top_n, 'change_percent')
    tickers = top_stocks['ticker'].tolist()
    
    print(f"\n✅ Выбрано акций: {len(tickers)}")
    print("\nТоп-5 по прогнозу:")
    for i, row in top_stocks.head().iterrows():
        print(f"  {row['ticker']:<10} {row['change_percent']:>+7.2f}%")
    
    # 3. Загружаем исторические данные
    print(f"\n📚 Шаг 3: Загрузка исторических данных...")
    
    print("\nВыберите период для анализа:")
    print("  1. 3 месяца")
    print("  2. 6 месяцев")
    print("  3. 1 год")
    print("  4. 2 года")
    
    period_choice = input("\nПериод (1-4): ").strip()
    
    period_days = {
        '1': 90,
        '2': 180,
        '3': 365,
        '4': 730
    }
    
    days = period_days.get(period_choice, 365)
    
    try:
        returns = manager.get_returns_data(tickers, days=days)
        print(f"✅ Загружено данных для {len(returns.columns)} акций")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # 4. Оптимизация портфеля
    print(f"\n🔧 Шаг 4: Оптимизация портфеля...")
    
    print("\nВыберите стратегию оптимизации:")
    print("  1. Max Sharpe Ratio (максимум доходности на единицу риска)")
    print("  2. Min Variance (минимальный риск)")
    print("  3. Risk Parity (равный вклад в риск)")
    print("  4. Equal Weight (равные веса)")
    print("  5. Сравнить все стратегии")
    
    strategy_choice = input("\nСтратегия (1-5): ").strip()
    
    optimizer = PortfolioOptimizer(returns, risk_free_rate=0.08)  # 8% безрисковая ставка
    
    try:
        if strategy_choice == '1':
            result = optimizer.max_sharpe_portfolio()
            strategy_name = "Max Sharpe Ratio"
        elif strategy_choice == '2':
            result = optimizer.min_variance_portfolio()
            strategy_name = "Min Variance"
        elif strategy_choice == '3':
            result = optimizer.risk_parity_portfolio()
            strategy_name = "Risk Parity"
        elif strategy_choice == '4':
            result = optimizer.equal_weight_portfolio()
            strategy_name = "Equal Weight"
        elif strategy_choice == '5':
            print("\n📊 Сравнение стратегий:")
            comparison = optimizer.compare_strategies()
            print(f"\n{comparison.to_string(index=False)}")
            
            print("\n🏆 Рекомендуемая стратегия: Max Sharpe Ratio")
            result = optimizer.max_sharpe_portfolio()
            strategy_name = "Max Sharpe Ratio"
        else:
            print("❌ Неверный выбор")
            return
        
        print(f"\n✅ Оптимизация завершена: {strategy_name}")
        
        # 5. Выводим результаты
        print(f"\n{'='*80}")
        print(f"📊 Результаты оптимизации")
        print(f"{'='*80}")
        print(f"Стратегия:          {strategy_name}")
        print(f"Ожидаемая доходность: {result['expected_return']*100:>6.2f}%")
        print(f"Волатильность:      {result['volatility']*100:>6.2f}%")
        print(f"Sharpe Ratio:       {result['sharpe_ratio']:>6.2f}")
        
        print(f"\n{'Состав портфеля:'}")
        print(f"\n{'Тикер':<10} {'Вес':>8} {'Прогноз':>10}")
        print(f"{'-'*80}")
        
        # Фильтруем только значимые позиции (> 1%)
        significant_weights = {k: v for k, v in result['weights'].items() if v > 0.01}
        
        # Рассчитываем ожидаемую доходность портфеля на основе ML прогнозов
        expected_portfolio_return = 0.0
        
        for ticker, weight in sorted(significant_weights.items(), key=lambda x: x[1], reverse=True):
            pred_row = predictions[predictions['ticker'] == ticker]
            pred_return = pred_row['change_percent'].values[0] if not pred_row.empty else 0
            expected_portfolio_return += weight * pred_return
            print(f"{ticker:<10} {weight*100:>7.1f}% {pred_return:>+9.2f}%")
        
        print(f"\n{'='*80}")
        print(f"🎯 Прогнозируемая доходность портфеля (на основе ML): {expected_portfolio_return:>+7.2f}%")
        print(f"📊 Историческая доходность (MPT):                   {result['expected_return']*100:>+7.2f}%")
        print(f"{'='*80}")
        
        # 6. Создаём портфель
        print(f"\n💰 Шаг 5: Создание портфеля...")
        
        initial_capital = float(input("\nВведите начальный капитал (₽): ").strip())
        
        portfolio = Portfolio("Оптимизированный портфель", initial_cash=initial_capital)
        
        # Получаем текущие цены
        prices = manager.get_latest_prices(list(significant_weights.keys()))
        
        # Распределяем капитал
        investable_cash = initial_capital
        
        for ticker, weight in significant_weights.items():
            if ticker in prices:
                target_value = initial_capital * weight
                price = prices[ticker]
                shares = int(target_value / price)
                
                if shares > 0:
                    try:
                        portfolio.add_position(ticker, shares, price)
                        print(f"  ✅ {ticker}: {shares} акций по {price:.2f} ₽")
                    except Exception as e:
                        print(f"  ⚠️  {ticker}: {e}")
        
        # 7. Анализ рисков
        print(f"\n🛡️  Шаг 6: Анализ рисков...")
        
        risk_manager = RiskManager(returns)
        risk_manager.print_risk_report(
            result['weights'],
            risk_free_rate=0.08,
            portfolio_value=initial_capital
        )
        
        # 8. Добавляем прогноз в метаданные портфеля
        portfolio.ml_expected_return = expected_portfolio_return
        portfolio.mpt_expected_return = result['expected_return'] * 100
        
        # Сохраняем портфель
        portfolio.print_summary()
        
        # Выводим дополнительную информацию о прогнозе
        print(f"\n{'='*80}")
        print(f"🔮 Прогноз доходности")
        print(f"{'='*80}")
        print(f"ML Модель (прогноз):        {expected_portfolio_return:>+7.2f}%  ({initial_capital * expected_portfolio_return / 100:>+,.2f} ₽)")
        print(f"MPT (историческая):         {result['expected_return']*100:>+7.2f}%  ({initial_capital * result['expected_return']:>+,.2f} ₽)")
        print(f"{'='*80}\n")
        
        save = input("\n💾 Сохранить портфель? (y/n): ").strip().lower()
        if save == 'y':
            filepath = portfolio.save()
            print(f"✅ Портфель сохранён: {filepath}")
        
        # 9. Визуализация
        visualize = input("\n📊 Построить графики? (y/n): ").strip().lower()
        if visualize == 'y':
            visualize_portfolio(optimizer, result, portfolio)
        
    except Exception as e:
        print(f"❌ Ошибка оптимизации: {e}")
        import traceback
        traceback.print_exc()


def analyze_existing_portfolio():
    """Анализ существующего портфеля."""
    print("\n" + "="*80)
    print("🔍 Анализ существующего портфеля")
    print("="*80)
    
    # Ищем сохранённые портфели
    if not os.path.exists('portfolios'):
        print("❌ Папка portfolios не найдена")
        return
    
    files = [f for f in os.listdir('portfolios') if f.endswith('.json')]
    
    if not files:
        print("❌ Сохранённые портфели не найдены")
        return
    
    print("\nДоступные портфели:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file}")
    
    try:
        choice = int(input("\nВыберите портфель (номер): ").strip())
    except ValueError:
        print("❌ Неверный ввод")
        return
    
    if choice < 1 or choice > len(files):
        print("❌ Неверный выбор")
        return
    
    filepath = os.path.join('portfolios', files[choice - 1])
    
    try:
        # Загружаем портфель
        portfolio = Portfolio.load(filepath)
        
        # Обновляем цены
        manager = PortfolioManager()
        tickers = list(portfolio.positions.keys())
        prices = manager.get_latest_prices(tickers)
        portfolio.update_prices(prices)
        
        # Выводим сводку
        portfolio.print_summary()
        
        # Анализ рисков
        print("\n🛡️  Анализ рисков...")
        
        returns = manager.get_returns_data(tickers, days=365)
        risk_manager = RiskManager(returns)
        
        weights = portfolio.get_weights()
        risk_manager.print_risk_report(
            weights,
            risk_free_rate=0.08,
            portfolio_value=portfolio.get_total_value()
        )
    except Exception as e:
        print(f"❌ Ошибка анализа портфеля: {e}")
        import traceback
        traceback.print_exc()


def rebalancing_recommendations():
    """Рекомендации по ребалансировке."""
    print("\n" + "="*80)
    print("⚖️  Рекомендации по ребалансировке")
    print("="*80)
    
    # Ищем сохранённые портфели
    if not os.path.exists('portfolios'):
        print("❌ Папка portfolios не найдена")
        return
    
    files = [f for f in os.listdir('portfolios') if f.endswith('.json')]
    
    if not files:
        print("❌ Сохранённые портфели не найдены")
        return
    
    print("\nДоступные портфели:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file}")
    
    try:
        choice = int(input("\nВыберите портфель (номер): ").strip())
    except ValueError:
        print("❌ Неверный ввод")
        return
    
    if choice < 1 or choice > len(files):
        print("❌ Неверный выбор")
        return
    
    filepath = os.path.join('portfolios', files[choice - 1])
    
    try:
        # Загружаем портфель
        portfolio = Portfolio.load(filepath)
        
        # Обновляем цены
        manager = PortfolioManager()
        tickers = list(portfolio.positions.keys())
        prices = manager.get_latest_prices(tickers)
        portfolio.update_prices(prices)
        
        print(f"\n📊 Текущий портфель:")
        portfolio.print_summary()
        
        # Загружаем последние прогнозы
        print(f"\n🔄 Загружаем новые прогнозы...")
        predictions = manager.load_predictions()
        
        # Выбираем стратегию ребалансировки
        print(f"\n⚙️  Выберите стратегию ребалансировки:")
        print("  1. На основе новых прогнозов (Max Sharpe)")
        print("  2. Equal Weight (равные веса)")
        print("  3. Сохранить текущие веса")
        
        strat_choice = input("\nСтратегия (1-3): ").strip()
        
        if strat_choice == '1':
            # Оптимизация на основе прогнозов
            print(f"\n🔧 Оптимизация на основе новых прогнозов...")
            
            # Берём топ акций
            top_n = len(tickers) * 2  # Расширяем выбор
            top_stocks = predictions.nlargest(top_n, 'change_percent')
            new_tickers = top_stocks['ticker'].tolist()
            
            # Загружаем данные
            returns = manager.get_returns_data(new_tickers, days=180)
            
            # Оптимизируем
            optimizer = PortfolioOptimizer(returns, risk_free_rate=0.08)
            result = optimizer.max_sharpe_portfolio()
            
            target_weights = result['weights']
            print(f"✅ Новые оптимальные веса рассчитаны")
            
        elif strat_choice == '2':
            # Equal Weight
            target_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
            print(f"✅ Равные веса для {len(tickers)} акций")
            
        elif strat_choice == '3':
            # Сохранить текущие
            target_weights = portfolio.get_weights()
            print(f"✅ Используем текущие веса")
        else:
            print("❌ Неверный выбор")
            return
        
        # Создаём ребалансировщик
        print(f"\n💰 Параметры ребалансировки:")
        print(f"  Комиссия брокера: 0.05%")
        print(f"  Налог на прибыль: 13%")
        print(f"  Мин. сумма сделки: 1,000 ₽")
        
        rebalancer = PortfolioRebalancer(
            commission_rate=0.0005,
            tax_rate=0.13,
            min_trade_value=1000.0
        )
        
        # Проверяем необходимость ребалансировки
        current_weights = portfolio.get_weights()
        
        print(f"\n🔍 Проверка отклонений...")
        
        # Threshold-based check
        threshold = 0.05  # 5%
        needs_rebalance, deviations = rebalancer.check_threshold_rebalance(
            current_weights, target_weights, threshold
        )
        
        if not needs_rebalance:
            print(f"\n✅ Ребалансировка не требуется!")
            print(f"   Все отклонения < {threshold*100}%")
            
            # Показываем отклонения
            print(f"\n📊 Текущие отклонения:")
            for ticker in sorted(deviations.keys(), key=lambda t: deviations[t], reverse=True)[:5]:
                print(f"  {ticker}: {deviations[ticker]*100:.2f}%")
            return
        
        print(f"\n⚠️  Обнаружены значительные отклонения (> {threshold*100}%)")
        
        # Рассчитываем необходимые сделки
        print(f"\n💼 Расчёт сделок...")
        
        trades = rebalancer.calculate_rebalance_trades(
            portfolio.positions,
            target_weights,
            prices,
            portfolio.get_total_value()
        )
        
        if not trades:
            print(f"\n✅ Мелкие отклонения, сделки не требуются")
            return
        
        # Создаём отчёт
        print(f"\n📋 Отчёт по ребалансировке:")
        report = rebalancer.generate_rebalance_report(
            current_weights, target_weights, trades, deviations
        )
        
        print(f"\n{report.to_string(index=False)}")
        
        # Рассчитываем издержки
        costs = rebalancer.calculate_rebalance_cost(trades)
        
        print(f"\n💸 Издержки ребалансировки:")
        print(f"  Комиссии:          {costs['total_commission']:>12,.2f} ₽")
        print(f"  Налоги:            {costs['total_tax']:>12,.2f} ₽")
        print(f"  Итого издержки:    {costs['total_cost']:>12,.2f} ₽")
        print(f"\n  Покупка:           {costs['buy_value']:>12,.2f} ₽")
        print(f"  Продажа (чистая):  {costs['sell_proceeds']:>12,.2f} ₽")
        print(f"  Нужно добавить:    {costs['net_cost']:>12,.2f} ₽")
        
        # Оптимизация с учётом издержек
        print(f"\n🤔 Стоит ли ребалансировать?")
        
        expected_return = 0.15  # 15% годовых (можно взять из оптимизации)
        should_rebalance, reason = rebalancer.optimize_rebalance_with_costs(
            trades, expected_return, time_horizon_days=365
        )
        
        if should_rebalance:
            print(f"✅ ДА - {reason}")
        else:
            print(f"❌ НЕТ - {reason}")
        
        # Tax Loss Harvesting
        print(f"\n🧾 Tax Loss Harvesting (продажа убыточных позиций):")
        
        tlh_recommendations = rebalancer.tax_loss_harvesting(
            portfolio.positions, prices, min_loss_percent=0.10
        )
        
        if tlh_recommendations:
            print(f"\n  Найдено {len(tlh_recommendations)} убыточных позиций:")
            for rec in tlh_recommendations[:5]:
                print(f"  {rec['ticker']}: {rec['loss_percent']*100:+.2f}% "
                      f"(экономия налогов: {rec['tax_benefit']:.2f} ₽)")
        else:
            print(f"  Убыточных позиций не найдено")
        
        # Детальный план сделок
        print(f"\n📝 Детальный план сделок:")
        
        for ticker, trade in sorted(trades.items()):
            action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
            print(f"\n  {action_emoji} {trade['action']} {ticker}:")
            print(f"     Акций: {trade['shares']}")
            print(f"     Цена: {trade['price']:.2f} ₽")
            print(f"     Сумма: {trade['value']:,.2f} ₽")
            print(f"     Комиссия: {trade['commission']:.2f} ₽")
            if 'tax' in trade:
                print(f"     Налог: {trade['tax']:.2f} ₽")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


def visualize_portfolio(optimizer: PortfolioOptimizer, portfolio_result: dict, portfolio: Portfolio):
    """
    Визуализация портфеля.
    
    Args:
        optimizer: PortfolioOptimizer
        portfolio_result: Результат оптимизации
        portfolio: Portfolio объект
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Efficient Frontier
    print("\n📊 Строим Efficient Frontier...")
    try:
        frontier = optimizer.efficient_frontier(num_portfolios=50)
        
        ax = axes[0, 0]
        ax.scatter(frontier['volatility'], frontier['return'], c=frontier['sharpe_ratio'], 
                  cmap='viridis', s=50, alpha=0.6)
        ax.scatter(portfolio_result['volatility'], portfolio_result['expected_return'], 
                  color='red', s=200, marker='*', label='Оптимальный портфель')
        ax.set_xlabel('Волатильность (риск)')
        ax.set_ylabel('Ожидаемая доходность')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=frontier['sharpe_ratio'].min(), 
                                                     vmax=frontier['sharpe_ratio'].max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Sharpe Ratio')
        
    except Exception as e:
        print(f"⚠️  Ошибка построения Efficient Frontier: {e}")
    
    # 2. Веса портфеля (пирог)
    ax = axes[0, 1]
    weights = portfolio.get_weights()
    
    if weights:
        labels = list(weights.keys())
        sizes = list(weights.values())
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Распределение активов')
    
    # 3. Доходности позиций
    ax = axes[1, 0]
    returns = portfolio.get_returns()
    
    if returns:
        tickers = list(returns.keys())
        values = list(returns.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        ax.barh(tickers, values, color=colors, alpha=0.7)
        ax.set_xlabel('Доходность (%)')
        ax.set_title('Доходность по позициям')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, axis='x')
    
    # 4. Стоимость позиций
    ax = axes[1, 1]
    summary = portfolio.get_summary()
    
    if summary['positions']:
        tickers = [p['ticker'] for p in summary['positions']]
        values = [p['value'] for p in summary['positions']]
        
        ax.barh(tickers, values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Стоимость (₽)')
        ax.set_title('Стоимость позиций')
        ax.grid(True, axis='x')
    
    plt.tight_layout()
    
    # Сохраняем
    os.makedirs('portfolio_charts', exist_ok=True)
    filename = f"portfolio_charts/portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ График сохранён: {filename}")
    
    plt.show()


def main():
    """Главное меню."""
    print("\n" + "="*80)
    print("🎯 Управление портфелем")
    print("="*80)
    print("\nДоступные опции:")
    print("  1. Создать оптимальный портфель")
    print("  2. Анализ существующего портфеля")
    print("  3. Рекомендации по ребалансировке")
    print("  0. Выход")
    
    choice = input("\nВыберите действие (0-3): ").strip()
    
    if choice == '1':
        create_optimal_portfolio()
    elif choice == '2':
        analyze_existing_portfolio()
    elif choice == '3':
        rebalancing_recommendations()
    elif choice == '0':
        print("👋 До свидания!")
        return
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

