"""
СКРИПТ 5: Продвинутый бэктестинг универсальной модели

Что делает:
- Загружает универсальную модель
- Тестирует на исторических данных
- Walk-Forward Analysis
- Monte Carlo симуляции
- Расширенные метрики (Sharpe, Sortino, Calmar, etc.)
- Визуализация результатов

Запуск: python 5_backtest_model.py
"""

from core import Config, Logger, Database
from predict.universal_model import UniversalModelTrainer
from backtesting import AdvancedBacktester
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from datetime import datetime


def simple_backtest(test_period_days: int = 30):
    """
    Простой бэктестинг универсальной модели.
    
    Args:
        test_period_days (int): Период для тестирования в днях
    
    Returns:
        Dict: Результаты бэктестинга
    """
    print("=" * 80)
    print("СКРИПТ 5: БЭКТЕСТИНГ МОДЕЛИ")
    print("=" * 80)
    print()
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("Backtest")
    trainer = UniversalModelTrainer(config, logger)
    db_path = config.base_path / "data" / "market_data.db"
    database = Database(db_path, logger)
    
    try:
        # Загрузка модели
        print("📦 Загрузка универсальной модели...")
        model_data = trainer.load_model()
        
        model = model_data['model']
        scaler = model_data['scaler']
        ticker_encoder = model_data['ticker_encoder']
        features = model_data['results']['features']
        
        print("✅ Модель загружена")
        print(f"   Обучена: {model_data['results']['trained_at']}")
        print(f"   Test R²: {model_data['results']['test_metrics']['r2']:.4f}")
        print()
        
        # Загрузка данных
        print(f"📊 Бэктестинг на последних {test_period_days} днях...")
        print()
        
        tickers = database.get_available_tickers()
        
        all_predictions = []
        all_actuals = []
        ticker_results = []
        returns_list = []
        
        for ticker in tickers:
            try:
                quotes = database.load_quotes(ticker)
                
                if len(quotes) < test_period_days + 30:
                    continue
                
                # Берём последние N дней для теста
                test_data = quotes.tail(test_period_days + 1)  # +1 для целевой переменной
                
                for i in range(len(test_data) - 1):
                    current_row = test_data.iloc[i:i+1]
                    actual_next_price = test_data.iloc[i+1]['close']
                    current_price = current_row['close'].iloc[0]
                    
                    # Загружаем индикаторы
                    indicators = database.load_indicators(ticker)
                    if indicators.empty:
                        continue
                    
                    # Находим соответствующий индикатор
                    current_date = current_row['date'].iloc[0]
                    indicator_row = indicators[indicators.index == current_date]
                    
                    if indicator_row.empty:
                        continue
                    
                    # Подготовка данных для прогноза
                    pred_data = current_row.copy()
                    pred_data['ticker'] = ticker
                    
                    try:
                        pred_data['ticker_encoded'] = ticker_encoder.transform([ticker])[0]
                    except:
                        continue
                    
                    # Добавляем индикаторы
                    for col in indicator_row.columns:
                        pred_data[col] = indicator_row[col].iloc[0]
                    
                    # Дополнительные признаки
                    pred_data['price_change_1d'] = quotes['close'].pct_change(1).iloc[i] if i > 0 else 0
                    pred_data['price_change_5d'] = quotes['close'].pct_change(5).iloc[i] if i > 4 else 0
                    pred_data['volume_ma_ratio'] = quotes['volume'].iloc[i] / quotes['volume'].rolling(20).mean().iloc[i] if i > 19 else 1
                    
                    # Выбираем признаки
                    X_pred = pd.DataFrame()
                    for f in features:
                        if f in pred_data.columns:
                            X_pred[f] = pred_data[f]
                        else:
                            X_pred[f] = 0
                    
                    # Прогноз
                    X_scaled = scaler.transform(X_pred)
                    
                    # Проверяем тип модели (Deep Learning или классическая)
                    model_type = model_data.get('model_type', None)
                    if model_type in ['lstm', 'gru']:
                        # Для Deep Learning нужно reshape
                        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                        pred = model.predict(X_reshaped, verbose=0)
                        predicted_price = float(pred[0][0])
                    else:
                        # Для классических моделей
                        predicted_price = float(model.predict(X_scaled)[0])
                    
                    all_predictions.append(predicted_price)
                    all_actuals.append(actual_next_price)
                    
                    # Рассчитываем доходность
                    actual_return = (actual_next_price - current_price) / current_price
                    returns_list.append(actual_return)
                
                # Метрики по тикеру
                if all_actuals:
                    recent_actuals = all_actuals[-min(test_period_days, len(all_actuals)):]
                    recent_predictions = all_predictions[-min(test_period_days, len(all_predictions)):]
                    
                    if len(recent_actuals) > 0 and len(recent_predictions) > 0:
                        ticker_mae = mean_absolute_error(recent_actuals, recent_predictions)
                        ticker_results.append({
                            'ticker': ticker,
                            'mae': ticker_mae,
                            'predictions': len(recent_actuals)
                        })
            
            except Exception as e:
                logger.error(f"Error backtesting {ticker}: {e}")
                continue
        
        # Общие метрики
        if all_predictions and all_actuals:
            mae = mean_absolute_error(all_actuals, all_predictions)
            rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
            r2 = r2_score(all_actuals, all_predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) / np.array(all_actuals))) * 100
            
            print("=" * 80)
            print("РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
            print("=" * 80)
            print()
            print(f"📊 Общие метрики ({len(all_predictions)} прогнозов):")
            print(f"   R² Score: {r2:.4f}")
            print(f"   MAE:      {mae:.2f} ₽")
            print(f"   RMSE:     {rmse:.2f} ₽")
            print(f"   MAPE:     {mape:.2f}%")
            print()
            
            print("📈 Оценка качества:")
            if r2 > 0.7:
                print("   🏆 ОТЛИЧНО! Модель дает надежные прогнозы")
            elif r2 > 0.5:
                print("   ✅ ХОРОШО! Модель работает неплохо")
            elif r2 > 0.3:
                print("   ⚠️  СРЕДНЕ. Есть куда расти")
            else:
                print("   ⚠️  НИЗКОЕ. Модель требует улучшения")
            
            print()
            print(f"💡 Средняя ошибка прогноза: {mae:.2f} ₽ ({mape:.2f}%)")
            print()
            
            # Топ и худшие акции
            if ticker_results:
                df_results = pd.DataFrame(ticker_results)
                df_sorted = df_results.sort_values('mae')
                
                print("🏆 ТОП-10 акций с лучшими прогнозами:")
                for _, row in df_sorted.head(10).iterrows():
                    print(f"   {row['ticker']:<6} MAE: {row['mae']:.2f} ₽")
                
                print()
                print("⚠️  10 акций с худшими прогнозами:")
                for _, row in df_sorted.tail(10).iterrows():
                    print(f"   {row['ticker']:<6} MAE: {row['mae']:.2f} ₽")
            
            print()
            
            # Возвращаем результаты для дальнейшего анализа
            return {
                'predictions': all_predictions,
                'actuals': all_actuals,
                'returns': returns_list,
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
        
        else:
            print("❌ Недостаточно данных для бэктестинга")
            return None
    
    except FileNotFoundError:
        print("❌ Модель не найдена!")
        print("   Сначала запустите: python 3_train_universal_model.py")
        return None
    
    except Exception as e:
        print(f"❌ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def advanced_backtest():
    """
    Продвинутый бэктестинг с расширенными метриками.
    """
    print("\n" + "=" * 80)
    print("🔬 ПРОДВИНУТЫЙ БЭКТЕСТИНГ")
    print("=" * 80)
    print()
    
    # Сначала запускаем простой бэктестинг для получения данных
    print("Шаг 1: Базовый бэктестинг...")
    results = simple_backtest(test_period_days=60)
    
    if not results:
        return
    
    # Создаём продвинутый бэктестер
    backtester = AdvancedBacktester(risk_free_rate=0.08)
    
    # Преобразуем доходности в Series
    returns = pd.Series(results['returns'])
    
    print("\n" + "=" * 80)
    print("📊 РАСШИРЕННЫЕ МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 80)
    
    # Рассчитываем все метрики
    metrics = backtester.calculate_all_metrics(returns)
    
    print()
    print(f"💰 Общая доходность:      {metrics['total_return']*100:>8.2f}%")
    print(f"📈 Годовая доходность:    {metrics['annual_return']*100:>8.2f}%")
    print(f"📊 Волатильность:         {metrics['volatility']*100:>8.2f}%")
    print()
    print(f"⚡ Sharpe Ratio:          {metrics['sharpe_ratio']:>8.2f}")
    print(f"⚡ Sortino Ratio:         {metrics['sortino_ratio']:>8.2f}")
    print(f"⚡ Calmar Ratio:          {metrics['calmar_ratio']:>8.2f}")
    print()
    print(f"⚠️  Max Drawdown:          {metrics['max_drawdown']*100:>8.2f}%")
    print(f"⚠️  MDD Duration:          {metrics['max_dd_duration_days']:>8.0f} дней")
    print()
    print(f"🎯 Win Rate:              {metrics['win_rate']*100:>8.2f}%")
    print(f"🎯 Average Win:           {metrics['avg_win']*100:>8.4f}%")
    print(f"🎯 Average Loss:          {metrics['avg_loss']*100:>8.4f}%")
    print(f"🎯 Win/Loss Ratio:        {metrics['win_loss_ratio']:>8.2f}")
    print(f"🎯 Total Trades:          {metrics['total_trades']:>8.0f}")
    
    # Генерируем текстовый отчёт
    print("\n💾 Сохранение отчёта...")
    report_path = Path("backtest_reports")
    report_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_path / f"backtest_report_{timestamp}.txt"
    
    backtester.generate_report(metrics, save_path=str(report_file))
    
    return backtester, results, metrics


def monte_carlo_simulation_mode():
    """
    Режим Monte Carlo симуляции.
    """
    print("\n" + "=" * 80)
    print("🎲 MONTE CARLO СИМУЛЯЦИЯ")
    print("=" * 80)
    print()
    
    # Сначала получаем данные
    print("Получение исторических данных...")
    results = simple_backtest(test_period_days=60)
    
    if not results:
        return
    
    # Преобразуем доходности
    returns = pd.Series(results['returns'])
    
    print("\n" + "=" * 80)
    print("Параметры симуляции:")
    print("=" * 80)
    
    # Запрашиваем параметры
    try:
        num_sims = int(input("\nКоличество симуляций (по умолчанию 10,000): ").strip() or "10000")
        num_days = int(input("Горизонт симуляции в днях (по умолчанию 252): ").strip() or "252")
        initial_capital = float(input("Начальный капитал в рублях (по умолчанию 1,000,000): ").strip() or "1000000")
    except ValueError:
        print("Используем значения по умолчанию...")
        num_sims = 10000
        num_days = 252
        initial_capital = 1000000
    
    # Создаём бэктестер и запускаем симуляцию
    backtester = AdvancedBacktester(risk_free_rate=0.08)
    mc_results = backtester.monte_carlo_simulation(
        returns=returns,
        num_simulations=num_sims,
        num_days=num_days,
        initial_capital=initial_capital
    )
    
    # Визуализация
    print("\n📊 Создание визуализации...")
    chart_path = Path("backtest_charts")
    chart_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = chart_path / f"monte_carlo_{timestamp}.png"
    
    backtester.plot_monte_carlo_results(mc_results, save_path=str(chart_file))
    
    print("\n✅ MONTE CARLO СИМУЛЯЦИЯ ЗАВЕРШЕНА")


def walk_forward_mode():
    """
    Режим Walk-Forward Analysis.
    """
    print("\n" + "=" * 80)
    print("🔄 WALK-FORWARD ANALYSIS")
    print("=" * 80)
    print()
    print("⚠️  Функция в разработке...")
    print("   Требуется интеграция с переобучением модели")
    print()


def main():
    """
    Главная функция.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "ПРОДВИНУТЫЙ БЭКТЕСТИНГ")
    print("=" * 80)
    print()
    print("ВЫБЕРИТЕ РЕЖИМ ТЕСТИРОВАНИЯ:")
    print()
    print("  1. 🔹 Простой бэктестинг (7 дней)")
    print("  2. 🔹 Простой бэктестинг (14 дней)")
    print("  3. 🔹 Простой бэктестинг (30 дней)")
    print("  4. 🔹 Простой бэктестинг (60 дней)")
    print()
    print("  5. 🔬 Продвинутый бэктестинг (расширенные метрики)")
    print("  6. 🎲 Monte Carlo симуляция")
    print("  7. 🔄 Walk-Forward Analysis (в разработке)")
    print()
    print("  0. Выход")
    print()
    
    choice = input("Ваш выбор: ").strip()
    
    if choice == "1":
        simple_backtest(test_period_days=7)
    
    elif choice == "2":
        simple_backtest(test_period_days=14)
    
    elif choice == "3":
        simple_backtest(test_period_days=30)
    
    elif choice == "4":
        simple_backtest(test_period_days=60)
    
    elif choice == "5":
        advanced_backtest()
    
    elif choice == "6":
        monte_carlo_simulation_mode()
    
    elif choice == "7":
        walk_forward_mode()
    
    elif choice == "0":
        print("До свидания!")
    
    else:
        print("❌ Неверный выбор")
    
    print("\n" + "=" * 80)
    print("✅ БЭКТЕСТИНГ ЗАВЕРШЁН")
    print("=" * 80)
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # CLI режим
        if sys.argv[1] == "--advanced":
            advanced_backtest()
        elif sys.argv[1] == "--monte-carlo":
            monte_carlo_simulation_mode()
        elif sys.argv[1] == "--walk-forward":
            walk_forward_mode()
        else:
            days = int(sys.argv[1])
            simple_backtest(test_period_days=days)
    else:
        # Интерактивный режим
        main()
