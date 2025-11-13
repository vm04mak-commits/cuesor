"""
Скрипт для обучения моделей прогнозирования.
Обучает модели на исторических данных из БД или CSV файлов.
"""

from core import Orchestrator, Config, Logger
from predict.model_trainer import ModelTrainer
from datetime import datetime, timedelta
import argparse


def train_single_ticker(ticker: str, models: list = None, from_db: bool = True, horizon: int = 1):
    """
    Обучение моделей для одного тикера.
    
    Args:
        ticker (str): Тикер акции
        models (list): Список моделей для обучения
        from_db (bool): Загружать данные из БД
        horizon (int): Горизонт прогноза в днях
    
    Returns:
        None
    """
    print("=" * 80)
    print(f"ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ {ticker}")
    print("=" * 80)
    print()
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("Training")
    trainer = ModelTrainer(config, logger)
    
    # Доступные модели
    if models is None:
        models = ['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']
    
    print(f"Тикер: {ticker}")
    print(f"Модели: {', '.join(models)}")
    print(f"Источник данных: {'База данных' if from_db else 'CSV файлы'}")
    print(f"Горизонт прогноза: {horizon} дней")
    print()
    
    try:
        # Обучение
        results = trainer.train_multiple_models(
            ticker=ticker,
            models=models,
            from_db=from_db,
            target_horizon=horizon
        )
        
        # Вывод результатов
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("=" * 80)
        
        for model_type, data in results.items():
            if 'error' in data:
                print(f"\n❌ {model_type.upper()}: Ошибка - {data['error']}")
            else:
                metrics = data['metrics']['test_metrics']
                print(f"\n✅ {model_type.upper()}:")
                print(f"   R² Score:  {metrics['r2']:.4f}")
                print(f"   MAE:       {metrics['mae']:.2f} ₽")
                print(f"   RMSE:      {metrics['rmse']:.2f} ₽")
                print(f"   Модель:    {data['model_path']}")
        
        # Лучшая модель
        best_model_path = trainer.get_best_model(ticker)
        if best_model_path:
            print(f"\n🏆 Лучшая модель сохранена: {best_model_path}")
        
        print("\n" + "=" * 80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {str(e)}")
        print()


def collect_and_train(ticker: str, days: int = 365):
    """
    Сбор данных и обучение моделей.
    
    Args:
        ticker (str): Тикер акции
        days (int): Количество дней истории для сбора
    
    Returns:
        None
    """
    print("=" * 80)
    print(f"ПОЛНЫЙ ЦИКЛ: СБОР ДАННЫХ + ОБУЧЕНИЕ")
    print("=" * 80)
    print()
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("Training")
    
    # 1. Сбор данных
    print("📊 Шаг 1: Сбор данных...")
    print()
    
    from data import DataCollector
    from analysis import Analyzer
    
    collector = DataCollector(config, logger)
    analyzer = Analyzer(config, logger)
    
    # Даты
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"Тикер: {ticker}")
    print(f"Период: {start_date} — {end_date}")
    print()
    
    # Загрузка котировок
    data = collector.fetch_stock_data(ticker, start_date, end_date)
    print(f"✅ Получено {len(data)} котировок")
    
    # Расчёт индикаторов
    print("\n📈 Расчёт индикаторов...")
    analysis = analyzer.analyze(data, ticker=ticker, save_indicators=True)
    print("✅ Индикаторы рассчитаны и сохранены")
    
    # 2. Обучение моделей
    print("\n🤖 Шаг 2: Обучение моделей...")
    print()
    
    trainer = ModelTrainer(config, logger)
    results = trainer.train_multiple_models(
        ticker=ticker,
        models=['linear', 'ridge', 'random_forest', 'gradient_boosting'],
        from_db=True,
        target_horizon=1
    )
    
    # Результаты
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 80)
    
    for model_type, data_result in results.items():
        if 'error' not in data_result:
            metrics = data_result['metrics']['test_metrics']
            print(f"\n{model_type.upper()}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}₽")
    
    print("\n" + "=" * 80)
    print("✅ ПОЛНЫЙ ЦИКЛ ЗАВЕРШЁН")
    print("=" * 80)
    print()


def batch_training(tickers: list, days: int = 365):
    """
    Пакетное обучение для нескольких тикеров.
    
    Args:
        tickers (list): Список тикеров
        days (int): Количество дней истории
    
    Returns:
        None
    """
    print("=" * 80)
    print(f"ПАКЕТНОЕ ОБУЧЕНИЕ ДЛЯ {len(tickers)} ТИКЕРОВ")
    print("=" * 80)
    print()
    
    results_summary = {}
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Обработка {ticker}...")
        print("-" * 80)
        
        try:
            collect_and_train(ticker, days)
            results_summary[ticker] = "✅ Успешно"
        except Exception as e:
            print(f"❌ Ошибка для {ticker}: {str(e)}")
            results_summary[ticker] = f"❌ Ошибка: {str(e)}"
    
    # Сводка
    print("\n" + "=" * 80)
    print("СВОДКА ПАКЕТНОГО ОБУЧЕНИЯ")
    print("=" * 80)
    
    for ticker, status in results_summary.items():
        print(f"{ticker}: {status}")
    
    print("=" * 80)
    print()


def main():
    """
    Главная функция с CLI интерфейсом.
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Обучение моделей для прогнозирования акций')
    
    parser.add_argument('ticker', nargs='?', help='Тикер акции (например, SBER)')
    parser.add_argument('--models', nargs='+', 
                       choices=['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'],
                       help='Модели для обучения')
    parser.add_argument('--from-csv', action='store_true', help='Загружать данные из CSV вместо БД')
    parser.add_argument('--horizon', type=int, default=1, help='Горизонт прогноза в днях (default: 1)')
    parser.add_argument('--collect', action='store_true', help='Сначала собрать данные')
    parser.add_argument('--days', type=int, default=365, help='Дней истории для сбора (default: 365)')
    parser.add_argument('--batch', nargs='+', help='Пакетное обучение для списка тикеров')
    
    args = parser.parse_args()
    
    # Интерактивный режим
    if not args.ticker and not args.batch:
        print("\n" + "=" * 80)
        print(" " * 20 + "🤖 ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("=" * 80)
        print()
        print("ВЫБЕРИТЕ РЕЖИМ:")
        print()
        print("  1. Обучить модели для одного тикера")
        print("  2. Собрать данные + обучить модели")
        print("  3. Пакетное обучение (несколько тикеров)")
        print()
        print("  0. Выход")
        print()
        
        choice = input("Ваш выбор: ").strip()
        
        if choice == "1":
            ticker = input("\nВведите тикер (например, SBER): ").strip().upper()
            if ticker:
                train_single_ticker(ticker, from_db=True, horizon=1)
        
        elif choice == "2":
            ticker = input("\nВведите тикер (например, SBER): ").strip().upper()
            days = input("Количество дней истории (default: 365): ").strip()
            days = int(days) if days else 365
            
            if ticker:
                collect_and_train(ticker, days)
        
        elif choice == "3":
            tickers_input = input("\nВведите тикеры через пробел (например, SBER GAZP LKOH): ").strip().upper()
            tickers = tickers_input.split()
            
            if tickers:
                batch_training(tickers, days=365)
        
        elif choice == "0":
            print("До свидания!")
        
        else:
            print("❌ Неверный выбор")
    
    # CLI режим
    elif args.batch:
        batch_training(args.batch, args.days)
    
    elif args.collect:
        collect_and_train(args.ticker, args.days)
    
    else:
        train_single_ticker(
            ticker=args.ticker,
            models=args.models,
            from_db=not args.from_csv,
            horizon=args.horizon
        )


if __name__ == "__main__":
    main()









