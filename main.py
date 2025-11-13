"""
Главный файл запуска Investment AI Assistant.
Примеры использования системы.
"""

from core import Orchestrator
from datetime import datetime, timedelta


def example_single_stock_analysis():
    """
    Пример анализа одной акции.
    
    Returns:
        None
    """
    print("=" * 80)
    print("ПРИМЕР 1: Анализ одной акции")
    print("=" * 80)
    print()
    
    # Инициализация системы
    orchestrator = Orchestrator()
    
    # Параметры анализа
    ticker = "SBER"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Тикер: {ticker}")
    print(f"Период: {start_date} - {end_date}")
    print()
    
    try:
        # Запуск полного пайплайна
        print("Запуск анализа...")
        results = orchestrator.run_pipeline(ticker, start_date, end_date)
        
        # Вывод результатов
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА")
        print("=" * 80)
        
        # Статистика
        if 'analysis' in results and 'statistics' in results['analysis']:
            stats = results['analysis']['statistics']
            print(f"\nТекущая цена: {stats.get('current', 0):.2f} ₽")
            print(f"Изменение за период: {stats.get('change', 0):.2f}%")
        
        # Тренд
        if 'analysis' in results and 'trend' in results['analysis']:
            trend = results['analysis']['trend']
            print(f"\nТренд: {trend.get('trend', 'unknown').upper()}")
            print(f"Сила: {trend.get('strength', 'unknown').upper()}")
        
        # Рекомендация
        if 'prediction' in results and 'recommendation' in results['prediction']:
            rec = results['prediction']['recommendation']
            print(f"\n{'='*80}")
            print(f"РЕКОМЕНДАЦИЯ: {rec.get('action', 'hold').upper()}")
            print(f"{'='*80}")
            print(f"Причина: {rec.get('reason', 'Нет данных')}")
            print(f"Уверенность: {rec.get('confidence', 'unknown').upper()}")
        
        # Отчёт
        if 'report' in results:
            print(f"\nОтчёт сохранён: {results['report'].get('html_report', 'N/A')}")
        
        print("\n✅ Анализ завершён успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    print()


def example_data_collection():
    """
    Пример сбора данных.
    
    Returns:
        None
    """
    print("=" * 80)
    print("ПРИМЕР 2: Сбор данных")
    print("=" * 80)
    print()
    
    from core import Config, Logger
    from data import DataCollector
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("Example")
    collector = DataCollector(config, logger)
    
    # Параметры
    ticker = "GAZP"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"Сбор данных для {ticker}")
    print(f"Период: {start_date} - {end_date}")
    print()
    
    try:
        # Сбор данных
        data = collector.fetch_stock_data(ticker, start_date, end_date)
        
        print(f"Получено {len(data)} записей\n")
        print("Первые 5 записей:")
        print(data.head())
        print()
        print("Последние 5 записей:")
        print(data.tail())
        
        print("\n✅ Данные получены успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    print()


def example_technical_analysis():
    """
    Пример технического анализа.
    
    Returns:
        None
    """
    print("=" * 80)
    print("ПРИМЕР 3: Технический анализ")
    print("=" * 80)
    print()
    
    from core import Config, Logger
    from data import DataCollector
    from analysis import Analyzer
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("Example")
    collector = DataCollector(config, logger)
    analyzer = Analyzer(config, logger)
    
    # Параметры
    ticker = "LKOH"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    print(f"Технический анализ для {ticker}")
    print(f"Период: {start_date} - {end_date}")
    print()
    
    try:
        # Сбор данных
        data = collector.fetch_stock_data(ticker, start_date, end_date)
        
        # Анализ
        analysis = analyzer.analyze(data)
        
        # Вывод результатов
        print("ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:")
        if 'technical' in analysis and 'current_values' in analysis['technical']:
            current = analysis['technical']['current_values']
            print(f"  Цена: {current.get('close', 0):.2f} ₽")
            print(f"  RSI: {current.get('rsi', 0):.2f}")
            print(f"  SMA(20): {current.get('sma_20', 0):.2f} ₽")
            print(f"  SMA(50): {current.get('sma_50', 0):.2f} ₽")
        
        print("\nВОЛАТИЛЬНОСТЬ:")
        if 'volatility' in analysis:
            vol = analysis['volatility']
            print(f"  Дневная: {vol.get('daily_volatility', 0):.4f}")
            print(f"  Годовая: {vol.get('annual_volatility', 0):.4f}")
        
        print("\nТРЕНД:")
        if 'trend' in analysis:
            trend = analysis['trend']
            print(f"  Направление: {trend.get('trend', 'unknown').upper()}")
            print(f"  Сила: {trend.get('strength', 'unknown').upper()}")
        
        print("\n✅ Анализ завершён успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    print()


def example_prediction():
    """
    Пример прогнозирования.
    
    Returns:
        None
    """
    print("=" * 80)
    print("ПРИМЕР 4: Прогнозирование")
    print("=" * 80)
    print()
    
    from core import Config, Logger
    from data import DataCollector
    from predict import Predictor
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("Example")
    collector = DataCollector(config, logger)
    predictor = Predictor(config, logger)
    
    # Параметры
    ticker = "ROSN"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Прогнозирование для {ticker}")
    print(f"Период обучения: {start_date} - {end_date}")
    print(f"Горизонт прогноза: 30 дней")
    print()
    
    try:
        # Сбор данных
        data = collector.fetch_stock_data(ticker, start_date, end_date)
        
        # Прогнозирование
        prediction = predictor.predict(data, horizon=30)
        
        # Вывод результатов
        current_price = float(data['close'].iloc[-1])
        
        print("РЕЗУЛЬТАТЫ ПРОГНОЗА:")
        
        if 'linear_regression' in prediction:
            lr = prediction['linear_regression']
            print(f"\nЛинейная регрессия:")
            print(f"  Прогноз: {lr.get('predicted_price', 0):.2f} ₽")
            print(f"  Направление: {lr.get('direction', 'unknown').upper()}")
        
        if 'time_series' in prediction:
            ts = prediction['time_series']
            print(f"\nВременные ряды:")
            print(f"  Прогноз: {ts.get('predicted_price', 0):.2f} ₽")
            print(f"  Направление: {ts.get('direction', 'unknown').upper()}")
        
        if 'ensemble' in prediction:
            ens = prediction['ensemble']
            print(f"\nАнсамбль (итоговый):")
            print(f"  Текущая цена: {current_price:.2f} ₽")
            print(f"  Прогноз: {ens.get('predicted_price', 0):.2f} ₽")
            print(f"  Направление: {ens.get('direction', 'unknown').upper()}")
            print(f"  Уверенность: {ens.get('confidence', 'unknown').upper()}")
        
        if 'recommendation' in prediction:
            rec = prediction['recommendation']
            print(f"\nРЕКОМЕНДАЦИЯ: {rec.get('action', 'hold').upper()}")
            print(f"  {rec.get('reason', 'Нет данных')}")
        
        print("\n✅ Прогноз завершён успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    print()


def example_api_server():
    """
    Пример запуска API сервера.
    
    Returns:
        None
    """
    print("=" * 80)
    print("ПРИМЕР 5: Запуск API сервера")
    print("=" * 80)
    print()
    
    orchestrator = Orchestrator()
    
    print("Инициализация API сервера...")
    api_server = orchestrator.initialize_module("api")
    
    print(f"Сервер запущен на http://{api_server.host}:{api_server.port}")
    print()
    print("Доступные endpoints:")
    print("  GET  /                         - Информация об API")
    print("  GET  /health                   - Проверка состояния")
    print("  GET  /api/stocks/<ticker>      - Информация об акции")
    print("  POST /api/analyze/<ticker>     - Анализ акции")
    print("  POST /api/predict/<ticker>     - Прогноз по акции")
    print("  POST /api/report/<ticker>      - Генерация отчёта")
    print()
    print("Нажмите Ctrl+C для остановки сервера")
    print()
    
    try:
        api_server.run()
    except KeyboardInterrupt:
        print("\n\nСервер остановлен")


def example_update_documentation():
    """
    Пример обновления документации.
    
    Returns:
        None
    """
    print("=" * 80)
    print("ПРИМЕР 6: Обновление документации")
    print("=" * 80)
    print()
    
    orchestrator = Orchestrator()
    
    print("Сканирование проекта и извлечение docstring'ов...")
    orchestrator.update_documentation()
    
    print("\n✅ Документация обновлена!")
    print("\nФайлы документации:")
    print("  docs/autodoc.md       - Автодокументация функций")
    print("  docs/session_log.json - Лог вызовов")
    print("  docs/roadmap.md       - План развития")
    print("  docs/readme.md        - Описание системы")
    print()


def main():
    """
    Главная функция с меню примеров.
    
    Returns:
        None
    """
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Investment AI Assistant" + " " * 35 + "║")
    print("║" + " " * 78 + "║")
    print("║" + " " * 15 + "Локальный ИИ-ассистент для фондового рынка РФ" + " " * 17 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    while True:
        print("ВЫБЕРИТЕ ПРИМЕР:")
        print()
        print("  1. Полный анализ одной акции (с отчётом)")
        print("  2. Сбор данных с MOEX")
        print("  3. Технический анализ")
        print("  4. Прогнозирование")
        print("  5. Запуск API сервера")
        print("  6. Обновление документации")
        print()
        print("  0. Выход")
        print()
        
        choice = input("Ваш выбор: ").strip()
        print()
        
        if choice == "1":
            example_single_stock_analysis()
        elif choice == "2":
            example_data_collection()
        elif choice == "3":
            example_technical_analysis()
        elif choice == "4":
            example_prediction()
        elif choice == "5":
            example_api_server()
        elif choice == "6":
            example_update_documentation()
        elif choice == "0":
            print("До свидания! 👋")
            break
        else:
            print("❌ Неверный выбор. Попробуйте снова.\n")


if __name__ == "__main__":
    main()









