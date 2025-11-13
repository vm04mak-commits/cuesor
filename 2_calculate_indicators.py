"""
СКРИПТ 2: Расчёт индикаторов для всех акций

Что делает:
- Загружает котировки из БД
- Рассчитывает технические индикаторы для каждой акции
- Сохраняет индикаторы в БД и CSV

Запуск: python 2_calculate_indicators.py
"""

from core import Config, Logger, Database
from analysis import Analyzer
import pandas as pd


def calculate_indicators_for_all(incremental: bool = True):
    """
    Расчёт индикаторов для всех акций в БД.
    
    Args:
        incremental (bool): Инкрементальное обновление (только новые данные)
    
    Returns:
        None
    """
    print("=" * 80)
    print("СКРИПТ 2: РАСЧЁТ ИНДИКАТОРОВ")
    print("=" * 80)
    print()
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("CalculateIndicators")
    db_path = config.base_path / "data" / "market_data.db"
    database = Database(db_path, logger)
    analyzer = Analyzer(config, logger)
    
    # Получение списка тикеров из БД
    print("📋 Получение списка акций из БД...")
    tickers = database.get_available_tickers()
    print(f"✅ Найдено {len(tickers)} акций")
    print(f"🔄 Режим: {'Инкрементальное обновление' if incremental else 'Полный пересчёт'}")
    print()
    
    if not tickers:
        print("❌ В базе данных нет акций!")
        print("   Сначала запустите: python 1_collect_all_stocks.py")
        return
    
    # Расчёт индикаторов
    print("🔄 Начинаем расчёт индикаторов...")
    print("-" * 80)
    
    success_count = 0
    updated_count = 0
    skipped_count = 0
    failed_count = 0
    failed_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}...", end=" ")
        
        try:
            # Загрузка котировок
            data = database.load_quotes(ticker)
            
            if data.empty or len(data) < 30:
                print("⚠️  Недостаточно данных")
                failed_count += 1
                failed_tickers.append(ticker)
                continue
            
            # Проверяем последнюю дату индикаторов
            need_update = True
            if incremental:
                last_indicator_date = analyzer.get_last_indicator_date(ticker)
                last_quote_date = data['date'].max().strftime('%Y-%m-%d')
                
                if last_indicator_date:
                    if last_indicator_date == last_quote_date:
                        print(f"⏭️  Актуально (последняя дата: {last_indicator_date})")
                        skipped_count += 1
                        need_update = False
                        continue
                    else:
                        print(f"🔄 Обновление с {last_indicator_date}...", end=" ")
                        updated_count += 1
                else:
                    print(f"📥 Первый расчёт...", end=" ")
                    success_count += 1
            else:
                success_count += 1
            
            if need_update:
                # Расчёт индикаторов
                analysis = analyzer.analyze(data, ticker=ticker, save_indicators=True)
                print(f"✅ {len(data)} записей")
        
        except Exception as e:
            print(f"❌ Ошибка: {str(e)[:50]}")
            failed_count += 1
            failed_tickers.append(ticker)
            continue
    
    # Итоги
    print()
    print("=" * 80)
    print("ИТОГИ РАСЧЁТА")
    print("=" * 80)
    
    if incremental:
        print(f"✅ Новых акций: {success_count}")
        print(f"🔄 Обновлено: {updated_count}")
        print(f"⏭️  Актуальных: {skipped_count}")
        print(f"❌ Ошибок: {failed_count}")
        print(f"\n📊 Обработано: {success_count + updated_count + skipped_count} из {len(tickers)}")
    else:
        print(f"✅ Успешно: {success_count}")
        print(f"❌ Ошибок: {failed_count}")
    
    if failed_tickers:
        print(f"\nНе удалось рассчитать: {', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            print(f"... и ещё {len(failed_tickers) - 10}")
    
    print()
    print("📊 Индикаторы сохранены в:")
    print(f"   - БД: {db_path}")
    print(f"   - CSV: {config.base_path / 'data' / 'csv' / '[TICKER]_indicators.csv'}")
    print()
    print("=" * 80)
    print("✅ РАСЧЁТ ЗАВЕРШЁН")
    print("=" * 80)
    print()
    print("Следующий шаг:")
    print("  python 3_train_universal_model.py")
    print()


def calculate_for_specific_tickers(tickers: list):
    """
    Расчёт индикаторов для конкретных акций.
    
    Args:
        tickers (list): Список тикеров
    
    Returns:
        None
    """
    print("=" * 80)
    print(f"РАСЧЁТ ИНДИКАТОРОВ ДЛЯ {len(tickers)} АКЦИЙ")
    print("=" * 80)
    print()
    
    config = Config()
    logger = Logger.get_logger("CalculateIndicators")
    db_path = config.base_path / "data" / "market_data.db"
    database = Database(db_path, logger)
    analyzer = Analyzer(config, logger)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...")
        
        try:
            data = database.load_quotes(ticker)
            
            if not data.empty:
                analyzer.analyze(data, ticker=ticker, save_indicators=True)
                print(f"  ✅ Готово")
            else:
                print(f"  ⚠️  Нет данных")
        
        except Exception as e:
            print(f"  ❌ Ошибка: {str(e)[:50]}")
    
    print()
    print("✅ РАСЧЁТ ЗАВЕРШЁН")
    print()


def main():
    """
    Главная функция.
    
    Returns:
        None
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "РАСЧЁТ ИНДИКАТОРОВ")
    print("=" * 80)
    print()
    print("ВЫБЕРИТЕ РЕЖИМ:")
    print()
    print("  1. Все акции - ОБНОВЛЕНИЕ (быстро, только новые)")
    print("  2. Все акции - ПОЛНЫЙ ПЕРЕСЧЁТ (медленно)")
    print("  3. Конкретные акции")
    print()
    print("  0. Выход")
    print()
    
    choice = input("Ваш выбор: ").strip()
    
    if choice == "1":
        calculate_indicators_for_all(incremental=True)
    
    elif choice == "2":
        calculate_indicators_for_all(incremental=False)
    
    elif choice == "3":
        tickers_input = input("\nВведите тикеры через пробел (например, SBER GAZP LKOH): ").strip().upper()
        if tickers_input:
            tickers = tickers_input.split()
            calculate_for_specific_tickers(tickers)
    
    elif choice == "0":
        print("До свидания!")
    
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # CLI режим
        if sys.argv[1] == "--update":
            # Быстрое обновление
            calculate_indicators_for_all(incremental=True)
        elif sys.argv[1] == "--all":
            # Полный пересчёт
            calculate_indicators_for_all(incremental=False)
        else:
            # Список тикеров как аргументы
            calculate_for_specific_tickers(sys.argv[1:])
    else:
        # Интерактивный режим
        main()

