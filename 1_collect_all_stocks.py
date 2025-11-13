"""
СКРИПТ 1: Сбор котировок всех акций MOEX

Что делает:
- Получает список всех торгуемых акций через moexalgo
- Скачивает исторические котировки для каждой акции
- Сохраняет в БД и CSV

Запуск: python 1_collect_all_stocks.py
"""

from core import Config, Logger
from data import DataCollector
from data.moex_stocks import MOEXStocks
from datetime import datetime, timedelta
import time


def collect_all_stocks(days: int = 365, top_n: int = None, incremental: bool = True):
    """
    Сбор котировок для всех акций.
    
    Args:
        days (int): Количество дней истории (при первом запуске)
        top_n (int): Если указано, собирать только топ-N акций
        incremental (bool): Инкрементальное обновление (только новые данные)
    
    Returns:
        None
    """
    print("=" * 80)
    print("СКРИПТ 1: СБОР КОТИРОВОК ВСЕХ АКЦИЙ")
    print("=" * 80)
    print()
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("CollectAll")
    stocks_getter = MOEXStocks(config, logger)
    collector = DataCollector(config, logger)
    
    # Получение списка акций
    print("📋 Получение списка акций с MOEX...")
    
    if top_n:
        tickers = stocks_getter.get_top_stocks(top_n)
        print(f"✅ Выбрано топ-{top_n} акций")
    else:
        stocks_df = stocks_getter.get_all_stocks()
        tickers = stocks_df['ticker'].tolist()
        print(f"✅ Получено {len(tickers)} акций")
    
    print(f"📦 Всего акций: {len(tickers)}")
    print(f"🔄 Режим: {'Инкрементальное обновление' if incremental else 'Полная загрузка'}")
    print()
    
    # Сбор данных
    print("🔄 Начинаем сбор...")
    print("-" * 80)
    
    success_count = 0
    updated_count = 0
    skipped_count = 0
    failed_count = 0
    failed_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}...", end=" ")
        
        try:
            # Проверяем последнюю дату
            if incremental:
                last_date = collector.get_last_date(ticker)
                
                if last_date:
                    # Есть данные, обновляем с последней даты
                    last_date_obj = datetime.strptime(last_date, '%Y-%m-%d')
                    
                    # Если последняя дата сегодня или вчера, пропускаем
                    days_diff = (datetime.now() - last_date_obj).days
                    
                    if days_diff <= 1:
                        print(f"⏭️  Актуально (последняя дата: {last_date})")
                        skipped_count += 1
                        continue
                    
                    # Обновляем с последней даты + 1 день
                    start_date = (last_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    
                    print(f"🔄 Обновление с {start_date}...", end=" ")
                else:
                    # Нет данных, скачиваем всю историю
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                    print(f"📥 Первая загрузка ({days} дней)...", end=" ")
            else:
                # Полная загрузка
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            data = collector.fetch_stock_data(ticker, start_date, end_date, use_cache=False)
            
            if not data.empty:
                if incremental and last_date:
                    print(f"✅ +{len(data)} записей")
                    updated_count += 1
                else:
                    print(f"✅ {len(data)} записей")
                    success_count += 1
            else:
                print("⚠️  Нет новых данных")
                skipped_count += 1
            
            # Небольшая задержка для избежания блокировки API
            time.sleep(0.2)
        
        except Exception as e:
            print(f"❌ Ошибка: {str(e)[:50]}")
            failed_count += 1
            failed_tickers.append(ticker)
            continue
    
    # Итоги
    print()
    print("=" * 80)
    print("ИТОГИ СБОРА")
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
        print(f"\nНе удалось загрузить: {', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            print(f"... и ещё {len(failed_tickers) - 10}")
    
    print()
    print("📊 Данные сохранены в:")
    print(f"   - БД: {config.base_path / 'data' / 'market_data.db'}")
    print(f"   - CSV: {config.base_path / 'data' / 'csv' / '[TICKER]'}")
    print()
    print("=" * 80)
    print("✅ СБОР ЗАВЕРШЁН")
    print("=" * 80)
    print()
    print("Следующий шаг:")
    print("  python 2_calculate_indicators.py")
    print()


def main():
    """
    Главная функция с выбором режима.
    
    Returns:
        None
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "СБОР КОТИРОВОК")
    print("=" * 80)
    print()
    print("ВЫБЕРИТЕ РЕЖИМ:")
    print()
    print("  1. Все акции - ОБНОВЛЕНИЕ (быстро, только новые данные)")
    print("  2. Топ-50 акций - ОБНОВЛЕНИЕ (1-5 минут)")
    print("  3. Топ-20 акций - ОБНОВЛЕНИЕ (< 1 минуты)")
    print("  4. Топ-10 акций - ОБНОВЛЕНИЕ (< 30 секунд)")
    print()
    print("  5. Все акции - ПОЛНАЯ ЗАГРУЗКА (1-2 часа)")
    print("  6. Топ-50 - ПОЛНАЯ ЗАГРУЗКА (15-30 минут)")
    print()
    print("  0. Выход")
    print()
    
    choice = input("Ваш выбор: ").strip()
    
    if choice == "1":
        collect_all_stocks(days=365, incremental=True)
    
    elif choice == "2":
        collect_all_stocks(days=365, top_n=50, incremental=True)
    
    elif choice == "3":
        collect_all_stocks(days=365, top_n=20, incremental=True)
    
    elif choice == "4":
        collect_all_stocks(days=365, top_n=10, incremental=True)
    
    elif choice == "5":
        days = input("\nКоличество дней истории (default: 365): ").strip()
        days = int(days) if days else 365
        collect_all_stocks(days=days, incremental=False)
    
    elif choice == "6":
        collect_all_stocks(days=365, top_n=50, incremental=False)
    
    elif choice == "0":
        print("До свидания!")
    
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    # Можно запустить напрямую с параметрами
    import sys
    
    if len(sys.argv) > 1:
        # CLI режим
        if sys.argv[1] == "--update":
            # Быстрое обновление
            top_n = int(sys.argv[2]) if len(sys.argv) > 2 else None
            collect_all_stocks(days=365, top_n=top_n, incremental=True)
        elif sys.argv[1] == "--all":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
            collect_all_stocks(days=days, incremental=False)
        elif sys.argv[1] == "--top":
            top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            collect_all_stocks(days=365, top_n=top_n, incremental=False)
        else:
            print("Использование:")
            print("  python 1_collect_all_stocks.py --update [top_n]  # Быстрое обновление")
            print("  python 1_collect_all_stocks.py --all [days]      # Полная загрузка")
            print("  python 1_collect_all_stocks.py --top [n]         # Полная загрузка топ-N")
    else:
        # Интерактивный режим
        main()

