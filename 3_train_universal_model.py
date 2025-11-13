"""
СКРИПТ 3: Обучение универсальной модели для ВСЕГО рынка

Одна модель предсказывает ВСЕ акции!

Что делает:
- Загружает данные всех акций из базы
- Обучает универсальную модель (Random Forest, Gradient Boosting или LSTM/GRU)
- Сохраняет модель для использования в прогнозах

Запуск: python 3_train_universal_model.py
"""

from core import Config, Logger
from predict import UniversalModelTrainer
import sys


def main():
    """
    Главная функция обучения универсальной модели.
    """
    print("\n" + "="*80)
    print(" " * 20 + "ОБУЧЕНИЕ УНИВЕРСАЛЬНОЙ МОДЕЛИ")
    print("="*80)
    print()
    print("💡 Одна модель для ВСЕХ акций рынка!")
    print()
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("UniversalModel")
    trainer = UniversalModelTrainer(config, logger)
    
    # Выбор типа модели
    print("📋 Выберите тип модели:")
    print()
    print("  1. 🌲 Random Forest (быстро, ~10-15 минут)")
    print("  2. 📈 Gradient Boosting (медленнее, точнее, ~30-60 минут)")
    print("  3. 🤖 LSTM Deep Learning (требует TensorFlow, ~20-40 минут)")
    print("  4. 🤖 GRU Deep Learning (требует TensorFlow, ~15-30 минут)")
    print()
    print("  0. ◀️  Выход")
    print()
    
    choice = input("Ваш выбор (1-4): ").strip()
    
    if choice == '0':
        print("\nДо свидания! 👋")
        return
    
    if choice not in ['1', '2', '3', '4']:
        print("❌ Неверный выбор")
        return
    
    # Определяем тип модели
    model_types = {
        '1': 'random_forest',
        '2': 'gradient_boosting',
        '3': 'lstm',
        '4': 'gru'
    }
    
    model_type = model_types[choice]
    is_deep_learning = model_type in ['lstm', 'gru']
    
    # Для Deep Learning проверяем TensorFlow
    if is_deep_learning:
        try:
            import tensorflow as tf
        except ImportError:
            print("\n❌ TensorFlow не установлен!")
            print("   Установите: pip install tensorflow>=2.12.0")
            print("   Или выберите Random Forest / Gradient Boosting")
            return
    
    # Подтверждение
    print("\n" + "="*80)
    print(f"⚙️  Выбрана модель: {model_type.upper()}")
    print("="*80)
    print()
    
    if is_deep_learning:
        print("⚠️  Deep Learning занимает больше времени и памяти")
        print("   Рекомендуется GPU для ускорения")
        print()
    
    print("⚠️  Обучение займёт время. Не прерывайте процесс!")
    print()
    
    continue_choice = input("Продолжить? (y/n): ").strip().lower()
    if continue_choice != 'y':
        print("Отменено")
        return
    
    # Обучение
    print("\n" + "="*80)
    print("🎓 НАЧИНАЕМ ОБУЧЕНИЕ")
    print("="*80)
    print()
    
    try:
        # Загрузка данных
        print("📊 Загрузка данных всех акций из базы...")
        quotes, indicators = trainer.load_all_data()
        
        if quotes.empty:
            print("❌ Нет данных в базе!")
            print("   Запустите: python 1_collect_all_stocks.py")
            return
        
        # Подготовка датасета
        print("🔧 Подготовка датасета...")
        X, y, features = trainer.prepare_dataset(quotes, indicators)
        
        print(f"\n✅ Датасет готов:")
        print(f"   Акций:     {quotes['ticker'].nunique()}")
        print(f"   Записей:   {len(X):,}")
        print(f"   Признаков: {len(features)}")
        print()
        
        # Проверка размера для Deep Learning
        if is_deep_learning and len(X) < 10000:
            print(f"⚠️  ВНИМАНИЕ: Мало данных ({len(X)} записей)")
            print("   Deep Learning лучше работает с 10,000+ записей")
            print("   Рекомендуется собрать больше данных или использовать RF/GB")
            print()
            
            continue_dl = input("Продолжить с Deep Learning? (y/n): ").strip().lower()
            if continue_dl != 'y':
                print("Отменено. Попробуйте Random Forest (вариант 1)")
                return
        
        # Обучение модели
        print(f"⚙️  Обучение модели: {model_type.upper()}")
        print("   Это займёт несколько минут...")
        print()
        
        model_data = trainer.train(X, y, model_type=model_type)
        
        if not model_data:
            print("❌ Ошибка при обучении модели")
            return
        
        # Результаты
        print("\n" + "="*80)
        print("✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА!")
        print("="*80)
        print()
        print("📊 Результаты:")
        print(f"   Train R²: {model_data['results']['train_metrics']['r2']:.4f}")
        print(f"   Test R²:  {model_data['results']['test_metrics']['r2']:.4f}")
        print(f"   Test MAE: {model_data['results']['test_metrics']['mae']:.2f}")
        print()
        
        # Оценка качества
        test_r2 = model_data['results']['test_metrics']['r2']
        if test_r2 > 0.7:
            print("   🏆 ОТЛИЧНО! Модель работает очень хорошо")
        elif test_r2 > 0.5:
            print("   ✅ ХОРОШО! Модель показывает приемлемые результаты")
        elif test_r2 > 0.3:
            print("   ⚠️  СРЕДНЕ. Можно улучшить собрав больше данных")
        else:
            print("   ⚠️  НИЗКОЕ. Рекомендуется переобучить или собрать больше данных")
        
        print()
        
        # Сохранение
        print("💾 Сохранение модели...")
        model_path = trainer.save_model(model_data)
        print(f"✅ Модель сохранена: {model_path}")
        print()
        
        # Следующие шаги
        print("="*80)
        print("🎉 ГОТОВО!")
        print("="*80)
        print()
        print("Что дальше:")
        print("  1. Сделайте прогнозы: python 4_predict_stocks.py")
        print("  2. Проверьте точность: python 5_backtest_model.py")
        print("  3. Создайте портфель: python 6_portfolio_optimization.py")
        print()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Обучение прервано пользователем")
    
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
