"""
СКРИПТ 4: Прогнозирование для акций

Что делает:
- Загружает универсальную модель
- Делает прогнозы для всех или выбранных акций
- Выводит топ акций по прогнозу роста

Запуск: python 4_predict_stocks.py
"""

from core import Config, Logger, Database
from predict.universal_model import UniversalModelTrainer
import pandas as pd
import numpy as np


def predict_all_stocks(top_n: int = 20):
    """
    Прогнозирование для всех акций.
    
    Args:
        top_n (int): Показать топ-N акций
    
    Returns:
        None
    """
    print("=" * 80)
    print("СКРИПТ 4: ПРОГНОЗИРОВАНИЕ")
    print("=" * 80)
    print()
    
    # Инициализация
    config = Config()
    logger = Logger.get_logger("Predict")
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
        print(f"   R² Score: {model_data['results']['test_metrics']['r2']:.4f}")
        print()
        
        # Получение списка акций
        print("📋 Получение списка акций...")
        tickers = database.get_available_tickers()
        print(f"✅ Найдено {len(tickers)} акций")
        print()
        
        # Прогнозирование
        print("🔮 Делаем прогнозы...")
        print("-" * 80)
        
        predictions = []
        
        for ticker in tickers:
            try:
                # Загрузка данных
                quotes = database.load_quotes(ticker)
                indicators = database.load_indicators(ticker)
                
                if quotes.empty or indicators.empty or len(quotes) < 30:
                    continue
                
                # Последняя строка
                last_quote = quotes.iloc[-1:]
                last_indicators = indicators.iloc[-1:]
                
                # Объединение
                last_quote_reset = last_quote.reset_index(drop=True)
                last_indicators_reset = last_indicators.reset_index(drop=True)
                last_data = pd.concat([last_quote_reset, last_indicators_reset], axis=1)
                
                # Добавляем тикер
                last_data['ticker'] = ticker
                last_data['ticker_encoded'] = ticker_encoder.transform([ticker])[0]
                
                # Дополнительные признаки (как при обучении)
                last_data['price_change_1d'] = quotes['close'].pct_change(1).iloc[-1]
                last_data['price_change_5d'] = quotes['close'].pct_change(5).iloc[-1]
                last_data['volume_ma_ratio'] = quotes['volume'].iloc[-1] / quotes['volume'].rolling(20).mean().iloc[-1]
                
                # Выбираем нужные признаки
                X_pred = last_data[[f for f in features if f in last_data.columns]]
                
                # Заполняем отсутствующие признаки нулями
                for f in features:
                    if f not in X_pred.columns:
                        X_pred[f] = 0
                
                # Упорядочиваем колонки
                X_pred = X_pred[features]
                
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
                
                current_price = float(quotes['close'].iloc[-1])
                change_percent = ((predicted_price - current_price) / current_price) * 100
                
                predictions.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'change_percent': change_percent
                })
                
                print(f"{ticker:<6} {current_price:>8.2f} ₽ → {predicted_price:>8.2f} ₽ ({change_percent:+.2f}%)")
            
            except Exception as e:
                continue
        
        print("-" * 80)
        print()
        
        # Результаты
        if predictions:
            df = pd.DataFrame(predictions)
            df_sorted = df.sort_values('change_percent', ascending=False)
            
            print("=" * 80)
            print(f"🏆 ТОП-{top_n} АКЦИЙ ПО ПРОГНОЗУ РОСТА")
            print("=" * 80)
            print()
            print(f"{'Тикер':<8} {'Текущая':<10} {'Прогноз':<10} {'Изменение':<12} {'Рекомендация'}")
            print("-" * 80)
            
            for _, row in df_sorted.head(top_n).iterrows():
                change = row['change_percent']
                
                if change > 5:
                    recommendation = "🟢 BUY"
                elif change > 2:
                    recommendation = "🟡 HOLD"
                elif change > -2:
                    recommendation = "🟡 HOLD"
                else:
                    recommendation = "🔴 SELL"
                
                print(f"{row['ticker']:<8} {row['current_price']:>9.2f} ₽ {row['predicted_price']:>9.2f} ₽ {change:>+10.2f}%  {recommendation}")
            
            print()
            print("=" * 80)
            print(f"✅ ПРОГНОЗ ЗАВЕРШЁН ({len(predictions)} акций)")
            print("=" * 80)
            print()
            
            # Сохранение прогнозов
            predictions_file = config.base_path / "predictions" / f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions_file.parent.mkdir(parents=True, exist_ok=True)
            df_sorted.to_csv(predictions_file, index=False)
            print(f"💾 Прогнозы сохранены: {predictions_file}")
            print()
        
        else:
            print("❌ Не удалось сделать прогнозы")
    
    except FileNotFoundError:
        print("❌ Модель не найдена!")
        print("   Сначала запустите: python 3_train_universal_model.py")
    
    except Exception as e:
        print(f"❌ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()


def predict_specific_tickers(tickers: list):
    """
    Прогноз для конкретных акций.
    
    Args:
        tickers (list): Список тикеров
    
    Returns:
        None
    """
    print("=" * 80)
    print(f"ПРОГНОЗ ДЛЯ {len(tickers)} АКЦИЙ")
    print("=" * 80)
    print()
    
    config = Config()
    logger = Logger.get_logger("Predict")
    trainer = UniversalModelTrainer(config, logger)
    
    # Загрузка модели
    model_data = trainer.load_model()
    
    print(f"{'Тикер':<8} {'Текущая':<12} {'Прогноз':<12} {'Изменение':<12} {'Рекомендация'}")
    print("-" * 80)
    
    # Прогнозы (упрощённо, без полной реализации)
    for ticker in tickers:
        print(f"{ticker:<8} (требуется реализация)")
    
    print()


def main():
    """
    Главная функция.
    
    Returns:
        None
    """
    print("\n" + "=" * 80)
    print(" " * 30 + "ПРОГНОЗ")
    print("=" * 80)
    print()
    print("ВЫБЕРИТЕ РЕЖИМ:")
    print()
    print("  1. Прогноз для всех акций (показать топ-20)")
    print("  2. Прогноз для всех акций (показать топ-50)")
    print("  3. Прогноз для конкретных акций")
    print()
    print("  0. Выход")
    print()
    
    choice = input("Ваш выбор: ").strip()
    
    if choice == "1":
        predict_all_stocks(top_n=20)
    
    elif choice == "2":
        predict_all_stocks(top_n=50)
    
    elif choice == "3":
        tickers_input = input("\nВведите тикеры через пробел: ").strip().upper()
        if tickers_input:
            tickers = tickers_input.split()
            predict_specific_tickers(tickers)
    
    elif choice == "0":
        print("До свидания!")
    
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # CLI режим
        if sys.argv[1] == "--all":
            top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            predict_all_stocks(top_n=top_n)
        else:
            predict_specific_tickers(sys.argv[1:])
    else:
        # Интерактивный режим
        main()




