# 💼 Portfolio Management Module

Модуль для управления инвестиционным портфелем на основе Modern Portfolio Theory (MPT).

---

## 📋 Содержание

- **`portfolio.py`** - Класс Portfolio для управления портфелем
- **`optimizer.py`** - Класс PortfolioOptimizer для оптимизации
- **`risk_manager.py`** - Класс RiskManager для анализа рисков

---

## 🚀 Быстрый старт

```python
from portfolio import Portfolio, PortfolioOptimizer, RiskManager
import pandas as pd

# 1. Создание портфеля
portfolio = Portfolio("Мой портфель", initial_cash=1000000)
portfolio.add_position("SBER", shares=100, price=250.0)
portfolio.add_position("GAZP", shares=50, price=180.0)

# 2. Оптимизация
returns = pd.DataFrame({...})  # Исторические доходности
optimizer = PortfolioOptimizer(returns, risk_free_rate=0.08)
result = optimizer.max_sharpe_portfolio()

# 3. Анализ рисков
risk_manager = RiskManager(returns)
metrics = risk_manager.calculate_all_metrics(result['weights'])
```

---

## 📚 Классы

### Portfolio

Управление инвестиционным портфелем:
- Добавление/удаление позиций
- Расчёт стоимости и доходностей
- История операций
- Сохранение/загрузка

**Основные методы:**
- `add_position(ticker, shares, price)` - Купить акции
- `remove_position(ticker, shares, price)` - Продать акции
- `update_prices(prices_dict)` - Обновить цены
- `get_total_value()` - Общая стоимость
- `get_weights()` - Веса позиций
- `get_returns()` - Доходности
- `print_summary()` - Вывести сводку
- `save(filepath)` / `load(filepath)` - Сохранить/загрузить

### PortfolioOptimizer

Оптимизация портфеля по MPT:
- Max Sharpe Ratio
- Min Variance
- Risk Parity
- Equal Weight
- Efficient Frontier
- Monte Carlo симуляции

**Основные методы:**
- `max_sharpe_portfolio()` - Максимизация Sharpe Ratio
- `min_variance_portfolio()` - Минимизация риска
- `risk_parity_portfolio()` - Равный вклад в риск
- `equal_weight_portfolio()` - Равные веса
- `efficient_frontier(num_portfolios)` - Efficient Frontier
- `compare_strategies()` - Сравнение стратегий
- `optimize_with_constraints()` - Оптимизация с ограничениями

### RiskManager

Управление рисками:
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Maximum Drawdown
- Sharpe/Sortino/Calmar Ratios
- Beta и корреляция

**Основные методы:**
- `value_at_risk(weights, confidence_level)` - VaR
- `conditional_var(weights, confidence_level)` - CVaR
- `maximum_drawdown(weights)` - Maximum Drawdown
- `sharpe_ratio(weights, risk_free_rate)` - Sharpe Ratio
- `sortino_ratio(weights, risk_free_rate)` - Sortino Ratio
- `calmar_ratio(weights)` - Calmar Ratio
- `beta(weights)` - Beta (требует бенчмарк)
- `calculate_all_metrics(weights)` - Все метрики
- `print_risk_report(weights, portfolio_value)` - Отчёт

---

## 📊 Примеры

### Пример 1: Простой портфель

```python
from portfolio import Portfolio

# Создаём портфель
portfolio = Portfolio("Пример", initial_cash=1000000)

# Добавляем позиции
portfolio.add_position("SBER", shares=100, price=250.0)
portfolio.add_position("GAZP", shares=50, price=180.0)

# Обновляем цены
portfolio.update_prices({"SBER": 260.0, "GAZP": 175.0})

# Выводим сводку
portfolio.print_summary()

# Сохраняем
portfolio.save()
```

### Пример 2: Оптимизация

```python
from portfolio import PortfolioOptimizer
import pandas as pd

# Подготовка данных
returns = pd.DataFrame({
    'SBER': [0.01, -0.02, 0.03, ...],
    'GAZP': [-0.01, 0.02, 0.01, ...],
    ...
})

# Оптимизатор
optimizer = PortfolioOptimizer(returns, risk_free_rate=0.08)

# Стратегии
max_sharpe = optimizer.max_sharpe_portfolio()
min_var = optimizer.min_variance_portfolio()

# Сравнение
comparison = optimizer.compare_strategies()
print(comparison)
```

### Пример 3: Анализ рисков

```python
from portfolio import RiskManager

risk_manager = RiskManager(returns)

# Веса портфеля
weights = {'SBER': 0.4, 'GAZP': 0.3, 'LKOH': 0.3}

# Метрики
var_95 = risk_manager.value_at_risk(weights)
cvar_95 = risk_manager.conditional_var(weights)
sharpe = risk_manager.sharpe_ratio(weights, risk_free_rate=0.08)

# Полный отчёт
risk_manager.print_risk_report(weights, portfolio_value=1000000)
```

---

## 📈 Теория

### Modern Portfolio Theory (MPT)

Разработана Гарри Марковицем (Нобелевская премия 1990).

**Основная идея:** Диверсификация снижает риск без уменьшения доходности.

**Ключевые концепции:**
- Efficient Frontier - множество оптимальных портфелей
- Sharpe Ratio - мера эффективности
- Оптимальное распределение капитала

### Метрики

| Метрика | Формула | Интерпретация |
|---------|---------|---------------|
| **Sharpe Ratio** | (Rp - Rf) / σp | > 1.0 хорошо, > 2.0 отлично |
| **Sortino Ratio** | (Rp - Rf) / σd | Учитывает только downside риск |
| **Calmar Ratio** | Rp / MDD | > 1.0 хорошо, > 3.0 отлично |
| **VaR 95%** | Квантиль 5% | Макс. убыток с 95% вероятностью |
| **CVaR 95%** | E[R \| R ≤ VaR] | Средний убыток в худших 5% |
| **Max Drawdown** | (Peak - Trough) / Peak | Макс. просадка |

---

## 🛠️ Зависимости

```python
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
```

---

## 📚 Документация

Полное руководство: **[docs/guides/PORTFOLIO_GUIDE.md](../docs/guides/PORTFOLIO_GUIDE.md)**

---

## 🎯 Use Cases

1. **Создание оптимального портфеля** на основе ML прогнозов
2. **Ребалансировка существующего портфеля**
3. **Анализ рисков** текущих позиций
4. **Сравнение стратегий** инвестирования
5. **Backtesting** портфельных стратегий

---

## ⚠️ Важно

1. **Исторические данные ≠ Будущие результаты**
2. **Транзакционные издержки** не учитываются
3. **Налоги** не учитываются
4. **Ликвидность** проверяйте вручную
5. **Регулярно обновляйте** данные

---

## 📞 Поддержка

Вопросы и проблемы:
- См. [PORTFOLIO_GUIDE.md](../docs/guides/PORTFOLIO_GUIDE.md)
- Запустите `6_portfolio_optimization.py` для интерактивного интерфейса

---

**Успешных инвестиций! 📈💰**









