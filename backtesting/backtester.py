"""
Advanced Backtester

Продвинутый бэктестинг для ML моделей.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class AdvancedBacktester:
    """
    Продвинутый бэктестер для ML моделей.
    
    Возможности:
    - Walk-Forward Analysis
    - Monte Carlo симуляции
    - Расширенные метрики (Sharpe, Sortino, Calmar, etc.)
    - Визуализация результатов
    """
    
    def __init__(self, risk_free_rate: float = 0.08):
        """
        Инициализация бэктестера.
        
        Args:
            risk_free_rate (float): Безрисковая ставка (8% годовых для РФ)
        """
        self.risk_free_rate = risk_free_rate
        self.results = {}
    
    # ========== МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ ==========
    
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Коэффициент Шарпа.
        
        Args:
            returns: Серия доходностей
            
        Returns:
            float: Sharpe Ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Аннуализируем (252 торговых дня)
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        return sharpe
    
    def sortino_ratio(self, returns: pd.Series) -> float:
        """
        Коэффициент Сортино (учитывает только downside volatility).
        
        Args:
            returns: Серия доходностей
            
        Returns:
            float: Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Только отрицательные доходности
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)
        sortino = excess_returns.mean() / downside_std * np.sqrt(252)
        
        return sortino
    
    def maximum_drawdown(self, returns: pd.Series) -> Dict:
        """
        Максимальная просадка.
        
        Args:
            returns: Серия доходностей
            
        Returns:
            Dict: {max_drawdown, peak_date, trough_date, duration_days}
        """
        if len(returns) == 0:
            return {
                'max_drawdown': 0.0,
                'peak_date': None,
                'trough_date': None,
                'duration_days': 0
            }
        
        # Кумулятивная доходность
        cumulative = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cumulative.expanding().max()
        
        # Drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Максимальная просадка
        max_dd = drawdown.min()
        
        if pd.isna(max_dd):
            max_dd = 0.0
        
        # Найти даты
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None
        
        if max_dd_date is not None:
            # Найти пик перед просадкой
            before_trough = cumulative.loc[:max_dd_date]
            if len(before_trough) > 0:
                peak_date = before_trough.idxmax()
                duration_days = (max_dd_date - peak_date).days if hasattr(max_dd_date - peak_date, 'days') else 0
            else:
                peak_date = cumulative.index[0]
                duration_days = 0
        else:
            peak_date = None
            duration_days = 0
        
        return {
            'max_drawdown': max_dd,
            'peak_date': peak_date,
            'trough_date': max_dd_date,
            'duration_days': duration_days
        }
    
    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Коэффициент Калмара (доходность / максимальная просадка).
        
        Args:
            returns: Серия доходностей
            
        Returns:
            float: Calmar Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        mdd = abs(self.maximum_drawdown(returns)['max_drawdown'])
        
        if mdd == 0:
            return 0.0
        
        return annual_return / mdd
    
    def win_rate(self, returns: pd.Series) -> Dict:
        """
        Win Rate и средние выигрыш/проигрыш.
        
        Args:
            returns: Серия доходностей
            
        Returns:
            Dict: {win_rate, avg_win, avg_loss, win_loss_ratio}
        """
        if len(returns) == 0:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_loss_ratio': 0.0
            }
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio
        }
    
    def calculate_all_metrics(self, returns: pd.Series) -> Dict:
        """
        Рассчитать все метрики.
        
        Args:
            returns: Серия доходностей
            
        Returns:
            Dict: Все метрики
        """
        mdd = self.maximum_drawdown(returns)
        win_metrics = self.win_rate(returns)
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(returns),
            'max_drawdown': mdd['max_drawdown'],
            'max_dd_duration_days': mdd['duration_days'],
            'win_rate': win_metrics['win_rate'],
            'avg_win': win_metrics['avg_win'],
            'avg_loss': win_metrics['avg_loss'],
            'win_loss_ratio': win_metrics['win_loss_ratio'],
            'total_trades': len(returns)
        }
        
        return metrics
    
    # ========== WALK-FORWARD ANALYSIS ==========
    
    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        train_period_days: int = 180,
        test_period_days: int = 30,
        step_days: int = 30,
        retrain_func: callable = None
    ) -> Dict:
        """
        Walk-Forward Analysis (Rolling Window Validation).
        
        Метод:
        1. Обучаем модель на train_period_days
        2. Тестируем на test_period_days
        3. Сдвигаем окно на step_days
        4. Повторяем
        
        Args:
            data: DataFrame с данными (должен содержать 'date', 'actual', 'predicted')
            train_period_days: Период обучения
            test_period_days: Период тестирования
            step_days: Шаг сдвига окна
            retrain_func: Функция переобучения модели (опционально)
            
        Returns:
            Dict: Результаты WFA
        """
        print("\n" + "="*80)
        print("🔄 WALK-FORWARD ANALYSIS")
        print("="*80)
        print(f"Период обучения:     {train_period_days} дней")
        print(f"Период тестирования: {test_period_days} дней")
        print(f"Шаг:                 {step_days} дней")
        print()
        
        if 'date' not in data.columns:
            data = data.reset_index()
        
        # Сортируем по дате
        data = data.sort_values('date')
        
        results = []
        
        # Начальная позиция
        start_idx = 0
        
        while start_idx + train_period_days + test_period_days <= len(data):
            # Окна данных
            train_end_idx = start_idx + train_period_days
            test_end_idx = train_end_idx + test_period_days
            
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:test_end_idx]
            
            if len(test_data) == 0:
                break
            
            # Если есть функция переобучения, используем её
            if retrain_func:
                try:
                    retrain_func(train_data)
                except Exception as e:
                    print(f"⚠️  Ошибка переобучения: {e}")
            
            # Оцениваем на тестовом периоде
            if 'actual' in test_data.columns and 'predicted' in test_data.columns:
                actuals = test_data['actual'].values
                predictions = test_data['predicted'].values
                
                # Метрики прогноза
                r2 = r2_score(actuals, predictions)
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                
                # Доходности (если есть)
                if 'return' in test_data.columns:
                    returns = test_data['return']
                    metrics = self.calculate_all_metrics(returns)
                else:
                    metrics = {}
                
                results.append({
                    'period_start': train_data['date'].iloc[0],
                    'period_end': test_data['date'].iloc[-1],
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    **metrics
                })
            
            # Сдвигаем окно
            start_idx += step_days
        
        # Агрегированные результаты
        if results:
            results_df = pd.DataFrame(results)
            
            summary = {
                'num_periods': len(results),
                'avg_r2': results_df['r2'].mean(),
                'std_r2': results_df['r2'].std(),
                'avg_mae': results_df['mae'].mean(),
                'avg_rmse': results_df['rmse'].mean(),
                'periods': results_df
            }
            
            print(f"✅ Протестировано периодов: {len(results)}")
            print(f"📊 Средний R²: {summary['avg_r2']:.4f} ± {summary['std_r2']:.4f}")
            print(f"📊 Средний MAE: {summary['avg_mae']:.2f}")
            print(f"📊 Средний RMSE: {summary['avg_rmse']:.2f}")
            
            return summary
        else:
            print("❌ Недостаточно данных для WFA")
            return {}
    
    # ========== MONTE CARLO СИМУЛЯЦИИ ==========
    
    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        num_simulations: int = 10000,
        num_days: int = 252,
        initial_capital: float = 1000000
    ) -> Dict:
        """
        Monte Carlo симуляция для оценки будущей производительности.
        
        Args:
            returns: Исторические доходности
            num_simulations: Количество симуляций
            num_days: Горизонт симуляции (дней)
            initial_capital: Начальный капитал
            
        Returns:
            Dict: Результаты симуляций
        """
        print("\n" + "="*80)
        print("🎲 MONTE CARLO СИМУЛЯЦИЯ")
        print("="*80)
        print(f"Симуляций:        {num_simulations:,}")
        print(f"Горизонт:         {num_days} дней")
        print(f"Начальный капитал: {initial_capital:,.0f} ₽")
        print()
        
        if len(returns) == 0:
            return {}
        
        # Параметры распределения
        mean_return = returns.mean()
        std_return = returns.std()
        
        print(f"📊 Средняя доходность: {mean_return*100:.4f}% / день")
        print(f"📊 Волатильность:      {std_return*100:.4f}% / день")
        print()
        
        # Симуляции
        simulations = np.zeros((num_simulations, num_days))
        
        for i in range(num_simulations):
            # Генерируем случайные доходности
            sim_returns = np.random.normal(mean_return, std_return, num_days)
            
            # Кумулятивная доходность
            sim_cumulative = (1 + sim_returns).cumprod()
            
            simulations[i] = sim_cumulative
        
        # Финальные значения портфеля
        final_values = simulations[:, -1] * initial_capital
        
        # Статистика
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(final_values, percentiles)
        
        # Вероятности
        prob_profit = (final_values > initial_capital).mean()
        prob_loss_10 = (final_values < initial_capital * 0.9).mean()
        prob_loss_20 = (final_values < initial_capital * 0.8).mean()
        
        results = {
            'simulations': simulations,
            'final_values': final_values,
            'mean_final_value': final_values.mean(),
            'std_final_value': final_values.std(),
            'percentiles': dict(zip(percentiles, percentile_values)),
            'prob_profit': prob_profit,
            'prob_loss_10_percent': prob_loss_10,
            'prob_loss_20_percent': prob_loss_20,
            'best_case': final_values.max(),
            'worst_case': final_values.min()
        }
        
        print("📈 Результаты симуляции:")
        print(f"   Средний итог:     {results['mean_final_value']:>15,.0f} ₽")
        print(f"   Медиана (50%):    {percentile_values[2]:>15,.0f} ₽")
        print(f"   Лучший случай:    {results['best_case']:>15,.0f} ₽")
        print(f"   Худший случай:    {results['worst_case']:>15,.0f} ₽")
        print()
        print("📊 Перцентили:")
        for p, v in zip(percentiles, percentile_values):
            print(f"   {p}%: {v:>20,.0f} ₽")
        print()
        print("🎯 Вероятности:")
        print(f"   Прибыль (>0%):        {prob_profit*100:>6.2f}%")
        print(f"   Убыток >10%:          {prob_loss_10*100:>6.2f}%")
        print(f"   Убыток >20%:          {prob_loss_20*100:>6.2f}%")
        
        return results
    
    # ========== ВИЗУАЛИЗАЦИЯ ==========
    
    def plot_walk_forward_results(self, wf_results: Dict, save_path: Optional[str] = None):
        """
        Визуализация результатов Walk-Forward Analysis.
        
        Args:
            wf_results: Результаты WFA
            save_path: Путь для сохранения графика
        """
        if not wf_results or 'periods' not in wf_results:
            print("❌ Нет данных для визуализации")
            return
        
        periods_df = wf_results['periods']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. R² по периодам
        axes[0, 0].plot(range(len(periods_df)), periods_df['r2'], marker='o', linewidth=2)
        axes[0, 0].axhline(y=periods_df['r2'].mean(), color='r', linestyle='--', label=f'Mean: {periods_df["r2"].mean():.4f}')
        axes[0, 0].set_title('R² Score по периодам')
        axes[0, 0].set_xlabel('Период')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MAE и RMSE
        axes[0, 1].plot(range(len(periods_df)), periods_df['mae'], marker='o', label='MAE', linewidth=2)
        axes[0, 1].plot(range(len(periods_df)), periods_df['rmse'], marker='s', label='RMSE', linewidth=2)
        axes[0, 1].set_title('MAE и RMSE по периодам')
        axes[0, 1].set_xlabel('Период')
        axes[0, 1].set_ylabel('Ошибка')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sharpe Ratio (если есть)
        if 'sharpe_ratio' in periods_df.columns:
            axes[1, 0].bar(range(len(periods_df)), periods_df['sharpe_ratio'], alpha=0.7)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1, 0].axhline(y=1.5, color='g', linestyle='--', label='Хороший (1.5)', linewidth=1)
            axes[1, 0].set_title('Sharpe Ratio по периодам')
            axes[1, 0].set_xlabel('Период')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Win Rate (если есть)
        if 'win_rate' in periods_df.columns:
            axes[1, 1].bar(range(len(periods_df)), periods_df['win_rate']*100, alpha=0.7, color='green')
            axes[1, 1].axhline(y=50, color='r', linestyle='--', label='50%', linewidth=1)
            axes[1, 1].set_title('Win Rate по периодам')
            axes[1, 1].set_xlabel('Период')
            axes[1, 1].set_ylabel('Win Rate (%)')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 График сохранён: {save_path}")
        else:
            plt.show()
    
    def plot_monte_carlo_results(self, mc_results: Dict, save_path: Optional[str] = None):
        """
        Визуализация результатов Monte Carlo симуляции.
        
        Args:
            mc_results: Результаты MC симуляции
            save_path: Путь для сохранения графика
        """
        if not mc_results or 'simulations' not in mc_results:
            print("❌ Нет данных для визуализации")
            return
        
        simulations = mc_results['simulations']
        final_values = mc_results['final_values']
        percentiles = mc_results['percentiles']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=16, fontweight='bold')
        
        # 1. Пути симуляций (показываем 100 случайных)
        sample_size = min(100, len(simulations))
        sample_indices = np.random.choice(len(simulations), sample_size, replace=False)
        
        for idx in sample_indices:
            axes[0, 0].plot(simulations[idx], alpha=0.1, color='blue')
        
        # Перцентили
        percentile_paths = np.percentile(simulations, [5, 50, 95], axis=0)
        axes[0, 0].plot(percentile_paths[0], color='red', linewidth=2, label='5th percentile')
        axes[0, 0].plot(percentile_paths[1], color='green', linewidth=2, label='50th percentile')
        axes[0, 0].plot(percentile_paths[2], color='red', linewidth=2, label='95th percentile')
        axes[0, 0].axhline(y=1, color='black', linestyle='--', linewidth=1)
        axes[0, 0].set_title('Пути симуляций (100 случайных)')
        axes[0, 0].set_xlabel('Дни')
        axes[0, 0].set_ylabel('Кумулятивная доходность')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Распределение финальных значений
        axes[0, 1].hist(final_values, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=mc_results['mean_final_value'], color='green', linestyle='--', linewidth=2, label=f'Mean: {mc_results["mean_final_value"]:,.0f}')
        axes[0, 1].axvline(x=percentiles[50], color='orange', linestyle='--', linewidth=2, label=f'Median: {percentiles[50]:,.0f}')
        axes[0, 1].set_title('Распределение финальных значений')
        axes[0, 1].set_xlabel('Финальное значение (₽)')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Перцентили
        percentile_keys = sorted(percentiles.keys())
        percentile_vals = [percentiles[k] for k in percentile_keys]
        
        axes[1, 0].bar(range(len(percentile_keys)), percentile_vals, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xticks(range(len(percentile_keys)))
        axes[1, 0].set_xticklabels([f'{k}%' for k in percentile_keys])
        axes[1, 0].set_title('Перцентили финальных значений')
        axes[1, 0].set_xlabel('Перцентиль')
        axes[1, 0].set_ylabel('Значение (₽)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for i, (k, v) in enumerate(zip(percentile_keys, percentile_vals)):
            axes[1, 0].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Вероятности
        prob_data = {
            'Прибыль (>0%)': mc_results['prob_profit'] * 100,
            'Убыток >10%': mc_results['prob_loss_10_percent'] * 100,
            'Убыток >20%': mc_results['prob_loss_20_percent'] * 100
        }
        
        colors = ['green', 'orange', 'red']
        axes[1, 1].barh(list(prob_data.keys()), list(prob_data.values()), color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Вероятности исходов')
        axes[1, 1].set_xlabel('Вероятность (%)')
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Добавляем значения
        for i, (k, v) in enumerate(prob_data.items()):
            axes[1, 1].text(v, i, f' {v:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 График сохранён: {save_path}")
        else:
            plt.show()
    
    def generate_report(self, metrics: Dict, save_path: Optional[str] = None) -> str:
        """
        Генерация текстового отчёта.
        
        Args:
            metrics: Словарь метрик
            save_path: Путь для сохранения отчёта
            
        Returns:
            str: Отчёт
        """
        report = []
        report.append("="*80)
        report.append("ОТЧЁТ БЭКТЕСТИНГА")
        report.append("="*80)
        report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
        report.append("-"*80)
        report.append(f"Общая доходность:      {metrics.get('total_return', 0)*100:>8.2f}%")
        report.append(f"Годовая доходность:    {metrics.get('annual_return', 0)*100:>8.2f}%")
        report.append(f"Волатильность:         {metrics.get('volatility', 0)*100:>8.2f}%")
        report.append("")
        
        report.append("📈 РИСК-ДОХОДНОСТЬ:")
        report.append("-"*80)
        report.append(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>8.2f}")
        report.append(f"Sortino Ratio:         {metrics.get('sortino_ratio', 0):>8.2f}")
        report.append(f"Calmar Ratio:          {metrics.get('calmar_ratio', 0):>8.2f}")
        report.append("")
        
        report.append("⚠️  РИСКИ:")
        report.append("-"*80)
        report.append(f"Max Drawdown:          {metrics.get('max_drawdown', 0)*100:>8.2f}%")
        report.append(f"MDD Duration:          {metrics.get('max_dd_duration_days', 0):>8.0f} дней")
        report.append("")
        
        report.append("🎯 ТОЧНОСТЬ:")
        report.append("-"*80)
        report.append(f"Win Rate:              {metrics.get('win_rate', 0)*100:>8.2f}%")
        report.append(f"Average Win:           {metrics.get('avg_win', 0)*100:>8.4f}%")
        report.append(f"Average Loss:          {metrics.get('avg_loss', 0)*100:>8.4f}%")
        report.append(f"Win/Loss Ratio:        {metrics.get('win_loss_ratio', 0):>8.2f}")
        report.append(f"Total Trades:          {metrics.get('total_trades', 0):>8.0f}")
        report.append("")
        
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"💾 Отчёт сохранён: {save_path}")
        
        return report_text









