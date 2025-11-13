"""
Модуль генерации отчётов.
Создаёт комплексные отчёты по анализу и прогнозированию акций.
"""

import pandas as pd
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
from .chart_builder import ChartBuilder
from .html_exporter import HTMLExporter


class ReportGenerator:
    """
    Класс для генерации инвестиционных отчётов.
    """
    
    def __init__(self, config, logger):
        """
        Инициализация генератора отчётов.
        
        Args:
            config: Объект конфигурации системы
            logger: Объект логгера
        """
        self.config = config
        self.logger = logger
        self.output_dir = config.base_path / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chart_builder = ChartBuilder(logger)
        self.html_exporter = HTMLExporter(logger)
        
        self.logger.info("ReportGenerator инициализирован")
    
    def generate_report(self, ticker: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерация полного отчёта по акции.
        
        Args:
            ticker (str): Тикер акции
            analysis_results (Dict[str, Any]): Результаты анализа и прогноза
        
        Returns:
            Dict[str, Any]: Информация о сгенерированном отчёте
        """
        self.logger.info(f"Генерация отчёта для {ticker}")
        
        try:
            # Создание директории для отчёта
            report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = self.output_dir / f"{ticker}_{report_timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Извлекаем данные из результатов
            data = analysis_results.get('data', pd.DataFrame())
            analysis = analysis_results.get('analysis', {})
            prediction = analysis_results.get('prediction', {})
            
            # Генерация графиков
            charts = self._generate_charts(ticker, data, analysis, prediction, report_dir)
            
            # Генерация текстового отчёта
            text_report = self._generate_text_report(ticker, analysis, prediction)
            
            # Сохранение текстового отчёта
            text_file = report_dir / "report.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            # Генерация HTML отчёта
            html_file = self.html_exporter.export(
                ticker, analysis, prediction, charts, report_dir
            )
            
            report_info = {
                'ticker': ticker,
                'timestamp': report_timestamp,
                'directory': str(report_dir),
                'text_report': str(text_file),
                'html_report': str(html_file),
                'charts': charts
            }
            
            self.logger.info(f"Отчёт сгенерирован: {report_dir}")
            return report_info
        
        except Exception as e:
            self.logger.exception(f"Ошибка при генерации отчёта для {ticker}")
            raise
    
    def _generate_charts(self, ticker: str, data: pd.DataFrame, analysis: Dict[str, Any], 
                        prediction: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
        """
        Генерация всех графиков для отчёта.
        
        Args:
            ticker (str): Тикер акции
            data (pd.DataFrame): Исторические данные
            analysis (Dict[str, Any]): Результаты анализа
            prediction (Dict[str, Any]): Результаты прогноза
            output_dir (Path): Директория для сохранения графиков
        
        Returns:
            Dict[str, str]: Словарь {название: путь к файлу}
        """
        self.logger.info("Генерация графиков")
        
        charts = {}
        
        if not data.empty:
            # График цен
            price_chart = self.chart_builder.build_price_chart(ticker, data, output_dir)
            charts['price_chart'] = price_chart
            
            # График с техническими индикаторами
            if 'technical' in analysis:
                tech_chart = self.chart_builder.build_technical_chart(
                    ticker, data, analysis['technical'], output_dir
                )
                charts['technical_chart'] = tech_chart
            
            # График волатильности
            if 'volatility' in analysis:
                vol_chart = self.chart_builder.build_volatility_chart(
                    ticker, data, output_dir
                )
                charts['volatility_chart'] = vol_chart
        
        return charts
    
    def _generate_text_report(self, ticker: str, analysis: Dict[str, Any], 
                             prediction: Dict[str, Any]) -> str:
        """
        Генерация текстового отчёта.
        
        Args:
            ticker (str): Тикер акции
            analysis (Dict[str, Any]): Результаты анализа
            prediction (Dict[str, Any]): Результаты прогноза
        
        Returns:
            str: Текстовый отчёт
        """
        self.logger.info("Генерация текстового отчёта")
        
        report = []
        report.append("=" * 80)
        report.append(f"ИНВЕСТИЦИОННЫЙ ОТЧЁТ: {ticker}")
        report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Статистика
        if 'statistics' in analysis:
            stats = analysis['statistics']
            report.append("СТАТИСТИКА")
            report.append("-" * 40)
            report.append(f"Текущая цена: {stats.get('current', 0):.2f} ₽")
            report.append(f"Средняя цена: {stats.get('mean', 0):.2f} ₽")
            report.append(f"Медиана: {stats.get('median', 0):.2f} ₽")
            report.append(f"Минимум: {stats.get('min', 0):.2f} ₽")
            report.append(f"Максимум: {stats.get('max', 0):.2f} ₽")
            report.append(f"Изменение: {stats.get('change', 0):.2f}%")
            report.append("")
        
        # Технический анализ
        if 'technical' in analysis and 'current_values' in analysis['technical']:
            current = analysis['technical']['current_values']
            report.append("ТЕХНИЧЕСКИЙ АНАЛИЗ")
            report.append("-" * 40)
            if current.get('rsi'):
                report.append(f"RSI: {current['rsi']:.2f}")
            if current.get('sma_20'):
                report.append(f"SMA(20): {current['sma_20']:.2f} ₽")
            if current.get('sma_50'):
                report.append(f"SMA(50): {current['sma_50']:.2f} ₽")
            report.append("")
        
        # Тренд
        if 'trend' in analysis:
            trend = analysis['trend']
            report.append("ТРЕНД")
            report.append("-" * 40)
            report.append(f"Направление: {trend.get('trend', 'unknown').upper()}")
            report.append(f"Сила: {trend.get('strength', 'unknown').upper()}")
            report.append("")
        
        # Волатильность
        if 'volatility' in analysis:
            vol = analysis['volatility']
            report.append("ВОЛАТИЛЬНОСТЬ")
            report.append("-" * 40)
            report.append(f"Дневная: {vol.get('daily_volatility', 0):.4f}")
            report.append(f"Годовая: {vol.get('annual_volatility', 0):.4f}")
            report.append("")
        
        # Прогноз
        if 'ensemble' in prediction:
            ensemble = prediction['ensemble']
            report.append("ПРОГНОЗ")
            report.append("-" * 40)
            report.append(f"Прогнозная цена: {ensemble.get('predicted_price', 0):.2f} ₽")
            report.append(f"Направление: {ensemble.get('direction', 'unknown').upper()}")
            report.append(f"Уверенность: {ensemble.get('confidence', 'unknown').upper()}")
            report.append("")
        
        # Рекомендация
        if 'recommendation' in prediction:
            rec = prediction['recommendation']
            report.append("РЕКОМЕНДАЦИЯ")
            report.append("-" * 40)
            report.append(f"Действие: {rec.get('action', 'hold').upper()}")
            report.append(f"Причина: {rec.get('reason', 'Нет данных')}")
            report.append(f"Уверенность: {rec.get('confidence', 'unknown').upper()}")
            if 'expected_change_percent' in rec:
                report.append(f"Ожидаемое изменение: {rec['expected_change_percent']:.2f}%")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_summary(self, multiple_reports: Dict[str, Dict[str, Any]]) -> str:
        """
        Генерация сводного отчёта по нескольким акциям.
        
        Args:
            multiple_reports (Dict[str, Dict[str, Any]]): Словарь {тикер: результаты}
        
        Returns:
            str: Путь к сводному отчёту
        """
        self.logger.info(f"Генерация сводного отчёта для {len(multiple_reports)} акций")
        
        summary = []
        summary.append("=" * 80)
        summary.append("СВОДНЫЙ ИНВЕСТИЦИОННЫЙ ОТЧЁТ")
        summary.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Количество акций: {len(multiple_reports)}")
        summary.append("=" * 80)
        summary.append("")
        
        # Таблица рекомендаций
        summary.append("РЕКОМЕНДАЦИИ ПО АКЦИЯМ")
        summary.append("-" * 80)
        summary.append(f"{'Тикер':<10} {'Действие':<10} {'Изменение':<12} {'Уверенность':<15}")
        summary.append("-" * 80)
        
        for ticker, results in multiple_reports.items():
            prediction = results.get('prediction', {})
            rec = prediction.get('recommendation', {})
            
            action = rec.get('action', 'hold').upper()
            change = rec.get('expected_change_percent', 0)
            confidence = rec.get('confidence', 'unknown').upper()
            
            summary.append(f"{ticker:<10} {action:<10} {change:>10.2f}% {confidence:<15}")
        
        summary.append("")
        summary.append("=" * 80)
        
        # Сохранение
        summary_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary))
        
        self.logger.info(f"Сводный отчёт сохранён: {summary_file}")
        return str(summary_file)









