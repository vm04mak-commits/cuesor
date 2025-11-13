"""
Модуль Report - генерация отчётов и визуализация.

Содержит:
- ReportGenerator (генератор отчётов)
- ChartBuilder (построение графиков)
- HTMLExporter (экспорт в HTML)
"""

from .report_generator import ReportGenerator
from .chart_builder import ChartBuilder
from .html_exporter import HTMLExporter

__all__ = ['ReportGenerator', 'ChartBuilder', 'HTMLExporter']









