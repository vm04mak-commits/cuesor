"""
Модуль Data - сбор данных о рынке.

Содержит:
- DataCollector (сбор котировок с MOEX)
- NewsCollector (сбор новостей)
- ReportCollector (сбор отчётностей компаний)
"""

from .data_collector import DataCollector
from .news_collector import NewsCollector
from .report_collector import ReportCollector

__all__ = ['DataCollector', 'NewsCollector', 'ReportCollector']









