"""
Модуль Core - ядро системы инвестиционного ИИ-ассистента.

Содержит:
- Оркестратор (управление модулями)
- Логгер (централизованное логирование)
- Конфигурация (настройки системы)
- Менеджер документации (автоматическое обновление docs)
- База данных (SQLite для хранения данных)
"""

from .logger import Logger
from .config import Config
from .doc_manager import DocManager
from .orchestrator import Orchestrator
from .database import Database

__all__ = ['Logger', 'Config', 'DocManager', 'Orchestrator', 'Database']

