"""
Модуль логирования.
Централизованная система логирования для всех модулей системы.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class Logger:
    """
    Класс для централизованного логирования.
    Управляет записью логов в файлы и консоль.
    """
    
    _instances = {}
    
    def __init__(self, name: str, log_dir: Optional[Path] = None, level: str = "INFO"):
        """
        Инициализация логгера.
        
        Args:
            name (str): Имя логгера (обычно имя модуля)
            log_dir (Optional[Path]): Директория для хранения логов
            level (str): Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = name
        self.log_dir = log_dir or Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание логгера
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Избегаем дублирования обработчиков
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """
        Настройка обработчиков логов (консоль и файл).
        
        Returns:
            None
        """
        # Формат логов
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Консольный обработчик
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Файловый обработчик
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(cls, name: str, log_dir: Optional[Path] = None, level: str = "INFO") -> 'Logger':
        """
        Получение экземпляра логгера (singleton для каждого имени).
        
        Args:
            name (str): Имя логгера
            log_dir (Optional[Path]): Директория для логов
            level (str): Уровень логирования
        
        Returns:
            Logger: Экземпляр логгера
        """
        if name not in cls._instances:
            cls._instances[name] = cls(name, log_dir, level)
        return cls._instances[name]
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Логирование сообщения уровня DEBUG.
        
        Args:
            message (str): Сообщение для логирования
            **kwargs: Дополнительные данные для логирования
        
        Returns:
            None
        """
        if kwargs:
            message = f"{message} | Data: {json.dumps(kwargs, ensure_ascii=False)}"
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Логирование сообщения уровня INFO.
        
        Args:
            message (str): Сообщение для логирования
            **kwargs: Дополнительные данные для логирования
        
        Returns:
            None
        """
        if kwargs:
            message = f"{message} | Data: {json.dumps(kwargs, ensure_ascii=False)}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Логирование сообщения уровня WARNING.
        
        Args:
            message (str): Сообщение для логирования
            **kwargs: Дополнительные данные для логирования
        
        Returns:
            None
        """
        if kwargs:
            message = f"{message} | Data: {json.dumps(kwargs, ensure_ascii=False)}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        """
        Логирование сообщения уровня ERROR.
        
        Args:
            message (str): Сообщение для логирования
            **kwargs: Дополнительные данные для логирования
        
        Returns:
            None
        """
        if kwargs:
            message = f"{message} | Data: {json.dumps(kwargs, ensure_ascii=False)}"
        self.logger.error(message)
    
    def critical(self, message: str, **kwargs) -> None:
        """
        Логирование сообщения уровня CRITICAL.
        
        Args:
            message (str): Сообщение для логирования
            **kwargs: Дополнительные данные для логирования
        
        Returns:
            None
        """
        if kwargs:
            message = f"{message} | Data: {json.dumps(kwargs, ensure_ascii=False)}"
        self.logger.critical(message)
    
    def exception(self, message: str, exc_info: bool = True) -> None:
        """
        Логирование исключения с трассировкой.
        
        Args:
            message (str): Сообщение об ошибке
            exc_info (bool): Включать ли информацию об исключении
        
        Returns:
            None
        """
        self.logger.exception(message, exc_info=exc_info)









