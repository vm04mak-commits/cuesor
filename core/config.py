"""
Модуль конфигурации системы.
Хранит все настройки, пути к файлам и параметры работы.
"""

import os
from pathlib import Path
from typing import Dict, Any
import json


class Config:
    """
    Класс конфигурации системы.
    Управляет настройками и путями к файлам.
    """
    
    def __init__(self, base_path: str = None):
        """
        Инициализация конфигурации.
        
        Args:
            base_path (str): Базовый путь проекта. Если None, используется текущая директория.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._config_data: Dict[str, Any] = {}
        self._init_paths()
        self._load_or_create_config()
    
    def _init_paths(self) -> None:
        """
        Инициализация путей к директориям и файлам.
        
        Returns:
            None
        """
        # Основные директории
        self.core_dir = self.base_path / "core"
        self.data_dir = self.base_path / "data"
        self.analysis_dir = self.base_path / "analysis"
        self.predict_dir = self.base_path / "predict"
        self.report_dir = self.base_path / "report"
        self.api_dir = self.base_path / "api"
        self.docs_dir = self.base_path / "docs"
        
        # Директории для данных
        self.cache_dir = self.data_dir / "cache"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Файлы документации
        self.roadmap_file = self.docs_dir / "roadmap.md"
        self.session_log_file = self.docs_dir / "session_log.json"
        self.autodoc_file = self.docs_dir / "autodoc.md"
        self.readme_file = self.docs_dir / "readme.md"
        
        # Файл конфигурации
        self.config_file = self.base_path / "config.json"
    
    def _load_or_create_config(self) -> None:
        """
        Загрузка конфигурации из файла или создание новой.
        
        Returns:
            None
        """
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._config_data = json.load(f)
        else:
            self._config_data = self._get_default_config()
            self.save_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Получение конфигурации по умолчанию.
        
        Returns:
            Dict[str, Any]: Словарь с настройками по умолчанию.
        """
        return {
            "system": {
                "name": "Investment AI Assistant",
                "version": "0.1.0",
                "log_level": "INFO"
            },
            "data": {
                "moex_api_url": "https://iss.moex.com/iss",
                "update_interval": 3600,  # секунды
                "cache_ttl": 86400  # секунды (24 часа)
            },
            "analysis": {
                "indicators": ["SMA", "EMA", "RSI", "MACD", "BB"],
                "default_period": 30
            },
            "predict": {
                "model_type": "linear_regression",
                "training_period": 365,  # дней
                "prediction_horizon": 30  # дней
            },
            "report": {
                "output_format": ["html", "pdf"],
                "include_charts": True
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "debug": False
            }
        }
    
    def save_config(self) -> None:
        """
        Сохранение конфигурации в файл.
        
        Returns:
            None
        """
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self._config_data, f, indent=4, ensure_ascii=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения конфигурации по ключу.
        
        Args:
            key (str): Ключ в формате "section.param" (например, "data.moex_api_url")
            default (Any): Значение по умолчанию, если ключ не найден
        
        Returns:
            Any: Значение конфигурации или default
        """
        keys = key.split('.')
        value = self._config_data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Установка значения конфигурации.
        
        Args:
            key (str): Ключ в формате "section.param"
            value (Any): Новое значение
        
        Returns:
            None
        """
        keys = key.split('.')
        config = self._config_data
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save_config()
    
    def ensure_directories(self) -> None:
        """
        Создание всех необходимых директорий, если они не существуют.
        
        Returns:
            None
        """
        directories = [
            self.core_dir, self.data_dir, self.analysis_dir,
            self.predict_dir, self.report_dir, self.api_dir,
            self.docs_dir, self.cache_dir, self.raw_data_dir,
            self.processed_data_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)









