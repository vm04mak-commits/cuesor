"""
Модуль оркестратора.
Управляет всеми модулями системы и координирует их работу.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from .config import Config
from .logger import Logger
from .doc_manager import DocManager


class Orchestrator:
    """
    Оркестратор системы.
    Координирует работу всех модулей и управляет жизненным циклом приложения.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Инициализация оркестратора.
        
        Args:
            base_path (Optional[str]): Базовый путь проекта
        """
        # Инициализация конфигурации
        self.config = Config(base_path)
        self.config.ensure_directories()
        
        # Инициализация логгера
        self.logger = Logger.get_logger(
            "Orchestrator",
            self.config.base_path / "logs",
            self.config.get("system.log_level", "INFO")
        )
        
        # Инициализация менеджера документации
        self.doc_manager = DocManager(
            self.config.base_path,
            self.config.docs_dir
        )
        
        # Модули системы (будут инициализированы по мере необходимости)
        self.modules: Dict[str, Any] = {}
        
        self.logger.info("Оркестратор инициализирован")
        self._log_session_start()
    
    def _log_session_start(self) -> None:
        """
        Логирование начала сессии.
        
        Returns:
            None
        """
        self.doc_manager.log_session(
            module="Orchestrator",
            function="__init__",
            input_data={"base_path": str(self.config.base_path)},
            result="Система инициализирована"
        )
    
    def initialize_module(self, module_name: str) -> Any:
        """
        Инициализация модуля по имени.
        
        Args:
            module_name (str): Имя модуля (data, analysis, predict, report, api)
        
        Returns:
            Any: Экземпляр модуля
        """
        if module_name in self.modules:
            return self.modules[module_name]
        
        try:
            if module_name == "data":
                from data import DataCollector
                self.modules[module_name] = DataCollector(self.config, self.logger)
            elif module_name == "analysis":
                from analysis import Analyzer
                self.modules[module_name] = Analyzer(self.config, self.logger)
            elif module_name == "predict":
                from predict import Predictor
                self.modules[module_name] = Predictor(self.config, self.logger)
            elif module_name == "report":
                from report import ReportGenerator
                self.modules[module_name] = ReportGenerator(self.config, self.logger)
            elif module_name == "api":
                from api import APIServer
                self.modules[module_name] = APIServer(self.config, self.logger)
            else:
                raise ValueError(f"Неизвестный модуль: {module_name}")
            
            self.logger.info(f"Модуль '{module_name}' инициализирован")
            self.doc_manager.log_session(
                module="Orchestrator",
                function="initialize_module",
                input_data={"module_name": module_name},
                result=f"Модуль {module_name} инициализирован"
            )
            
            return self.modules[module_name]
        
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации модуля '{module_name}'", error=str(e))
            self.doc_manager.log_session(
                module="Orchestrator",
                function="initialize_module",
                input_data={"module_name": module_name},
                error=str(e)
            )
            raise
    
    def update_documentation(self) -> None:
        """
        Обновление всей документации проекта.
        
        Returns:
            None
        """
        self.logger.info("Обновление документации...")
        self.doc_manager.update_autodoc()
        self.logger.info("Документация обновлена")
    
    def run_pipeline(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Запуск полного пайплайна анализа для тикера.
        
        Args:
            ticker (str): Тикер акции (например, "SBER")
            start_date (str): Начальная дата (YYYY-MM-DD)
            end_date (str): Конечная дата (YYYY-MM-DD)
        
        Returns:
            Dict[str, Any]: Результаты анализа и прогноза
        """
        self.logger.info(f"Запуск пайплайна для {ticker}")
        results = {}
        
        try:
            # 1. Сбор данных
            data_collector = self.initialize_module("data")
            data = data_collector.fetch_stock_data(ticker, start_date, end_date)
            results['data'] = data
            
            # 2. Анализ
            analyzer = self.initialize_module("analysis")
            analysis = analyzer.analyze(data, ticker=ticker, save_indicators=True)
            results['analysis'] = analysis
            
            # 3. Прогноз
            predictor = self.initialize_module("predict")
            prediction = predictor.predict(data)
            results['prediction'] = prediction
            
            # 4. Отчёт
            report_gen = self.initialize_module("report")
            report = report_gen.generate_report(ticker, results)
            results['report'] = report
            
            self.logger.info(f"Пайплайн для {ticker} завершён успешно")
            self.doc_manager.log_session(
                module="Orchestrator",
                function="run_pipeline",
                input_data={"ticker": ticker, "start_date": start_date, "end_date": end_date},
                result="Пайплайн выполнен успешно"
            )
            
        except Exception as e:
            self.logger.exception(f"Ошибка при выполнении пайплайна для {ticker}")
            self.doc_manager.log_session(
                module="Orchestrator",
                function="run_pipeline",
                input_data={"ticker": ticker, "start_date": start_date, "end_date": end_date},
                error=str(e)
            )
            raise
        
        return results
    
    def shutdown(self) -> None:
        """
        Корректное завершение работы системы.
        
        Returns:
            None
        """
        self.logger.info("Завершение работы системы...")
        self.update_documentation()
        self.logger.info("Система остановлена")

