"""
Модуль менеджера документации.
Автоматическое извлечение docstring'ов и обновление документации.
"""

import ast
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from .logger import Logger


class DocManager:
    """
    Менеджер документации.
    Автоматически извлекает docstring'и и обновляет документацию проекта.
    """
    
    def __init__(self, base_path: Path, docs_dir: Path):
        """
        Инициализация менеджера документации.
        
        Args:
            base_path (Path): Базовый путь проекта
            docs_dir (Path): Директория для документации
        """
        self.base_path = base_path
        self.docs_dir = docs_dir
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        self.autodoc_file = self.docs_dir / "autodoc.md"
        self.session_log_file = self.docs_dir / "session_log.json"
        self.roadmap_file = self.docs_dir / "roadmap.md"
        
        self.logger = Logger.get_logger("DocManager")
    
    def extract_docstrings_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Извлечение docstring'ов из Python файла.
        
        Args:
            file_path (Path): Путь к Python файлу
        
        Returns:
            List[Dict[str, Any]]: Список словарей с информацией о функциях/классах и их docstring'ах
        """
        docstrings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        doc_info = {
                            'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
                            'name': node.name,
                            'docstring': docstring,
                            'file': str(file_path.relative_to(self.base_path)),
                            'line': node.lineno
                        }
                        docstrings.append(doc_info)
        
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении docstring из {file_path}", error=str(e))
        
        return docstrings
    
    def scan_project(self) -> List[Dict[str, Any]]:
        """
        Сканирование всего проекта на предмет Python файлов и извлечение docstring'ов.
        
        Returns:
            List[Dict[str, Any]]: Список всех найденных docstring'ов
        """
        all_docstrings = []
        
        # Сканируем все директории проекта
        for py_file in self.base_path.rglob("*.py"):
            # Пропускаем виртуальные окружения и кэш
            if any(part in py_file.parts for part in ['venv', '__pycache__', '.venv', 'env']):
                continue
            
            docstrings = self.extract_docstrings_from_file(py_file)
            all_docstrings.extend(docstrings)
        
        self.logger.info(f"Найдено {len(all_docstrings)} docstring'ов в проекте")
        return all_docstrings
    
    def update_autodoc(self) -> None:
        """
        Обновление файла автоматической документации.
        
        Returns:
            None
        """
        docstrings = self.scan_project()
        
        # Группировка по файлам
        docs_by_file = {}
        for doc in docstrings:
            file_path = doc['file']
            if file_path not in docs_by_file:
                docs_by_file[file_path] = []
            docs_by_file[file_path].append(doc)
        
        # Генерация markdown
        content = "# Автоматическая документация\n\n"
        content += f"*Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        content += "---\n\n"
        
        for file_path, docs in sorted(docs_by_file.items()):
            content += f"## {file_path}\n\n"
            
            for doc in sorted(docs, key=lambda x: x['line']):
                icon = "📦" if doc['type'] == 'class' else "⚡"
                content += f"### {icon} `{doc['name']}` (строка {doc['line']})\n\n"
                content += f"```\n{doc['docstring']}\n```\n\n"
            
            content += "---\n\n"
        
        # Сохранение
        with open(self.autodoc_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Автодокументация обновлена: {self.autodoc_file}")
    
    def log_session(self, module: str, function: str, input_data: Any, result: Any = None, error: Any = None) -> None:
        """
        Логирование вызова функции в session_log.json.
        
        Args:
            module (str): Имя модуля
            function (str): Имя функции
            input_data (Any): Входные данные
            result (Any): Результат выполнения
            error (Any): Ошибка (если есть)
        
        Returns:
            None
        """
        # Загрузка существующего лога
        if self.session_log_file.exists():
            with open(self.session_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Добавление новой записи
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'module': module,
            'function': function,
            'input_data': str(input_data),
            'result': str(result) if result is not None else None,
            'error': str(error) if error is not None else None
        }
        logs.append(log_entry)
        
        # Сохранение
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        
        self.logger.debug(f"Логирование вызова: {module}.{function}")
    
    def update_roadmap(self, content: str) -> None:
        """
        Обновление roadmap.md.
        
        Args:
            content (str): Содержимое roadmap
        
        Returns:
            None
        """
        with open(self.roadmap_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info("Roadmap обновлён")
    
    def create_readme(self, content: str) -> None:
        """
        Создание или обновление readme.md.
        
        Args:
            content (str): Содержимое readme
        
        Returns:
            None
        """
        readme_file = self.docs_dir / "readme.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info("README обновлён")









