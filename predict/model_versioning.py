"""
Model Versioning Module

Система версионирования ML моделей:
- Сохранение истории моделей
- Сравнение версий
- Rollback к предыдущим версиям
- Метаданные моделей
"""

import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import logging


class ModelVersioning:
    """
    Система версионирования ML моделей.
    
    Структура хранения:
    models/
    ├── v1/
    │   ├── model.pkl
    │   ├── metadata.json
    │   └── metrics.json
    ├── v2/
    │   ├── model.pkl
    │   ├── metadata.json
    │   └── metrics.json
    └── versions.json (индекс всех версий)
    """
    
    def __init__(self, models_dir: str = "models", logger: Optional[logging.Logger] = None):
        """
        Инициализация Model Versioning.
        
        Args:
            models_dir: Директория для хранения моделей
            logger: Логгер
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logger
        
        self.versions_file = self.models_dir / "versions.json"
        self._load_versions_index()
    
    # ========== УПРАВЛЕНИЕ ИНДЕКСОМ ВЕРСИЙ ==========
    
    def _load_versions_index(self):
        """Загрузить индекс версий."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r', encoding='utf-8') as f:
                self.versions_index = json.load(f)
        else:
            self.versions_index = {
                'versions': [],
                'latest': None,
                'production': None
            }
    
    def _save_versions_index(self):
        """Сохранить индекс версий."""
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.versions_index, f, indent=2, ensure_ascii=False)
    
    def _get_next_version(self) -> str:
        """Получить следующий номер версии."""
        if not self.versions_index['versions']:
            return 'v1'
        
        # Извлекаем номера версий
        version_numbers = []
        for v in self.versions_index['versions']:
            try:
                num = int(v['version'].replace('v', ''))
                version_numbers.append(num)
            except ValueError:
                continue
        
        if version_numbers:
            next_num = max(version_numbers) + 1
        else:
            next_num = 1
        
        return f'v{next_num}'
    
    # ========== СОХРАНЕНИЕ МОДЕЛИ ==========
    
    def save_model(
        self,
        model: Any,
        metadata: Dict,
        metrics: Dict,
        version: Optional[str] = None,
        set_as_latest: bool = True,
        set_as_production: bool = False,
        additional_files: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Сохранить модель с версией.
        
        Args:
            model: Обученная модель
            metadata: Метаданные (автор, описание, параметры, etc.)
            metrics: Метрики модели
            version: Версия (если None, создаётся автоматически)
            set_as_latest: Установить как latest
            set_as_production: Установить как production
            additional_files: Дополнительные файлы (scaler, encoder, etc.)
            
        Returns:
            str: Версия модели
        """
        print("\n" + "="*80)
        print("💾 СОХРАНЕНИЕ МОДЕЛИ")
        print("="*80)
        print()
        
        # Версия
        if version is None:
            version = self._get_next_version()
        
        version_dir = self.models_dir / version
        version_dir.mkdir(exist_ok=True)
        
        print(f"Версия: {version}")
        print(f"Директория: {version_dir}")
        print()
        
        # Сохраняем модель
        model_path = version_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Модель сохранена: {model_path}")
        
        # Добавляем метаданные системы
        metadata['version'] = version
        metadata['created_at'] = datetime.now().isoformat()
        metadata['model_type'] = type(model).__name__
        
        # Сохраняем метаданные
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ Метаданные сохранены: {metadata_path}")
        
        # Сохраняем метрики
        metrics_path = version_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ Метрики сохранены: {metrics_path}")
        
        # Дополнительные файлы
        if additional_files:
            for filename, obj in additional_files.items():
                file_path = version_dir / filename
                with open(file_path, 'wb') as f:
                    pickle.dump(obj, f)
                print(f"✅ Дополнительный файл сохранён: {file_path}")
        
        # Обновляем индекс
        version_entry = {
            'version': version,
            'created_at': metadata['created_at'],
            'model_type': metadata['model_type'],
            'metrics': metrics
        }
        
        # Удаляем старую запись если есть
        self.versions_index['versions'] = [
            v for v in self.versions_index['versions'] if v['version'] != version
        ]
        
        # Добавляем новую
        self.versions_index['versions'].append(version_entry)
        
        # Latest
        if set_as_latest:
            self.versions_index['latest'] = version
            print(f"✅ Установлена как latest: {version}")
        
        # Production
        if set_as_production:
            self.versions_index['production'] = version
            print(f"✅ Установлена как production: {version}")
        
        self._save_versions_index()
        
        print()
        print(f"✅ Модель {version} успешно сохранена")
        print()
        
        return version
    
    # ========== ЗАГРУЗКА МОДЕЛИ ==========
    
    def load_model(
        self,
        version: Optional[str] = None,
        use_production: bool = False,
        load_additional: bool = True
    ) -> Dict:
        """
        Загрузить модель.
        
        Args:
            version: Версия (если None, загружается latest)
            use_production: Использовать production версию
            load_additional: Загрузить дополнительные файлы
            
        Returns:
            Dict: {model, metadata, metrics, additional_files}
        """
        print("\n" + "="*80)
        print("📦 ЗАГРУЗКА МОДЕЛИ")
        print("="*80)
        print()
        
        # Определяем версию
        if use_production:
            version = self.versions_index.get('production')
            if not version:
                raise ValueError("Production версия не установлена")
            print(f"Загрузка production версии: {version}")
        elif version is None:
            version = self.versions_index.get('latest')
            if not version:
                raise ValueError("Нет сохранённых версий")
            print(f"Загрузка latest версии: {version}")
        else:
            print(f"Загрузка версии: {version}")
        
        version_dir = self.models_dir / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Версия {version} не найдена")
        
        print()
        
        # Загружаем модель
        model_path = version_dir / "model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Модель загружена: {model_path}")
        
        # Загружаем метаданные
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✅ Метаданные загружены: {metadata_path}")
        
        # Загружаем метрики
        metrics_path = version_dir / "metrics.json"
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        print(f"✅ Метрики загружены: {metrics_path}")
        
        # Дополнительные файлы
        additional_files = {}
        if load_additional:
            for file_path in version_dir.glob("*.pkl"):
                if file_path.name != "model.pkl":
                    with open(file_path, 'rb') as f:
                        additional_files[file_path.stem] = pickle.load(f)
                    print(f"✅ Дополнительный файл загружен: {file_path}")
        
        print()
        print(f"✅ Версия {version} успешно загружена")
        print()
        
        return {
            'model': model,
            'metadata': metadata,
            'metrics': metrics,
            'additional_files': additional_files,
            'version': version
        }
    
    # ========== УПРАВЛЕНИЕ ВЕРСИЯМИ ==========
    
    def list_versions(self) -> pd.DataFrame:
        """
        Список всех версий.
        
        Returns:
            DataFrame с версиями
        """
        if not self.versions_index['versions']:
            print("Нет сохранённых версий")
            return pd.DataFrame()
        
        versions_list = []
        
        for v in self.versions_index['versions']:
            entry = {
                'version': v['version'],
                'created_at': v['created_at'],
                'model_type': v['model_type'],
                'is_latest': v['version'] == self.versions_index['latest'],
                'is_production': v['version'] == self.versions_index['production']
            }
            
            # Добавляем метрики
            if 'metrics' in v:
                for key, value in v['metrics'].items():
                    if isinstance(value, (int, float)):
                        entry[f'metric_{key}'] = value
            
            versions_list.append(entry)
        
        df = pd.DataFrame(versions_list)
        df = df.sort_values('created_at', ascending=False)
        
        return df
    
    def print_versions(self):
        """Вывести список версий."""
        print("\n" + "="*80)
        print("📋 СПИСОК ВЕРСИЙ МОДЕЛЕЙ")
        print("="*80)
        print()
        
        df = self.list_versions()
        
        if df.empty:
            print("Нет сохранённых версий")
            return
        
        for _, row in df.iterrows():
            status = []
            if row['is_latest']:
                status.append("LATEST")
            if row['is_production']:
                status.append("PRODUCTION")
            
            status_str = f" [{', '.join(status)}]" if status else ""
            
            print(f"📦 {row['version']}{status_str}")
            print(f"   Тип:        {row['model_type']}")
            print(f"   Создана:    {row['created_at']}")
            
            # Метрики
            metric_cols = [col for col in df.columns if col.startswith('metric_')]
            if metric_cols:
                print("   Метрики:")
                for col in metric_cols:
                    metric_name = col.replace('metric_', '')
                    metric_value = row[col]
                    if pd.notna(metric_value):
                        print(f"      {metric_name}: {metric_value:.4f}")
            
            print()
    
    def set_production(self, version: str):
        """
        Установить версию как production.
        
        Args:
            version: Версия модели
        """
        version_dir = self.models_dir / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Версия {version} не найдена")
        
        self.versions_index['production'] = version
        self._save_versions_index()
        
        print(f"✅ Версия {version} установлена как production")
    
    def delete_version(self, version: str, confirm: bool = False):
        """
        Удалить версию.
        
        Args:
            version: Версия для удаления
            confirm: Подтверждение удаления
        """
        if not confirm:
            print(f"⚠️  Для удаления версии {version} установите confirm=True")
            return
        
        version_dir = self.models_dir / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Версия {version} не найдена")
        
        # Нельзя удалить production
        if version == self.versions_index.get('production'):
            raise ValueError(f"Нельзя удалить production версию {version}")
        
        # Удаляем директорию
        shutil.rmtree(version_dir)
        
        # Обновляем индекс
        self.versions_index['versions'] = [
            v for v in self.versions_index['versions'] if v['version'] != version
        ]
        
        # Если была latest, назначаем новую
        if version == self.versions_index.get('latest'):
            if self.versions_index['versions']:
                # Берём последнюю по дате
                latest_version = max(
                    self.versions_index['versions'],
                    key=lambda x: x['created_at']
                )
                self.versions_index['latest'] = latest_version['version']
            else:
                self.versions_index['latest'] = None
        
        self._save_versions_index()
        
        print(f"✅ Версия {version} удалена")
    
    # ========== СРАВНЕНИЕ ВЕРСИЙ ==========
    
    def compare_versions(self, version1: str, version2: str) -> pd.DataFrame:
        """
        Сравнить две версии.
        
        Args:
            version1: Первая версия
            version2: Вторая версия
            
        Returns:
            DataFrame со сравнением
        """
        print("\n" + "="*80)
        print(f"🔍 СРАВНЕНИЕ ВЕРСИЙ: {version1} vs {version2}")
        print("="*80)
        print()
        
        # Загружаем метаданные и метрики
        v1_dir = self.models_dir / version1
        v2_dir = self.models_dir / version2
        
        if not v1_dir.exists():
            raise FileNotFoundError(f"Версия {version1} не найдена")
        if not v2_dir.exists():
            raise FileNotFoundError(f"Версия {version2} не найдена")
        
        # Метаданные
        with open(v1_dir / "metadata.json", 'r', encoding='utf-8') as f:
            metadata1 = json.load(f)
        with open(v2_dir / "metadata.json", 'r', encoding='utf-8') as f:
            metadata2 = json.load(f)
        
        # Метрики
        with open(v1_dir / "metrics.json", 'r', encoding='utf-8') as f:
            metrics1 = json.load(f)
        with open(v2_dir / "metrics.json", 'r', encoding='utf-8') as f:
            metrics2 = json.load(f)
        
        # Сравнение метрик
        comparison = []
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            val1 = metrics1.get(metric)
            val2 = metrics2.get(metric)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                diff_pct = (diff / val1 * 100) if val1 != 0 else 0
                
                comparison.append({
                    'metric': metric,
                    version1: val1,
                    version2: val2,
                    'diff': diff,
                    'diff_pct': diff_pct
                })
        
        df = pd.DataFrame(comparison)
        
        # Выводим
        print(f"📊 Метрики:")
        print()
        for _, row in df.iterrows():
            print(f"   {row['metric']}:")
            print(f"      {version1}: {row[version1]:.4f}")
            print(f"      {version2}: {row[version2]:.4f}")
            
            diff_sign = "+" if row['diff'] > 0 else ""
            print(f"      Разница:   {diff_sign}{row['diff']:.4f} ({diff_sign}{row['diff_pct']:.2f}%)")
            print()
        
        # Метаданные
        print(f"📝 Метаданные:")
        print(f"   {version1}: {metadata1.get('model_type')} (создана {metadata1.get('created_at')})")
        print(f"   {version2}: {metadata2.get('model_type')} (создана {metadata2.get('created_at')})")
        print()
        
        return df
    
    # ========== ROLLBACK ==========
    
    def rollback(self, version: str):
        """
        Rollback к предыдущей версии (установить как production).
        
        Args:
            version: Версия для rollback
        """
        print("\n" + "="*80)
        print(f"⏮️  ROLLBACK К ВЕРСИИ {version}")
        print("="*80)
        print()
        
        version_dir = self.models_dir / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Версия {version} не найдена")
        
        old_production = self.versions_index.get('production')
        
        self.set_production(version)
        
        print(f"✅ Rollback выполнен")
        print(f"   Старая production: {old_production}")
        print(f"   Новая production:  {version}")
        print()






