"""
Модуль сбора данных о котировках акций.
Работает с API Московской биржи (MOEX).
Сохраняет данные в БД и CSV файлы.
"""

import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.database import Database


class DataCollector:
    """
    Класс для сбора данных о котировках с Московской биржи.
    """
    
    def __init__(self, config, logger):
        """
        Инициализация сборщика данных.
        
        Args:
            config: Объект конфигурации системы
            logger: Объект логгера
        """
        self.config = config
        self.logger = logger
        self.base_url = config.get("data.moex_api_url", "https://iss.moex.com/iss")
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = config.get("data.cache_ttl", 86400)
        
        # CSV директория
        self.csv_dir = config.base_path / "data" / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        # База данных
        db_path = config.base_path / "data" / "market_data.db"
        self.database = Database(db_path, logger)
        
        self.logger.info("DataCollector инициализирован")
    
    def _get_cache_file(self, ticker: str, start_date: str, end_date: str) -> Path:
        """
        Получение пути к файлу кэша для данных.
        
        Args:
            ticker (str): Тикер акции
            start_date (str): Начальная дата
            end_date (str): Конечная дата
        
        Returns:
            Path: Путь к файлу кэша
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """
        Проверка валидности кэша.
        
        Args:
            cache_file (Path): Путь к файлу кэша
        
        Returns:
            bool: True если кэш валиден, False если устарел
        """
        if not cache_file.exists():
            return False
        
        file_age = time.time() - cache_file.stat().st_mtime
        return file_age < self.cache_ttl
    
    def _load_from_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """
        Загрузка данных из кэша.
        
        Args:
            cache_file (Path): Путь к файлу кэша
        
        Returns:
            Optional[pd.DataFrame]: DataFrame с данными или None
        """
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            self.logger.info(f"Данные загружены из кэша: {cache_file}")
            return df
        except Exception as e:
            self.logger.error(f"Ошибка загрузки из кэша: {str(e)}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_file: Path) -> None:
        """
        Сохранение данных в кэш.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
            cache_file (Path): Путь к файлу кэша
        
        Returns:
            None
        """
        try:
            data_dict = data.to_dict(orient='records')
            # Конвертация datetime в строки для JSON
            for record in data_dict:
                if 'date' in record and isinstance(record['date'], pd.Timestamp):
                    record['date'] = record['date'].strftime('%Y-%m-%d')
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Данные сохранены в кэш: {cache_file}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения в кэш: {str(e)}")
    
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Получение исторических данных по акции.
        
        Args:
            ticker (str): Тикер акции (например, "SBER")
            start_date (str): Начальная дата в формате YYYY-MM-DD
            end_date (str): Конечная дата в формате YYYY-MM-DD
            use_cache (bool): Использовать ли кэш
        
        Returns:
            pd.DataFrame: DataFrame с историческими данными
        """
        self.logger.info(f"Запрос данных для {ticker} с {start_date} по {end_date}")
        
        # Проверка кэша
        cache_file = self._get_cache_file(ticker, start_date, end_date)
        if use_cache and self._is_cache_valid(cache_file):
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        try:
            # Запрос к API MOEX
            url = f"{self.base_url}/history/engines/stock/markets/shares/securities/{ticker}.json"
            params = {
                'from': start_date,
                'till': end_date,
                'start': 0
            }
            
            all_data = []
            
            # MOEX возвращает данные порциями, нужно пагинировать
            while True:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Извлекаем данные из ответа
                if 'history' not in data or 'data' not in data['history']:
                    break
                
                history_data = data['history']['data']
                if not history_data:
                    break
                
                # Получаем названия колонок
                columns = data['history']['columns']
                
                # Преобразуем в DataFrame
                df_chunk = pd.DataFrame(history_data, columns=columns)
                all_data.append(df_chunk)
                
                # Проверяем, есть ли ещё данные
                if len(history_data) < 100:  # MOEX возвращает по 100 записей
                    break
                
                params['start'] += 100
                time.sleep(0.1)  # Небольшая задержка для избежания rate limit
            
            if not all_data:
                self.logger.warning(f"Нет данных для {ticker}")
                return pd.DataFrame()
            
            # Объединяем все порции
            df = pd.concat(all_data, ignore_index=True)
            
            # Обработка данных
            df = self._process_stock_data(df)
            
            # Сохранение в кэш
            if use_cache:
                self._save_to_cache(df, cache_file)
            
            # Сохранение в базу данных
            self.save_to_database(ticker, df)
            
            # Сохранение в CSV
            self.save_to_csv(ticker, df)
            
            self.logger.info(f"Получено {len(df)} записей для {ticker}")
            return df
        
        except Exception as e:
            self.logger.exception(f"Ошибка при получении данных для {ticker}")
            raise
    
    def _process_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка сырых данных с биржи.
        
        Args:
            df (pd.DataFrame): Сырые данные
        
        Returns:
            pd.DataFrame: Обработанные данные
        """
        # Выбираем нужные колонки
        required_columns = ['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VALUE']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if not available_columns:
            self.logger.warning("Не найдены необходимые колонки данных")
            return df
        
        df = df[available_columns].copy()
        
        # Переименовываем колонки
        column_mapping = {
            'TRADEDATE': 'date',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close',
            'VOLUME': 'volume',
            'VALUE': 'value'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Конвертируем типы
        df['date'] = pd.to_datetime(df['date'])
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'value']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Удаляем строки с NaN в ключевых колонках
        df.dropna(subset=['date', 'close'], inplace=True)
        
        # Сортируем по дате
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def get_available_tickers(self) -> List[str]:
        """
        Получение списка доступных тикеров на MOEX.
        
        Returns:
            List[str]: Список тикеров
        """
        try:
            url = f"{self.base_url}/engines/stock/markets/shares/securities.json"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            securities = data['securities']['data']
            columns = data['securities']['columns']
            
            df = pd.DataFrame(securities, columns=columns)
            
            # Фильтруем только акции (убираем фонды и другие инструменты)
            if 'SECTYPE' in df.columns:
                df = df[df['SECTYPE'].isin(['1', 'common_share'])]
            
            tickers = df['SECID'].tolist() if 'SECID' in df.columns else []
            
            self.logger.info(f"Получено {len(tickers)} доступных тикеров")
            return tickers
        
        except Exception as e:
            self.logger.exception("Ошибка при получении списка тикеров")
            return []
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Получение текущей цены акции.
        
        Args:
            ticker (str): Тикер акции
        
        Returns:
            Optional[float]: Текущая цена или None при ошибке
        """
        try:
            url = f"{self.base_url}/engines/stock/markets/shares/securities/{ticker}.json"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'marketdata' in data and 'data' in data['marketdata']:
                marketdata = data['marketdata']['data']
                columns = data['marketdata']['columns']
                
                if marketdata and 'LAST' in columns:
                    last_idx = columns.index('LAST')
                    price = marketdata[0][last_idx]
                    
                    self.logger.info(f"Текущая цена {ticker}: {price}")
                    return float(price) if price else None
            
            return None
        
        except Exception as e:
            self.logger.error(f"Ошибка при получении текущей цены {ticker}: {str(e)}")
            return None
    
    def save_to_database(self, ticker: str, data: pd.DataFrame) -> None:
        """
        Сохранение данных в базу данных.
        
        Args:
            ticker (str): Тикер акции
            data (pd.DataFrame): Данные для сохранения
        
        Returns:
            None
        """
        try:
            self.database.save_quotes(ticker, data)
            self.logger.info(f"Данные для {ticker} сохранены в БД")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения в БД: {str(e)}")
    
    def save_to_csv(self, ticker: str, data: pd.DataFrame) -> str:
        """
        Сохранение данных в CSV файл.
        
        Args:
            ticker (str): Тикер акции
            data (pd.DataFrame): Данные для сохранения
        
        Returns:
            str: Путь к сохранённому файлу
        """
        try:
            # Создаём директорию для тикера
            ticker_dir = self.csv_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            
            # Путь к файлу
            csv_file = ticker_dir / f"{ticker}.csv"
            
            # Если файл существует, объединяем данные
            if csv_file.exists():
                existing_data = pd.read_csv(csv_file, parse_dates=['date'])
                
                # Объединяем и удаляем дубликаты
                combined = pd.concat([existing_data, data], ignore_index=True)
                combined = combined.drop_duplicates(subset=['date'], keep='last')
                combined = combined.sort_values('date').reset_index(drop=True)
                
                combined.to_csv(csv_file, index=False)
                self.logger.info(f"CSV обновлён: {csv_file} ({len(combined)} записей)")
            else:
                data.to_csv(csv_file, index=False)
                self.logger.info(f"CSV создан: {csv_file} ({len(data)} записей)")
            
            return str(csv_file)
        
        except Exception as e:
            self.logger.error(f"Ошибка сохранения CSV: {str(e)}")
            return ""
    
    def load_from_database(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Загрузка данных из базы данных.
        
        Args:
            ticker (str): Тикер акции
            start_date (str): Начальная дата
            end_date (str): Конечная дата
        
        Returns:
            pd.DataFrame: Данные из БД
        """
        try:
            df = self.database.load_quotes(ticker, start_date, end_date)
            self.logger.info(f"Загружено {len(df)} записей из БД для {ticker}")
            return df
        except Exception as e:
            self.logger.error(f"Ошибка загрузки из БД: {str(e)}")
            return pd.DataFrame()
    
    def load_from_csv(self, ticker: str) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла.
        
        Args:
            ticker (str): Тикер акции
        
        Returns:
            pd.DataFrame: Данные из CSV
        """
        try:
            csv_file = self.csv_dir / ticker / f"{ticker}.csv"
            
            if not csv_file.exists():
                self.logger.warning(f"CSV файл не найден: {csv_file}")
                return pd.DataFrame()
            
            df = pd.read_csv(csv_file, parse_dates=['date'])
            self.logger.info(f"Загружено {len(df)} записей из CSV для {ticker}")
            return df
        
        except Exception as e:
            self.logger.error(f"Ошибка загрузки из CSV: {str(e)}")
            return pd.DataFrame()
    
    def get_last_date(self, ticker: str) -> str:
        """
        Получает последнюю дату данных для тикера из БД.
        
        Args:
            ticker (str): Тикер акции
        
        Returns:
            str: Последняя дата в формате 'YYYY-MM-DD' или None
        """
        try:
            df = self.database.load_quotes(ticker)
            
            if df.empty:
                return None
            
            last_date = df['date'].max()
            return last_date.strftime('%Y-%m-%d')
        
        except Exception as e:
            self.logger.error(f"Ошибка получения последней даты для {ticker}: {str(e)}")
            return None

