"""
Модуль для работы с базой данных.
Использует SQLite для хранения исторических данных и индикаторов.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime


class Database:
    """
    Класс для работы с базой данных котировок и индикаторов.
    """
    
    def __init__(self, db_path: Path, logger):
        """
        Инициализация подключения к базе данных.
        
        Args:
            db_path (Path): Путь к файлу базы данных
            logger: Объект логгера
        """
        self.db_path = db_path
        self.logger = logger
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self.logger.info(f"Database инициализирована: {db_path}")
    
    def _init_database(self) -> None:
        """
        Инициализация структуры базы данных.
        
        Returns:
            None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица котировок
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL NOT NULL,
                volume INTEGER,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        
        # Таблица индикаторов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                indicator_name TEXT NOT NULL,
                indicator_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date, indicator_name)
            )
        """)
        
        # Таблица метаданных
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                ticker TEXT PRIMARY KEY,
                last_update TIMESTAMP,
                records_count INTEGER,
                first_date DATE,
                last_date DATE
            )
        """)
        
        # Индексы для ускорения поиска
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotes_ticker_date ON quotes(ticker, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_ticker_date ON indicators(ticker, date)")
        
        conn.commit()
        conn.close()
    
    def save_quotes(self, ticker: str, data: pd.DataFrame) -> int:
        """
        Сохранение котировок в базу данных.
        
        Args:
            ticker (str): Тикер акции
            data (pd.DataFrame): DataFrame с котировками
        
        Returns:
            int: Количество сохранённых записей
        """
        if data.empty:
            self.logger.warning(f"Пустой DataFrame для {ticker}")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Подготовка данных
            df = data.copy()
            df['ticker'] = ticker
            
            # Конвертация даты в строку
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Сохранение (replace для обновления существующих)
            records_saved = df.to_sql('quotes', conn, if_exists='append', index=False, 
                                     method='multi', chunksize=1000)
            
            # Обновление метаданных
            self._update_metadata(conn, ticker, df)
            
            conn.commit()
            self.logger.info(f"Сохранено {len(df)} котировок для {ticker}")
            return len(df)
        
        except sqlite3.IntegrityError:
            # Если есть дубликаты, обновляем построчно
            saved = 0
            for _, row in data.iterrows():
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO quotes 
                        (ticker, date, open, high, low, close, volume, value)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker,
                        pd.to_datetime(row['date']).strftime('%Y-%m-%d') if 'date' in row else None,
                        row.get('open'),
                        row.get('high'),
                        row.get('low'),
                        row.get('close'),
                        row.get('volume'),
                        row.get('value')
                    ))
                    saved += 1
                except Exception as e:
                    self.logger.error(f"Ошибка сохранения записи: {str(e)}")
            
            conn.commit()
            self.logger.info(f"Обновлено {saved} котировок для {ticker}")
            return saved
        
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Ошибка сохранения котировок: {str(e)}")
            raise
        
        finally:
            conn.close()
    
    def save_indicators(self, ticker: str, date: str, indicators: dict) -> int:
        """
        Сохранение индикаторов для определённой даты.
        
        Args:
            ticker (str): Тикер акции
            date (str): Дата в формате YYYY-MM-DD
            indicators (dict): Словарь {имя_индикатора: значение}
        
        Returns:
            int: Количество сохранённых индикаторов
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            saved = 0
            for indicator_name, value in indicators.items():
                if pd.notna(value):  # Пропускаем NaN
                    cursor.execute("""
                        INSERT OR REPLACE INTO indicators
                        (ticker, date, indicator_name, indicator_value)
                        VALUES (?, ?, ?, ?)
                    """, (ticker, date, indicator_name, float(value)))
                    saved += 1
            
            conn.commit()
            self.logger.debug(f"Сохранено {saved} индикаторов для {ticker} на {date}")
            return saved
        
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Ошибка сохранения индикаторов: {str(e)}")
            raise
        
        finally:
            conn.close()
    
    def save_indicators_bulk(self, ticker: str, indicators_df: pd.DataFrame) -> int:
        """
        Массовое сохранение индикаторов.
        
        Args:
            ticker (str): Тикер акции
            indicators_df (pd.DataFrame): DataFrame с индикаторами (индекс - даты)
        
        Returns:
            int: Количество сохранённых записей
        """
        if indicators_df.empty:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            saved = 0
            for date_idx, row in indicators_df.iterrows():
                date_str = pd.to_datetime(date_idx).strftime('%Y-%m-%d')
                for col_name, value in row.items():
                    if pd.notna(value):
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO indicators
                            (ticker, date, indicator_name, indicator_value)
                            VALUES (?, ?, ?, ?)
                        """, (ticker, date_str, col_name, float(value)))
                        saved += 1
            
            conn.commit()
            self.logger.info(f"Сохранено {saved} индикаторов для {ticker}")
            return saved
        
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Ошибка массового сохранения индикаторов: {str(e)}")
            raise
        
        finally:
            conn.close()
    
    def load_quotes(self, ticker: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Загрузка котировок из базы данных.
        
        Args:
            ticker (str): Тикер акции
            start_date (Optional[str]): Начальная дата (YYYY-MM-DD)
            end_date (Optional[str]): Конечная дата (YYYY-MM-DD)
        
        Returns:
            pd.DataFrame: DataFrame с котировками
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = "SELECT date, open, high, low, close, volume, value FROM quotes WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            
            self.logger.info(f"Загружено {len(df)} котировок для {ticker}")
            return df
        
        except Exception as e:
            self.logger.error(f"Ошибка загрузки котировок: {str(e)}")
            return pd.DataFrame()
        
        finally:
            conn.close()
    
    def load_indicators(self, ticker: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Загрузка индикаторов из базы данных.
        
        Args:
            ticker (str): Тикер акции
            start_date (Optional[str]): Начальная дата
            end_date (Optional[str]): Конечная дата
        
        Returns:
            pd.DataFrame: DataFrame с индикаторами (колонки - индикаторы, индекс - даты)
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = "SELECT date, indicator_name, indicator_value FROM indicators WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return pd.DataFrame()
            
            # Pivot для удобства (даты в индексе, индикаторы в колонках)
            pivot_df = df.pivot(index='date', columns='indicator_name', values='indicator_value')
            pivot_df.index = pd.to_datetime(pivot_df.index)
            
            self.logger.info(f"Загружено {len(pivot_df)} записей индикаторов для {ticker}")
            return pivot_df
        
        except Exception as e:
            self.logger.error(f"Ошибка загрузки индикаторов: {str(e)}")
            return pd.DataFrame()
        
        finally:
            conn.close()
    
    def get_available_tickers(self) -> List[str]:
        """
        Получение списка доступных тикеров в базе.
        
        Returns:
            List[str]: Список тикеров
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM quotes ORDER BY ticker")
            tickers = [row[0] for row in cursor.fetchall()]
            return tickers
        
        finally:
            conn.close()
    
    def get_metadata(self, ticker: str) -> dict:
        """
        Получение метаданных по тикеру.
        
        Args:
            ticker (str): Тикер акции
        
        Returns:
            dict: Метаданные
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT last_update, records_count, first_date, last_date
                FROM metadata WHERE ticker = ?
            """, (ticker,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'last_update': row[0],
                    'records_count': row[1],
                    'first_date': row[2],
                    'last_date': row[3]
                }
            return {}
        
        finally:
            conn.close()
    
    def _update_metadata(self, conn, ticker: str, data: pd.DataFrame) -> None:
        """
        Обновление метаданных по тикеру.
        
        Args:
            conn: Подключение к БД
            ticker (str): Тикер
            data (pd.DataFrame): Данные
        
        Returns:
            None
        """
        cursor = conn.cursor()
        
        # Получаем статистику
        cursor.execute("""
            SELECT COUNT(*), MIN(date), MAX(date)
            FROM quotes WHERE ticker = ?
        """, (ticker,))
        
        count, first_date, last_date = cursor.fetchone()
        
        # Обновляем метаданные
        cursor.execute("""
            INSERT OR REPLACE INTO metadata
            (ticker, last_update, records_count, first_date, last_date)
            VALUES (?, ?, ?, ?, ?)
        """, (ticker, datetime.now(), count, first_date, last_date))
    
    def clear_ticker_data(self, ticker: str) -> None:
        """
        Удаление всех данных по тикеру.
        
        Args:
            ticker (str): Тикер акции
        
        Returns:
            None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM quotes WHERE ticker = ?", (ticker,))
            cursor.execute("DELETE FROM indicators WHERE ticker = ?", (ticker,))
            cursor.execute("DELETE FROM metadata WHERE ticker = ?", (ticker,))
            conn.commit()
            
            self.logger.info(f"Данные для {ticker} удалены из БД")
        
        finally:
            conn.close()
    
    def get_available_tickers(self) -> List[str]:
        """
        Получает список всех доступных тикеров в базе данных.
        
        Returns:
            List[str]: Список тикеров
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM quotes ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        self.logger.debug(f"Найдено {len(tickers)} тикеров в БД.")
        return tickers

