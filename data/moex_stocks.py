"""
Модуль для работы со списком акций MOEX через moexalgo.
Получение актуального списка торгуемых акций и их характеристик.
"""

import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import json


class MOEXStocks:
    """
    Класс для работы со списком акций MOEX.
    """
    
    def __init__(self, config, logger):
        """
        Инициализация.
        
        Args:
            config: Объект конфигурации системы
            logger: Объект логгера
        """
        self.config = config
        self.logger = logger
        self.stocks_cache_file = config.base_path / "data" / "moex_stocks_list.json"
        
        self.logger.info("MOEXStocks инициализирован")
    
    def get_all_stocks(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Получение списка всех торгуемых акций на MOEX.
        
        Args:
            use_cache (bool): Использовать кэш
        
        Returns:
            pd.DataFrame: Список акций с характеристиками
        """
        # Проверка кэша
        if use_cache and self.stocks_cache_file.exists():
            try:
                with open(self.stocks_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                self.logger.info(f"Загружено {len(df)} акций из кэша")
                return df
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки кэша: {str(e)}")
        
        try:
            # Используем moexalgo
            import moexalgo
            
            # Получение списка акций
            stocks = moexalgo.Market('stocks').tickers()
            
            # Фильтруем только акции (board='TQBR' - основной режим)
            stocks_df = pd.DataFrame(stocks)
            
            if 'board' in stocks_df.columns:
                stocks_df = stocks_df[stocks_df['board'].isin(['TQBR', 'TQTF'])]
            
            # Выбираем нужные колонки
            columns_to_keep = ['ticker', 'name', 'shortname', 'lotsize', 'board']
            available_columns = [col for col in columns_to_keep if col in stocks_df.columns]
            stocks_df = stocks_df[available_columns]
            
            # Сохранение в кэш
            self.stocks_cache_file.parent.mkdir(parents=True, exist_ok=True)
            stocks_dict = stocks_df.to_dict(orient='records')
            with open(self.stocks_cache_file, 'w', encoding='utf-8') as f:
                json.dump(stocks_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Получено {len(stocks_df)} акций с MOEX")
            return stocks_df
        
        except ImportError:
            self.logger.error("moexalgo не установлен. Установите: pip install moexalgo")
            return self._get_fallback_stocks()
        
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка акций: {str(e)}")
            return self._get_fallback_stocks()
    
    def _get_fallback_stocks(self) -> pd.DataFrame:
        """
        Резервный список популярных акций.
        
        Returns:
            pd.DataFrame: Список акций
        """
        self.logger.info("Использование резервного списка акций")
        
        popular_stocks = [
            {'ticker': 'SBER', 'name': 'Сбербанк', 'board': 'TQBR'},
            {'ticker': 'GAZP', 'name': 'Газпром', 'board': 'TQBR'},
            {'ticker': 'LKOH', 'name': 'Лукойл', 'board': 'TQBR'},
            {'ticker': 'ROSN', 'name': 'Роснефть', 'board': 'TQBR'},
            {'ticker': 'YNDX', 'name': 'Яндекс', 'board': 'TQBR'},
            {'ticker': 'GMKN', 'name': 'Норникель', 'board': 'TQBR'},
            {'ticker': 'NVTK', 'name': 'Новатэк', 'board': 'TQBR'},
            {'ticker': 'TATN', 'name': 'Татнефть', 'board': 'TQBR'},
            {'ticker': 'VTBR', 'name': 'ВТБ', 'board': 'TQBR'},
            {'ticker': 'MGNT', 'name': 'Магнит', 'board': 'TQBR'},
            {'ticker': 'AFLT', 'name': 'Аэрофлот', 'board': 'TQBR'},
            {'ticker': 'MTSS', 'name': 'МТС', 'board': 'TQBR'},
            {'ticker': 'SNGS', 'name': 'Сургутнефтегаз', 'board': 'TQBR'},
            {'ticker': 'ALRS', 'name': 'АЛРОСА', 'board': 'TQBR'},
            {'ticker': 'PLZL', 'name': 'Полюс', 'board': 'TQBR'},
        ]
        
        return pd.DataFrame(popular_stocks)
    
    def get_top_stocks(self, n: int = 50) -> List[str]:
        """
        Получение топ-N акций по ликвидности.
        
        Args:
            n (int): Количество акций
        
        Returns:
            List[str]: Список тикеров
        """
        stocks_df = self.get_all_stocks()
        
        # Берём первые N (moexalgo обычно возвращает отсортированные по ликвидности)
        top_tickers = stocks_df['ticker'].head(n).tolist()
        
        self.logger.info(f"Выбрано топ-{n} акций")
        return top_tickers
    
    def get_stocks_by_board(self, board: str = 'TQBR') -> List[str]:
        """
        Получение акций определённого режима торгов.
        
        Args:
            board (str): Режим торгов (TQBR - основной)
        
        Returns:
            List[str]: Список тикеров
        """
        stocks_df = self.get_all_stocks()
        
        if 'board' in stocks_df.columns:
            filtered = stocks_df[stocks_df['board'] == board]
            tickers = filtered['ticker'].tolist()
        else:
            tickers = stocks_df['ticker'].tolist()
        
        self.logger.info(f"Найдено {len(tickers)} акций в режиме {board}")
        return tickers
    
    def save_stocks_list(self, filename: str = "stocks_to_collect.txt") -> str:
        """
        Сохранение списка акций в текстовый файл.
        
        Args:
            filename (str): Имя файла
        
        Returns:
            str: Путь к файлу
        """
        stocks_df = self.get_all_stocks()
        tickers = stocks_df['ticker'].tolist()
        
        file_path = self.config.base_path / "data" / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tickers))
        
        self.logger.info(f"Список из {len(tickers)} акций сохранён в {file_path}")
        return str(file_path)









