"""
Модуль главного анализатора.
Координирует технический и фундаментальный анализ.
"""

import pandas as pd
from typing import Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from .technical_indicators import TechnicalIndicators
from .correlations import CorrelationAnalyzer
from core.database import Database


class Analyzer:
    """
    Главный класс анализа данных.
    Выполняет технический и фундаментальный анализ.
    """
    
    def __init__(self, config, logger):
        """
        Инициализация анализатора.
        
        Args:
            config: Объект конфигурации системы
            logger: Объект логгера
        """
        self.config = config
        self.logger = logger
        self.technical = TechnicalIndicators(logger)
        self.correlation = CorrelationAnalyzer(logger)
        
        # База данных для сохранения индикаторов
        db_path = config.base_path / "data" / "market_data.db"
        self.database = Database(db_path, logger)
        
        # CSV директория для индикаторов
        self.indicators_csv_dir = config.base_path / "data" / "csv"
        self.indicators_csv_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Analyzer инициализирован")
    
    def analyze(self, data: pd.DataFrame, ticker: str = None, save_indicators: bool = True) -> Dict[str, Any]:
        """
        Полный анализ данных о акции.
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            ticker (str): Тикер акции (для сохранения индикаторов)
            save_indicators (bool): Сохранять ли индикаторы в БД и CSV
        
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        self.logger.info(f"Запуск анализа для {len(data)} записей")
        
        if data.empty:
            self.logger.warning("Пустой DataFrame для анализа")
            return {}
        
        results = {}
        
        try:
            # Технический анализ
            results['technical'] = self._technical_analysis(data)
            
            # Базовая статистика
            results['statistics'] = self._calculate_statistics(data)
            
            # Волатильность
            results['volatility'] = self._calculate_volatility(data)
            
            # Тренд
            results['trend'] = self._identify_trend(data)
            
            # Сохранение индикаторов
            if save_indicators and ticker:
                self.save_indicators(ticker, data, results['technical'])
            
            self.logger.info("Анализ завершён успешно")
            return results
        
        except Exception as e:
            self.logger.exception("Ошибка при анализе данных")
            raise
    
    def _technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Технический анализ с индикаторами.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
        
        Returns:
            Dict[str, Any]: Результаты технического анализа
        """
        self.logger.info("Выполнение технического анализа")
        
        results = {}
        
        # Вычисление индикаторов
        indicators_config = self.config.get("analysis.indicators", ["SMA", "EMA", "RSI"])
        
        if "SMA" in indicators_config:
            results['sma_20'] = self.technical.sma(data['close'], period=20)
            results['sma_50'] = self.technical.sma(data['close'], period=50)
        
        if "EMA" in indicators_config:
            results['ema_12'] = self.technical.ema(data['close'], period=12)
            results['ema_26'] = self.technical.ema(data['close'], period=26)
        
        if "RSI" in indicators_config:
            results['rsi'] = self.technical.rsi(data['close'], period=14)
        
        if "MACD" in indicators_config:
            macd_data = self.technical.macd(data['close'])
            results['macd'] = macd_data
        
        if "BB" in indicators_config:
            bb_data = self.technical.bollinger_bands(data['close'], period=20)
            results['bollinger_bands'] = bb_data
        
        # Текущие значения
        current_values = {
            'close': float(data['close'].iloc[-1]),
            'rsi': float(results['rsi'].iloc[-1]) if 'rsi' in results else None,
            'sma_20': float(results['sma_20'].iloc[-1]) if 'sma_20' in results else None,
            'sma_50': float(results['sma_50'].iloc[-1]) if 'sma_50' in results else None
        }
        results['current_values'] = current_values
        
        return results
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчёт базовой статистики.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
        
        Returns:
            Dict[str, float]: Статистические показатели
        """
        self.logger.info("Расчёт статистики")
        
        stats = {
            'mean': float(data['close'].mean()),
            'median': float(data['close'].median()),
            'std': float(data['close'].std()),
            'min': float(data['close'].min()),
            'max': float(data['close'].max()),
            'current': float(data['close'].iloc[-1]),
            'change': float((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100)
        }
        
        return stats
    
    def _calculate_volatility(self, data: pd.DataFrame, period: int = 30) -> Dict[str, float]:
        """
        Расчёт волатильности.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
            period (int): Период для расчёта
        
        Returns:
            Dict[str, float]: Показатели волатильности
        """
        self.logger.info(f"Расчёт волатильности за {period} дней")
        
        # Дневная доходность
        returns = data['close'].pct_change()
        
        # Волатильность (стандартное отклонение доходности)
        volatility = float(returns.std())
        
        # Годовая волатильность (умножаем на корень из числа торговых дней)
        annual_volatility = volatility * (252 ** 0.5)
        
        # Волатильность за последние N дней
        recent_volatility = float(returns.tail(period).std())
        
        result = {
            'daily_volatility': volatility,
            'annual_volatility': annual_volatility,
            f'volatility_{period}d': recent_volatility
        }
        
        return result
    
    def _identify_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Определение тренда.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
        
        Returns:
            Dict[str, Any]: Информация о тренде
        """
        self.logger.info("Определение тренда")
        
        # Простое определение на основе SMA
        sma_20 = self.technical.sma(data['close'], period=20)
        sma_50 = self.technical.sma(data['close'], period=50)
        
        current_price = float(data['close'].iloc[-1])
        current_sma_20 = float(sma_20.iloc[-1])
        current_sma_50 = float(sma_50.iloc[-1])
        
        # Определение тренда
        if current_sma_20 > current_sma_50 and current_price > current_sma_20:
            trend = "uptrend"
            strength = "strong"
        elif current_sma_20 > current_sma_50 and current_price < current_sma_20:
            trend = "uptrend"
            strength = "weak"
        elif current_sma_20 < current_sma_50 and current_price < current_sma_20:
            trend = "downtrend"
            strength = "strong"
        elif current_sma_20 < current_sma_50 and current_price > current_sma_20:
            trend = "downtrend"
            strength = "weak"
        else:
            trend = "sideways"
            strength = "neutral"
        
        result = {
            'trend': trend,
            'strength': strength,
            'current_price': current_price,
            'sma_20': current_sma_20,
            'sma_50': current_sma_50
        }
        
        return result
    
    def compare_stocks(self, stocks_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Сравнительный анализ нескольких акций.
        
        Args:
            stocks_data (Dict[str, pd.DataFrame]): Словарь {тикер: данные}
        
        Returns:
            Dict[str, Any]: Результаты сравнительного анализа
        """
        self.logger.info(f"Сравнительный анализ {len(stocks_data)} акций")
        
        results = {}
        
        # Корреляционный анализ
        correlation_matrix = self.correlation.calculate_correlation_matrix(stocks_data)
        results['correlation_matrix'] = correlation_matrix
        
        # Сравнение доходностей
        returns_comparison = {}
        for ticker, data in stocks_data.items():
            if not data.empty:
                total_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
                returns_comparison[ticker] = float(total_return)
        
        results['returns_comparison'] = returns_comparison
        
        return results
    
    def save_indicators(self, ticker: str, data: pd.DataFrame, technical: Dict[str, Any]) -> None:
        """
        Сохранение индикаторов в БД и CSV.
        
        Args:
            ticker (str): Тикер акции
            data (pd.DataFrame): Данные с датами
            technical (Dict[str, Any]): Технические индикаторы
        
        Returns:
            None
        """
        try:
            # Создаём DataFrame с индикаторами
            indicators_df = pd.DataFrame()
            
            # Добавляем дату из исходных данных
            if 'date' in data.columns:
                indicators_df['date'] = data['date']
            else:
                indicators_df['date'] = pd.date_range(start='2020-01-01', periods=len(data))
            
            # Добавляем индикаторы
            for indicator_name, indicator_values in technical.items():
                if indicator_name == 'current_values':
                    continue  # Пропускаем текущие значения
                
                if isinstance(indicator_values, pd.Series):
                    indicators_df[indicator_name] = indicator_values.values
                elif isinstance(indicator_values, dict):
                    # Для MACD, Bollinger Bands и т.д.
                    for sub_name, sub_values in indicator_values.items():
                        if isinstance(sub_values, pd.Series):
                            col_name = f"{indicator_name}_{sub_name}"
                            indicators_df[col_name] = sub_values.values
            
            # Сохранение в БД
            indicators_df_indexed = indicators_df.set_index('date')
            self.database.save_indicators_bulk(ticker, indicators_df_indexed)
            
            # Сохранение в CSV
            self._save_indicators_csv(ticker, indicators_df)
            
            self.logger.info(f"Индикаторы для {ticker} сохранены")
        
        except Exception as e:
            self.logger.error(f"Ошибка сохранения индикаторов: {str(e)}")
    
    def _save_indicators_csv(self, ticker: str, indicators_df: pd.DataFrame) -> None:
        """
        Сохранение индикаторов в CSV файл.
        
        Args:
            ticker (str): Тикер акции
            indicators_df (pd.DataFrame): DataFrame с индикаторами
        
        Returns:
            None
        """
        try:
            # Создаём директорию для тикера
            ticker_dir = self.indicators_csv_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            
            # Путь к файлу индикаторов
            csv_file = ticker_dir / f"{ticker}_indicators.csv"
            
            # Если файл существует, объединяем
            if csv_file.exists():
                existing_data = pd.read_csv(csv_file, parse_dates=['date'])
                
                # Объединяем по дате
                combined = pd.merge(
                    existing_data, indicators_df,
                    on='date', how='outer',
                    suffixes=('', '_new')
                )
                
                # Обновляем значения из новых данных
                for col in indicators_df.columns:
                    if col != 'date' and f"{col}_new" in combined.columns:
                        combined[col] = combined[f"{col}_new"].fillna(combined[col])
                        combined = combined.drop(columns=[f"{col}_new"])
                
                combined = combined.sort_values('date').reset_index(drop=True)
                combined.to_csv(csv_file, index=False)
            else:
                indicators_df.to_csv(csv_file, index=False)
            
            self.logger.info(f"Индикаторы сохранены в CSV: {csv_file}")
        
        except Exception as e:
            self.logger.error(f"Ошибка сохранения индикаторов в CSV: {str(e)}")
    
    def load_indicators(self, ticker: str, from_db: bool = True) -> pd.DataFrame:
        """
        Загрузка индикаторов из БД или CSV.
        
        Args:
            ticker (str): Тикер акции
            from_db (bool): Загрузить из БД (True) или из CSV (False)
        
        Returns:
            pd.DataFrame: DataFrame с индикаторами
        """
        if from_db:
            return self.database.load_indicators(ticker)
        else:
            csv_file = self.indicators_csv_dir / ticker / f"{ticker}_indicators.csv"
            if csv_file.exists():
                return pd.read_csv(csv_file, parse_dates=['date'], index_col='date')
            return pd.DataFrame()
    
    def get_last_indicator_date(self, ticker: str) -> str:
        """
        Получает последнюю дату рассчитанных индикаторов.
        
        Args:
            ticker (str): Тикер акции
        
        Returns:
            str: Последняя дата в формате 'YYYY-MM-DD' или None
        """
        try:
            indicators_df = self.database.load_indicators(ticker)
            
            if indicators_df.empty:
                return None
            
            last_date = indicators_df.index.max()
            return last_date.strftime('%Y-%m-%d')
        
        except Exception as e:
            self.logger.error(f"Ошибка получения последней даты индикаторов для {ticker}: {str(e)}")
            return None

