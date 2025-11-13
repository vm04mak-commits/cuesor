"""
Модуль построения графиков.
Создаёт визуализации для анализа акций.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any
from pathlib import Path


class ChartBuilder:
    """
    Класс для построения графиков и визуализаций.
    """
    
    def __init__(self, logger):
        """
        Инициализация построителя графиков.
        
        Args:
            logger: Объект логгера
        """
        self.logger = logger
        
        # Настройка стиля matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def build_price_chart(self, ticker: str, data: pd.DataFrame, output_dir: Path) -> str:
        """
        Построение графика цен.
        
        Args:
            ticker (str): Тикер акции
            data (pd.DataFrame): DataFrame с данными
            output_dir (Path): Директория для сохранения
        
        Returns:
            str: Путь к сохранённому графику
        """
        self.logger.info(f"Построение графика цен для {ticker}")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # График цены закрытия
            ax.plot(data['date'], data['close'], label='Цена закрытия', linewidth=2)
            
            # Оформление
            ax.set_title(f'{ticker} - История цен', fontsize=14, fontweight='bold')
            ax.set_xlabel('Дата', fontsize=12)
            ax.set_ylabel('Цена (₽)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Форматирование дат на оси X
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Сохранение
            chart_path = output_dir / f"{ticker}_price_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"График сохранён: {chart_path}")
            return str(chart_path)
        
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика цен: {str(e)}")
            plt.close()
            return ""
    
    def build_technical_chart(self, ticker: str, data: pd.DataFrame, 
                            technical: Dict[str, Any], output_dir: Path) -> str:
        """
        Построение графика с техническими индикаторами.
        
        Args:
            ticker (str): Тикер акции
            data (pd.DataFrame): DataFrame с данными
            technical (Dict[str, Any]): Технические индикаторы
            output_dir (Path): Директория для сохранения
        
        Returns:
            str: Путь к сохранённому графику
        """
        self.logger.info(f"Построение графика технических индикаторов для {ticker}")
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Верхний график: цены и SMA
            ax1.plot(data['date'], data['close'], label='Цена', linewidth=2)
            
            if 'sma_20' in technical:
                ax1.plot(data['date'], technical['sma_20'], 
                        label='SMA(20)', linestyle='--', alpha=0.7)
            
            if 'sma_50' in technical:
                ax1.plot(data['date'], technical['sma_50'], 
                        label='SMA(50)', linestyle='--', alpha=0.7)
            
            ax1.set_title(f'{ticker} - Технический анализ', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Цена (₽)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Нижний график: RSI
            if 'rsi' in technical:
                ax2.plot(data['date'], technical['rsi'], label='RSI', color='purple', linewidth=2)
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Перекупленность')
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Перепроданность')
                ax2.set_ylabel('RSI', fontsize=12)
                ax2.set_ylim(0, 100)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Форматирование дат
            ax2.set_xlabel('Дата', fontsize=12)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Сохранение
            chart_path = output_dir / f"{ticker}_technical_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"График сохранён: {chart_path}")
            return str(chart_path)
        
        except Exception as e:
            self.logger.error(f"Ошибка при построении технического графика: {str(e)}")
            plt.close()
            return ""
    
    def build_volatility_chart(self, ticker: str, data: pd.DataFrame, output_dir: Path) -> str:
        """
        Построение графика волатильности.
        
        Args:
            ticker (str): Тикер акции
            data (pd.DataFrame): DataFrame с данными
            output_dir (Path): Директория для сохранения
        
        Returns:
            str: Путь к сохранённому графику
        """
        self.logger.info(f"Построение графика волатильности для {ticker}")
        
        try:
            # Расчёт волатильности (скользящее стандартное отклонение доходности)
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(data['date'], volatility * 100, label='Волатильность (20 дней)', 
                   color='orange', linewidth=2)
            
            # Средняя волатильность
            mean_vol = volatility.mean() * 100
            ax.axhline(y=mean_vol, color='r', linestyle='--', alpha=0.5, 
                      label=f'Средняя: {mean_vol:.2f}%')
            
            ax.set_title(f'{ticker} - Волатильность', fontsize=14, fontweight='bold')
            ax.set_xlabel('Дата', fontsize=12)
            ax.set_ylabel('Волатильность (%)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Форматирование дат
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Сохранение
            chart_path = output_dir / f"{ticker}_volatility_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"График сохранён: {chart_path}")
            return str(chart_path)
        
        except Exception as e:
            self.logger.error(f"Ошибка при построении графика волатильности: {str(e)}")
            plt.close()
            return ""









