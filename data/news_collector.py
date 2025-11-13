"""
Модуль сбора новостей о компаниях.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import json
from pathlib import Path


class NewsCollector:
    """
    Класс для сбора новостей о компаниях.
    """
    
    def __init__(self, config, logger):
        """
        Инициализация сборщика новостей.
        
        Args:
            config: Объект конфигурации системы
            logger: Объект логгера
        """
        self.config = config
        self.logger = logger
        self.news_dir = config.raw_data_dir / "news"
        self.news_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("NewsCollector инициализирован")
    
    def fetch_news(self, ticker: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Получение новостей о компании за последние N дней.
        
        Args:
            ticker (str): Тикер компании
            days (int): Количество дней для анализа
        
        Returns:
            List[Dict[str, Any]]: Список новостей
        """
        self.logger.info(f"Запрос новостей для {ticker} за последние {days} дней")
        
        # TODO: Интеграция с реальными источниками новостей
        # Пока возвращаем заглушку
        
        news = [
            {
                'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'title': f'Новость {i} о {ticker}',
                'source': 'Example Source',
                'sentiment': 'neutral',
                'url': f'https://example.com/news/{i}'
            }
            for i in range(days)
        ]
        
        self.logger.info(f"Найдено {len(news)} новостей для {ticker}")
        return news
    
    def analyze_sentiment(self, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализ тональности новостей.
        
        Args:
            news (List[Dict[str, Any]]): Список новостей
        
        Returns:
            Dict[str, Any]: Результаты анализа тональности
        """
        self.logger.info(f"Анализ тональности {len(news)} новостей")
        
        # Подсчёт тональности
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for item in news:
            sentiment = item.get('sentiment', 'neutral')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        total = len(news)
        sentiment_scores = {
            k: v / total if total > 0 else 0
            for k, v in sentiment_counts.items()
        }
        
        # Общий индекс тональности (-1 до 1)
        sentiment_index = (
            sentiment_scores['positive'] - sentiment_scores['negative']
        )
        
        result = {
            'total_news': total,
            'sentiment_counts': sentiment_counts,
            'sentiment_scores': sentiment_scores,
            'sentiment_index': sentiment_index
        }
        
        self.logger.info(f"Индекс тональности: {sentiment_index:.2f}")
        return result
    
    def save_news(self, ticker: str, news: List[Dict[str, Any]]) -> None:
        """
        Сохранение новостей в файл.
        
        Args:
            ticker (str): Тикер компании
            news (List[Dict[str, Any]]): Список новостей
        
        Returns:
            None
        """
        file_path = self.news_dir / f"{ticker}_news.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(news, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Новости сохранены: {file_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении новостей: {str(e)}")









