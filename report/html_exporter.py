"""
Модуль экспорта отчётов в HTML.
Создаёт красивые HTML-отчёты с графиками.
"""

from typing import Dict, Any
from pathlib import Path
from datetime import datetime


class HTMLExporter:
    """
    Класс для экспорта отчётов в HTML формат.
    """
    
    def __init__(self, logger):
        """
        Инициализация экспортёра.
        
        Args:
            logger: Объект логгера
        """
        self.logger = logger
    
    def export(self, ticker: str, analysis: Dict[str, Any], 
              prediction: Dict[str, Any], charts: Dict[str, str], 
              output_dir: Path) -> str:
        """
        Экспорт отчёта в HTML.
        
        Args:
            ticker (str): Тикер акции
            analysis (Dict[str, Any]): Результаты анализа
            prediction (Dict[str, Any]): Результаты прогноза
            charts (Dict[str, str]): Словарь с путями к графикам
            output_dir (Path): Директория для сохранения
        
        Returns:
            str: Путь к HTML файлу
        """
        self.logger.info(f"Экспорт отчёта в HTML для {ticker}")
        
        try:
            html_content = self._generate_html(ticker, analysis, prediction, charts, output_dir)
            
            html_file = output_dir / f"{ticker}_report.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML отчёт сохранён: {html_file}")
            return str(html_file)
        
        except Exception as e:
            self.logger.error(f"Ошибка при экспорте в HTML: {str(e)}")
            return ""
    
    def _generate_html(self, ticker: str, analysis: Dict[str, Any], 
                      prediction: Dict[str, Any], charts: Dict[str, str],
                      output_dir: Path) -> str:
        """
        Генерация HTML контента.
        
        Args:
            ticker (str): Тикер акции
            analysis (Dict[str, Any]): Результаты анализа
            prediction (Dict[str, Any]): Результаты прогноза
            charts (Dict[str, str]): Пути к графикам
            output_dir (Path): Директория отчёта
        
        Returns:
            str: HTML контент
        """
        # Извлекаем данные
        stats = analysis.get('statistics', {})
        trend = analysis.get('trend', {})
        vol = analysis.get('volatility', {})
        rec = prediction.get('recommendation', {})
        
        # Генерация HTML
        html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчёт: {ticker}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .metric-card h3 {{
            color: #555;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .recommendation {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        
        .recommendation h3 {{
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        
        .recommendation .action {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .charts {{
            margin-top: 30px;
        }}
        
        .chart-container {{
            margin-bottom: 30px;
        }}
        
        .chart-container img {{
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Инвестиционный отчёт: {ticker}</h1>
            <p>Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- Статистика -->
            <div class="section">
                <h2>📊 Статистика</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Текущая цена</h3>
                        <div class="value">{stats.get('current', 0):.2f} ₽</div>
                    </div>
                    <div class="metric-card">
                        <h3>Изменение</h3>
                        <div class="value">{stats.get('change', 0):+.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Минимум</h3>
                        <div class="value">{stats.get('min', 0):.2f} ₽</div>
                    </div>
                    <div class="metric-card">
                        <h3>Максимум</h3>
                        <div class="value">{stats.get('max', 0):.2f} ₽</div>
                    </div>
                </div>
            </div>
            
            <!-- Тренд -->
            <div class="section">
                <h2>📈 Тренд</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Направление</h3>
                        <div class="value">{trend.get('trend', 'unknown').upper()}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Сила</h3>
                        <div class="value">{trend.get('strength', 'unknown').upper()}</div>
                    </div>
                </div>
            </div>
            
            <!-- Волатильность -->
            <div class="section">
                <h2>⚡ Волатильность</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Дневная</h3>
                        <div class="value">{vol.get('daily_volatility', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Годовая</h3>
                        <div class="value">{vol.get('annual_volatility', 0):.4f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Рекомендация -->
            <div class="recommendation">
                <h3>💡 Рекомендация</h3>
                <div class="action">{rec.get('action', 'HOLD').upper()}</div>
                <p><strong>Причина:</strong> {rec.get('reason', 'Нет данных')}</p>
                <p><strong>Уверенность:</strong> {rec.get('confidence', 'unknown').upper()}</p>
            </div>
            
            <!-- Графики -->
            <div class="section">
                <h2>📉 Графики</h2>
                <div class="charts">
"""
        
        # Добавляем графики
        for chart_name, chart_path in charts.items():
            if chart_path:
                # Получаем относительный путь от HTML файла
                chart_file = Path(chart_path).name
                html += f"""
                    <div class="chart-container">
                        <img src="{chart_file}" alt="{chart_name}">
                    </div>
"""
        
        html += """
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Investment AI Assistant • Создано автоматически</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html









