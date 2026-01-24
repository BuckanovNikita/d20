# d20

Инструмент для загрузки датасетов COCO8 и COCO12-Formats из Ultralytics.

## Описание

Этот инструмент позволяет автоматически загружать и распаковывать датасеты COCO из репозитория Ultralytics. Поддерживаются следующие датасеты:

- **COCO8**: Компактный датасет для тестирования, состоящий из первых 8 изображений из COCO train 2017
- **COCO12-Formats**: Датасет в различных форматах (если доступен)

## Установка

```bash
uv sync
```

## Использование

### Базовое использование

```bash
uv run python main.py
```

Это загрузит датасеты COCO8 и COCO12-Formats в директорию `datasets/`.

### Программное использование

```python
from pathlib import Path
from d20.downloader import DownloadConfig, download_datasets

config = DownloadConfig(
    output_dir=Path("my_datasets"),
    datasets=["coco8", "coco12-formats"],
)

download_datasets(config)
```

## Структура проекта

```
d20/
├── d20/
│   ├── __init__.py
│   └── downloader.py  # Основной модуль загрузки
├── main.py            # Точка входа
├── pyproject.toml     # Конфигурация проекта
└── README.md          # Этот файл
```

## Зависимости

- `loguru` - для логирования
- `pydantic` - для валидации конфигурации
- `requests` - для загрузки файлов

## Лицензия

MIT
