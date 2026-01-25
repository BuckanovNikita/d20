# d20

CLI и библиотека для конвертации датасетов детекции между форматами COCO, YOLO и PASCAL VOC.

## Возможности

- Конвертация между `coco`, `yolo`, `voc` в обе стороны.
- Сохранение сплитов `train/val/test` или работа с одним сплитом `data`.
- Настраиваемые директории `images/labels/annotations`.
- HTML-отчет по результатам конвертаций через FiftyOne.

## Установка

```bash
uv sync
```

Для разработки и тестов:

```bash
uv sync --extra dev
```

## Конфигурация

Конфиг задается YAML-файлом и валидируется через Pydantic.

```yaml
class_names:
  - person
  - bicycle
splits:
  - train
  - val
images_dir: images
labels_dir: labels
annotations_dir: annotations
```

Поля:
- `class_names` — список классов (обязателен для YOLO).
- `splits` — список сплитов, по умолчанию `["data"]`.
- `images_dir`, `labels_dir`, `annotations_dir` — имена каталогов внутри датасета.

## Форматы датасетов

### YOLO (Ultralytics-style)

```
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Если сплиты не используются, допускается структура без подпапок `train/val`.

### COCO

```
dataset/
  images/
    train/
    val/
  annotations/
    train.json
    val.json
```

### PASCAL VOC

```
dataset/
  JPEGImages/
  Annotations/
  ImageSets/
    Main/
      train.txt
      val.txt
```

## CLI

Примеры:

```bash
d20 yolo coco --input path/to/yolo --output path/to/coco --config config.yaml
d20 coco voc --input path/to/coco --output path/to/voc --config config.yaml
d20 voc yolo --input path/to/voc --output path/to/yolo --config config.yaml
```

Переопределение настроек:

```bash
d20 yolo coco \
  --input path/to/yolo \
  --output path/to/coco \
  --config config.yaml \
  --splits train,val \
  --images-dir images \
  --labels-dir labels \
  --annotations-dir annotations
```

## Тесты

Фикстуры должны находиться в `tests/fixtures/datasets/`:

```
tests/fixtures/datasets/
  coco8/
  coco8.yaml
  coco12-formats/
  coco12-formats.yaml
```

Если фикстуры отсутствуют, тесты должны падать.

Запуск тестов:

```bash
uv run --extra dev pytest tests/
```

HTML-отчеты FiftyOne сохраняются в `tests/reports/<dataset>/<step>/index.html`.
