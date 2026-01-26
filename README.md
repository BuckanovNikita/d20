# d20

CLI и библиотека для конвертации датасетов детекции объектов между форматами COCO, YOLO и PASCAL VOC.

## 🚀 Быстрый старт

```bash
# Установка
uv sync

# Простейшая конвертация YOLO → COCO
d20 convert yolo coco \
  --input ./my_yolo_dataset \
  --output ./my_coco_dataset \
  --class-names-file classes.txt
```

## ✨ Возможности

- 🔄 **Двусторонняя конвертация** между `coco`, `yolo`, `voc`
- 📊 **Автоматическое определение** структуры датасета
- 🎯 **Гибкая настройка** директорий и сплитов
- 📈 **Визуализация** через FiftyOne App
- 🛠️ **Простой CLI** и программный API

## 📦 Установка

### Базовая установка

```bash
uv sync
```

После установки команда `d20` будет доступна в вашем терминале.

> 💡 **Для разработчиков:** См. [CONTRIBUTING.md](CONTRIBUTING.md) для информации о разработке и внесении вклада.

## 📖 Примеры использования

### Базовые конвертации

#### YOLO → COCO

```bash
d20 convert yolo coco \
  --input ./datasets/yolo_dataset \
  --output ./datasets/coco_dataset \
  --class-names-file classes.txt
```

#### COCO → YOLO

```bash
d20 convert coco yolo \
  --input ./datasets/coco_dataset \
  --output ./datasets/yolo_dataset
```

#### PASCAL VOC → YOLO

```bash
d20 convert voc yolo \
  --input ./datasets/voc_dataset \
  --output ./datasets/yolo_dataset \
  --class-names-file classes.txt
```

### Работа с классами

#### Из файла

Создайте файл `classes.txt`:
```
person
bicycle
car
motorcycle
```

Используйте его при конвертации:
```bash
d20 convert yolo coco \
  --input ./dataset \
  --output ./output \
  --class-names-file classes.txt
```

#### Автоматическое определение

Для COCO и YOLO (с `data.yaml`) классы определяются автоматически из аннотаций.

### Настройка сплитов

#### Указание конкретных сплитов

```bash
d20 convert coco yolo \
  --input ./dataset \
  --output ./output \
  --splits train,val
```

#### Работа без сплитов (один набор данных)

Если ваш датасет не разделен на train/val/test:
```bash
d20 convert yolo coco \
  --input ./single_dataset \
  --output ./output \
  --splits data
```

### Кастомные директории

Если структура вашего датасета отличается от стандартной:

```bash
d20 convert yolo coco \
  --input ./my_dataset \
  --output ./output \
  --images-dir photos \
  --labels-dir annotations \
  --annotations-dir metadata \
  --class-names-file classes.txt
```

### COCO с одним JSON файлом

Если у вас один JSON файл COCO и отдельная папка с изображениями:

```bash
d20 convert coco yolo \
  --input ./annotations.json \
  --output ./output \
  --images-path ./images
```

### Визуализация датасета (FiftyOne)

Просмотрите ваш датасет в интерактивном интерфейсе:

```bash
# Экспорт всего датасета
d20 export yolo \
  --input ./dataset \
  --class-names-file classes.txt

# Экспорт только train сплита
d20 export coco \
  --input ./dataset \
  --split train
```

После выполнения команды откроется браузер с интерактивным просмотром датасета.

## 📁 Структура форматов

### YOLO (Ultralytics-style)

Стандартная структура:
```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── val/
│       ├── img003.jpg
│       └── img004.jpg
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── img002.txt
    └── val/
        ├── img003.txt
        └── img004.txt
```

Без сплитов:
```
yolo_dataset/
├── images/
│   ├── img001.jpg
│   └── img002.jpg
└── labels/
    ├── img001.txt
    └── img002.txt
```

С `data.yaml`:
```
yolo_dataset/
├── data.yaml
├── images/
│   └── train/
└── labels/
    └── train/
```

### COCO

Стандартная структура:
```
coco_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── val/
│       ├── img003.jpg
│       └── img004.jpg
└── annotations/
    ├── train.json
    └── val.json
```

Один JSON файл:
```
project/
├── annotations.json
└── images/
    ├── img001.jpg
    └── img002.jpg
```

### PASCAL VOC

```
voc_dataset/
├── JPEGImages/
│   ├── img001.jpg
│   └── img002.jpg
├── Annotations/
│   ├── img001.xml
│   └── img002.xml
└── ImageSets/
    └── Main/
        ├── train.txt
        └── val.txt
```

## 🎯 Типичные сценарии

### Сценарий 1: Подготовка датасета для обучения YOLO

У вас есть датасет в формате COCO, нужно конвертировать в YOLO:

```bash
# 1. Создайте файл с классами (если нужно)
echo -e "person\nbicycle\ncar" > classes.txt

# 2. Конвертируйте
d20 convert coco yolo \
  --input ./coco_dataset \
  --output ./yolo_dataset \
  --class-names-file classes.txt

# 3. Проверьте результат
d20 export yolo \
  --input ./yolo_dataset \
  --class-names-file classes.txt
```

### Сценарий 2: Конвертация для разных фреймворков

```bash
# Исходный датасет в YOLO
SOURCE="./my_dataset"

# Для PyTorch (YOLO)
d20 convert yolo yolo \
  --input "$SOURCE" \
  --output ./pytorch_dataset \
  --class-names-file classes.txt

# Для TensorFlow (COCO)
d20 convert yolo coco \
  --input "$SOURCE" \
  --output ./tensorflow_dataset \
  --class-names-file classes.txt

# Для старых инструментов (VOC)
d20 convert yolo voc \
  --input "$SOURCE" \
  --output ./voc_dataset \
  --class-names-file classes.txt
```

### Сценарий 3: Объединение и разделение сплитов

```bash
# Конвертируем только train и val (без test)
d20 convert coco yolo \
  --input ./full_dataset \
  --output ./trainval_dataset \
  --splits train,val
```

### Сценарий 4: Работа с нестандартной структурой

Если ваш датасет имеет нестандартную структуру:

```bash
d20 convert yolo coco \
  --input ./custom_dataset \
  --output ./standard_dataset \
  --images-dir photos \
  --labels-dir labels_txt \
  --annotations-dir coco_annotations \
  --splits train,val,test \
  --class-names-file my_classes.txt
```

## ⚙️ Конфигурация через YAML

Создайте файл `config.yaml`:

```yaml
class_names:
  - person
  - bicycle
  - car
  - motorcycle
  - airplane
  - bus
  - train
  - truck
splits:
  - train
  - val
images_dir: images
labels_dir: labels
annotations_dir: annotations
```

Использование (если поддерживается в будущих версиях):
```bash
d20 convert yolo coco \
  --input ./dataset \
  --output ./output \
  --config config.yaml
```

## 🐛 Решение проблем

### Ошибка: "Class names are required"

**Проблема:** Для YOLO формата нужны имена классов.

**Решение:**
```bash
# Создайте файл classes.txt
echo -e "class1\nclass2\nclass3" > classes.txt

# Укажите его при конвертации
d20 convert yolo coco \
  --input ./dataset \
  --output ./output \
  --class-names-file classes.txt
```

### Ошибка: "Split not found"

**Проблема:** Указанный сплит отсутствует в датасете.

**Решение:** Проверьте доступные сплиты или используйте автоопределение:
```bash
# Без указания --splits (автоопределение)
d20 convert coco yolo \
  --input ./dataset \
  --output ./output
```

### Изображения не найдены

**Проблема:** Пути к изображениям не совпадают с аннотациями.

**Решение:** Укажите правильные директории:
```bash
d20 convert coco yolo \
  --input ./dataset \
  --output ./output \
  --images-dir photos \
  --class-names-file classes.txt
```

## 📚 Дополнительная информация

### Поддерживаемые форматы

- **YOLO** (Ultralytics) - самый популярный формат для YOLO моделей
- **COCO** - стандартный формат для многих фреймворков
- **PASCAL VOC** - классический формат для детекции объектов

### Автоматическое определение

d20 автоматически определяет:
- Структуру датасета (сплиты, директории)
- Формат аннотаций
- Классы (для COCO и YOLO с YAML)
