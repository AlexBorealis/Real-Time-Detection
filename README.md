# Real-Time-Detection

## Результаты и прогнозы
Результаты обучения моделей `/Real-Time-Detection/results/models`

Визуализации прогнозов модели с вероятностями детекции - `/Real-Time-Detection/results/visualizations/model_name/predict`

Визуализации прогнозов модели в сравнении с реальными рамками - `/Real-Time-Detection/results/visualizations/model_name/comparison`

Предсказания на видео `/Real-Time-Detection/results/visualizations/model_name/videos`

## Методики проведения экспериментов

### Цель экспериментов
Оценка производительности моделей YOLOv8 для задач детекции объектов в реальном времени на датасете BDD100K, включая базовую модель, оптимизированные версии (обрезка, квантизация) и тестирование на edge-устройствах.

### Этапы экспериментов
- **Подготовка данных**: Использование датасета BDD100K. Скачивание изображений (100k images, Labels) и видео (BDD-Attention). Разделение на тренировочную, валидационную и тестовую выборки согласно конфигурации `/Real-Time-Detection/config/datasets/dataset.yaml`.
- **Предобработка**: Запуск скрипта `preprocess.py` для аугментации, нормализации и подготовки данных. Команда: `pipenv run python -m src.data.preprocess`.
- **Обучение моделей**: Основной цикл обучения через `train.py`. Параметры: эпохи, learning rate, batch size из `/Real-Time-Detection/config/models/config.yaml`. Команда: `pipenv run python -m src.models.train.train`.
- **Оптимизация**: 
  - Обрезка модели (`pruning.py`) с параметром `prune-amount` (например, 0.05). Команда: `pipenv run python -m src.models.optimization.pruning --model 1 --prune-amount 0.05 --config yolo8_baseline.yaml`.
  - Квантизация модели (`quantization.py`). Команда: `pipenv run python -m src.models.optimization.quantization --model 1 --config yolo8_baseline.yaml`.
- **Тестирование**: 
  - Предсказания на изображениях и видео через `predict_image.py` и `predict_video.py`. Метрики: mAP, precision, recall, FPS, IoU. Примеры команд: `pipenv run python -m src.models.test.predict_video --model 1 --config yolo8_baseline.yaml`.
- **Имитация на edge-устройствах**: Тестирование преобразованной модели (например, в ONNX) через `edge_imitation.py`. Команда: `pipenv run python -m src.models.test.edge_imitation --model 1 --format onnx --config yolo8_baseline.yaml`.

### Конфигурации
Основная конфигурация: `yolo8_baseline.yaml`. Влияет на гиперпараметры обучения, оптимизации и тестирования. Преобразование модели в форматы (например, ONNX) через `transform_model.py`.

### Оборудование
Эксперименты проводились на GPU/CPU - (8 CPU - AMD Ryzen 7 7435HS; 1 GPU (8 GB) - NVIDIA GEFORCE RTX 4060 с CUDA).

## Обзор результатов

### Метрики производительности
Сравнение базовой (model 1), оптимизированной (model 2) моделей YOLOv8 и (model 3) YOLO11. Метрики включают mAP@0.5, precision, recall, FPS на изображениях/видео.

### Сравнение моделей
| Модель                  | mAP@0.5 | Precision | Recall | F1   |
|-------------------------|---------|-----------|--------|------|
| Базовая (YOLOv8)        | 0.390   | 0.663     | 0.362  | 0.41 |
| Обрезанная (prune 0.05) | 0.399   | 0.689     | 0.361  | 0.42 |
| YOLOv11                 | 0.383   | 0.657     | 0.35   | 0.4  |

(Актуальные значения в ноутбуках [`/notebooks/model_name`](https://github.com/AlexBorealis/Real-Time-Detection/blob/master/notebooks/yolo8/train.ipynb).)

### Визуализации
- `/results/visualizations/model_name/predict`: Детекции с вероятностями (bounding boxes и scores).
- `/results/visualizations/model_name/comparison`: Сравнение предсказанных и ground truth рамок (IoU визуализация).
- `/results/visualizations/model_name/videos`: Обработанные видео с детекциями.

### Ноутбуки
Ноутбуки (.ipynb) с визуализациями и метриками - `/Real-Time-Detection/notebooks/model_name`. Содержат графики обучения (loss curves), confusion matrices и расчеты метрик.

### Выводы
1) При обучении на датасете BDD100K на протяжении 100 эпох, модели YOLO 8 и 11 версий показывают приблизительно равные результаты по метрикам качества.
2) После 50–60 эпох прирост производительности замедляется, что указывает на возможное достижение предела возможностей моделей на данном датасете.
3) На основе анализа матрицы сходства (confusion matrix/confusion matrix normalized), а также учитывая все вышеизложенное, можно сделать вывод о том, для данного датасета и данной модели достигнуты предельные показатели точности детекции объектов.
4) Для улучшения качества детекции, в частности для объектов overcrowded предлагается увеличить сложность модели (yolov8l, yolov8m) с последующей оптимизацией и дообучением, а также разнообразить аугментации тренировочных изображений.
Также дополнительно желательно использовать дополнительные данные с большим количеством overcrowded сцен с объектами классов person, ryder, так как на основании все той же confusion matrix, можно заметить что объекты этих классов ложно определяются моделью как фон.
5) Также для улучшения качества детекции объектов в overcrowded сценах, предлагается производить онлайн-обучение, с полуавтоматической разметкой изображений с помощью обученной модели в процессе реальной работы.
6) Модель успешно была преобразована в формат .onnx и протестирована на CPU. По результатам тестирования после оптимизации, модель показала прирост в скорости обработки видео и изображений на GPU в 3-5 раз, на CPU в 1.5-2 раза. 

## Данные
[BDD100K](http://bdd-data.berkeley.edu/download.html)

1) Изображения для тренировки, валидации и тестирования моделей; скачивается вручную, по верхней ссылке и там выбираются эти объекты - 100k images, Labels
2) Видео для тестирования; также скачиваются вручную - BDD-Attention button

файл конфигурации датасета - `/Real-Time-Detection/config/datasets/dataset.yaml`

файлы с первоначальными конфигурациями `/Real-Time-Detection/config/models/config.yaml`

## Запуск скриптов
## Для запуска справки по командам `pipenv run python -m command --help`
1) Запуск пред обработки данных (preprocess.py) -
    ```
   cd /project/path
   pipenv run python -m src.data.preprocess
   ```

2) Запуска основного цикла обучения (train.py) - 
    ```
   cd /project/path
   pipenv run python -m src.models.train.train
   ```

3) Запуск создания предсказаний (predict_image.py/predict_video.py) - 
    ```
   cd /project/path
   pipenv run python -m src.models.test.predict_video --model 1 --config yolo8_baseline.yaml #(base model)
   pipenv run python -m src.models.test.predict_video --model 2 --config yolo8_baseline.yaml #(optimized model)
   pipenv run python -m src.models.test.predict_image --model 1 --config yolo8_baseline.yaml --nimage 5 #(base model, nimage - любое число)
   pipenv run python -m src.models.test.predict_image --model 2 --config yolo8_baseline.yaml --nimage 5 #(optimized model - любое число)
   ```

4) Запуск обрезки модели (pruning.py) -
    ```
   cd /project/path
   pipenv run python -m src.models.optimization.pruning --model 1 --prune-amount 0.1 --config yolo8_baseline.yaml
    ```
   
5) Запуск квантизации модели (quantization.py) -
    ```
   cd /project/path
   pipenv run python -m src.models.optimization.quantization --model 1 --config yolo8_baseline.yaml
    ```
   
6) Запуск преобразования модели в нужный формат (transform_model.py) -
    ```
   cd /project/path
   pipenv run python -m src.models.transform_model.transform_model --model 1 --format onnx --config yolo8_baseline.yaml
    ```

7) Запуск имитации модели на edge устройстве (edge_imitation.py) -
    ```
   cd /project/path
   pipenv run python -m src.models.test.edge_imitation --model 1 --format onnx --config yolo8_baseline.yaml
    ```
