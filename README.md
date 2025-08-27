# Real-Time-Detection

[BDD100K](http://bdd-data.berkeley.edu/download.html)

1) Изображения для тренировки, валидации и тестирования моделей; скачивается вручную, по верхней ссылке и там выбираются эти объекты - 100k images, Labels
2) Видео для тестирования; также скачиваются вручную - BDD-Attention button


# Запуск скриптов
## Для запуска справки по командам ```python3 -m command --help```
1) Запуск пред обработки данных (preprocess.py) -
    ```
   cd /project/path
   python3 -m src.data.preprocess
   ```

2) Запуска основного цикла обучения (train.py) - 
    ```
   cd /project/path
   python3 -m src.models.train.train
   ```

3) Запуск создания предсказаний (predict_image.py/predict_image_video.py) - 
    ```
   cd /project/path
   python3 -m src.models.test.predict_video --model 1 --config yolo8_baseline.yaml #(base model)
   python3 -m src.models.test.predict_video --model 2 --config yolo8_baseline.yaml #(optimized model)
   python3 -m src.models.test.predict_image --model 1 --config yolo8_baseline.yaml --nimage 5 #(base model, nimage - любое число)
   python3 -m src.models.test.predict_image --model 2 --config yolo8_baseline.yaml --nimage 5 #(optimized model - любое число)
   ```

4) Запуск обрезки модели (pruning.py) -
    ```
   cd /project/path
   python3 -m src.models.optimization.pruning --model 1 --prune-amount 0.1 --config yolo8_baseline.yaml
    ```
   
5) Запуск квантизации модели (quantization.py) -
    ```
   cd /project/path
   python3 -m src.models.optimization.quantization --model 1 --config yolo8_baseline.yaml
    ```
   
6) Запуск преобразования модели в нужный формат (transform_model.py) -
    ```
   cd /project/path
   python3 -m src.models.transform_model.transform_model --model 1 --format onnx --config yolo8_baseline.yaml
    ```

7) Запуск имитации модели на edge устройстве (edge_imitation.py) -
    ```
   cd /project/path
   python3 -m src.models.test.edge_imitation --model 1 --format onnx --config yolo8_baseline.yaml
    ```