from typing import List, Optional, Tuple

import torch
from super_gradients.training import Trainer, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from torch import nn
from torch.utils.data import DataLoader


def load_model(
    num_classes: int = 2,
    pretrained: bool = True,
    input_size: Tuple[int, int] = (640, 640),
    path: Optional[str] = None,
) -> nn.Module:
    """
    Загрузка модели YOLO NAS с возможностью загрузки предобученных весов или сохраненной модели.

    :param num_classes: Количество классов.
    :param pretrained: Использовать предобученные веса.
    :param input_size: Размер входного изображения.
    :param path: Путь к файлу с сохраненной моделью (если есть).
    :return: Модель YOLO NAS.
    """
    if path:
        # Если указан путь к файлу, загружаем сохраненную модель
        model = models.get("yolo_nas_l", num_classes=num_classes)
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        # Иначе загружаем новую модель с предобученными весами или без них
        if pretrained:
            model = models.get("yolo_nas_l", pretrained_weights="coco")
        else:
            model = models.get("yolo_nas_l", num_classes=num_classes)
        model.prep_model_for_conversion(input_size=input_size)

    return model


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 2,
    epochs: int = 100,
    lr: float = 0.001,
    experiment_name: str = "aircraft_detection",
    ckpt_root_dir: str = "models",
    score_threshold: float = 0.1,
    nms_threshold: float = 0.6,
    nms_top_k: int = 1000,
    max_predictions: int = 300,
    lr_mode: str = "StepLRScheduler",
    lr_updates: List[int] = [70, 90],
    lr_decay_factor: float = 0.1,
) -> nn.Module:
    """
    Обучение модели YOLO NAS.

    :param train_loader: DataLoader для обучающего набора данных.
    :param val_loader: DataLoader для валидационного набора данных.
    :param num_classes: Количество классов.
    :param epochs: Количество эпох обучения.
    :param lr: Начальная скорость обучения.
    :param experiment_name: Имя эксперимента.
    :param ckpt_root_dir: Директория для сохранения контрольных точек.
    :param score_threshold: Порог уверенности для постобработки.
    :param nms_threshold: Порог NMS (Non-Maximum Suppression).
    :param nms_top_k: Максимальное количество предсказаний после NMS.
    :param max_predictions: Максимальное количество предсказаний.
    :param lr_mode: Режим изменения скорости обучения.
    :param lr_updates: Эпохи изменения скорости обучения.
    :param lr_decay_factor: Коэффициент уменьшения скорости обучения.
    :return: Обученная модель YOLO NAS.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.get(
        "yolo_nas_l", num_classes=num_classes, pretrained_weights="coco"
    ).to(device)

    trainer = Trainer(
        experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir
    )

    post_prediction_callback = PPYoloEPostPredictionCallback(
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        nms_top_k=nms_top_k,
        max_predictions=max_predictions,
    )

    trainer.train(
        model=model,
        training_params={
            "max_epochs": epochs,
            "lr_mode": lr_mode,
            "lr_updates": lr_updates,
            "lr_decay_factor": lr_decay_factor,
            "initial_lr": lr,
            "loss": PPYoloELoss(num_classes=num_classes),
            "optimizer": "Adam",
            "optimizer_params": {},
            "train_metrics_list": [
                DetectionMetrics_050(
                    num_cls=num_classes,
                    post_prediction_callback=post_prediction_callback,
                )
            ],
            "valid_metrics_list": [
                DetectionMetrics_050(
                    num_cls=num_classes,
                    post_prediction_callback=post_prediction_callback,
                )
            ],
            "metric_to_watch": "mAP@0.50",
            "device": device,
        },
        train_loader=train_loader,
        valid_loader=val_loader,
    )

    return model


def save_model(model: nn.Module, path: str) -> None:
    """
    Сохранение модели в указанный путь.

    :param model: Модель для сохранения.
    :param path: Путь для сохранения модели.
    """
    model.save(path)


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, num_classes: int = 2
) -> float:
    """
    Оценка модели на тестовом наборе данных с использованием метрики mAP@0.50.

    :param model: Обученная модель YOLO NAS.
    :param test_loader: DataLoader для тестового набора данных.
    :param num_classes: Количество классов в модели.
    :return: Вычисленное значение mAP@0.50.
    """
    model.eval()  # Переключение модели в режим оценки

    post_prediction_callback = PPYoloEPostPredictionCallback(
        score_threshold=0.1,
        nms_threshold=0.6,
        nms_top_k=1000,
        max_predictions=300,
    )

    # Инициализация метрики
    metric = DetectionMetrics_050(
        num_cls=num_classes, post_prediction_callback=post_prediction_callback
    )

    with torch.no_grad():  # Отключение вычисления градиентов
        for images, targets in test_loader:
            images = images.to(model.device)
            outputs = model(images)

            # Обновление метрики
            metric.update(outputs, targets)

    # Вычисление финального значения метрики
    results = metric.compute()

    # Возвращаем значение mAP@0.50
    return results["mAP@0.50"]
