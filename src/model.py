import torch
from super_gradients.training import Trainer, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from super_gradients.training.utils import compute_mAP


def load_model(num_classes=2, pretrained=True):
    if pretrained:
        model = models.get("yolo_nas_l", pretrained_weights="coco")
    else:
        model = models.get("yolo_nas_l", num_classes=num_classes)
    model.prep_model_for_conversion(input_size=(640, 640))
    return model


def train_model(train_loader, val_loader, num_classes=2, epochs=100, lr=0.001):
    model = models.get(
        "yolo_nas_l", num_classes=num_classes, pretrained_weights="coco"
    )

    trainer = Trainer(
        experiment_name="aircraft_detection", ckpt_root_dir="models"
    )

    trainer.train(
        model=model,
        training_params={
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Adam",
            "loss": PPYoloELoss(),
            "train_metrics_list": [
                DetectionMetrics_050(
                    post_prediction_callback=PPYoloEPostPredictionCallback()
                )
            ],
            "valid_metrics_list": [
                DetectionMetrics_050(
                    post_prediction_callback=PPYoloEPostPredictionCallback()
                )
            ],
            "metric_to_watch": "mAP@0.50",
        },
        train_loader=train_loader,
        valid_loader=val_loader,
    )

    return model


def save_model(model, path):
    model.save(path)


def load_trained_model(path, num_classes=2):
    model = models.get("yolo_nas_l", num_classes=num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def evaluate_model(model, test_loader):
    """
    Оценка модели на тестовом наборе данных с использованием метрики mAP@0.50.

    :param model: Обученная модель YOLO NAS.
    :param test_loader: DataLoader для тестового набора данных.
    :return: Вычисленное значение mAP@0.50.
    """
    model.eval()  # Переключение модели в режим оценки
    all_detections = []
    all_targets = []

    with torch.no_grad():  # Отключение вычисления градиентов
        for images, targets in test_loader:
            outputs = model(images)

            # Преобразование выходных данных модели в нужный формат
            detections = outputs["pred"][
                0
            ]  # Предполагается, что это тензор формы (N, 6), где N - количество обнаружений

            # Разделение детекций на боксы, скоры и классы
            boxes = detections[:, :4]
            scores = detections[:, 4]
            labels = detections[:, 5]

            all_detections.append(
                {"boxes": boxes, "scores": scores, "labels": labels}
            )
            all_targets.append(targets)

    # Вычисление mAP@0.50 с использованием функции из super_gradients
    mAP = compute_mAP(all_targets, all_detections, iou_threshold=0.50)
    return mAP
