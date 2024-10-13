import torch
from super_gradients.training import Trainer, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)


def load_model(num_classes=2, pretrained=True):
    if pretrained:
        model = models.get("yolo_nas_l", pretrained_weights="coco")
    else:
        model = models.get("yolo_nas_l", num_classes=num_classes)
    model.prep_model_for_conversion(input_size=(640, 640))
    return model


def train_model(train_loader, val_loader, num_classes=2, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.get(
        "yolo_nas_l", num_classes=num_classes, pretrained_weights="coco"
    ).to(device)

    trainer = Trainer(
        experiment_name="aircraft_detection", ckpt_root_dir="models"
    )

    post_prediction_callback = PPYoloEPostPredictionCallback(
        score_threshold=0.1,
        nms_threshold=0.6,
        nms_top_k=1000,
        max_predictions=300,
    )

    trainer.train(
        model=model,
        training_params={
            "max_epochs": epochs,
            "lr_mode": "StepLRScheduler",
            "lr_updates": [70, 90],
            "lr_decay_factor": 0.1,
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


def save_model(model, path):
    model.save(path)


def load_trained_model(path, num_classes=2):
    model = models.get("yolo_nas_l", num_classes=num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def evaluate_model(model, test_loader, num_classes=2):
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
