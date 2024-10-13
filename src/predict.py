import argparse
from pathlib import Path

import cv2
import torch

from src.model import load_model


def predict_video(
    video_path: str, output_dir: str, model_path: str, num_classes: int
):
    """
    Выполняет предсказания объектов на видео с использованием обученной модели YOLO NAS.

    :param video_path: Путь к входному видеофайлу.
    :param output_dir: Директория для сохранения выходного видео с предсказаниями.
    :param model_path: Путь к файлу с обученной моделью.
    :param num_classes: Количество классов для детекции.
    """
    # Загрузка модели из указанного пути
    model = load_model(num_classes=num_classes, path=model_path)
    model.eval()  # Переключение модели в режим оценки

    # Открытие видеофайла для чтения
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Настройка параметров для сохранения обработанного видео
    output_path = Path(output_dir) / f"predicted_{Path(video_path).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в тензор для модели
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        # Выполнение предсказания
        with torch.no_grad():
            predictions = model(img)

        # Обработка предсказаний и отрисовка боксов на кадре
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred[:6]
            if conf > 0.5:  # Порог уверенности
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                label = f"{cls}: {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Запись обработанного кадра в выходное видео
        out.write(frame)

    cap.release()
    out.release()
    print(f"Predicted video saved to {output_path}")


def predict_image(
    image_path: str, output_dir: str, model_path: str, num_classes: int
):
    """
    Выполняет предсказания объектов на изображении с использованием обученной модели YOLO NAS.

    :param image_path: Путь к входному изображению.
    :param output_dir: Директория для сохранения выходного изображения с предсказаниями.
    :param model_path: Путь к файлу с обученной моделью.
    :param num_classes: Количество классов для детекции.
    """
    # Загрузка модели из указанного пути
    model = load_model(num_classes=num_classes, path=model_path)
    model.eval()  # Переключение модели в режим оценки

    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image file {image_path}")
        return

    # Преобразование изображения в тензор для модели
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = (
        torch.tensor(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    )

    # Выполнение предсказания
    with torch.no_grad():
        predictions = model(img_tensor)

    # Обработка предсказаний и отрисовка боксов на изображении
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > 0.5:  # Порог уверенности
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            label = f"{cls}: {conf:.2f}"
            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Сохранение обработанного изображения
    output_path = Path(output_dir) / f"predicted_{Path(image_path).name}"
    cv2.imwrite(str(output_path), img)

    print(f"Predicted image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict objects in a video or image using a trained YOLO NAS model."
    )

    parser.add_argument(
        "--video-path",
        type=str,
        help="Path to the input video file.",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to the input image file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the output video or image.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes for detection.",
    )

    args = parser.parse_args()

    if args.video_path:
        predict_video(
            args.video_path, args.output_dir, args.model_path, args.num_classes
        )

    if args.image_path:
        predict_image(
            args.image_path, args.output_dir, args.model_path, args.num_classes
        )
