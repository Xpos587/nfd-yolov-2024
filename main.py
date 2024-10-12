import cv2

from preprocessing import process_frame
from src.data_loader import VideoLoader
from src.model import load_model


def main():
    # Загрузка модели
    model = load_model()

    # Инициализация загрузчика видео
    video_loader = VideoLoader("path/to/your/video.mp4")

    # Инициализация видеозаписи
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(
        "output_video.mp4", fourcc, video_loader.fps, video_loader.frame_size
    )

    while True:
        frame = video_loader.next_frame()
        if frame is None:
            break

        # Предобработка кадра
        processed_frame = process_frame(frame)

        # Детекция объектов
        detections = model.predict(processed_frame)

        # Отрисовка результатов
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.25:  # Порог уверенности
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{cls}: {conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

        # Запись обработанного кадра
        out.write(frame)

    video_loader.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
