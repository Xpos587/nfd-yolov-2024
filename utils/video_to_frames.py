import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, frame_interval=1, max_frames=None):
    """
    Извлекает кадры из видео и сохраняет их в указанную директорию.

    :param video_path: Путь к видеофайлу
    :param output_dir: Директория для сохранения кадров
    :param frame_interval: Интервал между сохраняемыми кадрами
    :param max_frames: Максимальное количество кадров для извлечения (None для всех кадров)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = int(video.get(cv2.CAP_PROP_FPS))

    if max_frames is None:
        max_frames = total_frames

    frame_count = 0
    saved_count = 0

    for _ in tqdm(
        range(min(total_frames, max_frames)),
        desc=f"Обработка {os.path.basename(video_path)}",
    ):
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_dir, f"frame_{saved_count:06d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    return saved_count


def process_video_directory(
    input_dir, output_dir, frame_interval=1, max_frames=None, num_workers=4
):
    """
    Обрабатывает все видеофайлы в указанной директории, извлекая из них кадры.

    :param input_dir: Директория с видеофайлами
    :param output_dir: Директория для сохранения кадров
    :param frame_interval: Интервал между сохраняемыми кадрами
    :param max_frames: Максимальное количество кадров для извлечения из каждого видео
    :param num_workers: Количество параллельных процессов
    """
    video_files = [
        f for f in os.listdir(input_dir) if f.endswith((".mp4", ".avi", ".mov"))
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for video_file in video_files:
            video_path = os.path.join(input_dir, video_file)
            video_output_dir = os.path.join(
                output_dir, os.path.splitext(video_file)[0]
            )
            futures.append(
                executor.submit(
                    extract_frames,
                    video_path,
                    video_output_dir,
                    frame_interval,
                    max_frames,
                )
            )

        total_frames = 0
        for future in as_completed(futures):
            total_frames += future.result()

    print(f"Всего извлечено {total_frames} кадров из {len(video_files)} видео.")
