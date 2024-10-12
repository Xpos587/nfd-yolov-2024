import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class VideoLoader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        self.cap.release()


class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        self.img_size = img_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Загрузка и предобработка изображения
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img) / 255.0  # Нормализация
        img = img.transpose(2, 0, 1)  # HWC to CHW

        # Загрузка и обработка меток
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                labels = np.array(
                    [x.split() for x in f.read().splitlines()], dtype=np.float32
                )

        return img, labels


def collate_fn(batch):
    images, targets = [], []
    for img, lbl in batch:
        images.append(torch.tensor(img, dtype=torch.float32))
        if len(lbl) > 0:
            targets.append(torch.tensor(lbl, dtype=torch.float32))
        else:
            targets.append(
                torch.empty((0, 5), dtype=torch.float32)
            )  # Пустая метка
    images = torch.stack(images)
    return images, targets


def load_dataset(data_dir, batch_size=16, img_size=640):
    train_dataset = YOLODataset(
        os.path.join(data_dir, "train", "images"),
        os.path.join(data_dir, "train", "labels"),
        img_size=img_size,
    )
    val_dataset = YOLODataset(
        os.path.join(data_dir, "valid", "images"),
        os.path.join(data_dir, "valid", "labels"),
        img_size=img_size,
    )
    test_dataset = YOLODataset(
        os.path.join(data_dir, "test", "images"),
        os.path.join(data_dir, "test", "labels"),
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
