import os
from typing import Dict, List

import torch
import torchvision
from torch.utils.data import Dataset


class UCF101(Dataset):
    def __init__(
        self,
        videos_dir: str,
        class_ind_path: str,
        labels_path: List[str],
        n_frames: int,
        image_size: int,
    ):
        self.videos_dir = videos_dir
        self.n_frames = n_frames
        self.image_size = image_size
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomResizedCrop(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1)),
            ]
        )

        self.class_ind = {}
        self.labels = []
        self._load_class_ind(class_ind_path)
        for path in labels_path:
            self._load_labels(path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        # return 0  # debug

        video_path = self.labels[idx]["video_path"]
        video = torchvision.io.read_video(video_path, pts_unit="sec")[0].float()

        # [t, n, h, w]
        interval = video.shape[0] // self.n_frames
        video = video.permute(0, 3, 1, 2)
        new_video = torch.zeros(
            (self.n_frames, video.shape[1], self.image_size, self.image_size)
        )
        for t in range(0, self.n_frames):
            frame = self.transforms(video[t * interval])
            new_video[t] = frame

        return {"video": new_video, "class": self.labels[idx]["class"] - 1}

    def _load_class_ind(self, class_ind_path: str):
        with open(class_ind_path) as f:
            lines = iter(f)

            for line in lines:
                items = line.split(" ")
                index = items[0]
                class_name = items[1].split("\n")[0]
                self.class_ind[class_name] = int(index)

    def _load_labels(self, labels_path: str):
        with open(labels_path) as f:
            lines = iter(f)

            for line in lines:
                items = line.split(" ")
                video_path = os.path.join(self.videos_dir, items[0].split("/")[-1])
                class_num = self.class_ind[items[0].split("/")[0]]

                self.labels.append(
                    {
                        "video_path": video_path,
                        "class": torch.as_tensor(class_num),
                    }
                )
