from typing import Dict, List
import os

import torch
import torchvision
from torch.utils.data import Dataset


class UCF101(Dataset):
    def __init__(self, videos_dir: str, labels_path: List[str]):
        self.videos_dir = videos_dir
        self.labels = []
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.CenterCrop(240),
            torchvision.transforms.ToTensor(),
        ])
        for path in labels_path:
            self._load_labels(path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        video_path =self.labels[idx]["video_path"]
        video = torchvision.io.read_video(video_path, pts_unit="sec")[0].float()

        # [t, n, h, w]
        video = video.permute(0, 3, 1, 2)
        new_video = torch.zeros((video.shape[0], video.shape[1], video.shape[2], video.shape[2]))
        for t in range(video.shape[0]):
            frame = self.transforms(video[t])
            new_video[t] = (frame - 127.5) / 127.5

        return {"video": new_video.cuda(), "class": self.labels[idx]["class"]}

    def _load_labels(self, labels_path: str):
        with open(labels_path) as f:
            lines = iter(f)
            next(lines)

            for line in lines:
                items = line.split(" ")
                video_path = os.path.join(self.videos_dir, items[0].split("/")[-1])
                class_num = int(items[1].split("\n")[0])

                self.labels.append(
                    {
                        "video_path": video_path,
                        "class": torch.as_tensor(class_num),
                    }
                )
