from typing import Dict, List

import torchvision
from torch.utils.data import Dataset


class UCF101(Dataset):
    def __init__(self, videos_dir: str, labels_path: List[str]):
        self.videos_dir = videos_dir
        self.labels = []
        for path in labels_path:
            self._load_labels(path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        return {"video": self.labels[idx]["video"], "class": self.labels[idx]["class"]}

    def _load_labels(self, labels_path: str):
        with open(labels_path) as f:
            lines = iter(f)
            next(lines)

            for line in lines:
                items = line.split(" ")
                video_path = items[0].split("/")[-1]
                video = torchvision.io.read_video(video_path, pts_unit="sec")[0]
                class_num = items[1]

                self.labels.append(
                    {
                        "video": video,
                        "class": class_num,
                    }
                )
