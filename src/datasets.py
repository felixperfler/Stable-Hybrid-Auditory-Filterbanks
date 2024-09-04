import os
import random

import numpy as np
import soundfile
import torch
from scipy import signal
from torch.utils.data import Dataset

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)


class Chime2(Dataset):
    def __init__(
        self,
        dataset: str,
        signal_length: int,
        fs: int,
        type="train",
        return_file_id=False,
    ):
        self.fs = fs
        self.signal_length = signal_length
        self.return_file_id = return_file_id

        if type not in ["train", "test", "dev"]:
            raise ValueError(f"Type must be one of: train, test, dev not {type}")

        if type == "train":
            annotations_folder = dataset + "/data/chime2-wsj0/annotations/train"
        elif type == "test":
            annotations_folder = dataset + "/data/chime2-wsj0/annotations/test5k"
        elif type == "dev":
            annotations_folder = dataset + "/data/chime2-wsj0/annotations/dev5k"

        self.samples = []
        for root, dirs, files in os.walk(annotations_folder):
            for file in files:
                with open(f"{root}/{file}", "r") as reader:
                    for line in reader.readlines():
                        self.samples.append(line.split(":")[0])

        self.path_to_samples_clean = []
        for root, dirs, files in os.walk(dataset + "/data/chime2-wsj0/scaled"):
            for file in files:
                if file.split(".")[0] in self.samples:
                    self.path_to_samples_clean.append(f"{root}/{file}")

        self.path_to_samples_noisy = []
        for root, dirs, files in os.walk(dataset + "/data/chime2-wsj0/isolated"):
            for file in files:
                if file.split(".")[0] in self.samples:
                    self.path_to_samples_noisy.append(f"{root}/{file}")

        if not (len(self.path_to_samples_noisy) == len(self.path_to_samples_clean)):
            raise Exception(
                "Dataset faulty as number of clean and noisy samples are differnet"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        clean_sample_path = list(
            filter(lambda k: sample in k, self.path_to_samples_clean)
        )[0]
        noisy_sample_path = list(
            filter(lambda k: sample in k, self.path_to_samples_noisy)
        )[0]

        clean_sample, _ = soundfile.read(clean_sample_path)
        noisy_sample, _ = soundfile.read(noisy_sample_path)

        clean_sample = np.array(clean_sample[:, 0], dtype=np.float32)
        noisy_sample = np.array(noisy_sample[:, 0], dtype=np.float32)

        while clean_sample.shape[0] < self.fs * self.signal_length:
            clean_sample = np.append(clean_sample, clean_sample)

        while noisy_sample.shape[0] < self.fs * self.signal_length:
            noisy_sample = np.append(noisy_sample, noisy_sample)

        if self.return_file_id:
            return {
                "clean": clean_sample[: self.fs * self.signal_length],
                "noisy": noisy_sample[: self.fs * self.signal_length],
                "clean_sample_path": clean_sample_path,
                "noisy_sample_path": noisy_sample_path,
            }
        else:
            return {
                "clean": clean_sample[: self.fs * self.signal_length],
                "noisy": noisy_sample[: self.fs * self.signal_length],
            }


if __name__ == "__main__":

    data = Chime2(dataset="datasets/chime2_wsj0", signal_length=5, fs=16000)
    #     data = Chime2(dataset="datasets/chime2_wsj0", type="test", signal_length=5, fs=16000)
    #     data = Chime2(dataset="datasets/chime2_wsj0", type="dev", signal_length=5, fs=16000)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    for sample in dataloader:

        #         soundfile.write("clean.wav", sample["clean"][0], 16000)
        #         soundfile.write("noisy.wav", sample["noisy"][0], 16000)

        print("lala")

#         break
