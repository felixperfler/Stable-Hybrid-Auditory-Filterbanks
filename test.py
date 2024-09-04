import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm
from pesq import pesq

from src.datasets import Chime2
from src.model import HybridfilterbankModel, FFTModel

def main(args):
    LOGGING_DIR = args.logging_dir
    DATASET = args.dataset
    RESULT_CSV = args.result_csv

    best_pesq = 0
    MODEL_FILE = ''
    best_epoch = 0

    for model_file in os.listdir(LOGGING_DIR+'/models/'):
        pesq_i = torch.load(LOGGING_DIR+'/models/'+model_file, map_location="cpu", weights_only=False)["PESQ"]
        if pesq_i > best_pesq:
            best_pesq = pesq_i
            MODEL_FILE = LOGGING_DIR+'/models/'+model_file
            best_epoch = torch.load(LOGGING_DIR+'/models/'+model_file, map_location="cpu", weights_only=False)["epoch"]

    print(f"Evaluating model at epoch {best_epoch} with a validation PESQ of {best_pesq}.")

    RESULT_CSV += f'results {best_epoch}.csv'
    checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    SIGNAL_LENGTH = checkpoint["signal_length"]
    FFT_INPUT = checkpoint["fft_input"]

    test_dataset = Chime2(
        dataset=DATASET,
        signal_length=SIGNAL_LENGTH,
        fs=16000,
        type="test",
        return_file_id=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    if FFT_INPUT :
        model = FFTModel()
    else:
        model = HybridfilterbankModel()

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    pesq_values = []
    sisdr = []
    noisy_filename = []
    clean_filename = []

    with torch.no_grad():
        for batch in tqdm(test_loader):

            noisy_signal = batch["noisy"]
            target_signal = batch["clean"]
            enhanced_signal = model(noisy_signal)
            target_signal = target_signal[..., : enhanced_signal.shape[-1]]

            pesq_values.append(
                np.mean(
                    pesq(
                        16000,
                        np.array(target_signal[0].detach().numpy()),
                        np.array(enhanced_signal[0].detach().numpy()),
                        "nb",
                    )
                ).item()
            )
            sisdr.append(
                float(
                    ScaleInvariantSignalDistortionRatio()(
                        enhanced_signal.detach(),
                        target_signal.detach(),
                    )
                )
            )

            clean_filename.append(batch["clean_sample_path"])
            noisy_filename.append(batch["noisy_sample_path"])

    export_df = pd.DataFrame(
        list(zip(clean_filename, noisy_filename, pesq_values, sisdr)),
        columns=["Clean File", "Noisy File", "PESQ", "SISDR"],
    )

    print(f"PESQ: {export_df.PESQ.mean()}")
    print(f"SI-SDR: {export_df.SISDR.mean()}")

    export_df.to_csv(RESULT_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--logging_dir",
        type=str,
        required=True,
        help="Path to model file",
    )
    parser.add_argument(
        "--result_csv",
        type=str,
        required=True,
        help="CSV file to write results to.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/chime2_wsj0",
        help="Path to dataset",
    )
    main(parser.parse_args())
