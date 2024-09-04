import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pesq import pesq

from src.datasets import Chime2
from src.models import GRUModel
from torch.utils.data import DataLoader
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)


def main(args):
    MODEL_FILE = args.model_file
    DATASET = args.dataset
    SIGNAL_LENGTH = args.signal_length
    USE_FFT_INPUT = args.use_fft_input
    NUM_FILTERS = args.num_filters
    FILTER_LEN = args.filter_len
    STRIDE = args.stride
    RESULT_CSV = args.result_csv
    AUDITORY_BIAS_INIT = args.auditory_bias_init
    AUDITORY_INIT = args.auditory_init
    FS = 16000
    SMALLER_MODEL = args.smaller_model
    DEVICE = torch.device("cpu")

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

    if AUDITORY_BIAS_INIT or AUDITORY_INIT:
        if SMALLER_MODEL:
            with open("filters/auditory_filters_256_short.npy", "rb") as f:
                auditory_filters = torch.tensor(np.load(f), dtype=torch.complex64)
        else:
            with open("filters/auditory_filters_256.npy", "rb") as f:
                auditory_filters = torch.tensor(np.load(f), dtype=torch.complex64)
    else:
        auditory_filters = None

    model = GRUModel(
        use_encoder_filterbank=not USE_FFT_INPUT,
        num_filters=NUM_FILTERS,
        filter_len=FILTER_LEN,
        stride=STRIDE,
        signal_length=SIGNAL_LENGTH,
        fs=FS,
        apply_fir_tightener3000=False,
        auditory_bias_init=AUDITORY_BIAS_INIT,
        auditory_init=AUDITORY_INIT,
        auditory_filters=auditory_filters,
        fft_input=USE_FFT_INPUT,
        device=torch.device("cpu"),
        smaller_model=SMALLER_MODEL,
        random_dual_encoder=False,
    )

    checkpoint = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]

    model = model.to(DEVICE)

    model.eval()

    pesq_values = []
    sisdr = []
    noisy_filename = []
    clean_filename = []
    

    with torch.no_grad():
        for batch in tqdm(test_loader):

            noisy_signal = batch["noisy"].to(DEVICE)
            target_signal = batch["clean"].to(DEVICE)
            enhanced_signal, enhanced_signal_fft = model(noisy_signal)
            target_signal = target_signal[..., : enhanced_signal.shape[-1]]

            pesq_values.append(
                np.mean(
                    pesq(
                        16000,
                        np.array(target_signal[0].cpu().detach().numpy()),
                        np.array(enhanced_signal[0].cpu().detach().numpy()),
                        "nb",
                    )
                ).item()
            )
            sisdr.append(
                float(
                    ScaleInvariantSignalDistortionRatio()(
                        enhanced_signal.cpu().detach(),
                        target_signal.cpu().detach(),
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
        "--model_file",
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
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--signal_length",
        type=int,
        default=5,
        help="Signal length in seconds (default: 5)",
    )
    parser.add_argument(
        "--use_fft_input",
        action=argparse.BooleanOptionalAction,
        help="Use STFT",
    )
    parser.add_argument(
        "--kappa_beta",
        type=float,
        default=None,
        help="Kappa Beta (if None, no kappa beta loss is used, default:  None)",
    )
    parser.add_argument(
        "--auditory_init",
        action=argparse.BooleanOptionalAction,
        help="Use auditory initialization",
    )
    parser.add_argument(
        "--auditory_bias_init",
        action=argparse.BooleanOptionalAction,
        help="Use auditory bias init",
    )
    parser.add_argument(
        "--fir_tightener3000",
        action=argparse.BooleanOptionalAction,
        help="Use FIR tightener 3000",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=257,
        help="Number of filters of the encoder filterbank",
    )
    parser.add_argument(
        "--filter_len",
        type=int,
        default=64,
        help="Length of the filter in the encoder filterbank",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride of the filter in the encoder filterbank",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate of the optimizer.",
    )
    parser.add_argument(
        "--time_domain_loss",
        type=str,
        default=None,
        help="Use time domain loss (None, SNR, SISDR; default: None)",
    )
    parser.add_argument(
        "--compressed_auditory_loss",
        action=argparse.BooleanOptionalAction,
        help="LaLa loss",
    )
    parser.add_argument(
        "--smaller_model",
        action=argparse.BooleanOptionalAction,
        help="Use smaller Model",
    )
    main(parser.parse_args())
