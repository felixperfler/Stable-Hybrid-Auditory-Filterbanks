import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pesq import pesq_batch

from src.datasets import Chime2
from src.losses import ComplexCompressedMSELoss, SISDRLoss, SNRLoss
from src.models import GRUModel, TightFilterbankEncoder
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)


def main(args):
    EPOCHS = args.epochs
    VAL_EVERY = args.val_every
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    MODEL_FILE = args.model_file
    FS = args.fs
    LOGGING_DIR = args.logging_dir
    DATASET = args.dataset
    SIGNAL_LENGTH = args.signal_length
    USE_FFT_INPUT = args.use_fft_input
    KAPPA_BETA = args.kappa_beta
    AUDITORY_INIT = args.auditory_init
    AUDITORY_BIAS_INIT = args.auditory_bias_init
    FIR_TIGHTENER3000 = args.fir_tightener3000
    NUM_FILTERS = args.num_filters
    FILTER_LEN = args.filter_len
    STRIDE = args.stride
    LEARNING_RATE = args.learning_rate
    TIME_DOMAIN_LOSS = args.time_domain_loss
    COMPRESSED_AUDITORY_LOSS = args.compressed_auditory_loss
    SMALLER_MODEL = args.smaller_model
    RANDOM_DUAL_ENCODER = args.random_dual_encoder
    RANDOM_DUAL_ENCODER_AUD_INIT = args.random_dual_encoder_aud_init
    AUDITORY_INIT_TRAIN = args.auditory_init_train

    if TIME_DOMAIN_LOSS is None:
        TIME_DOMAIN_LOSS = False
    else:
        TIME_DOMAIN_LOSS = True
        TIME_DOMAIN_LOSS_TYPE = args.time_domain_loss

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if (
        AUDITORY_BIAS_INIT
        or AUDITORY_INIT
        or RANDOM_DUAL_ENCODER_AUD_INIT
        or AUDITORY_INIT_TRAIN
    ):
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
        apply_fir_tightener3000=FIR_TIGHTENER3000,
        auditory_bias_init=AUDITORY_BIAS_INIT,
        auditory_init=AUDITORY_INIT,
        auditory_filters=auditory_filters,
        fft_input=USE_FFT_INPUT,
        device=device,
        smaller_model=SMALLER_MODEL,
        random_dual_encoder=RANDOM_DUAL_ENCODER,
        auditory_init_train=AUDITORY_INIT_TRAIN,
    )

    print(
        "Number of model parameters: ",
        sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        ),
    )

    # init tensorboard
    writer = SummaryWriter(f"{LOGGING_DIR}")

    model = model.to(device)

    #     writer.add_graph(model, TraceWrapper(torch.zeros(1,FS)))
    #     writer.close()

    if MODEL_FILE is not None:
        checkpoint = torch.load(MODEL_FILE, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
    else:
        epoch = 0

    if TIME_DOMAIN_LOSS:
        if KAPPA_BETA is None:
            if TIME_DOMAIN_LOSS_TYPE.lower() == "snr":
                loss_fn = SNRLoss()
            elif TIME_DOMAIN_LOSS_TYPE.lower() == "sisdr":
                loss_fn = SISDRLoss()
            else:
                raise ValueError("Loss function not valid.")
        else:
            if TIME_DOMAIN_LOSS_TYPE.lower() == "snr":
                loss_fn = SNRLoss(beta=KAPPA_BETA)
            elif TIME_DOMAIN_LOSS_TYPE.lower() == "sisdr":
                loss_fn = SISDRLoss(beta=KAPPA_BETA)
            else:
                raise ValueError("Loss function not valid.")
    else:
        if KAPPA_BETA is None:
            loss_fn = ComplexCompressedMSELoss()
        else:
            loss_fn = ComplexCompressedMSELoss(beta=KAPPA_BETA)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if MODEL_FILE is not None:
        checkpoint = torch.load(MODEL_FILE, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    dataset_train = Chime2(
        dataset=DATASET,
        type="train",
        fs=FS,
        signal_length=SIGNAL_LENGTH,
    )
    g = torch.Generator()
    g.manual_seed(0)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=g,
    )

    dataset_val = Chime2(
        dataset=DATASET,
        type="dev",
        fs=FS,
        signal_length=SIGNAL_LENGTH,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        generator=g,
    )

    while epoch < EPOCHS:
        running_loss = 0
        model.train()
        with tqdm(dataloader_train, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                noisy_signal = batch["noisy"].to(device)
                target_signal = batch["clean"].to(device)

                optimizer.zero_grad()

                enhanced_signal, enhanced_signal_fft = model(noisy_signal)
                # enhanced signal and target signal should be same length
                target_signal = target_signal[..., : enhanced_signal.shape[-1]]

                if COMPRESSED_AUDITORY_LOSS:
                    if AUDITORY_BIAS_INIT or AUDITORY_INIT:
                        target_signal_fft = F.conv1d(
                            target_signal.unsqueeze(1),
                            model.filterbank.auditory_filters_real,
                            stride=model.filterbank.auditory_filterbank_stride,
                        ) + 1j * F.conv1d(
                            target_signal.unsqueeze(1),
                            model.filterbank.auditory_filters_imag,
                            stride=model.filterbank.auditory_filterbank_stride,
                        )
                    elif model.filterbank.random_dual_encoder:

                        filters_real = (
                            model.filterbank.encoder_real.weight.detach().clone()
                        )
                        filters_imag = (
                            model.filterbank.encoder_imag.weight.detach().clone()
                        )

                        target_signal_fft = F.conv1d(
                            target_signal.unsqueeze(1),
                            filters_real,
                            stride=model.filterbank.stride,
                        ) + 1j * F.conv1d(
                            target_signal.unsqueeze(1),
                            filters_imag,
                            stride=model.filterbank.stride,
                        )

                    else:
                        enhanced_signal_fft = F.conv1d(
                            enhanced_signal.unsqueeze(1),
                            model.filterbank.encoder.weight.detach().clone(),
                            stride=model.filterbank.stride,
                        )
                        target_signal_fft = F.conv1d(
                            target_signal.unsqueeze(1),
                            model.filterbank.encoder.weight.detach().clone(),
                            stride=model.filterbank.stride,
                        )
                if USE_FFT_INPUT:
                    target_signal_fft = model.specgram(target_signal)

                if KAPPA_BETA is not None:
                    if model.filterbank.random_dual_encoder:
                        filterbank_weights_real = (
                            model.filterbank.encoder_real.weight.squeeze(1)
                        )
                        filterbank_weights_imag = (
                            model.filterbank.encoder_imag.weight.squeeze(1)
                        )

                        filterbank_weights = (
                            filterbank_weights_real,
                            filterbank_weights_imag,
                        )
                    else:
                        filterbank_weights = (
                            model.filterbank.encoder.weight.squeeze(1),
                        )
                else:
                    filterbank_weights = None

                if TIME_DOMAIN_LOSS:
                    loss, loss_ = loss_fn(
                        enhanced_signal, target_signal, filterbank_weights
                    )
                else:
                    loss, loss_ = loss_fn(
                        enhanced_signal_fft, target_signal_fft, filterbank_weights
                    )

                running_loss += loss.item()
                if loss_ is not None:
                    loss_.backward()
                else:
                    loss.backward()

                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        model.eval()
        if epoch % VAL_EVERY == 0:
            running_val_loss = 0
            pesq = 0.0
            sisdr = 0.0

            with torch.no_grad():

                with tqdm(dataloader_val, unit="batch") as tepoch:
                    for batch in tepoch:
                        tepoch.set_description(f"Validation epoch {epoch}")

                        noisy_signal = batch["noisy"].to(device)
                        target_signal = batch["clean"].to(device)

                        enhanced_signal, enhanced_signal_fft = model(noisy_signal)
                        # enhanced signal and target signal should be same length
                        target_signal = target_signal[..., : enhanced_signal.shape[-1]]

                        if COMPRESSED_AUDITORY_LOSS:
                            if AUDITORY_BIAS_INIT or AUDITORY_INIT:
                                target_signal_fft = F.conv1d(
                                    target_signal.unsqueeze(1),
                                    model.filterbank.auditory_filters_real,
                                    stride=model.filterbank.auditory_filterbank_stride,
                                ) + 1j * F.conv1d(
                                    target_signal.unsqueeze(1),
                                    model.filterbank.auditory_filters_imag,
                                    stride=model.filterbank.auditory_filterbank_stride,
                                )
                            elif model.filterbank.random_dual_encoder:

                                filters_real = (
                                    model.filterbank.encoder_real.weight.detach().clone()
                                )
                                filters_imag = (
                                    model.filterbank.encoder_imag.weight.detach().clone()
                                )

                                target_signal_fft = F.conv1d(
                                    target_signal.unsqueeze(1),
                                    filters_real,
                                    stride=model.filterbank.stride,
                                ) + 1j * F.conv1d(
                                    target_signal.unsqueeze(1),
                                    filters_imag,
                                    stride=model.filterbank.stride,
                                )
                            else:
                                enhanced_signal_fft = F.conv1d(
                                    enhanced_signal.unsqueeze(1),
                                    model.filterbank.encoder.weight.detach().clone(),
                                    stride=model.filterbank.stride,
                                )
                                target_signal_fft = F.conv1d(
                                    target_signal.unsqueeze(1),
                                    model.filterbank.encoder.weight.detach().clone(),
                                    stride=model.filterbank.stride,
                                )
                        if USE_FFT_INPUT:
                            target_signal_fft = model.specgram(target_signal)

                        if KAPPA_BETA is not None:
                            if model.filterbank.random_dual_encoder:
                                filterbank_weights_real = (
                                    model.filterbank.encoder_real.weight.squeeze(1)
                                )
                                filterbank_weights_imag = (
                                    model.filterbank.encoder_imag.weight.squeeze(1)
                                )

                                filterbank_weights = (
                                    filterbank_weights_real,
                                    filterbank_weights_imag,
                                )
                            else:
                                filterbank_weights = (
                                    model.filterbank.encoder.weight.squeeze(1),
                                )
                        else:
                            filterbank_weights = None

                        if TIME_DOMAIN_LOSS:
                            loss, loss_ = loss_fn(
                                enhanced_signal, target_signal, filterbank_weights
                            )
                        else:
                            loss, loss_ = loss_fn(
                                enhanced_signal_fft,
                                target_signal_fft,
                                filterbank_weights,
                            )

                        running_val_loss += loss.item()
                        pesq_loop = np.mean(
                            pesq_batch(
                                FS,
                                np.array(target_signal.cpu().detach().numpy()),
                                np.array(enhanced_signal.cpu().detach().numpy()),
                                "nb",
                            )
                        ).item()
                        pesq += pesq_loop
                        sisdr_loop = ScaleInvariantSignalDistortionRatio()(
                            enhanced_signal.cpu().detach(), target_signal.cpu().detach()
                        )
                        sisdr += sisdr_loop

                        tepoch.set_postfix(
                            loss=loss.item(), PESQ=pesq_loop, SISDR=sisdr_loop
                        )

            writer.add_audio(
                "Prediction Val", enhanced_signal[0], epoch, sample_rate=FS
            )
            writer.add_audio("Target Val", target_signal[0], epoch, sample_rate=FS)
            writer.add_audio("Noisy Val", noisy_signal[0], epoch, sample_rate=FS)

            writer.add_scalar("PESQ Val", pesq / len(dataloader_val), epoch)
            writer.add_scalar("SISDR Val", sisdr / len(dataloader_val), epoch)

            if not USE_FFT_INPUT:
                if not AUDITORY_INIT or AUDITORY_BIAS_INIT:
                    if RANDOM_DUAL_ENCODER:
                        cn_real, cn_imag = model.filterbank.condition_number
                        writer.add_scalars(
                            "Condition Number",
                            {
                                "Real": cn_real,
                                "Imag": cn_imag,
                            },
                            epoch,
                        )
                    else:
                        writer.add_scalar(
                            "Condition Number", model.filterbank.condition_number, epoch
                        )

            writer.add_scalars(
                "Loss",
                {
                    "Train": running_loss / len(dataloader_train),
                    "Val": running_val_loss / len(dataloader_val),
                },
                epoch,
            )
        else:
            writer.add_scalars(
                "Loss",
                {
                    "Train": running_loss / len(dataloader_train),
                },
                epoch,
            )

        # check if model save dir exists
        if not os.path.exists(f"{LOGGING_DIR}/models"):
            os.makedirs(f"{LOGGING_DIR}/models")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"{LOGGING_DIR}/models/model_{epoch}.pth",
        )

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=301, help="Number of epochs (default: 301)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers (default: 4)"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Path to model file (default: None)",
    )
    parser.add_argument(
        "--fs", type=int, default=16000, help="Sampling rate (default: 16000)"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=f"{os.getcwd()}/logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Logging directory (default: ./logs/<timestamp>)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/chime2_wsj0",
        help="Path to dataset",
    )
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
        default=256,
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
    parser.add_argument(
        "--random_dual_encoder",
        action=argparse.BooleanOptionalAction,
        help="Use seperate encoders for real and imag",
    )
    parser.add_argument(
        "--random_dual_encoder_aud_init",
        action=argparse.BooleanOptionalAction,
        help="why not",
    )
    parser.add_argument(
        "--auditory_init_train",
        action=argparse.BooleanOptionalAction,
        help="initialize conv1d with audlet",
    )

    main(parser.parse_args())
