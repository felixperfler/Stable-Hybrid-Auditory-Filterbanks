import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from pesq import pesq_batch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from hybra import HybrALoss

from src.datasets import Chime2
from src.losses import ComplexCompressedMSELoss
from src.model import HybridfilterbankModel

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
    FS = args.fs
    LOGGING_DIR = args.logging_dir
    DATASET = args.dataset
    SIGNAL_LENGTH = args.signal_length
    KAPPA_BETA = args.kappa_beta
    LEARNING_RATE = args.learning_rate

    print(device := torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = HybridfilterbankModel()

    print(
        "Number of model parameters: ",
        sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        ),
    )

    model = model.to(device)

    # init tensorboard
    writer = SummaryWriter(f"{LOGGING_DIR}")

    loss_fn = HybrALoss(ComplexCompressedMSELoss(), 1, 0, 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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

    epoch = 0

    while epoch < EPOCHS:
        running_loss = 0
        model.train()
        with tqdm(dataloader_train, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                noisy_signal = batch["noisy"].to(device)
                target_signal = batch["clean"].to(device)

                enhanced_signal_coefficients = model(noisy_signal)

                target_signal_coefficients = model.filterbank.encoder(target_signal)

                loss, loss_ = loss_fn(
                    enhanced_signal_coefficients, target_signal_coefficients,
                    model.filterbank
                )

                running_loss += loss.item()
                
                optimizer.zero_grad()
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

                        enhanced_signal_coefficients = model(noisy_signal)
                        enhanced_signal = model.filterbank.decoder(enhanced_signal_coefficients)

                        target_signal_coefficients = model.filterbank.encoder(target_signal)

                        loss, loss_ = loss_fn(
                            enhanced_signal_coefficients,
                            target_signal_coefficients,
                            model.filterbank,
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
                            loss=loss.item(), PESQ=pesq_loop, SISDR=sisdr_loop.item()
                        )

            writer.add_audio(
                "Prediction Val", enhanced_signal[0], epoch, sample_rate=FS
            )
            writer.add_audio("Target Val", target_signal[0], epoch, sample_rate=FS)
            writer.add_audio("Noisy Val", noisy_signal[0], epoch, sample_rate=FS)

            writer.add_scalar("PESQ Val", pesq / len(dataloader_val), epoch)
            writer.add_scalar("SISDR Val", sisdr / len(dataloader_val), epoch)

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

            if not os.path.exists(f"{LOGGING_DIR}/models"):
                os.makedirs(f"{LOGGING_DIR}/models")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": running_loss / len(dataloader_train),
                    "val_loss": running_val_loss / len(dataloader_val),
                    "PESQ": pesq / len(dataloader_val),
                    "SISDR":  sisdr / len(dataloader_val),
                    "fs": FS,
                    "signal_length": SIGNAL_LENGTH,
                },
                f"{LOGGING_DIR}/models/model_{epoch}.pth",
            )
        else:
            writer.add_scalars(
                "Loss",
                {
                    "Train": running_loss / len(dataloader_train),
                },
                epoch,
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
        default="chime2_wsj0",
        help="Path to dataset",
    )
    parser.add_argument(
        "--signal_length",
        type=int,
        default=5,
        help="Signal length in seconds (default: 5)",
    )
    parser.add_argument(
        "--kappa_beta",
        type=float,
        default=None,
        help="Kappa Beta (if None, no kappa beta loss is used, default:  None)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate of the optimizer.",
    )

    main(parser.parse_args())
