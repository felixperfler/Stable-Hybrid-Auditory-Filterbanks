import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import InverseSpectrogram, Spectrogram

from src.fb_utils import filterbank_response, fir_tightener3000


def calculate_condition_number(w):
    w_hat = torch.sum(torch.abs(torch.fft.fft(w, dim=1)) ** 2, dim=0)
    B = torch.max(w_hat, dim=0).values
    A = torch.min(w_hat, dim=0).values

    return B / A


class TightFilterbankEncoder(nn.Module):
    def __init__(
        self,
        num_filters=256,
        filter_len=32,
        stride=4,
        signal_length=5,
        fs=16000,
        apply_fir_tightener3000=False,
        auditory_init=False,
        auditory_bias_init=False,
        auditory_filters=None,
        device=torch.device("cpu"),
        random_dual_encoder=False,
        smaller_model=False,
        auditory_init_train=False,
    ):
        super().__init__()

        if auditory_bias_init:
            auditory_init = True

        self.num_filters = num_filters
        self.filter_len = filter_len
        self.stride = stride
        self.auditory_init = auditory_init
        self.auditory_bias_init = auditory_bias_init
        self.auditory_filterbank_stride = 128
        self.random_dual_encoder = random_dual_encoder
        self.auditory_init_train = auditory_init_train

        if auditory_filters is not None:
            self.auditory_filters_real = auditory_filters.real.to(device).unsqueeze(1)
            self.auditory_filters_imag = auditory_filters.imag.to(device).unsqueeze(1)

            if auditory_bias_init or auditory_init_train:
                self.__create_encoder(apply_fir_tightener3000)

        else:
            self.__create_encoder(apply_fir_tightener3000)

    def __create_encoder(self, apply_fir_tightener3000):

        self.encoder = nn.Conv1d(
            self.num_filters if self.auditory_bias_init else 1,
            self.num_filters,
            self.filter_len,
            bias=False,
            stride=self.stride,
            padding="same" if self.auditory_bias_init else 0,
            groups=self.num_filters if self.auditory_bias_init else 1,
        )

        if self.auditory_init_train:
            self.encoder.weight = torch.nn.Parameter(self.auditory_filters_real)

        if apply_fir_tightener3000:
            encoder_filterbank = self.encoder.weight.squeeze(1).detach().numpy()
            encoder_filterbank = fir_tightener3000(
                encoder_filterbank, self.filter_len, eps=1.01
            )
        # else:
        #     self.encoder.weight = torch.nn.Parameter(torch.tensor(
        #             encoder_filterbank[:, : self.filter_len], dtype=torch.float32
        #         ).unsqueeze(1))

        #         print("Initial Condition Number: ", self.condition_number)

        if self.random_dual_encoder:
            self.encoder_real = nn.Conv1d(
                self.num_filters if self.auditory_bias_init else 1,
                self.num_filters,
                self.filter_len,
                bias=False,
                stride=self.stride,
                padding="same" if self.auditory_bias_init else 0,
                groups=self.num_filters if self.auditory_bias_init else 1,
            )

            self.encoder_imag = nn.Conv1d(
                self.num_filters if self.auditory_bias_init else 1,
                self.num_filters,
                self.filter_len,
                bias=False,
                stride=self.stride,
                padding="same" if self.auditory_bias_init else 0,
                groups=self.num_filters if self.auditory_bias_init else 1,
            )

            if self.auditory_bias_init:
                self.encoder_real.weight = torch.nn.Parameter(
                    self.auditory_filters_real[..., : self.filter_len]
                )
                self.encoder_imag.weight = torch.nn.Parameter(
                    self.auditory_filters_imag[..., : self.filter_len]
                )
            else:
                self.encoder_real.weight = torch.nn.Parameter(
                    torch.tensor(
                        encoder_filterbank[:, : self.filter_len], dtype=torch.float32
                    ).unsqueeze(1)
                )
                self.encoder_imag.weight = torch.nn.Parameter(
                    torch.tensor(
                        encoder_filterbank[:, : self.filter_len], dtype=torch.float32
                    ).unsqueeze(1)
                )

            del self.encoder

    def forward(self, input):
        if self.auditory_init:
            output_real = F.conv1d(
                input,
                self.auditory_filters_real,
                stride=self.auditory_filterbank_stride,
                padding=0,
            )
            self.output_real_forward = output_real.clone()
            output_imag = F.conv1d(
                input,
                self.auditory_filters_imag,
                stride=self.auditory_filterbank_stride,
                padding=0,
            )
            self.output_imag_forward = output_imag.clone()
            if self.auditory_bias_init:
                output_real = self.encoder(output_real)
                output_imag = self.encoder(output_imag)

            output = torch.log10(
                torch.max(
                    output_real**2 + output_imag**2, 1e-8 * torch.ones_like(output_real)
                )
            )
        elif self.random_dual_encoder:
            output_real = self.encoder_real(input)
            self.output_real_forward = output_real.clone()
            output_imag = self.encoder_imag(input)
            self.output_imag_forward = output_imag.clone()

            output = torch.log10(
                torch.max(
                    output_real**2 + output_imag**2, 1e-8 * torch.ones_like(output_real)
                )
            )
        else:
            output = self.encoder(input)

        return output

    @property
    def condition_number(self):
        if self.random_dual_encoder:
            coefficients_real = self.encoder_real.weight.detach().clone().squeeze(1)
            coefficients_imag = self.encoder_imag.weight.detach().clone().squeeze(1)
            return float(calculate_condition_number(coefficients_real)), float(
                calculate_condition_number(coefficients_imag)
            )
        else:
            coefficients = self.encoder.weight.detach().clone().squeeze(1)
            return float(calculate_condition_number(coefficients))

    def decoder(self, input):
        decoder_filterbank = self.encoder.weight.detach().clone()
        output = F.conv_transpose1d(
            input,
            decoder_filterbank,
            stride=self.stride,
        ).squeeze(1)

        return output


class GRUModel(nn.Module):
    def __init__(
        self,
        use_encoder_filterbank=False,
        num_filters=256,
        filter_len=512,
        stride=128,
        signal_length=10,
        fs=16000,
        apply_fir_tightener3000=False,
        auditory_bias_init=False,
        auditory_init=False,
        auditory_filters=None,
        fft_input=False,
        device=torch.device("cpu"),
        smaller_model=False,
        random_dual_encoder=False,
        auditory_init_train=False,
    ):

        self.auditory_init = auditory_init
        self.auditory_bias_init = auditory_bias_init
        self.fft_input = fft_input
        self.fs = fs
        self.signal_length = signal_length
        self.stride = stride
        self.smaller_model = smaller_model

        if self.smaller_model:
            scaling_factor = 1
        else:
            scaling_factor = 1

        super().__init__()

        input_size = num_filters if use_encoder_filterbank else 255

        self.linear_before = nn.Linear(input_size, 400 // scaling_factor)

        self.gru = nn.GRU(
            input_size=400 // scaling_factor,
            hidden_size=400 // scaling_factor,
            num_layers=2,
            batch_first=True,
        )

        self.linear_after = nn.Linear(400 // scaling_factor, 600 // scaling_factor)
        self.linear_after2 = nn.Linear(600 // scaling_factor, 600 // scaling_factor)
        self.linear_after3 = nn.Linear(600 // scaling_factor, input_size)

        self.use_encoder_filterbank = use_encoder_filterbank

        if self.use_encoder_filterbank:
            self.filterbank = TightFilterbankEncoder(
                num_filters=num_filters,
                filter_len=filter_len,
                stride=stride,
                signal_length=signal_length,
                fs=fs,
                apply_fir_tightener3000=apply_fir_tightener3000,
                auditory_bias_init=auditory_bias_init,
                auditory_init=auditory_init,
                auditory_filters=auditory_filters,
                device=device,
                random_dual_encoder=random_dual_encoder,
                auditory_init_train=auditory_init_train,
            )

        elif self.fft_input:
            self.specgram = Spectrogram(
                n_fft=512, win_length=512, hop_length=256, power=None
            ).to(device)
            self.power_specgram = Spectrogram(
                n_fft=512, win_length=512, hop_length=256, power=2
            ).to(device)
            self.inverse_specgram = InverseSpectrogram(
                n_fft=512, win_length=512, hop_length=256
            ).to(device)

    def forward(self, x):
        if self.use_encoder_filterbank:
            filterbank_responses = self.filterbank(x.unsqueeze(1))
            filterbank_responses_copy = filterbank_responses.clone().detach()
            x = torch.log10(
                torch.max(
                    torch.abs(filterbank_responses) ** 2,
                    torch.ones_like(filterbank_responses, dtype=torch.float32) * 1e-6,
                )
            )
        elif self.fft_input:
            x_fft = self.specgram(x)
            x = torch.log10(
                torch.max(
                    self.power_specgram(x),
                    torch.ones(
                        x.shape[0],
                        257,
                        self.fs * self.signal_length // 255,
                        dtype=torch.float32,
                    ).to(x.device)
                    * 1e-6,
                )
            )[:, 1:-1, :]

        x = x.permute(0, 2, 1)
        x = torch.relu(self.linear_before(x))
        x, _ = self.gru(x)
        x = torch.relu(self.linear_after(x))
        x = torch.relu(self.linear_after2(x))
        x = torch.sigmoid(self.linear_after3(x))
        x = x.permute(0, 2, 1)

        if self.use_encoder_filterbank:
            if self.auditory_init:
                x_real = x * self.filterbank.output_real_forward
                x_imag = x * self.filterbank.output_imag_forward
                if self.smaller_model:
                    renorm = 16
                else:
                    renorm = 8
                x = renorm * (
                    F.conv_transpose1d(
                        x_real,
                        self.filterbank.auditory_filters_real,
                        stride=self.filterbank.auditory_filterbank_stride,
                        padding=0,
                    )
                    + F.conv_transpose1d(
                        x_imag,
                        self.filterbank.auditory_filters_imag,
                        stride=self.filterbank.auditory_filterbank_stride,
                        padding=0,
                    )
                )

                return x.squeeze(1), x_real + 1j * x_imag

            elif self.filterbank.random_dual_encoder:
                x_real = x * self.filterbank.output_real_forward
                x_imag = x * self.filterbank.output_imag_forward

                filterbank_real_weights = (
                    self.filterbank.encoder_real.weight.detach().clone()
                )
                filterbank_imag_weights = (
                    self.filterbank.encoder_imag.weight.detach().clone()
                )

                x = (self.filterbank.stride / 2) * (
                    F.conv_transpose1d(
                        x_real, filterbank_real_weights, stride=self.filterbank.stride
                    )
                    + F.conv_transpose1d(
                        x_imag, filterbank_imag_weights, stride=self.filterbank.stride
                    )
                )

                return x.squeeze(1), x_real + 1j * x_imag

            else:
                x = x * filterbank_responses_copy
                x = self.filterbank.decoder(x)
                return x, None
        elif self.fft_input:
            x = (
                torch.cat(
                    [
                        torch.ones(x.shape[0], 1, x.shape[2]).to(x.device),
                        x,
                        torch.ones(x.shape[0], 1, x.shape[2]).to(x.device),
                    ],
                    axis=1,
                )
                * x_fft
            )

            return self.inverse_specgram(x), x
