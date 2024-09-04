import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import calculate_condition_number, fir_tightener3000

class HybrA(nn.Module):
    def __init__(self, path_to_auditory_filter_config, start_tight:bool=True):
        super().__init__()

        config = torch.load(path_to_auditory_filter_config, weights_only=False, map_location="cpu")

        self.auditory_filters_real = torch.tensor(config['auditory_filters_real'])
        self.auditory_filters_imag = torch.tensor(config['auditory_filters_imag'])
        self.auditory_filters_stride = config['auditory_filters_stride']
        self.auditory_filter_length = self.auditory_filters_real.shape[-1]
        self.n_filters = config['n_filters']
        self.kernel_size = config['kernel_size']

        self.output_real_forward = None
        self.output_imag_forward = None

        k = torch.tensor(self.n_filters / (self.kernel_size * self.n_filters))
        encoder_weight = (-torch.sqrt(k) - torch.sqrt(k)) * torch.rand([self.n_filters, 1, self.kernel_size]) + torch.sqrt(k)

        if start_tight:
            encoder_weight = torch.tensor(fir_tightener3000(
                encoder_weight.squeeze(1).numpy(), self.kernel_size, eps=1.01
            ),  dtype=torch.float32).unsqueeze(1)
        
        self.encoder_weight_real = nn.Parameter(encoder_weight, requires_grad=True)
        self.encoder_weight_imag = nn.Parameter(encoder_weight, requires_grad=True)

        self.hybra_filters_real = torch.empty(1)
        self.hybra_filters_imag = torch.empty(1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the HybridFilterbank.
        
        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, 1, signal_length)
        
        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, n_filters, signal_length//hop_length)
        """

        kernel_real = F.conv1d(
            self.auditory_filters_real.to(x.device).squeeze(1),
            self.encoder_weight_real,
            groups=self.n_filters,
            padding="same",
        ).unsqueeze(1)
        self.hybra_filters_real = kernel_real.clone().detach()

        kernel_imag = F.conv1d(
            self.auditory_filters_imag.to(x.device).squeeze(1),
            self.encoder_weight_imag,
            groups=self.n_filters,
            padding="same",
        ).unsqueeze(1)
        self.hybra_filters_imag = kernel_imag.clone().detach()
        
        output_real = F.conv1d(
            F.pad(x.unsqueeze(1), (self.auditory_filter_length//2, self.auditory_filter_length//2), mode='circular'),
            kernel_real,
            stride=self.auditory_filters_stride,
        )
        
        output_imag = F.conv1d(
            F.pad(x.unsqueeze(1), (self.auditory_filter_length//2,self.auditory_filter_length//2), mode='circular'),
            kernel_imag,
            stride=self.auditory_filters_stride,
        )

        out = output_real + 1j* output_imag

        return out

    def encoder(self, x:torch.Tensor):
        """For learning use forward method

        """
        out = F.conv1d(
                    F.pad(x.unsqueeze(1),(self.auditory_filter_length//2, self.auditory_filter_length//2), mode='circular'),
                    self.hybra_filters_real.to(x.device),
                    stride=self.auditory_filters_stride,
                ) + 1j * F.conv1d(
                    F.pad(x.unsqueeze(1),(self.auditory_filter_length//2, self.auditory_filter_length//2), mode='circular'),
                    self.hybra_filters_imag.to(x.device),
                    stride=self.auditory_filters_stride,
                )
                
        return out
    
    def decoder(self, x_real:torch.Tensor, x_imag:torch.Tensor) -> torch.Tensor:
        """Forward pass of the dual HybridFilterbank.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, n_filters, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, signal_length)
        """
        x = (
            F.conv_transpose1d(
                x_real,
                self.hybra_filters_real,
                stride=self.auditory_filters_stride,
                padding=self.auditory_filter_length//2,
            )
            + F.conv_transpose1d(
                x_imag,
                self.hybra_filters_imag,
                stride=self.auditory_filters_stride,
                padding=self.auditory_filter_length//2,
            )
        )

        return x.squeeze(1)

    @property
    def condition_number(self):
        # coefficients = self.hybra_filters_real.detach().clone().squeeze(1) + 1j* self.hybra_filters_imag.detach().clone().squeeze(1)
        coefficients = self.encoder_weight_real.squeeze(1) + 1j*self.encoder_weight_imag.squeeze(1)
        return float(calculate_condition_number(coefficients))
