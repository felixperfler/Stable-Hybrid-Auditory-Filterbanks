import torch

class ComplexCompressedMSELoss(torch.nn.Module):
    def __init__(self, beta: float = 0.0):
        super().__init__()

        self.c = 0.3
        self.l = 0.3
        self.beta = beta

    def forward(self, enhanced: torch.tensor, clean: torch.tensor, w=None):
        enhanced_mag = torch.max(
            torch.abs(enhanced), 1e-8 * torch.ones(enhanced.shape).to(enhanced.device)
        )
        clean_mag = torch.max(
            torch.abs(clean), 1e-8 * torch.ones(clean.shape).to(clean.device)
        )

        enhanced_unit_phasor = torch.div(enhanced, enhanced_mag)
        clean_unit_phasor = torch.div(clean, clean_mag)

        mag_compressed_loss = torch.mean(
            (clean_mag**self.c - enhanced_mag**self.c) ** 2
        )
        phasor_loss = torch.mean(
            (
                torch.abs(
                    clean_mag**self.c * clean_unit_phasor
                    - enhanced_mag**self.c * enhanced_unit_phasor
                )
            )
            ** 2
        )

        loss = (1 - self.l) * mag_compressed_loss + self.l * phasor_loss

        if w is not None:
            w_hat = torch.sum(torch.abs(torch.fft.fft(w, dim=1)) ** 2, dim=0)
            B = torch.max(w_hat, dim=0).values
            A = torch.min(w_hat, dim=0).values

            loss_ = loss +  self.beta * (B / A - 1)

            return loss, loss_
        else:
            return loss, None
