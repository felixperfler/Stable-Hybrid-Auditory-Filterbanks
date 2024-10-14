import torch

class ComplexCompressedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.c = 0.3
        self.l = 0.3

    def forward(self, enhanced: torch.tensor, clean: torch.tensor):
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

        return (1 - self.l) * mag_compressed_loss + self.l * phasor_loss
