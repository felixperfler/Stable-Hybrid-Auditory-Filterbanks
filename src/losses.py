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
            if len(w) == 1:
                w_hat = torch.sum(torch.abs(torch.fft.fft(w[0], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                ahh = self.beta * (B / A - 1)
            elif len(w) == 2:
                w_hat = torch.sum(torch.abs(torch.fft.fft(w[0], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                first = self.beta * (B / A - 1)

                w_hat = torch.sum(torch.abs(torch.fft.fft(w[1], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                second = self.beta * (B / A - 1)

                ahh = (first + second) / 2

            loss_ = loss + ahh

            return loss, loss_
        else:
            return loss, None


class SNRLoss(torch.nn.Module):

    def __init__(self, eps=1e-8, beta: float = 0.0):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.beta = beta
        self.zero_mean = True

    def forward(self, preds, target, w=None):

        if self.zero_mean:
            target = target - torch.mean(target, dim=-1, keepdim=True)
            preds = preds - torch.mean(preds, dim=-1, keepdim=True)

        noise = target - preds

        val = (torch.sum(target**2, dim=-1) + self.eps) / (
            torch.sum(noise**2, dim=-1) + self.eps
        )
        loss = -torch.mean(10 * torch.log10(val))

        if w is not None:
            if len(w) == 1:
                w_hat = torch.sum(torch.abs(torch.fft.fft(w[0], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                ahh = self.beta * (B / A - 1)
            elif len(w) == 2:
                w_hat = torch.sum(torch.abs(torch.fft.fft(w[0], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                first = self.beta * (B / A - 1)

                w_hat = torch.sum(torch.abs(torch.fft.fft(w[1], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                second = self.beta * (B / A - 1)

                ahh = (first + second) / 2

            loss_ = loss + ahh

            return loss, loss_
        else:
            return loss, None


class SISDRLoss(torch.nn.Module):

    def __init__(self, beta: float = 0.0):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.beta = beta
        self.zero_mean = True

    def forward(self, preds, target, w=None):

        if self.zero_mean:
            target = target - torch.mean(target, dim=-1, keepdim=True)
            preds = preds - torch.mean(preds, dim=-1, keepdim=True)

        alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + self.eps) / (
            torch.sum(target**2, dim=-1, keepdim=True) + self.eps
        )
        target_scaled = alpha * target

        noise = target_scaled - preds

        val = (torch.sum(target_scaled**2, dim=-1) + self.eps) / (
            torch.sum(noise**2, dim=-1) + self.eps
        )
        loss = -torch.mean(10 * torch.log10(val))

        if w is not None:
            if len(w) == 1:
                w_hat = torch.sum(torch.abs(torch.fft.fft(w[0], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                ahh = self.beta * (B / A - 1)
            elif len(w) == 2:
                w_hat = torch.sum(torch.abs(torch.fft.fft(w[0], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                first = self.beta * (B / A - 1)

                w_hat = torch.sum(torch.abs(torch.fft.fft(w[1], dim=1)) ** 2, dim=0)
                B = torch.max(w_hat, dim=0).values
                A = torch.min(w_hat, dim=0).values

                second = self.beta * (B / A - 1)

                ahh = (first + second) / 2

            loss_ = loss + ahh

            return loss, loss_
        else:
            return loss, None
