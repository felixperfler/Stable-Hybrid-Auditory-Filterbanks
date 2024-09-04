import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F

# parameters for the different filterbanks

HYPERPARAMS = {
    "CQT": {
        "N": 4092,
        "J": 58,
        "T": 1024,
        "sr": 16000,
        "fmin": 64,
        "fmax": 8000,
        "stride": 12,
        "batch_size": 64,
    },
    "MEL": {
        "N": 4095,
        "J": 42,
        "T": 1024,
        "sr": 16000,
        "fmin": 45,
        "fmax": 8000,
        "stride": 585,
        "batch_size": 64,
    },
    "VQT": {
        "N": 2**12,
        "J": 96,
        "T": 1024,
        "sr": 16000,
        "fmin": 64,
        "fmax": 8000,
        "stride": 512,
        "batch_size": 64,
    },
    "ART": {
        "N": 1024,
        "J": 40,
        "T": 1024,
        "sr": 16000,
        "fmin": 64,
        "fmax": 8000,
        "batch_size": 32,
    },
}

# computes the frame bounds of a filterbank given as a numpy array of row vectors via Littlewood-Payley


def frame_bounds_lp(w, freq=False):
    # if the filters are given already as frequency responses
    if freq:
        w_hat = np.sum(np.abs(w) ** 2, axis=1)
    else:
        w_hat = np.sum(np.abs(np.fft.fft(w, axis=1)) ** 2, axis=0)
    B = np.max(w_hat)
    A = np.min(w_hat)

    return A, B


# creates a random filterbank of J filters of support T, padded with zeros to have length N
# and optionally its tightened version


def random_filterbank(
    N, J, T, norm=True, tight=True, to_torch=True, support_only=False
):
    if T is None:
        T = N
    if norm:
        w = np.random.randn(J, T) / np.sqrt(T) / np.sqrt(J)
    else:
        w = np.random.randn(J, T)
    if support_only:
        w_cat = w
    else:
        w_cat = np.pad(w, ((0, 0), (0, N - T)), constant_values=0)
    if tight:
        W = np.concatenate(
            [sp.linalg.circulant(w_cat[k, :]) for k in range(w_cat.shape[0])]
        )
        S = np.matmul(W.T, W)
        S_sq = np.linalg.inv(sp.linalg.sqrtm(S))
        w_tight = np.matmul(S_sq, w_cat.T).T
        if to_torch:
            return torch.from_numpy(w_cat), torch.from_numpy(w_tight)
        else:
            return w_cat, w_tight
    if to_torch:
        return torch.from_numpy(w_cat)
    else:
        return w_cat


# computes filterbank responses using conv1D (circulant)


def filterbank_response(x, w, mag=True):
    # some shaping
    x = x.reshape(x.shape[0], 1, x.shape[-1])
    x = F.pad(
        x,
        (0, x.shape[-1] - 1),
        mode="circular",
    )
    w = w.unsqueeze(1).float()

    # filtering
    out = F.conv1d(x, w, bias=None, stride=1, padding=0)
    # magnitude
    if mag:
        out = torch.abs(out)
    return out


# computes filterbank responses via FFT


def filterbank_response_fft(x, w, spec):
    # some shaping
    x = x.reshape(x.shape[0], 1, x.shape[-1])
    w = w.unsqueeze(0).float()
    Wx = torch.fft.ifft(torch.fft.fft(x, dim=-1) * w, dim=-1)
    Wx = torch.abs(Wx)
    Wx = Wx[:, :, :: spec["stride"]]
    # hann = torch.hann_window(spec["N"]//spec["stride"]).unsqueeze(0).unsqueeze(0)
    # phi = torch.ones(spec["J"], spec["J"], spec["N"]//spec["stride"])*hann
    # Ux = F.conv1d(Wx, phi, bias=None, stride=1, padding=0)

    Ux = F.avg_pool1d(Wx, kernel_size=spec["N"] // spec["stride"], stride=1)
    return Ux


# generate a dataset of random sine waves


def generate_random_sine(sample_rate, f_min, f_max, length, batch_size):
    time = torch.arange(length).reshape(1, -1) / sample_rate
    log2_min = np.log2(f_min)
    log2_range = np.log2(f_max) - np.log2(f_min)
    while True:
        log2_f0 = log2_min + log2_range * torch.rand(batch_size)
        f0 = (2**log2_f0).reshape(-1, 1)
        yield torch.sin(2 * torch.pi * f0 * time)


# computes the circulant matrix version of w (tensor)


def circulant(w):
    dim = 1
    N = w.shape[dim]
    J = w.shape[0]
    tmp = torch.cat(
        [w.flip((dim,)), torch.narrow(w.flip((dim,)), dim=dim, start=0, length=N - 1)],
        dim=dim,
    )
    tmp = tmp.unfold(dim, N, 1).flip((-1,))
    return tmp.reshape(J * N, N)


# make tight


def tight(w, ver="S"):
    """
    Construction of the canonical tight filterbank
    :param w: analysis filterbank
    :param ver: version of the tight filterbank: 'S' yields canonical tight filterbank, 'flat_spec' yields tight filterbank with flat spectral response
    :return: canonical tight filterbank
    """
    if ver == "S":
        w_freqz = np.fft.fft(w, axis=1)
        lp = np.sum(np.abs(w_freqz) ** 2, axis=0)
        w_freqz_tight = w_freqz * lp ** (-0.5)
        w_tight = np.fft.ifft(w_freqz_tight, axis=1)
    elif ver == "flat_spec":
        M, N = w.shape
        w_freqz = np.fft.fft(w, axis=1).T
        w_tight = np.zeros((M, N), dtype=np.complex64)
        for k in range(N):
            H = w_freqz[k, :]
            U = H / np.linalg.norm(H)
            w_tight[:, k] = np.conj(U)
        w_tight = np.fft.ifft(w_tight.T, axis=0).T
    else:
        raise NotImplementedError
    return w_tight


# make tight with fixed support


def fir_tightener3000(w, supp, eps=1.1, print_kappa=False):
    """
    Iterative construction of a tight filterbank with a given support
    :param w: analysis filterbank
    :param supp: desired support of the tight filterbank
    :param eps: desired precision for kappa = B/A
    :return: tight filterbank
    """
    A, B = frame_bounds_lp(w)
    w_tight = w.copy()
    while B / A > eps:
        w_tight = tight(w_tight)
        w_tight[:, supp:] = 0
        w_tight = np.real(w_tight)
        A, B = frame_bounds_lp(w_tight)
        kappa = B / A
        error = np.linalg.norm(w - w_tight)
        if print_kappa:
            print("kappa:", "%.4f" % kappa, ", error:", "%.4f" % error)
    return w_tight
