import numpy as np
import torch

def calculate_condition_number(w) -> torch.Tensor:
    """""
    Calculate the condition number of a convolution operator via the Littlewood Payley sum.
    Input: w (torch.tensor) - matrix with filter impulse respones in the columns
    Output: kappa (torch.tensor) - condition number of the convolution operator
    """""
    w_hat: torch.Tensor = torch.sum(torch.abs(torch.fft.fft(w, dim=1)) ** 2, dim=0)
    B: torch.Tensor = torch.max(w_hat, dim=0).values
    A: torch.Tensor = torch.min(w_hat, dim=0).values
    kappa: torch.Tensor = B/A
    return kappa

def audfilters(n_filters, filter_length, hop_length, frequency_scale, sr):
    """""
    Constructs a set of filters *g* that are equidistantly spaced on a perceptual frequency scale (see |freqtoaud|) between 0 and the Nyquist frequency.
    The filter bandwidths are proportional to the  critical bandwidth of the auditory filters |audfiltbw|.
    The filters are intended to work with signals with a sampling rate of *fs*.
    The signal length *Ls* is mandatory, since we need to avoid too narrow frequency windows.
    """""
    g = np.zeros((n_filters, filter_length), dtype=np.complex64)
    return g

def smooth_fir(frequency_responses, support):
    """""
    Takes a matrix of frequency responses (as columns) and constructs a smoothed FIR version with support length *support*.
    """""
    g = np.exp(-np.pi * np.arange(-support//2,support//2)**2 / ((support-12)/2)**2)
    supper = g.reshape(1,-1)
    gi = np.fft.ifft(frequency_responses, axis=0)
    gi = np.roll(gi, support//2, axis=0)
    g_re = np.real(gi[:support]).T * supper
    g_im = np.imag(gi[:support]).T * supper
    return g_re + 1j * g_im

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

def frame_bounds_lp(w, freq=False):
    # if the filters are given already as frequency responses
    if freq:
        w_hat = np.sum(np.abs(w) ** 2, axis=1)
    else:
        w_hat = np.sum(np.abs(np.fft.fft(w, axis=1)) ** 2, axis=0)
    B = np.max(w_hat)
    A = np.min(w_hat)

    return A, B

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