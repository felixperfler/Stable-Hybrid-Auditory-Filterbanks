# Stable Hybrid Auditory Filterbanks

![Filters](filters/filters.png "Filters")
> **Hold me Tight: Trainable and stable hybrid auditory filterbanks for speech enhancement**\
> Daniel Haider*, Felix Perfler*, Vincent Lostanlen, Martin Ehler, Peter Balazs\
> **Equal contribution*\
> Paper: https://arxiv.org/abs/2408.17358

## About

Convolutional layers with 1-D filters are often used as frontend to encode audio signals. Unlike fixed time—frequency representations, they can adapt to the local characteristics of input data.
However, 1-D filters on raw audio are hard to train and often suffer from instabilities.
In this paper, we address these problems with hybrid solutions, i.e., combining theory-driven and data-driven approaches. 
First, we preprocess the audio signals via a auditory filterbank, guaranteeing good frequency localization for the learned encoder.
Second, we use results from frame theory to define an unsupervised learning objective that encourages energy conservation and perfect reconstruction. Third, we adapt mixed compressed spectral norms as learning objectives to the encoder coefficients. 
Using these solutions in a low-complexity encoder—mask—decoder model significantly improves the perceptual evaluation of speech quality (PESQ) in speech enhancement.

| Encoder              | Params | Objective              | $\kappa$-penalization  | PESQ | SI-SDR | $\kappa$  |
| -------------------- | ------ | ---------------------- | ---------------------- | ---- | ------ | --------- |
| STFT (baseline)      | 0      | $MCS$                  | ❌                     | 3.19 | 9.85   | 2         |
| audlet (ours)        | 0      | $MCS$                  | ❌                     | 3.23 | 9.58   | 1         |
| conv1D               | 8.1k   | $MCS$                  | ❌                     | 2.66 | 11.69  | 3.2       |
| conv1D               | 8.1k   | $MCS_{\beta}$          | ✅                     | 2.77 | 11.99  | 1         |
| hybrid audlet (ours) | 2.8k   | $MCS$                  | ❌                     | 3.38 | 8.86   | 1.2       |
| hybrid audlet (ours) | 2.8k   | $MCS_{\beta}$          | ✅                     | 3.39 | 8.68   | 1         |

## Usage

### Installation

Install the necessary packages with:
```
$ pip install -r requirements.txt
```
### Training

The training script can be started with:
```
$ python main.py
```
The script takes various input arguments with most of them offering default sensible parameter choises. The argument `--dataset` has to be provided and is the path to the dataset as a string. The dataset used was the [CHiME-2 dataset](https://www.chimechallenge.org/challenges/chime2/index). For all other parameters please refer to the help or code.

### Testing

The test script can be executed using:
```
$ python test.py
```
The required arguments are `--dataset`, `--model_file` representing a string to the trained model weight file, and `--result_csv`, which is the file where the results are written to. 

## Citation

If you find our work valuable, please cite

```
@article{HaiderTight2024,
  title={Hold me Tight: Trainable and stable hybrid auditory filterbanks for speech enhancement},
  author={Haider, Daniel and Perfler, Felix and Lostanlen, Vincent and Ehler, Martin and Balazs, Peter},
  journal={arXiv preprint arXiv:2408.17358},
  year={2024}
}
```
