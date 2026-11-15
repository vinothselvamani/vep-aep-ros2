"""Signal-processing core shared by the ROS 2 nodes and the offline evaluation.

Everything here is causal (sample-by-sample state is kept) so that the
streaming ROS 2 pipeline and the offline analysis produce identical epochs.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import eigh

EVENT_ID = {"aud/left": 1, "aud/right": 2, "vis/left": 3, "vis/right": 4}
MODALITY = {1: 0, 2: 0, 3: 1, 4: 1}  # 0 = AEP, 1 = VEP


class CausalBandpass:
    """Butterworth band-pass in SOS form with persistent per-channel state."""

    def __init__(self, fs: float, n_ch: int, lo: float = 1.0, hi: float = 30.0, order: int = 4):
        self.sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, n_ch))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (n_samples, n_ch) -> filtered (n_samples, n_ch)."""
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def car(x: np.ndarray) -> np.ndarray:
    """Common average reference over channels; x: (n_samples, n_ch)."""
    return x - x.mean(axis=1, keepdims=True)


class RingBuffer:
    """Fixed-length buffer of the most recent samples with absolute indexing."""

    def __init__(self, n_ch: int, length: int):
        self.buf = np.zeros((length, n_ch))
        self.length = length
        self.n_written = 0  # absolute index of the next sample

    def push(self, x: np.ndarray) -> None:
        n = x.shape[0]
        idx = (self.n_written + np.arange(n)) % self.length
        self.buf[idx] = x
        self.n_written += n

    def get(self, start: int, stop: int) -> np.ndarray | None:
        """Return samples [start, stop) in absolute index, or None if unavailable."""
        if stop > self.n_written or start < self.n_written - self.length:
            return None
        idx = np.arange(start, stop) % self.length
        return self.buf[idx].copy()


def make_epoch(seg: np.ndarray, n_pre: int) -> np.ndarray:
    """Baseline-correct a (n_samples, n_ch) segment, return (n_ch, n_samples)."""
    seg = seg - seg[:n_pre].mean(axis=0, keepdims=True)
    return seg.T


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------
class XdawnLDA:
    """Multi-class xDAWN spatial filtering + shrinkage LDA.

    For every class c the generalised eigenproblem  C_c w = lambda C_x w  is
    solved (C_c: covariance of the class-average evoked response, C_x: covariance
    of single-trial data). The top `n_filters` filters per class are stacked,
    epochs are projected, decimated in time and fed to Ledoit-Wolf LDA.
    """

    def __init__(self, n_filters: int = 3, decim: int = 4, reg: float = 1e-3):
        self.n_filters, self.decim, self.reg = n_filters, decim, reg

    def _features(self, X: np.ndarray) -> np.ndarray:
        Z = np.einsum("fc,ncs->nfs", self.W_, X)
        Z = signal.decimate(Z, self.decim, axis=-1, zero_phase=True) if self.decim > 1 else Z
        return Z.reshape(len(X), -1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.classes_ = np.unique(y)
        Xc = np.concatenate(list(X), axis=1)  # (n_ch, n_trials*n_s)
        Cx = np.cov(Xc)
        Cx += self.reg * np.trace(Cx) / len(Cx) * np.eye(len(Cx))
        filters = []
        for c in self.classes_:
            P = X[y == c].mean(axis=0)
            Cp = np.cov(P)
            evals, evecs = eigh(Cp, Cx)
            filters.append(evecs[:, ::-1][:, : self.n_filters].T)
        self.W_ = np.concatenate(filters, axis=0)
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda_.fit(self._features(X), y)
        return self

    def predict(self, X):
        return self.lda_.predict(self._features(X))

    def decision_function(self, X):
        return self.lda_.decision_function(self._features(X))

    def predict_proba(self, X):
        return self.lda_.predict_proba(self._features(X))


class TemplateMatch:
    """Baseline: correlate each epoch with class-average templates (all channels)."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.T_ = np.stack([X[y == c].mean(0).ravel() for c in self.classes_])
        self.T_ = (self.T_ - self.T_.mean(1, keepdims=True)) / self.T_.std(1, keepdims=True)
        return self

    def decision_function(self, X):
        F = X.reshape(len(X), -1)
        F = (F - F.mean(1, keepdims=True)) / F.std(1, keepdims=True)
        S = F @ self.T_.T / F.shape[1]
        return S[:, 1] - S[:, 0] if len(self.classes_) == 2 else S

    def predict(self, X):
        F = X.reshape(len(X), -1)
        F = (F - F.mean(1, keepdims=True)) / F.std(1, keepdims=True)
        return self.classes_[np.argmax(F @ self.T_.T, axis=1)]


class ChannelLDA:
    """Baseline: shrinkage LDA on decimated waveforms of a few fixed channels."""

    def __init__(self, picks, decim: int = 4):
        self.picks, self.decim = picks, decim

    def _f(self, X):
        Z = X[:, self.picks]
        Z = signal.decimate(Z, self.decim, axis=-1, zero_phase=True)
        return Z.reshape(len(X), -1)

    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(self._f(X), y)
        self.classes_ = self.lda_.classes_
        return self

    def predict(self, X):
        return self.lda_.predict(self._f(X))

    def decision_function(self, X):
        return self.lda_.decision_function(self._f(X))


# --------------------------------------------------------------------------
# Component (peak) measurement on averaged responses
# --------------------------------------------------------------------------
def peak_measure(wave: np.ndarray, times: np.ndarray, win: tuple, polarity: int):
    """Return (latency_s, amplitude) of the extreme of given polarity in window."""
    m = (times >= win[0]) & (times <= win[1])
    seg = wave[m] * polarity
    i = int(np.argmax(seg))
    return float(times[m][i]), float(wave[m][i])


def component_detected(wave, times, win, polarity, k_sd: float = 3.0):
    """Classical criterion: peak of required polarity exceeds k x baseline SD."""
    lat, amp = peak_measure(wave, times, win, polarity)
    sd = wave[times < 0].std()
    return (amp * polarity) > k_sd * sd, lat, amp
