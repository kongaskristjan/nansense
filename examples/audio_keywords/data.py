"""Mini Speech Commands data loading: WAV files -> log-mel spectrograms.

The dataset is Google's "mini Speech Commands" set (8 keywords, ~8000 mono
16 kHz ~1 s clips). It is fetched once with torchvision's archive downloader and
read off disk with `scipy.io.wavfile`. Every audio operation — framing, the STFT,
the mel filterbank, the log compression — lives in the transform, so the model
only ever sees a fixed `[1, n_mels, n_frames]` log-mel tensor and stays a plain
2D CNN over an "image".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_and_extract_archive

DATA_URL: str = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
EXTRACT_DIRNAME: str = "mini_speech_commands"

# The eight keyword classes, sorted so the label index is deterministic.
KEYWORDS: tuple[str, ...] = ("down", "go", "left", "no", "right", "stop", "up", "yes")


@dataclass(frozen=True)
class AudioConfig:
    """Static description of the audio task and the log-mel front end.

    `mean` / `std` are scalar normalization constants for the log-mel features,
    passed to `nansense.start` as 1-tuples so the spectrogram renders
    denormalized in the UI. They are deliberately approximate — the log-mel of
    near-silent padding dominates and pulls the global mean low.
    """

    num_classes: int = len(KEYWORDS)
    in_channels: int = 1
    sample_rate: int = 16_000
    clip_length: int = 16_000  # 1 s, pad/truncate target
    n_fft: int = 400  # 25 ms window at 16 kHz
    hop_length: int = 160  # 10 ms hop
    n_mels: int = 40
    f_min: float = 20.0
    f_max: float = 8_000.0  # Nyquist at 16 kHz
    log_eps: float = 1e-6
    # Approximate global log-mel statistics (see class docstring).
    mean: tuple[float, ...] = (-6.0,)
    std: tuple[float, ...] = (3.0,)


def _hz_to_mel(hz: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: Tensor) -> Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(config: AudioConfig) -> Tensor:
    """A `[n_mels, n_fft // 2 + 1]` triangular mel filterbank matrix.

    Triangles are spaced evenly on the mel scale between `f_min` and `f_max` and
    interpolated back onto the linear STFT frequency bins — the standard
    Slaney-style construction, built from scratch with torch ops.
    """
    n_freqs = config.n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, config.sample_rate / 2.0, n_freqs)

    mel_min = _hz_to_mel(torch.tensor(config.f_min))
    mel_max = _hz_to_mel(torch.tensor(config.f_max))
    # n_mels + 2 points define n_mels overlapping triangles (lower / peak / upper).
    mel_points = torch.linspace(float(mel_min), float(mel_max), config.n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    filterbank = torch.zeros(config.n_mels, n_freqs)
    for m in range(config.n_mels):
        lower, center, upper = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        left = (fft_freqs - lower) / (center - lower)
        right = (upper - fft_freqs) / (upper - center)
        filterbank[m] = torch.clamp(torch.minimum(left, right), min=0.0)
    return filterbank


class LogMelTransform:
    """Waveform `[clip_length]` -> log-mel spectrogram `[1, n_mels, n_frames]`.

    Pure torch: `torch.stft` -> power -> mel projection -> `log(x + eps)`. The
    filterbank is precomputed once and reused for every clip.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.window = torch.hann_window(config.n_fft)
        self.filterbank = build_mel_filterbank(config)

    def __call__(self, waveform: Tensor) -> Tensor:
        cfg = self.config
        spectrum = torch.stft(
            waveform,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spectrum.abs() ** 2  # [n_freqs, n_frames]
        mel = self.filterbank @ power  # [n_mels, n_frames]
        log_mel = torch.log(mel + cfg.log_eps)
        return log_mel.unsqueeze(0)  # [1, n_mels, n_frames]


def load_waveform(path: Path, clip_length: int) -> Tensor:
    """Read a mono 16 kHz WAV as float32 in [-1, 1], padded/truncated to length.

    `scipy.io.wavfile` returns int16 PCM for these clips; the divide maps it to
    the [-1, 1] float range the front end expects.
    """
    _sample_rate, samples = wavfile.read(path)
    audio = torch.from_numpy(np.asarray(samples)).to(torch.float32)
    if audio.ndim > 1:  # collapse any stray stereo to mono
        audio = audio.mean(dim=1)
    if audio.dtype != torch.float32:  # pragma: no cover - scipy returns int16
        audio = audio.to(torch.float32)
    audio = audio / 32768.0
    return fix_length(audio, clip_length)


def fix_length(audio: Tensor, clip_length: int) -> Tensor:
    """Right-pad with zeros or truncate so `audio` has exactly `clip_length`."""
    if audio.numel() < clip_length:
        return torch.nn.functional.pad(audio, (0, clip_length - audio.numel()))
    return audio[:clip_length]


def _is_train(path: Path, val_fraction: float) -> bool:
    """Deterministic per-file train/val split from a hash of the filename.

    Hashing the name (not a shuffle) keeps a clip on the same side of the split
    regardless of enumeration order or worker count.
    """
    digest = hashlib.sha1(path.name.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket >= val_fraction


class SpeechCommandsDataset(Dataset):
    """Map-style dataset of (log-mel spectrogram, label) over the keyword clips.

    The (path, label) index is enumerated once at construction; `__getitem__`
    reads the WAV and applies the `LogMelTransform`.
    """

    def __init__(
        self,
        data_dir: Path,
        config: AudioConfig,
        train: bool,
        val_fraction: float = 0.15,
        download: bool = True,
    ) -> None:
        self.config = config
        self.transform = LogMelTransform(config)
        root = ensure_downloaded(data_dir, download=download)

        self.samples: list[tuple[Path, int]] = []
        for label, keyword in enumerate(KEYWORDS):
            for wav_path in sorted((root / keyword).glob("*.wav")):
                if _is_train(wav_path, val_fraction) == train:
                    self.samples.append((wav_path, label))
        if not self.samples:
            raise RuntimeError(f"No WAV files found under {root}; expected {KEYWORDS} subdirs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self.samples[index]
        waveform = load_waveform(path, self.config.clip_length)
        return self.transform(waveform), label


def ensure_downloaded(data_dir: Path, download: bool = True) -> Path:
    """Fetch + extract the archive if missing; return the extracted root dir."""
    root = data_dir / EXTRACT_DIRNAME
    if not root.is_dir():
        if not download:
            raise RuntimeError(f"{root} not found and download=False.")
        download_and_extract_archive(DATA_URL, download_root=str(data_dir))
    return root


def build_dataloaders(
    config: AudioConfig,
    data_dir: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    val_fraction: float = 0.15,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_set = SpeechCommandsDataset(
        data_dir, config, train=True, val_fraction=val_fraction, download=download
    )
    val_set = SpeechCommandsDataset(
        data_dir, config, train=False, val_fraction=val_fraction, download=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
