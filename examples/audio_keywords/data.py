"""Mini Speech Commands data loading: WAV files -> log-mel spectrograms.

The dataset is Google's "mini Speech Commands" set (8 keywords, ~8000 mono
16 kHz ~1 s clips), fetched once with torchvision's archive downloader. The
clips are plain 16-bit PCM, so they are decoded with the stdlib `wave` module:
torchaudio 2.x routes `torchaudio.load` through TorchCodec (FFmpeg), which we
don't want to require just to read PCM. The interesting audio processing — the
STFT, the mel filterbank, the log compression — does run on torchaudio, via
`torchaudio.transforms.MelSpectrogram` in the transform, so the model only ever
sees a fixed `[1, n_mels, n_frames]` log-mel tensor and stays a plain 2D CNN
over an "image".

torchaudio is intentionally absent on some PyTorch builds (notably ROCm); the
import below fails with a friendly message rather than a cryptic
`ModuleNotFoundError`.
"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_and_extract_archive

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram
except ImportError as exc:  # pragma: no cover - exercised only where torchaudio is absent
    raise SystemExit(
        "the audio_keywords example requires torchaudio, which is unavailable on "
        "the ROCm PyTorch build. Install a CPU/CUDA build (e.g. `uv sync --group "
        "cu130`) to run this example."
    ) from exc

DATA_URL: str = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
EXTRACT_DIRNAME: str = "mini_speech_commands"

# The eight keyword classes, sorted so the label index is deterministic.
KEYWORDS: tuple[str, ...] = ("down", "go", "left", "no", "right", "stop", "up", "yes")


@dataclass(frozen=True)
class AudioConfig:
    """Static description of the audio task and the log-mel front end.

    `mean` / `std` are the display window for the single-channel spectrogram,
    passed to `nansense.start` as 1-tuples. nansense renders the input as
    `value * std + mean` clamped to `[0, 1]`, so `std = 1 / 20` and
    `mean = 0.5` map a log-mel value of `-10` to black and `+10` to white —
    a fixed `[-10, +10]` window that comfortably brackets the raw log-mel
    values the model sees (log of a power spectrogram, dominated by the
    near-silent padding floor around `log(1e-6) ≈ -13.8`).
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
    # `[-10, +10]` display window for the log-mel spectrogram (see class
    # docstring): mean + std map raw log-mel values to the UI's [0, 1] range.
    mean: tuple[float, ...] = (0.5,)
    std: tuple[float, ...] = (0.05,)


class LogMelTransform:
    """Waveform `[clip_length]` -> log-mel spectrogram `[1, n_mels, n_frames]`.

    Wraps `torchaudio.transforms.MelSpectrogram` (a power spectrogram projected
    through a Slaney-style mel filterbank) and applies `log(x + eps)`. The
    transform is an `nn.Module`, so it deliberately lives here in the dataset —
    never inside the model's `forward` — keeping the model fx-traceable.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.mel = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,  # power spectrogram, matching a classic log-mel front end
        )
        self.mel.eval()

    @torch.no_grad()
    def __call__(self, waveform: Tensor) -> Tensor:
        mel = self.mel(waveform)  # [n_mels, n_frames]
        log_mel = torch.log(mel + self.config.log_eps)
        return log_mel.unsqueeze(0)  # [1, n_mels, n_frames]


def load_waveform(path: Path, config: AudioConfig) -> Tensor:
    """Read a WAV as a mono float32 waveform in [-1, 1], padded/truncated to length."""
    waveform = _load_pcm_wav(path, config.sample_rate)
    return fix_length(waveform, config.clip_length)


def _load_pcm_wav(path: Path, expected_rate: int) -> Tensor:
    """Decode a 16-bit PCM WAV to a mono float32 `[samples]` tensor in [-1, 1].

    Uses the stdlib `wave` module: torchaudio 2.x's `torchaudio.load` decodes
    via TorchCodec (FFmpeg), which we avoid requiring just to read these plain
    PCM clips. torchaudio is still used for the mel front end (see above).
    """
    with wave.open(str(path), "rb") as wav:
        n_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:  # pragma: no cover - mini Speech Commands is 16-bit PCM
        raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit: {path}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    audio = torch.from_numpy(samples.copy())
    if n_channels > 1:  # pragma: no cover - clips are mono
        audio = audio.reshape(-1, n_channels).mean(dim=1)
    if sample_rate != expected_rate:  # pragma: no cover - clips are 16 kHz
        audio = torchaudio.functional.resample(audio, sample_rate, expected_rate)
    return audio


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
        waveform = load_waveform(path, self.config)
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
    batch_size: int = 64,
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
