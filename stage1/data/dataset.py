"""
dataset.py — PyTorch Dataset that reads one sample per manifest row,
loading each enabled modality on-the-fly.

Supported modalities and their loaded representation:
  image  -> torch.Tensor [C, H, W]        (after transforms)
  audio  -> torch.Tensor [n_mels, T]      (log-mel spectrogram)
  video  -> torch.Tensor [max_frames, C, H, W]  (uniformly sampled frames)
  text   -> dict with keys 'input_ids', 'attention_mask'
             OR torch.Tensor [embedding_dim] if using static embeddings
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Optional heavy imports (guarded) ──────────────────────────────────────────
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

try:
    import torchvision.transforms as T
    import torchvision.io as tvio
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Default transforms
# ──────────────────────────────────────────────────────────────────────────────

def _default_image_transform(train: bool):
    if _TORCHVISION_AVAILABLE:
        if train:
            return T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return T.Compose([
            T.Resize((224, 224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    # Fallback using PIL and NumPy
    def _fallback_img(img):
        img = img.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()
    return _fallback_img


def _default_frame_transform(train: bool):
    """Per-frame transform for video."""
    if _TORCHVISION_AVAILABLE:
        if train:
            return T.Compose([
                T.Resize((112, 112)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def _fallback_frame(img):
        img = img.convert("RGB").resize((112, 112))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()
    return _fallback_frame



# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Manifest rows (already filtered to the desired split).
    modalities : list of str
        Active modalities.
    encoder_cfg : ConfigNode
        ``cfg.encoder`` sub-config.
    train : bool
        Enables training-time augmentations.
    tokenizer_name : str, optional
        If text is enabled and not using pre-computed paths, the HuggingFace
        tokenizer name (default: ``bert-base-uncased``).
    max_text_length : int
        Tokenizer max length.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        modalities: List[str],
        encoder_cfg,
        train: bool = True,
        tokenizer_name: str = "bert-base-uncased",
        max_text_length: int = 128,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.modalities = modalities
        self.cfg = encoder_cfg
        self.train = train
        self.max_text_length = max_text_length

        # ── Per-modality setup ──
        if "image" in modalities:
            self._img_transform = _default_image_transform(train)

        if "video" in modalities:
            self._frame_transform = _default_frame_transform(train)
            self.max_frames: int = getattr(encoder_cfg.video, "max_frames", 16)

        if "audio" in modalities:
            self.n_mels: int = getattr(encoder_cfg.audio, "n_mels", 64)

        if "text" in modalities:
            if _TRANSFORMERS_AVAILABLE and "text" in df.columns:
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            else:
                self._tokenizer = None

    # ── len / getitem ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sample: Dict[str, Any] = {
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
        }

        for mod in self.modalities:
            sample[mod] = self._load_modality(mod, row)

        return sample

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_modality(self, mod: str, row: pd.Series) -> Any:
        if mod == "image":
            return self._load_image(str(row["image_path"]))
        if mod == "audio":
            return self._load_audio(str(row["audio_path"]))
        if mod == "video":
            return self._load_video(str(row["video_path"]))
        if mod == "text":
            if "text" in row and pd.notna(row["text"]):
                return self._load_text(str(row["text"]))
            return self._load_text_from_file(str(row["text_path"]))
        raise ValueError(f"Unknown modality: {mod}")

    def _load_image(self, path: str) -> torch.Tensor:
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow (PIL) is required for image modality.")
        img = PILImage.open(path).convert("RGB")
        return self._img_transform(img)  # [3, H, W]

    def _load_audio(self, path: str) -> torch.Tensor:
        if _TORCHAUDIO_AVAILABLE:
            waveform, sr = torchaudio.load(path)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample to 16 kHz
            target_sr = 16_000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            # Mel spectrogram
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sr, n_mels=self.n_mels
            )(waveform)
            log_mel = torch.log(mel + 1e-9)  # [1, n_mels, T]
            return log_mel.squeeze(0)         # [n_mels, T]
        else:
            import wave
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                n_ch = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                sampwidth = wf.getsampwidth()
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            raw_arr = np.frombuffer(frames, dtype=dtype_map.get(sampwidth, np.int16))
            if n_ch > 1:
                raw_arr = raw_arr.reshape(-1, n_ch).mean(axis=1)
            sig = torch.from_numpy(raw_arr.astype(np.float32) / 32768.0).unsqueeze(0)
            n_fft = 512
            hop = 256
            win = torch.hann_window(n_fft)
            stft = torch.stft(sig, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            mag = torch.abs(stft).squeeze(0)  # [F, T]
            if mag.shape[0] != self.n_mels:
                mag_4d = mag.unsqueeze(0).unsqueeze(0)
                mag_res = torch.nn.functional.interpolate(mag_4d, size=(self.n_mels, mag.shape[-1]), mode="bilinear", align_corners=False)
                mag = mag_res.squeeze(0).squeeze(0)
            return torch.log(mag + 1e-9)

    def _load_video(self, path: str) -> torch.Tensor:
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for video modality.")
        # Read all frames as uint8 [T, H, W, C]
        try:
            frames, _, _ = tvio.read_video(path, pts_unit="sec")
        except Exception as e:
            raise RuntimeError(f"Failed to read video {path}: {e}") from e

        total = frames.shape[0]
        n = min(self.max_frames, total)
        if total == 0:
            raise RuntimeError(f"Video has no frames: {path}")

        # Uniform sampling
        indices = np.linspace(0, total - 1, n, dtype=int)
        selected = frames[indices]          # [n, H, W, C]

        # Per-frame transform (PIL-based)
        out = []
        for i in range(n):
            frame_np = selected[i].numpy()  # [H, W, C]
            pil = PILImage.fromarray(frame_np)
            out.append(self._frame_transform(pil))  # [3, H, W]

        tensor = torch.stack(out, dim=0)    # [n, 3, H, W]

        # Pad to max_frames if fewer frames
        if n < self.max_frames:
            pad = torch.zeros(
                self.max_frames - n, *tensor.shape[1:], dtype=tensor.dtype
            )
            tensor = torch.cat([tensor, pad], dim=0)

        return tensor                       # [max_frames, 3, H, W]

    def _load_text(self, text: str) -> Dict[str, torch.Tensor]:
        if self._tokenizer is None:
            # Fallback: return raw string (encoder must handle it)
            return {"raw": text}
        enc = self._tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def _load_text_from_file(self, path: str) -> Dict[str, torch.Tensor]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return self._load_text(text)
