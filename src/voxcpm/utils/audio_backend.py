"""Audio processing backend abstraction layer.

Provides a unified interface for audio loading and processing,
supporting multiple backends (torchaudio, librosa).

This module enables:
- GPU acceleration via torchaudio (default)
- Backward compatibility via librosa fallback
- Clean separation of audio processing concerns

Usage:
    >>> from voxcpm.utils.audio_backend import get_audio_backend
    >>> backend = get_audio_backend()
    >>> audio, sr = backend.load("audio.wav", sample_rate=16000)
    >>> trimmed = backend.trim_silence(audio, sr, top_db=35.0)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class AudioBackend(ABC):
    """Abstract base class for audio processing backends.
    
    Subclasses must implement load() and trim_silence() methods.
    All backends should return audio in (channels, samples) format.
    """
    
    @abstractmethod
    def load(
        self,
        path: str,
        sample_rate: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """Load audio file.
        
        Args:
            path: Path to audio file
            sample_rate: Target sample rate (None to keep original)
            mono: Convert to mono if True
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
            audio_tensor shape: (1, samples) for mono, (channels, samples) for multi-channel
        """
        pass
    
    @abstractmethod
    def trim_silence(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        top_db: float = 35.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        max_silence_ms: float = 200.0
    ) -> torch.Tensor:
        """Trim silence from audio boundaries.
        
        Args:
            audio: Audio tensor of shape (1, samples) or (channels, samples)
            sample_rate: Audio sample rate
            top_db: Threshold in dB below reference for silence detection
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            max_silence_ms: Maximum silence to preserve at boundaries
            
        Returns:
            Trimmed audio tensor
        """
        pass
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend is available.
        
        Returns:
            True if backend can be instantiated, False otherwise
        """
        return True


class TorchaudioBackend(AudioBackend):
    """Torchaudio-based audio processing backend.
    
    Advantages:
    - GPU acceleration support
    - Native torch tensor operations (no numpy conversion)
    - Already a project dependency
    - Consistent with PyTorch ecosystem
    
    Note:
        For silence trimming, this backend implements custom energy-based
        detection that matches librosa.effects.trim behavior, since
        torchaudio.functional.vad is designed for voice activity detection
        rather than silence trimming.
    """
    
    def __init__(self):
        import torchaudio
        self._ta = torchaudio
        self._ta_vad_available = hasattr(torchaudio.functional, 'vad')
        logger.debug("TorchaudioBackend initialized (vad_available=%s)", self._ta_vad_available)
    
    def load(
        self,
        path: str,
        sample_rate: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """Load audio using torchaudio.

        Implementation notes:
        - Uses soundfile directly to avoid torchcodec dependency in newer torchaudio
        - Returns (channels, samples) format
        - Resampling uses torchaudio.transforms.Resample for quality

        Args:
            path: Path to audio file
            sample_rate: Target sample rate (None to keep original)
            mono: Convert to mono if True

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Use soundfile directly to avoid torchcodec dependency
        # Newer torchaudio versions require torchcodec which is not always available
        import soundfile as sf
        import numpy as np
        
        # Load audio with soundfile
        audio_np, sr = sf.read(path)
        
        # Convert to tensor
        # soundfile returns (samples, channels), we need (channels, samples)
        if audio_np.ndim == 1:
            # Mono audio
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            # Multi-channel audio: transpose to (channels, samples)
            waveform = torch.from_numpy(audio_np.T)
        
        # Convert to float32 for consistency with torchaudio
        waveform = waveform.float()
        
        # Convert to mono if requested
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate is not None and sr != sample_rate:
            resampler = self._ta.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate
        
        return waveform, sr
        
        # Convert to mono if requested
        if mono and waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif mono and waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample if needed
        if sample_rate is not None and sr != sample_rate:
            resampler = self._ta.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate
        
        return waveform, sr
    
    def trim_silence(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        top_db: float = 35.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        max_silence_ms: float = 200.0
    ) -> torch.Tensor:
        """Trim silence using energy-based detection.
        
        This implementation matches the behavior of librosa.effects.trim:
        1. Calculate RMS energy for each frame
        2. Find frames above threshold (ref - top_db)
        3. Trim to first and last non-silent frame
        4. Add margin for max_silence_ms
        
        Args:
            audio: Audio tensor of shape (1, samples) or (channels, samples)
            sample_rate: Audio sample rate
            top_db: Threshold in dB below reference for silence detection
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            max_silence_ms: Maximum silence to preserve at boundaries
            
        Returns:
            Trimmed audio tensor
        """
        if audio.numel() == 0:
            return audio
        
        # Ensure 2D tensor (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Use first channel for silence detection
        y = audio[0] if audio.shape[0] == 1 else audio.mean(dim=0)
        num_samples = y.shape[-1]
        
        # Calculate reference level (maximum absolute amplitude)
        ref = torch.max(torch.abs(y))
        if ref <= 0:
            return audio
        
        # Calculate threshold from top_db
        threshold = ref * (10.0 ** (-top_db / 20.0))
        
        # Calculate number of frames
        num_frames = max(0, (num_samples - frame_length) // hop_length + 1)
        if num_frames == 0:
            return audio
        
        # Calculate RMS energy for each frame using vectorized operations
        # Create frame indices
        frame_starts = torch.arange(num_frames, device=audio.device) * hop_length
        frame_ends = torch.minimum(frame_starts + frame_length, torch.tensor(num_samples, device=audio.device))
        
        # Calculate energy for each frame
        energy = torch.zeros(num_frames, device=audio.device)
        for i in range(num_frames):
            start = frame_starts[i].item()
            end = frame_ends[i].item()
            frame = y[start:end]
            energy[i] = torch.sqrt(torch.mean(frame ** 2))
        
        # Find non-silent frames
        non_silent = energy >= threshold
        if not torch.any(non_silent):
            return audio
        
        # Find start and end indices
        indices = torch.where(non_silent)[0]
        start_frame = indices[0].item()
        end_frame = indices[-1].item()
        
        # Convert frame indices to sample indices
        start_sample = max(0, start_frame * hop_length)
        end_sample = min(num_samples, (end_frame + 1) * hop_length + (frame_length - hop_length))
        
        # Add margin for max_silence_ms
        max_silence_samples = int(max_silence_ms * sample_rate / 1000.0)
        start_sample = max(0, start_sample - max_silence_samples)
        end_sample = min(num_samples, end_sample + max_silence_samples)
        
        return audio[:, start_sample:end_sample]
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            import torchaudio
            return True
        except ImportError:
            return False


class LibrosaBackend(AudioBackend):
    """Librosa-based audio processing backend.
    
    Provided for backward compatibility and as fallback.
    Uses librosa.load() and librosa.effects.trim() internally.
    
    Note:
        This backend requires librosa to be installed.
        Audio is converted between numpy and torch tensors,
        which may have performance implications.
    """
    
    def __init__(self):
        import librosa
        import numpy as np
        self._librosa = librosa
        self._np = np
        logger.debug("LibrosaBackend initialized")
    
    def load(
        self,
        path: str,
        sample_rate: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """Load audio using librosa.
        
        Args:
            path: Path to audio file
            sample_rate: Target sample rate (None to keep original)
            mono: Convert to mono if True
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        audio, sr = self._librosa.load(
            path, 
            sr=sample_rate, 
            mono=mono
        )
        # Convert numpy array to torch tensor
        # librosa returns (samples,) for mono, we want (1, samples)
        audio = torch.from_numpy(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return audio, sr
    
    def trim_silence(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        top_db: float = 35.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        max_silence_ms: float = 200.0
    ) -> torch.Tensor:
        """Trim silence using librosa.effects.trim.
        
        This implementation exactly matches the original voxcpm2.py behavior,
        including the pseudo-silence trimming for low-energy background noise.
        
        Args:
            audio: Audio tensor of shape (1, samples) or (channels, samples)
            sample_rate: Audio sample rate
            top_db: Threshold in dB below reference for silence detection
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            max_silence_ms: Maximum silence to preserve at boundaries
            
        Returns:
            Trimmed audio tensor
        """
        if audio.numel() == 0:
            return audio
        
        # Convert to numpy for librosa
        y = audio.squeeze(0).numpy()
        n = len(y)
        
        ref = self._np.max(self._np.abs(y))
        if ref <= 0:
            return audio
        
        threshold = ref * (10.0 ** (-top_db / 20.0))
        
        try:
            _, (start, end) = self._librosa.effects.trim(
                y, 
                top_db=top_db, 
                ref=self._np.max, 
                frame_length=frame_length, 
                hop_length=hop_length
            )
        except Exception:
            start, end = 0, n
        
        # Find the last frame with continuous energy
        # This matches the original implementation for trimming pseudo-silence
        n_frames = max(0, (n - frame_length) // hop_length + 1)
        last_voice_frame = -1
        
        for j in range(n_frames):
            idx = j * hop_length
            if idx + frame_length > n:
                break
            rms = self._np.sqrt(self._np.mean(y[idx:idx + frame_length] ** 2))
            if rms >= threshold:
                last_voice_frame = j
        
        if last_voice_frame >= 0:
            end_by_vad = min(n, (last_voice_frame + 1) * hop_length + (frame_length - hop_length))
            end = min(end, end_by_vad)
        
        # Add margin
        max_silence_samples = int(max_silence_ms * sample_rate / 1000.0)
        new_start = max(0, start - max_silence_samples)
        new_end = min(n, end + max_silence_samples)
        
        return audio[:, new_start:new_end]
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            import librosa
            return True
        except ImportError:
            return False


# Module-level singleton
_audio_backend: Optional[AudioBackend] = None


def get_audio_backend(prefer_torchaudio: bool = True) -> AudioBackend:
    """Get the audio processing backend.
    
    Backend selection order:
    1. If prefer_torchaudio=True (default): Try TorchaudioBackend first
    2. Fallback to LibrosaBackend if preferred not available
    3. Raise RuntimeError if no backend available
    
    Args:
        prefer_torchaudio: If True, prefer torchaudio backend
        
    Returns:
        AudioBackend instance (singleton)
        
    Raises:
        RuntimeError: If no backend is available
    """
    global _audio_backend
    
    if _audio_backend is not None:
        return _audio_backend
    
    backends_to_try = []
    if prefer_torchaudio:
        backends_to_try = [TorchaudioBackend, LibrosaBackend]
    else:
        backends_to_try = [LibrosaBackend, TorchaudioBackend]
    
    for backend_cls in backends_to_try:
        if backend_cls.is_available():
            _audio_backend = backend_cls()
            logger.info("Using %s for audio processing", backend_cls.__name__)
            return _audio_backend
    
    raise RuntimeError(
        "No audio backend available. "
        "Please install torchaudio (pip install torchaudio) or librosa (pip install librosa)."
    )


def set_audio_backend(backend: Optional[AudioBackend]) -> None:
    """Set the audio processing backend.
    
    Use this to override the default backend selection.
    Pass None to reset and allow automatic selection on next get_audio_backend() call.
    
    Args:
        backend: AudioBackend instance or None to reset
        
    Example:
        >>> from voxcpm.utils.audio_backend import LibrosaBackend, set_audio_backend
        >>> set_audio_backend(LibrosaBackend())  # Force librosa
        >>> set_audio_backend(None)  # Reset to automatic selection
    """
    global _audio_backend
    _audio_backend = backend
    if backend is not None:
        logger.info("Audio backend manually set to %s", type(backend).__name__)
    else:
        logger.info("Audio backend reset to automatic selection")


__all__ = [
    'AudioBackend',
    'TorchaudioBackend',
    'LibrosaBackend',
    'get_audio_backend',
    'set_audio_backend',
]
