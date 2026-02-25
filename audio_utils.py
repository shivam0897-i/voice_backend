"""
Audio utility functions for Base64 decoding and audio loading.
"""
import base64
import io
import tempfile
import os
import logging
from typing import Tuple, Optional
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

# Magic bytes for common audio formats
AUDIO_MAGIC_BYTES = {
    b'\xff\xfb': 'mp3',      # MP3 (MPEG Audio Layer 3)
    b'\xff\xfa': 'mp3',      # MP3 variant
    b'\xff\xf3': 'mp3',      # MP3 variant
    b'\xff\xf2': 'mp3',      # MP3 variant
    b'ID3': 'mp3',           # MP3 with ID3 tag
    b'RIFF': 'wav',          # WAV
    b'fLaC': 'flac',         # FLAC
    b'OggS': 'ogg',          # OGG
    b'\x1a\x45\xdf\xa3': 'webm',  # WebM / Matroska container
    # M4A/MP4 detected via ftyp at offset 4 (see validate_audio_content)
}


def validate_audio_content(audio_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate that the bytes actually contain audio data.
    
    Args:
        audio_bytes: Raw bytes to validate
        
    Returns:
        Tuple of (is_valid, detected_format_or_error_message)
    """
    if len(audio_bytes) < 12:
        return False, "Audio data too small to be valid"
    
    # Check for text content (common mistake: uploading CSV/JSON as audio)
    # ASCII printable range check on first 100 bytes
    sample = audio_bytes[:100]
    printable_ratio = sum(1 for b in sample if 32 <= b <= 126 or b in (9, 10, 13)) / len(sample)
    if printable_ratio > 0.9:
        # Likely text content
        preview = sample[:50].decode('utf-8', errors='replace')
        return False, f"File appears to be text, not audio. Preview: {preview[:30]}..."
    
    # Check magic bytes
    for magic, fmt in AUDIO_MAGIC_BYTES.items():
        if audio_bytes.startswith(magic):
            return True, fmt
    
    # Check for M4A/MP4 (ftyp at offset 4)
    if len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp':
        return True, "m4a"
    
    # Unknown format but not text - allow it and let librosa try
    logger.warning("Unknown audio format, attempting to load anyway")
    return True, "unknown"


def decode_base64_audio(base64_string: str) -> bytes:
    """
    Decode a Base64-encoded audio string to raw bytes.
    
    Args:
        base64_string: Base64-encoded audio data
        
    Returns:
        Raw audio bytes
        
    Raises:
        ValueError: If the Base64 string is invalid
    """
    try:
        # Strip data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        
        # Remove any whitespace
        base64_string = base64_string.strip()
        
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid Base64 encoding: {str(e)}")


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 22050, audio_format: str = "mp3") -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes into a numpy array using librosa.
    
    Args:
        audio_bytes: Raw audio file bytes
        target_sr: Target sample rate (default 22050 Hz)
        audio_format: Audio format extension (mp3, wav, flac, ogg, m4a, mp4)
        
    Returns:
        Tuple of (audio waveform as numpy array, sample rate)
        
    Raises:
        ValueError: If audio cannot be loaded or is invalid
    """
    is_valid, validation_result = validate_audio_content(audio_bytes)
    if not is_valid:
        raise ValueError(f"Invalid audio file: {validation_result}")
    
    logger.info("Audio validation passed. Detected format hint: %s", validation_result)
    
    tmp_path = None
    try:
        audio_format = audio_format.lower().strip()
        if audio_format.startswith("."):
            audio_format = audio_format[1:]
        
        # Reject suspicious format strings
        if not audio_format.isalnum() or len(audio_format) > 5:
            raise ValueError(f"Invalid audio format: {audio_format}")
        
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        audio, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        
        if len(audio) == 0:
            raise ValueError("Audio file is empty or could not be decoded")
        
        duration = len(audio) / sr
        logger.info("Audio loaded: %.2fs at %dHz", duration, sr)
        
        return audio, sr
                
    except Exception as e:
        raise ValueError(f"Failed to load audio: {str(e)}")
    finally:
        # Always clean up temp file, even on exceptions
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # Best effort cleanup


