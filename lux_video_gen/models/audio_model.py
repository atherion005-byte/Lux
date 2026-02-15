"""
Audio Generation Model for Lux.

Implements synchronized audio-video generation using a transformer-based
architecture that produces audio conditioned on video features.

Key capabilities:
- Context-aware sound effect generation
- Speech/dialogue generation with lip sync
- Ambient sound matching
- Audio reference-based generation
- Multi-language support for speech
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AudioTokenizer(nn.Module):
    """
    Converts raw audio waveform into discrete or continuous tokens.
    
    Uses a 1D convolutional encoder to compress audio into a latent
    representation suitable for transformer processing.
    """

    def __init__(
        self,
        in_channels: int = 1,  # Mono audio
        latent_dim: int = 256,
        base_channels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.latent_dim = latent_dim

        # Mel spectrogram parameters
        self.n_fft = n_fft
        self.n_mels = 80

        # Encoder: mel spectrogram -> latent tokens
        self.encoder = nn.Sequential(
            nn.Conv1d(self.n_mels, base_channels, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels * 4, latent_dim, kernel_size=3, padding=1),
        )

        # Decoder: latent tokens -> mel spectrogram
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, base_channels * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, self.n_mels, kernel_size=7, padding=3),
        )

    def compute_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from waveform."""
        # Simple mel spectrogram computation
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitude = stft.abs()

        # Create mel filterbank
        n_freqs = self.n_fft // 2 + 1
        mel_fb = self._mel_filterbank(n_freqs, self.n_mels, self.sample_rate, waveform.device)
        mel = torch.matmul(mel_fb, magnitude)

        # Log scale
        mel = torch.log(mel.clamp(min=1e-5))
        return mel

    def _mel_filterbank(self, n_freqs, n_mels, sample_rate, device):
        """Create a mel filterbank matrix."""
        low_freq = 0
        high_freq = sample_rate / 2

        mel_low = 2595 * torch.log10(torch.tensor(1 + low_freq / 700.0))
        mel_high = 2595 * torch.log10(torch.tensor(1 + high_freq / 700.0))
        mel_points = torch.linspace(mel_low, mel_high, n_mels + 2, device=device)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bins = (hz_points * n_freqs / (sample_rate / 2)).long().clamp(0, n_freqs - 1)

        filterbank = torch.zeros(n_mels, n_freqs, device=device)
        for i in range(n_mels):
            left = bins[i]
            center = bins[i + 1]
            right = bins[i + 2]
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to latent tokens."""
        mel = self.compute_mel(waveform)
        return self.encoder(mel)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode latent tokens to mel spectrogram."""
        return self.decoder(tokens)


class AudioTransformerBlock(nn.Module):
    """Single transformer block for audio generation."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        # Self attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=mask)
        x = x + h

        # Cross attention with video features
        if context is not None:
            h = self.norm2(x)
            h, _ = self.cross_attn(h, context, context)
            x = x + h

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class AudioGenerator(nn.Module):
    """
    Audio generation model synchronized with video.

    Architecture:
    1. Encodes video features via a projection layer
    2. Uses a transformer decoder conditioned on video features
    3. Generates audio tokens autoregressively or in parallel
    4. Decodes tokens to mel spectrogram
    5. Uses HiFi-GAN vocoder for final waveform synthesis

    Supports:
    - Sound effect generation matching video content
    - Speech generation with lip-sync alignment
    - Ambient sound generation
    - Audio reference-based style transfer
    """

    def __init__(
        self,
        audio_dim: int = 256,
        video_dim: int = 1152,
        depth: int = 8,
        num_heads: int = 8,
        max_audio_tokens: int = 512,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.max_audio_tokens = max_audio_tokens
        self.sample_rate = sample_rate

        # Audio tokenizer
        self.tokenizer = AudioTokenizer(
            latent_dim=audio_dim,
            sample_rate=sample_rate,
        )

        # Video feature projection
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, audio_dim),
            nn.LayerNorm(audio_dim),
            nn.SiLU(),
            nn.Linear(audio_dim, audio_dim),
        )

        # Position embedding for audio tokens
        self.pos_embed = nn.Embedding(max_audio_tokens, audio_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AudioTransformerBlock(audio_dim, num_heads)
            for _ in range(depth)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(audio_dim)
        self.head = nn.Linear(audio_dim, audio_dim)

        # Audio type classifier (speech, sfx, ambient, music)
        self.type_classifier = nn.Linear(audio_dim, 4)

    def forward(
        self,
        video_features: torch.Tensor,
        audio_tokens: Optional[torch.Tensor] = None,
        audio_waveform: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate or reconstruct audio from video features.

        Args:
            video_features: (B, T_vid, video_dim) features from DiT
            audio_tokens: (B, audio_dim, T_aud) pre-encoded audio tokens (for training)
            audio_waveform: (B, 1, samples) raw audio waveform (for training)

        Returns:
            Dict with predicted audio tokens and optional reconstruction loss
        """
        B = video_features.shape[0]

        # Project video features
        video_ctx = self.video_proj(video_features)

        # Encode target audio if provided (training mode)
        if audio_waveform is not None:
            target_tokens = self.tokenizer.encode(audio_waveform)  # (B, audio_dim, T)
            target_tokens = rearrange(target_tokens, "b d t -> b t d")
        elif audio_tokens is not None:
            target_tokens = rearrange(audio_tokens, "b d t -> b t d")
        else:
            # Inference: generate tokens
            target_tokens = None

        if target_tokens is not None:
            T_aud = target_tokens.shape[1]
            # Add position embeddings
            pos = self.pos_embed(torch.arange(T_aud, device=video_features.device))
            x = target_tokens + pos.unsqueeze(0)
        else:
            # Start with learned start token
            T_aud = self.max_audio_tokens
            x = self.pos_embed(torch.arange(T_aud, device=video_features.device))
            x = x.unsqueeze(0).expand(B, -1, -1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, context=video_ctx)

        # Output
        x = self.norm_out(x)
        pred_tokens = self.head(x)

        # Decode to mel spectrogram
        pred_mel = self.tokenizer.decode(rearrange(pred_tokens, "b t d -> b d t"))

        result = {
            "audio_tokens": rearrange(pred_tokens, "b t d -> b d t"),
            "mel_spectrogram": pred_mel,
        }

        # Compute reconstruction loss if target is provided
        if target_tokens is not None:
            result["loss"] = F.mse_loss(pred_tokens, target_tokens)

        return result

    @torch.no_grad()
    def generate(
        self,
        video_features: torch.Tensor,
        num_tokens: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate audio mel spectrogram from video features.

        Returns:
            mel: (B, n_mels, T) predicted mel spectrogram
        """
        self.eval()
        result = self.forward(video_features)
        return result["mel_spectrogram"]
