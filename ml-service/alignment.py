"""The trainable 'brain' — projection heads + cross-attention that align audio and text into a
shared space. Frozen encoders feed this; only this is trained."""
import torch.nn as nn


class AlignmentEngine(nn.Module):
    def __init__(self, audio_dim: int = 1024, text_dim: int = 768, shared_dim: int = 256):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=8, batch_first=True)
        self.norm_audio = nn.LayerNorm(shared_dim)
        self.norm_text = nn.LayerNorm(shared_dim)

    def forward(self, audio_features, text_features):
        audio_proj = self.audio_proj(audio_features)
        text_proj = self.text_proj(text_features)
        # text queries audio: fuse the audio frames that belong to each word (localization).
        # attn_weights [B, n_text_tokens, n_audio_frames] drive the attention-entropy signal.
        aligned_audio, attn_weights = self.cross_attn(query=text_proj, key=audio_proj, value=audio_proj)
        return self.norm_audio(aligned_audio), self.norm_text(text_proj), attn_weights
