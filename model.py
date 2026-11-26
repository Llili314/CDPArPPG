"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math

from .transformer_layer import Transformer_ST_TDC_gra_sharp


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


# ================= FFT Amplitude-based Input Temporal Attention (Scheme A) =================
class FFTAmplitudeSpatialAttention(nn.Module):
    """
    Scheme A: Compute spatial attention weights from the *input* video via spatial FFT amplitude.

    Input : x [B, C, T, H, W]
    Output: w [B, 1, T, H, W] in (0,1)

    Steps:
      1) convert to luminance (mean over C)
      2) (optional) spatial downsample for efficiency
      3) rfft2 over (H,W), take amplitude |FFT| (optionally log1p)
      4) crop a low-mid frequency band (avoid DC)
      5) average over freq bins -> amp(t)
      6) inverse FFT of band-pass filtered spectrum -> spatial map w(x,y,t)
    """
    def __init__(
        self,
        kernel_size: int = 5,  # UNUSED (kept for compatibility)
        hidden: int = 16,      # UNUSED (kept for compatibility)
        use_log: bool = True,
        downsample: int = 2,
        band_h: tuple = (0.05, 0.50),
        band_w: tuple = (0.05, 0.50),
        eps: float = 1e-6,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.use_log = use_log
        self.downsample = downsample
        self.band_h = band_h
        self.band_w = band_w
        self.eps = eps

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # 1) luminance proxy (avoid extra cost across RGB)
        y = x.mean(dim=1, keepdim=False)  # [B, T, H, W]

        # 2) downsample spatially for efficiency
        if self.downsample is not None and self.downsample > 1:
            ds = int(self.downsample)
            y2 = y.reshape(B*T, 1, H, W)
            y2 = F.avg_pool2d(y2, kernel_size=ds, stride=ds)
            H2, W2 = y2.shape[-2], y2.shape[-1]
            y = y2.reshape(B, T, H2, W2)
        else:
            H2, W2 = H, W

        # 3) FFT (complex), then magnitude + phase
        Yf = torch.fft.rfft2(y, dim=(-2, -1))      # [B, T, H2, W2//2+1] complex
        mag = Yf.abs()                              # magnitude |F|
        phase = torch.angle(Yf)                     # phase angle(F)

        # optional: log compression on magnitude (attention-only, improves stability)
        if self.use_log:
            mag = torch.log1p(mag)

        # 4) build low-mid frequency mask (avoid DC and very high freq)
        h0 = max(0, int(self.band_h[0] * H2))
        h1 = min(H2, int(self.band_h[1] * H2))
        wR = (W2 // 2) + 1
        w0 = max(0, int(self.band_w[0] * wR))
        w1 = min(wR, int(self.band_w[1] * wR))

        mask = torch.zeros((H2, wR), device=Yf.device, dtype=mag.dtype)
        mask[h0:h1, w0:w1] = 1.0
        mask = mask.view(1, 1, H2, wR)             # [1,1,H2,wR]

        # 5) magnitude-domain band-pass filtering (key requirement)
        mag_bp = mag * mask                         # [B,T,H2,wR]

        # 6) reconstruct complex spectrum using original phase (keep spatial structure)
        Yf_bp = mag_bp * torch.exp(1j * phase)      # complex

        # 7) inverse FFT to get band-limited spatial component
        y_bp = torch.fft.irfft2(Yf_bp, s=(H2, W2), dim=(-2, -1))  # [B,T,H2,W2]

        # 8) spatial attention map: abs + per-frame min-max normalization -> (0,1)
        att = y_bp.abs()                             # [B,T,H2,W2]
        att = att.view(B*T, 1, H2, W2)

        att_min = att.amin(dim=(-2, -1), keepdim=True)
        att_max = att.amax(dim=(-2, -1), keepdim=True)
        att = (att - att_min) / (att_max - att_min + self.eps)   # [B*T,1,H2,W2]

        # 9) upsample back to original H,W if downsampled
        if (H2 != H) or (W2 != W):
            att = F.interpolate(att, size=(H, W), mode='bilinear', align_corners=False)

        att = att.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4)      # [B,1,T,H,W]
        return att
# ==========================================================================================






# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 8,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        #positional_embedding: str = '1d',
        in_channels: int = 3, 
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
    ):
        super().__init__()

        
        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim
        # --- Attention options ---
        # Scheme A: FFT amplitude attention before Stem (recommended)
        self.use_fft_amp_attn = True
        self.fft_amp_attn = FFTAmplitudeSpatialAttention(kernel_size=5, hidden=16, use_log=True, downsample=2)
        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t//ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        #self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads,
                                       # ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        
        
        
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        #self.normLast = nn.LayerNorm(dim, eps=1e-6)
        
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        
        
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x, gra_sharp,return_att=False):

        b, c, t, fh, fw = x.shape
        amp_w = None
        # === FFT amplitude spatial attention (Scheme A) ===
        if self.use_fft_amp_attn and self.fft_amp_attn is not None:
            amp_w = self.fft_amp_attn(x)              # [B,1,T,H,W]
            x = x * amp_w  # broadcast over channel dimension automatically (C)

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, 64, 160, 64, 64]
        
        x = self.patch_embedding(x)  # [B, 64, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 40*4*4, 64]
        
        
        Trans_features, Score1 =  self.transformer1(x, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features2, Score2 =  self.transformer2(Trans_features, gra_sharp)  # [B, 4*4*40, 64]
        #Trans_features3, Score3 =  self.transformer3(Trans_features2, gra_sharp)  # [B, 4*4*40, 64]
        
        

        # upsampling heads
        features_last = Trans_features2.transpose(1, 2).view(b, self.dim, t//4, 4, 4) # [B, 64, 40, 4, 4]
        
        features_last = self.upsample(features_last)		    # x [B, 64, 7*7, 80]
        features_last = self.upsample2(features_last)		    # x [B, 32, 7*7, 160]
        
        features_last = torch.mean(features_last,3)     # x [B, 32, 160, 4]  
        features_last = torch.mean(features_last,3)     # x [B, 32, 160]    
        rPPG = self.ConvBlockLast(features_last)    # x [B, 1, 160]
        
        #pdb.set_trace()
        
        rPPG = rPPG.squeeze(1)

        if return_att:
            return rPPG, Trans_features2, Score1, Score2, amp_w
        return rPPG, Trans_features2, Score1, Score2




