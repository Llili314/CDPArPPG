# sfda_physformer_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def update_ema(teacher, student, m: float):
    # 1) update parameters
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

    # 2) copy buffers (BatchNorm running stats, num_batches_tracked, etc.)
    for b_t, b_s in zip(teacher.buffers(), student.buffers()):
        b_t.data.copy_(b_s.data)


def z_norm(x: torch.Tensor, eps: float = 1e-8):
    # x: [B,T]
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + eps)

class NegPearson(nn.Module):
    """
    向量化 Pearson loss：
    输入:
        preds: [B, T]
        labels: [B, T]
    输出:
        标量 loss
    """
    def __init__(self, eps: float = 1e-8):
        super(NegPearson, self).__init__()
        self.eps = eps

    def forward(self, preds, labels):
        # 确保是 float 类型
        preds = preds.float()
        labels = labels.float()

        # 如果是 [B, 1, T] 或 [B, T, 1] 这类，先压到 [B, T]
        if preds.dim() > 2:
            preds = preds.view(preds.size(0), -1)
            labels = labels.view(labels.size(0), -1)

        # [B, T]
        N = preds.size(1)

        sum_x  = torch.sum(preds, dim=1)
        sum_y  = torch.sum(labels, dim=1)
        sum_xy = torch.sum(preds * labels, dim=1)
        sum_x2 = torch.sum(preds * preds, dim=1)
        sum_y2 = torch.sum(labels * labels, dim=1)

        # 分子 & 分母: [B]
        numerator = N * sum_xy - sum_x * sum_y
        denominator = torch.sqrt(
            (N * sum_x2 - sum_x * sum_x) *
            (N * sum_y2 - sum_y * sum_y) + self.eps
        )

        pearson = numerator / denominator  # [B]

        # 可以按 batch 求平均
        loss = 1.0 - pearson
        return loss.mean()

def pearson_normalize(x: torch.Tensor, eps: float = 1e-6):
    """
    x: [B, T', C] or [B, P, C]
    Pearson-style normalization along feature dim
    """
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True) + eps)
    return x

#把B,512,96维度变为B，32,96
def pool_tokens_spatial(tokens: torch.Tensor, spatial_tokens: int = 16):
    """
    tokens: [B, P, C], where P = T' * 16 (16 = 4*4)
    return: [B, T', C]
    """
    B, P, C = tokens.shape
    assert P % spatial_tokens == 0
    T_ = P // spatial_tokens
    x = tokens.view(B, T_, spatial_tokens, C).mean(dim=2)
    return x

def build_st_graphs(tokens_s: torch.Tensor, tokens_t: torch.Tensor):
    """
    return: GS, GT, GTS [B,P,P]
    """

    # 2. Pearson normalize（关键）
    S = pearson_normalize(tokens_s)
    T = pearson_normalize(tokens_t)
    TS = pearson_normalize((tokens_s + tokens_t) / 2.0)
    # 3. 用 dot-product（现在等价于 Pearson）

    GS = torch.clamp(torch.matmul(S, S.transpose(1, 2)), -1.0, 1.0)
    GT = torch.clamp(torch.matmul(T, T.transpose(1, 2)), -1.0, 1.0)
    GTS = torch.clamp(torch.matmul(TS, TS.transpose(1, 2)), -1.0, 1.0)

    return S, T, TS, GS, GT, GTS

def compute_inter_time_graph_consistency(tokens_s, tokens_t):
    """
    ECA-style inter-graph consistency on time-token graphs
    """
    tokens_s = pool_tokens_spatial(tokens_s)  # [B,T',C]
    tokens_t = pool_tokens_spatial(tokens_t)  # [B,T',C]
    _, _, _, GS, GT, GTS = build_st_graphs(tokens_s, tokens_t)

    loss_gs_gt  = torch.mean((GS - GT.detach())  ** 2)
    loss_gs_gts = torch.mean((GS - GTS.detach()) ** 2)
    loss_gt_gts = torch.mean((GT.detach() - GTS.detach()) ** 2)

    loss_eca = loss_gs_gt + loss_gs_gts + loss_gt_gts
    return loss_eca

def compute_intra_graph_consistency(tokens_s, tokens_t, thr: float = 0.6,detach_teacher: bool = True):
    """
    Intra-graph consistency without pseudo labels:
    use teacher graph similarity GT to build mask M:
        M_ij = 1 if GT_ij > thr else 0, and i!=j
    then compute ACE-like loss on masked pairs:
        (1 - ES) + (1 - ET) + (1 - ETS)
    """
    device = tokens_s.device
    B = tokens_s.size(0)

    # 你原来这里写了 B<=1 直接 return，这会让 batch=1 时 loss 恒为 0
    # 可以保留
    if B <= 0:
        return torch.tensor(0.0, device=device)

    # 用你现有的公共建图函数（里面已经做了 pearson_normalize）
    S, T, TS, GS, GT, GTS = build_st_graphs(tokens_s, tokens_t)  # GS/GT/GTS: [B,P,P]

    if detach_teacher:
        GT = GT.detach()
        GTS = GTS.detach()

    # 1) 用 teacher 图生成 mask：相似度大于阈值 -> 1，否则 0
    # 注意：GT 值域约 [-1,1]（你 build_st_graphs 里 clamp 了）
    M = (GT > thr).float()  # [B,P,P]

    # 2) 去掉对角线（i==j 不参与）
    eye = torch.eye(M.size(-1), device=device).unsqueeze(0)  # [1,P,P]
    M = M * (1.0 - eye)

    # 3) denom 防空（防止阈值太大导致没有 pair）
    denom = M.sum()
    if denom.item() <= 0:
        return torch.tensor(0.0, device=device)

    # 4) 把相似度当作 ε，并 clamp 到 [0,1]（保证 1-ε 有意义）
    E_S  = torch.clamp(GS,  0.0, 1.0)
    E_T  = torch.clamp(GT,  0.0, 1.0)
    E_TS = torch.clamp(GTS, 0.0, 1.0)

    # 5) ACE-style term（对应论文 Eq.(7) 的三项，只不过你的 M 来自阈值而不是 pseudo label）
    term = (1.0 - E_S) + (1.0 - E_T) + (1.0 - E_TS)  # [B,P,P]

    loss_ace = (M * term).sum() / denom
    return loss_ace

