import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
#  Utils: Pearson autocorr
# -------------------------
def pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x,y: (M,) float tensor
    return: scalar r
    """
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(eps)
    r = (x * y).sum() / denom
    # 数值安全：限制到 [-1,1]
    return r.clamp(-1.0, 1.0)


def pvalue_from_r_approx_torch(r: torch.Tensor, n: int, eps: float = 1e-8) -> torch.Tensor:
    """
    可选：p-value 近似（纯 torch，使用 Fisher z 近似，n>=25 左右比较稳）
    这是“近似”，不是严格等价于 scipy.stats.pearsonr 的精确 p。
    """
    r = r.clamp(-1 + 1e-6, 1 - 1e-6)
    z = 0.5 * torch.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(max(n - 3, 1))
    zscore = z / (se + eps)
    # two-sided p from normal approx
    # p = 2*(1 - Phi(|z|))
    p = 2.0 * (1.0 - 0.5 * (1.0 + torch.erf(zscore.abs() / math.sqrt(2.0))))
    return p.clamp(0.0, 1.0)


def calculate_autocorrelation_torch(
    x: torch.Tensor,
    lag: int,
    level_of_significance_for_pearson: float = 0.01,
    consider_only_significant_correlation: bool = True,
    use_pvalue_approx: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    对应你原来的 calculate_autocorrelation。:contentReference[oaicite:2]{index=2}
    返回 (r, p, corfunc_value, lag)
    """
    if lag == 0:
        r = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        p = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return r, p, r, 0

    x1 = x[lag:]
    x2 = x[:-lag]
    r = pearsonr_torch(x1, x2)

    if use_pvalue_approx:
        p = pvalue_from_r_approx_torch(r, n=x1.numel())
    else:
        # 不算 p-value 时：给一个占位（不会用于筛选）
        p = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    if consider_only_significant_correlation and use_pvalue_approx:
        cor = torch.where(p > level_of_significance_for_pearson,
                          torch.tensor(0.0, device=x.device, dtype=x.dtype),
                          r)
    else:
        cor = r
    return r, p, cor, lag


def autocor_torch(
    x: torch.Tensor,
    lags: torch.Tensor,
    level_of_significance_for_pearson: float = 0.01,
    consider_only_significant_correlation: bool = True,
    use_pvalue_approx: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对应你原来的 autocor。:contentReference[oaicite:3]{index=3}
    x: (T,)
    lags: (K,) int64
    returns: r_list, p_list, corfunc, lag_list
    """
    rs, ps, cs = [], [], []
    for lag in lags.tolist():
        r, p, c, _ = calculate_autocorrelation_torch(
            x, lag,
            level_of_significance_for_pearson,
            consider_only_significant_correlation,
            use_pvalue_approx,
        )
        rs.append(r)
        ps.append(p)
        cs.append(c)
    r_list = torch.stack(rs)
    p_list = torch.stack(ps)
    corfunc = torch.stack(cs)
    lag_list = lags.to(device=x.device)
    return r_list, p_list, corfunc, lag_list


# -------------------------
#  shift_diff / sum_shifted_function
# -------------------------
def shift_diff_torch(i: int, corfunc: torch.Tensor) -> torch.Tensor:
    """
    对应 shift_diff。:contentReference[oaicite:4]{index=4}
    """
    if i == 0:
        return torch.tensor(0.0, device=corfunc.device, dtype=corfunc.dtype)
    w1 = corfunc[i:]
    w2 = corfunc[:-i]
    return (w1.sub(w2).abs().mean())


def sum_shifted_function_torch(i: int, corfunc: torch.Tensor) -> torch.Tensor:
    """
    对应 sum_shifted_function。:contentReference[oaicite:5]{index=5}
    """
    if i == 0:
        return 2.0 * corfunc.abs().mean()
    w1 = corfunc[i:]
    w2 = corfunc[:-i]
    return w1.abs().mean() + w2.abs().mean()


# -------------------------
#  local minima detection (替代 scipy find_peaks(-diffs))
# -------------------------
def local_minima_indices_torch(v: torch.Tensor) -> torch.Tensor:
    """
    返回 v 的“严格局部最小值”索引（1..len-2），边界不算。
    """
    if v.numel() < 3:
        return torch.empty(0, device=v.device, dtype=torch.long)
    left = v[:-2]
    mid = v[1:-1]
    right = v[2:]
    mask = (mid < left) & (mid < right)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1) + 1
    return idx


def get_relevant_diffs_torch(diffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    对应 get_relevant_diffs。:contentReference[oaicite:6]{index=6}
    返回 (relevant_diffs, peaks, stop_calculation)
    """
    stop_calculation = 0
    peaks_inner = local_minima_indices_torch(diffs)

    if peaks_inner.numel() <= 0:
        # 只有 shift=0 一个“人为插入”的 minima 时，算法就没法给出周期
        stop_calculation = 1

    # 插入 shift=0
    peaks = torch.cat([torch.zeros(1, device=diffs.device, dtype=torch.long), peaks_inner.long()], dim=0)
    relevant_diffs = diffs[peaks]
    return relevant_diffs, peaks, stop_calculation


# -------------------------
#  Torch model for phase->value  (替代 RandomForestRegressor)
# -------------------------
class PhaseMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def fit_model_torch(
    phase_mod: torch.Tensor,   # (T,) in [0, period)
    y: torch.Tensor,           # (T,)
    epochs: int = 400,
    lr: float = 5e-2,
    hidden: int = 64,
    weight_decay: float = 0.0,
) -> Tuple[torch.Tensor, nn.Module]:
    """
    对应 fit_model（把 RF 换成 torch MLP）。:contentReference[oaicite:7]{index=7}
    返回 y_model: (T,1) 和 model
    """
    device = y.device
    dtype = y.dtype

    # normalize phase to [0,1] helps training
    period = phase_mod.max().clamp_min(1e-6)
    x_in = (phase_mod / period).unsqueeze(1).to(device=device, dtype=dtype)
    y_in = y.unsqueeze(1).to(device=device, dtype=dtype)

    model = PhaseMLP(hidden=hidden).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        pred = model(x_in)
        loss = F.mse_loss(pred, y_in)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_model = model(x_in)  # (T,1)
    return y_model, model


# -------------------------
#  Main: find_period (Torch)
# -------------------------
@dataclass
class ResultsTorch:
    period: float
    model: Optional[nn.Module]
    criteria: float


def find_period(
    x: torch.Tensor,                         # (T,)
    dt_minutes: float,                       # 相邻采样点间隔（分钟）
    tol_norm_diff: float = 1e-3,
    number_steps: int = 1000,
    minimum_number_of_relevant_shifts: int = 2,
    minimum_number_of_datapoints_for_correlation_test: int = 30,
    minimum_ratio_of_datapoints_for_shift_autocorrelation: float = 0.3,
    consider_only_significant_correlation: bool = True,
    level_of_significance_for_pearson: float = 0.01,
    use_pvalue_approx: bool = False,
    output_flag: int = 1,                    # 这里只控制是否“全 lag”算 residual acf（与原逻辑一致）
) -> Tuple[ResultsTorch, torch.Tensor]:
    """
    基于你的 find_period 主流程改写。:contentReference[oaicite:8]{index=8}
    返回：ResultsTorch + model_data(或原信号)
    """
    device = x.device
    x = x.flatten()
    T = x.numel()

    # Step 2: autocorrelation
    max_lag = max(T - minimum_number_of_datapoints_for_correlation_test, 1)
    lag_list = torch.arange(0, max_lag, device=device, dtype=torch.long)

    r_list, p_list, corfunc, lag_list = autocor_torch(
        x, lag_list,
        level_of_significance_for_pearson,
        consider_only_significant_correlation,
        use_pvalue_approx,
    )

    # Step 3: diffs between acf and shifted acf
    max_shift = int(corfunc.numel() - corfunc.numel() * minimum_ratio_of_datapoints_for_shift_autocorrelation)
    max_shift = max(max_shift, 1)
    diffs = torch.stack([shift_diff_torch(i, corfunc) for i in range(max_shift)])

    # Step 4: local minima of diffs -> candidate shifts
    relevant_diffs, peaks, stop_calculation = get_relevant_diffs_torch(diffs)
    if stop_calculation == 1:
        return ResultsTorch(period=-1.0, model=None, criteria=0.0), x

    # 准备 peaks 的额外筛选指标 sum_of_norms
    sum_of_shifted = torch.stack([sum_shifted_function_torch(int(i), corfunc) for i in peaks.tolist()])

    # Step 5: sweep tolerances
    list_relv_pos: List[int] = []
    size_list_relv_pos = 0

    suggested_periods: List[float] = []
    criteria_list: List[float] = []
    norms_sum_list: List[float] = []
    norm_diff_list: List[float] = []
    model_data_list: List[torch.Tensor] = []
    model_list: List[nn.Module] = []
    tolerances_list: List[float] = []

    # 用 torch linspace 扫 tolerance
    for tol_for_zero in torch.linspace(0.0, 1.0, number_steps + 1, device=device):
        tol = float(tol_for_zero.item())

        # vec_bool: diffs<=tol and sum_norms>tol
        vec_bool = (relevant_diffs <= tol_for_zero) & (sum_of_shifted > tol_for_zero)
        relv = peaks[vec_bool].tolist()

        if len(relv) >= minimum_number_of_relevant_shifts and len(relv) > size_list_relv_pos:
            size_list_relv_pos = len(relv)
            list_relv_pos = relv

            corr_at_peaks = corfunc[torch.tensor(list_relv_pos, device=device, dtype=torch.long)]
            all_positive = (corr_at_peaks <= 0).sum().item() == 0
            if not all_positive:
                break

            tolerances_list.append(tol)

            # Step 5(c)(d): periods from diffs of relevant peak times
            relv_time_minutes = torch.tensor(list_relv_pos, device=device, dtype=x.dtype) * float(dt_minutes)
            # diffs between consecutive relevant positions
            if relv_time_minutes.numel() >= 2:
                list_of_periods = relv_time_minutes[1:] - relv_time_minutes[:-1]
                suggested_period = float(list_of_periods.median().item())
            else:
                continue

            # Step 5(e): phase folding (modulo)
            # 这里时间轴 = index * dt_minutes，相对 reference_time 的偏移等价于 index*dt
            t_minutes = torch.arange(T, device=device, dtype=x.dtype) * float(dt_minutes)
            phase_mod = torch.remainder(t_minutes, suggested_period)  # (T,)

            # Step 5(e): fit model (phase -> value)
            model_data, model = fit_model_torch(phase_mod, x)

            # Step 5(f)(g): residual and its autocorr at relevant peaks
            residual = x.unsqueeze(1) - model_data  # (T,1)

            if output_flag == 1:
                # 全 lag residual acf
                r_d, p_d, cor_d, _ = autocor_torch(
                    residual.squeeze(1),
                    lag_list,
                    level_of_significance_for_pearson,
                    consider_only_significant_correlation,
                    use_pvalue_approx,
                )
                corr_resid_at_peaks = cor_d[torch.tensor(list_relv_pos, device=device, dtype=torch.long)]
            else:
                # 只算 relevant lags
                relv_lags = torch.tensor(list_relv_pos, device=device, dtype=torch.long)
                r_d, p_d, cor_d, _ = autocor_torch(
                    residual.squeeze(1),
                    relv_lags,
                    level_of_significance_for_pearson,
                    consider_only_significant_correlation,
                    use_pvalue_approx,
                )
                corr_resid_at_peaks = cor_d

            # reduction criterion
            base = corfunc[torch.tensor(list_relv_pos, device=device, dtype=torch.long)]
            # 跳过第一个 (shift=0)
            if base.numel() > 1:
                reduction = 1.0 - (corr_resid_at_peaks[1:].abs().mean() / base[1:].abs().mean()).item()
            else:
                reduction = 0.0

            # norms (对应原实现的 mean abs)
            norm_signal = x.abs().mean().item()
            norm_model = model_data.squeeze(1).abs().mean().item()
            norm_sum = norm_signal + norm_model
            norm_diff = residual.squeeze(1).abs().mean().item()

            suggested_periods.append(suggested_period)
            criteria_list.append(reduction)
            norms_sum_list.append(norm_sum)
            norm_diff_list.append(norm_diff)
            model_data_list.append(model_data)
            model_list.append(model)

    if len(suggested_periods) == 0:
        return ResultsTorch(period=-1.0, model=None, criteria=0.0), x

    # Step 6: choose best period
    # (1) very close fit: norm_diff <= tol_norm_diff and sum_norms > tol_norm_diff
    idx_close = [i for i in range(len(suggested_periods))
                 if (norm_diff_list[i] <= tol_norm_diff and norms_sum_list[i] > tol_norm_diff)]
    if len(idx_close) > 0:
        best_i = idx_close[0]
        return ResultsTorch(period=suggested_periods[best_i], model=model_list[best_i], criteria=1.5), model_data_list[best_i]

    # (2) otherwise: max reduction criterion
    best_i = int(torch.tensor(criteria_list).argmax().item())
    return ResultsTorch(period=suggested_periods[best_i], model=model_list[best_i], criteria=criteria_list[best_i]), model_data_list[best_i]
