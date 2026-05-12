import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from torch.nn.utils.parametrizations import spectral_norm
import torchvision.transforms.functional as TF
import torch.fft

# 基础工具部分
class DevelopmentalVisualDiet(nn.Module):
    """
    DVD 热力学退火调度器
    物理职责：模拟生物发育，控制微观层激波 J_ext 的香农信息熵注入速率。
    """
    def __init__(self, max_epochs=300, alpha=2, beta=1e-4, lambda_c=150):
        super().__init__()
        self.max_months = max_epochs # 模拟人类发育到 25岁(300个月)
        self.alpha = alpha    # 1 epoch = alpha 个月
        self.beta = beta      # 初始对比度阈值
        self.lambda_c = lambda_c # 敏感度映射因子
        
    def forward(self, image_tensor, current_epoch):
        """
        image_tensor:[B, C, H, W] 原始高清图片
        """
        # 1. 计算当前的"生物学年龄" (月)
        age_months = min(current_epoch * self.alpha, self.max_months)
        
        # 2. 视觉敏锐度发育 (Acuity - Gaussian Blur)
        # 婴儿期 sigma 大(极度模糊)，成年后 sigma -> 0
        sigma = 4.0 * math.exp(-age_months / 20.0) 
        if sigma > 0.1:
            kernel_size = int(2 * math.ceil(2 * sigma) + 1)
            img_blurred = TF.gaussian_blur(image_tensor, kernel_size, [sigma, sigma])
        else:
            img_blurred = image_tensor
            
        # 3. 对比度敏感度发育 (Contrast - Frequency Thresholding)
        # 这是论文指出的最关键因素：过滤低频但低振幅的纹理信号
        img_fft = torch.fft.fft2(img_blurred)
        amp_spectrum = torch.abs(img_fft)
        
        # 计算随年龄降低的过滤阈值 T_t
        C_t = min(age_months / 300.0, 1.0) # 0 到 1
        threshold = (amp_spectrum.amax(dim=(-2, -1), keepdim=True) * self.beta * (1 - C_t)) / max(age_months / self.lambda_c, 1.0)
        
        # 抹除低于阈值的频率 (强行去除高频纹理细节)
        fft_filtered = torch.where(amp_spectrum < threshold, torch.zeros_like(img_fft), img_fft)
        img_contrast = torch.fft.ifft2(fft_filtered).real
        
        # 4. 色彩保真度发育 (Chromatic - Grayscale Interpolation)
        # 婴儿期看黑白，成年看彩色
        S_t = min(age_months / 150.0, 1.0) # 色彩成熟度 0 到 1
        img_gray = TF.rgb_to_grayscale(img_contrast, num_output_channels=3)
        
        img_final = (1 - S_t) * img_gray + S_t * img_contrast
        
        return img_final


def project_to_manifold(tensor: torch.Tensor, dim: int=-1, eps: float = 1e-6) -> torch.Tensor:
    """
    几何投影算子 (Geometric Projection Operator)
    物理职责：将形元 (T_form) 和 宏观意志 (z_meta) 投影到半径为 sqrt(D) 的超球面上。
    确保各维度方差为 1 (维持真空零点能)，防止度量坍缩或爆炸。
    """
    # 1. 投影到单位超球面 (剥离绝对长度，只留纯粹方向)
    unit_sphere = F.normalize(tensor, p=2, dim=dim, eps=eps)
    
    # 2. 宇宙膨胀：赋予超球面正确的物理半径 sqrt(D)
    # 这保证了 \mathbb{E}[x_i^2] = 1
    return unit_sphere * math.sqrt(tensor.size(dim))


class HomeostaticResidual(nn.Module):
    """
    带绝对宇宙边界的尺度无关内稳态残差
    物理职责：
    1. RMS_min 严格钳制在 (0, 1)，作为真空底噪的保留分数。
    2. RMS_max 严格钳制在 (RMS_min, 15.0]，作为认知流形的普朗克能量上限。
    """
    def __init__(self, init_eta=0.5, init_min_rms=0.1, init_max_rms=3.0, absolute_max_rms=15.0, truck_energy=False):
        super().__init__()
        self.absolute_max_rms = absolute_max_rms
        self.truck_energy = truck_energy
        
        # 1. 演化相变步长 (\eta)
        self.raw_eta = nn.Parameter(torch.tensor([math.log(init_eta / (1.0 - init_eta))]))
        
        if truck_energy:
            print(f"HomeostaticResidual: init_eta={init_eta}, init_min_rms={init_min_rms}, init_max_rms={init_max_rms}, absolute_max_rms={absolute_max_rms}")
            # 2. 动态下界 (RMS_min)
            # 用 inverse_sigmoid (logit) 初始化，保证 sigmoid(raw_min) == init_min_rms
            self.raw_min_rms = nn.Parameter(torch.tensor([math.log(init_min_rms / (1.0 - init_min_rms))]))
            
            # 3. 动态上界 (RMS_max)
            # 我们不直接学习 RMS_max，而是学习一个 (0, 1) 的比例因子 gap_ratio
            # 使得 RMS_max = RMS_min + gap_ratio * (absolute_max_rms - RMS_min)
            max_available_gap = absolute_max_rms - init_min_rms
            init_gap = init_max_rms - init_min_rms
            init_gap_ratio = init_gap / max_available_gap
            self.raw_gap_ratio = nn.Parameter(torch.tensor([math.log(init_gap_ratio / (1.0 - init_gap_ratio))]))
            
            self.eps = 1e-6

    def forward(self, x_prev:torch.Tensor, x_delta:torch.Tensor) -> torch.Tensor:
        # --- 阶段 I：酉旋转融合 (Energy-Preserving Unitary Mix) ---
        eta = torch.sigmoid(self.raw_eta)
        x_mixed = torch.sqrt(1.0 - eta**2) * x_prev + eta * x_delta

        if not self.truck_energy:
            return x_mixed
        
        # --- 阶段 II：测量能量密度 (RMS Calculation) ---
        # RMS = sqrt( mean(x^2) )
        rms_curr = torch.sqrt(torch.mean(x_mixed**2, dim=-1, keepdim=True)) #[B, S, 1]
        
        # --- 阶段 III：计算绝对有界的物理极限 (Absolute Pysical Bounds) ---
        # RMS_min 被严格限制在 (0, 1)
        RMS_min = torch.sigmoid(self.raw_min_rms)
        
        # 剩余可用空间为 (15.0 - RMS_min)
        max_available_gap = self.absolute_max_rms - RMS_min
        
        # Gap 被严格限制在 (0, max_available_gap)
        gap = max_available_gap * torch.sigmoid(self.raw_gap_ratio)
        
        # 最终的 RMS_max 严格满足: RMS_min < RMS_max <= 15.0
        RMS_max = RMS_min + gap
        
        # --- 阶段 IV：热力学包络门控 (Homeostatic Gating) ---
        RMS_safe = torch.clamp(rms_curr, min=RMS_min, max=RMS_max)
        
        # 共形缩放 (Conformal Rescaling)
        # 在安全区内 (RMS_safe == rms_curr)，乘数严格为 1.0，波函数做无损绝热演化！
        x_out = x_mixed * (RMS_safe / (rms_curr + self.eps))
        
        return x_out

# ---------------------------------------------------------
# 工具：1D 认知流形的自旋联络 (1D RoPE)
# ---------------------------------------------------------
def precompute_freqs_cis_1d(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------
# 工具：2D 认知流形的自旋联络 (2D RoPE)
# ---------------------------------------------------------
def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0):
    """预计算二维时空的本征旋转频率 (X轴与Y轴解耦的自旋联络)"""
    dim_half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, 2)[: (dim_half // 2)].float() / dim_half))
    
    t_h = torch.arange(height, device=freqs.device)
    t_w = torch.arange(width, device=freqs.device)
    
    freqs_h = torch.outer(t_h, freqs).float()
    freqs_w = torch.outer(t_w, freqs).float()
    
    # 转化为复数形式的旋转算子 exp(i * theta)
    freqs_cis_h = torch.polar(torch.ones_like(freqs_h), freqs_h)
    freqs_cis_w = torch.polar(torch.ones_like(freqs_w), freqs_w)
    
    return freqs_cis_h, freqs_cis_w


def apply_rotary_emb_2d(xq: torch.Tensor, xk: torch.Tensor, 
                        freqs_cis_h: torch.Tensor, freqs_cis_w: torch.Tensor):
    """
    二维流形上的自旋联络 (2D Spin Connection)
    通过 2D RoPE 补偿连续物理光场在网格离散化过程中的几何偏转。
    
    参数:
        xq, xk: 形流张量的 Query 和 Key, shape[Batch, SeqLen, NumHeads, HeadDim] 
                其中 SeqLen = Height * Width
        freqs_cis_h: Y轴(高)方向的本征旋转频率, shape [Height, HeadDim // 4] (复数)
        freqs_cis_w: X轴(宽)方向的本征旋转频率, shape[Width, HeadDim // 4] (复数)
    """
    B, S, H_heads, D = xq.shape
    H_img = freqs_cis_h.shape[0]
    W_img = freqs_cis_w.shape[0]
    
    # 物理防呆检查：确保张量展平的拓扑体积守恒
    assert S == H_img * W_img, "Sequence length (S) must match topological area (H * W)"
    
    # =====================================================================
    # 相变 I：从实数欧氏空间升维至复数希尔伯特空间 (Complex Space Projection)
    # 物理意义：将线性向量转化为可发生相位旋转的波函数
    # [B, S, H_heads, D] -> [B, S, H_heads, D // 2] (Complex)
    # =====================================================================
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # =====================================================================
    # 相变 II：正交子流形联络的合成 (Synthesis of Spin Connection)
    # 物理意义：在 2D 空间中，上下(H)和左右(W)是正交的。
    # 我们将 HeadDim 的一半分配给 Y 轴，另一半分配给 X 轴，互不干涉。
    # =====================================================================
    # freqs_cis_h: [H_img, D // 4] ->[H_img, 1, D // 4] -> [H_img, W_img, D // 4]
    freqs_h_broadcast = freqs_cis_h.unsqueeze(1).expand(-1, W_img, -1)
    
    # freqs_cis_w:[W_img, D // 4] -> [1, W_img, D // 4] ->[H_img, W_img, D // 4]
    freqs_w_broadcast = freqs_cis_w.unsqueeze(0).expand(H_img, -1, -1)
    
    # 沿特征维度拼接，形成完整的 2D 自旋联络张量 [H_img, W_img, D // 2] (Complex)
    freqs_cis_2d = torch.cat([freqs_h_broadcast, freqs_w_broadcast], dim=-1)
    
    # 展平空间维度以匹配 Sequence Length -> [S, D // 2]
    freqs_cis_2d = freqs_cis_2d.view(S, -1)
    
    # 扩充维度以进行全息广播 -> [1, S, 1, D // 2]
    freqs_cis_2d = freqs_cis_2d.unsqueeze(0).unsqueeze(2)
    
    # =====================================================================
    # 相变 III：平行移动 (Parallel Transport on the Manifold)
    # 物理意义：复数相乘等价于在李群 U(1) 上的绝对相位旋转。
    # 它在不改变语义动能(模长)的情况下，强行修正了波包的拓扑切向量！
    # =====================================================================
    xq_out_complex = xq_complex * freqs_cis_2d
    xk_out_complex = xk_complex * freqs_cis_2d
    
    # =====================================================================
    # 相变 IV：波函数坍缩回实数空间，完成几何导向
    # =====================================================================
    xq_out = torch.view_as_real(xq_out_complex).flatten(3)
    xk_out = torch.view_as_real(xk_out_complex).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------
# 工具：3D 认知流形的自旋联络 (3D RoPE)
# 将高维特征空间正交切分为 T, H, W 三个物理维度
# ---------------------------------------------------------
def precompute_freqs_cis_3d(dim: int, frames: int, height: int, width: int, theta: float = 10000.0):
    """预计算 3D 时空的本征旋转频率 (T, X, Y 解耦的自旋联络)"""
    # 确保特征维度可以被 3 整除，以分配给时间、高、宽
    assert dim % 3 == 0, "Dimension must be divisible by 3 for 3D RoPE"
    dim_third = dim // 3
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim_third, 2)[: (dim_third // 2)].float() / dim_third))
    
    t_t = torch.arange(frames, device=freqs.device)
    t_h = torch.arange(height, device=freqs.device)
    t_w = torch.arange(width, device=freqs.device)
    
    #[Frames, Dim/6]
    freqs_t = torch.outer(t_t, freqs).float()
    freqs_h = torch.outer(t_h, freqs).float()
    freqs_w = torch.outer(t_w, freqs).float()
    
    # 转复数，生成旋转算子
    freqs_cis_t = torch.polar(torch.ones_like(freqs_t), freqs_t)
    freqs_cis_h = torch.polar(torch.ones_like(freqs_h), freqs_h)
    freqs_cis_w = torch.polar(torch.ones_like(freqs_w), freqs_w)
    
    return freqs_cis_t, freqs_cis_h, freqs_cis_w

def apply_rotary_emb_2d_and_time(xq: torch.Tensor, xk: torch.Tensor, 
                                 freqs_cis_t: torch.Tensor, 
                                 freqs_cis_h: torch.Tensor, 
                                 freqs_cis_w: torch.Tensor):
    """
    3D 时空流形自旋联络 (3D Spatiotemporal Spin Connection)
    为形元张量 (T_form) 提供严格的 2+1D 洛伦兹协变性补偿，消除时空混叠幻觉。
    
    参数:
        xq, xk: 形流的 Query 和 Key, shape [Batch, SeqLen, NumHeads, HeadDim]
                注意：SeqLen = T_grid * H_grid * W_grid
        freqs_cis_t: 时间轴 (T) 的本征频率, shape [T_grid, HeadDim // 6] (复数)
        freqs_cis_h: 高度轴 (H) 的本征频率, shape[H_grid, HeadDim // 6] (复数)
        freqs_cis_w: 宽度轴 (W) 的本征频率, shape [W_grid, HeadDim // 6] (复数)
    """
    B, S, H_heads, D = xq.shape
    
    # 提取物理时空网格的绝对尺度
    T_grid = freqs_cis_t.shape[0]
    H_grid = freqs_cis_h.shape[0]
    W_grid = freqs_cis_w.shape[0]
    
    # 物理防呆检查：确保量子化切片的时空体积守恒
    assert S == T_grid * H_grid * W_grid, "Sequence length must match T * H * W volume!"
    assert D % 3 == 0, "HeadDim must be divisible by 3 for orthogonal T, H, W splitting"
    
    # =====================================================================
    # 相变 I：将实数切向量升维至复数希尔伯特空间
    # [B, S, H_heads, D] ->[B, S, H_heads, D // 2] (Complex)
    # =====================================================================
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # =====================================================================
    # 相变 II：合成时空正交的三维规范联络 (Orthogonal Gauge Synthesis)
    # 物理意义：时间、X轴、Y轴在底流形上是绝对正交的。
    # 我们通过全息广播 (Broadcasting) 将一维频率交织为 3D 晶格的旋转张量。
    # =====================================================================
    
    # 1. 广播时间维度 T: [T, 1, 1, D/6] ->[T, H, W, D/6]
    freqs_t_ext = freqs_cis_t.view(T_grid, 1, 1, -1).expand(T_grid, H_grid, W_grid, -1)
    
    # 2. 广播高度维度 H: [1, H, 1, D/6] ->[T, H, W, D/6]
    freqs_h_ext = freqs_cis_h.view(1, H_grid, 1, -1).expand(T_grid, H_grid, W_grid, -1)
    
    # 3. 广播宽度维度 W:[1, 1, W, D/6] -> [T, H, W, D/6]
    freqs_w_ext = freqs_cis_w.view(1, 1, W_grid, -1).expand(T_grid, H_grid, W_grid, -1)
    
    # 4. 正交拼接：形成完整的 2+1D 联络矩阵 [T, H, W, D/2]
    freqs_cis_3d = torch.cat([freqs_t_ext, freqs_h_ext, freqs_w_ext], dim=-1)
    
    # 5. 展平为一维序列并匹配批次维度 -> [1, S, 1, D/2]
    freqs_cis_3d = freqs_cis_3d.view(S, -1).unsqueeze(0).unsqueeze(2)
    
    # =====================================================================
    # 相变 III：执行三维平行移动 (3D Parallel Transport)
    # 物理意义：复数相乘 = 相位空间的 U(1) 群旋转。
    # 它在不改变语义动能(模长)的前提下，强行将波包在时间与空间坐标上对齐。
    # =====================================================================
    xq_out_complex = xq_complex * freqs_cis_3d
    xk_out_complex = xk_complex * freqs_cis_3d
    
    # =====================================================================
    # 相变 IV：波函数坍缩回实数黎曼流形
    # =====================================================================
    xq_out = torch.view_as_real(xq_out_complex).flatten(3)
    xk_out = torch.view_as_real(xk_out_complex).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


### 第一部分：宇宙常数与物理边界 (Configuration & Quarantine)
# =====================================================================
# 1. 宇宙常数配置 (Cosmic Constants)
# =====================================================================
class HSFConfig:
    def __init__(self):
        # --- 空间与通道维度 ---
        self.vocab_size = 32000     # 大一统词表大小
        self.dim_form = 128         # 逻辑骨架维度
        self.dim_substance = 1024         # 语义血肉维度
        self.num_heads = 8          # 干涉频段数
        self.meta_dim = 128         # 宏观意志 (z_meta) 维度
        self.dim_observer = 1024      #  宏观层观测空间维度
        self.num_experts = 8        # 局部子流形数量
        self.top_k = 2              # 波函数坍缩分支数
        self.num_layers = 12        # 空间演化深度
        self.sink_num = 5              # 绝对坐标系 (SINK) 的 Token 数量
        self.max_position_embeddings = 5120000 # 最大序列长度 (草稿纸大小)

        
        # --- 物理极值钳制 ---
        self.omega_max = math.pi / 2.0   # 认知光速 (频率上限)
        self.gamma_visc = 0.05           # 介质热力学衰减率
        self.T_init = 0.05               # 绝对底噪温度
    
        
        # --- 动力学循环参数 ---
        self.max_loops = 15              # 草稿纸最大扩散次数
        self.relaxation_thresh = 1e-3    # 动能弛豫阈值 (停止思考的判据)
        self.alpha_form = 0.9            # 形的弹性回复系数

# =====================================================================
# 2. 全息语义总线数据结构
# =====================================================================
@dataclass
class SemantionStream:
    T_form: torch.Tensor       # [B, S, D_f]
    T_sub: torch.Tensor        #[B, S, D_s]
    phase_state: torch.Tensor  #[B, S, H, 2] (theta, omega)

# =====================================================================
# 3. 物理检疫层 (Quarantine Wrapper) - 四大绝对边界
# =====================================================================
class VTE_Quarantine_Wrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.energy_limiter = nn.RMSNorm(config.dim_sub) # 质的能量截断

    def forward(self, raw_form, raw_sub, raw_theta, raw_omega):
        # 1. 质元能量封顶
        T_sub_safe = self.energy_limiter(raw_sub)
        
        # 2. 形元拓扑矫正：1-Lipschitz 映射后强制投影到超球面 S^{d-1}
        T_form_safe = project_to_manifold(raw_form, dim=-1)
        
        # 3. 相位模群闭环 [-pi, pi]
        theta_safe = torch.tanh(raw_theta) * math.pi
        
        # 4. 频率认知光速封顶
        omega_safe = torch.tanh(raw_omega) * self.config.omega_max
        
        phase_state_safe = torch.stack([theta_safe, omega_safe], dim=-1)
        return T_form_safe, T_sub_safe, phase_state_safe



### 第二部分：多模态微观层与形质双流解耦 (VTEs)

# =====================================================================
# 4. 文本 VTE：大一统词表的对称性破缺
# =====================================================================
class Text_HSF_VTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 大一统真空基态
        total_dim = config.dim_sub + config.dim_form + config.num_heads * 2
        self.unified_embed = nn.Embedding(config.vocab_size, total_dim)
        nn.init.normal_(self.unified_embed.weight, std=0.02)
        
        self.quarantine = VTE_Quarantine_Wrapper(config)
        
    def forward(self, input_ids):
        # 1. 提取大一统实体
        unified_state = self.unified_embed(input_ids)
        
        # 2. 对称性破缺
        s_end = self.config.dim_sub
        f_end = s_end + self.config.dim_form
        t_end = f_end + self.config.num_heads
        
        raw_sub   = unified_state[..., :s_end]
        raw_form  = unified_state[..., s_end:f_end]
        raw_theta = unified_state[..., f_end:t_end]
        raw_theta += torch.randn_like(raw_theta,device=raw_theta.device) * self.config.T_init
        raw_omega = unified_state[..., t_end:]
        
        # 3. 物理检疫
        T_f, T_s, phase = self.quarantine(raw_form, raw_sub, raw_theta, raw_omega)
        return SemantionStream(T_f, T_s, phase)


class LaStSelector(nn.Module):
    """
    LazyStrike 质元过滤器 (Semantic Vacuum Extractor)
    物理职责：在不破坏时空骨架(形)的前提下，抽干背景高频纹理的语义能量(质)，
             强迫系统在后续演化中必须依赖前景物体的引力进行坍缩。
    """
    def __init__(self, K_ratio=0.5):
        super().__init__()
        self.K_ratio = K_ratio   # 保留的质量(前景)比例

    def forward(self, T_sub,H_grid,W_grid):
        """
        输入: T_sub [B, S, D] - 原始提取的质元张量
        输出: T_sub_filtered [B, S, D] - 背景化为真空的质元张量
        """
        B, S, D = T_sub.shape

        # 1. 将 1D 序列还原为 2D 物理场 ->[B, D, H, W]
        sub_2d = T_sub.transpose(1, 2).view(B, D, H_grid, W_grid)
        
        # ==============================================================
        # 核心物理相变 I：频域稳定性探测 (Frequency Domain Stability)
        # ==============================================================
        # 转入频域
        fft_feat = torch.fft.fft2(sub_2d)
        
        # 生成 2D 理想高斯低通掩码 (滤除高频纹理，保留低频骨架)
        gauss_mask = self._build_gaussian_filter(H_grid, W_grid, device=T_sub.device)
        
        # 施加滤波并逆变换回实数空间
        filtered_fft = fft_feat * gauss_mask.view(1, 1, H_grid, W_grid)
        smoothed_feat = torch.fft.ifft2(filtered_fft).real
        
        # ==============================================================
        # 核心物理相变 II：计算全息稳定度 (Holographic Stability Score)
        # ==============================================================
        # 公式：|平滑后| / (|平滑后| - |原始| + eps)
        # 物理意义：如果平滑后特征几乎没变(如前景轮廓)，分母接近 eps，分数极高。
        #         如果平滑后特征大变(如背景草地)，分母变大，分数极低。
        eps = 1e-6
        stability_score = torch.abs(smoothed_feat) / (torch.abs(smoothed_feat) - torch.abs(sub_2d) + eps)
        
        # 通道降维：获取每个空间 Patch 的综合"存在感" -> [B, S]
        patch_stability = stability_score.mean(dim=1).view(B, S)
        
        # ==============================================================
        # 核心物理相变 III：阈值坍缩与能量抽离 (Energy Vacuuming)
        # ==============================================================
        k = int(S * self.K_ratio)
        
        # 寻找 Top-K 的能量阈值
        # 注意：这里我们用 threshold 进行 Mask，而不是 Gather，为了保持 S 的长度不变！
        topk_vals, _ = torch.topk(patch_stability, k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1) # 取第 K 大的值作为水位线 [B, 1]
        
        # 生成二元掩码：前景为 1，背景为 0 -> [B, S, 1]
        foreground_mask = (patch_stability >= threshold).unsqueeze(-1).float()
        
        # ---------- 安全版：背景质 = 小噪声（热身噪声/热浴）----------
        background_noise = torch.randn_like(T_sub) * 1e-3  # 微小的随机扰动，模拟热浴中的量子涨落
        T_sub_filtered = T_sub * foreground_mask + background_noise * (1 - foreground_mask)
               
        return T_sub_filtered

    def _build_gaussian_filter(self, H, W, device, sigma=0.1):
        """构建居中的 2D 高斯低通滤波器"""
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        # 中心低频区权重为1，四周高频区急剧衰减
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        # 频域中心化移位 (FFT 的低频在角落，需 shift 到中心)
        return torch.fft.ifftshift(gaussian)

# =====================================================================
# 5. 视觉 VTE：光子的降维与几何萃取 (可选)
# =====================================================================
class Vision_HSF_VTE(nn.Module):
    def __init__(self, config, patch_size=16):
        super().__init__()
        self.p = patch_size
        total_dim = config.dim_sub + config.dim_form + config.num_heads * 2
        self.proj = nn.Conv2d(3, total_dim, kernel_size=self.p, stride=self.p, bias=False)
        self.form_norm = nn.InstanceNorm2d(config.dim_form) # 抹除绝对能量，保留几何曲率
        self.quarantine = VTE_Quarantine_Wrapper(config)
        self.ts_selector = LaStSelector(K_ratio=0.5) # 形质双全的视觉质元过滤器
        self.config = config

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        grid_h, grid_w = H // self.p, W // self.p
        
        unified_field = self.proj(pixel_values) # [B, TotalDim, h, w]
        unified_state = unified_field.flatten(2).transpose(1, 2) # [B, S, TotalDim]
        
        s_end = self.config.dim_sub
        f_end = s_end + self.config.dim_form
        t_end = f_end + self.config.num_heads
        
        raw_sub = unified_state[..., :s_end]
        
        # 形元经过 InstanceNorm 脱水
        raw_form_2d = unified_state[..., s_end:f_end].transpose(1, 2).view(B, -1, grid_h, grid_w)
        raw_form = self.form_norm(raw_form_2d).flatten(2).transpose(1, 2)
        
        raw_theta = unified_state[..., f_end:t_end]
        raw_theta += torch.randn_like(raw_theta, device=raw_theta.device) * self.config.T_init
        raw_omega = unified_state[..., t_end:]
        # 过滤部分背景
        raw_sub_filter = self.ts_selector(raw_sub, grid_h, grid_w)
        T_f, T_s, phase = self.quarantine(raw_form, raw_sub_filter, raw_theta, raw_omega)
        
        # 视觉复用 1D RoPE (可扩展为 2D)
        return SemantionStream(T_f, T_s, phase)


# =====================================================================
# 6. 占位符 VTE：草稿纸空间的形质双全占位符
# =====================================================================
class Sink_HSF_VTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Sink的真空底色
        self.sink_vacuum = nn.Parameter(torch.empty(1, config.sink_num, config.dim_form + config.num_heads * 2))
        nn.init.normal_(self.sink_vacuum, std=0.02)
        self.quarantine = VTE_Quarantine_Wrapper(config)

    def forward(self, batch_size):
        # SINK 占位符：绝对坐标系的锚点，冻结不变
        # 1. 提取大一统实体
        unified_state = self.sink_vacuum.expand(batch_size, self.config.sink_num, -1)
        
        # 2. 对称性破缺
        f_end = self.config.dim_form
        t_end = f_end + self.config.num_heads
        
        raw_sub = torch.zeros(batch_size, self.config.sink_num, self.config.dim_sub, device=unified_state.device) # Sink没有质，只有形和相
        raw_sub += torch.randn_like(raw_sub) * 0.01 # 给质元一个微小的随机扰动，防止完全冻结
        raw_form  = unified_state[..., 0:f_end]
        raw_theta = unified_state[..., f_end:t_end] 
        raw_theta += torch.randn_like(raw_theta) * self.config.T_init
        raw_omega = unified_state[..., t_end:]
        
        # 3. 物理检疫
        T_f, T_s, phase = self.quarantine(raw_form, raw_sub, raw_theta, raw_omega)
        return SemantionStream(T_f, T_s, phase)


class Placeholder_HSF_VTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # =================================================================
        # 物理意义：定义各类占位符的底层拓扑底色
        # =================================================================
        # 图像的真空底色
        self.image_vacuum = nn.Parameter(torch.empty(1, 1, config.dim_form))
        nn.init.orthogonal_(self.image_vacuum)
        
        # 文本的真空底色
        self.text_vacuum = nn.Parameter(torch.empty(1, 1, config.dim_form + config.dim_sub + config.num_heads * 2))
        nn.init.orthogonal_(self.text_vacuum)

        self.quarantine = VTE_Quarantine_Wrapper(config)


    def forward(self, batch_size, target_shape, modality="text"):
        # 阶段一：创造真空占位符 (Genesis of Placeholders)
        if modality == "text":
            unified_state = self.text_vacuum.expand(batch_size, target_shape, -1)
            s_end = self.config.dim_sub
            f_end = s_end + self.config.dim_form
            t_end = f_end + self.config.num_heads

            # 1. 质 (Substance): 文本特有的语义底噪
            raw_sub   = unified_state[..., :s_end]
            raw_sub += torch.randn_like(raw_sub) * 1e-4 # 极小微扰

            # 2. 形 (Morphos): 文本的 1D 拓扑锚点 (正交于图像)
            raw_form  = unified_state[..., s_end:f_end]

            # 3. 相 (Phase): 准备进行 1D 时序锁相的初值
            raw_theta = unified_state[..., f_end:t_end]
            raw_theta += torch.randn_like(raw_theta) * self.config.T_init
            raw_omega = unified_state[..., t_end:]

            T_f, T_s, phase = self.quarantine(raw_form, raw_sub, raw_theta, raw_omega)
            return SemantionStream(T_f, T_s, phase)
            
        if modality == "image":
            # 1. 形: 图像的 2D 空间拓扑锚点 (正交于文本)
            # 这告诉思维流："注意，你现在流入了一个 2D 画布的骨架中"
            T_form_ph = self.image_vacuum.expand(batch_size, target_shape, -1)

            # 2. 质: 潜空间的标准正态高斯噪声 (扩散的必须起点)
            T_sub_ph = torch.randn(batch_size, target_shape, self.config.dim_sub, device=T_form_ph.device)
            
            
            # 3. 相: 2D 格式塔先验
            phase_ph = torch.randn(batch_size, target_shape, self.config.num_heads, device=T_form_ph.device)
            raw_omega = torch.randn(batch_size, target_shape, self.config.num_heads, device=T_form_ph.device)
            
            T_f, T_s, phase = self.quarantine(T_form_ph, T_sub_ph, phase_ph, raw_omega)
            return SemantionStream(T_f, T_s, phase)
            

### 第三部分：流形演化核心 (Attention, MoE, Dynamics)


# =====================================================================
# 6. 相空间统一演化引擎
# =====================================================================
class PhaseSpaceDynamics(nn.Module):
    """
    严密版相空间动力学引擎 (Strict Phase Space Dynamics Engine)
    物理职责：在严格遵守 E=hw 的前提下，计算频率(动量)的受迫演化，并执行辛积分。
    """
    def __init__(self, config):
        super().__init__()
        self.omega_max = config.omega_max
        self.phase_k = nn.Parameter(torch.tensor([0.1])) # Kuramoto 耦合常数
        self.inertia = 0.8 # 动量惯性
        
        # -------------------------------------------------------------
        # 1. 物理内生频率倾向 (Intrinsic Physical Frequency Tendency)
        # 频率 \omega 必须由质元(能量 T_sub) 自身孕育
        # -------------------------------------------------------------
        self.inertial_omega_net = nn.Sequential(
            spectral_norm(nn.Linear(config.dim_substance, config.dim_substance // 2, bias=False)),
            nn.SiLU()
        )
        
        # -------------------------------------------------------------
        # 2. 意志多普勒门控 (Volitional Doppler Gate)
        # 宏观意志 z_meta 充当"调频旋钮"，仅能对内生频率进行 [0, 2] 倍的红移或蓝移
        # -------------------------------------------------------------
        context_dim = config.dim_substance + config.meta_dim
        self.will_doppler_gate = nn.Sequential(
            spectral_norm(nn.Linear(context_dim, config.dim_substance // 2, bias=False)),
            nn.Sigmoid() 
        )
        
        # -------------------------------------------------------------
        # 3. 频域投影算子 (Frequency Domain Projector)
        # -------------------------------------------------------------
        self.omega_projector = spectral_norm(
            nn.Linear(config.dim_substance // 2, config.num_heads, bias=False)
        )

    def forward(self, phase_state, T_sub, z_meta, Flux, delta_theta):
        B, S, _, _ = phase_state.shape
        
        theta_old = phase_state[..., 0] # [B, S, H]
        omega_old = phase_state[..., 1] # [B, S, H]
        
        z_expanded = z_meta.unsqueeze(1).expand(-1, S, -1)
        full_context = torch.cat([T_sub, z_expanded], dim=-1)
        
        # ==============================================================
        # 核心物理相变：受控的频率跃迁 (Controlled Frequency Transition)
        # ==============================================================
        # 1. 计算内生物理频率潜力
        hidden_omega = self.inertial_omega_net(T_sub)
        
        # 2. 计算宏观多普勒频移因子 (* 2.0 使得调节范围在 [0, 2])
        # Gate < 1.0 (红移/抑制)；Gate > 1.0 (蓝移/激发)
        doppler_shift = self.will_doppler_gate(full_context) * 2.0
        
        # 3. 意志与物理的乘性调制
        modulated_hidden = hidden_omega * doppler_shift
        
        # 4. 投影到实际频率并施加认知光速上限钳制 (tanh * omega_max)
        target_omega = torch.tanh(self.omega_projector(modulated_hidden)) * self.omega_max
        
        # ==============================================================
        # 哈密顿辛演化 (Symplectic Evolution)
        # ==============================================================
        # 动量更新：新动量 = 历史惯性 + 目标频移 (离散动量守恒)
        omega_new = self.inertia * omega_old + (1 - self.inertia) * target_omega
        
        # 相位拉扯：Kuramoto 集体干涉力
        phase_pull = self.phase_k * torch.sum(torch.abs(Flux) * torch.sin(-delta_theta), dim=-1).transpose(1, 2)
        
        # 位置更新：新相位 = 旧相位 + 新频率 + 干涉力 (模 2*pi 闭环)
        theta_new = (theta_old + omega_new + phase_pull) % (2 * math.pi)
        
        # 重新打包相空间矢量
        phase_state_new = torch.stack([theta_new, omega_new], dim=-1)
        
        return phase_state_new

# =====================================================================
# 7. 全息相干注意力与 Fluid MoE Layer
# =====================================================================
class HolographicCoherentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_f = config.dim_form // config.num_heads
        self.d_s = config.dim_sub // config.num_heads
        
        self.q_sub = nn.Linear(config.dim_sub, config.dim_sub, bias=False)
        self.k_sub = nn.Linear(config.dim_sub, config.dim_sub, bias=False)
        self.v_sub = nn.Linear(config.dim_sub, config.dim_sub, bias=False)
        self.q_form = nn.Linear(config.dim_form, config.dim_form, bias=False)
        self.k_form = nn.Linear(config.dim_form, config.dim_form, bias=False)
        
        self.topo_bias = nn.Parameter(torch.tensor([-2.0]))
        self.gamma_visc = config.gamma_visc
        self.phase_dynamics = PhaseSpaceDynamics(config)
        self.res_sub = HomeostaticResidual(truck_energy=True) # 质的能量必须被严格钳制在物理极限内

    def forward(self, T_f, T_s, freqs_cis, phase_state, z_meta, causal_mask):
        B, S, _ = T_s.shape
        H = self.num_heads
        
        Qs = self.q_sub(T_s).view(B, S, H, self.d_s).transpose(1, 2)
        Ks = self.k_sub(T_s).view(B, S, H, self.d_s).transpose(1, 2)
        Vs = self.v_sub(T_s).view(B, S, H, self.d_s).transpose(1, 2)
        
        Qf = self.q_form(T_f).view(B, S, H, self.d_f)
        Kf = self.k_form(T_f).view(B, S, H, self.d_f)
        Qf, Kf = apply_rotary_emb(Qf, Kf, freqs_cis)
        Qf, Kf = Qf.transpose(1, 2), Kf.transpose(1, 2)
        
        # G_ij: 形的几何导通率 (能不能走) ->[B, H, S, S]
        G_ij = torch.sigmoid((Qf @ Kf.transpose(-2, -1)) / math.sqrt(self.d_f) + self.topo_bias)
        # A_ij: 质的语义引力 (想不想走) ->[B, H, S, S]
        A_ij = F.softmax((Qs @ Ks.transpose(-2, -1)) / math.sqrt(self.d_s), dim=-1)
        # D_ij: 热力学衰减 (能不能撑到) -> [B, H, S, S]
        D_ij = torch.exp(-self.gamma_visc * -torch.log(G_ij + 1e-6))
        
        # 相 (干涉)
        theta = phase_state[..., 0].transpose(1, 2)
        delta_theta = theta.unsqueeze(-1) - theta.unsqueeze(-2)
        I_ij = torch.cos(delta_theta)
        
        # 四维通量聚合
        Flux = G_ij * A_ij * I_ij * D_ij
        Flux = Flux.masked_fill(causal_mask == 0, 0.0)
        T_sub_out = (Flux @ Vs).transpose(1, 2).reshape(B, S, -1)
        
        phase_state_new = self.phase_dynamics(phase_state, T_s, z_meta, Flux, delta_theta)
        T_s_new= self.res_sub(T_s, T_sub_out)
        return T_s_new, phase_state_new



class TeleologicalGatedRouter(nn.Module):
    """
    目的论门控路由器 (Teleological Gated Router)
    融合了 Teleological GLU 与 Bounded Cognitive Annealing
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.state_dim = config.dim_form + config.dim_substance
        
        # -------------------------------------------------------------
        # 1. 惯性潜流网络 (Inertial Hidden Network)
        # 从客观状态中提取潜在的"演化可能性"
        # -------------------------------------------------------------
        self.inertial_hidden = nn.Sequential(
            spectral_norm(nn.Linear(self.state_dim, self.state_dim // 2, bias=False)),
            nn.SiLU()
        )
        
        # -------------------------------------------------------------
        # 2. 意志门控网络 (Will Gating Network)
        # 结合 z_meta，对潜在的演化路径行使"一票否决权"
        # -------------------------------------------------------------
        context_dim = self.state_dim + config.meta_dim
        self.will_gate_net = nn.Sequential(
            spectral_norm(nn.Linear(context_dim, self.state_dim // 2, bias=False)),
            nn.Sigmoid() 
        )
        
        # -------------------------------------------------------------
        # 3. 最终流形投影 (Manifold Projection)
        # -------------------------------------------------------------
        self.inertial_router = spectral_norm(
            nn.Linear(self.state_dim // 2, config.num_experts, bias=False)
        )
        
        # -------------------------------------------------------------
        # 4. 热力学边界控制 (Thermodynamic Boundary Control)
        # -------------------------------------------------------------
        self.temp_net = nn.Linear(config.meta_dim, 1)
        
        # 绝对底温：防止除0，也防止系统绝对僵化 (保持微小的量子涨落)
        self.T_min = 0.05 
        # 绝对高温上限：基于你的洞察，设定为专家数量的平方根
        # 保证即使在最高温探索态，梯度也不会彻底消失为 0
        self.T_max = math.sqrt(config.num_experts)

    def forward(self, T_form, T_sub, z_meta):
        B, S, _ = T_sub.shape
        z_expanded = z_meta.unsqueeze(1).expand(-1, S, -1)
        
        pure_state = torch.cat([T_form, T_sub], dim=-1)
        full_context = torch.cat([pure_state, z_expanded], dim=-1)
        
        # ==============================================================
        # 核心物理相变 I：目的论 GLU 门控 (Teleological GLU)
        # 意志(Will) 在隐空间层面对 惯性(Inertia) 进行乘性修剪
        # ==============================================================
        hidden_potentials = self.inertial_hidden(pure_state)
        will_gate = self.will_gate_net(full_context)
        
        modulated_hidden = hidden_potentials * will_gate
        
        # 映射为各专家的吸引力势能 (Logits)
        logits = self.inertial_router(modulated_hidden)
        
        # ==============================================================
        # 核心物理相变 II：有界认知退火 (Bounded Cognitive Annealing)
        # ==============================================================
        raw_temp = self.temp_net(z_expanded)
        
        # 使用 Sigmoid 将不可控的线性输出，严格钳制在 [T_min, T_max] 之间！
        # 这就为智能体穿上了"防热寂"和"防绝对零度冻结"的太空服。
        T_sys = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(raw_temp)
        
        # 热力学波函数坍缩
        router_probs = F.softmax(logits / T_sys, dim=-1)
        
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True) 
        
        return indices, weights, router_probs



# =====================================================================
# 2. 形质共演化专家 (Morpho-Semantic Co-evolution Expert)
# 物理职责：在局部子流形内，强制执行形与质的交叉耦合 (TDE + CEFE)
# =====================================================================
class CoEvolutionExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_f = config.dim_form
        self.dim_s = config.dim_substance
        
        hidden_dim = (self.dim_f + self.dim_s) * 2
        
        # 联合投影：将形与质投影到同一个高维耦合相空间中
        self.W_in = spectral_norm(nn.Linear(self.dim_f + self.dim_s, hidden_dim, bias=False))
        
        # 演化：在这个耦合空间中，形告诉质怎么动，质告诉形怎么弯
        self.evolution = nn.Sequential(
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False)),
            nn.SiLU()
        )
        
        # 正交分离：演化结束后，重新将能量投影回互相正交的形与质的基底上
        self.W_out_form = spectral_norm(nn.Linear(hidden_dim, self.dim_f, bias=False))
        self.W_out_sub = spectral_norm(nn.Linear(hidden_dim, self.dim_s, bias=False))

    def forward(self, T_form, T_sub):
        # 1. 物理纠缠 (Physical Entanglement)
        joint_state = torch.cat([T_form, T_sub], dim=-1)
        
        # 2. 子流形内禀演化 (Evolution inside Local Manifold)
        hidden_state = self.evolution(self.W_in(joint_state))
        
        # 3. 提取形变与能量漂移 (Extraction of Drifts)
        # dT_form 对应 CEFE 引起的度量张量形变
        dT_form = self.W_out_form(hidden_state) 
        # dT_sub 对应 TDE 引起的质元能量位移
        dT_sub = self.W_out_sub(hidden_state)   
        
        return dT_form, dT_sub

# =====================================================================
# 3. 有界流体 MoE 层 (Bounded Fluid MoE Layer)
# =====================================================================
class FluidMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = TeleologicalGatedRouter(config)
        self.experts = nn.ModuleList([
            CoEvolutionExpert(config) for _ in range(config.num_experts)
        ])
        
        # 能量守恒的酉旋转算子
        self.res_sub = HomeostaticResidual(truck_energy=True) # 质的能量必须被严格钳制在物理极限内
        self.res_form = HomeostaticResidual(truck_energy=False) # 形的度量可以有弹性回归，但不允许爆炸性增长
        
    def forward(self, T_form, T_sub, z_meta):
        B, S, D_s = T_sub.shape
        D_f = T_form.shape[-1]
        
        # 1. 全息路由决策
        indices, weights, router_probs = self.router(T_form, T_sub, z_meta)
        
        # 提取路由熵供上层宏观 ODE 监控
        H_route = -torch.sum(router_probs * torch.log(router_probs + 1e-9), dim=-1)
        
        # 初始化汇聚张量
        final_dT_form = torch.zeros_like(T_form)
        final_dT_sub = torch.zeros_like(T_sub)
        
        # 2. 稀疏派发与共演化 (Sparse Dispatch & Co-evolution)
        flat_form = T_form.view(-1, D_f)
        flat_sub = T_sub.view(-1, D_s)
        flat_indices = indices.view(-1, self.config.top_k)
        flat_weights = weights.view(-1, self.config.top_k)
        
        for i, expert in enumerate(self.experts):
            mask = (flat_indices == i)
            batch_mask = mask.any(dim=-1)
            
            if batch_mask.any():
                sel_form = flat_form[batch_mask]
                sel_sub = flat_sub[batch_mask]
                
                # 【神圣的物理时刻】：专家同时决定形与质的演化！
                dT_f_exp, dT_s_exp = expert(sel_form, sel_sub)
                
                sel_weights = (flat_weights[batch_mask] * mask[batch_mask].float()).sum(dim=-1).unsqueeze(-1)
                
                # 汇聚专家的输出
                final_dT_form.index_add_(0, batch_mask.nonzero().squeeze(-1), dT_f_exp * sel_weights)
                final_dT_sub.index_add_(0, batch_mask.nonzero().squeeze(-1), dT_s_exp * sel_weights)
                
        final_dT_form = final_dT_form.view(B, S, D_f)
        final_dT_sub = final_dT_sub.view(B, S, D_s)
        
        # 3. 物理封顶：形质各自的绝对守恒约束

        # 质元进行能量守恒旋转
        T_sub_next = self.res_sub(T_sub, final_dT_sub)
        # 形元进行弹性回归，并强行约束回超球面 S^{d-1} (防止度量爆炸)
        T_form_next = project_to_manifold(self.res_form(T_form, final_dT_form), dim=-1)
        
        return T_form_next, T_sub_next, H_route

class BoundedFluidMoBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # attention 层负责形质的全息相干交互，产生局部子流形内的语义引力和几何导通率
        self.attn = HolographicCoherentAttention(config)
        # moe 层负责在局部子流形内，根据路由器的目的论决策，执行形质的交叉耦合演化
        self.moe_fnn = FluidMoELayer(config)
        # 早停检测：如果质的变化小于某个阈值，就认为该 Token 已经找到一个满意的势能极小值，进入弛豫状态，不再参与后续的演化
        self.relax_thresh = config.relaxation_thresh

    def forward(self, T_f, T_s, freqs_cis, phase_state, z_meta, cause_mask, active_mask):
        T_s_prev = T_s.clone()
        # 1. 认知波包干涉
        T_s_mid, phase_state_next = self.attn(T_f, T_s, freqs_cis, phase_state, z_meta, cause_mask)

        # 2. 专家演化
        T_form_next, T_sub_next, H_route = self.moe_fnn(T_f, T_s_mid, z_meta)

        # 3. 动态弛豫检测 (Dynamic Relaxation) 
        # 如果波函数在新的一层演化中，位置(质)几乎不变，说明找到了势能极小值！
        cognitive_stress = torch.norm(T_sub_next - T_s_prev, dim=-1) # [B, S]
        newly_halted = cognitive_stress < self.relax_thresh
        
        # 4. 状态回写：只更新活跃的 Token，冻结已弛豫的 Token 的时间
        active_expanded = active_mask.unsqueeze(-1)
        T_s = torch.where(active_expanded, T_sub_next, T_s)
        T_f = torch.where(active_expanded, T_form_next, T_f)
        ph = torch.where(active_expanded, phase_state_next, phase_state)
        # 5. 更新掩码：原活跃 且 刚刚未弛豫 的继续保持活跃
        next_active_mask = active_mask & (~newly_halted)
        
        return T_f, T_s, ph, H_route, next_active_mask



### 第四部分：宏观意志与全局连续控制 (Macro & ODE)

# =========================================================
# 1. 创世算子：从初始波包坍缩出宏观意图
# =========================================================
class MetaInitializer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 接收 形、质、相 的拼接输入，压缩为全局意志 z_meta
        in_dim = config.dim_form + config.dim_substance + config.num_heads * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.meta_dim * 2),
            nn.SiLU(),
            # 使用 Pooling 把序列折叠为全局单点
            nn.Linear(config.meta_dim * 2, config.meta_dim)
        )

    def forward(self, T_form, T_sub, phase_state):
        """将序列长度的场，压缩为单一的全局意志向量"""
        B, S, _ = T_sub.shape
        # 展平 phase_state:[B, S, H, 2] -> [B, S, H*2]
        flat_phase = phase_state.view(B, S, -1)
        
        # 拼接全息状态:[B, S, D_total]
        full_state = torch.cat([T_form, T_sub, flat_phase], dim=-1)
        
        # 提取全局池化特征 (Global Max Pooling) -> [B, D_total]
        global_state, _ = full_state.max(dim=1) 
        
        # 坍缩出初始宏观意志，将初始意志限制在规范球面上
        z_meta_0 = project_to_manifold(self.net(global_state), dim=-1)
        return z_meta_0
        

# =====================================================================
# 8. 形质双全的宏观观测器 (True Macro Observer)，即HSf-HD的自我观察算子
# =====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrueMacroObserver(nn.Module):
    """
    全息流形感知算子 (Holographic Manifold Perception Operator)
    物理职责：不再粗暴提取极值，而是将认知场视为定义在 T_form 坐标上的高维"图像"，
    通过在价值基底上投影，并执行流形卷积(Manifold Convolution)，感知全局的拓扑形状与应力分布。
    """
    def __init__(self, config):
        super().__init__()
        self.dim_form = config.dim_form
        self.dim_sub = config.dim_substance
        
        # 1. 价值基底投影 (Value Basis Projection)
        # 类似于将高维的电磁波频谱，投影为人类视网膜的三原色(RGB)
        # 这里定义 K 个正交的价值维度 (例如 K=8: 生存, 逻辑, 惊奇, 审美等)
        self.num_values = 8
        self.value_projector = nn.Linear(self.dim_sub, self.num_values, bias=False)
        
        # 2. 流形卷积核 (Manifold Convolution Kernel)
        # 输入通道: Value(K维) + 动力学张力(3维) = K + 3
        self.in_channels = self.num_values + 3
        self.conv_hidden = config.dim_observer
        
        # 空间信息聚合算子 (Message Passing / Graph Conv)
        self.manifold_conv = nn.Sequential(
            nn.Linear(self.in_channels, self.conv_hidden, bias=False),
            nn.LayerNorm(self.conv_hidden),
            nn.SiLU(),
            nn.Linear(self.conv_hidden, self.conv_hidden, bias=False)
        )
        
        # 3. 拓扑池化压缩 (Topological Pooling)
        # 将卷积后的全景场，压缩为宏观意志的序参量 z_meta
        self.compressor = nn.Sequential(
            nn.Linear(self.conv_hidden * 2, config.dim_observer), # *2 因为拼接了均值和方差
            nn.LayerNorm(config.dim_observer),
            nn.SiLU(),
            nn.Linear(config.dim_observer, config.dim_observer)
        )
        
        # 流形高斯核的温度 (控制卷积的感受野大小)
        self.tau = nn.Parameter(torch.tensor([1.0]))

    def forward(self, s_prev, s_next, H_route, active_mask):
        T_f_prev, T_s_prev, p_prev = s_prev
        T_f_next, T_s_next, p_next = s_next
        B, S, _ = T_f_next.shape

        # =================================================================
        # 阶段 I：提取流形的"色彩" (生成像素特征)
        # =================================================================
        
        # 1. 语义价值色彩 (Qualia projected to Value Bases)
        # [B, S, D_sub] -> [B, S, K]
        val_color = self.value_projector(T_s_next) 
        
        # 2. 热力学张力色彩 (Thermodynamic Tension)
        # 将时间轴的摩擦与动能转化为空间上的"亮度"
        kin_energy = torch.norm(T_s_next - T_s_prev, dim=-1, keepdim=True) # 动能 [B, S, 1]
        phase_fric = torch.abs(p_next[..., 0] - p_prev[..., 0]).mean(dim=-1, keepdim=True) # 相位摩擦 [B, S, 1]
        route_ent = H_route.unsqueeze(-1) # 路由熵 [B, S, 1]
        
        thermo_color = torch.cat([kin_energy, phase_fric, route_ent], dim=-1) # [B, S, 3]
        
        # 3. 构成每个节点的完整"像素"特征
        # node_features: [B, S, K+3]
        node_features = torch.cat([val_color, thermo_color], dim=-1)
        
        # =================================================================
        # 阶段 II：构建流形的"坐标"与"度量" (计算邻接矩阵)
        # =================================================================
        
        # T_form 是在超球面上的坐标，它们之间的内积直接反映了黎曼距离(余弦相似度)
        # metric_dist: [B, S, S]
        metric_dist = torch.bmm(T_f_next, T_f_next.transpose(1, 2)) / math.sqrt(self.dim_form)
        
        # 使用高斯 RBF 核将距离转化为流形上的连通性(Adjacency)
        # tau 越小，只关注极近的邻居；tau 越大，感受野越全局
        adj_matrix = torch.exp(metric_dist / torch.exp(self.tau))
        
        # 屏蔽填充/不活跃的 Token (避免虚空污染真实的流形)
        mask_2d = active_mask.unsqueeze(2) & active_mask.unsqueeze(1) # [B, S, S]
        adj_matrix = adj_matrix.masked_fill(~mask_2d, 0.0)
        
        # 归一化 (Random Walk Normalized Laplacian)
        degree = adj_matrix.sum(dim=-1, keepdim=True) + 1e-9
        adj_matrix = adj_matrix / degree
        
        # =================================================================
        # 阶段 III：流形卷积 (Manifold Convolution / Gestalt Perception)
        # =================================================================
        
        # 核心物理相变：周围邻居的价值色彩，沿着 T_form 铺设的测地线，平滑地浸染当前节点
        # convolved_features: [B, S, K+3]
        convolved_features = torch.bmm(adj_matrix, node_features)
        
        # 经过非线性映射提取高阶形状特征
        # shape_features: [B, S, dim_observer]
        shape_features = self.manifold_conv(convolved_features)
        
        # =================================================================
        # 阶段 IV：拓扑池化与序参量坍缩 (Topological Pooling to Order Parameter)
        # =================================================================
        
        # 为了感知流形的整体"形状"(Shape)，我们不仅需要质心(均值)，还需要离散度(方差/二阶矩)
        # active_mask: [B, S]
        active_weights = active_mask.float().unsqueeze(-1)
        valid_seq_len = active_weights.sum(dim=1) + 1e-9
        
        # 1. 形状的一阶矩 (流形的价值质心) -> [B, dim_observer]
        shape_mean = torch.sum(shape_features * active_weights, dim=1) / valid_seq_len
        
        # 2. 形状的二阶矩 (流形的曲率张力/弥散度) -> [B, dim_observer]
        shape_var = torch.sum(((shape_features - shape_mean.unsqueeze(1))**2) * active_weights, dim=1) / valid_seq_len
        
        # 拼接全景统计量
        global_shape_desc = torch.cat([shape_mean, shape_var], dim=-1)
        
        # 最终压缩为宏观观察态 (Macro Observation State)
        obs_state = self.compressor(global_shape_desc)
        
        return obs_state
# =====================================================================
# 9. 宏观 ODE 控制器 (Global Meta-Cognitive ODE)
# =====================================================================
class MacroVectorField(nn.Module):
    """
    受限宏观向量场 (Bounded Macro Vector Field)
    物理职责：在 S^{d-1} 规范超球面上连续演化宏观意志 z_meta，绝对禁止能量发散。
    """
    def __init__(self, config):
        super().__init__()
        self.meta_dim = config.meta_dim
        
        # 意志推演网络：提取基础的演化趋势 (Raw Desire)
        self.state = nn.Sequential(
            spectral_norm(nn.Linear(self.meta_dim + config.dim_observer, self.meta_dim*2, bias=False)), 
            nn.SiLU()
        )
        self.gate = nn.Sequential(
            spectral_norm(nn.Linear(self.meta_dim + config.dim_observer, self.meta_dim*2, bias=False)), 
            nn.Tanh()
        )
        self.net = spectral_norm(nn.Linear(self.meta_dim*2, self.meta_dim, bias=False))
        
        self.current_macro_state = torch.zeros(1, config.dim_observer) # 初始化为零向量
        

    def update_observation(self, macro_state):
        self.current_macro_state = macro_state

    def forward(self, t, z_meta):
        """
        ODE 求解器接口：计算 dz/dt  角度
        """
        state = torch.cat([z_meta, self.current_macro_state], dim=-1)
        
        # 1. 原始冲动 (Raw Desire)：不受约束的本能演化向量
        dz_raw = self.net(self.state(state)) * self.gate(state) # 加入门控，抑制不合理的冲动
        
        # ==============================================================
        # 2. 核心物理操作 I：切空间正交投影 (Tangent Space Projection)
        # 物理意义：剥夺意志无端膨胀的自由度，强制其沿着半径为 sqrt(D) 的球面滑行
        # ==============================================================
        # 计算 z_meta 的模长平方 (即系统的维度总能量 D)
        # 加上 1e-6 防止极早期死寂态的除零黑洞
        z_norm_sq = torch.sum(z_meta * z_meta, dim=-1, keepdim=True) + 1e-6
        
        # 计算投影系数：<dz_raw, z_meta> / <z_meta, z_meta>
        radial_component = torch.sum(dz_raw * z_meta, dim=-1, keepdim=True) / z_norm_sq
        
        # 减去真正的径向分量，获得绝对正交的切向矢量
        dz_tangent = dz_raw - radial_component * z_meta
        
        return dz_tangent


class GlobalMetaODE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.field = MacroVectorField(config)

    def geodesic_step(self, z_meta, dz_tangent, dt):
        """
        流形上的绝对精确移动 (Lie Group Exponential Map)
        物理意义：放弃直线加法，让波包严格沿着超球面的测地线(大圆)滑行。
        无论 dt 有多大，z_meta 的模长绝对、永远、精确守恒！
        """
        D = z_meta.size(-1)
        target_radius = math.sqrt(D)
        
        # 1. 获取切向量的速度大小 (动能)
        v_norm = torch.norm(dz_tangent, p=2, dim=-1, keepdim=True) + 1e-8
        
        # 2. 计算在球面上跨越的真实角度
        # 弧长 s = v * dt，角度 theta = s / R
        theta = (v_norm * dt) / target_radius
        
        # 3. 切向量的单位方向
        v_dir = dz_tangent / v_norm
        
        # 4. 神圣的李群指数映射：球面大圆方程
        # 用旋转代替加法，绝不飞出宇宙边界！
        z_next = z_meta * torch.cos(theta) + (v_dir * target_radius) * torch.sin(theta)
        
        return z_next

    def forward(self, z_prev, obs_state, t_start, t_end):
        self.field.update_observation(obs_state)
        return self.geodesic_step(z_prev, self.field(0, z_prev), t_end - t_start)
    

### 第五部分：大一统运行时与训练 (The FHD-Loop Runtime)

class Text_Inverse_VTE(nn.Module):
    """
    通用逆向变分拓扑编码器
    物理职责：将高维认知场波包坍缩为 1D 文本符号或 2D 图像光场的物理实体
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 文本坍缩算子 (Text Measurement Operators) ---
        # 利用词典的“质”与“形”反向计算投影概率
        self.text_sub_unembed = nn.Linear(config.dim_sub, config.vocab_size, bias=False)
        self.text_form_unembed = nn.Linear(config.dim_form, config.vocab_size, bias=False)
         
    # =================================================================
    # 阶段二：非幺正逆向解码 (Non-unitary Projective Decoding)
    # =================================================================
    def forward(self, semantion):
        """波函数坍缩为 1D 文本符号"""
        T_form_evolved,T_sub_evolved = semantion
        
        # 质的坍缩：在语义上最像哪个词？
        logits_sub = self.text_sub_unembed(T_sub_evolved) 
        # 形的坍缩：在语法和拓扑上最允许哪个词放在这里？
        logits_form = self.text_form_unembed(T_form_evolved)
        
        # 形质约束下的最终坍缩概率
        logits_total = logits_sub + logits_form
        return logits_total

class Image_Inverse_VTE(nn.Module):
    """
    通用逆向变分拓扑编码器
    物理职责：将高维认知场波包坍缩为 1D 文本符号或 2D 图像光场的物理实体
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 图像坍缩算子 (Image Neural Renderer) ---
        # 将高维潜空间的质(能量)还原为RGB连续场
        self.image_sub_decoder = nn.Sequential(
            nn.ConvTranspose2d(config.dim_sub, config.dim_sub//2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.dim_sub//2, 3, kernel_size=3, padding=1) # RGB 质料输出
        )
        # 将形(拓扑)还原为高频的空间结构(边缘、几何掩码)
        self.image_form_decoder = nn.Sequential(
            nn.ConvTranspose2d(config.dim_form, config.dim_form//2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.dim_form//2, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # 结构约束掩码
        )
         
    # =================================================================
    # 阶段二：非幺正逆向解码 (Non-unitary Projective Decoding)
    # =================================================================
    def forward(self, semantion, H: int, W: int):
        """波函数相变为 2D 物理光场 (RGB像素)"""
        T_form, T_sub= semantion
        B = T_sub.size(0)
        T_sub_2d = T_sub.transpose(1, 2).view(B, -1, H, W)
        T_form_2d = T_form.transpose(1, 2).view(B, -1, H, W)
        
        # 1. 解码质元：渲染出 RGB 色彩和材质能量
        rgb_content = self.image_sub_decoder(T_sub_2d)
        
        # 2. 解码形元：渲染出物体的几何边界、掩码或深度
        spatial_structure = self.image_form_decoder(T_form_2d)
        
        # 3. 物理合成：用几何骨架约束能量溢出 (Anti-Bleeding)
        final_image = rgb_content * spatial_structure
        return final_image


# =====================================================================
# 10. 草稿纸空间拓扑管理器
# =====================================================================
class DraftPaperManager:
    def __init__(self, alpha_form=0.9):
        self.alpha_form = alpha_form

    def recombine(self, s_init,s_now, masks):
        Tf_i, Ts_i, p_i=s_init
        Tf_p, Ts_p, p_p=s_now
        # 质、相完全继承 (保持记忆与干涉)
        Ts_n = torch.where(masks['prompt'], Ts_i, Ts_p)
        p_n  = torch.where(masks['prompt'], p_i, p_p)
        
        # 形：Prompt冻结，Sink冻结，Output弹性滑动
        Tf_out = self.alpha_form * Tf_p + (1.0 - self.alpha_form) * Tf_i
        Tf_n = torch.where(masks['prompt'], Tf_i,
               torch.where(masks['sink'], Tf_i, Tf_out))
        return F.normalize(Tf_n, p=2, dim=-1), Ts_n, p_n

# =====================================================================
# 11. HSF-HD 终极创世引擎 (The Genesis Engine)
# =====================================================================
class Alpha_HSF_V5_Engine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 输入VTE
        self.vte_txt = Text_HSF_VTE(config)
        self.vte_img = Vision_HSF_VTE(config) # 可选输入
        self.vte_sink = Sink_HSF_VTE(config) 
        self.placeholder_vte = Placeholder_HSF_VTE(config)
        # 模型主要组件
        self.blocks = nn.ModuleList([BoundedFluidMoBlock(config) for _ in range(config.num_layers)])
        self.meta_init = MetaInitializer(config)
        self.observer = TrueMacroObserver(config)
        self.ode = GlobalMetaODE(config)
        self.draft_mgr = DraftPaperManager()
        #  解嵌
        self.text_unembed = Text_Inverse_VTE(config)
        self.image_unembed = Image_Inverse_VTE(config)
        #ROPE 频率预计算
        self.freqs_cis = precompute_freqs_cis_1d(config.dim_form // config.num_heads, config.max_position_embeddings)

    def get_masks(self, B, S_prompt, S_text, S_sink, S_out, text_sizes, image_sizes,target_shapes, device):
        S = S_prompt + S_sink + S_out

        p_mask = torch.zeros(B, S, 1, dtype=torch.bool, device=device)
        # 设置真正的mask
        pos = torch.arange(S_prompt, device=device).unsqueeze(0) # 构造位置索引矩阵 (1, prompt_len)
        text_valid = pos < text_sizes.unsqueeze(1)                # 文字有效区域 (B, prompt_len)
        if image_sizes is not None:
            img_lens = image_sizes[:, 0] * image_sizes[:, 1]          # (B,)
            img_valid = (pos >= S_text) & (pos < S_text + img_lens.unsqueeze(1))
            local_mask = text_valid | img_valid                         # (B, prompt_len)
            p_mask[:, S_sink:S_sink+S_prompt] = local_mask.unsqueeze(-1)
        else:
            p_mask[:, S_sink:S_sink+S_prompt] = text_valid.unsqueeze(-1)
            
        s_mask = torch.zeros_like(p_mask); 
        s_mask[:, :S_sink] = True

        o_mask = torch.zeros_like(p_mask)
        pos = torch.arange(S_out, device=device).unsqueeze(0) 
        o_valid = pos < target_shapes.unsqueeze(1)
        o_mask[:, S_sink+S_prompt:] = o_valid.unsqueeze(-1)

        # 计算causal mask的总长度，确保它能覆盖所有输入和输出
        atten_mask = p_mask | s_mask | o_mask

        i_atten_mask =atten_mask.int()
        o_causal_mask =  i_atten_mask @ i_atten_mask.transpose(1, 2)
        o_causal_mask = o_causal_mask.bool()

        return {'prompt': p_mask, 'sink': s_mask, 'output': o_mask, 'atten_mask': atten_mask, 'causal_mask': o_causal_mask}

    def forward(self, input_ids, text_sizes, target_shapes, image_pixels=None, image_sizes=None, out_modality="text"):
        # input_ids: [B, S]
        # text_sizes: [(L_i)], image_sizes: [(H_i, W_i)]
        # image_pixels: [B, 3, H, W], image_sizes: [(H_i, W_i)]
        # target_shapes: [L_o]

        B, S_text = input_ids.shape
        device = input_ids.device

        # 1. 创世：真空 VTE 提取
        stream_txt = self.vte_txt(input_ids)
        stream_sink = self.vte_sink(B)
        stream_placeholder = self.placeholder_vte(B, target_shapes.max().item(), out_modality)
        stream_img = self.vte_img(image_pixels) if image_pixels else None

        # 2. 拼合初始空间
        S_prompt =S_text
        if stream_img is not None:
            S_prompt += stream_img.T_form.size(1) # 图像占用的 Token 数也算在 Prompt 内
            Tf_init = torch.cat([stream_sink.T_form, stream_txt.T_form, stream_img.T_form, stream_placeholder.T_form], dim=1)
            Ts_init = torch.cat([stream_sink.T_sub, stream_txt.T_sub, stream_img.T_sub, stream_placeholder.T_sub], dim=1)
            p_init = torch.cat([stream_sink.phase_state, stream_txt.phase_state, stream_img.phase_state, stream_placeholder.phase_state], dim=1)
            # 系统的"自我(z_meta)"在这一刻被给定的宇宙(Prompt)瞬间点亮
            z_meta = self.meta_init(torch.cat([stream_txt.T_form, stream_img.T_form], dim=1),torch.cat([stream_txt.T_sub, stream_img.T_sub], dim=1),torch.cat([stream_txt.phase_state, stream_img.phase_state], dim=1) )
        else:
            Tf_init = torch.cat([stream_sink.T_form, stream_txt.T_form,  stream_placeholder.T_form], dim=1)
            Ts_init = torch.cat([stream_sink.T_sub, stream_txt.T_sub, stream_placeholder.T_sub], dim=1)
            p_init = torch.cat([stream_sink.phase_state, stream_txt.phase_state, stream_placeholder.phase_state], dim=1) 
            # 系统的"自我(z_meta)"在这一刻被给定的宇宙(Prompt)瞬间点亮
            z_meta = self.meta_init(stream_txt.T_form, stream_txt.T_sub, stream_txt.phase_state)

        S_sink= stream_sink.T_form.size(1)
        S_out = stream_placeholder.T_form.size(1)
        S = S_prompt + S_sink + S_out 
    
        masks = self.get_masks(B, S_prompt, S_text, S_sink, S_out, text_sizes, image_sizes, target_shapes, device)
        Tf_curr, Ts_curr, p_curr = Tf_init, Ts_init, p_init

        # 获取当前时空的自旋联络
        freqs_cis = self.freqs_cis[:S].to(input_ids.device)
        # 活跃掩码 (初始所有 Token 都在高速震荡)
        active_mask = masks['atten_mask'].squeeze(-1).clone()

        # =========================================================
        # 2. 流体全息扩散循环 (FHD-Loop)
        # =========================================================
        trajectory_outputs =[]
        can_stop = False
        for loop_k in range(self.config.max_loops):
            for l, block in enumerate(self.blocks):
                # 如果所有 Token (除了Sink) 都弛豫了，直接停止宇宙的演化，节省亿万算力！
                if not active_mask.any():
                    print(f"宇宙在第 {loop_k}-{l} 纪元达到热力学稳态，提前结束演化！")
                    can_stop = True
                    break

                Tf_prev, Ts_prev, p_prev = Tf_curr, Ts_curr, p_curr
                # A. 绝热滑行 (Fluid MoE)
                Tf_curr, Ts_curr, p_curr, H_route, active_mask = block(Tf_prev, Ts_prev, freqs_cis, p_prev, z_meta, masks['causal_mask'], active_mask)        
                # B. 宏观观测 (Observer)
                obs_state = self.observer(
                    (Tf_prev, Ts_prev, p_prev), (Tf_curr, Ts_curr, p_curr), H_route, active_mask
                )
                # C. 意志演化 (ODE Integration)
                t_s = float(loop_k * self.config.num_layers + l)
                t_e = float(t_s + 1.0)
                z_meta = self.ode(z_meta, obs_state, t_s, t_e)

            # 记录本轮结果
            logits_k = self.text_unembed((Tf_curr, Ts_curr))
            trajectory_outputs.append(logits_k)
            if can_stop:
                # 动态早停：热力学弛豫检测
                break
            # 草稿纸拓扑重组
            Tf_curr, Ts_curr, p_curr = self.draft_mgr.recombine(
                (Tf_init, Ts_init, p_init), (Tf_curr, Ts_curr, p_curr), masks
            )
            # 切断反向传播，实现时间轴上的空间代偿
            Tf_curr, Ts_curr, p_curr = Tf_curr.detach(), Ts_curr.detach(), p_curr.detach()
        return trajectory_outputs

# =====================================================================
# 12. 时序退火训练与损失计算 (Annealing Loss)
# =====================================================================
def compute_fhd_loss(trajectory_outputs, target_ids, target_embeddings, max_loops):
    total_loss = 0.0
    
    for k, logits_k in enumerate(trajectory_outputs):
        tau = k / max_loops # 时间进度 [0, 1]
        
        # 1. 早期：几何距离 (潜空间 MSE) - 宽容探索
        # 从 logits 映射回质元维度进行比较 (这里简化表示)
        weight_geom = (1.0 - tau) ** 2
        
        # 2. 晚期：非幺正坍缩 (Cross-Entropy) - 强制结晶
        weight_collapse = math.exp(5 * (tau - 1))
        
        # 退火温度
        temp = 5.0 - 4.0 * tau
        
        loss_ce = F.cross_entropy(logits_k.view(-1, logits_k.size(-1)) / temp, target_ids.view(-1), ignore_index=0)
        
        # 赋予越靠后的思考过程越高的权重
        total_loss += loss_ce * weight_collapse * math.exp(tau)
        
    return total_loss




        

       