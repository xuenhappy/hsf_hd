import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from torch.nn.utils.parametrizations import spectral_norm
# 需 pip install torchdiffeq
from torchdiffeq import odeint_adjoint as odeint

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
        self.num_experts = 8        # 局部子流形数量
        self.top_k = 2              # 波函数坍缩分支数
        self.num_layers = 12        # 空间演化深度
        self.sink_num = 5              # 绝对坐标系 (SINK) 的 Token 数量
        self.max_position_embeddings = 5120000 # 最大序列长度 (草稿纸大小)

        # --- 记忆 ---
        self.use_memory_bank = False          # 是否启用热力学记忆库
        self.memory_capacity = 10000     # 记忆库容量 (条目数)
        self.memory_gamma=0.01           # 质(能量)的耗散系数
        self.memory_kappa=0.005          # 频(动量)的阻尼系数
        
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
        self.form_projector = spectral_norm(nn.Linear(config.dim_form, config.dim_form, bias=False))

    def forward(self, raw_form, raw_sub, raw_theta, raw_omega):
        # 1. 质元能量封顶
        T_sub_safe = self.energy_limiter(raw_sub)
        
        # 2. 形元拓扑矫正：1-Lipschitz 映射后强制投影到单位超球面 S^{d-1}
        T_form_safe = F.normalize(self.form_projector(raw_form), p=2, dim=-1)
        
        # 3. 相位模群闭环 [-pi, pi]
        theta_safe = torch.tanh(raw_theta) * math.pi
        
        # 4. 频率认知光速封顶
        omega_safe = torch.tanh(raw_omega) * self.config.omega_max
        
        phase_state_safe = torch.stack([theta_safe, omega_safe], dim=-1)
        return T_form_safe, T_sub_safe, phase_state_safe

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
        
        T_f, T_s, phase = self.quarantine(raw_form, raw_sub, raw_theta, raw_omega)
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

class EnergyPreservingResidual(nn.Module):
    def __init__(self, init_eta=0.5):
        super().__init__()
        self.raw_eta = nn.Parameter(torch.tensor([math.log(init_eta / (1 - init_eta))]))
    def forward(self, old, delta):
        eta = torch.sigmoid(self.raw_eta)
        return math.sqrt(1.0 - eta**2) * old + eta * delta
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
        self.res_sub = EnergyPreservingResidual()

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


class CognitiveSurprisalExtractor(nn.Module):
    """
    认知惊奇提取器 (Cognitive Surprisal Extractor)
    物理职责：在推理(Inference)或演化(Evolution)期，实时计算认知场在当前流形局部产生的内生热力学惊奇能量。
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=1.0):
        super().__init__()
        # 物理感受系数 (Sensitivity Coefficients)
        # 将其设为可学习参数，允许系统在重整化群流中自发调节"痛觉"敏感度
        self.alpha = nn.Parameter(torch.tensor(alpha)) # 几何应力敏感度 (质元位移)
        self.beta = nn.Parameter(torch.tensor(beta))   # 相位摩擦敏感度 (Kuramoto 扭转)
        self.gamma = nn.Parameter(torch.tensor(gamma)) # 路由纠结敏感度 (相空间分岔熵)
        self.delta = nn.Parameter(torch.tensor(delta)) # 外部强迫源敏感度 (如预测Loss)

    def forward(self, T_sub_prev, T_sub_next, theta_prev, theta_next, router_probs):
        """
        输入:
            T_sub_prev, T_sub_next:[B, S, D] 演化前后的质元能量态
            theta_prev, theta_next:[B, S, H] 演化前后的本征相位
            router_probs: [B, S, Num_Experts] 目的论路由器的概率分布
            ext_loss: [B, S] (可选) 来自外部环境或 Next-Token 预测的真实物理激波
        输出:
            surprisal_energy: [B, S] 序列中每个 Token 的总惊奇能量密度
        """
        # ==============================================================
        # 1. 几何应力 (Geometric Stress) 
        # 物理意义：思维流包 (质元) 在底流形上演化时发生的动量改变。
        # 改变越大，说明遇到的流形曲率(阻力)越大，做功越多。
        # ==============================================================
        stress_energy = torch.norm(T_sub_next - T_sub_prev, dim=-1) # [B, S]

        # ==============================================================
        # 2. 相位摩擦 (Phase Friction) 
        # 物理意义：在 Kuramoto 锁相动力学中，由于外界强行拉扯导致的相位跃迁。
        # 对应于"顿悟"的火花，或者被强迫改变观念时的"认知撕裂感"。
        # ==============================================================
        phase_friction = torch.abs(theta_next - theta_prev) # [B, S, H]
        phase_energy = phase_friction.mean(dim=-1)          # [B, S]

        # ==============================================================
        # 3. 路由困惑度 (Routing Entropy)
        # 物理意义：衡量在多岔路口(Expert子流形)的纠结程度。
        # 熵越大，说明系统在多种等价路径间发生叠加，这是产生高度创造性(或迷茫)的标志。
        # ==============================================================
        # 加上 1e-9 防止 log(0) 产生数学奇点黑洞
        router_entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-9), dim=-1) # [B, S]

        # ==============================================================
        # 4. 热力学总能量合成 (Thermodynamic Energy Synthesis)
        # 使用 F.softplus 确保所有感受系数为正，满足热力学能量非负的公理
        # ==============================================================
        surprisal_energy = (F.softplus(self.alpha) * stress_energy +
                            F.softplus(self.beta)  * phase_energy +
                            F.softplus(self.gamma) * router_entropy)


        return surprisal_energy


class ThermodynamicMemoryBank(nn.Module):
    """
    高性能时空谐振腔 (High-Performance Thermodynamic Memory Bank)
    物理职责：在固定显存容量下，维护一个环形缓冲区 (Ring Buffer)。
    利用布尔掩码仅提取发生"惊奇相变"的 Token，实现高密度的相空间张量存储与并行召回。
    """
    def __init__(self, config):
        super().__init__()
        self.capacity = config.memory_capacity  # N: 存储的语义子(Token)总数量上限
        self.gamma = config.memory_gamma        # 质(能量)的耗散系数
        self.kappa = config.memory_kappa        # 频(动量)的阻尼系数
        self.shock_threshold = config.shock_threshold
        
        self.dim_form = config.dim_form
        self.dim_sub = config.dim_substance
        self.num_heads = config.num_heads

        # -------------------------------------------------------------
        # 1. 物理容器预分配 (Pre-allocated Physical Containers)
        # 使用 register_buffer 确保它们与模型一起分布在正确的 GPU 上，
        # 且作为模型的持久化状态 (State_Dict) 保存，但不参与梯度更新。
        # -------------------------------------------------------------
        # 形 (逻辑坐标/测地线锚点) -> [Capacity, D_f]
        self.register_buffer('T_form_keys', torch.zeros(self.capacity, self.dim_form))
        # 质 (语义血肉/能量载荷) -> [Capacity, D_s]
        self.register_buffer('T_sub_values', torch.zeros(self.capacity, self.dim_sub))
        # 相空间矢量 (Theta, Omega) ->[Capacity, H, 2]
        self.register_buffer('phase_states', torch.zeros(self.capacity, self.num_heads, 2))
        
        # 热力学元数据
        self.register_buffer('timestamps', torch.zeros(self.capacity, dtype=torch.int64))
        self.register_buffer('importances', torch.zeros(self.capacity, dtype=torch.float32))

        # -------------------------------------------------------------
        # 2. 环形指针与状态元数据
        # -------------------------------------------------------------
        self.register_buffer('mem_ptr', torch.tensor(0, dtype=torch.int64))   # 插入指针
        self.register_buffer('mem_size', torch.tensor(0, dtype=torch.int64))  # 当前有效记忆量
        self.register_buffer('current_time', torch.tensor(0, dtype=torch.int64)) # 绝对物理时钟
        
        self.surprisal_extractor = CognitiveSurprisalExtractor()
        self.layer_norm = nn.LayerNorm(config.dim_substance)
        self.kuramoto_k = 0.2 # 记忆对当下的相位拖拽系数
        
    def tick(self):
        """推进全局物理时钟"""
        self.current_time += 1

    @torch.no_grad() # 记忆的沉积是被动的地质学过程，不保留计算图
    def add_memory(self, S_prev, S_next, router_probs, active_mask):
        """
        动态沉积：筛选超高能量的激波 Token，将其作为孤立子压入环形存储器
        """
        _, T_sub_prev, theta_prev = S_prev
        T_form_next, T_sub_next, phase_state_next = S_next
        theta_next = phase_state_next[..., 0]
        
        # 1. 计算热力学惊奇能量 [B, S]
        surprisal_energy = self.surprisal_extractor(
            T_sub_prev=T_sub_prev, T_sub_next=T_sub_next, 
            theta_prev=theta_prev, theta_next=theta_next, 
            router_probs=router_probs
        )
        
        # 2. 物理相变过滤 (Topological Filtering)
        # 仅保留：活跃状态 (排除Padding/Prompt冻结区) 且 惊奇度击穿阈值
        is_shocking = (surprisal_energy > self.shock_threshold)
        valid_mask = active_mask & is_shocking  # [B, S] 的布尔矩阵
        
        # 如果当前没有任何 Token 产生足够的认知张力，直接返回，不消耗任何显存带宽
        if not valid_mask.any():
            self.tick()
            return
            
        # 3. 语义子降维坍缩 (Flatten to Tokens)
        # 将 [B, S, D] 转换为 [N, D]，N 是本次被选中的高能 Token 总数
        new_keys = T_form_next[valid_mask]             # [N, D_f]
        new_vals = T_sub_next[valid_mask]              # [N, D_s]
        new_phases = phase_state_next[valid_mask]      # [N, H*2]
        new_imps = surprisal_energy[valid_mask]        # [N]
        
        N = new_keys.shape[0]
        curr_time = self.current_time.item()
        new_times = torch.full((N,), curr_time, dtype=torch.int64, device=new_keys.device)

        # 4. 环形缓冲区极速写入 (Ring Buffer Update)
        ptr = self.mem_ptr.item()
        
        if N >= self.capacity:
            # 极端情况：单次涌入激波太多，只保留最新的 capacity 个
            self.T_form_keys.copy_(new_keys[-self.capacity:])
            self.T_sub_values.copy_(new_vals[-self.capacity:])
            self.phase_states.copy_(new_phases[-self.capacity:])
            self.importances.copy_(new_imps[-self.capacity:])
            self.timestamps.copy_(new_times[-self.capacity:])
            self.mem_ptr.fill_(0)
            self.mem_size.fill_(self.capacity)
        else:
            # 正常插入，处理末尾循环(Wrap-around)
            end_ptr = ptr + N
            if end_ptr <= self.capacity:
                self.T_form_keys[ptr:end_ptr] = new_keys
                self.T_sub_values[ptr:end_ptr] = new_vals
                self.phase_states[ptr:end_ptr] = new_phases
                self.importances[ptr:end_ptr] = new_imps
                self.timestamps[ptr:end_ptr] = new_times
            else:
                overflow = end_ptr - self.capacity
                first_part = N - overflow
                
                # 写入尾部
                self.T_form_keys[ptr:] = new_keys[:first_part]
                self.T_sub_values[ptr:] = new_vals[:first_part]
                self.phase_states[ptr:] = new_phases[:first_part]
                self.importances[ptr:] = new_imps[:first_part]
                self.timestamps[ptr:] = new_times[:first_part]
                
                # 写入头部覆盖
                self.T_form_keys[:overflow] = new_keys[first_part:]
                self.T_sub_values[:overflow] = new_vals[first_part:]
                self.phase_states[:overflow] = new_phases[first_part:]
                self.importances[:overflow] = new_imps[first_part:]
                self.timestamps[:overflow] = new_times[first_part:]
                
            self.mem_ptr.fill_(end_ptr % self.capacity)
            self.mem_size.fill_(min(self.capacity, self.mem_size.item() + N))

        self.tick()

    @torch.no_grad() # 记忆的沉积是被动的地质学过程，不保留计算图
    def resonant_recall(self, T_form_now, phase_state_now, top_k=3):
        """
        高性能相干召回
        T_form_now:[B, S, D_f]
        phase_state_now: [B, S, H, 2]
        """
        size = self.mem_size.item()
        if size == 0:
            # 宇宙诞生之初，没有记忆
            B, S, D_s = T_form_now.shape[0], T_form_now.shape[1], self.dim_sub
            return torch.zeros((B, S, D_s), device=T_form_now.device), None, 0.0

        # 1. 提取当前有效流形截面 (Active Manifold Section)
        # 取前 size 个，避免计算 padding zeroes
        K_form = self.T_form_keys[:size]     # [size, D_f]
        V_sub = self.T_sub_values[:size]     # [size, D_s]
        P_state = self.phase_states[:size]   # [size, H, 2]
        T_stamp = self.timestamps[:size]     # [size]
        E_shock = self.importances[:size]    # [size]

        # 2. 测地线寻址：全矩阵一次性乘法，极致并行
        # T_form_now:[B, S, D_f] @ K_form.T:[D_f, size] -> scores:[B, S, size]
        scores = torch.matmul(T_form_now, K_form.transpose(0, 1)) 
        
        actual_top_k = min(top_k, size)
        topk_scores, topk_indices = torch.topk(scores, int(actual_top_k), dim=-1) # [B, S, TopK]
        
        # 几何导通率
        G_dist = F.softmax(topk_scores / math.sqrt(self.dim_form), dim=-1)

        # 3. 提取 TopK 相关记忆的物理属性
        delta_t = self.current_time.item() - T_stamp[topk_indices] # [B, S, TopK]
        delta_t = delta_t.float()
        
        # P_state[topk_indices] ->[B, S, TopK, H, 2]
        topk_phase_states = P_state[topk_indices]
        theta_old = topk_phase_states[..., 0] # [B, S, TopK, H]
        omega_old = topk_phase_states[..., 1] #[B, S, TopK, H]
        
        # 【物理法则一：频的阻尼衰减】
        omega_recalled = omega_old * torch.exp(-self.kappa * delta_t.unsqueeze(-1))
        
        # 【物理法则二：相的时间积分】
        phase_shift = (omega_old / (self.kappa + 1e-6)) * (1 - torch.exp(-self.kappa * delta_t.unsqueeze(-1)))
        theta_recalled = (theta_old + phase_shift) % (2 * math.pi)

        # 4. 干涉计算
        theta_now = phase_state_now[..., 0].unsqueeze(2) #[B, S, 1, H]
        I_mem = torch.cos(theta_now - theta_recalled)    # [B, S, TopK, H]
        
        # 5. 能量坍缩
        decay_factor = torch.exp(-self.gamma * delta_t)  #[B, S, TopK]
        importance_factor = torch.sigmoid(E_shock[topk_indices]) #[B, S, TopK]
        
        # 总通量计算：注意 I_mem 有 H 维度，而 G_dist 没有。
        # 我们需要在 Head 维度上进行精细干涉，因此 Flux 为[B, S, TopK, H]
        Flux = G_dist.unsqueeze(-1) * I_mem * decay_factor.unsqueeze(-1) * importance_factor.unsqueeze(-1)

        # 6. 提取 V_sub[B, S, TopK, D_s] 并按照 Head 分解，应用不同的干涉权重
        B, S, K_top = topk_indices.shape
        D_s = self.dim_sub
        H = self.num_heads
        head_dim = D_s // H
        
        V_sub_topk = V_sub[topk_indices] #[B, S, TopK, D_s]
        # 重塑为 [B, S, TopK, H, head_dim] 以精确匹配每个 Head 的干涉通量
        V_sub_topk_heads = V_sub_topk.view(B, S, K_top, H, head_dim)
        
        # 乘性干涉并对 TopK 路径进行求和融合
        # [B, S, TopK, H, 1] * [B, S, TopK, H, head_dim] -> sum(dim=2) ->[B, S, H, head_dim]
        recalled_T_sub_heads = torch.sum(Flux.unsqueeze(-1) * V_sub_topk_heads, dim=2)
        # 还原回原维度
        recalled_T_sub = recalled_T_sub_heads.reshape(B, S, D_s)
        
        # 重组召回的相空间 [B, S, TopK, H, 2]
        recalled_phase_state = torch.stack([theta_recalled, omega_recalled], dim=-1)

        return recalled_T_sub, recalled_phase_state, Flux



    def forward(self, T_form, T_sub, phase_state_now):
        # =========================================================
        # 无黑盒的纯物理波函数干涉融合 (Wave Superposition)
        # =========================================================
        
        # 从谐振腔中召回历史波包
        recalled_T_sub, recalled_phase_state, Flux = self.resonant_recall(T_form, phase_state_now)
        
        if recalled_phase_state is not None:
            # 1. 质元能量的物理叠加 (直接相加，因为 Flux 中已经包含了 cos 干涉门控)
            # 我们彻底去掉了上一版中多余的 nn.Linear 神经网络 Gate！大自然不需要 Linear！
            T_sub_fused = self.layer_norm(T_sub + recalled_T_sub)
            
            # 2. 相空间矢量的强迫扭转 (Kuramoto Pulling)
            # 回忆的频率 \omega 越高，对你当下的影响就越暴烈
            theta_now, omega_now = phase_state_now[..., 0], phase_state_now[..., 1]
            theta_rec = recalled_phase_state[..., 0] # [B, S, TopK, H]
            omega_rec = recalled_phase_state[..., 1]
            
            # 记忆对当下的拉扯：频率差的逼近
            omega_pull = torch.sum(Flux * (omega_rec - omega_now.unsqueeze(2)), dim=2)
            omega_fused = omega_now + self.kuramoto_k * omega_pull
            
            # 记忆对当下的拉扯：相位的对齐
            theta_pull = torch.sum(Flux * torch.sin(theta_rec - theta_now.unsqueeze(2)), dim=2)
            theta_fused = theta_now + self.kuramoto_k * theta_pull
            
            phase_state_fused = torch.stack([theta_fused, omega_fused], dim=-1)

        else:
            T_sub_fused = T_sub
            phase_state_fused = phase_state_now

        return T_sub_fused, phase_state_fused


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
        self.res_sub = EnergyPreservingResidual(init_eta=0.5)
        self.res_form = EnergyPreservingResidual(init_eta=0.2)

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
        
        # 形元进行弹性回归，并强行约束回单位超球面 S^{d-1} (防止度量爆炸)
        T_form_next = self.res_form(T_form, final_dT_form)
        T_form_next = F.normalize(T_form_next, p=2, dim=-1)
        
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

    def forward(self, T_f, T_s, freqs_cis, phase_state, z_meta, cause_mask, active_mask, memory_bank=None):
        T_s_prev = T_s.clone()
        # 1. 认知波包干涉
        T_s_mid, phase_state_next = self.attn(T_f, T_s, freqs_cis, phase_state, z_meta, cause_mask)
        # 2. 从谐振腔中召回历史波包
        if memory_bank is not None:
            T_s_mid, phase_state_next = memory_bank(T_s_mid, phase_state_next)

        # 3. 专家演化
        T_form_next, T_sub_next, H_route = self.moe_fnn(T_f, T_s_mid, z_meta)

        # 4. 物理时钟推进与高能刻蚀
        if memory_bank is not None:
            memory_bank.add_memory((T_f,T_s_prev,phase_state),(T_form_next, T_sub_next, phase_state_next),H_route,active_mask)

        # 5. 动态弛豫检测 (Dynamic Relaxation) 
        # 如果波函数在新的一层演化中，位置(质)几乎不变，说明找到了势能极小值！
        cognitive_stress = torch.norm(T_sub_next - T_s_prev, dim=-1) # [B, S]
        newly_halted = cognitive_stress < self.relax_thresh
        
        # 6. 状态回写：只更新活跃的 Token，冻结已弛豫的 Token 的时间
        active_expanded = active_mask.unsqueeze(-1)
        T_s = torch.where(active_expanded, T_sub_next, T_s)
        T_f = torch.where(active_expanded, T_form_next, T_f)
        ph = torch.where(active_expanded, phase_state_next, phase_state)
        # 7. 更新掩码：原活跃 且 刚刚未弛豫 的继续保持活跃
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
        z_meta_0 = F.normalize(self.net(global_state), p=2, dim=-1)
        return z_meta_0
        

# =====================================================================
# 8. 形质双全的宏观观测器 (True Macro Observer)
# =====================================================================
class TrueMacroObserver(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = 3 + config.dim_sub + config.dim_form + 1
        self.compressor = nn.Sequential(
            nn.Linear(in_dim, config.meta_dim*2), nn.LayerNorm(config.meta_dim*2), nn.SiLU(),
            nn.Linear(config.meta_dim*2, config.meta_dim)
        )

    def forward(self, s_prev, s_next, H_route, active_mask):
        T_f_prev, T_s_prev, p_prev= s_prev
        T_f_next, T_s_next, p_next = s_next
        # 1. 极值提取
        kin_energy = torch.norm(T_s_next - T_s_prev, dim=-1)
        kin_energy = kin_energy.masked_fill(~active_mask, 0.0)

        max_kin, _ = torch.max(kin_energy, dim=-1)
        phase_fric = torch.abs(p_next[..., 0] - p_prev[..., 0]).mean(dim=-1)
        max_fric, _ = torch.max(phase_fric, dim=-1)
        max_ent, _ = torch.max(H_route, dim=-1)
        thermo_ext = torch.stack([max_kin, max_fric, max_ent], dim=-1) # [B, 3]

        # 2. 能量加权要旨 (Gist)
        weights = F.softmax(kin_energy / 0.1, dim=-1).unsqueeze(-1)
        gist_sub = torch.sum(T_s_next * weights, dim=1)
        gist_form_raw = torch.sum(T_f_next * weights, dim=1)
        
        form_norm = torch.norm(gist_form_raw, dim=-1, keepdim=True)
        geom_dispersion = 1.0 - form_norm
        gist_form = gist_form_raw / (form_norm + 1e-9)

        # 3. 压缩
        macro_in = torch.cat([thermo_ext, gist_sub, gist_form, geom_dispersion], dim=-1)
        return self.compressor(macro_in)

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
        # 强制加入谱范数 (Spectral Norm)，保证内部映射为 1-Lipschitz，防止内生爆炸
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(self.meta_dim *2, self.meta_dim, bias=False)), 
            nn.SiLU(),
            spectral_norm(nn.Linear(self.meta_dim, self.meta_dim, bias=False))
        )
        self.current_macro_state = torch.zeros(1, config.meta_dim)
        
        # 数值漂移校正系数 (Lyapunov Restoring Constant)
        self.kappa_restore = 0.5 

    def update_observation(self, macro_state):
        self.current_macro_state = macro_state

    def forward(self, t, z_meta):
        """
        ODE 求解器接口：计算 dz/dt
        """
        state = torch.cat([z_meta, self.current_macro_state], dim=-1)
        
        # 1. 原始冲动 (Raw Desire)：不受约束的本能演化向量
        dz_raw = self.net(state)
        
        # ==============================================================
        # 2. 核心物理操作 I：切空间正交投影 (Tangent Space Projection)
        # 物理意义：剥夺意志无端膨胀的自由度，强制其沿着球面测地线滑行
        # 数学：dz_dt = dz_raw - proj_z(dz_raw)
        # ==============================================================
        # 计算原始向量在径向 (z_meta方向) 上的投影大小: <dz_raw, z_meta>
        radial_component = torch.sum(dz_raw * z_meta, dim=-1, keepdim=True)
        
        # 减去径向分量，剩下的就是严格与 z_meta 正交的切向分量 (Tangent Vector)
        dz_tangent = dz_raw - radial_component * z_meta
        
        # ==============================================================
        # 3. 核心物理操作 II：李雅普诺夫拓扑回弹 (Topological Restoring Force)
        # 物理意义：ODE 求解器 (如 rk4/dopri5) 在离散步长下会有微小的截断误差，
        # 这会导致 z_meta 随时间极缓慢地"渗漏"出球面。必须施加物理回弹力。
        # ==============================================================
        # 计算当前状态的实际能量 ||z_meta||^2
        current_norm_sq = torch.sum(z_meta * z_meta, dim=-1, keepdim=True)
        
        # 构造负反馈阻尼：如果能量 > 1，产生向内的拉力；如果 < 1，产生向外的推力
        drift_correction = -self.kappa_restore * (current_norm_sq - 1.0) * z_meta
        
        # 最终的受迫演化速度 = 切向真实演化 + 法向误差纠正
        dz_dt_final = dz_tangent + drift_correction
        
        return dz_dt_final


class GlobalMetaODE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.field = MacroVectorField(config)

    def forward(self, z_prev, obs_state, t_start, t_end):
        self.field.update_observation(obs_state)
        t_span = torch.tensor([t_start, t_end], dtype=torch.float32, device=z_prev.device)
        # O(1) 显存连续积分，放弃 fixed step_size，让大自然的曲率自己决定走多快
        # z_traj = odeint(self.field, z_prev, t_span, method='rk4', options={'step_size':0.1})
        z_traj = odeint(
            self.field, z_prev, t_span, 
            method='dopri5',   # 自适应步长
            atol=1e-4, rtol=1e-4 # 当地形崎岖(误差大)时，步长自动缩短到 0.001
        )
        return z_traj[-1]


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
        self.memory_bank = ThermodynamicMemoryBank(config) if config.use_memory_bank else None
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
                Tf_curr, Ts_curr, p_curr, H_route, active_mask = block(Tf_prev, Ts_prev, freqs_cis, p_prev, z_meta, masks['causal_mask'], active_mask, memory_bank=self.memory_bank)        
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




        

       