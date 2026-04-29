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
        self.dim_sub = 1024         # 语义血肉维度
        self.num_heads = 8          # 干涉频段数
        self.meta_dim = 128         # 宏观意志 (z_meta) 维度
        self.num_experts = 8        # 局部子流形数量
        self.top_k = 2              # 波函数坍缩分支数
        self.num_layers = 12        # 空间演化深度
        self.sink_num = 5              # 绝对坐标系 (SINK) 的 Token 数量
        
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
        
        raw_sub   = torch.zeros(batch_size, self.config.sink_num, self.config.dim_sub, device=unified_state.device) # Sink没有质，只有形和相
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
        return math.sqrt(1.0 - eta.item()**2) * old + eta * delta
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

    def forward(self, T_f, T_s, freqs_cis, phase_state, z_meta, cause_mask, active_mask):
        T_s_prev = T_s.clone()
        # 1. 认知波包干涉
        T_s_mid, phase_state_next = self.attn(T_f, T_s, freqs_cis, phase_state, z_meta, cause_mask)
        # 2. 专家演化
        T_form_next, T_sub_next, H_route = self.moe_fnn(T_f, T_s_mid, z_meta)

        # --- 动态弛豫检测 (Dynamic Relaxation) ---
        # 如果波函数在新的一层演化中，位置(质)几乎不变，说明找到了势能极小值！
        cognitive_stress = torch.norm(T_sub_next - T_s_prev, dim=-1) # [B, S]
        newly_halted = cognitive_stress < self.relax_thresh
        
        # 状态回写：只更新活跃的 Token，冻结已弛豫的 Token 的时间
        active_expanded = active_mask.unsqueeze(-1)
        T_s = torch.where(active_expanded, T_sub_next, T_s)
        T_f = torch.where(active_expanded, T_form_next, T_f)
        ph = torch.where(active_expanded, phase_state_next, phase_state)
        # 更新掩码：原活跃 且 刚刚未弛豫 的继续保持活跃
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
            # 使用 Mean Pooling 把序列折叠为全局单点
            nn.Linear(config.meta_dim * 2, config.meta_dim),
            nn.Tanh() # 将初始意志限制在规范球面上
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
        
        # 坍缩出初始宏观意志
        z_meta_0 = self.net(global_state)
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
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.meta_dim * 2, 128), nn.SiLU(),
            nn.Linear(128, config.meta_dim)
        )
        self.obs_state : torch.Tensor  = torch.zeros(1, config.meta_dim) # 占位符，实际在 forward 时动态赋值

    def forward(self, t, z_meta):
        state = torch.cat([z_meta, self.obs_state], dim=-1)
        return self.net(state)

class GlobalMetaODE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.field = MacroVectorField(config)

    def forward(self, z_prev, obs_state, t_start, t_end):
        self.field.obs_state = obs_state
        t_span = torch.tensor([t_start, t_end], dtype=torch.float32, device=z_prev.device)
        # O(1) 显存连续积分
        z_traj = odeint(self.field, z_prev, t_span, method='rk4', options={'step_size':0.1})
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
            z_meta = self.meta_init(stream_txt, stream_txt.T_sub, stream_txt.phase_state)

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
                t_s, t_e = loop_k + l/self.config.num_layers, loop_k + (l+1)/self.config.num_layers
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

# 训练步示例
def train_step(model, optimizer, input_ids, target_ids, target_embeddings):
    optimizer.zero_grad()
    # 包含了完整的 ODE 演化、草稿纸迭代
    trajectory_outputs = model(input_ids)
    
    # 时序退火损失计算
    loss = compute_fhd_loss(trajectory_outputs, target_ids, target_embeddings, model.config.max_loops)
    
    # 伴随算子的 O(1) 反向传播，加持 Sophia-HSF 几何优化器
    loss.backward()
    optimizer.step()
    return loss.item()
