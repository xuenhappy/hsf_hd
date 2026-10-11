#!/usr/bin/env python3
"""One-time, explicitly authored computational rewrite of the theoretical spine.

Original figure source and every existing label remain available. Legacy section
labels become chapter-level compatibility anchors, recorded in the manifest.
"""
from pathlib import Path
import json
import re

R = Path(__file__).resolve().parents[1]
_manifest = json.loads((R / 'docs/chapter-manifest.json').read_text())
if any(row['status'] == 'computational-rewrite' for row in _manifest['chapters']):
    raise SystemExit('Core rewrite already applied; edit chapter files directly.')
def put(path, text):
    p = R / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + '\n')

put('frontmatter/preface.tex', r'''
\chapter*{前言：从智能结构到状态动力学}
\addcontentsline{toc}{chapter}{前言：从智能结构到状态动力学}
\markboth{前言}{从智能结构到状态动力学}
本书研究一个贯穿智能理论与工程的问题：一个具有有限资源的系统，如何在持续交互中组织当前活动、积累历史、改变结构，并维持目标导向的行动？我们将这一问题表述为智能状态的计算动力学。

我们保留全息语义场与层级动力学（HSF-HD）的名称。“语义场”表示在关系结构上分布并相互作用的语义状态；“全息”表示局部活动受到整体上下文约束，并能够参与重建任务相关的整体信息；“层级动力学”表示不同作用范围与更新速度的过程相互耦合。这些术语是计算建模的约定，不预设认知过程遵循某种量子或引力方程。

本书的理论位置介于基础系统科学与智能运行时架构之间。系统论、协同论、耗散结构理论和复杂系统理论提供组织、反馈、开放性与多尺度演化的视角，认知科学提供记忆、注意、控制和自我模型的经验约束。HSF-HD 将这些约束组织为可以分析和验证的模型族，AgentOS 则将模型实例化为记忆管理、状态监控、资源调度、工具交互与持续学习机制。

我们把“智能的一般空气动力学”作为研究方向的类比：如同研究飞行需要区分运动状态、结构、边界和控制，研究智能也需要区分当前活动、持久结构、输入输出、控制策略与资源预算。类比帮助提出问题，结论仍须由明确假设下的数学推导和实现中的观测支持。

全书保留三卷十一篇。第一卷建立结构与属性的表示，第二卷研究状态、记忆、控制和多尺度耦合，第三卷比较不同系统并讨论工程实现。几何方法是可选的表示工具，物理学语言用于说明载体约束或提供辅助直觉。计算量、信息量、状态范数和实际能耗分别定义，不能相互替代。

我们使用“定义”建立模型语言，使用“假设”声明适用条件，使用“命题”给出能够推导的结果。尚缺验证的机制保持为模型假说。一个方程可以在数学上成立而在经验上不适用；一个原型可以执行而尚未具有理论预期的性质。这些区分构成理论走向工程的必要步骤。

本书的目标不是用一种术语覆盖所有智能现象，而是建立一套可以逐步修正的共同语言，使状态演化、结构学习和目标控制能够在同一框架中被讨论、实现与检验。
\hfill XuEn
''')
put('frontmatter/literary-preface.tex', r'''
\chapter*{序言：结构、活动与持续生成}
\addcontentsline{toc}{chapter}{序言：结构、活动与持续生成}
\label{chp:prelude_geometry}
我们理解智能，不能只看某一时刻的输出，也不能只看保存下来的知识。当前活动决定系统此刻能够处理什么，历史影响它如何解释输入，结构规定可达的路径，目标决定哪些路径值得探索。智能的连续性来自这些过程的共同演化。

河道与流水可以提供一个有限的直觉：结构约束活动，活动在一定条件下改变结构。我们借助这个意象说明相互作用，但不会由它推断思维具有水的物性。计算模型必须进一步给出状态变量、更新规则和边界条件。

同样，自我可以被讨论为系统维护的连续约束与自我模型。其是否对应主观体验，需要认知科学与哲学层面的独立论证。我们不从形式相似性推导意识，也不从模型中的几何轨迹推导个体生命之外的持续存在。

本书由此出发：把富有启发性的意象转化为能够检验的机制，把宏观解释连接到局部更新，把持续生成连接到有限资源下的具体运行。
''')
put('frontmatter/reading-guide.tex', r'''
\chapter*{阅读约定：对象、符号与论证层级}
\addcontentsline{toc}{chapter}{阅读约定：对象、符号与论证层级}
\label{front:reading_conventions}
我们首先区分四类内部变量：当前状态 $\mathbf{x}$、持久结构 $S$、历史及中间记忆 $\mathbf{m}$、目标与价值状态 $\mathbf{v}$。输入为 $\mathbf{u}$，控制为 $\mathbf{c}$，动作为 $\mathbf{a}$。符号的具体维度与空间在各模型中声明，不能仅凭名称执行运算。

\begin{table}[htbp]\centering
\begin{tabularx}{\textwidth}{l X}
\toprule 对象 & 使用约定 \\\midrule
状态强度 & 由范数或指定统计量测量，不自动具有能量单位。\\
信息熵 & 对指定概率分布定义；特征向量本身不是概率。\\
认知温度 & 若用于 softmax，表示尺度参数；若用于相图，须重新定义观测量。\\
稳定性 & 区分有界、平衡点稳定、渐近收敛及任务成功。\\
相位与干涉 & 可作为周期变量和带符号耦合的计算机制，不预设量子实现。\\
几何与拓扑 & 必须声明空间、度量、边界与离散化，不由类比自动获得。\\
\bottomrule\end{tabularx}\end{table}

在使用连续时间模型时，我们研究的是计算状态的连续近似；实际系统可以采用离散步、事件触发或异步更新。各尺度的步长和延迟需要单独给出。

\section*{插图与历史符号的解释}
我们保留原有插图及其可编辑源文件。图中的“波”“场”“势能”“宏观意志”分别可用于辅助理解状态传播、分布式表示、目标偏置与元控制；它们不构成相应物理规律在认知系统中成立的证据。几何示意图不保证空间真的满足图中描绘的正交、距离或曲率。

部分历史图和专门模型仍使用 $\Psi$ 表示认知活动场。该符号只在图或局部模型中有效；一般模型使用 $\mathbf{x}$ 或 $\phi$，AgentOS 的自我使用 $\Psi_{self}$，并可在 AgentOS 专用语境中简写为 $\Psi$。历史的 TCE 控制方程与 AgentOS 的 Temperature \& Control Engine 不使用同一对象身份。

\section*{证明、模型与实现}
我们仅在条件明确且推导成立时赋予命题数学保证。边界条件、噪声、离散步长或结构更新发生变化后，原结论需要重新检查。专门模型中的“守恒”“收敛”和“等价”不能直接推广到任意智能系统。物理载体的功耗、散热和通信延迟作为外部实现约束测量。
''')
put('introduction/introduction.tex', r'''
\chapter*{绪论：智能状态动力学的研究对象}
\addcontentsline{toc}{chapter}{绪论：智能状态动力学的研究对象}
\markboth{绪论}{智能状态动力学的研究对象}
\label{intro:scope}

\section*{问题与理论层级}
我们研究智能系统在有限资源下的持续交互过程。系统论使我们区分整体、部件和环境；协同论提示我们寻找能够组织局部活动的低维变量；耗散结构理论提示我们注意开放交换与持续维持；复杂系统理论提供非线性、网络、分岔与跨尺度视角；认知科学约束记忆、注意、控制和自我模型的具体解释。上述来源提供研究原则，而不是直接给出唯一的智能方程。

HSF-HD 承担中间层的工作：明确状态空间，提出更新机制，推导有条件的性质，建立可观测指标。AgentOS 承担实例化工作：将状态、历史、结构、控制和资源约束落实为可运行的记忆与调度系统。不同架构可以实现同一个动力学模型族，同一个架构也可以承载不同模型。

\section*{最小模型族与对象类型}
令 $\mathbf{x}\in\mathbb{R}^{d_x}$ 表示快速活动状态，$\mathbf{m}\in\mathbb{R}^{d_m}$ 表示历史及中间记忆，$S\in\mathcal{S}$ 表示持久结构，$\mathbf{v}\in\mathbb{R}^{d_v}$ 表示目标与偏好状态。若结构是矩阵，则 $\mathcal{S}$ 可以是约束矩阵集合；若结构是图，则其离散改变由事件更新定义。下面的微分形式只用于结构具有连续参数化的情形：
\begin{align}
\dot{\mathbf{x}}&=F(\mathbf{x},\mathbf{m},S,\mathbf{v},\mathbf{u},\mathbf{c}),\\
\dot{\mathbf{m}}&=\varepsilon_m H(\mathbf{x},\mathbf{m},S,\mathbf{v}),\\
\dot{S}&=\varepsilon_S G(\mathbf{x},\mathbf{m},S,\mathbf{v}),\\
\dot{\mathbf{v}}&=\varepsilon_v Q(\mathbf{x},\mathbf{m},S,\mathbf{v},\mathbf{u}),\\
\mathbf{a}&=R(\mathbf{x},\mathbf{m},S,\mathbf{v}).
\end{align}
这些等式定义一个接口，而不是宣称所有智能系统具有相同的右端函数。向量场在工作域内局部 Lipschitz，可提供局部解的存在唯一性；若需要全局解，还须控制增长或证明状态留在有界不变域中。$\varepsilon_m,\varepsilon_S,\varepsilon_v$ 指定相对于快速状态的更新速率，只有满足尺度分离条件时才可采用准静态近似。

输入 $\mathbf{u}$ 必须经过编码进入状态空间，控制 $\mathbf{c}$ 由观测和策略生成。输出动作还需通过环境转移与感知返回，形成闭环。观测不完整、通信延迟和计算预算都可能改变闭环性质。

\section*{从闭环到验证}
给定模型后，我们首先检查维度和定义域，再研究有界性、平衡点、收敛和扰动响应。随后将模型离散化，测量实现中的活动强度、记忆误差、更新代价、任务表现与干预效果。任务成功不由低状态范数自动推出，内部预测改善也不由输出流畅自动推出。

三卷依次讨论表示、演化与工程。第一卷的形质分解建立对象；第二卷的传播、学习与控制建立过程；第三卷的比较与实现检查这些机制在不同载体上的成立程度。我们保留数学工具的多样性，但用统一的对象类型和可验证条件约束它们。
''')

chapters = {
'chp:dual_basis': ('智能表示的双重基底：结构与属性', r'''
我们将智能表示分解为结构与属性，以区分信息之间的组织关系和关系节点上的具体内容。这是一个建模选择，不是关于宇宙实体必然由两种本原组成的断言。
\section{结构、属性与绑定}
设 $V$ 为有限位置集合，$S$ 描述位置之间的关系，$q_i\in\mathbb{R}^d$ 为位置 $i$ 上的属性。一个表示为 $\mathcal{R}=(V,S,q)$，其中 $q:V\to\mathbb{R}^d$。若节点数量固定，属性可以堆叠为矩阵 $X\in\mathbb{R}^{|V|\times d}$。结构不等于坐标顺序，属性不等于主观体验。

例如，同样的节点属性置于不同关系图上，可以对应不同事件；同样的关系图填入不同属性，也可以对应不同对象。绑定要求结构索引与属性索引一致，并在变换时共同更新。
\section{表示变换与不变性}
对节点排列矩阵 $P$，我们定义 $S'=PSP^\top$、$X'=PX$。若任务读出满足 $r(PSP^\top,PX)=r(S,X)$，则结果不依赖任意节点编号。这是一项可测试的置换不变性要求，不表示不同语义结构具有相同意义。

结构和属性可以从同一输入联合学习，因此我们不假定二者统计独立。对于编码器 $e(u)=(S,X)$，仅当解码或任务约束充分时，分解才具有可辨识性；否则不同编码器可产生相同输出。
\section{从表示到动力学}
我们采用 $\dot X=F(X,S,u,c)$ 和 $\dot S=\varepsilon_S G(X,S,m,v)$ 区分快速活动与慢速组织变化。第一式在当前结构上运行，第二式改变未来可用的结构。尺度分离是一项需要验证的条件，不由“形”和“质”的名称推出。

形质分解由此提供状态动力学的对象基础：形对应组织关系，质对应局部内容，两者通过绑定和更新规则共同构成任务表示。
'''),
'chp:morphos_skeleton': ('形：关系结构与语义组织', r'''
我们用“形”描述信息之间的组织关系，包括邻接、顺序、层级和约束。不同任务需要不同结构，不要求统一为一种连续流形。
\section{离散结构与连续参数}
在图模型中，结构由节点集合与边集合定义；在固定图上，非负权重 $w_{ij}$ 可以连续更新。对无向图令 $W=W^\top$，$D_{ii}=\sum_jw_{ij}$，图拉普拉斯为 $L=D-W$。由展开平方可得
\begin{equation}z^\top Lz=\frac12\sum_{i,j}w_{ij}(z_i-z_j)^2\ge0.\end{equation}
因此 $L$ 半正定，常量向量属于其零空间。若图连通，零空间仅由常量方向构成。这些结论依赖无向和非负权重；有向或带符号关系需要另外分析。
\section{传播规则由结构约束}
对于节点活动 $x$，模型 $\dot x=-Lx$ 的差异量逐渐减少。在连通固定图上，它趋于初始状态的均值。这个结果描述一种一致化机制，不自动构成语义推理，因为任务相关的差异可能恰恰需要保留。

推理需要加入内容变换、关系类型和目标偏置。例如 $\dot X=-LX+f_S(X,u,c)$，其中 $f_S$ 可以区分因果、包含和时间关系。结构的表达能力与动力学的任务功能必须分别验证。
\section{几何作为可选表示}
当位置存在可微坐标并指定度量时，我们可以使用流形与联络。图距离、嵌入距离和语义距离并非同一对象。只有给出映射和误差约束后，才能以一种距离近似另一种距离。
'''),
'chp:qualia_activation': ('质：属性表示与活动状态', r'''
我们将“质”作为属性与内容的建模术语。颜色、词义特征、置信度和情绪标签都可以作为属性，但其编码值与真实体验不能直接等同。
\section{向量、概率与周期变量}
属性向量 $q\in\mathbb{R}^d$ 不必归一化；概率 $p$ 则满足 $p_i\ge0$、$\sum_i p_i=1$。对 logits $z$ 与尺度 $T>0$，我们定义
\begin{equation}p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.\end{equation}
信息熵 $H(p)=-\sum_i p_i\log p_i$ 对分布定义，不能直接用于任意向量的分量。$T$ 调整分布的相对集中程度，不具有热力学温度单位。

相位 $\theta\in\mathbb{R}/2\pi\mathbb{Z}$ 可表示周期状态，复数形式 $a e^{i\theta}$ 可表达带方向的组合。模平方只有在模型明确规定并归一化时才能解释为概率。
\section{活动强度与资源代价}
状态强度可定义为 $I(q)=\|q\|^2$，执行代价可以定义为运算次数、延迟或实际功耗。两者需要通过实现测量建立关系。我们不会从状态强度增加直接推断硬件能耗按相同比例增加。
\section{有界投影的能力与边界}
对闭凸集合 $C$，投影 $P_C(q)$ 给出最近可行点。若 $C$ 为半径 $r$ 的球，则 $\|P_C(q)\|\le r$。这只保证属性位于指定域内，不保证任务正确、梯度有效或闭环收敛。投影也可能删除任务所需的信息，需通过消融检查。
'''),
'chp:universe_geometry': ('结构化状态空间：图、纤维与张量表示', r'''
我们建立结构化状态空间，以统一表达位置之间的关系和位置上的内容。表示选择由任务与实现决定。
\section{离散纤维表示}
给定位置集合 $V$，每个位置 $i$ 对应属性空间 $F_i$。总空间为 $\mathcal{E}=\bigsqcup_{i\in V}F_i$，投影 $\pi$ 返回位置。截面 $q$ 满足 $\pi(q(i))=i$。若所有 $F_i\simeq\mathbb{R}^d$ 且选定共同坐标，截面可表示为属性矩阵。

这里的离散表示不自动具有连续纤维丛的光滑性质。讨论可微联络或曲率之前，必须补充连续结构或定义离散对应物。
\section{跨位置传输}
令 $A_{ij}:F_j\to F_i$ 为传输映射，局部更新可以写为
\begin{equation}\dot q_i=\sum_{j\in\mathcal{N}(i)}w_{ij}(A_{ij}q_j-q_i)+f_i(q_i,u_i,c_i).\end{equation}
当 $A_{ij}$ 为恒等映射时，该式退化为内容扩散和局部反应。映射具有方向、类型或上下文依赖时，更新能够表达更丰富的关系，但稳定性需要针对具体算子分析。
\section{张量分解的适用范围}
若位置和内容可分离，表示可以写成 $s\otimes q$；一般状态为 $X=\sum_r s_r\otimes q_r$。单个张量积仅描述秩一可分离状态，不能单独证明不可分离的绑定。耦合由多项表示、非线性更新或约束产生，需明确其机制。
'''),
'chp:constitutive_equation': ('构成方程：表示绑定与结构适应', r'''
我们把构成方程定义为结构与内容怎样共同产生任务表示的计算规则。不同实现可以使用不同绑定函数。
\section{绑定函数与类型}
令 $S\in\mathcal S$、$X\in\mathbb R^{n\times d}$，任务表示为 $Z=B(S,X)$。$B$ 可以是关系编码、消息传递或联合注意力，其值域由后续读出规定。若使用 $S\otimes X$，必须说明张量阶数和解码规则。
\section{从目标函数到结构更新}
设 $S$ 暂以欧氏参数向量表示，目标函数为
\begin{equation}\mathcal J(S;X,m,v)=\ell(B(S,X),y)+\lambda\Omega(S)+\beta\mathcal C(S,m,v).\end{equation}
我们要求各项为无量纲或通过系数完成尺度匹配。固定其余变量时，若 $\mathcal J$ 可微，梯度更新为 $\dot S=-\varepsilon_S\nabla_S\mathcal J$。链式法则给出
\begin{equation}\frac{d\mathcal J}{dt}=-\varepsilon_S\|\nabla_S\mathcal J\|^2\le0.\end{equation}
这个结论保证固定目标下的单调性，不保证全局最优。若 $X,m,v$ 同时改变，还存在关于这些变量的偏导项，不能继续无条件宣称单调。
\section{约束与离散实现}
若 $S\in C$，可采用 $S_{k+1}=P_C(S_k-\eta\nabla\mathcal J(S_k))$。对凸可微目标、闭凸集合和适当步长，可进一步分析收敛。图节点或边的新增属于离散事件，需要单独的结构编辑规则及代价，不能由连续梯度公式直接覆盖。
'''),
'chp:holomorphic_isomorphism': ('环境与认知的结构对应：可观测性与近似等价', r'''
我们将环境与内部模型的关系表述为任务相关的结构对应。预测性能可以约束内部表示，但不能单独证明内部空间与整个环境同胚。
\section{任务相关的等价类}
令环境状态为 $z$，允许的动作和观测历史共同确定任务。若两状态在所有允许策略下产生相同的未来任务观测分布，我们将它们记为 $z\sim z'$。这个等价关系依赖任务、传感器和可用干预；改变这些条件后，原等价类可能需要细分。
\section{动力学映射的定义}
考虑确定性特例 $z_{k+1}=T(z_k,a_k)$，内部状态 $x=e(z)$，内部预测转移为 $\widehat T$。严格对应要求
\begin{equation}e(T(z,a))=\widehat T(e(z),a).\end{equation}
这是转移的交换关系。如果 $e$ 非单射，它至多构成任务相关的压缩映射；只有补充双射、逆映射的连续性等条件，才能讨论同胚。随机系统须比较转移核而非单个状态点。
\section{近似误差的传播}
设编码域内的一步误差不超过 $\delta$，$\widehat T$ 对状态为 $L$-Lipschitz，且比较轨迹使用同一动作序列。令 $E_k=\|e(z_k)-x_k\|$，三角不等式给出 $E_{k+1}\le\delta+LE_k$。若 $E_0=0$，则
\begin{equation}E_k\le\delta\sum_{j=0}^{k-1}L^j.\end{equation}
当 $L<1$ 时误差有统一上界；$L\ge1$ 时长期误差可能增长。闭环动作随内部预测改变时，还需计入策略敏感性。本书据此将结构对应视为有条件、可测量的模型性质。
'''),
'chp:evolution_equation_ch': ('状态演化方程：传播、反馈与结构耦合', r'''
我们从明确的更新机制构造状态演化方程，并在给定条件下推导其性质。一般模型使用 $\dot x=F(x,S,m,v,u,c)$，下面给出可分析的线性特例。
\section{局部传播与输入控制}
在固定的无向非负权图上，$L_S$ 为拉普拉斯，$R\succeq0$ 为抑制矩阵，$B$ 与 $U$ 为输入和控制映射。我们定义
\begin{equation}\dot x=-(L_S+R)x+Bu+Uc.\label{eq:hsf_linear_state}\end{equation}
所有项具有相同状态变化率单位。控制 $c$ 可以是反馈，也可以是外部设定；二者的稳定性条件不同。该方程不需要量子或引力假设。
\section{有界响应的推导}
令 $A=L_S+R$，假设 $A=A^\top\succeq\alpha I$，$\alpha>0$。对 $V=\frac12\|x\|^2$，记 $b=Bu+Uc$，可得
\begin{align}\dot V&=-x^\top Ax+x^\top b\\
&\le-\alpha\|x\|^2+\|x\|\|b\|\\
&\le-\frac\alpha2\|x\|^2+\frac{1}{2\alpha}\|b\|^2.
\end{align}
若 $\|b(t)\|\le M$，则由积分因子得到
\begin{equation}V(t)\le e^{-\alpha t}V(0)+\frac{M^2}{2\alpha^2}(1-e^{-\alpha t}).\end{equation}
因此有界输入产生有界状态；输入为零时状态指数衰减。这不意味着任务获得解决，因为任务读出和目标满足性尚未纳入结论。
\section{结构变化与闭环反馈}
若 $A(S(t))\succeq\alpha I$ 在所有时刻统一成立，同一欧氏范数估计仍适用。若采用随结构变化的度量 $P(t)$，则 $V=x^\top P(t)x/2$ 的导数还包含 $x^\top\dot P x/2$，不能省略。

对于反馈 $c=Kx$，有效矩阵为 $A-UK$。必须重新检查其对称部分是否正定；控制幅度更大并不保证更稳定。非线性模型则需要局部线性化、共同 Lyapunov 函数或其他明确条件。
\section{时间离散与步长}
对无输入模型，显式 Euler 更新为 $x_{k+1}=(I-\Delta t A)x_k$。若 $A$ 对称正定，各模态乘子为 $1-\Delta t\lambda_i$，收敛要求 $0<\Delta t<2/\lambda_{\max}(A)$。连续模型稳定不自动意味着任意离散步长稳定。
\section{复数传播的专门模型}
若某实现需要相位组合，可以定义 $\dot\phi=(-iH-D)\phi+j$，其中 $H=H^\dagger$、$D=D^\dagger\succeq0$。直接计算得到
\begin{equation}\frac{d}{dt}\|\phi\|^2=-2\phi^\dagger D\phi+2\operatorname{Re}(\phi^\dagger j).\end{equation}
只有 $D=0$ 且 $j=0$ 时模长守恒。该式是复数线性系统的性质，不赋予认知状态量子测量含义。
'''),
'chp:macro_stability_ig': ('控制稳定性：目标函数、反馈与投影条件', r'''
我们区分状态有界、平衡点稳定、目标函数下降和任务成功。不同性质需要不同条件，不能通过“宏观控制”这一名称统一保证。
\section{梯度流与下降性质}
设 $J:\mathbb R^d\to\mathbb R$ 为可微固定目标，$M(x)$ 为对称正定矩阵，采用 $\dot x=-M(x)\nabla J(x)$。链式法则给出
\begin{equation}\dot J=-\nabla J(x)^\top M(x)\nabla J(x)\le0.\end{equation}
若目标具有唯一极小点、轨迹有界且正定性和连续性满足所需条件，可进一步证明收敛。若目标随时间变化，则 $\dot J$ 还包含 $\partial_tJ$；若存在外部输入，也需要加入相应交叉项。
\section{凸性与离散步长}
设 $J$ 的梯度为 $L$-Lipschitz，对 $x^+=x-\eta\nabla J(x)$ 使用下降引理，得到
\begin{equation}J(x^+)\le J(x)-\eta(1-L\eta/2)\|\nabla J(x)\|^2.\end{equation}
因此 $0<\eta<2/L$ 保证该次更新不增加目标。严格凸性可保证极小点至多一个，但极小点的存在仍需另外条件。任意控制输入不由凸性自动获得稳定性。
\section{Bregman 投影与精确条件}
对严格凸可微函数 $\psi$，定义
\begin{equation}D_\psi(r,p)=\psi(r)-\psi(p)-\langle\nabla\psi(p),r-p\rangle.\end{equation}
设 $C$ 为闭凸集合，$q$ 是 $D_\psi(z,p)$ 在 $C$ 内存在的极小点。最优性给出 $\langle\nabla\psi(q)-\nabla\psi(p),r-q\rangle\ge0$。结合三点恒等式可得
\begin{equation}D_\psi(r,p)\ge D_\psi(r,q)+D_\psi(q,p),\qquad r\in C.\end{equation}
只有交叉项为零时取得等式。$\psi(p)=\sum_i p_i\log p_i$ 在正概率单纯形上产生 KL 散度；此时还需处理归一化约束和边界。该投影保证特定散度的关系，不保证任意任务指标改善。
\section{不完整观测与控制验证}
实际控制器从 $y=O(x)$ 估计状态，观测误差、执行延迟和模型偏差可以破坏理想下降性质。我们通过扰动上界和实际反馈轨迹评估鲁棒性，不将低维观察者视为全状态的无损副本。
'''),
'chp:phenomenological_physics': ('记忆与时间：活动维持、历史积累与结构学习', r'''
我们将记忆区分为当前活动的维持、历史经历的积累和可复用结构的学习。这种区分首先依据更新机制和作用时间，而不是物理相态。
\section{工作状态与有限维持}
令 $x$ 为当前工作状态，模型 $\dot x=-Ax+Bu+Kx$ 表示输入与重入共同维持活动。工作状态能否持续取决于 $-A+K$ 的谱及输入条件。仅使谱接近零可能延长保持时间，也可能放大噪声；持久维持需要控制与资源约束共同分析。
\section{历史记忆与指数积累}
给定状态特征 $\varphi(x)$，我们定义 $\dot m=-\lambda m+\varphi(x)$，$\lambda>0$。积分因子给出
\begin{equation}m(t)=e^{-\lambda(t-t_0)}m(t_0)+\int_{t_0}^t e^{-\lambda(t-s)}\varphi(x(s))\,ds.\end{equation}
这是一种衰减历史摘要，不保留所有经历的完整顺序。需要事件顺序时，可以采用带时间戳的离散事件记录；二者在信息保真度、检索成本与更新延迟上不同。
\section{结构学习与持续适应}
结构更新 $\dot S=-\varepsilon_S\nabla_S\mathcal J(S;m)$ 使用历史证据改变未来处理规则。当前活动、历史积累和结构学习形成闭环；在 AgentOS 中，它们分别由工作记忆 $h(t)$、情景记忆 $E$ 和结构记忆 $\Theta$ 的机制实例化，而不是简单的三个存储位置。
\section{计算时间与结构时间}
计算时间记录更新顺序和延迟，事件时间记录外部发生顺序。若要定义结构变化量，可取
\begin{equation}\tau_S(t)=\int_{t_0}^t\|\dot S(s)\|_W\,ds,\end{equation}
其中范数和参数化必须固定。它描述结构轨迹的累积长度，不天然等于主观时间；参数重写后保持不变需要适当的几何度量。关于时间体验的解释仍需独立经验检验。
\section{体验与自我模型的边界}
我们可以建模自我相关的筛选、价值评价和历史连续性，但不能由这些计算功能直接证明主观感受存在。体验研究需要在可观测行为、报告与内部机制之间建立可反驳的联系。
'''),
'chap:software_control_msos': ('智能运行时：从 HSF-HD 动力学到 AgentOS', r'''
HSF-HD 描述状态、结构、历史与控制的共同演化，AgentOS 将这些对象实现为运行时机制。我们不把某种专用芯片作为该关系成立的前提。
\section{一般模型与运行时对象}
在一般模型中，$x$ 表示当前活动，$m$ 表示历史及中间记忆，$S$ 表示持久结构，$v$ 表示目标与价值状态。AgentOS 分别以 $h(t)$、$E$、$\Theta$ 实例化前三者；自我 $\Psi_{self}$ 参与价值、筛选、历史连续性和自我模型，不等同于单一目标向量。
\section{观测、控制与资源管理}
SPC 对 $h(t)$ 的分布和关系按自我相关方向进行压缩，并把结果重新提供给工作状态；TCE（Temperature \& Control Engine）依据观测调节探索强度、增益、抑制及工作模式。CCM 管理输入和运行态复杂度，Boundary 管理内外交换，Offloading 将可重复的信息处理迁移到外部记忆 $M_e$。

我们将这些机制写成离散闭环：
\begin{align}
y_k&=O_{SPC}(h_k,E_k,\Theta_k,\Psi_{self,k}),\\
c_k&=\pi_{TCE}(y_k,v_k,b_k),\\
h_{k+1}&=F_h(h_k,\Theta_k,E_k,u_k,c_k,r_k),\\
E_{k+1}&=F_E(E_k,h_k,a_k,\Psi_{self,k}),\\
\Theta_{k+1}&=F_\Theta(\Theta_k,E_k,\mathcal E_k).
\end{align}
其中 $b_k$ 为资源预算，$r_k$ 为外部召回，$\mathcal E_k$ 为学习证据。各函数可以在不同频率执行。SPC 的压缩通常有损，控制器需要根据任务检测关键状态是否被遗漏。
\section{调度的可行域与约束}
对待执行任务集合 $\mathcal T_k$，选择任务子集 $A_k$，可定义预算约束
\begin{equation}\sum_{i\in A_k}\widehat C_i\le B_k.\end{equation}
估计代价 $\widehat C_i$ 可以包含上下文占用、时间和工具成本，但多种单位应分别约束或归一化。预算满足只保证调度可行；任务依赖、优先级、误差和中断恢复仍需要额外规则。
\section{外部记忆与边界}
外部记忆包括文件、数据库、程序、工具、环境与其他智能体。卸载只有在未来检索和维护成本可接受时才有利。我们比较保存成本、调用延迟、可靠性与修改代价，而不将一切外化都视为复杂度下降。
\section{实现与验证}
运行时可以构建在现有模型、图结构和普通计算硬件上。DEC 或其他专用载体属于可选实现路线。我们以状态过载率、恢复时间、长期记忆保真度、单位任务成本和控制干预效果验证机制，不以模块名称证明智能能力。
'''),
'sec:section_5204': ('理论与工程约化：从一般动力学到离散网络', r'''
我们通过离散化将一般状态动力学连接到可执行网络。离散模型必须说明所保留的性质与近似误差，而不是仅凭名称宣称与连续模型精确等价。
\section{图模型的构造}
设 $W=W^\top\ge0$，$L=D-W$。对节点活动 $x$，定义
\begin{equation}\dot x=-Lx-\nabla V(x)+Bu.\end{equation}
固定输入为零时，该式是目标函数 $J(x)=x^\top Lx/2+V(x)$ 的梯度流，所以 $\dot J=-\|\nabla J\|^2\le0$。有输入时需加入 $\nabla J^\top Bu$，不能继续无条件宣称下降。
\section{连续近似的条件}
若图来自空间采样，边权按指定规则近似微分算子，可研究采样尺度趋零时的算子一致性。边界、权重尺度和采样分布都会影响极限。任意知识图不自动对应某个连续流形上的拉普拉斯。
\section{实现与复杂度}
稀疏图更新的单步成本与边数和特征维数相关；稠密注意力仍需要序列长度的二次交互。并行或模拟实现可以改变延迟与常数，但不能免除硬件数量、通信、精度和输入输出成本。
\section{工程不变量与实验}
我们先验证维度、边界、掩码与数值稳定，再检查状态范数、目标下降和任务表现。每项保证只在对应条件内成立。参数归一化可提供范围约束，谱约束可限制局部增益，二者均不能独立保证整个智能体正确行动。
'''),
}

manifest = json.loads((R / 'docs/chapter-manifest.json').read_text())
for row in manifest['chapters']:
    p = R / row['path']
    old = p.read_text()
    labels = re.findall(r'\\label\{([^}]+)\}', old)
    primary = next((label for label in labels if label in chapters), None)
    if primary is None:
        continue
    title, body = chapters[primary]
    # Preserve every figure verbatim in a separate, still-compiled TeX module.
    figs = re.findall(r'\\begin\{figure\}(?:\[[^]]*\])?.*?\\end\{figure\}', old, re.S)
    fig_path = f'figures/preserved/{p.stem}.tex'
    figure_labels = set(re.findall(r'\\label\{([^}]+)\}', '\n'.join(figs)))
    aliases = [x for x in labels if x != primary and x not in figure_labels]
    head = '\\chapter{' + title + '}\n\\label{' + primary + '}\n'
    if aliases:
        head += '% Compatibility anchors: old section references resolve to this chapter.\n'
        head += '\n'.join('\\label{' + x + '}' for x in dict.fromkeys(aliases)) + '\n'
    if figs:
        put(fig_path, '\n\n'.join(figs))
        body += r'''
\section{表示与示意图}
我们保留以下示意图以展示原有结构关系。图中历史物理术语遵循阅读约定中的计算解释；图的形式相似性不代替本章的假设、推导和验证。
''' + '\\input{' + fig_path + '}\n'
    put(row['path'], head + body)
    row['new_title'] = title
    row['status'] = 'computational-rewrite'
    row['legacy_section_aliases'] = list(dict.fromkeys(aliases))
    row['preserved_figure_count'] = len(figs)

put('docs/chapter-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
print('Authored prefaces, introduction, reading conventions and', len(chapters), 'core chapters.')
