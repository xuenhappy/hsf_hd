#!/usr/bin/env python3
"""Complete the authored computational rewrite of volume one, preserving figures."""
from pathlib import Path
import json
import re

R = Path(__file__).resolve().parents[1]
_manifest = json.loads((R / 'docs/chapter-manifest.json').read_text())
if all(row['status'] == 'computational-rewrite' for row in _manifest['chapters'] if row['volume'] == 1):
    raise SystemExit('Volume one rewrite already applied; edit chapter files directly.')
CONTENT = {
'chp:physics_anatomy': ('环境对象的结构化表示：测量、关系与约束', r'''
我们从环境中的可观测对象建立结构与属性表示。模型首先服务于感知、预测和行动，不要求复制环境的全部自由度。
\section{测量与状态估计}
令环境隐状态为 $z$，观测为 $o=H(z)+\xi$。传感器给出观测而不是隐状态本身，编码器 $e(o)$ 因而只能恢复任务相关信息。对指定噪声模型可以建立似然 $p(o\mid z)$；在缺少该模型时，我们只能给出经验误差，不能宣称表示无损。

结构 $S$ 可包含邻接、包含、相对位置和事件先后关系；属性 $X$ 可包含测量值、类别特征和置信度。对于位置和速度等具有物理单位的属性，单位必须保留；不能将这些测量值与无量纲语义向量直接相加。
\section{约束、预测与模型误差}
设任务需要预测量 $y$，采用 $\widehat y=r(S,X)$。目标函数可以写成
\begin{equation}\mathcal J=\mathbb E[\ell(\widehat y,y)]+\lambda\mathcal R(S,X),\end{equation}
其中正则项表达结构复杂度或约束违背。误差较小只表明该模型在指定任务和数据分布上有效，不能推出内部表示与真实环境在全部尺度上同构。
\section{物理知识作为特定领域约束}
对真实机械或电路系统，守恒关系、运动方程和单位制可以作为模型约束。这些规律适用于所描述的环境对象；它们不因被智能体学习就成为内部活动的普遍规律。我们通过这种区分保留物理知识的作用，同时避免将环境模型和认知动力学混为一谈。
'''),
'chp:semantic_anatomy': ('认知对象的结构化表示：语言、视觉与关系', r'''
我们将认知表示理解为系统为任务建立的关系与内容组织。语言、视觉和推理可以共享某些操作，但不必具有相同的几何结构。
\section{语言与视觉的关系类型}
语言表示包含词项、依赖关系、语境和事件角色。线性序列保存表达顺序，依赖图补充跨位置关系。视觉表示包含区域、对象、相对位置和外观属性。两类表示都可以使用节点与关系，但各自的节点身份和关系含义需要独立定义。

结构与内容不是预先独立的两组信息。词义会影响句法消歧，对象识别会影响关系判断。我们采用联合编码 $(S,X)=e(o)$，并通过任务损失检验分解是否有用。
\section{消息传递与关系绑定}
在固定位置集合上，设关系类型为 $r$，可以定义
\begin{equation}x_i^{k+1}=f\left(x_i^k,\sum_r\sum_{j\in\mathcal N_r(i)}A_r x_j^k,u_i\right).\end{equation}
映射 $A_r$ 将关系类型转化为内容变换。加和要求各消息属于同一空间，函数 $f$ 则规定更新的非线性形式。该表达只是机制接口；是否能完成推理必须通过组合泛化、关系干预和错误分析检查。
\section{语义一致性与可验证范围}
两个表示产生相同输出，不代表内部机制相同。我们需要分别检查对象身份、关系保真、上下文变化和读出一致性。对于主观体验，本章只讨论可编码内容，不能以向量相似度判定体验相同。
'''),
'chp:domain_manifolds_heterogeneity': ('领域表示的异构性：任务尺度与跨域接口', r'''
我们将不同领域视为具有不同变量、关系、时间尺度和观测限制的模型空间。通用性来自可组合接口与适应机制，不来自强行采用同一表示。
\section{领域状态与局部适用性}
令领域 $d$ 的状态为 $x_d\in\mathcal X_d$，更新为 $x_d^+=F_d(x_d,u_d)$。文本、场景图、程序状态和运动控制可能分别采用序列、图、离散状态机与连续向量。每种空间具有自己的合法操作和约束。
\section{跨域映射与误差}
映射 $E_{ab}:\mathcal X_a\to\mathcal X_b$ 应声明任务相关的保真量。我们可以检验转移的一致性误差
\begin{equation}\delta_{ab}(x,u)=d_b\left(E_{ab}(F_a(x,u)),F_b(E_{ab}(x),U_{ab}(u))\right),\end{equation}
其中 $U_{ab}$ 转换输入，$d_b$ 是目标域的指定距离。两域操作维度不同或信息不足时，误差可能无法消除。任务相关的近似对应不等同于整个领域之间的严格同构。
\section{多尺度接口与调度}
局部控制、语义推演和长期规划具有不同截止时间。接口需要规定采样率、最大延迟、误差预算和失败后的处理方式。低频过程可以施加持久约束，但其影响范围必须由架构和耦合通道确定，不能由低频本身推出。
'''),
'chp:mst_architecture': ('MST 架构：结构与内容的联合注意力', r'''
我们提出形质联合 Transformer（MST）作为结构与属性分解的一种工程实例。架构假设必须通过实验检验，它不是 HSF-HD 唯一可能的实现。
\section{双通道与类型约束}
设输入包含 $n$ 个对齐位置，结构通道为 $T\in\mathbb R^{n\times d_s}$，内容通道为 $X\in\mathbb R^{n\times d_x}$。两通道可以使用不同编码器，但位置对应关系必须由数据或对齐模块定义。未经对齐的两个序列不能直接按同一节点索引融合。
\section{结构引导的内容聚合}
取 $Q=TW_Q$、$K=XW_K$、$V=XW_V$，其中 $Q,K\in\mathbb R^{n\times d_k}$，$V\in\mathbb R^{n\times d_v}$。定义
\begin{equation}A=\operatorname{softmax}_{row}\left(\frac{QK^\top}{\sqrt{d_k}}+M(S)\right),\qquad Z=AV.\end{equation}
掩码 $M(S)$ 表示允许关系，不允许的边可设为负无穷。每一行至少有一个合法位置，才能避免归一化未定义。$A$ 是聚合权重，不是结构与内容在物理意义上的纠缠。
\section{目标与训练}
给定结构目标 $S^*$、属性目标 $X^*$ 和任务目标 $y$，可以定义
\begin{equation}\mathcal L=\lambda_s\ell_s(\widehat S,S^*)+\lambda_x\ell_x(\widehat X,X^*)+\lambda_y\ell_y(r(Z),y)+\lambda_b\ell_b(T,X).\end{equation}
绑定损失 $\ell_b$ 的正负样本需要符合任务语义；错误的负样本会惩罚合法组合。不同损失项应按尺度设置权重，训练损失下降不保证分解可辨识或域外泛化。
\section{复杂度与对照实验}
稠密 $QK^\top$ 仍具有二次交互成本。稀疏关系可以减少实际交互数量，但必须说明稀疏模式和实现。我们以单通道基线、双通道无绑定、关系扰动和跨域测试区分结构分解带来的收益与参数增加带来的收益。
'''),
'chp:generative_creation': ('结构化生成：从输入描述到可执行结果', r'''
我们把生成任务表述为在输入条件下构造满足约束的结构与内容，并通过读出转化为可执行或可验证的结果。
\section{生成变量与条件}
给定描述 $u$，系统生成 $Y=(S,X)$，目标可以写为
\begin{equation}Y^*\in\argmin_{Y\in\mathcal C(u)}\mathcal J(Y;u),\end{equation}
其中 $\mathcal C(u)$ 表示合法性约束。我们不假定极小点必然存在或唯一；不可行输入需要报告约束冲突，不能通过输出流畅掩盖不可行性。
\section{离散结构与连续属性}
结构编辑可采用候选搜索，连续属性可采用梯度或其他优化方法。对于交替更新，只有每一步都不增加同一目标时，才能推导目标序列单调；结构编辑改变目标定义后，该结论需要重新分析。
\section{读出、执行与闭环校准}
读出器 $D(Y)$ 可以生成文本、程序、场景或控制计划。计划执行后，环境反馈通过编码器返回状态更新。内部生成的一致性与现实执行的正确性是不同指标，后者需要观测、实验或形式验证。
\section{从表示生成到持续智能}
MST 描述联合表示与生成的一条路线，HSF-HD 还需要历史、结构适应和目标控制，AgentOS 进一步管理这些过程的调度与边界。因此单次生成模型不是完整的持续智能体。
'''),
}
m=json.loads((R/'docs/chapter-manifest.json').read_text())
for row in m['chapters']:
    p=R/row['path'];old=p.read_text();labels=re.findall(r'\\label\{([^}]+)\}',old)
    key=next((x for x in labels if x in CONTENT),None)
    if key is None: continue
    title,body=CONTENT[key]
    figs=re.findall(r'\\begin\{figure\}(?:\[[^]]*\])?.*?\\end\{figure\}',old,re.S)
    fig_labels=set(re.findall(r'\\label\{([^}]+)\}','\n'.join(figs)))
    aliases=list(dict.fromkeys(x for x in labels if x!=key and x not in fig_labels))
    head='\\chapter{'+title+'}\n\\label{'+key+'}\n'
    head+='% Legacy section labels temporarily resolve to the chapter.\n'+'\n'.join('\\label{'+x+'}' for x in aliases)+'\n'
    if figs:
        fp=R/'figures/preserved'/f'{p.stem}.tex';fp.parent.mkdir(parents=True,exist_ok=True);fp.write_text('\n\n'.join(figs)+'\n')
        body+='\n\\section{结构示意图}\n我们保留以下原图说明结构与内容的组织关系；图中的物理措辞只在指定领域或辅助解释中适用。\n\\input{'+fp.relative_to(R).as_posix()+'}\n'
    p.write_text(head+body.strip()+'\n')
    row.update(new_title=title,status='computational-rewrite',legacy_section_aliases=aliases,preserved_figure_count=len(figs))
(R/'docs/chapter-manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print('Completed the remaining five chapters of volume one.')
