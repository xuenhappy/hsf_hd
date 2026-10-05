# 第18章实写检查点

状态：已完成作者章首语与第一编号小节；正式章节保持 `queued-for-full-rewrite`，草稿未加入任何编译入口，也不计入已完成数量。

固定来源：`6fee4834649091437c012bc736e3dac28cfca793`；正式文件为 `chapters/part-06/chp-microscopic_origins.tex`，持久草稿为 `docs/drafts/chapter-018-microscopic-origins.tex`。

已实写内容：新标题为“统计状态的微观建模：信息几何、估计界与学习坐标”。章首语305汉字；第一节1184汉字并保留 `sec:section_8204`，定义样本空间、事件族、参考测度、参数空间、模型族和真实数据机制；区分世界结构快照与条件分布；以任务身份、绑定及读出联合局部结构网络与全局分布表示；用任务等价关系说明不可辨识参数方向；限定光滑可逆重参数化与离散结构编辑的边界，并将 AgentOS 仅作为可审计工程实例。

后续按以下顺序继续，每节至少1000汉字实质正文，完整章正文同时超过原源码5191汉字和可读单位5255：

1. 保留 `sec:section_1922`：在共同支持、可微、可交换积分微分及平方可积条件下，从评分期望为零推导 Fisher 半正定与 KL 局部二阶展开；退化方向在商空间处理，不把 Fisher 等同逻辑真理。
2. 保留 `sec:geometry_of_measurement_crlb`：先标量后向量推导 Cramér--Rao 界，声明无偏、正则性、样本量和矩阵序条件；说明它不推出量子不确定性、创造力—精确度必然对偶或幻觉必然性。
3. 保留 `sec:semantion_revisited_probability_wavepacket`：把分布表示重建为任务条件下的信念状态，区分混合分布、概率幅和模糊逻辑；纠正微分熵坐标依赖及“熵即质量”的错误，保留自然/期望参数的指数族实例。
4. 保留 `sec:dual_connections_learning_and_perception`：定义指数族/混合族的对偶坐标与散度投影条件；区分推断、在线更新和离线训练；不把 Transformer 前向传播、Adam 或所有学习过程规定为某种联络的严格同一物。

全章完成并迁入正式路径后，才能更新 manifest 为 `rewritten-awaiting-layout-review`；通过 XeLaTeX、半页门槛与逐页视觉检查后，才能标记 `rewritten-reviewed`。
