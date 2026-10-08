# 第55章改写检查点

- 正式路径：`chapters/part-08/chp-dynamical_reconstruction.tex`
- 分段草稿：`docs/drafts/chapter-055-draft.tex`
- 拟用标题：学习动力学的重建：从变分目标到受约束随机更新
- 状态：长章分段实写中；正式章、`REWRITE_PROGRESS.csv` 与 manifest 仍保持 `queued-for-full-rewrite`
- 固定基线：源码汉字7346、实质正文汉字6206、可读单位8070、九个原编号小节
- 本轮完成：章首语和前四个编号小节。章首语259汉字；四节实质正文分别为1022、1007、1052、1069汉字，草稿正文合计4150汉字
- 结构约束：草稿只使用章首语与编号 `section`；未使用 `subsection`、`subsubsection`、`paragraph`、`smalltitle` 或无编号小节
- 已迁移标签：`chp:dynamical_reconstruction`、`sec:free_energy_hsf_hd`、`sec:cognitive_einstein_information_geometry`、`sec:statistical_physics_view`、`sec:engineering_mapping_optimizer`
- 本轮核心修订：把“严格同构”降为有适用条件的模型比较；定义参数、结构、分布、时钟、单位和变化目标；限定费希尔度量只表达局部预测可辨识性；给出状态相关预条件朗之万方程的漂移修正边界；把Sophia--HSF重建为带曲率代理、信赖域、接受测试、版本控制和四本验收账的AgentOS工程实例
- 图像：原章无 `figure`；正式全书视觉资源未改动
- 验证：草稿字符门槛与深层标题检查通过；`make check` 对当前54章正式已写子集通过。草稿尚未纳入审阅PDF，不能标记全文完成或排版通过

## 原问题到新论证的对应

| 原标签/问题 | 新论证 | 进度 |
| --- | --- | --- |
| `sec:free_energy_hsf_hd`：损失、自由能与作用量被宣称严格同构 | 区分监督目标、变分自由能与轨迹泛函；目标变化进入更新；给出标签污染和过强正则反例 | 已实写 |
| `sec:cognitive_einstein_information_geometry`：费希尔度量被等同为全部认知几何 | 定义模型流形、量纲、正定条件和自然梯度信赖域；离散结构与连续分布分层 | 已实写 |
| `sec:statistical_physics_view`：SGD被直接等同为朗之万过程 | 从条件噪声出发，区分抽样噪声与显式噪声，声明伊藤修正、步长与实验判据 | 已实写 |
| `sec:engineering_mapping_optimizer`：物理类比替代优化器条件 | 构造曲率代理、逐坐标信赖域、随机候选、两阶段结构提交及数值/安全/任务/资源四账 | 已实写 |
| `sec:entropy_minimization`：信息熵、计算代价、热力学熵产和电能混同 | 分账定义活动、代价、信息量、资源与焦耳能耗；只在已校准物理系统中谈熵产 | 待续写 |
| `sec:cefe_as_gradient_flow`：驻点方程被直接当作动态梯度流 | 定义状态—结构泛函、迁移算子与显式时间项；区分驻点、局部稳定和全局收敛 | 待续写 |
| `sec:proof_of_minimum_future_entropy`：由训练分布稳态推出任意未来输入全局最优 | 改为分布内期望风险的条件命题，并给出漂移、模型错设和多稳态反例 | 待续写 |
| `sec:physical_consequences`：训练被等同吸热手术、推理被等同绝热超导 | 分别记录训练资源、部署延迟和实测功耗；把物理措辞限制为辅助解释 | 待续写 |
| `sec:double_descent_topology`：双重下降被宣称拓扑必然 | 用载荷比、谱条件和正则路径描述可检验机制，保留不存在双重下降的数据反例 | 待续写 |

## 精确续写位置

从第五节 `sec:entropy_minimization` 开始，先完成“代价、信息与实际耗散的分账”，目标每节至少1000实质汉字；随后依次完成余下四节。九节全部实写、全章汉字与可读单位达到固定基线后，才把草稿替换到正式章并更新进度状态、manifest、论证映射、审阅入口和排版记录。
