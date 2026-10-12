# 第78章分段改写检查点

- 正式路径：`chapters/part-10/chp-social_artificial_intelligence.tex`
- 草稿路径：`docs/drafts/chapter-078-draft.tex`
- 当前状态：长章分段实写中；正式章仍保留完整恢复稿，manifest 与进度表继续为 `queued-for-full-rewrite`
- 已完成：章首语，以及前两个编号正文小节 `sec:section_4093`、`sec:section_9110`
- 已迁移旧标签：市场节内的 `subsec:micro_semantion`、`subsec:economic_cefe` 已置于语义相应位置；章标签和两个旧节标签已经保留
- 已完成内容：市场部分以主体状态、契约图、订单事件、快慢更新和政策控制重建价格形成、反馈放大、路径依赖与宏观调节；交通部分以图上守恒流、逐车时延控制、路口排队、导航反馈和慢速治理重建拥堵、网格锁死与多目标交通控制
- 精确续写位置：第三个编号正文小节 `sec:section_4389`（现有LLM），随后处理 `sec:section_4089`（训练阶段LLM）和 `sec:finetuning_gauge_field`（微调动力学）
- 待迁移旧标签：从 `subsec:llm_spatial_topology_pathology` 到 `subsec:global_manifold_mapping` 的LLM相关旧子节标签，均须在后三个编号正文小节内迁移到语义对应位置
- 视觉资源：原章无figure主体；正文中的公式、列表和引文环境将在正式替换后参与编译与布局验收
- 固定验收基线：原章源码汉字21011、可读单位21452；完整新章必须达到两项基线，五个编号正文小节各至少1000汉字且实际PDF不少于半页
- 验收边界：当前草稿不是完整章节，未替换正式文件、未加入 `rewrite-review.tex`，不计入累计完成数，不标记 `rewritten-awaiting-layout-review` 或 `rewritten-reviewed`
