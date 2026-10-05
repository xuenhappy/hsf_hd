# HSF-HD：智能状态的计算动力学

本项目研究智能系统的状态、结构、历史与控制如何在有限资源下共同演化。HSF-HD 位于系统科学与具体运行时架构之间，AgentOS 是其重要工程实例。

当前正在按全章字数与二级结构要求实质重写 117 章。前 12 章已实写，其余 105 章仍待写；恢复原文及增补分析不计为完成。三卷十一篇、封面、前置章节、附录与原图保留。详见 [重构状态](docs/REFACTOR_STATUS.md)、[写作规范](docs/EDITORIAL_STANDARD.md) 与 [逐章进度](docs/REWRITE_PROGRESS.csv)。

## 编译

推荐完整 TeX Live；依赖包括 XeLaTeX、latexmk、CTeX/xeCJK、Fandol 中文字体、TeX Gyre 西文字体、TikZ、standalone、tcolorbox 和 biblatex。

Ubuntu 可安装：

```sh
apt-get install texlive-xetex texlive-latex-extra texlive-lang-chinese texlive-lang-cjk texlive-bibtex-extra fonts-texgyre latexmk biber
```

在仓库根目录运行：

```sh
make check       # 主编译图、标签、资源与插图完整性
make smoke       # 第一卷审阅本（不是全书）
make book        # 全书装配（目前包含待写原文）
make rewrite-review # 仅已实写的前十二章
```

结果在 `build/`。根目录已有的 `hsf_hd.pdf` 是原稿成品，不是新版结果。发布日期固定在 `config/metadata.tex`，不随编译日期改变。

## 文件职责

| 位置 | 内容 |
| --- | --- |
| hsf_hd.tex | 全书装配入口 |
| config/、styles/ | 字体、宏包、版式、元数据与作者命令 |
| vendor/ | 固定版本 ElegantBook；调整西文字体加载并为可选装饰字体提供替代符号 |
| frontmatter/、introduction/ | 前置内容与独立绪论 |
| volumes/、parts/ | 三卷与十一篇装配 |
| chapters/ | 每章一个独立 TeX 文件 |
| appendices/ | 附录装配与独立附录 |
| backmatter/ | 后置内容 |
| figure/、image/ | 原有图像、TikZ 与 PDF，保留原路径 |
| figures/preserved/、drafts/previous-condensed/ | 上轮压缩稿与抽取图的历史材料，不参与主书编译 |
| docs/ | 章节清单、迁移证据、理论规范与审核状态 |
| scripts/ | 结构迁移与检查工具 |

章文件不含导言区。修改内容时直接编辑相应章节；修改顺序时编辑篇装配文件。不要在已经编辑的工作树重跑一次性迁移脚本，它们会从基线重建文件并覆盖后续修改。

`fluid_moe.py` 保留原型原文件，附录仍可引用；运行正确性与训练性质尚需独立验证。

## 许可证

保留原项目 LPPL 1.3c 许可证及 ElegantBook 来源与版权说明。模板来源：[ElegantBook](https://github.com/ElegantLaTeX/ElegantBook)。原文关系由 Git 历史和章节清单记录。
