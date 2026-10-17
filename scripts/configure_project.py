#!/usr/bin/env python3
"""One-time extraction of shared configuration from the pinned manuscript."""
from pathlib import Path
import subprocess

R = Path(__file__).resolve().parents[1]
if (R / 'config/metadata.tex').exists():
    raise SystemExit('Configuration already extracted; edit config/ and styles/ directly.')
source = subprocess.check_output(['git', 'show', '6fee4834649091437c012bc736e3dac28cfca793:hsf_hd.tex'], cwd=R).decode()
def put(path, text):
    p = R / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)

# Vendor the existing class; only its hardcoded Latin fonts are made portable.
cls = (R / 'elegantbook.cls').read_text()
start = cls.index('  \\setmainfont{TeXGyreTermesX}')
end = cls.index('\\fi', start)
cls = cls[:start] + r'''  \IfFontExistsTF{TeX Gyre Termes}{\setmainfont{TeX Gyre Termes}}{\setmainfont{Nimbus Roman}}
  \IfFontExistsTF{TeX Gyre Heros}{\setsansfont{TeX Gyre Heros}[Scale=0.9]}{\setsansfont{Nimbus Sans}[Scale=0.9]}
''' + cls[end:]
cls = cls.replace('\\RequirePackage{pifont,manfnt,bbding}', r'''\RequirePackage{pifont,manfnt}
\IfFileExists{bbding.sty}{\RequirePackage{bbding}}{\providecommand{\HandPencilLeft}{\ding{43}}\providecommand{\SquareShadowBottomRight}{\ding{110}}}''')
cls = cls.replace('\\RequirePackage{adforn}', r'''\IfFileExists{adforn.sty}{\RequirePackage{adforn}}{\providecommand{\adftripleflourishleft}{\ensuremath{\diamond}}\providecommand{\adftripleflourishright}{\ensuremath{\diamond}}}''')
put('vendor/elegantbook.cls', cls)

put('config/packages.tex', r'''% Shared packages not already provided by ElegantBook.
\usepackage{ragged2e,microtype,float,makecell,tabularx,url}
\usepackage{standalone,tikz-3dplot}
\usetikzlibrary{shapes.geometric,shapes.misc,arrows.meta,fadings,positioning,calc,backgrounds,fit,shadows}
\AtBeginDocument{\special{pdf:minorversion 7}}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\usepackage{styles/hsfhd}
''')
put('config/fonts.tex', r'''% CTeX/Fandol is the portable default supplied by texlive-lang-chinese.
% Override fonts here, not in chapter files. Latin defaults live in the vendored class.
\xeCJKsetup{CJKspace=true}
''')
put('config/layout.tex', r'''\setcounter{tocdepth}{1}
\hypersetup{pdftitle={HSF-HD：智能状态的计算动力学},pdfauthor={XuEn},pdfkeywords={HSF-HD,AgentOS,计算动力学,智能状态}}
\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single,
 backgroundcolor=\color{gray!10},keywordstyle=\color{blue},
 commentstyle=\color{green!50!black},stringstyle=\color{red},language=Python,
 breakatwhitespace=true,showspaces=false,showstringspaces=false}
''')
put('config/metadata.tex', r'''\title{智能状态的计算动力学}
\subtitle{全息语义场与层级动力学（HSF-HD）}
\author{XuEn\\\url{nanhangxuen@gmail.com}}
\institute{panda lab}
\date{2026年10月5日}
\version{2.0 工作稿}
\logo{figure/balls.pdf}
''')
put('styles/hsfhd.sty', r'''\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{hsfhd}[2026/10/05 HSF-HD manuscript conventions]
\newtcolorbox{bquote}{colback=gray!10,grow to left by=-5mm,boxrule=0pt,
 leftrule=3pt,colframe=green!50,sharp corners,breakable,enhanced,
 borderline west={2pt}{0pt}{gray!50},before skip=10pt,after skip=10pt}
\newcommand{\smalltitle}[1]{\vspace{1em}\noindent\textbf{\textcolor{structurecolor}{#1}}}
\newcommand{\redtext}[1]{{\textcolor{red}{#1}}}
\newcommand{\bluetext}[1]{{\textcolor{blue}{#1}}}
\newcommand{\purpletext}[1]{{\textcolor{purple}{#1}}}
\newcommand{\bookvolume}[1]{\cleardoublepage\phantomsection
 \addcontentsline{toc}{part}{#1}\markboth{#1}{#1}
 \thispagestyle{empty}\vspace*{0.28\textheight}
 \begin{center}\Huge\bfseries #1\end{center}\cleardoublepage}
\newcommand{\hsfstate}{\mathbf{x}}
\newcommand{\hsfstructure}{\mathbf{S}}
\newcommand{\hsfmemory}{\mathbf{m}}
\newcommand{\hsfvalue}{\mathbf{v}}
\newenvironment{modelassumption}[1]{\begin{bquote}\textbf{建模假设：#1}\par}{\end{bquote}}
\newenvironment{auxiliaryexplanation}[1]{\begin{bquote}\textbf{辅助解释：#1}\par}{\end{bquote}}
''')
cover = source[source.index('% 临时设置当前页面'):source.index('\\maketitle')]
back = source[source.index('% ========== 插入封底页'):source.index('\\end{document}')]
put('frontmatter/cover.tex', cover)
put('backmatter/back-cover.tex', back)
put('hsf_hd.tex', r'''% !TEX program = xelatex
% Build from the repository root: latexmk -xelatex hsf_hd.tex
\documentclass[lang=cn,10pt,scheme=chinese]{vendor/elegantbook}
\input{config/packages.tex}
\input{config/fonts.tex}
\input{config/layout.tex}
\input{config/metadata.tex}
\begin{document}
\input{frontmatter/cover.tex}
\maketitle
\frontmatter
\input{frontmatter/preface.tex}
\input{frontmatter/literary-preface.tex}
\input{frontmatter/reading-guide.tex}
\tableofcontents
\mainmatter
\input{introduction/introduction.tex}
\input{volumes/volume-01.tex}
\input{volumes/volume-02.tex}
\input{volumes/volume-03.tex}
\input{appendices/appendices.tex}
\backmatter
\input{backmatter/back-cover.tex}
\end{document}
''')
put('.latexmkrc', '''$pdf_mode = 5;
$out_dir = 'build';
$xelatex = 'xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$max_repeat = 5;
''')
put('Makefile', '''.PHONY: book smoke check clean
book:
\tlatexmk -xelatex hsf_hd.tex
smoke:
\tlatexmk -xelatex smoke.tex
check:
\tpython3 scripts/check_manuscript.py
clean:
\tlatexmk -C hsf_hd.tex smoke.tex
''')
put('smoke.tex', r'''% Fast review of the newly rewritten theoretical spine.
\documentclass[lang=cn,10pt,scheme=chinese]{vendor/elegantbook}
\input{config/packages.tex}\input{config/fonts.tex}\input{config/layout.tex}
\begin{document}\frontmatter
\input{frontmatter/preface.tex}\input{frontmatter/reading-guide.tex}\tableofcontents\mainmatter
\input{introduction/introduction.tex}
\input{chapters/part-01/chp-dual_basis.tex}
\input{chapters/part-02/chp-constitutive_equation.tex}
\input{chapters/part-03/chp-holomorphic_isomorphism.tex}
\input{chapters/part-07/chp-macro_stability_ig.tex}
\input{chapters/part-08/chp-evolution_equation_ch.tex}
\input{chapters/part-09/chp-phenomenological_physics.tex}
\input{chapters/part-11/chap-software_control_msos.tex}
\end{document}
''')
with (R / '.gitignore').open('a') as f:
    f.write('\n/build/\n/__pycache__/\n*.pyc\n.DS_Store\n')
print('Root, shared configuration, build entry points and vendored class configured.')
