import re
import sys
import os

class MarkdownToLatexConverter:
    def __init__(self, content):
        self.content = content
        self.lines = content.split('\n')
        self.latex_lines = []
        self.in_table = False
        self.table_buffer = []
        self.table_alignments = []

    def escape_latex(self, text):
        """
        转义 LaTeX 特殊字符，但保护数学公式部分。
        """
        # 特殊字符映射表
        special_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',  # 注意：这里我们单独处理公式，所以这个是在文本中的美元符号
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}',
        }

        # 简单的状态机分离公式和文本
        # 模式匹配: $$...$$ (块公式) 或 $...$ (行内公式)
        # 注意：这里假设 Markdown 语法规范
        pattern = r'(\$\$[\s\S]*?\$\$|\$[^\$\n]+\$)'
        parts = re.split(pattern, text)
        
        escaped_text = ""
        for part in parts:
            if part.startswith('$'):
                # 这是一个公式，保持原样（去除 Markdown 可能的多余转义）
                # 在 LaTeX 中，$$ 通常建议用 \[ \]，这里为了兼容保持 $$ 或转换
                escaped_text += part
            else:
                # 这是普通文本，进行转义
                clean_part = part
                for char, replacement in special_chars.items():
                    # 此时的 $ 符号如果出现在非公式部分，说明是文字，应该转义
                    if char == '$': continue 
                    clean_part = clean_part.replace(char, replacement)
                escaped_text += clean_part
        
        return escaped_text

    def process_headers(self, line):
        """处理标题层级"""
        # HSF-HD 结构宏大，建议映射如下：
        # # -> \part
        # ## -> \chapter
        # ### -> \section
        # #### -> \subsection
        
        if line.startswith('# '):
            return r'\part{' + self.escape_latex(line[2:].strip()) + '}'
        elif line.startswith('## '):
            return r'\chapter{' + self.escape_latex(line[3:].strip()) + '}'
        elif line.startswith('### '):
            return r'\section{' + self.escape_latex(line[4:].strip()) + '}'
        elif line.startswith('#### '):
            return r'\subsection{' + self.escape_latex(line[5:].strip()) + '}'
        elif line.startswith('##### '):
            return r'\subsubsection{' + self.escape_latex(line[6:].strip()) + '}'
        return None

    def process_formatting(self, line):
        """处理粗体、斜体"""
        # 粗体 **text**
        line = re.sub(r'\*\*(.*?)\*\*', lambda m: r'\textbf{' + m.group(1) + '}', line)
        # 斜体 *text*
        line = re.sub(r'\*(.*?)\*', lambda m: r'\textit{' + m.group(1) + '}', line)
        return line

    def parse_table_row(self, line):
        """解析表格行"""
        # 去除首尾的 |
        line = line.strip('|')
        cells = [c.strip() for c in line.split('|')]
        return cells

    def flush_table(self):
        """将缓存的表格转换为 LaTeX"""
        if not self.table_buffer:
            return ""
        
        # 确定列数
        cols = len(self.table_buffer[0])
        col_def = "|".join(["l"] * cols) # 默认左对齐，带竖线
        
        tex = []
        tex.append(r'\begin{table}[h!]')
        tex.append(r'\centering')
        tex.append(r'\begin{tabular}{|' + col_def + r'|}')
        tex.append(r'\hline')
        
        # 表头
        headers = self.table_buffer[0]
        tex.append(" & ".join([self.escape_latex(h) for h in headers]) + r' \\ \hline')
        
        # 内容 (跳过分隔符行)
        for row in self.table_buffer[2:]:
            tex.append(" & ".join([self.escape_latex(c) for c in row]) + r' \\ \hline')
            
        tex.append(r'\end{tabular}')
        tex.append(r'\end{table}')
        
        self.table_buffer = []
        self.in_table = False
        return "\n".join(tex)

    def convert(self):
        # 写入导言区
        self.latex_lines.append(r'\documentclass[a4paper,12pt]{report}')
        self.latex_lines.append(r'\usepackage{geometry}')
        self.latex_lines.append(r'\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}')
        self.latex_lines.append(r'\usepackage{amsmath, amssymb}') # 数学公式
        self.latex_lines.append(r'\usepackage{xeCJK}') # 中文支持
        self.latex_lines.append(r'\setCJKmainfont{SimSun}') # 设置中文字体，根据系统调整，如 "PingFang SC" 或 "Microsoft YaHei"
        self.latex_lines.append(r'\usepackage{hyperref}') # 超链接
        self.latex_lines.append(r'\usepackage{graphicx}') # 图片
        self.latex_lines.append(r'\usepackage{listings}') # 代码块
        self.latex_lines.append(r'\usepackage{xcolor}') # 颜色
        self.latex_lines.append(r'\usepackage{float}') # 浮动体
        
        # 代码块样式设置
        self.latex_lines.append(r'''
\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10},
    keywordstyle=\color{blue},
    commentstyle=\color{green!50!black},
    stringstyle=\color{red}
}
''')
        
        self.latex_lines.append(r'\begin{document}')
        # 生成目录
        self.latex_lines.append(r'\tableofcontents')
        self.latex_lines.append(r'\newpage')

        in_code_block = False
        
        for line in self.lines:
            line = line.rstrip()
            
            # 1. 处理代码块
            if line.startswith('```'):
                if in_code_block:
                    self.latex_lines.append(r'\end{lstlisting}')
                    in_code_block = False
                else:
                    self.latex_lines.append(r'\begin{lstlisting}')
                    in_code_block = True
                continue
            
            if in_code_block:
                self.latex_lines.append(line)
                continue

            # 2. 处理表格
            # 简单的表格检测：以 | 开头并包含 |
            if line.strip().startswith('|') and line.strip().endswith('|'):
                if not self.in_table:
                    self.in_table = True
                self.table_buffer.append(self.parse_table_row(line))
                continue
            else:
                if self.in_table:
                    # 表格结束，输出表格
                    self.latex_lines.append(self.flush_table())

            # 3. 处理标题
            header = self.process_headers(line)
            if header:
                self.latex_lines.append(header)
                continue

            # 4. 处理引用块
            if line.startswith('> '):
                content = self.process_formatting(self.escape_latex(line[2:]))
                self.latex_lines.append(r'\begin{quote}' + content + r'\end{quote}')
                continue

            # 5. 处理列表 (简单处理)
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                content = self.process_formatting(self.escape_latex(line.strip()[2:]))
                self.latex_lines.append(r'\begin{itemize}')
                self.latex_lines.append(r'\item ' + content)
                self.latex_lines.append(r'\end{itemize}') # 这种逐行处理方式很粗糙，但能跑通，TeX 会自动忽略多余的列表环境闭合开启
                continue
            
            # 6. 处理数学公式块 $$ ... $$
            if line.strip() == '$$':
                self.latex_lines.append(r'\[')
                continue
            if line.strip() == '$$': # 结束
                self.latex_lines.append(r'\]')
                continue

            # 7. 普通文本
            # 只有非空行才处理
            if line.strip():
                # 处理行内格式
                processed_line = self.process_formatting(self.escape_latex(line))
                self.latex_lines.append(processed_line + r' \\') # 强制换行，或者依靠段落间距
            else:
                self.latex_lines.append(r'\par') # 段落分隔

        # 确保表格缓存被清空
        if self.in_table:
            self.latex_lines.append(self.flush_table())

        self.latex_lines.append(r'\end{document}')
        return '\n'.join(self.latex_lines)

def convert_file(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        converter = MarkdownToLatexConverter(content)
        latex_content = converter.convert()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
            
        print(f"转换成功！已生成文件: {output_path}")
        print("提示: 请使用 xelatex 编译该文件以支持中文。")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    # 使用示例：
    # python script.py input.md output.tex
    
    if len(sys.argv) < 2:
        print("请提供文件名，例如:")
        print("python md2tex.py hsf_hd_theory.md")
        # 为了演示，如果没参数，尝试读取同目录下的文件
        # convert_file("hsf-hd.md", "hsf-hd.tex")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.rsplit('.', 1)[0] + '.tex'
        convert_file(input_file, output_file)



