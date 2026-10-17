#!/usr/bin/env python3
"""Rebuild the modular manuscript from the pinned, unmodified Git baseline.

Run once during migration. Subsequent editing happens in the chapter files.
No formula or figure is transformed by this structural migration.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'docs/chapter-manifest.json').exists():
    raise SystemExit('Migration already exists; edit modular files instead of rebuilding.')
BASE = '6fee4834649091437c012bc736e3dac28cfca793'
SOURCE = subprocess.check_output(['git', 'show', f'{BASE}:hsf_hd.tex'], cwd=ROOT).decode()


def write(path, content):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')


def argument(text, start):
    depth = 1
    pos = start
    while depth:
        if pos >= len(text):
            raise ValueError('unbalanced structural title')
        if text[pos] == '{' and (pos == 0 or text[pos - 1] != '\\'):
            depth += 1
        elif text[pos] == '}' and (pos == 0 or text[pos - 1] != '\\'):
            depth -= 1
        pos += 1
    return text[start:pos - 1], pos


tokens = []
for m in re.finditer(r'^\\(chapter\*?|part|appendix)\{', SOURCE, re.M):
    title, end = argument(SOURCE, m.end())
    tokens.append({'kind': m[1], 'title': title.strip(), 'start': m.start(), 'end': end})

part_titles = [
    '结构与属性 — 形质表示基础',
    '数学形式化 — 状态空间与结构表示',
    '跨域映射 — 环境与认知表示',
    '工程实现 — 形质联合表示与 MST 架构',
    '动态过程元理论 — 交互、目标与反馈',
    '状态空间 — 语义结构与状态分布',
    '实现约束 — 边界、资源与局部机制',
    '整体动力学 — 演化、控制与多尺度耦合',
    '涌现机制 — 目的、自我与体验',
    '比较动力学 — 智能系统的结构与演化',
    '智能工程 — 架构、运行时与培育',
]
volume_titles = ['第一卷：智能的结构与表示', '第二卷：智能的状态动力学', '第三卷：智能的比较与工程']
records = []
volume, part, chapter = 0, 0, 0
volume_inputs = {}
part_inputs = {}
appendix_inputs = []
appendix_mode = False
front = []
for i, token in enumerate(tokens):
    end = tokens[i + 1]['start'] if i + 1 < len(tokens) else SOURCE.index('% ========== 插入封底页')
    block = SOURCE[token['start']:end]
    kind = token['kind']
    if kind == 'chapter*' and token['title'].startswith(('第一卷', '第二卷', '第三卷')):
        volume += 1
        volume_inputs[volume] = []
        # Replace the legacy volume prose with a focused computational overview.
        continue
    if kind == 'chapter*':
        path = 'frontmatter/literary-preface.tex' if not front else 'frontmatter/preface.tex'
        # The second legacy preface ends in TOC/mainmatter; these belong in the root.
        block = block.split('\\tableofcontents')[0]
        write(path, block)
        front.append(path)
        continue
    if kind == 'part':
        part += 1
        part_inputs[part] = []
        volume_inputs[volume].append(part)
        label = re.search(r'\\label\{([^}]+)\}', block)
        label_tex = '\\label{' + label[1] + '}\n' if label else ''
        write(f'parts/part-{part:02d}.tex', '\\part{' + part_titles[part - 1] + '}\n' + label_tex)
        continue
    if kind == 'appendix':
        appendix_mode = True
        continue
    if kind != 'chapter':
        raise ValueError(kind)
    chapter += 1
    label = re.search(r'\\label\{([^}]+)\}', block)
    identifier = re.sub(r'[^a-zA-Z0-9_-]+', '-', label[1] if label else f'chapter-{chapter}').strip('-')
    folder = 'appendices' if appendix_mode else f'chapters/part-{part:02d}'
    path = f'{folder}/{identifier}.tex'
    if any(r['path'] == path for r in records):
        raise ValueError('duplicate chapter path ' + path)
    write(path, block)
    row = {
        'path': path, 'title': token['title'], 'volume': volume if not appendix_mode else None,
        'part': part if not appendix_mode else None, 'kind': 'appendix' if appendix_mode else 'chapter',
        'source_start_line': SOURCE.count('\n', 0, token['start']) + 1,
        'source_end_line': SOURCE.count('\n', 0, end) + 1,
        'source_sha256': hashlib.sha256(block.encode()).hexdigest(),
        'status': 'migrated-unedited',
    }
    records.append(row)
    (appendix_inputs if appendix_mode else part_inputs[part]).append(path)

assert chapter == 117 and part == 11 and volume == 3
for number, paths in part_inputs.items():
    target = ROOT / f'parts/part-{number:02d}.tex'
    target.write_text(target.read_text() + '\n' + '\n'.join('\\input{' + p + '}' for p in paths) + '\n')
for number, parts in volume_inputs.items():
    write(f'volumes/volume-{number:02d}.tex', '\\bookvolume{' + volume_titles[number - 1] + '}\n' +
          '\n'.join(f'\\input{{parts/part-{p:02d}.tex}}' for p in parts) + '\n')
write('appendices/appendices.tex', '\\appendix\n\\phantomsection\n\\label{part:appendix}\n' +
      '\n'.join('\\input{' + p + '}' for p in appendix_inputs) + '\n')

# Record exact migration before any editorial work.
write('docs/chapter-manifest.json', json.dumps({'baseline': BASE, 'chapters': records}, ensure_ascii=False, indent=2) + '\n')
figures = re.findall(r'\\begin\{figure\}(?:\[[^]]*\])?.*?\\end\{figure\}', SOURCE, re.S)
write('docs/figure-baseline.json', json.dumps({'figure_count': len(figures), 'sha256':
      [hashlib.sha256(x.encode()).hexdigest() for x in figures]}, indent=2) + '\n')
write('docs/migration-verification.json', json.dumps({
    'baseline': BASE, 'numbered_chapters': chapter, 'body_chapters': chapter - len(appendix_inputs),
    'appendix_chapters': len(appendix_inputs), 'parts': part, 'volumes': volume,
    'all_chapter_blocks_exact_at_migration': all(
        hashlib.sha256((ROOT / r['path']).read_bytes()).hexdigest() == r['source_sha256'] for r in records),
}, indent=2) + '\n')
print(f'Migrated {chapter} complete chapter blocks; preserved figure and formula source.')
