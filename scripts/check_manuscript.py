#!/usr/bin/env python3
"""Check the actual root include graph, chapter identity, labels and figure conservation."""
from pathlib import Path
import collections
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
errors = []
visited = []
stack = []

def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)

def resolve(path):
    p = ROOT / path
    if p.suffix == '':
        p = p.with_suffix('.tex')
    return p

def expand(path):
    p = resolve(path)
    if not p.exists():
        errors.append(f'Missing include: {path}')
        return ''
    if p in stack:
        errors.append(f'Include cycle: {path}')
        return ''
    visited.append(p)
    stack.append(p)
    raw = p.read_text()
    # Follow uncommented include positions while retaining exact figure blocks.
    def replace(m):
        line_start = raw.rfind('\n', 0, m.start()) + 1
        prefix = raw[line_start:m.start()]
        if re.search(r'(?<!\\)%', prefix):
            return m[0]
        return expand(m[1])
    out = re.sub(r'\\input\{([^}]+)\}', replace, raw)
    stack.pop()
    return out

text = expand('hsf_hd.tex')
active = uncomment(text)
manifest = json.loads((ROOT / 'docs/chapter-manifest.json').read_text())
for row in manifest['chapters']:
    p = ROOT / row['path']
    if visited.count(p) != 1:
        errors.append(f'Chapter included {visited.count(p)} times: {row["path"]}')
    if len(re.findall(r'\\chapter\{', uncomment(p.read_text()))) != 1:
        errors.append(f'Expected exactly one chapter in {row["path"]}')
labels = collections.Counter(re.findall(r'\\label\{([^}]+)\}', active))
labels.update(re.findall(r'label=\{([^}]+)\}', active))
errors += [f'Duplicate label: {k}' for k, n in labels.items() if n > 1]
refs = re.findall(r'\\(?:ref|eqref|autoref|pageref|figref|tabref)\{([^}]+)\}', active)
errors += [f'Undefined reference: {k}' for k in sorted(set(refs) - set(labels))]
assets = re.findall(r'\\(?:includegraphics|logo)(?:\[[^]]*\])?\{([^}]+)\}', active)
assets += re.findall(r'\\lstinputlisting(?:\[[^]]*\])?\{([^}]+)\}', active)
for asset in assets:
    if not (ROOT / asset).exists():
        errors.append(f'Missing asset: {asset}')
figs = re.findall(r'\\begin\{figure\}(?:\[[^]]*\])?.*?\\end\{figure\}',
                  '\n'.join(p.read_text() for p in visited), re.S)
def visual_source(block):
    # Captions are author text and may be corrected while the original drawing
    # and image references must stay unchanged. Remove balanced caption arguments.
    out = ''
    pos = 0
    for m in list(re.finditer(r'\\caption\{', block)):
        if m.start() < pos:
            continue
        out += block[pos:m.start()]
        depth, end = 1, m.end()
        while depth:
            if block[end] == '{' and block[end - 1] != '\\': depth += 1
            if block[end] == '}' and block[end - 1] != '\\': depth -= 1
            end += 1
        pos = end
    out = out + block[pos:]
    # Layout may change; drawing commands, asset paths and labels must survive.
    out = re.sub(r'\\begin\{figure\}(?:\[[^]]*\])?', r'\\begin{figure}', out)
    out = re.sub(r'(\\includegraphics)\[[^]]*\]', r'\1', out)
    out = re.sub(r'\\begin\{adjustbox\}\{[^\n]*\}\s*|\\end\{adjustbox\}', '', out)
    return re.sub(r'\s+', '', out)

current = collections.Counter(hashlib.sha256(visual_source(x).encode()).hexdigest() for x in figs)
baseline = json.loads((ROOT / 'docs/figure-baseline.json').read_text())
baseline_source = subprocess.check_output(['git', 'show', f'{manifest["baseline"]}:hsf_hd.tex'], cwd=ROOT).decode()
original_figs = re.findall(r'\\begin\{figure\}(?:\[[^]]*\])?.*?\\end\{figure\}', baseline_source, re.S)
old = collections.Counter(hashlib.sha256(visual_source(x).encode()).hexdigest() for x in original_figs)
errors += [f'Original figure no longer compiled: {key}' for key, count in (old - current).items() for _ in range(count)]
# Verify original binary/image/TikZ asset bytes, independently of presentation.
original_assets = set(re.findall(r'\\(?:includegraphics|logo)(?:\[[^]]*\])?\{([^}]+)\}', baseline_source))
original_assets.add('figure/tdci.tex')
asset_bytes_verified = 0
for asset in sorted(original_assets):
    baseline_blob = subprocess.check_output(['git','rev-parse',f'{manifest["baseline"]}:{asset}'],cwd=ROOT).strip()
    current_blob = subprocess.check_output(['git','hash-object',asset],cwd=ROOT).strip()
    if baseline_blob != current_blob:
        errors.append(f'Original visual asset modified: {asset}')
    else:
        asset_bytes_verified += 1
old_captions = collections.Counter(x.strip() for x in re.findall(r'\\caption\{([^\n]+)', '\n'.join(original_figs)))
new_captions = collections.Counter(x.strip() for x in re.findall(r'\\caption\{([^\n]+)', '\n'.join(figs)))
counts = {
    'numbered_chapters': len(re.findall(r'\\chapter\{', active)),
    'parts': len(re.findall(r'\\part\{', active)),
    'volumes': len(re.findall(r'\\bookvolume\{', active)),
    'original_figure_blocks_preserved': sum((old & current).values()),
    'original_figures_with_layout_adjustments': sum(x not in original_figs for x in figs),
    'original_figures_with_revised_captions': sum((old_captions-new_captions).values()),
    'original_visual_assets_byte_verified': asset_bytes_verified,
    'figure_blocks': len(figs), 'labels': len(labels), 'reference_targets': len(set(refs)),
    'chapter_status': dict(collections.Counter(r['status'] for r in manifest['chapters'])),
}
if counts['numbered_chapters'] != 117 or counts['parts'] != 11 or counts['volumes'] != 3:
    errors.append('Book structure changed unexpectedly')
report = {'checks_passed': not errors, 'counts': counts, 'errors': errors}
(ROOT / 'docs/verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
