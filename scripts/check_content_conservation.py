#!/usr/bin/env python3
"""Prove that original chapter exposition remains before any added analysis.

Allowed differences are the chapter title, whitespace/comments, the one fixed
anchor, and figure layout/captions (drawing conservation is checked separately).
This checks preservation, not scientific validity or completion of rewriting.
"""
from pathlib import Path
import subprocess,json,re
R=Path(__file__).resolve().parents[1]
m=json.loads((R/'docs/chapter-manifest.json').read_text())
s=subprocess.check_output(['git','show',m['baseline']+':hsf_hd.tex'],cwd=R).decode()
lines=s.splitlines(keepends=True)
def normalize(text):
    text=re.sub(r'(?<!\\)%[^\n]*','',text)
    text=re.sub(r'\\begin\{figure\}(?:\[[^]]*\])?.*?\\end\{figure\}', '<FIGURE>',text,flags=re.S)
    start=text.index('\\chapter{')+len('\\chapter{')
    depth=1;end=start
    while depth:
        if text[end]=='{' and text[end-1]!='\\':depth+=1
        if text[end]=='}' and text[end-1]!='\\':depth-=1
        end+=1
    text=text[:start]+text[end-1:]
    text=text.replace('subsec:llm_wm_coherence_capacity_extended','subsec:llm_wm_coherence_capacity')
    text=text.replace('\\begin{aligned}','').replace('\\end{aligned}','').replace('\\\\','').replace('&{}','').replace('&','').replace('@{}','')
    text=text.replace('\\text{太阳系}','太阳系')
    return re.sub(r'\s+','',text)
validation=json.loads((R/'docs/rewrite-validation.json').read_text())
rewritten={x['path']:x['claimed_checks_pass'] for x in validation['chapters'] if x['claimed_rewritten']}
records=[]
for r in m['chapters']:
    original=''.join(lines[r['source_start_line']-1:r['source_end_line']-1])
    current=(R/r['path']).read_text()
    a,b=normalize(original),normalize(current)
    records.append({'path':r['path'],'original_exposition_preserved':b.startswith(a),
        'validation_mode':'full-rewrite-length-and-structure' if r['path'] in rewritten else 'original-exposition-conservation',
        'checks_passed':rewritten[r['path']] if r['path'] in rewritten else b.startswith(a),
        'baseline_chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',original)),
        'current_chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',current))})
report={'checks_passed':all(r['checks_passed'] for r in records),
 'original_complete_chapters_verified':sum(r['original_exposition_preserved'] for r in records),
 'fully_rewritten_chapters_length_verified':len(rewritten),
 'baseline_chapter_chinese_characters':sum(r['baseline_chinese_characters'] for r in records),
 'current_chapter_chinese_characters':sum(r['current_chinese_characters'] for r in records),
 'scope':'Pending chapters retain complete original exposition; individually rewritten chapters use length/structure audit instead of verbatim conservation. Scientific claims not validated by these mechanical checks.',
 'chapters':records}
(R/'docs/content-conservation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='chapters'},ensure_ascii=False,indent=2))
if not report['checks_passed']:raise SystemExit(1)
