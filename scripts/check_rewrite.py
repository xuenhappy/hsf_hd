#!/usr/bin/env python3
"""Audit genuine chapter rewrites; pending originals are never counted as done.

Two independent length rules: all baseline Han characters, including headings
and source comments, must be matched by the new substantive chapter prose; and
readable Han/Latin-word/number units must not decline. Section prose excludes
headings, display/inline math, code, captions, figures and source metadata.
"""
from pathlib import Path
import argparse,collections,csv,json,re,subprocess
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args()
m=json.loads((R/'docs/chapter-manifest.json').read_text())
s=subprocess.check_output(['git','show',m['baseline']+':hsf_hd.tex'],cwd=R).decode();lines=s.splitlines(keepends=True)
MIN_SECTION=1000;MIN_OPENING=200

def argument(text,start):
    depth=1;i=start
    while i<len(text) and depth:
        if text[i]=='{' and (i==0 or text[i-1]!='\\'):depth+=1
        elif text[i]=='}' and (i==0 or text[i-1]!='\\'):depth-=1
        i+=1
    if depth:raise ValueError('Unbalanced argument')
    return text[start:i-1],i

def drop_commands(text,names):
    pattern=re.compile(r'\\(?:'+ '|'.join(names)+r')\*?(?:\[[^]]*\])?\{')
    while True:
        match=pattern.search(text)
        if not match:return text
        _,end=argument(text,match.end());text=text[:match.start()]+' '+text[end:]

def prose(text,keep_code=False):
    text=re.sub(r'(?<!\\)%[^\n]*','',text)
    text=re.sub(r'\\begin\{(?:figure|table)\}.*?\\end\{(?:figure|table)\}', ' ',text,flags=re.S)
    if not keep_code:text=re.sub(r'\\begin\{lstlisting\}(?:\[[^]]*\])?.*?\\end\{lstlisting\}',' ',text,flags=re.S)
    text=re.sub(r'\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?)\}.*?\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?)\}', ' ',text,flags=re.S)
    text=re.sub(r'\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|(?<!\\)\$[^$]*?(?<!\\)\$', ' ',text,flags=re.S)
    text=drop_commands(text,['chapter','section','subsection','subsubsection','label','ref','eqref','pageref','autoref','includegraphics','input','lstinputlisting','caption'])
    text=re.sub(r'\\(?:begin|end)\{[^}]+\}',' ',text)
    text=re.sub(r'\\[A-Za-z]+\*?(?:\[[^]]*\])?',' ',text)
    return text

def han(text):return len(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]',text))
def units(text):return han(text)+len(re.findall(r'[A-Za-z]+(?:[-\'][A-Za-z]+)*|\d+(?:\.\d+)?',text))

rows=[];errors=[]
for i,c in enumerate(m['chapters'],1):
    original=''.join(lines[c['source_start_line']-1:c['source_end_line']-1]);current=(R/c['path']).read_text()
    claimed=c['status'].startswith('rewritten-')
    blocks=[]
    matches=list(re.finditer(r'\\section\{',current))
    opening=current[:matches[0].start()] if matches else current
    for k,x in enumerate(matches):
        title,end=argument(current,x.end());stop=matches[k+1].start() if k+1<len(matches) else len(current)
        blocks.append({'title':title,'prose_han':han(prose(current[end:stop]))})
    original_han=han(original);new_han=han(prose(current));base_units=units(prose(original,keep_code=True));new_units=units(prose(current,keep_code=True))
    issues=[]
    if claimed:
        if new_han<original_han:issues.append(f'Chapter prose {new_han} < original Han {original_han}')
        if new_units<base_units:issues.append(f'Chapter readable units {new_units} < original {base_units}')
        if han(prose(opening))<MIN_OPENING:issues.append('Opening too short')
        if not blocks:issues.append('No numbered body sections')
        if re.search(r'\\(?:subsection|subsubsection|paragraph|subparagraph|smalltitle)\*?\b|\\section\*',current):issues.append('Third-level or unnumbered heading in chapter')
        for b in blocks:
            if b['prose_han']<MIN_SECTION:issues.append(f"Section too short: {b['title']} ({b['prose_han']})")
        paragraphs=[re.sub(r'\s+','',p) for p in prose(current).split('\n\n') if han(p)>80]
        if any(n>1 for n in collections.Counter(paragraphs).values()):issues.append('Repeated substantive paragraph')
        for term in ['认知波函数','认知爱因斯坦场方程','目的论狄拉克方程','全息同构定理','宇宙的源代码','实体即纠缠','非幺正坍缩']:
            if term in prose(current):issues.append('Unresolved core physical formulation: '+term)
        errors.extend(c['path']+': '+x for x in issues)
    rows.append({'order':i,'path':c['path'],'title':c.get('new_title',c['title']),'status':c['status'],
      'original_han_floor':original_han,'new_chapter_prose_han':new_han,'original_readable_units':base_units,'new_readable_units':new_units,
      'opening_prose_han':han(prose(opening)),'sections':blocks,'claimed_rewritten':claimed,'claimed_checks_pass':claimed and not issues,'issues':issues})
report={'claimed_chapters_pass':not errors,'all_chapters_rewritten':all(r['claimed_rewritten'] for r in rows),'total_chapters':len(rows),
 'rewritten_chapters':sum(r['claimed_rewritten'] for r in rows),'pending_chapters':sum(not r['claimed_rewritten'] for r in rows),
 'minimum_section_prose_han':MIN_SECTION,'minimum_opening_prose_han':MIN_OPENING,
 'count_policy':'New prose Han >= all original source Han; readable Han/Latin-word/number units also compared with baseline (code included). Section prose excludes math, code, figures, tables, captions, headings and TeX metadata. Half-page layout reviewed separately.',
 'errors':errors,'chapters':rows}
(R/'docs/rewrite-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (R/'docs/REWRITE_PROGRESS.csv').open('w') as f:
    w=csv.writer(f,lineterminator="\n");w.writerow(['order','path','status','original_han_floor','new_chapter_prose_han','original_readable_units','new_readable_units','section_count','minimum_section_prose_han'])
    for r in rows:w.writerow([r['order'],r['path'],r['status'],r['original_han_floor'],r['new_chapter_prose_han'],r['original_readable_units'],r['new_readable_units'],len(r['sections']),min((b['prose_han'] for b in r['sections']),default=0)])
print(json.dumps({k:v for k,v in report.items() if k!='chapters'},ensure_ascii=False,indent=2))
for r in rows:
    if r['claimed_rewritten']:print(json.dumps({k:r[k] for k in ['order','original_han_floor','new_chapter_prose_han','original_readable_units','new_readable_units','opening_prose_han','sections']},ensure_ascii=False))
if errors or (args.all and not report['all_chapters_rewritten']):raise SystemExit(1)
