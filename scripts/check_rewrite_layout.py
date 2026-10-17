#!/usr/bin/env python3
from pathlib import Path
import fitz,re,json,math,statistics
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];D=R/'build'/'rewrite-render';D.mkdir(exist_ok=True)
p=fitz.open(R/'build/rewrite-review.pdf');toc=p.get_toc()
sections=[(k,x) for k,x in enumerate(toc) if x[0]==3]
normalize=lambda x:re.sub(r'\s+','',x)
all_rows=[];all_pitch=[]
for i,page in enumerate(p):
 rows=[]
 for b in page.get_text('dict')['blocks']:
  for line in b.get('lines',[]):
   line_text=''.join(s['text'] for s in line['spans']).strip()
   if __import__('re').match(r'^图\s*\d',line_text):continue
   sp=[s for s in line['spans'] if 9.8<=s['size']<=10.2 and len(re.findall('[\u4e00-\u9fff]',s['text']))>=4]
   if sp:
    y=min(s['bbox'][1] for s in sp)
    if 72<=y<=770 and not any(abs(y-v)<2 for v in rows):rows.append(y)
 rows.sort();all_rows.append(rows)
 all_pitch.extend(b-a for a,b in zip(rows,rows[1:]) if 12<b-a<25)
pitch=statistics.median(all_pitch)
log=(R/'build/rewrite-review.log').read_text(errors='replace')
height=float(re.search(r'HSFHD_REVIEW_TEXTHEIGHT=([0-9.]+)pt',log).group(1))*72/72.27
# Locate chapter/section heading top independently of header text.
def heading(entry):
 level,title,page=entry
 if level==1:return page-1,72
 # Match the stable CJK portion because some Type-1 fonts expose incomplete
 # ToUnicode maps even when the visible mixed-script heading is correct.
 needle=''.join(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]+',normalize(title)))
 for actual in range(max(0,page-2),min(len(p),page+2)):
  for b in p[actual].get_text('dict')['blocks']:
   for line in b.get('lines',[]):
    t=''.join(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]+',normalize(''.join(s['text'] for s in line['spans']))))
    if needle[:10] in t and any(s['size']>13 for s in line['spans']):return actual,line['bbox'][1]
 raise ValueError('Heading not found: '+title)
report=[]
for k,entry in sections:
 start=heading(entry);stop=heading(toc[k+1]) if k+1<len(toc) else (len(p)-1,770)
 n=0
 for i in range(start[0],stop[0]+1):
  for y in all_rows[i]:
   if (i,y)>start and (i,y)<stop:n+=1
 equiv=n*pitch/height
 report.append({'title':entry[1],'start_pdf_page':entry[2],'prose_lines':n,'calibrated_prose_page_equivalents':round(equiv,3),'half_page_prose_pass':equiv>=.5})
clipped=[]
for i,page in enumerate(p):
 for b in page.get_text('dict')['blocks']:
  for l in b.get('lines',[]):
   for s in l['spans']:
    x0,y0,x1,y1=s['bbox']
    if s['text'].strip() and (x0< -1 or y0< -1 or x1>page.rect.width+1 or y1>page.rect.height+1):clipped.append({'page':i+1,'text':s['text'][:45]})
for start in range(0,len(p),16):
 sheet=Image.new('RGB',(1600,2400),'#ddd');draw=ImageDraw.Draw(sheet)
 for i in range(start,min(start+16,len(p))):
  pix=p[i].get_pixmap(matrix=fitz.Matrix(.7,.7));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((390,565));j=i-start;col=j%4;row=j//4;sheet.paste(im,(col*400+(400-im.width)//2,row*600));draw.text((col*400+10,row*600+570),str(i+1),fill='black')
 sheet.save(D/f'pages-{start+1:03d}.jpg')
for i in sorted(set([heading(toc[k])[0] for k,_ in sections]+[0,1])):
 pix=p[i].get_pixmap(matrix=fitz.Matrix(1.25,1.25));pix.save(str(D/f'page-{i+1:03d}.png'))
result={'pages':len(p),'bytes':(R/'build/rewrite-review.pdf').stat().st_size,'text_height_pdf_points':round(height,2),'median_prose_line_pitch_pdf_points':round(pitch,2),'numbered_sections':len(report),'half_page_checks_pass':all(r['half_page_prose_pass'] for r in report),'physical_page_clipping':clipped,'sections':report,'scope':f'Review subset: {sum(x[0]==2 for x in toc)} chapters; does not validate the full book.'}
(R/'docs/rewrite-layout-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))

if not result["half_page_checks_pass"] or clipped:
 raise SystemExit("Review layout check failed")
