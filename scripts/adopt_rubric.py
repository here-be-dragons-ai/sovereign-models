"""Adopt the rubric's values into families.csv and scorecard.csv.

Where the answers in evidence.csv decide a field and the stored value differs, the rubric's value
is written (at family level if all models of the family agree, else per model), together with a
source row citing the decisive verified quote. Run after extract.py import and verify_quotes.py;
review the diff and give reasons in changes.csv after build.py.

    python scripts/adopt_rubric.py
"""
import csv
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rubric
from scorecard import Data, SOURCED, COUNTING_ISSUERS
TODAY=dt.date.today().isoformat()
ROOT=Path(__file__).resolve().parent.parent
d=Data()
def load(p):
    with open(ROOT/p,newline='') as f: r=csv.DictReader(f); return r.fieldnames,list(r)
ffn,fam=load('data/families.csv'); F={r['id']:r for r in fam}
sfn,sc=load('data/scorecard.csv'); M={r['id']:r for r in sc}
srcfn,src=load('data/sources.csv')
QS={'origin':['init_random','base_model','base_scratch','base_publisher','post_training_only'],
    'data':['data_downloadable','datasets_named','data_shares'],'compute':['pretraining_site'],
    'license_class':['licence_text','licence_modified','use_policy_binding','inherited_terms','noncommercial'],
    'terms_can_change':['policy_updatable']}
def decisive(m, field, provider_id):
    own=[r for r in d.evidence if (r['scope'],r['subject'])==('model',m['id'])]
    asked={r['question'] for r in own}
    fam_=[r for r in d.evidence if (r['scope'],r['subject'])==('family',m['family_id']) and r['question'] not in asked]
    rows=[r for r in rubric.usable(own+fam_, d.quote_status) if r['question'] in QS[field] and r['_quoted']]
    def issuer(r): return d.answer_issuer(r['url'],provider_id)
    rows.sort(key=lambda r:(issuer(r) not in COUNTING_ISSUERS, QS[field].index(r['question'])))
    return rows[0] if rows else None, (issuer(rows[0]) if rows else None)
changes=[]; added=[]
for fid,f in F.items():
    models=[m for m in d.resolved() if m['family_id']==fid]
    for field in rubric.FIELDS:
        got={m['id']:d.rubric(m)[field] for m in models}
        diff=[m for m in models if got[m['id']] is not None and got[m['id']]!=d.current(m,field)]
        if not diff: continue
        vals={v for v in got.values()}
        if (len(vals)==1 and None not in vals) or field not in sfn:
            vals={v for v in vals if v is not None}
            if len(vals)!=1: print('SKIP (models disagree, family-only field)',fid,field,got); continue
            v=vals.pop(); scope=('family',fid); old=f[field]; f[field]=v
            for m in models:
                if field in sfn: M[m['id']][field]=''   # family value now holds for all
            targets=[(scope,models[0])]
        else:
            targets=[]
            for m in diff:
                old=M[m['id']].get(field,''); M[m['id']][field]=got[m['id']]; targets.append((('model',m['id']),m))
        for (scope,m) in targets:
            v=got[m['id']]
            if field=='origin' and v.startswith('continued') or v.startswith('finetune'):
                b=[r['answer'] for r in d.evidence if r['question']=='base_model' and r['subject'] in (m['id'],fid) and r['answer'].lower()!='none']
                if b and field=='origin':
                    (F[fid] if scope[0]=='family' else M[m['id']])['base_model']=b[0]
            row,iss=decisive(m,field,f['provider_id'])
            changes.append((scope,field,v,row['url'] if row else '-'))
            if row and (field in SOURCED['family'] or field=='terms_can_change'):
                arch=next((s['archive_url'] for s in src if s['url']==row['url'] and s['archive_url']),'')
                src.append(dict(scope=scope[0],subject=scope[1],field=field,url=row['url'],issuer=iss,checked='yes',
                                retrieved=TODAY,archive_url=arch,note=f'rubric ({row["question"]}={row["answer"]}): "{row["quote"]}"'))
                added.append(scope)
for p,fn,rows,crlf in (('data/families.csv',ffn,fam,False),('data/scorecard.csv',sfn,sc,False),('data/sources.csv',srcfn,src,True)):
    w=csv.DictWriter(open(ROOT/p,'w',newline=''),fieldnames=fn,lineterminator='\r\n' if crlf else '\n'); w.writeheader(); w.writerows(rows)
for c in changes: print(c)
print(len(changes),'changes,',len(added),'sources added')
