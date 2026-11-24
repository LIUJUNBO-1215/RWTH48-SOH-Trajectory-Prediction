#!/usr/bin/env python3
"""Validate structure and retained CSV evidence; not an end-to-end model rerun."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,ast,re
ROOT=Path(__file__).resolve().parents[1]
TABLE_COUNTS={1:73,2:60,3:5,4:10,5:8,6:25,7:47,8:3}
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
for num,count in TABLE_COUNTS.items():
    file=ROOT/f'data/tables/Table_S{num}.csv'
    n=len(read_csv(file));assert n==count,(file,n,count)
    print(f'PASS {file.relative_to(ROOT)}: {n} rows')
ranking=read_csv(ROOT/'data/rankings/01_source_only_hi_ranking.csv')
assert len(ranking)==10293
folds=defaultdict(list)
for row in ranking:folds[(row['Horizon'],row['TestBattery'])].append(row)
assert len(folds)==141
assert all(len(x)==73 for x in folds.values())
for group in folds.values():
    ranks={str(r['Rank']):r['HI'] for r in group if str(r.get('Rank')) in {'1','2'}}
    assert ranks.get('1')=='V_TimeWeightedMean',(group[0],ranks)
    assert ranks.get('2')=='P_Last',(group[0],ranks)
print('PASS 141 HI-ranking groups, 73 candidates/group, stable top two')
source=list((ROOT/'src').glob('*.py'))
assert len(source)==25,len(source)
assert {int(p.name.split('_',1)[0]) for p in source}==set(range(1,26))
names={p.name for p in source}
for p in source:
    text=p.read_text(encoding='utf-8-sig')
    tree=ast.parse(text)
    compile(text,str(p),'exec')
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            value=node.value
            if value.endswith('.py') and '\n' not in value and '/' not in value and chr(92) not in value and (re.match(r'^[0-9]{2}_',value) or value.startswith('RWTH48_')):
                assert value in names,(p.name,node.lineno,value)
            assert not re.search(r'(?i)(?:[A-Z]:[\\/]+(?:Users|桌面|desktop|data|home)[\\/]|/home/|/mnt/|/Users/)',value),(p.name,node.lineno,value[:100])
print('PASS 25 Python scripts compile; numbered script references resolve; no known machine-specific paths')
figures=list((ROOT/'figures/main').glob('*.png'))+list((ROOT/'figures/appendix').glob('*.png'))
assert {p.name for p in figures}=={f'Figure_{i:02}.png' for i in range(1,9)}|{'Figure_A1.png','Figure_A2.png'}
print('PASS 10 final figure files')
# Integrity: canonical files do not have exact byte duplicates (including photos and CSVs).
files=[p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&{'.git','workspace','rawdata','raw_data','.venv','venv','__pycache__','.pytest_cache'} and p.suffix not in {'.pyc','.pyo'}]
hashes=defaultdict(list)
for p in files: hashes[hashlib.sha256(p.read_bytes()).hexdigest()].append(str(p.relative_to(ROOT)))
dups=[v for v in hashes.values() if len(v)>1]
assert not dups,dups
print(f'PASS no byte-identical duplicate artifacts ({len(files)} files)')
print('Retained-results checks passed. Full source-data model rerun NOT performed.')
