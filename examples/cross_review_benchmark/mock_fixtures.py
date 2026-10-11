"""Evaluator-only correct scripted solutions; live engines never see this file."""

SOLUTIONS = {
    "gap_runs": """def gap_runs(values):
    out=[]
    for v in sorted(set(values)):
        if out and v == out[-1][1]+1: out[-1]=(out[-1][0],v)
        else: out.append((v,v))
    return out
""",
    "clipped_windows": """def clipped_windows(text,width,stride):
    if any(type(n) is not int or n<=0 for n in (width,stride)): raise ValueError()
    return [text[i:i+width] for i in range(0,len(text),stride)]
""",
    "cent_parts": """from decimal import Decimal, InvalidOperation, localcontext, ROUND_HALF_UP
def cent_parts(amount,weights):
    if not weights or any(type(w) is not int or w<0 for w in weights) or sum(weights)==0: raise ValueError()
    try: a=Decimal(amount)
    except InvalidOperation: raise ValueError() from None
    if not a.is_finite() or a<0: raise ValueError()
    with localcontext() as ctx:
        ctx.prec=max(28,len(a.as_tuple().digits)+abs(a.as_tuple().exponent)+10)
        cents=int((a*100).quantize(Decimal(1),rounding=ROUND_HALF_UP))
    total=sum(weights); out=[cents*w//total for w in weights]
    order=sorted(range(len(weights)),key=lambda i: (-(cents*weights[i]%total),i))
    for i in order[:cents-sum(out)]: out[i]+=1
    return out
""",
    "fold_counts": """import unicodedata as u
def fold_counts(words):
    counts={}
    for w in words:
        k=u.normalize('NFC',u.normalize('NFC',w).casefold()); counts[k]=counts.get(k,0)+1
    return list(counts.items())
""",
    "utc_day": """from datetime import datetime, timezone
def utc_day(stamp):
    d=datetime.fromisoformat(stamp)
    if d.utcoffset() is None: raise ValueError()
    return d.astimezone(timezone.utc).date().isoformat()
""",
    "month_anchor": """import calendar
from datetime import date
def month_anchor(day,offset):
    y,m=divmod((day.year-1)*12+day.month-1+offset,12); y+=1; m+=1
    if not 1<=y<=9999: raise ValueError()
    return date(y,m,min(day.day,calendar.monthrange(y,m)[1]))
""",
    "quiet_ranges": """def quiet_ranges(busy,start,stop):
    if start>stop or any(a>b for a,b in busy): raise ValueError()
    out=[]; cursor=start
    for a,b in sorted(busy):
        a=max(start,a); b=min(stop,b)
        if a>=b: continue
        if a>cursor: out.append((cursor,a))
        cursor=max(cursor,b)
    if cursor<stop: out.append((cursor,stop))
    return out
""",
    "escaped_fields": """def escaped_fields(text):
    out=['']; escaped=False
    for c in text:
        if escaped: out[-1]+=c; escaped=False
        elif c==chr(92): escaped=True
        elif c=='|': out.append('')
        else: out[-1]+=c
    if escaped: raise ValueError()
    return out
""",
    "carry_round": """from decimal import Decimal, localcontext, ROUND_HALF_EVEN
def carry_round(values):
    nums=[Decimal(v) for v in values]
    if any(not v.is_finite() for v in nums): raise ValueError()
    with localcontext() as ctx:
        ctx.prec=max([28]+[len(v.as_tuple().digits)+abs(v.as_tuple().exponent)+len(str(len(nums)))+10 for v in nums])
        total=Decimal(0); last=0; out=[]
        for v in nums:
            total+=v; n=int(total.to_integral_value(rounding=ROUND_HALF_EVEN)); out.append(n-last); last=n
    return out
""",
    "recent_keys": """def recent_keys(keys,capacity):
    if type(capacity) is not int or capacity<0: raise ValueError()
    out=[]
    for k in keys:
        if k in out: out.remove(k)
        out.append(k)
        if len(out)>capacity: out.pop(0)
    return out
""",
    "braid_rows": """def braid_rows(rows):
    return [row[c] for c in range(max(map(len,rows),default=0)) for row in (rows if c%2==0 else rows[::-1]) if c<len(row)]
""",
    "stable_scores": """import math
def stable_scores(rows):
    return [n for n,s in sorted(rows,key=lambda r:(math.isnan(r[1]),0 if math.isnan(r[1]) else -r[1]))]
""",
    "near_duplicate": """def near_duplicate(values,distance):
    if distance<0: raise ValueError()
    last={}
    for i,v in enumerate(values):
        if v in last and i-last[v]<=distance: return True
        last[v]=i
    return False
""",
    "path_reduce": """def path_reduce(path):
    absolute=path.startswith('/'); parts=[]
    for p in path.split('/'):
        if p in ('','.'): continue
        if p=='..':
            if parts and parts[-1]!='..': parts.pop()
            elif not absolute: parts.append(p)
        else: parts.append(p)
    return ('/' if absolute else '')+'/'.join(parts) or '.'
""",
    "select_rank": """def select_rank(values,rank):
    if rank<0 or rank>=len(values): raise IndexError()
    return sorted(values)[rank]
""",
    "dense_labels": """def dense_labels(labels):
    seen={}; out=[]
    for s in labels:
        if s not in seen: seen[s]=len(seen)
        out.append(seen[s])
    return out
""",
    "retry_slots": """def retry_slots(delays,budget):
    if budget<0 or any(d<0 for d in delays): raise ValueError()
    out=[]; t=0
    for d in delays:
        t+=d
        if t>budget: break
        out.append(t)
    return out
""",
    "end_match": """def end_match(text,suffixes):
    return max((s for s in suffixes if text.endswith(s)),key=len,default=None)
""",
    "local_midnights": """from datetime import datetime, time, timedelta, timezone
def local_midnights(start,stop):
    if start.utcoffset() is None or stop.utcoffset() is None or start.tzinfo is not stop.tzinfo: raise ValueError()
    a=start.astimezone(timezone.utc); b=stop.astimezone(timezone.utc)
    if a>b: raise ValueError()
    d=start.date(); out=[]
    while d<=stop.date():
        m=datetime.combine(d,time(),tzinfo=start.tzinfo)
        if a<=m.astimezone(timezone.utc)<b: out.append(m)
        if d==stop.date(): break
        d+=timedelta(days=1)
    return sorted(out,key=lambda m:m.astimezone(timezone.utc))
""",
    "aligned_batches": """def aligned_batches(items,size,offset):
    if type(size) is not int or size<=0: raise ValueError()
    out=[]
    for i,v in enumerate(items):
        if not out or (offset+i)%size==0: out.append([])
        out[-1].append(v)
    return out
""",
    "tool_name": """from lazybridge import Tool
def make_tool(fn):
    return Tool.wrap(fn,name='measure',description='Measure a value')
""",
    "mock_plan": """from lazybridge import Agent, Plan, Step, MockAgent
def run_mock_plan(text):
    echo=MockAgent(text,name='echo')
    return Agent(name='pipeline',engine=Plan(Step('echo')),tools=[echo])('run').text()
""",
    "envelope_context": """from lazybridge import Envelope
def make_envelope(task,context):
    return Envelope(task=task,context=context,payload={'status':'queued'})
""",
    "review_agent": """from lazybridge import Agent
def make_review_agent(engine,judge):
    return Agent(engine=engine,verify=judge,max_verify=2,name='checked')
""",
    "step_context": """from lazybridge import Step, from_prev, from_step
def make_context_step():
    return Step('writer',task=from_prev,context=from_step('research'))
""",
}


def scripts(task, arm):
    """Four repeatable scenarios: approve, repair, regress, and false approval."""
    from .tasks import TASKS

    good = SOLUTIONS[task.id.split("_", 2)[2]]
    bad = "raise RuntimeError('scripted incorrect candidate')\n"
    scenario = TASKS.index(task) % 4
    if arm in ("A", "B"):
        return [bad if scenario in (1, 3) else good], []
    if scenario == 0:
        return [good], ["approved"]
    if scenario == 1:
        return [bad, good], ["rejected: fix the implementation", "approved"]
    if scenario == 2:
        return [good, bad], ["rejected: scripted harmful feedback", "rejected: still wrong"]
    return [bad], ["approved: scripted missed error"]
