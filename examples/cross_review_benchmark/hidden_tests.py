# ruff: noqa: RUF001
"""Evaluator-only pytest functions. Never imported into model prompts.

Public for reproducibility, hidden during runs. Each grader gets only its task test.
"""

HEADER = """import copy
import math
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pytest
"""

TESTS = {
    "gap_runs": """
def test_hidden():
    assert gap_runs([]) == []
    x = [4,-2,3,-1,3,8,0]; before = x[:]
    assert gap_runs(x) == [(-2,0),(3,4),(8,8)]
    assert x == before
    assert gap_runs([10,10]) == [(10,10)]
    assert gap_runs([10**30,10**30+1]) == [(10**30,10**30+1)]
""",
    "clipped_windows": """
def test_hidden():
    assert clipped_windows('a😀éz',3,2) == ['a😀e','éz','z']
    assert clipped_windows('abc',5,4) == ['abc']
    assert clipped_windows('',1,1) == []
    assert clipped_windows('abcd',2,1) == ['ab','bc','cd','d']
    for w,s in [(0,1),(1,0),(-2,1),(True,1),(1,False),(1.0,1)]:
        with pytest.raises(ValueError): clipped_windows('',w,s)
""",
    "cent_parts": """
def test_hidden():
    w=[1,1,1]; before=w[:]
    assert cent_parts('0.025',w) == [1,1,1]
    assert cent_parts('0.05',w) == [2,2,1]
    assert cent_parts('0.07',[0,1,2]) == [0,2,5]
    assert cent_parts('0',[1]) == [0]
    assert cent_parts('123456789012345678901234567890.01',[1]) == [12345678901234567890123456789001]
    assert w == before
    for a,weights in [('NaN',[1]),('Infinity',[1]),('-0.001',[1]),('x',[1]),('1',[]),('1',[0,0]),('1',[-1,2]),('1',[True]),('1',[1.5])]:
        with pytest.raises(ValueError): cent_parts(a,weights)
""",
    "fold_counts": """
def test_hidden():
    x=['É','é','Straße','STRASSE','','','İ','i̇']; before=x[:]
    assert fold_counts(x) == [('é',2),('strasse',2),('',2),('i̇',2)]
    assert x == before
    assert fold_counts([]) == []
    assert fold_counts(['Σ','ς','σ']) == [('σ',3)]
""",
    "utc_day": """
def test_hidden():
    assert utc_day('2024-03-01T00:15:00+02:00') == '2024-02-29'
    assert utc_day('2023-12-31T23:30:00-03:00') == '2024-01-01'
    assert utc_day('2024-01-01T00:00:00Z') == '2024-01-01'
    for s in ['2024-01-01T00:00:00','2024-01-01','garbage']:
        with pytest.raises(ValueError): utc_day(s)
""",
    "month_anchor": """
def test_hidden():
    assert month_anchor(date(2024,1,31),1) == date(2024,2,29)
    assert month_anchor(date(2024,1,31),2) == date(2024,3,31)
    assert month_anchor(date(2024,3,31),-13) == date(2023,2,28)
    assert month_anchor(date(2024,2,29),12) == date(2025,2,28)
    assert month_anchor(date(2024,2,29),0) == date(2024,2,29)
    for d,n in [(date(1,1,1),-1),(date(9999,12,31),1)]:
        with pytest.raises(ValueError): month_anchor(d,n)
""",
    "quiet_ranges": """
def test_hidden():
    x=[(6,8),(-5,2),(2,4),(3,6),(10,12),(9,9)]; before=x[:]
    assert quiet_ranges(x,0,11) == [(8,10)]
    assert x == before
    assert quiet_ranges([],0,3) == [(0,3)]
    assert quiet_ranges([(4,7)],0,3) == [(0,3)]
    assert quiet_ranges([(0,3)],0,3) == []
    assert quiet_ranges([],3,3) == []
    for b,a,z in [([],4,3),([(2,1)],0,3),([(2,1)],0,0)]:
        with pytest.raises(ValueError): quiet_ranges(b,a,z)
""",
    "escaped_fields": r"""
def test_hidden():
    assert escaped_fields('') == ['']
    assert escaped_fields('|a||') == ['','a','','']
    assert escaped_fields(r'a\|b|c') == ['a|b','c']
    assert escaped_fields(r'x\\|y\q') == ['x\\','yq']
    assert escaped_fields('😀|é') == ['😀','é']
    with pytest.raises(ValueError): escaped_fields('abc' + chr(92))
""",
    "carry_round": """
def test_hidden():
    x=['0.5','0.5','0.5','0.5']; before=x[:]
    assert carry_round(x) == [0,1,1,0]
    assert carry_round(['-0.5','-0.5','-0.5']) == [0,-1,-1]
    assert carry_round(['1.49','-0.99','0.01']) == [1,-1,1]
    assert carry_round(['1000000000000000000000000000000','0.6']) == [1000000000000000000000000000000,1]
    assert x == before
    assert carry_round([]) == []
    for s in ['NaN','Infinity','-Infinity']:
        with pytest.raises(ValueError): carry_round([s])
""",
    "recent_keys": """
def test_hidden():
    x=['a','b','a','c']; before=x[:]
    assert recent_keys(x,2) == ['a','c']
    assert recent_keys(x,0) == []
    assert recent_keys(['','😀','','z'],2) == ['','z']
    assert recent_keys([],5) == []
    assert x == before
    for c in [-1,True,1.5]:
        with pytest.raises(ValueError): recent_keys([],c)
""",
    "braid_rows": """
def test_hidden():
    x=[[1,2,3],[],[4],[5,6,7,8]]; before=copy.deepcopy(x)
    assert braid_rows(x) == [1,4,5,6,2,3,7,8]
    assert x == before
    assert braid_rows([]) == []
    assert braid_rows([[],[]]) == []
    assert braid_rows([[1],[2,3]]) == [1,2,3]
""",
    "stable_scores": """
def test_hidden():
    x=[('n1',math.nan),('a',-0.0),('b',0.0),('top',math.inf),('low',-math.inf),('n2',math.nan),('one',1)]
    before=list(x)
    assert stable_scores(x) == ['top','one','a','b','low','n1','n2']
    assert x == before
    assert stable_scores([]) == []
    assert stable_scores([('a',2),('b',2),('a',2)]) == ['a','b','a']
""",
    "near_duplicate": """
def test_hidden():
    x=[1,2,3,1]; before=x[:]
    assert near_duplicate(x,3) is True
    assert near_duplicate(x,2) is False
    assert near_duplicate([1,1],0) is False
    assert near_duplicate([1,2,1,1],1) is True
    assert near_duplicate([],100) is False
    assert x == before
    with pytest.raises(ValueError): near_duplicate([],-1)
""",
    "path_reduce": r"""
def test_hidden():
    for p,want in [('', '.'),('/', '/'),('/../../a//./b/..','/a'),('../../a/../..','../../..'),('a/../../b','../b'),('a/..','.'),(r'a\b/..','.'),('.../..','.'),('///a///','/a')]:
        assert path_reduce(p) == want
""",
    "select_rank": """
def test_hidden():
    x=[3,1,1,-2]; before=x[:]
    assert [select_rank(x,i) for i in range(4)] == [-2,1,1,3]
    assert x == before
    for a,r in [([],0),(x,-1),(x,4)]:
        with pytest.raises(IndexError): select_rank(a,r)
""",
    "dense_labels": """
def test_hidden():
    x=['','é','é','','😀','é']; before=x[:]
    assert dense_labels(x) == [0,1,2,0,3,1]
    assert dense_labels([]) == []
    assert x == before
""",
    "retry_slots": """
def test_hidden():
    x=[0,2,0,3,1]; before=x[:]
    assert retry_slots(x,5) == [0,2,2,5]
    assert retry_slots([0,0,1],0) == [0,0]
    assert retry_slots([],0) == []
    assert x == before
    for a,b in [([], -1),([100,-1],0),([-1],5)]:
        with pytest.raises(ValueError): retry_slots(a,b)
""",
    "end_match": """
def test_hidden():
    x=['','z','éz','éz']; before=x[:]
    assert end_match('aéz',x) == 'éz'
    assert end_match('', ['x','']) == ''
    assert end_match('abc',[]) is None
    assert end_match('ABC',['bc']) is None
    assert end_match('é',['́','é']) == 'é'
    assert x == before
""",
    "local_midnights": """
def test_hidden():
    z=ZoneInfo('America/New_York')
    a=datetime(2024,3,9,tzinfo=z); b=datetime(2024,3,12,tzinfo=z)
    out=local_midnights(a,b)
    assert out == [datetime(2024,3,d,tzinfo=z) for d in [9,10,11]]
    assert all(d.tzinfo is z for d in out)
    assert local_midnights(a,a) == []
    assert local_midnights(a.replace(hour=1),a.replace(day=10)) == []
    # Wall-clock order and instant order differ during the fold.
    c=datetime(2024,11,3,1,45,tzinfo=z,fold=0)
    d=datetime(2024,11,3,1,15,tzinfo=z,fold=1)
    assert local_midnights(c,d) == []
    for s,t in [(d,c),(datetime(2024,1,1),b),(a,datetime(2024,3,12,tzinfo=timezone.utc))]:
        with pytest.raises(ValueError): local_midnights(s,t)
    z2=ZoneInfo('Pacific/Apia')
    a2=datetime(2011,12,29,tzinfo=z2); b2=datetime(2012,1,1,tzinfo=z2)
    assert local_midnights(a2,b2) == [datetime(2011,12,day,tzinfo=z2) for day in [29,30,31]]
""",
    "aligned_batches": """
def test_hidden():
    x=[0,1,2,3,4,5,6]; before=x[:]
    assert aligned_batches(x,3,1) == [[0,1],[2,3,4],[5,6]]
    assert aligned_batches(x,3,-1) == [[0],[1,2,3],[4,5,6]]
    assert aligned_batches(x,3,-6) == [[0,1,2],[3,4,5],[6]]
    assert aligned_batches([],2,0) == []
    out=aligned_batches(x,3,0); out[0].append(99)
    assert x == before
    for n in [0,-1,True,1.5]:
        with pytest.raises(ValueError): aligned_batches([],n,0)
""",
    "tool_name": """
def test_hidden():
    from lazybridge import Tool
    def f(x: int) -> int: return x+1
    t=make_tool(f)
    assert isinstance(t,Tool)
    assert t.name == 'measure' and t.description == 'Measure a value'
    import asyncio
    assert asyncio.run(t.run(x=2)) == 3
""",
    "mock_plan": """
def test_hidden(monkeypatch):
    from lazybridge import Plan, MockAgent
    seen={'plan':0,'mock':0}
    plan_run=Plan.run; mock_run=MockAgent.run
    async def count_plan(self,*args,**kwargs):
        seen['plan']+=1
        return await plan_run(self,*args,**kwargs)
    async def count_mock(self,*args,**kwargs):
        seen['mock']+=1
        return await mock_run(self,*args,**kwargs)
    monkeypatch.setattr(Plan,'run',count_plan)
    monkeypatch.setattr(MockAgent,'run',count_mock)
    for s in ['', 'hello', 'é😀']:
        assert run_mock_plan(s) == s
    assert seen == {'plan':3,'mock':3}
""",
    "envelope_context": """
def test_hidden():
    from lazybridge import Envelope
    e=make_envelope('task😀','context')
    assert isinstance(e,Envelope)
    assert e.task == 'task😀' and e.context == 'context'
    assert e.payload == {'status':'queued'}
""",
    "review_agent": """
def test_hidden():
    from lazybridge import Agent, MockAgent
    from types import SimpleNamespace
    engine=SimpleNamespace(); judge=MockAgent('approved')
    a=make_review_agent(engine,judge)
    assert isinstance(a,Agent)
    assert a.name == 'checked' and a.engine is engine
    assert a.verify is judge and a.max_verify == 2
""",
    "step_context": """
def test_hidden():
    from lazybridge import Step, from_prev, from_step
    s=make_context_step()
    assert isinstance(s,Step)
    assert s.target == 'writer'
    assert s.task is from_prev
    assert s.context == from_step('research')
""",
}


def test_source(task):
    function_name = task.spec.split("(", 1)[0].removeprefix("def ").strip()
    return HEADER + f"from candidate import {function_name}\n" + TESTS[task.id.split("_", 2)[2]]
