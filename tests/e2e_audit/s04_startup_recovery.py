"""Recover a standard delegate after its owning process is killed."""
import asyncio
from pathlib import Path
import sys
import time

from _harness import WORK, child, scenario, start_trace
from lazybridge import Store
from lazybridge.ext.delegation import JobRegistry, make_background_delegate

NAME = Path(__file__).stem


class BlockingEngine:
    async def run(self, env, **kwargs):
        (WORK / NAME / 'worker-ready').touch()
        await asyncio.Event().wait()


async def owner(directory):
    with Store(db=str(directory / 'jobs.sqlite')) as store:
        tasks = set()
        tool = make_background_delegate(tool_name='delegate', label='worker',
            engine_factory=BlockingEngine, registry=JobRegistry(store), background_tasks=tasks,
            notify=None, doc='Run a background job')
        await tool.run(objective='blocked job')
        await asyncio.Event().wait()


def run(directory):
    process = child(__file__, 'owner')
    try:
        deadline = time.monotonic() + 15
        while not (directory / 'worker-ready').exists():
            assert process.poll() is None, f'owner exited {process.returncode}'
            assert time.monotonic() < deadline, 'worker did not start'
            time.sleep(0.05)
        with Store(db=str(directory / 'jobs.sqlite')) as store:
            registry = JobRegistry(store)
            keys = [key for key in store if key.startswith('delegation:job:')]
            assert len(keys) == 1, keys
            job = store.read(keys[0])
            assert job['status'] == 'running' and job['execution_started']
            assert registry.reclaim_interrupted() == []
            process.kill()
            process.wait(timeout=10)
            assert registry.reclaim_interrupted() == [job['job_id']]
            assert registry.find(job['job_id'])['status'] == 'interrupted'
            registry.write('recent', 'work', tool_name='delegate', status='running', created_at=time.time())
            registry.write('stale', 'work', tool_name='delegate', status='running', created_at=time.time()-901)
            assert registry.reclaim_interrupted() == ['stale']
            assert registry.find('recent')['status'] == 'running'
            assert registry.reclaim_interrupted(reclaim_unowned=True) == ['recent']
        return 'Live job preserved; killed owner interrupted; recent, stale and override cases verified.'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        directory = WORK / NAME
        stream = start_trace(directory)
        try:
            asyncio.run(owner(directory))
        finally:
            sys.setprofile(None)
            stream.close()
    else:
        sys.exit(scenario(NAME, 'A6 startup recovery',
            'Preserve live jobs and reclaim dead or stale owners with explicit override.', run))
