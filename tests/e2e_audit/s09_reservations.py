"""Cancellation during batch admission settles every granted reservation once."""
import asyncio
from pathlib import Path
import sqlite3
import sys

from _harness import scenario
from lazybridge import Store
from lazybridge.ext.delegation import JobRegistry, make_parallel_delegate


async def check(directory):
    ledger = sqlite3.connect(directory / 'ledger.sqlite')
    ledger.execute('CREATE TABLE events (reservation INTEGER, action TEXT)')
    ledger.commit()
    entered, finish, blocked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class Grant:
        allowed = True
        def __init__(self, number):
            self.number = number
        async def refund(self):
            ledger.execute('INSERT INTO events VALUES (?, ?)', (self.number, 'refund'))
            ledger.commit()
            if self.number == 0:
                entered.set()
                await finish.wait()
    count = 0
    async def gate():
        nonlocal count
        count += 1
        if count == 4:
            blocked.set()
            await asyncio.Event().wait()
        return Grant(count - 1)
    with Store(db=str(directory / 'jobs.sqlite')) as store:
        tool = make_parallel_delegate(engine_factory=object, registry=JobRegistry(store),
            background_tasks=set(), doc='delegate', admission_gate=gate)
        batch = asyncio.create_task(tool.run(objectives=['one', 'two', 'three', 'four']))
        try:
            await asyncio.wait_for(blocked.wait(), 5)
            batch.cancel()
            await asyncio.wait_for(entered.wait(), 5)
            batch.cancel()
            await asyncio.sleep(0)
        finally:
            finish.set()
            [result] = await asyncio.wait_for(asyncio.gather(batch, return_exceptions=True), 5)
        assert isinstance(result, asyncio.CancelledError)
        rows = ledger.execute('SELECT reservation, action FROM events ORDER BY reservation').fetchall()
        assert rows == [(0, 'refund'), (1, 'refund'), (2, 'refund')], rows
        assert not list(store.items(prefix='delegation:job:'))
    ledger.close()
    return 'Three acquired reservations refunded once despite repeated cancellation; no job started.'


if __name__ == '__main__':
    sys.exit(scenario(Path(__file__).stem, 'M3 reservations',
        'Cancellation during scheduling cleanup refunds each outstanding reservation once.',
        lambda d: asyncio.run(check(d))))
