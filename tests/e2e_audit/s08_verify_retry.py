"""An ordinary producer error is judged and retried with feedback."""
import asyncio
from pathlib import Path
import sys

from _harness import scenario
from lazybridge import Agent, Envelope
from lazybridge.envelope import ErrorInfo


async def check(directory):
    inputs = []
    judgments = []
    class Producer:
        async def run(self, env, **kwargs):
            inputs.append(env.context)
            if len(inputs) == 1:
                return Envelope(task=env.task, error=ErrorInfo(type='Failure', message='ordinary error'))
            return Envelope(task=env.task, payload='success')
    def judge(text):
        judgments.append(text)
        return 'rejected: fix the error' if len(judgments) == 1 else 'approved'
    result = await Agent(engine=Producer(), name='producer', verify=judge, max_verify=2).run('work')
    assert result.ok and result.text() == 'success', result
    assert len(inputs) == len(judgments) == 2
    assert 'fix the error' in inputs[1]
    return 'Ordinary error judged; feedback reached second attempt; final result succeeded.'


if __name__ == '__main__':
    sys.exit(scenario(Path(__file__).stem, 'A7 verify retry',
        'Judge an ordinary failure and retry once with feedback.', lambda d: asyncio.run(check(d))))
