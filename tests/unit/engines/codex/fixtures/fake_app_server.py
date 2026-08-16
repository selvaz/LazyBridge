"""Standalone stand-in for ``codex app-server``'s JSON-RPC stdio protocol.

Spawned as a real subprocess by ``test_app_server.py`` so
``CodexAppServerClient``'s actual pipe/JSON-RPC read loop is exercised end
to end, without requiring a real, authenticated Codex installation.

The message shapes here were captured from a live ``codex app-server``
(codex-cli 0.148.0): usage arrives only via ``thread/tokenUsage/updated``
notifications, ``turn/completed`` carries no usage and no cost, and the
server numbers its own ``item/tool/call`` requests from 0 with a counter
independent of the client's.

Usage: ``python fake_app_server.py <scenario>`` where scenario is one of
"happy", "turn_failed", "error_notification", "id_collision",
"developer_instructions", "huge_message" or "exit_immediately" (see
``test_app_server.py``).
"""

from __future__ import annotations

import json
import sys


def read_message() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise SystemExit("fake_app_server: stdin closed unexpectedly")
    return json.loads(line)


def write_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def token_usage(input_tokens: int, output_tokens: int, turn_id: str = "turn-1") -> dict:
    # ``total`` is cumulative over the THREAD, not the turn (verified live:
    # 15137 after turn 1, 30292 after turn 2), which is why it is tagged with
    # a turn id and why the client subtracts a baseline on a resumed thread.
    return {
        "method": "thread/tokenUsage/updated",
        "params": {
            "threadId": "thread-1",
            "turnId": turn_id,
            "tokenUsage": {
                "total": {
                    "totalTokens": input_tokens + output_tokens,
                    "inputTokens": input_tokens,
                    "cachedInputTokens": 0,
                    "outputTokens": output_tokens,
                    "reasoningOutputTokens": 0,
                },
                "last": {
                    "totalTokens": input_tokens + output_tokens,
                    "inputTokens": input_tokens,
                    "cachedInputTokens": 0,
                    "outputTokens": output_tokens,
                    "reasoningOutputTokens": 0,
                },
                "modelContextWindow": 258400,
            },
        },
    }


def native_review_main() -> None:
    """``review/start``: the typed-target review path.

    Captured from codex-cli 0.148.0: the response carries a ``turn`` exactly
    like ``turn/start``, the findings arrive as ONE ``agentMessage`` in
    ``turn/completed`` (severity-tagged text, not a structured findings array),
    and an inline review streams no ``item/agentMessage/delta`` at all.
    """
    init = read_message()
    assert init["method"] == "initialize", init
    write_message({"id": init["id"], "result": {"userAgent": "fake", "platformOs": "test"}})
    assert read_message()["method"] == "initialized"

    start = read_message()
    assert start["method"] == "thread/start", start
    write_message({"id": start["id"], "result": {"thread": {"id": "thread-1"}}})

    review = read_message()
    assert review["method"] == "review/start", review
    params = review["params"]
    assert params["threadId"] == "thread-1", params
    assert params["target"] == {"type": "baseBranch", "branch": "main"}, params
    # inline only: a detached review completes on ANOTHER thread and raises an
    # approval request the parent never sees (measured — it hung).
    assert params["delivery"] == "inline", params
    write_message(
        {"id": review["id"], "result": {"reviewThreadId": "thread-1", "turn": {"id": "turn-1", "status": "inProgress"}}}
    )
    write_message(token_usage(70, 9))
    write_message(
        {
            "method": "turn/completed",
            "params": {
                "turn": {
                    "id": "turn-1",
                    "status": "completed",
                    "items": [{"type": "agentMessage", "text": "- [P1] Preserve the empty-input result — stats.py:3"}],
                }
            },
        }
    )


def resume_main(scenario: str) -> None:
    """The ``thread/resume`` path: a durable thread being picked up again.

    Mirrors what the live App Server does (captured from codex-cli 0.148.0):
    on resume it replays a ``thread/tokenUsage/updated`` carrying the
    *previous* turn's id and the thread-cumulative total, so this turn's cost
    is only the delta on top of it.
    """
    init = read_message()
    assert init["method"] == "initialize", init
    write_message({"id": init["id"], "result": {"userAgent": "fake", "platformOs": "test"}})
    assert read_message()["method"] == "initialized"

    resume = read_message()
    assert resume["method"] == "thread/resume", resume
    params = resume["params"]
    assert params["threadId"] == "thread-1", params
    # Everything is re-supplied on resume — the tool callbacks in particular
    # live in this process and cannot be inherited from the one that started
    # the thread.
    assert params["cwd"] is not None, params
    assert params["sandbox"] == "read-only", params
    assert "ephemeral" not in params, params
    dynamic_tools = params.get("dynamicTools", [])
    assert dynamic_tools, "resume must re-register dynamic tools"
    write_message({"id": resume["id"], "result": {"thread": {"id": "thread-1"}}})
    # History: 100 in / 20 out already spent on turn-1, before this turn.
    write_message(token_usage(100, 20, turn_id="turn-1"))

    turn_start = read_message()
    assert turn_start["method"] == "turn/start", turn_start

    if scenario == "resume_dies_before_ack":
        # Accepted the request, died before answering it: from outside, "was
        # this turn committed?" is unanswerable — so it must not be retried.
        return

    if scenario == "rejects_the_turn":
        write_message(
            {"id": turn_start["id"], "error": {"code": -32602, "message": "invalid review target"}}
        )
        return

    if scenario == "resume_completed_before_ack":
        # OUR completion outrunning its own acknowledgement. Dropping it hangs
        # the call until timeout and then reports a turn that demonstrably
        # finished as "outcome unknown" (reproduced by the Claude reviewer).
        write_message(token_usage(155, 27, turn_id="turn-2"))
        write_message(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "id": "turn-2",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "resumed answer"}],
                    }
                },
            }
        )
        write_message({"id": turn_start["id"], "result": {"turn": {"id": "turn-2", "status": "inProgress"}}})
        return

    if scenario == "resume_replay_before_ack":
        # The window the ack normally closes: a completion for an OLD turn,
        # replayed before we learn our own turn id.
        write_message(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "STALE"}],
                    }
                },
            }
        )

    if scenario == "resume_usage_before_ack":
        # ...and the mirror image: OUR turn's first usage report, arriving
        # before the ack. Counting it as history would undercount the turn.
        write_message(token_usage(140, 24, turn_id="turn-2"))

    write_message({"id": turn_start["id"], "result": {"turn": {"id": "turn-2", "status": "inProgress"}}})

    if scenario == "resume_dies_mid_turn":
        # Accepted the turn, then died: outcome unknown, not retryable.
        return

    if scenario == "resume_stale_turn":
        # A completion for the turn that ran *before* this one. Taking it
        # would hand the caller a previous answer as this call's result.
        write_message(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "STALE"}],
                    }
                },
            }
        )

    write_message(token_usage(155, 27, turn_id="turn-2"))
    write_message(
        {
            "method": "turn/completed",
            "params": {
                "turn": {
                    "id": "turn-2",
                    "status": "completed",
                    "items": [{"type": "agentMessage", "text": "resumed answer"}],
                }
            },
        }
    )


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "happy"
    if scenario == "native_review":
        native_review_main()
        return
    if scenario in (
        "resume",
        "resume_stale_turn",
        "resume_dies_mid_turn",
        "resume_replay_before_ack",
        "resume_dies_before_ack",
        "resume_usage_before_ack",
        "resume_completed_before_ack",
        "rejects_the_turn",
    ):
        resume_main(scenario)
        return

    if scenario == "exit_immediately":
        return

    init = read_message()
    assert init["method"] == "initialize", init
    write_message({"id": init["id"], "result": {"userAgent": "fake", "platformOs": "test"}})

    initialized = read_message()
    assert initialized["method"] == "initialized", initialized

    thread_start = read_message()
    assert thread_start["method"] == "thread/start", thread_start
    params = thread_start["params"]
    # Lock in the enum spelling the real CLI accepts — "readOnly" is
    # rejected live with "unknown variant `readOnly`".
    expected_sandbox = "workspace-write" if scenario == "command_approval" else "read-only"
    expected_approval = "on-request" if scenario == "command_approval" else "never"
    assert params["sandbox"] == expected_sandbox, params
    assert params["approvalPolicy"] == expected_approval, params
    assert params["ephemeral"] is True, params
    if scenario == "developer_instructions":
        assert params["developerInstructions"] == "Be concise.", params
    dynamic_tools = params.get("dynamicTools", [])
    write_message({"id": thread_start["id"], "result": {"thread": {"id": "thread-1"}}})

    turn_start = read_message()
    assert turn_start["method"] == "turn/start", turn_start
    turn_started_result = {"id": turn_start["id"], "result": {"turn": {"id": "turn-1", "status": "inProgress"}}}
    if scenario != "id_collision":
        # Real ordering: turn/start is acknowledged immediately, long before
        # the turn finishes.
        write_message(turn_started_result)

    if scenario == "command_approval":
        write_message(
            {
                "id": 0,
                "method": "item/commandExecution/requestApproval",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "itemId": "item-1",
                    "command": "git status",
                    "cwd": "C:/work/project",
                    "reason": "Inspect the worktree",
                },
            }
        )
        approval = read_message()
        assert approval == {"id": 0, "result": {"decision": "acceptForSession"}}, approval
        write_message(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "approved"}],
                    }
                },
            }
        )
    elif scenario in ("happy", "id_collision", "developer_instructions"):
        if dynamic_tools:
            # Server-side request ids start at 0 and are independent of the
            # client's counter; under "id_collision" this id is deliberately
            # one the client still has in flight (its turn/start).
            call_id = turn_start["id"] if scenario == "id_collision" else 0
            write_message(
                {
                    "id": call_id,
                    "method": "item/tool/call",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "callId": "exec-1",
                        "namespace": None,
                        "tool": dynamic_tools[0]["name"],
                        "arguments": {"symbol": "AMZN"},
                    },
                }
            )
            tool_response = read_message()
            assert tool_response["id"] == call_id, tool_response
            assert tool_response["result"]["success"] is True, tool_response

        if scenario == "id_collision":
            write_message(turn_started_result)

        write_message(token_usage(42, 3))
        write_message({"method": "item/agentMessage/delta", "params": {"delta": "AMZN is "}})
        write_message({"method": "item/agentMessage/delta", "params": {"delta": "123.45"}})
        write_message(token_usage(55, 7))

        write_message(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "error": None,
                        "items": [{"type": "agentMessage", "text": "AMZN is 123.45", "phase": "final_answer"}],
                    },
                },
            }
        )
    elif scenario == "huge_message":
        # One notification far past StreamReader's 64 KiB default limit —
        # what a real turn sends the moment it reads a large file or a real
        # `git diff`. Without an explicit limit= on the subprocess, readline()
        # raises "Separator is found, but chunk is longer than limit" and the
        # whole turn dies.
        write_message(
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": "x" * (256 * 1024)},
            }
        )
        write_message(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "big", "phase": "final_answer"}],
                    }
                },
            }
        )
    elif scenario == "exit_mid_turn":
        # Acknowledged turn/start, then died without ever sending
        # turn/completed — the path that hangs forever when the client has no
        # request_timeout, unless the reader resolves the completion waiter.
        return
    elif scenario == "turn_failed":
        write_message(
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "failed", "error": {"message": "rate limited"}}},
            }
        )
    elif scenario == "error_notification":
        # A retryable error must not end the run...
        write_message(
            {
                "method": "error",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "willRetry": True,
                    "error": {"message": "transient blip"},
                },
            }
        )
        # ...a terminal one must.
        write_message(
            {
                "method": "error",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "willRetry": False,
                    "error": {"message": "stream disconnected"},
                },
            }
        )
    else:
        raise SystemExit(f"fake_app_server: unknown scenario {scenario!r}")


if __name__ == "__main__":
    main()
