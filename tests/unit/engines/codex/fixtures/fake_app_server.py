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
"happy", "turn_failed", "error_notification" or "id_collision" (see
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


def token_usage(input_tokens: int, output_tokens: int) -> dict:
    return {
        "method": "thread/tokenUsage/updated",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
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


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "happy"

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
    assert params["sandbox"] == "read-only", params
    assert params["approvalPolicy"] == "never", params
    assert params["ephemeral"] is True, params
    dynamic_tools = params.get("dynamicTools", [])
    write_message({"id": thread_start["id"], "result": {"thread": {"id": "thread-1"}}})

    turn_start = read_message()
    assert turn_start["method"] == "turn/start", turn_start
    turn_started_result = {"id": turn_start["id"], "result": {"turn": {"id": "turn-1", "status": "inProgress"}}}
    if scenario != "id_collision":
        # Real ordering: turn/start is acknowledged immediately, long before
        # the turn finishes.
        write_message(turn_started_result)

    if scenario in ("happy", "id_collision"):
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
