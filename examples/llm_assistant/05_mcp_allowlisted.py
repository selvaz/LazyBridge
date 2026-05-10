"""MCP server with an explicit allow-list — the canonical safe shape.

Since 0.7.9 both ``MCP.stdio()`` and ``MCP.http()`` are deny-by-default:
omitting both ``allow=`` and ``deny=`` raises ``ValueError`` at
construction.  Pass an explicit fnmatch glob list (matched against the
namespaced ``"<server-name>.<tool>"``) so the LLM sees only the
sub-surface you've audited.
"""

from __future__ import annotations

from lazybridge.ext.mcp import MCP


def main() -> None:
    # NB: this example doesn't actually spawn npx — the construction
    # path is what demonstrates the canonical safety posture.  In real
    # code put `fs` into a real ``Agent(engine=LLMEngine(...), tools=[fs])``.
    try:
        fs = MCP.stdio(
            "fs",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
            allow=["fs.read_*", "fs.list_*"],  # read-only filesystem
            deny=["fs.delete_*", "fs.write_*"],  # belt + braces
        )
    except (FileNotFoundError, OSError):
        print("(npx not installed — the construction path is the point)")
        return

    print("MCPServer name:", fs.name)
    print("MCPServer allow:", fs._allow)
    print("MCPServer deny:", fs._deny)
    print("\nTo use it:\n  agent = Agent(engine=LLMEngine('claude-opus-4-7'), tools=[fs])")

    # Anti-example: omitting allow=/deny= raises since 0.7.9.
    try:
        MCP.stdio("unsafe", command="echo", args=["hi"])
    except ValueError as e:
        print("\nOmitting allow=/deny= now raises:")
        print(" ", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
