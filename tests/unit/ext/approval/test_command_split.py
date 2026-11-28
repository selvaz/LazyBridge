"""Public compound-command splitting contract."""

import pytest

from lazybridge.ext.approval import CommandSegment, CommandSplit, split_command


def test_operators_and_redirection():
    assert split_command("cd /repo; git status 2>/dev/null | head -5 && ls || pwd\nwhoami") == CommandSplit(
        (
            CommandSegment("cd /repo", ";"),
            CommandSegment("git status 2>/dev/null", "|"),
            CommandSegment("head -5", "&&"),
            CommandSegment("ls", "||"),
            CommandSegment("pwd", "\n"),
            CommandSegment("whoami", None),
        )
    )


def test_quotes_and_escapes_without_separators():
    assert split_command(r'''echo '$(literal)' "hello world" x\ y; ls''') == CommandSplit(
        (CommandSegment(r'''echo '$(literal)' "hello world" x\ y''', ";"), CommandSegment("ls", None))
    )


@pytest.mark.parametrize("command", ['echo "a; b"', "echo 'a|b'", 'echo "a&&b"', r"echo a\;b", r"echo a\&&b"])
def test_quoted_or_escaped_separators_are_rejected(command):
    assert split_command(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "echo $(whoami)", 'echo "$(whoami)"', "echo `whoami`", "(ls)",
        "cat <<EOF", "echo > file", "cat < file", "echo >/dev/null.bak",
        "echo 'unfinished", 'echo "unfinished', "echo x\\", "ls & pwd",
        "ls # comment", "ls;", ";ls", "ls || | pwd", "",
    ],
)
def test_unsupported_commands_are_rejected(command):
    assert split_command(command) is None
