"""Conservative splitting of command text for approval-policy consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandOperator = Literal[";", "&&", "||", "|", "\n"]


@dataclass(frozen=True)
class CommandSegment:
    """A stripped command segment and the operator joining it to the next."""

    command: str
    operator: CommandOperator | None


@dataclass(frozen=True)
class CommandSplit:
    """Ordered segments; the final segment has no following operator."""

    segments: tuple[CommandSegment, ...]


def split_command(command: str) -> CommandSplit | None:
    """Split a restricted shell command, or return ``None`` if unsupported.

    Unquoted ``;``, ``&&``, ``||``, ``|`` and newlines join nonempty
    segments. Quotes and backslash escapes are retained in the segment text;
    separators inside quotes or escaped with a backslash are not split.
    Output redirections to exactly ``/dev/null`` (optionally with a numeric
    file descriptor and whitespace after ``>``) are retained. Other
    redirections, command substitution, backticks, unquoted parentheses,
    unbalanced quotes, trailing escapes, empty segments, unquoted ``&`` and
    shell comments are rejected. Single-quoted text is literal.

    This is a text heuristic, not a shell parser or a security boundary. It
    does not interpret expansions, shell dialects, or the meaning of commands.
    Consumers must apply their own policy to every returned segment.
    """
    segments: list[CommandSegment] = []
    start = 0
    i = 0
    quote = ""
    while i < len(command):
        char = command[i]
        if char == "\\" and quote != "'":
            if i + 1 == len(command):
                return None
            i += 2
            continue
        if char == "'" and not quote:
            quote = "'"
        elif char == '"' and not quote:
            quote = '"'
        elif char == quote:
            quote = ""
        elif quote != "'" and (char == "`" or command.startswith("$(", i)):
            return None
        elif not quote:
            if (char in "()<&" or (char == "#" and (i == 0 or command[i - 1].isspace()))) and not command.startswith(
                "&&", i
            ):
                return None
            if char == ">":
                end = i + 1
                while end < len(command) and command[end] in " \t":
                    end += 1
                if not command.startswith("/dev/null", end):
                    return None
                end += len("/dev/null")
                if end < len(command) and command[end] not in " \t\r\n;|&":
                    return None
                i = end
                continue
            operator: CommandOperator | None = None
            if command.startswith("&&", i):
                operator = "&&"
            elif command.startswith("||", i):
                operator = "||"
            elif char in ";|\n":
                operator = char  # type: ignore[assignment]
            if operator is not None:
                segment = command[start:i].strip()
                if not segment:
                    return None
                segments.append(CommandSegment(segment, operator))
                i += len(operator)
                start = i
                continue
        i += 1
    segment = command[start:].strip()
    if quote or not segment:
        return None
    segments.append(CommandSegment(segment, None))
    return CommandSplit(tuple(segments))
