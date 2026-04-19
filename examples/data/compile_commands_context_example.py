"""Compile-commands parsing example.

This shows how build metadata is recovered from `compile_commands.json`.
The problem it solves is missing compiler context: C and C++ files are harder
to enrich correctly if the data pipeline does not know include paths and modes.
"""

from __future__ import annotations

import json
import os
import shlex


def split_shell_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def sanitize_compile_args(argv: list[str], filepath: str) -> tuple[list[str], str | None]:
    compiler = argv[0] if argv else None
    keep: list[str] = []
    i = 1
    while i < len(argv):
        token = argv[i]
        if token in {"-I", "-isystem", "-include", "-x", "-std"} and i + 1 < len(argv):
            keep.extend([token, argv[i + 1]])
            i += 2
            continue
        if token.startswith(("-I", "-D", "-std=", "-Winvalid-")):
            keep.append(token)
        i += 1
    return keep + [filepath], compiler


def parse_compile_commands_entries(entries: list[dict]) -> list[dict[str, object]]:
    """Normalize compile_commands entries into build-context records."""
    normalized: list[dict[str, object]] = []
    for raw in entries:
        directory = str(raw.get("directory", ""))
        filepath = str(raw.get("file", ""))
        arguments = raw.get("arguments")
        argv = [str(arg) for arg in arguments] if isinstance(arguments, list) and arguments else split_shell_command(str(raw.get("command", "")))
        if not filepath or not argv:
            continue
        preferred = os.path.normpath(os.path.join(directory, filepath)) if directory and not os.path.isabs(filepath) else os.path.normpath(filepath)
        compile_args, compiler = sanitize_compile_args(argv, preferred)
        normalized.append(
            {
                "filepath": preferred,
                "compile_args": compile_args,
                "build_info": {
                    "build_system": "compile_commands",
                    "source": "compile_commands",
                    **({"compiler": compiler} if compiler else {}),
                },
            }
        )
    return normalized


def load_compile_commands_text(text: str | None) -> list[dict[str, object]] | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    rows = parse_compile_commands_entries(data)
    return rows or None
