from toolery.tools.registry import ToolSpec, register

BASH_EXEC = ToolSpec(
    name="bash_exec",
    description="Execute a shell command and return stdout, stderr, exit code, and duration.",
    json_schema={
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Execute a shell command and return stdout, stderr, exit code, and duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout_s": {"type": "integer", "description": "Timeout in seconds.", "default": 30},
                    "cwd": {"type": "string", "description": "Working directory.", "default": "/home/user"},
                },
                "required": ["command"],
            },
        },
    },
)

register(BASH_EXEC)


PROCESS_START = ToolSpec(
    name="process_start",
    description="Start a background process and return its pid.",
    json_schema={
        "type": "function",
        "function": {
            "name": "process_start",
            "description": "Start a background process and return its pid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "name": {"type": "string", "default": ""},
                },
                "required": ["command"],
            },
        },
    },
)

register(PROCESS_START)


PROCESS_STATUS = ToolSpec(
    name="process_status",
    description="Poll the status of a running process by pid.",
    json_schema={
        "type": "function",
        "function": {
            "name": "process_status",
            "description": "Poll the status of a running process by pid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer"},
                },
                "required": ["pid"],
            },
        },
    },
)

register(PROCESS_STATUS)


PROCESS_KILL = ToolSpec(
    name="process_kill",
    description="Send a signal to a running process.",
    json_schema={
        "type": "function",
        "function": {
            "name": "process_kill",
            "description": "Send a signal to a running process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer"},
                    "signal": {"type": "string", "enum": ["TERM", "KILL", "INT", "HUP"], "default": "TERM"},
                },
                "required": ["pid"],
            },
        },
    },
)

register(PROCESS_KILL)


PROCESS_SEND_INPUT = ToolSpec(
    name="process_send_input",
    description="Send text to the stdin of a running process (e.g. answer an interactive prompt).",
    json_schema={
        "type": "function",
        "function": {
            "name": "process_send_input",
            "description": "Send text to the stdin of a running process (e.g. answer an interactive prompt).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["pid", "text"],
            },
        },
    },
)

register(PROCESS_SEND_INPUT)


READ_TTY_BUFFER = ToolSpec(
    name="read_tty_buffer",
    description="Read the current buffer of a TTY/tmux/screen session, optionally stripping ANSI codes.",
    json_schema={
        "type": "function",
        "function": {
            "name": "read_tty_buffer",
            "description": "Read the current buffer of a TTY/tmux/screen session, optionally stripping ANSI codes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "strip_ansi": {"type": "boolean", "default": False},
                },
                "required": ["session_id"],
            },
        },
    },
)

register(READ_TTY_BUFFER)
