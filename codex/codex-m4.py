#!/usr/bin/env python3
"""Start the dedicated profile without touching the shared daemon."""
import os
import sys


if __name__ == "__main__":
    os.execvp("codex", ["codex", "--no-daemon", "--profile", "m4-qwen", *sys.argv[1:]])
