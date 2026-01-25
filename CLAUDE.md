# Claude Code Development Guide

## Project Policies

1. Yolo dataset doesnt have classes.txt
1. Always check project with ruff format, ruff check and mypy before finalize work 
1. Use pathlib, try minimally use os and open for working with paths
1. Always prefix var name with _  if it will not used.
1. prefer way to use pillow is 'from pillow import Image'
1. always use loguru for logging not print
1. always write to README.md in russian
1. Use pydantic schemas for all configs
1. not check output logs in cli tests
1. Wirte README.md as user friendly intro, not as wall of documentation
1. Try avoid using strings instead ofg types if it's not strictly neccessary. Do it for bypassing ruff checks it's not strictly necessary use other.

### Package Management

**Always use `uv` interface, NOT `pip` or `uv pip`.**
Incorrect commands (DO NOT USE):
```bash
pip install ...           # ❌ Don't use
uv pip install ...        # ❌ Don't use
pip install -e .          # ❌ Don't use
```

### Running Commands
use `uv run`:
```bash
uv run pytest tests/
uv run python3 -m pytest tests/
```
