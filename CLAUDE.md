# Claude Code Development Guide

## Project Policies

### Archtecture and code
1. Yolo dataset doesnt have classes.txt

### Linters
1. prefer automatically fix with 'ruff format && ruff check --fix' before manual fix. 
1. Always check project with linters after proiimary work.
Run 
```
ruff format .
ruff check .
mypy .
pytest .
```
then fix all erros. 
1. tets also must be checked with linters
1. Explicitly ask to manual add 'type: ignore / noqa' with arguments or fix linter  errors.
1. Use pathlib, try minimally use os and open for working with paths
1. Always prefix var name with _  if it will not used.
1. Respect ruff PLR and PLC errors. Refactor instead of ignore
1. Do it for bypassing ruff checks it's not strictly necessary use other
1. **Never use hardcoded `/tmp` paths (ruff S108)**. In tests, use `tmp_path` pytest fixture. In code, use `tempfile` module:
   ```python
   # ❌ Bad
   with open("/tmp/foo.txt", "w") as file:
       ...
   
   # ✅ Good (in tests)
   def test_something(tmp_path: Path) -> None:
       file_path = tmp_path / "foo.txt"
       ...
   
   # ✅ Good (in code)
   import tempfile
   with tempfile.NamedTemporaryFile() as file:
       ...
   ```

### Style preferences 
1. prefer way to use pillow is 'from PIL import Image'
1. always use loguru for logging not print
1. Use pydantic schemas for all configs
1. preffer f-strings to loguru structured output

### Antipatterns
1. never check output logs in cli tests

### Documentation guidelines 
1. always write to README.md in russian
1. Write README.md as user friendly intro rich with examples, not as wall of internals
1. Try avoid using strings instead ofg types if it's not strictly neccessary..

## Other instructions 
Refers to AGENT_DOCS.md for short project documentation. Add short descriptions about implicit or hard code solutions to it if needed. 

## Package Management

**Always use `uv` interface, NOT `pip` or `uv pip`.**
Incorrect commands (DO NOT USE):
```bash
pip install ...           # ❌ Don't use
uv pip install ...        # ❌ Don't use
pip install -e .          # ❌ Don't use
```

Correct Commands
use `uv run`:
```bash
uv run pytest tests/
```
