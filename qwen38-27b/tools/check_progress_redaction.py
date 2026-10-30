#!/usr/bin/env python3
"""Check the Qwen progress-event credential redaction without importing TPU dependencies."""
import ast
import json
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "kernel" / "serve_qwen38.py"
source_text = SOURCE.read_text()
tree = ast.parse(source_text)
nodes = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name == "sanitize_progress")
         or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_SENSITIVE_PROGRESS_KEYS" for target in node.targets))]
namespace = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
notebook = SOURCE.parents[1] / "notebook" / "qwen38-tpu-serve.ipynb"
notebook_text = "\n".join("".join(cell.get("source", [])) for cell in json.loads(notebook.read_text())["cells"])
assert all(ast.get_source_segment(source_text, node) in notebook_text for node in nodes), "kernel/notebook sanitizer drift"
sanitize = namespace["sanitize_progress"]
marker = "test-only-secret-marker"
clean = sanitize({
    "api_key": marker,
    "nested": [{"Authorization": f"Bearer {marker}", "refresh-token": marker}],
    "model": "qwen3.8-27b",
    "mtp_tokens": 3,
})
assert clean["api_key"] == "[REDACTED]"
assert clean["nested"][0] == {"Authorization": "[REDACTED]", "refresh-token": "[REDACTED]"}
assert clean["model"] == "qwen3.8-27b" and clean["mtp_tokens"] == 3
assert marker not in repr(clean)
print("Qwen progress-event redaction + notebook parity: PASS")
