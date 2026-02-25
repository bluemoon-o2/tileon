import argparse
import ast
import importlib.util
import re
from pathlib import Path


def _load_docs(module_path: Path) -> list[dict]:
    spec = importlib.util.spec_from_file_location("tileon_pyi_docs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    docs = getattr(module, "C_PYI_DOCS", None)
    if docs is None:
        raise RuntimeError("C_PYI_DOCS not found in docs module")
    return list(docs)


def _function_name(signature: str) -> str:
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
    if not match:
        return "unknown"
    return match.group(1)


def _parse_signature(signature: str) -> tuple[list[tuple[str, str]], str]:
    code = f"{signature}:\n    ...\n"
    module = ast.parse(code)
    func = module.body[0]
    if not isinstance(func, ast.FunctionDef):
        return [], ""
    params: list[tuple[str, str]] = []
    for arg in func.args.args:
        if arg.annotation is None:
            params.append((arg.arg, ""))
        else:
            params.append((arg.arg, ast.unparse(arg.annotation)))
    return_type = ast.unparse(func.returns) if func.returns is not None else ""
    return params, return_type


def _render_docstring(entry: dict, param_order: list[tuple[str, str]], return_type: str) -> list[str]:
    lines: list[str] = []
    summary = entry.get("summary") or entry.get("doc")
    if summary:
        lines.append(summary)
    params = entry.get("params", {})
    args_lines: list[str] = []
    for name, type_text in param_order:
        if name not in params:
            continue
        desc = params.get(name, "")
        if type_text:
            label = f"{name} ({type_text})"
        else:
            label = name
        if desc:
            args_lines.append(f"    {label}: {desc}")
        else:
            args_lines.append(f"    {label}:")
    if args_lines:
        if lines:
            lines.append("")
        lines.append("Args:")
        lines.extend(args_lines)
    returns = entry.get("returns")
    if returns or return_type:
        if lines:
            lines.append("")
        lines.append("Returns:")
        if return_type and returns:
            lines.append(f"    {return_type}: {returns}")
        elif return_type:
            lines.append(f"    {return_type}:")
        else:
            lines.append(f"    {returns}")
    return lines


def _render_pyi(entries: list[dict], module_doc: str) -> str:
    counts: dict[str, int] = {}
    for entry in entries:
        name = _function_name(entry.get("signature", ""))
        counts[name] = counts.get(name, 0) + 1
    overload_needed = any(count > 1 for count in counts.values())

    lines: list[str] = []
    lines.append(f'"""{module_doc}"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    if overload_needed:
        lines.append("from typing import overload")
    lines.append("")

    seen: set[str] = set()
    for entry in entries:
        signature = entry.get("signature", "").strip()
        name = _function_name(signature)
        if counts.get(name, 0) > 1:
            lines.append("@overload")
        lines.append(f"{signature}:")
        param_order, return_type = _parse_signature(signature)
        doc_lines: list[str] = []
        if name not in seen:
            doc_lines = _render_docstring(entry, param_order, return_type)
        seen.add(name)
        if doc_lines:
            lines.append('    """')
            for doc_line in doc_lines:
                lines.append(f"    {doc_line}")
            lines.append('    """')
        lines.append("    ...")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--module-doc", required=True)
    parser.add_argument("--docs-module", required=True)
    args = parser.parse_args()

    docs_module = Path(args.docs_module)
    entries = _load_docs(docs_module)
    content = _render_pyi(entries, args.module_doc)
    Path(args.output).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
