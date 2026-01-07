import ast
import os
from typing import List

# --- Configuration for penaltyblog ---
CONFIG = {
    "summary_file": "README.md",
    "key_files": [
        "pyproject.toml",
        "requirements.txt",
    ],
    "source_directory": {
        "path": "penaltyblog",
        "description": "Main source code for the library. This will be summarized.",
    },
    "tutorial_directory": {
        "path": "docs",
        "description": "Docs demonstrating library usage. This will be included in full.",
    },
    "test_directory": {
        "path": "test",
        "description": "Unit and integration tests. This will be summarized.",
    },
    "exclude_patterns": [
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "dist",
        "build",
        ".pytest_cache",
        ".tox",
        "penaltyblog.egg-info",
    ],
    "include_extensions": [
        ".py",
        "pyx",
        "pyi",
        ".rst",
        ".md",
        ".yml",
        ".yaml",
        ".toml",
        ".txt",
    ],
    "max_file_size_kb": 50,
}


def is_excluded(path: str, exclude_patterns: List[str]) -> bool:
    return any(excl in path for excl in exclude_patterns)


def read_file_content(filepath: str) -> str:
    try:
        file_size_kb = os.path.getsize(filepath) / 1024
        if file_size_kb > CONFIG["max_file_size_kb"]:
            return f"# File {filepath} is too large ({file_size_kb:.2f} KB) and has been skipped.\n"
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# Error reading file {filepath}: {e}\n"


def _get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extracts a clean function signature from an AST node."""
    args = []
    # Regular arguments
    for a in node.args.args:
        args.append(a.arg)
    # Default arguments
    defaults = [ast.unparse(d) for d in node.args.defaults]
    # Combine args and defaults
    num_defaults = len(defaults)
    final_args = []
    for i, arg in enumerate(args):
        if i >= len(args) - num_defaults:
            default_val = defaults[i - (len(args) - num_defaults)]
            final_args.append(f"{arg}={default_val}")
        else:
            final_args.append(arg)

    signature = f"{node.name}({', '.join(final_args)})"
    if node.returns:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def summarize_python_file(filepath: str) -> str:
    """Parses a Python file to extract a structured summary."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        summary_lines = [f"### File: `{os.path.relpath(filepath)}`\n"]

        for item in tree.body:
            if isinstance(item, ast.ClassDef):
                docstring = ast.get_docstring(item)
                summary_lines.append(f"- **Class**: `{item.name}`")
                if docstring:
                    summary_lines.append(
                        f"  - *Description*: {docstring.strip().split(chr(10))[0]}"
                    )

                for sub_item in item.body:
                    if isinstance(sub_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sub_docstring = ast.get_docstring(sub_item)
                        signature = _get_signature(sub_item)
                        summary_lines.append(f"  - **Method**: `{signature}`")
                        if sub_docstring:
                            summary_lines.append(
                                f"    - *Description*: {sub_docstring.strip().split(chr(10))[0]}"
                            )

            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(item)
                signature = _get_signature(item)
                summary_lines.append(f"- **Function**: `{signature}`")
                if docstring:
                    summary_lines.append(
                        f"  - *Description*: {docstring.strip().split(chr(10))[0]}"
                    )

        return "\n".join(summary_lines) + "\n"

    except Exception as e:
        return f"# Error summarizing Python file {filepath}: {e}\n"


def generate_llms_content() -> str:
    """Generates the full, structured content for the llms.txt file."""
    output = []
    toc = []  # Table of Contents

    # 1. Table of Contents
    toc.append("# Table of Contents")
    toc.append("- [Project Overview](#project-overview-and-summary)")
    toc.append("- [Configuration](#key-configuration-and-dependencies)")
    toc.append("- [High-Level Documentation](#high-level-documentation)")
    toc.append("- [Tutorials](#tutorials)")
    toc.append("- [Source Code Summary](#source-code-summary)")
    toc.append("- [Test Summary](#test-summary)")
    output.append("\n".join(toc))
    output.append("\n" + "=" * 80 + "\n")

    # 2. High-Level Summary
    output.append("## Project Overview and Summary")
    output.append(f"Source: `{CONFIG['summary_file']}`\n")
    output.append(read_file_content(CONFIG["summary_file"]))
    output.append("\n" + "=" * 80 + "\n")

    # 3. Key Configuration
    output.append("## Key Configuration and Dependencies")
    for file_path in CONFIG["key_files"]:
        if os.path.exists(file_path):
            output.append(f"\n### {file_path}")
            output.append("```toml" if file_path.endswith(".toml") else "```")
            output.append(read_file_content(file_path))
            output.append("```")
    output.append("\n" + "=" * 80 + "\n")

    # 4. Docs
    tutorial_dir = CONFIG["tutorial_directory"]
    if os.path.isdir(tutorial_dir["path"]):
        output.append(f"## Tutorials")
        output.append(f"{tutorial_dir['description']}\n")
        for root, _, files in os.walk(tutorial_dir["path"]):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                if ext in CONFIG["include_extensions"] and not is_excluded(
                    file_path, CONFIG["exclude_patterns"]
                ):
                    rel_file_path = os.path.relpath(file_path)
                    output.append(f"### Tutorial: {rel_file_path}\n")
                    output.append("```rst")
                    output.append(read_file_content(file_path))
                    output.append("```")
        output.append("\n" + "=" * 80 + "\n")

    # 5. Summarized Source Code
    source_dir = CONFIG["source_directory"]
    if os.path.isdir(source_dir["path"]):
        output.append(f"## Source Code Summary")
        output.append(f"{source_dir['description']}\n")
        for root, _, files in os.walk(source_dir["path"]):
            for file in sorted(files):
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    if not is_excluded(file_path, CONFIG["exclude_patterns"]):
                        output.append(summarize_python_file(file_path))
        output.append("\n" + "=" * 80 + "\n")

    # 6. Summarized Tests
    test_dir = CONFIG["test_directory"]
    if os.path.isdir(test_dir["path"]):
        output.append(f"## Test Summary")
        output.append(f"{test_dir['description']}\n")
        for root, _, files in os.walk(test_dir["path"]):
            for file in sorted(files):
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    if not is_excluded(file_path, CONFIG["exclude_patterns"]):
                        output.append(summarize_python_file(file_path))
        output.append("\n" + "=" * 80 + "\n")

    return "".join(output)


if __name__ == "__main__":
    content = generate_llms_content()
    output_file = "llms.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully generated {output_file}")
