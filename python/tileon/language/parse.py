from __future__ import annotations

from ..runtime.jit import JITCallable


def get_jit_fn_file_line(fn: JITCallable):
    """
    Get the file name and line number of the first line of the original function.
    """
    base_fn = fn
    while not isinstance(base_fn, JITCallable):
        base_fn = base_fn.fn
    file_name = base_fn.fn.__code__.co_filename
    begin_line = base_fn.starting_line_number
    # Find the line number of the function definition
    for idx, line in enumerate(base_fn.raw_src):
        if line.strip().startswith("def "):
            begin_line += idx
            break
    return file_name, begin_line
