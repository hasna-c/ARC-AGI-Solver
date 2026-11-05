import subprocess
import tempfile
import json
import os
import textwrap
import contextlib

def run_arc_sandbox(code_str, train_pairs, timeout=10.0):
    """Executes model-generated Python solver code safely against ARC training pairs."""
    wrapped_code = f"""
import numpy as np
import json

def solve(input_grid):
    inp = np.array(input_grid)
{textwrap.indent(code_str, '    ')}

train_pairs = {json.dumps(train_pairs)}
results = []
passed = 0
for pair in train_pairs:
    try:
        out = solve(pair['input'])
        if out is not None and np.array_equal(np.array(out), np.array(pair['output'])):
            passed += 1
            results.append({{"success": True, "output": np.array(out).tolist()}})
        else:
            results.append({{"success": False, "output": out if out is not None else "None"}})
    except Exception as e:
        results.append({{"success": False, "error": str(e)}})

print(json.dumps({{ "passed": passed, "total": len(train_pairs), "results": results }}))
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapped_code)
        temp_name = f.name

    try:
        res = subprocess.run(["python", temp_name], capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0:
            return json.loads(res.stdout)
        return {"passed": 0, "error": res.stderr}
    except Exception as e:
        return {"passed": 0, "error": str(e)}
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)