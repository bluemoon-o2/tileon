import sys
import glob
import os

target_path = sys.argv[1]

if os.path.isdir(target_path):
    files = glob.glob(os.path.join(target_path, "**/*.pyi"), recursive=True)
else:
    files = [target_path]

for p in files:
    try:
        lines = open(p, encoding='utf-8').readlines()
    except Exception:
        continue
        
    content = ''.join(lines)

    if "numpy" in content:
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('"""') and i > 0:
                insert_idx = i + 1
                break
        
        header = 'try:\n    import numpy\nexcept ImportError:\n    numpy = None\n'
        if header not in content:
            lines.insert(insert_idx, header)
            open(p, 'w', encoding='utf-8').write(''.join(lines))
