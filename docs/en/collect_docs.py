#!/usr/bin/env python

import shutil
from pathlib import Path

root_dir = Path(__file__).parents[2]
doc_dir = Path(__file__).parent
tmp_dir = doc_dir / '_tmp'


def format_tool_readme(fn):

    with open(fn, 'r') as f:
        contents = f.readlines()

    in_code_block = False
    h1 = []
    for lineno, line in enumerate(contents):
        if line.startswith('```'):
            in_code_block = not in_code_block

        if line.startswith('# ') and not in_code_block:
            h1.append(lineno)

    for start, end in zip(h1, h1[1:] + [len(contents)]):
        cls_name = contents[start][2:].strip()
        target = tmp_dir / 'tools' / (cls_name + '.md')
        target.write_text(''.join(contents[start:end]))


if __name__ == '__main__':

    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)

    (tmp_dir / 'tools').mkdir(exist_ok=True, parents=True)

    # collect README.md of tools
    for readme_fn in root_dir.glob('mmlmtools/tools/*/README.md'):
        # format readme content
        format_tool_readme(readme_fn)
