#!/usr/bin/env python

import glob
import os
import os.path as osp
import shutil


def format_tool_readme(fn):

    with open(fn, 'r') as f:
        readme = f.readlines()

    in_code_block = False
    for i, line in enumerate(readme):
        if in_code_block:
            if line.startswith('```'):
                in_code_block = False
            else:
                continue

        if line.startswith('```'):
            in_code_block = True
            continue

        # add indent level for all section titles
        if line.startswith('#'):
            readme[i] = '#' + line

    # write formatted content
    with open(fn, 'w') as f:
        f.writelines(readme)


if __name__ == '__main__':
    if osp.isdir('_tmp'):
        shutil.rmtree('_tmp')

    os.makedirs('_tmp/tools', exist_ok=True)

    # collect README.md of tools
    for readme_fn in glob.glob('../../mmlmtools/tools/**/README.md'):
        tgt_fn = osp.join('_tmp/tools',
                          osp.basename(osp.dirname(readme_fn)) + '.md')
        shutil.copy(readme_fn, tgt_fn)

        # format readme content
        format_tool_readme(tgt_fn)
