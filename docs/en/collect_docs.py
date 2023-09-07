#!/usr/bin/env python

import glob
import os
import os.path as osp
import re

if __name__ == '__main__':
    os.makedirs('_tmp/tools', exist_ok=True)

    # collect README.md of tools
    for readme_fn in glob.glob('../../mmlmtools/tools/**/README.md'):
        with open(readme_fn, 'r') as f:
            readme = f.read()

            # add indent level for all section titles
            pattern = r'(^|\n)#{1,}\s*\w+'
            readme = re.sub(
                pattern, lambda m: f'#{m.group()}', readme, flags=re.M)

            # write to _tmp/tools
            tgt_fn = osp.join('_tmp/tools',
                              osp.basename(osp.dirname(readme_fn)) + '.md')
            with open(tgt_fn, 'w') as f:
                f.write(readme)
