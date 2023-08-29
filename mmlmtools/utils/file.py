# Copyright (c) OpenMMLab. All rights reserved.
import os
import uuid


def get_new_file_path(org_file_path: str, func_name: str = 'update'):
    """Create a new file path for the tool output based on the original file
    path and tool function. The file path is unique and can be identified by
    the agent. The file name consists of uuid, function name of all appled
    tools and the original file name.

    Args:
        org_file_path (str): Original file path
        func_name (str, optional): Descriptions. Defaults to `'update'`

    Returns:
        str: The new file path
    """
    dirname, basename = os.path.split(org_file_path)
    os.makedirs(dirname, exist_ok=True)
    basename_splits = basename.split('.')

    img_format = basename_splits[-1]
    name_split = basename_splits[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    most_org_file_name = name_split[-1]
    recent_prev_file_name = name_split[0]
    if len(name_split) in [1, 4]:
        new_file_name = '_'.join([
            this_new_uuid, func_name, recent_prev_file_name, most_org_file_name
        ])
    elif len(name_split) == 3:
        new_file_name = '_'.join(
            [recent_prev_file_name, this_new_uuid, most_org_file_name])
    else:
        raise NotImplementedError
    new_file_name += f'.{img_format}'
    new_file_path = os.path.join(dirname, new_file_name)
    return new_file_path
