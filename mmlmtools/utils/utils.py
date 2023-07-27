# Copyright (c) OpenMMLab. All rights reserved.
import os
import uuid

import mmengine


def get_new_image_name(org_img_name, func_name='update'):
    """create a temporary path for the image.

    Args:
        org_img_name (str): Original image path
        func_name (str, optional): Descriptions. Defaults to 'update'.

    Returns:
        new_image_path (str): The new image path
    """
    dirname, basename = os.path.split(org_img_name)
    mmengine.mkdir_or_exist(dirname)
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
    new_image_path = os.path.join(dirname, new_file_name)
    return new_image_path
