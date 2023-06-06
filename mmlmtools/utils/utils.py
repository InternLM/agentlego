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

    name_split = basename.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = '_'.join(
        [this_new_uuid, func_name, recent_prev_file_name, most_org_file_name])
    new_file_name += '.png'
    new_image_path = os.path.join(dirname, new_file_name)
    return new_image_path
