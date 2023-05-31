# Copyright (c) OpenMMLab. All rights reserved.
import os
import uuid


def get_new_image_name(org_img_name, func_name='update'):
    """create a temporary path for the image.

    Args:
        org_img_name (str): Original image path
        func_name (str, optional): Descriptions. Defaults to 'update'.

    Returns:
        new_image_path (str): The new image path
    """
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
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
    new_image_path = os.path.join(head, new_file_name)
    return new_image_path


def convert_description(src_description, src_style, dst_style):
    """convert description from one style into another to adapt different LLM.

    Args:
        src_description (str): Tool description
    """

    return src_description
