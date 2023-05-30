# Copyright (c) OpenMMLab. All rights reserved.
import os
import uuid


def get_new_image_name(org_img_name, func_name='update'):
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
    return os.path.join(head, new_file_name)