# Copyright (c) OpenMMLab. All rights reserved.
import os
import uuid

import mmengine
from PIL import Image


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


def identity(x):
    return x


image_path_to_inputs = identity
image_path_to_outputs = identity
text_to_inputs = identity
text_to_outputs = identity


def pil_image_to_inputs(inputs: Image) -> str:
    temp_image_path = get_new_image_name('image/temp.jpg', func_name='temp')
    inputs.save(temp_image_path)
    return temp_image_path


def outputs_to_pil_image(outputs: str) -> Image:
    outputs = Image.open(outputs)
    return outputs


inputs_conversions = {
    'image_path': image_path_to_inputs,
    'text': text_to_inputs,
    'pil image': pil_image_to_inputs,
    'eval': eval,
    'identity': identity
}

outputs_conversions = {
    'image_path': image_path_to_outputs,
    'text': text_to_outputs,
    'pil image': outputs_to_pil_image,
    'eval': eval,
    'identity': identity,
}
