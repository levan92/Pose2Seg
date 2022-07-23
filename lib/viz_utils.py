## from mmdet.core.visualization.image
## modified due to buffer.reshape bug

# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pathlib import Path

import cv2

from mmdet.core import mask2ndarray
# from mmdet.core.visualization import get_palette, palette_val

EPS = 1e-2

def get_palette(palette, num_classes):
    """Get palette from various inputs.
    copied from `mmdet/core/visualization/palette.py`

    Args:
        palette (list[tuple] | str | tuple | :obj:`Color`): palette inputs.
        num_classes (int): the number of classes.
    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)

    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == 'random' or palette is None:
        state = np.random.get_state()
        # random color
        # np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        # np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif palette == 'coco':
        from mmdet.datasets import CocoDataset, CocoPanopticDataset
        dataset_palette = CocoDataset.PALETTE
        if len(dataset_palette) < num_classes:
            dataset_palette = CocoPanopticDataset.PALETTE
    elif palette == 'citys':
        from mmdet.datasets import CityscapesDataset
        dataset_palette = CityscapesDataset.PALETTE
    elif palette == 'voc':
        from mmdet.datasets import VOCDataset
        dataset_palette = VOCDataset.PALETTE
    elif mmcv.is_str(palette):
        dataset_palette = [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')

    assert len(dataset_palette) >= num_classes, \
        'The length of palette should not be less than `num_classes`.'
    return dataset_palette


def draw(img,
        segms,
        kpts,
        outpath,
        mask_color=(72, 101, 241),
        thickness=2,
        font_size=13,
        win_name='',
        mask_alpha=0.5,
        ):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.
        pasted (list[int], optional): a list of flags of whether instance is pasted or not (for copy paste augmentation)
        pasted_color (list[tuple] | tuple | str | None, optional): Colors of
           pasted instances. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        mask_alpha (float): between 0. to 1., how much "transparency" masks should appear as. 

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    img = mmcv.imread(img).astype(np.uint8)
    img = np.ascontiguousarray(img)
    mask_color = get_palette('random', len(segms))
    mask_color = np.array(mask_color, dtype=np.uint8)

    for i, (segm, kpt) in enumerate(zip(segms, kpts)):
        mask = segm.astype(bool)
        img[mask] = img[mask] * (1-mask_alpha) + mask_color[i] * mask_alpha

        for kp in kpt:
            x,y,v = kp
            if v > 0:
                if v == 2:
                    color = (255,255,255)
                else: 
                    color = (0,0,255)
                cv2.circle(img, (int(x),int(y)), radius=2, color=color, thickness=-1)
    
    outpath = outpath.parent / f'{outpath.stem}_viz.jpg'
    cv2.imwrite(str(outpath), img)

