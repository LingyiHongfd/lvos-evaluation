import os
import sys

import numpy as np
from PIL import Image


class Results(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.a = root_dir

    def _read_mask(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f"{frame_id}.png")
            return np.array(Image.open(mask_path))
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write(
                "The frames have to be indexed PNG files placed inside the corespondent sequence "
                "folder.\nThe indexes have to match with the initial frame.\n"
            )
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_masks(self, sequence, masks_id):
        mask_0 = self._read_mask(sequence, masks_id[0])
        masks = np.zeros((len(masks_id), *mask_0.shape))
        for ii, m in enumerate(masks_id):
            masks[ii, ...] = self._read_mask(sequence, m)
        num_objects = int(np.max(masks))
        tmp = np.ones((num_objects, *masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = (tmp == masks[None, ...]) > 0
        return masks

    def read_mask(self, sequence, frame_id, target_obj=None):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f"{frame_id}.png")
            masks = np.array(Image.open(mask_path))
            if target_obj is not None:
                tmp_masks = np.zeros_like(masks)
                tmp_masks[masks == int(target_obj)] = 1
                masks = tmp_masks
            masks = np.expand_dims(masks, axis=0)
            return masks
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write(
                "The frames have to be indexed PNG files placed inside the corespondent sequence "
                "folder.\nThe indexes have to match with the initial frame.\n"
            )
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_mask_seperate(self, sequence, frame_id, max_n_proposals=None):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f"{frame_id}.png")
            masks = np.array(Image.open(mask_path))
            n_masks = np.zeros((max_n_proposals, masks.shape[0], masks.shape[1]))
            if max_n_proposals is not None:
                for pi in range(1, max_n_proposals + 1):
                    n_masks[pi - 1] = masks == pi
                masks = n_masks
            masks = np.expand_dims(masks, axis=0)
            return masks
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write(
                "The frames have to be indexed PNG files placed inside the corespondent sequence "
                "folder.\nThe indexes have to match with the initial frame.\n"
            )
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_mask_salient(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f"{frame_id}.png")
            masks = np.array(Image.open(mask_path))
            tmp_masks = np.zeros_like(masks)
            mask_values = list(np.unique(masks))
            if len(mask_values) > 2:
                tmp_masks[masks >= 127] = 1
                tmp_masks[masks < 127] = 0
            elif len(mask_values) == 1:
                if mask_values[0] == 0:
                    masks = tmp_masks
                elif mask_values[0] == 255:
                    tmp_masks[masks == 255] = 1
                elif mask_values[0] == 1:
                    tmp_masks[masks == 1] = 1
                else:
                    raise ValueError("Unknown mask value")
            elif len(mask_values) == 2:
                if mask_values == [0, 1]:
                    tmp_masks[masks == 1] = 1
                elif mask_values == [0, 255]:
                    tmp_masks[masks == 255] = 1
                else:
                    raise ValueError("Unknown mask value")
            masks = tmp_masks
            masks = np.expand_dims(masks, axis=0)
            return masks
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write(
                "The frames have to be indexed PNG files placed inside the corespondent sequence "
                "folder.\nThe indexes have to match with the initial frame.\n"
            )
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()
