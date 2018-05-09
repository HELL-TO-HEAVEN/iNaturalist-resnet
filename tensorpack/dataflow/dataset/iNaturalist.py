#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: iNaturalist.py

import os
import json
import os.path as osp
import tarfile
import numpy as np
import tqdm


from ...utils import logger
from ...utils.loadcaffe import get_caffe_pb
from ...utils.fs import mkdir_p, download, get_dataset_path
from ...utils.timer import timed_operation
from ..base import RNGDataFlow

__all__ = ['iNaturalistMeta', 'iNaturalist', 'iNaturalistFiles']
ground_truth_dir = '/home/huzhikun/DataSet/iNaturalist2018/ground_truth'

class iNaturalistMeta(object):
    """
    Provide methods to access metadata for ILSVRC dataset.
    """

    def __init__(self, dir=None):
        pass

    def get_image_list(self, name):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
            dir_structure (str): same as in :meth:`iNaturalist.__init__()`.
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        ret = []

        if name == 'train':
            fname = osp.join(ground_truth_dir, 'train2018.json')
            assert os.path.isfile(fname), fname

            with open(fname, 'r') as f:
                train2018 = json.load(f)
            for i in range(len(train2018['images'])):
                name = train2018['images'][i]['file_name'] #train_va/2018/Aves/2820/285hy8uryu8w989.jpg
                cls = train2018['annotations'][i]['category_id']
                ret.append((name.strip(), cls))

        if name == 'val':
            fname = osp.join(ground_truth_dir, 'val2018.json')
            assert os.path.isfile(fname), fname

            with open(fname, 'r') as f:
                val2018 = json.load(f)
            for i in range(len(val2018['images'])):
                name = val2018['images'][i]['file_name'] #train_va/2018/Aves/2820/285hy8uryu8w989.jpg
                cls = val2018['annotations'][i]['category_id']
                ret.append((name.strip(), cls))

        if name == 'test':
            fname = osp.join(ground_truth_dir, 'test2018.json')
            assert os.path.isfile(fname), fname

            with open(fname, 'r') as f:
                test2018 = json.load(f)
            for i in range(len(test2018['images'])):
                name = test2018['images'][i]['file_name'] #train_va/2018/Aves/2820/285hy8uryu8w989.jpg
                id = test2018['images'][i]['id']
                ret.append((name.strip(), id))

        assert len(ret)
        return ret

class iNaturalistFiles(RNGDataFlow):
    """
    Same as :class:`iNaturalist`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None):
        """
        Same as in :class:`iNaturalist`.
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = dir
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        assert meta_dir is None or os.path.isdir(meta_dir), meta_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        meta = iNaturalistMeta(meta_dir)
        self.imglist = meta.get_image_list(name)

        for fname, _ in self.imglist[:10]:
            fname = os.path.join(self.full_dir, fname)
            assert os.path.isfile(fname), fname

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            fname = os.path.join(self.full_dir, fname)
            yield [fname, label]

class iNaturalist(iNaturalistFiles):
    """
    Produces uint8 iNaturalist images of shape [h, w, 3(BGR)], and a label between [0, 8141]. num_classes=8142
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``,
                containing the images in a structure described below.
            name (str): One of 'train' or 'val' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.
            dir_structure (str): One of 'original' or 'train'.
        Examples:
        When `dir_structure=='original'`, `dir` should have the following structure:
            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val(test)/
                ILSVRC2012_val_00000001.JPEG
        When `dir_structure=='train'`, `dir` should have the following structure:
            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
              val/
                n01440764/
                  ILSVRC2012_val_00000293.JPEG
                  ...
              test/
                ILSVRC2012_test_00000001.JPEG
        """
        super(iNaturalist, self).__init__(
            dir, name, meta_dir, shuffle)

    """
    There are some CMYK / png images, but cv2 seems robust to them.
    https://github.com/tensorflow/models/blob/c0cd713f59cfe44fa049b3120c417cc4079c17e3/research/inception/inception/data/build_imagenet_data.py#L264-L300
    """
    def get_data(self):
        for fname, label in super(iNaturalist, self).get_data():
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            yield [im, label]

try:
    import cv2
except ImportError:
    from ...utils.develop import create_dummy_class
    ILSVRC12 = create_dummy_class('iNaturalist', 'cv2')  # noqa

if __name__ == '__main__':
    #meta = iNaturalistMeta()
    #print(meta.get_synset_words_1000())

    ds = iNaturalist('/home/huzhikun/DataSet/iNaturalist', 'val', shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break
