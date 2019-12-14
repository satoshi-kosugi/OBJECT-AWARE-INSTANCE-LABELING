# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Peng Tang for Online Instance Classifier Refinement
#
# Modified by Satoshi Kosugi for OBJECT-AWARE-INSTANCE-LABELING
# --------------------------------------------------------

"""Test an OICR network on an imdb (image database), for trainval set (CorLoc)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        im_shapes: the list of image shapes
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []

    im = cv2.resize(im_orig, (417, 417), interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_orig.shape[:2])

    processed_ims.append(im_list_to_blob([im]))

    blob = processed_ims
    return blob, np.array(im_scale_factors)


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    # blobs['rois'] = _get_rois_blob(rois, im_scale_factors)

    return blobs, im_scale_factors

def im_detect(net, im, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    for i in xrange(len(blobs['data'])):
        # reshape network inputs
        # net.blobs['data'].reshape(*(blobs['data'][i].shape))
        blobs_out = net.forward(data=blobs['data'][i].astype(np.float32, copy=False))

        saliency_map = blobs_out['sigmoid-output']
        saliency_map = cv2.resize(saliency_map[0, 0],
                        (unused_im_scale_factors[0][1], unused_im_scale_factors[0][0]),
                        interpolation=cv2.INTER_LINEAR)
        saliency_map = (saliency_map * 255).astype(np.uint8)

    return saliency_map

def detect_saliency(net, imdb):
    """Test an OICR network on an image database,
    and generate pseudo ground truths for training fast rcnn."""
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    roidb = imdb.roidb

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        saliency_map = im_detect(net, im, roidb[i]['boxes'])
        _t['im_detect'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time)

        cv2.imwrite(os.path.join(output_dir, imdb.image_index[i]+".png"), saliency_map)
