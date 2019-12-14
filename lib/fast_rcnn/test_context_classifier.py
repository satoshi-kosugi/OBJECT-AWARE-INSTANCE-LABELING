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

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im_list_to_blob([im]))

    blob = processed_ims
    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois_blob_real = []

    for i in xrange(len(im_scale_factors)):
        rois, levels = _project_im_rois(im_rois, np.array([im_scale_factors[i]]))
        rois_blob = np.hstack((levels, rois))
        rois_blob_real.append(rois_blob.astype(np.float32, copy=False))

    return rois_blob_real

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1
        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)

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
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(blobs['rois'][i] * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            blobs['rois'][i] = blobs['rois'][i][index, :]
            boxes_tmp = boxes[index, :]
        else:
            boxes_tmp = boxes

        # reshape network inputs
        net.blobs['data'].reshape(*(blobs['data'][i].shape))
        net.blobs['rois'].reshape(*(blobs['rois'][i].shape))

        blobs_out = net.forward(data=blobs['data'][i].astype(np.float32, copy=False),
                                rois=blobs['rois'][i].astype(np.float32, copy=False))

        scores_tmp = blobs_out['CAM_fc']
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes_tmp, (1, scores_tmp.shape[1]))

        if cfg.TEST.USE_FLIPPED:
            blobs['data'][i] = blobs['data'][i][:, :, :, ::-1]
            width = blobs['data'][i].shape[3]
            oldx1 = blobs['rois'][i][:, 1].copy()
            oldx2 = blobs['rois'][i][:, 3].copy()
            blobs['rois'][i][:, 1] = width - oldx2 - 1
            blobs['rois'][i][:, 3] = width - oldx1 - 1
            assert (blobs['rois'][i][:, 3] >= blobs['rois'][i][:, 1]).all()

            net.blobs['data'].reshape(*(blobs['data'][i].shape))
            net.blobs['rois'].reshape(*(blobs['rois'][i].shape))

            blobs_out = net.forward(data=blobs['data'][i].astype(np.float32, copy=False),
                                    rois=blobs['rois'][i].astype(np.float32, copy=False))

            scores_tmp += blobs_out['CAM_fc']

        if cfg.DEDUP_BOXES > 0:
            # Map scores and predictions back to the original set of boxes
            scores_tmp = scores_tmp[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]

        if i == 0:
            scores = np.copy(scores_tmp)
        else:
            scores += scores_tmp

    scores /= len(blobs['data']) * (1. + cfg.TEST.USE_FLIPPED)

    return scores, pred_boxes

def split_segments(saliency_map):
    binary128 = saliency_map.copy()
    binary128[saliency_map > 128] = 255
    binary128[saliency_map <= 128] = 0
    binary = saliency_map.copy()
    binary[saliency_map > cfg.TRAIN.SALIENCY_THRESH] = 255
    binary[saliency_map <= cfg.TRAIN.SALIENCY_THRESH] = 0

    nLabels, labelImage = cv2.connectedComponents(binary128)
    if nLabels == 1:
        return [np.zeros_like(saliency_map)]

    max_area = 0
    for i in range(nLabels):
        segment = labelImage.copy()
        segment[segment==i] = 255
        segment[segment!=255] = 0

        if (segment == binary128).sum() != 0:
            area = (segment / 255).sum()
            max_area = max(max_area, area)

    segments = []
    for i in range(nLabels):
        segment = labelImage.copy()
        segment[segment==i] = 255
        segment[segment!=255] = 0

        area = (segment / 255).sum()
        if (segment == binary128).sum() != 0 and area >= max_area*0.1:
            segments.append(segment)

    if len(segments) == 1:
        return [binary / 255.]

    finals = []
    for i, segment in enumerate(segments):
        kernel = np.ones((24, 24), np.uint8)
        expanded = cv2.dilate(segment.astype(np.uint8), kernel, iterations = 1)
        final = binary / 255.0 * expanded
        finals.append(final.astype(np.uint8) / 255.)
    return finals

def assign_regions(saliency_map_segments, boxes):
    if len(saliency_map_segments) == 1:
        return [np.arange(len(boxes))]

    ious = np.zeros((len(saliency_map_segments), len(boxes)))
    for i in range(len(saliency_map_segments)):
        segment = saliency_map_segments[i] / 255
        area_segment = segment.sum()
        for j in range(len(boxes)):
            box = boxes[j]
            area_box = (box[2] * 1. - box[0]) * (box[3] * 1. - box[1])
            intersection = segment[box[1]:box[3], box[0]:box[2]].sum()
            ious[i, j] = intersection * 1.0 / (area_segment + area_box - intersection)

    ious_argmax = ious.argmax(axis=0)
    assigned_indices = []
    for i in range(len(saliency_map_segments)):
        assigned_indices.append(np.where(ious_argmax==i)[0])
    return assigned_indices

def test_context_classifier(net, imdb, saliency_dir):
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

    scores_all = []
    boxes_all = []

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        scores = np.zeros((roidb[i]['boxes'].shape[0], 20))

        saliency_map_name = os.path.join(saliency_dir,
                imdb.image_path_at(i).split("/")[-1].replace("jpg", "png"))
        saliency_map = cv2.imread(saliency_map_name, cv2.IMREAD_GRAYSCALE)

        assert saliency_map is not None, "Cannot load saliency map"

        saliency_map_segments = split_segments(saliency_map)
        assigned_indices = assign_regions(saliency_map_segments, roidb[i]['boxes'])

        _t['im_detect'].tic()
        for j in range(len(saliency_map_segments)):
            masked_im = im.copy()

            masked_im = masked_im - cfg.PIXEL_MEANS
            masked_im[saliency_map_segments[j] == 0] = 0
            masked_im = masked_im + cfg.PIXEL_MEANS
            
            if len(assigned_indices[j]) != 0:
                scores[assigned_indices[j]], boxes = im_detect(net, masked_im, roidb[i]['boxes'][assigned_indices[j]])
        _t['im_detect'].toc()
        scores_all.append(scores)
        boxes_all.append(boxes)

        print 'im_detect: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time)

    dis_file_all = os.path.join(output_dir, 'discovery_all.pkl')
    results_all = {'scores_all' : scores_all, 'boxes_all' : boxes_all}
    with open(dis_file_all, 'wb') as f:
        cPickle.dump(results_all, f, cPickle.HIGHEST_PROTOCOL)
