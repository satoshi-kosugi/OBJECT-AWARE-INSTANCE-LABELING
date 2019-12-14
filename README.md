# Object-Aware Instance Labeling for Weakly Supervised Object Detection

Official implementation of [Object-Aware Instance Labeling for Weakly Supervised Object Detection](https://arxiv.org/abs/1908.03792) (accepted to ICCV 2019, oral).
Our code is based on [Caffe](http://caffe.berkeleyvision.org/), [fast r-cnn](https://github.com/rbgirshick/fast-rcnn), [faster r-cnn](https://github.com/rbgirshick/py-faster-rcnn), and [OICR](https://github.com/ppengtang/oicr).

<p align="left">
<img src="images/architecture.png" alt="architecture" width="850px">
</p>

## Requirements
1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

2. Python 2.7 and the following packages:
    - easydict
    - numpy
    - scikit-image
    - protobuf
    - opencv-python
    - cython
    - pyyaml

3. MATLAB

## Installation

1. Clone the repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/satoshi-kosugi/OBJECT-AWARE-INSTANCE-LABELING_kari.git
  ```

2. Build the Cython modules
  ```Shell
  cd $ROOT/lib
  make
  ```

3. Build Caffe and pycaffe
  ```Shell
  cd $ROOT/caffe-oicr
  # Now follow the Caffe installation instructions here:
  #   http://caffe.berkeleyvision.org/installation.html

  # If you're experienced with Caffe and have all of the requirements installed
  # and your Makefile.config in place, then simply do:
  make all -j 8
  make pycaffe
  ```


## Download
### Datasets
1. Download the training, validation, test data and VOCdevkit

  ```Shell
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
  ```
2. Extract all of these tars into one directory named `VOCdevkit`

  ```Shell
  tar xvf VOCtrainval_06-Nov-2007.tar
  tar xvf VOCtest_06-Nov-2007.tar
  tar xvf VOCdevkit_18-May-2011.tar
  ```
3. It should have this basic structure

  ```Shell
  $VOCdevkit/                           # development kit
  $VOCdevkit/VOCcode/                   # VOC utility code
  $VOCdevkit/VOC2007                    # image sets, annotations, etc.
  # ... and several other directories ...
  ```

4. Create symlinks for the PASCAL VOC dataset

  ```Shell
  cd $ROOT/data
  ln -s $VOCdevkit VOCdevkit2007
  ```
  Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.

5. [Optional] follow similar steps to get PASCAL VOC 2012.


### Pre-computed Selective Search object proposals

Pre-computed selective search boxes can also be downloaded for VOC2007 and VOC2012.

  ```Shell
  cd $ROOT
  ./data/scripts/fetch_selective_search_data.sh
  ```

This will populate the `$ROOT/data` folder with `selective_selective_data`.
(The script is copied from the [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)).

### Pre-trained models

Pre-trained ImageNet models can be downloaded.

  ```Shell
  cd $ROOT
  ./data/scripts/fetch_imagenet_models.sh
  ```
These models are all available in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but are provided here for your convenience.
(The script is copied from the [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)).

Pre-trained Saliency Detection model can be downloaded.

  ```Shell
  cd $ROOT
  ./data/scripts/fetch_saliency_models.sh
  ```

## Training and testing
1. Detect saliency maps.
   ```Shell
    ./tools/test_net.py --gpu 0 --def models/SaliencyDetection/snet.prototxt \
      --net data/senet.caffemodel \
      --imdb voc_2007_trainval \
      --mode saliency
   ```
2. Train a context classifier.
   ```Shell
    ./tools/train_net.py --gpu 0 --solver models/ContextClassifer/solver.prototxt \
      --weights data/imagenet_models/VGG16.v2.caffemodel --iters 10000 \
      --saliency output/default/voc_2007_trainval/senet/ \
      --mode context_classifier
   ```
3. Test a context classifier.
   ```Shell
    ./tools/test_net.py --gpu 0 --def models/ContextClassifer/test.prototxt \
      --net output/default/voc_2007_trainval/vgg16_ContextClassifer_iter_10000.caffemodel \
      --imdb voc_2007_trainval \
      --saliency output/default/voc_2007_trainval/senet/ \
      --mode context_classifier
   ```
4. Train the OICR model.
   ```Shell
    ./tools/train_net.py --gpu 0 --solver models/OICR/solver.prototxt \
      --weights data/imagenet_models/VGG16.v2.caffemodel --iters 70000 \
      --flg output/default/voc_2007_trainval/vgg16_ContextClassifer_iter_10000/discovery_all.pkl \
      --mode oicr
   ```
5. Test the OICR model.  
   On trainval
    ```Shell
    ./tools/test_net.py --gpu 0 --def models/OICR/test.prototxt \
      --net output/default/voc_2007_trainval/vgg16_oicr_iter_70000.caffemodel \
      --imdb voc_2007_trainval \
      --mode oicr
    ```
   On test  
    ```Shell
    ./tools/test_net.py --gpu 0 --def models/OICR/test.prototxt \
      --net output/default/voc_2007_trainval/vgg16_oicr_iter_70000.caffemodel \
      --imdb voc_2007_test \
      --mode oicr
    ```
6. Evaluate the OICR model.
    For mAP, run the python code tools/reval.py
      ```Shell
      ./tools/reval.py output/default/voc_2007_test/vgg16_oicr_iter_70000/ --imdb voc_2007_test --matlab
      ```

    For CorLoc, run the python code tools/reval_discovery.py
      ```Shell
      ./tools/reval_discovery.py output/default/voc_2007_trainval/vgg16_oicr_iter_70000/ --imdb voc_2007_trainval
      ```

## License

Our code is released under the MIT License (refer to the LICENSE file for details).

## Citation

If you find our research useful in your research, please consider citing:

    @inproceedings{kosugi2019object,
        title={Object-Aware Instance Labeling for Weakly Supervised Object Detection},
        author={Kosugi, Satoshi and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
        booktitle = {ICCV},
        year = {2019}
    }
