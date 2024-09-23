import numpy as np
import tensorflow as tf
from pathlib import Path
from glob import glob
import os
from .base_dataset import BaseDataset
from .utils import pipeline

from utils.tools import dict_update
DATA_PATH = '/root/Internship-Valeo/Project/data/'
EXPER_PATH = '/root/Internship-Valeo/Project/data/kitti'
import sys, getopt
import json


default_config = {
        'labels_train': None,
        'labels_val': None,
        'segmasks_train': None,
        'segmasks_val': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [160, 480]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }
class Kitti(BaseDataset):
    def _init_dataset(self, config):
        config = dict_update(default_config, config)
        #base_path = Path(DATA_PATH, 'kitti' ,'train_images')
        #image_paths = glob(DATA_PATH+'kitti/kitti-odometry-gray/sequences/0[4-8]/image_0/*.png')
        #image_paths = list(base_path.iterdir())
        #base_path = Path(DATA_PATH, 'KITTI-360' ,'val_images')
        #image_paths = list(base_path.iterdir())
        label_paths = Path(DATA_PATH, 'kitti' ,'KP_labels')
        label_paths = list(label_paths.iterdir())
        names = [p.stem for p in label_paths]
        image_paths = [DATA_PATH+'kitti/kitti-odometry-gray/sequences/'+name[:2]+'/image_0/'+name[2:]+'.png' for name in names]
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
        #names = [p.stem for p in image_paths]
        #names = [p[72:74]+p[83:-4] for p in image_paths]
    
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, #'mask_paths': mask_paths, 
                 'names': names}
        if config['labels_train']:
            label_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['labels_train'],'{}.npz'.format(n))
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths
        elif config['labels_val']:
            label_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['labels_val'],'{}.npz'.format(n))
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths
        if config['segmasks_train']:
            mask_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['segmasks_train'],'{}.jpg'.format(n))
                assert p.exists(), 'Image {} has no corresponding segmentation_mask {}'.format(n, p)
                mask_paths.append(str(p))
            files['mask_paths'] = mask_paths
        elif config['segmasks_val']:
            mask_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['segmasks_val'],'{}.jpg'.format(n))
                assert p.exists(), 'Image {} has no corresponding segmentation_mask {}'.format(n, p)
                mask_paths.append(str(p))
            files['mask_paths'] = mask_paths
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        return files

    def _get_data(self, files, split_name, config):
        config = dict_update(default_config, config)
        has_keypoints = 'label_paths' in files
        has_seg_masks = 'mask_paths' in files
        is_training = split_name == 'training'

        def _read_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return tf.cast(image, tf.float32)

        def _read_mask(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image)
            return tf.cast(image, tf.float32)

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return image
        def _preprocess_mask(mask_image):
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_mask_resize(mask_image,
                                                         **config['preprocessing'])
            return image

        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['arr_0'].astype(np.float32)

        names = tf.data.Dataset.from_tensor_slices(files['names'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths']) 
        images = images.map(_read_image)
        images = images.map(_preprocess)            
        data = tf.data.Dataset.zip({'image': images, #'mask_image': mask_images, 
                                    'name': names})

        # Add keypoints
        if has_keypoints:
            kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            kp = kp.map(lambda path: tf.numpy_function(_read_points, [path], tf.float32))
            kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
            data = tf.data.Dataset.zip((data, kp)).map(
                    lambda d, k: {**d, 'keypoints': k})
            data = data.map(pipeline.add_dummy_valid_mask)
        if has_seg_masks:
            mask_images = tf.data.Dataset.from_tensor_slices(files['mask_paths'])
            mask_images = mask_images.map(_read_mask)
            mask_images = mask_images.map(_preprocess_mask)
            data = tf.data.Dataset.zip((data, mask_images)).map(
                    lambda d, k: {**d, 'mask_image': k})
        # Keep only the first elements for validation
        if split_name == 'validation':
            data = data.take(config['validation_size'])

        # Cache to avoid always reading from disk
        if config['cache_in_memory']:
            tf.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        # Generate the warped pair
        if config['warped_pair']['enable']:
            assert has_keypoints
            warped = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                d, add_homography=True, **config['warped_pair']))
            if is_training and config['augmentation']['photometric']['enable']:
                warped = warped.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            warped = warped.map_parallel(pipeline.add_keypoint_map)
            # Merge with the original data
            data = tf.data.Dataset.zip((data, warped))
            data = data.map(lambda d, w: {**d, 'warped': w})
                                   # Data augmentation
        if has_keypoints and is_training:
            if config['augmentation']['photometric']['enable']:
                data = data.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            if config['augmentation']['homographic']['enable']:
                assert not config['warped_pair']['enable']  # doesn't support hom. aug.
                data = data.map_parallel(lambda d: pipeline.homographic_augmentation(
                    d, **config['augmentation']['homographic']))

        # Generate the keypoint map
        if has_keypoints:
            data = data.map_parallel(pipeline.add_keypoint_map)
        data = data.map_parallel(
            lambda d: {**d, 'image': tf.cast(d['image'], tf.float32) / 255.})

        if has_seg_masks:
            data = data.map_parallel(
            lambda d: {**d, 'mask_image': tf.cast(d['mask_image'], tf.float32) / 255})

        if config['warped_pair']['enable']:
            data = data.map_parallel(
                lambda d: {
                    **d, 'warped': {**d['warped'],
                                    'image': tf.cast(d['warped']['image'], tf.float32) / 255.}})
            if has_seg_masks:
                data = data.map_parallel(
                    lambda d: {
                        **d, 'warped': {**d['warped'],
                                        'mask_image': tf.cast(d['warped']['mask_image'], tf.float32) / 255}})

        return data
