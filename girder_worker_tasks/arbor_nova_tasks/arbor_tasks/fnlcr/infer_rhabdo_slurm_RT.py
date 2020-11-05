from tempfile import NamedTemporaryFile
import time
import random
import torch
import torch.nn as nn
import cv2

import os, glob
import numpy as np
from skimage.io import imread, imsave
from skimage import filters
from skimage.color import rgb2gray
import gc

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import albumentations as albu
# import segmentation_models_pytorch as smp

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    binding_to_type = {"input.1": np.float32, "1419": np.float32}
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = binding_to_type[str(binding)]
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# trt_engine_path = './rhabdo_80_3_96_96.engine'
trt_engine_path = '/mnt/hpc/webdata/server/fr-s-ivg-ssr-d1/RTEngines/rhabdo_80_3_384_384.engine'
trt_runtime = trt.Runtime(TRT_LOGGER)
trt_engine = load_engine(trt_runtime, trt_engine_path)
onnx_inputs, onnx_outputs, onnx_bindings, onnx_stream = allocate_buffers(trt_engine)
print(onnx_inputs)
context = trt_engine.create_execution_context()
ml = nn.Softmax(dim=1)

NE = 50
ST = 100
ER = 150
AR = 200
PRINT_FREQ = 20
BATCH_SIZE = 80

ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DEVICE = 'cuda'

# the weights file is in the same directory, so make this path reflect that.  If this is 
# running in a docker container, then we should assume the weights are at the toplevel 
# directory instead

if (os.getenv('DOCKER') == 'True') or (os.getenv('DOCKER') == 'True'):
    WEIGHT_PATH = '/'
else:
    WEIGHT_PATH = '/'

# these aren't used in the girder version, no files are directly written out 
# by the routines written by FNLCR (Hyun Jung)
WSI_PATH = '.'
PREDICTION_PATH = '.'

IMAGE_SIZE = 384
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
CHANNELS = 3
NUM_CLASSES = 5
CLASS_VALUES = [0, 50, 100, 150, 200]

BLUE = [0, 0, 255] # ARMS: 200
RED = [255, 0, 0] # ERMS: 150
GREEN = [0, 255, 0] # STROMA: 100
YELLOW = [255, 255, 0] # NECROSIS: 50
EPSILON = 1e-6

rot90 = albu.Rotate(limit=(90, 90), always_apply=True)
rotn90 = albu.Rotate(limit=(-90, -90), always_apply=True)

rot180 = albu.Rotate(limit=(180, 180), always_apply=True)
rotn180 = albu.Rotate(limit=(-180, -180), always_apply=True)

rot270 = albu.Rotate(limit=(270, 270), always_apply=True)
rotn270 = albu.Rotate(limit=(-270, -270), always_apply=True)

hflip = albu.HorizontalFlip(always_apply=True)
vflip = albu.VerticalFlip(always_apply=True)
tpose = albu.Transpose(always_apply=True)

pad = albu.PadIfNeeded(p=1.0, min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, value=(255, 255, 255), mask_value=0)

# supporting subroutines
#-----------------------------------------------------------------------------

def _generate_th(image_org):
    image = np.copy(image_org)

    org_height = image.shape[0]
    org_width = image.shape[1]

    otsu_seg = np.zeros((org_height//4, org_width//4), np.uint8)

    aug = albu.Resize(p=1.0, height=org_height // 4, width=org_width // 4)
    augmented = aug(image=image)
    thumbnail = augmented['image']

    thumbnail_gray = rgb2gray(thumbnail)
    val = filters.threshold_otsu(thumbnail_gray)     
    otsu_seg[thumbnail_gray <= val] = 255

    aug = albu.Resize(p=1.0, height=org_height, width=org_width)
    augmented = aug(image=otsu_seg, mask=otsu_seg)
    otsu_seg = augmented['mask']

    print('Otsu segmentation finished')

    return otsu_seg

def _infer_batch(test_patch):
    np.copyto(onnx_inputs[0].host, test_patch[:, :, :, :].ravel())
    logits_all = do_inference(context, bindings=onnx_bindings, inputs=onnx_inputs, outputs=onnx_outputs, stream=onnx_stream)
    logits_all = np.asarray(logits_all).reshape(80, 5, IMAGE_SIZE, IMAGE_SIZE)
    logits_all = torch.from_numpy(logits_all)

    logits = logits_all[:, 0:NUM_CLASSES, :, :]
    prob_classes_int = ml(logits)
    prob_classes_all = prob_classes_int.cpu().numpy().transpose(0, 2, 3, 1)

    return prob_classes_all

def _augment(index, image):

    if index == 0:
        image= image

    if index == 1:
        augmented = rot90(image=image)
        image = augmented['image']

    if index ==2:
        augmented = rot180(image=image)
        image= augmented['image']

    if index == 3:
        augmented = rot270(image=image)
        image = augmented['image']

    if index == 4:
        augmented = vflip(image=image)
        image = augmented['image']

    if index == 5:
        augmented = hflip(image=image)
        image = augmented['image']

    if index == 6:
        augmented = tpose(image=image)
        image = augmented['image']

    return image
    
def _unaugment(index, image):

    if index == 0:
        image= image

    if index == 1:
        augmented = rotn90(image=image)
        image = augmented['image']

    if index ==2:
        augmented = rotn180(image=image)
        image= augmented['image']

    if index == 3:
        augmented = rotn270(image=image)
        image = augmented['image']

    if index == 4:
        augmented = vflip(image=image)
        image = augmented['image']

    if index == 5:
        augmented = hflip(image=image)
        image = augmented['image']

    if index == 6:
        augmented = tpose(image=image)
        image = augmented['image']

    return image

def _gray_to_color(input_probs):

    index_map = (np.argmax(input_probs, axis=-1)*50).astype('uint8')
    height = index_map.shape[0]
    width = index_map.shape[1]

    heatmap = np.zeros((height, width, 3), np.float32)

    # Background
    heatmap[index_map == 0, 0] = input_probs[:, :, 0][index_map == 0]
    heatmap[index_map == 0, 1] = input_probs[:, :, 0][index_map == 0]
    heatmap[index_map == 0, 2] = input_probs[:, :, 0][index_map == 0]

    # Necrosis
    heatmap[index_map==50, 0] = input_probs[:, :, 1][index_map==50]
    heatmap[index_map==50, 1] = input_probs[:, :, 1][index_map==50]
    heatmap[index_map==50, 2] = 0.

    # Stroma
    heatmap[index_map==100, 0] = 0.
    heatmap[index_map==100, 1] = input_probs[:, :, 2][index_map==100]
    heatmap[index_map==100, 2] = 0.

    # ERMS
    heatmap[index_map==150, 0] = input_probs[:, :, 3][index_map==150]
    heatmap[index_map==150, 1] = 0.
    heatmap[index_map==150, 2] = 0.

    # ARMS
    heatmap[index_map==200, 0] = 0.
    heatmap[index_map==200, 1] = 0.
    heatmap[index_map==200, 2] = input_probs[:, :, 4][index_map==200]

    heatmap[np.average(heatmap, axis=-1)==0, :] = 1.

    return heatmap

#---------------- main inferencing routine ------------------
def _inference(image_path, BATCH_SIZE, num_classes, kernel, num_tta=1):
    image_org = imread(image_path)
    height_org = image_org.shape[0]
    width_org = image_org.shape[1]

    basename_string = os.path.splitext(os.path.basename(image_path))[0]
    print('Basename String: ', basename_string)

    otsu_org = _generate_th(image_org)//255
    prob_map_seg_stack = np.zeros((height_org, width_org, num_classes), dtype=np.float32)

    for b in range(num_tta):
        image_working = np.copy(image_org)
        image_working = _augment(b, image_working)

        height = image_working.shape[0]
        width = image_working.shape[1]

        PATCH_OFFSET = IMAGE_SIZE // 2
        SLIDE_OFFSET = IMAGE_SIZE // 2

        heights = (height+ PATCH_OFFSET * 2 - IMAGE_SIZE) // SLIDE_OFFSET + 1
        widths = (width+ PATCH_OFFSET * 2 - IMAGE_SIZE) // SLIDE_OFFSET + 1

        height_ext = SLIDE_OFFSET * heights + PATCH_OFFSET * 2
        width_ext = SLIDE_OFFSET * widths + PATCH_OFFSET * 2

        org_slide_ext = np.ones((height_ext, width_ext, 3), np.uint8) * 255
        otsu_ext = np.zeros((height_ext, width_ext), np.uint8)
        prob_map_seg = np.zeros((height_ext, width_ext, num_classes), dtype=np.float32)
        weight_sum = np.zeros((height_ext, width_ext, num_classes), dtype=np.float32)

        org_slide_ext[PATCH_OFFSET: PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width, 0:3] = image_working[:, :, 0:3]
        otsu_ext[PATCH_OFFSET: PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width] = otsu_org[:, :]

        linedup_predictions = np.zeros((heights*widths, IMAGE_SIZE, IMAGE_SIZE, num_classes), dtype=np.float32)
        linedup_predictions[:, :, :, 0] = 1.0
        # test_patch_tensor = torch.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=torch.float).cuda(non_blocking=True)
        test_patch_array = np.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32)
        patch_iter = 0
        inference_index = []
        position = 0
        for i in range(heights):
            for j in range(widths):
                test_patch = org_slide_ext[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE, j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE, 0:3]
                otsu_patch = otsu_ext[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE, j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE]
                if np.sum(otsu_patch) > int(0.05*IMAGE_SIZE*IMAGE_SIZE):
                    inference_index.append(patch_iter)
                    test_patch_tensor[position, :, :, :] = torch.from_numpy(test_patch.transpose(2, 0, 1)
                                                                     .astype('float32')/255.0)
                    position += 1
                patch_iter+=1

                if position==BATCH_SIZE:
                    # batch_predictions = _infer_batch(model, test_patch_tensor)
                    batch_predictions = _infer_batch(test_patch_array)
                    for k in range(BATCH_SIZE):
                        linedup_predictions[inference_index[k], :, :, :] = batch_predictions[k, :, :, :]

                    position = 0
                    inference_index = []

        # Very last part of the region
        # batch_predictions = _infer_batch(model, test_patch_tensor)
        batch_predictions = _infer_batch(test_patch_array)
        for k in range(position):
            linedup_predictions[inference_index[k], :, :, :] = batch_predictions[k, :, :, :]

        print('GPU inferencing complete. Constructing out image from patches')

        patch_iter = 0
        for i in range(heights):
            for j in range(widths):
                prob_map_seg[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE, j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE, :] \
                                += np.multiply(linedup_predictions[patch_iter, :, :, :], kernel)
                weight_sum[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE, j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE, :] \
                                += kernel
                patch_iter += 1

        prob_map_seg = np.true_divide(prob_map_seg, weight_sum)
        prob_map_valid = prob_map_seg[PATCH_OFFSET:PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width, :]
        prob_map_valid = _unaugment(b, prob_map_valid)

        prob_map_seg_stack += prob_map_valid/num_tta

    pred_map_final = np.argmax(prob_map_seg_stack, axis=-1)
    pred_map_final_gray = pred_map_final.astype('uint8') * 50
    pred_map_final_ones = [(pred_map_final_gray == v) for v in CLASS_VALUES]
    pred_map_final_stack = np.stack(pred_map_final_ones, axis=-1).astype('uint8')

    # for girder task, don't return this image, so commented out
    #prob_colormap = _gray_to_color(prob_map_seg_stack)
    #imsave( basename_string + '_prob.png', (prob_colormap * 255.0).astype('uint8'))

    pred_colormap = _gray_to_color(pred_map_final_stack)
    # return image instead of saving directly here
    #imsave(basename_string + '_pred.png', (pred_colormap*255.0).astype('uint8'))
    return (pred_colormap*255.0).astype('uint8')


def _gaussian_2d(num_classes, sigma, mu):
    x, y = np.meshgrid(np.linspace(-1, 1, IMAGE_SIZE), np.linspace(-1, 1, IMAGE_SIZE))
    d = np.sqrt(x * x + y * y)
    # sigma, mu = 1.0, 0.0
    k = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    k_min = np.amin(k)
    k_max = np.amax(k)

    k_normalized = (k - k_min) / (k_max - k_min)
    k_normalized[k_normalized<=EPSILON] = EPSILON

    kernels = [(k_normalized) for i in range(num_classes)]
    kernel = np.stack(kernels, axis=-1)

    print('Kernel shape: ', kernel.shape)
    print('Kernel Min value: ', np.amin(kernel))
    print('Kernel Max value: ', np.amax(kernel))

    return kernel

def inference_image(image_path, BATCH_SIZE, num_classes):
    kernel = _gaussian_2d(num_classes, 0.5, 0.0)
    start_predict = time.time()
    predict_image = _inference(image_path, BATCH_SIZE, num_classes, kernel, 1)
    end_predict = time.time()
    print("TOTAL predict takes:{}".format(end_predict - start_predict))
    return predict_image

def start_inference(image_file):
    predict_image = inference_image(image_file, BATCH_SIZE, len(CLASS_VALUES))
    return predict_image


import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputImage', help='tmp directory for current slurm job data input/ouput.', required=True)
parser.add_argument('-d', '--directory', help='output directory names format like ..hpc/tmp/slurm-jobname.jobid.', required=True)

kwargs = vars(parser.parse_args())

inputImage = kwargs.pop('inputImage')
outPath = kwargs.pop('directory')

print(" input image filename = {}".format(inputImage))

# setup the GPU environment for pytorch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda'

print('perform forward inferencing')
predict_image = start_inference(inputImage)
predict_bgr = cv2.cvtColor(predict_image,cv2.COLOR_RGB2BGR)
print('output conversion and inferencing complete')

# generate unique names for multiple runs.  Add extension so it is easier to use
outname = NamedTemporaryFile(delete=False, dir=outPath).name+'.png'

# write the output object using openCV  
print('writing output')
cv2.imwrite(outname, predict_bgr)
print('writing completed')