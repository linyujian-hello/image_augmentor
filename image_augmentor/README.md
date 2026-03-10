# Image Augmentation Tool for Small Sample Scenarios

## Project Overview

This project provides a set of image enhancement tools based on Python/OpenCV, suitable for data augmentation in small sample scenarios. It includes various classic augmentation techniques such as geometric transformations, color adjustments, noise injection, and advanced mixing methods like MixUp, CutMix, and Mosaic.

## Directory Structure

| Part              | Description                              |
| :---------------- | :--------------------------------------- |
| utils             | format conversion, resizing utilities    |
| operators         | Augmentation operators                   |
| flip              | Random flipping                          |
| rotate            | Random rotation                          |
| crop_resize       | Random cropping and resizing             |
| hsv               | HSV color perturbation, grayscale, CLAHE |
| noise             | Gaussian noise and salt-and-pepper noise |
| mixup             | MixUp mixing                             |
| cutmix            | CutMix mixing                            |
| mosaic            | Mosaic four-image stitching              |
| preprocess_images | Preprocess raw images                    |

## Contributing

Contributions are welcome! Please feel free to contact me. Before contributing, ensure your code adheres to the existing style and includes appropriate tests.