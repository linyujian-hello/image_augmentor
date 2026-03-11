# Image Augmentation Tool for Small Sample Scenarios

## Project Overview

This project provides a set of image augmentation tools based on Python and OpenCV, suitable for data augmentation in small-sample scenarios. It includes various classic augmentation techniques such as geometric transformations, color adjustments, noise injection, and advanced mixing methods like MixUp, CutMix, and Mosaic.

## Directory Structure

| Part              | Description                              |
| ----------------- | ---------------------------------------- |
| utils             | format conversion, resizing utilities    |
| operators         | augmentation operators                   |
| flip              | random flipping                          |
| rotate            | random rotation                          |
| crop_resize       | random cropping and resizing             |
| hsv               | HSV color perturbation, grayscale, CLAHE |
| noise             | gaussian noise and salt-and-pepper noise |
| mixup             | MixUp mixing                             |
| cutmix            | CutMix mixing                            |
| mosaic            | Mosaic four-image stitching              |
| preprocess_images | preprocess raw images                    |

------

## Application Scenarios

This augmentation toolset is designed primarily for machine learning and computer vision workflows, especially when labeled data are scarce. Representative usage scenarios include:

- **Image classification** — expand small labeled datasets to reduce overfitting and improve generalization.
- **Object detection** — use Mosaic, CutMix and geometric transformsto enrich scenes and increase small-object prevalence.
- **Semantic segmentation** — when augmented in sync with masks, improves robustness to occlusions, lighting, and noise.
- **Domain adaptation & transfer learning** — simulate diverse imaging conditions  to bridge domain gaps between source and target datasets.
- **Robotics & autonomous systems** — simulate sensor noise and viewpoint changes to make perception models more robust.

------

## Performance & Scaling

- Current scripts are single-process and work well for small-to-moderate datasets. For large-scale preprocessing, consider:
  - Parallel processing.
  - Batch-level augmentation implemented inside the training `DataLoader`.
  - Caching and streaming augmented images if IO is the bottleneck.

## Contributing

Contributions are welcome! Please feel free to contact me. Before contributing, ensure your code adheres to the existing style and includes appropriate tests.