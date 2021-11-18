# MSI-RES

## Requirements
- [DRLN](https://github.com/saeed-anwar/DRLN)
- [ESRGAN](https://github.com/idealo/image-super-resolution)
- [CARE](https://github.com/CSBDeep/CSBDeep)

## Models

The notebooks in this folder demonstrate how to train and test denoising and upsampling using the three models above.

## Evaluations

The metrics used to assess the model performances are the image quality metrics: **PSNR**, **Piqe** and **Brisque**. In addition, resolution of each image is also approximated using the script `msi_res.py`. These information are then collected and compared in `boxplot.ipynb` and `boxplot_x4.ipynb` Finally biological implications are drawn using the `Brain_clf.ipynb`.


## Utilities

This provides the necessary tools for changing the data to the appropriate formats. 

- CARE uses .tiff
- ESRGAN uses .png or .jpeg
- DRLN uses .npy
