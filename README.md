# Lawn_Segmentation
Lawn boundary detection and lawn segmentation

![](https://github.com/nitin02s/Lawn_Segmentation/blob/main/result.gif)

Run python predict.py to view the output of the model. The input images are present in test_images\ directory. A custom input can also be passed by parsing the directory of the image as an argument. The script takes --input/-i as the argument for the input image directory. All images in this directory are passed as input by default to the model.

Reference code : 
Network: milesial/Pytorch-UNet: PyTorch implementation of the U-Net for image semantic segmentation with high quality images (github.com)
Metrics : VainF/DeepLabV3Plus-Pytorch: Pretrained DeepLabv3 and DeepLabv3+ for Pascal VOC & Cityscapes (github.com)

The report is present in PDF form as report.pdf

