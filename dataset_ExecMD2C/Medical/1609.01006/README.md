# Combining Fully Convolutional and Recurrent Neural Networks for 3D Biomedical Image Segmentation (1609.01006)

## Project source

- **Project ID:** `1609.01006`
- **Paper:** Combining Fully Convolutional and Recurrent Neural Networks for 3D Biomedical Image Segmentation
- **PDF:** http://arxiv.org/pdf/1609.01006v2.pdf
- **GitHub:** https://github.com/shreyaspadhy/unet-zoo

## Reproduction task

The network segments 3D biomedical images by combining convolutional and recurrent layers. In `template.py`, complete `UNet.forward`, `BDCLSTM.forward`, and the other TODO-marked functions to run the full four-level UNet encoder-decoder path with skip concatenations and fuse three adjacent UNet feature maps with bidirectional ConvLSTM for segmentation.
