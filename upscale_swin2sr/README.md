Run `upscale_images.py` to upscale the images of [timbrooks/instructpix2pix-clip-filtered](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered). The upscaler is Swin2SR from [this repository](https://github.com/mv-lab/swin2sr). `swin2sr` is entirely from the original repository. `upscale_image.py` is a modified version of `predict.py` from [here](https://github.com/mv-lab/swin2sr/blob/main/predict.py). 

* `upscale_images.py` will produce a dataset (of 🤗 datasets) format locally. 
* Then use `generate_wds_shards.py` to get the dataset converted to the `webdataset` format.