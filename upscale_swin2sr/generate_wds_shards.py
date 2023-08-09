import os

import ray
import webdataset as wds
from datasets import Dataset

ray.init()


def main():
    dataset_path = "/scratch/suraj/instructpix2pix-clip-filtered-upscaled"
    wds_shards_path = "/scratch/suraj/instructpix2pix-clip-filtered-upscaled-wds"
    # get all .arrow files in the dataset path
    dataset_files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.endswith(".arrow")
    ]

    @ray.remote
    def create_shard(path):
        # get basename of the file
        basename = os.path.basename(path)
        # get the shard number data-00123-of-01034.arrow -> 00123
        shard_num = basename.split("-")[1]
        dataset = Dataset.from_file(path)
        # create a webdataset shard
        shard = wds.TarWriter(os.path.join(wds_shards_path, f"{shard_num}.tar"))
        for i, example in enumerate(dataset):
            wds_example = {
                "__key__": str(i),
                "original_prompt.txt": example["original_prompt"],
                "original_image.jpg": example["original_image"].convert("RGB"),
                "edit_prompt.txt": example["edit_prompt"],
                "edited_prompt.txt": example["edited_prompt"],
                "edited_image.jpg": example["edited_image"].convert("RGB"),
            }
            shard.write(wds_example)
        shard.close()

    futures = [create_shard.remote(path) for path in dataset_files]
    ray.get(futures)


if __name__ == "__main__":
    main()
