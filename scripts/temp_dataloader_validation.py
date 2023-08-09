from dataloader import get_dataloader
from argparse import Namespace
from PIL import Image
from huggingface_hub import create_repo, upload_folder
import os

OUTPUT_DIR = "verify_samples"

if __name__ == "__main__":
    args = Namespace(
        dataset_path="pipe:aws s3 cp s3://muse-datasets/instructpix2pix-clip-filtered-upscaled-wds/{00000..00519}.tar -",
        num_train_examples=313010,
        per_gpu_batch_size=8,
        global_batch_size=64,
        num_workers=4,
        center_crop=False,
        random_flip=True,
        resolution=256,
        original_image_column="original_image",
        edit_prompt_column="edit_prompt",
        edited_image_column="edited_image",
    )
    dataloader = get_dataloader(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sample in dataloader:
        print(sample.keys())
        print(sample["original_image"].shape)
        print(sample["edited_image"].shape)
        print(len(sample["edit_prompt"]))

        for i in range(len(sample["original_image"])):
            current_orig_sample = sample["original_image"][i].numpy().squeeze()
            current_orig_sample = current_orig_sample.transpose((1, 2, 0))
            current_orig_sample *= 255.0
            current_orig_sample = current_orig_sample.round().astype("uint8")
            current_orig_sample = Image.fromarray(current_orig_sample)

            current_edited_sample = sample["edited_image"][i].numpy().squeeze()
            current_edited_sample = current_edited_sample.transpose((1, 2, 0))
            current_edited_sample *= 255.0
            current_edited_sample = current_edited_sample.round().astype("uint8")
            current_edited_sample = Image.fromarray(current_edited_sample)

            current_orig_sample.save(os.path.join(OUTPUT_DIR, f"{i}_orig.png"))
            current_edited_sample.save(os.path.join(OUTPUT_DIR, f"{i}_edited.png"))
            with open(os.path.join(OUTPUT_DIR, f"{i}_edited_prompt.txt"), "w") as f:
                f.write(sample["edit_prompt"][i])

        break

    repo_id = create_repo(repo_id="upscaled-validation-logging", exist_ok=True).repo_id
    upload_folder(repo_id=repo_id, folder_path=OUTPUT_DIR)
