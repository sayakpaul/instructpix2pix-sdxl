import torch 
import torchvision.transforms as transforms 
import webdataset as wds
from torch.utils.data import default_collate
from huggingface_hub import create_repo, upload_folder
import PIL
import os

DOWNSAMPLE_TO = 256
BATCH_SIZE = 64
DATASET_PATH = "pipe:aws s3 cp s3://muse-datasets/instructpix2pix-clip-filtered-wds/{000000..000062}.tar -"

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f

def postprocess_image(output: torch.Tensor) -> PIL.Image.Image:
    output = output.float().cpu().clamp_(0, 1).numpy()
    output = (output * 255).round().astype("uint8")
    output = output.transpose(1, 2, 0)  # CHW -> HWC
    return PIL.Image.fromarray(output)


def get_dataloader(num_workers):
    resize = transforms.Resize((DOWNSAMPLE_TO, DOWNSAMPLE_TO))

    def preprocess_images(sample):
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = torch.stack(
            [
                transforms.ToTensor()(sample["original_image"]),
                transforms.ToTensor()(sample["edited_image"]),
            ]
        )
        transformed_images = resize(images)

        # Separate the original and edited images.
        original_image, edited_image = transformed_images.chunk(2)

        # Pad the images.
        def pad(img_lq):
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // WINDOW_SIZE + 1) * WINDOW_SIZE - h_old
            w_pad = (w_old // WINDOW_SIZE + 1) * WINDOW_SIZE - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                :, :, : h_old + h_pad, :
            ]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                :, :, :, : w_old + w_pad
            ]
            return img_lq.squeeze(0)

        # original_image = pad(original_image)
        # edited_image = pad(edited_image)

        return {
            "original_image": original_image,
            "edited_image": edited_image,
            "edit_prompt": sample["edit_prompt"],
            "original_prompt": sample["original_prompt"],
        }

    dataset = (
        wds.WebDataset(DATASET_PATH, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .rename(
            original_prompt="original_prompt.txt",
            original_image="original_image.jpg",
            edit_prompt="edit_prompt.txt",
            edited_image="edited_image.jpg",
            handler=wds.warn_and_continue,
        )
        .map(
            filter_keys(
                {"original_image", "edited_image", "edit_prompt", "original_prompt"}
            ),
            handler=wds.warn_and_continue,
        )
        .map(preprocess_images, handler=wds.warn_and_continue)
        .batched(BATCH_SIZE, partial=True, collation_fn=default_collate)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader

if __name__ == "__main__":
    folder_path = os.makedirs("sample_images", exist_ok=True)

    dataloader = get_dataloader(4)
    for sample in dataloader:
        break 
    
    repo_id = create_repo(repo_id="pipe-instructpix2pix", repo_type="dataset", exist_ok=True).repo_id
    for i in range(len(sample)):
        if i == 5:
            break
        
        original_image = postprocess_image(sample["original_image"][i].squeeze())
        edited_image = postprocess_image(sample["edited_image"][i].squeeze())
        original_image.save(os.path.join(folder_path, f"{i}_original.png"))
        edited_image.save(os.path.join(folder_path, f"{i}_edited.png"))
    
    upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="dataset")