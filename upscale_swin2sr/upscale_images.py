from swin2sr.models.network_swin2sr import Swin2SR as net

import torch
import webdataset as wds
from torch.utils.data import default_collate
from torchvision import transforms

from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import PIL
import os
import requests
import torch
from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator
import tempfile
from tqdm import tqdm

torch.set_grad_enabled(False)

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

MODEL_PATH = "model_zoo/swin2sr/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth"
PARAM_KEY_G = "params_ema"
SCALE = 4
WINDOW_SIZE = 8
DOWNSAMPLE_TO = 256
BATCH_SIZE = 64

NUM_WORKERS = 4

DATASET_PATH = "pipe:aws s3 cp s3://muse-datasets/instructpix2pix-clip-filtered-wds/{000000..000062}.tar -"
NEW_DATASET_NAME = "instructpix2pix-clip-filtered-upscaled"
PROJECT_DIR = "/scratch"


def download_model_weights() -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/{}".format(
        os.path.basename(MODEL_PATH)
    )
    r = requests.get(url, allow_redirects=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)


def load_model() -> torch.nn.Module:
    if not os.path.exists(MODEL_PATH):
        download_model_weights()
    model = net(
        upscale=SCALE,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )
    pretrained_model = torch.load(MODEL_PATH)
    model.load_state_dict(
        pretrained_model[PARAM_KEY_G]
        if PARAM_KEY_G in pretrained_model.keys()
        else pretrained_model,
        strict=True,
    )
    return model


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


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

        original_image = pad(original_image)
        edited_image = pad(edited_image)

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


def postprocess_image(output: torch.Tensor) -> PIL.Image.Image:
    output = output.data.float().cpu().clamp_(0, 1).numpy()
    output = (output * 255).round().astype("uint8")
    output = output.transpose(1, 2, 0)  # CHW -> HWC
    return PIL.Image.fromarray(output)


def gen_examples(
    original_prompts, original_images, edit_prompts, edited_images
):
    def fn():
        for i in range(len(original_prompts)):
            yield {
                "original_prompt": original_prompts[i],
                "original_image": {"path": original_images[i]},
                "edit_prompt": edit_prompts[i],
                "edited_image": {"path": edited_images[i]},
            }

    return fn


if __name__ == "__main__":
    accelerator_project_config = ProjectConfiguration(
        project_dir=PROJECT_DIR, logging_dir=PROJECT_DIR
    )
    accelerator = Accelerator(project_config=accelerator_project_config)

    model = load_model().eval()
    model = accelerator.prepare(model)

    dataloader = get_dataloader(num_workers=NUM_WORKERS)
    if accelerator.is_main_process:
        print("Model loaded.")
        print("Dataloader prepared.")

    all_upscaled_original_paths = []
    all_upscaled_edited_paths = []
    all_original_prompts = []
    all_edit_prompts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, batch in enumerate(tqdm(dataloader)):
            if idx == 1:
                break
            # Collate the original and edited images so that we do only a single
            # forward pass.
            images = [image for image in batch["original_image"]]
            images += [image for image in batch["edited_image"]]
            images = torch.stack(images).to(
                accelerator.device,
                memory_format=torch.contiguous_format,
                non_blocking=True,
            )

            # Inference.
            with torch.autocast(
                device_type=accelerator.device.type, dtype=torch.float16
            ):
                output_images = model(images).float()

            # Post-process.
            original_images, edited_images = output_images.chunk(2)
            original_images = [postprocess_image(image) for image in original_images]
            edited_images = [postprocess_image(image) for image in edited_images]

            all_original_prompts += [prompt for prompt in batch["original_prompt"]]
            all_edit_prompts += [prompt for prompt in batch["edit_prompt"]]

            orig_img_paths = [
                os.path.join(PROJECT_DIR, tmpdir, f"{idx}_{i}_original_img.png")
                for i in range(len(original_images))
            ]
            all_upscaled_original_paths += [path for path in orig_img_paths]
            edited_img_paths = [
                os.path.join(PROJECT_DIR, tmpdir, f"{idx}_{i}_edited_img.png")
                for i in range(len(edited_images))
            ]
            all_upscaled_edited_paths += [path for path in edited_img_paths]

            for i in range(len(orig_img_paths)):
                original_images[i].save(orig_img_paths[i])
                edited_images[i].save(edited_img_paths[i])

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        generator_fn = gen_examples(
            original_prompts=all_original_prompts,
            original_images=all_upscaled_original_paths,
            edit_prompts=all_edit_prompts,
            edited_images=all_upscaled_edited_paths,
        )
        ds = Dataset.from_generator(
            generator_fn,
            features=Features(
                original_prompt=Value("string"),
                original_image=ImageFeature(),
                edit_prompt=Value("string"),
                edited_image=ImageFeature(),
            ),
        )
        ds.push_to_hub(NEW_DATASET_NAME)
