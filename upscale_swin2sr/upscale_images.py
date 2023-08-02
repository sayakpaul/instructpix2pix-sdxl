from swin2sr.models.network_swin2sr import Swin2SR as net

from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value
import numpy as np
import PIL
import os
import requests
import torch
import datasets
from torch.utils.data import DataLoader
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
BATCH_SIZE = 96

DATASET_NAME = "timbrooks/instructpix2pix-clip-filtered"
NEW_DATASET_NAME = "instructpix2pix-clip-filtered-upscaled"


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


def preprocesss_image(image: PIL.Image.Image) -> torch.FloatTensor:
    image = image.resize((DOWNSAMPLE_TO, DOWNSAMPLE_TO))
    image = np.array(image).astype("float32") / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    img_lq = torch.from_numpy(image).float().unsqueeze(0)

    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // WINDOW_SIZE + 1) * WINDOW_SIZE - h_old
    w_pad = (w_old // WINDOW_SIZE + 1) * WINDOW_SIZE - w_old
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, : h_old + h_pad, :]
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, : w_old + w_pad]
    return img_lq.squeeze(0)


def postprocess_image(output: torch.Tensor) -> PIL.Image.Image:
    output = output.data.float().cpu().clamp_(0, 1).numpy()
    output = (output * 255).round().astype("uint8")
    output = output.transpose(1, 2, 0)  # CHW -> HWC
    return PIL.Image.fromarray(output)


def gen_examples(
    original_prompts, original_images, edit_prompts, edited_prompts, edited_images
):
    def fn():
        for i in range(len(original_prompts)):
            yield {
                "original_prompt": original_prompts[i],
                "original_image": {"path": original_images[i]},
                "edit_prompt": edit_prompts[i],
                "edited_prompt": edited_prompts[i],
                "edited_image": {"path": edited_images[i]},
            }

    return fn


if __name__ == "__main__":
    dataset = datasets.load_dataset(
        DATASET_NAME, split="train", num_proc=4, cache_dir="/scratch"
    )
    print(f"Dataset has got {len(dataset)} samples.")

    model = load_model()
    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs.")
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
        BATCH_SIZE *= num_gpus
    model = model.eval().cuda()
    print("Model loaded.")

    def pp(examples):
        examples["original_image"] = [
            preprocesss_image(image) for image in examples["original_image"]
        ]
        examples["edited_image"] = [
            preprocesss_image(image) for image in examples["edited_image"]
        ]
        examples["original_prompt"] = [prompt for prompt in examples["original_prompt"]]
        examples["edit_prompt"] = [prompt for prompt in examples["edit_prompt"]]
        examples["edited_prompt"] = [prompt for prompt in examples["edited_prompt"]]
        return examples

    dataset = dataset.with_transform(pp)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True)
    print("Dataloader prepared.")

    all_upscaled_original_paths = []
    all_upscaled_edited_paths = []
    all_original_prompts = []
    all_edit_prompts = []
    all_edited_prompts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, batch in enumerate(tqdm(dataloader)):
            original_images = model(batch["original_image"].cuda())
            original_images = [postprocess_image(image) for image in original_images]
            edited_images = model(batch["edited_image"].cuda())
            edited_images = [postprocess_image(image) for image in edited_images]

            all_original_prompts += [prompt for prompt in batch["original_prompt"]]
            all_edit_prompts += [prompt for prompt in batch["edit_prompt"]]
            all_edited_prompts += [prompt for prompt in batch["edited_prompt"]]

            orig_img_paths = [
                os.path.join(tmpdir, f"{idx}_{i}_original_img.png")
                for i in range(len(original_images))
            ]
            all_upscaled_original_paths += [path for path in orig_img_paths]
            edited_img_paths = [
                os.path.join(tmpdir, f"{idx}_{i}_edited_img.png")
                for i in range(len(edited_images))
            ]
            all_upscaled_edited_paths += [path for path in edited_img_paths]

            for i in range(len(orig_img_paths)):
                original_images[i].save(orig_img_paths[i])
                edited_images[i].save(edited_img_paths[i])

    generator_fn = gen_examples(
        original_prompts=all_original_prompts,
        original_images=all_upscaled_original_paths,
        edit_prompts=all_edit_prompts,
        edited_prompts=all_edited_prompts,
        edited_images=all_upscaled_edited_paths,
    )
    ds = Dataset.from_generator(
        generator_fn,
        features=Features(
            original_prompt=Value("string"),
            original_image=ImageFeature(),
            edit_prompt=Value("string"),
            edited_prompt=Value("string"),
            edited_image=ImageFeature(),
        ),
    )
    ds.push_to_hub(NEW_DATASET_NAME)
