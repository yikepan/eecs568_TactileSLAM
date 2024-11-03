# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

import os
from glob import glob

import git
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from neuralfeels.contrib.tactile_transformer.dpt_model import DPTModel
from neuralfeels.contrib.tactile_transformer.utils import (
    apply_jet_colormap,
    concat_images,
    create_dir,
)

root = git.Repo(".", search_parent_directories=True).working_tree_dir


class TouchVIT:
    """
    Image to 3D model for DIGIT
    """

    def __init__(self, cfg: DictConfig):
        super(TouchVIT, self).__init__()

        self.config = cfg
        input_dir = to_absolute_path(self.config["General"]["path_input_images"])
        self.input_images = glob(f"{input_dir}/*.jpg") + glob(f"{input_dir}/*.png")

        self.type = self.config["General"]["type"]

        self.device = torch.device(
            self.config["General"]["device"] if torch.cuda.is_available() else "cpu"
        )
        # print("device: %s" % self.device)
        resize = self.config["Dataset"]["transforms"]["resize"]
        self.model = DPTModel(
            image_size=(3, resize[0], resize[1]),
            emb_dim=self.config["General"]["emb_dim"],
            resample_dim=self.config["General"]["resample_dim"],
            read=self.config["General"]["read"],
            nclasses=len(self.config["Dataset"]["classes"]),
            hooks=self.config["General"]["hooks"],
            model_timm=self.config["General"]["model_timm"],
            type=self.type,
            patch_size=self.config["General"]["patch_size"],
        )
        path_model = to_absolute_path(
            os.path.join(
                self.config["General"]["path_model"],
                f"{self.config['weights']}.p",
            )
        )

        # print(f"TouchVIT path: {path_model}")
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()
        self.model.to(self.device)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((resize[0], resize[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.output_dir = self.config["General"]["path_predicted_images"]

    def image2heightmap(self, image):
        image = Image.fromarray(image)
        original_size = image.size
        image = self.transform_image(image).unsqueeze(0)
        image = image.to(self.device).float()

        output_depth, _ = self.model(image)  # [0 - 1] output

        output_depth = transforms.ToPILImage()(output_depth.squeeze(0).float()).resize(
            original_size, resample=Image.BICUBIC
        )  # [0 - 255] output
        return transforms.PILToTensor()(output_depth).squeeze()

    def run(self):
        path_dir_depths = os.path.join(self.output_dir, "depths")
        create_dir(self.output_dir)
        create_dir(path_dir_depths)

        output_depths, input_images = [], []
        for images in self.input_images[:10]:
            pil_im = Image.open(images)
            im = np.array(pil_im)
            with torch.no_grad():
                output_depth = self.image2heightmap(im)
                output_depths.append(output_depth)
            input_images.append(pil_im)

        # Convert list of tensors to image collage
        output_depths = [transforms.ToPILImage()(depth) for depth in output_depths]
        # Concatenate all 10 PIL images
        collage_depth = concat_images(output_depths, direction="horizontal")
        collage_depth = apply_jet_colormap(collage_depth)
        collage_images = concat_images(input_images, direction="horizontal")
        collage = concat_images([collage_images, collage_depth], direction="vertical")
        # add jet colormap to the collage
        collage.show()


@hydra.main(
    version_base=None,
    config_path=os.path.join(root, "scripts/config/main/touch_depth"),
    config_name="vit",
)
def main(cfg: DictConfig):
    cfg.weights = "dpt_sim"
    t = TouchVIT(cfg)
    t.run()


if __name__ == "__main__":
    main()
