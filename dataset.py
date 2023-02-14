import os
from typing import Union

import datasets
import pandas as pd
import json
from PIL import Image

_CITATION = ""
_DESCRIPTION = "VizWiz-Captions, consists of 39,181 images originating from people who are blind that are each paired with 5 captions" 
_URL = ""
_HOMEPAGE = ""
_LICENSE = ""

DATA_DIR = {"train": "vizwiz"}

class vizwiz(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 256
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="image_captioning", description="Train Set.")]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"caption": datasets.Value("string"), "image": datasets.Image()}
            ),
            supervised_keys=("caption", "image"),
            homepage=_HOMEPAGE,
        )
    
    
    def _split_generators(
            self, dl_manager: datasets.utils.download_manager.DownloadManager
        ):
            data_dir = '/s/mlsc/enandhag/V+L/BLIP/data/vizwiz/val'
            caption_file="/s/mlsc/enandhag/V+L/BLIP/data/vizwiz/val_format.json"
            train_splits = [
                datasets.SplitGenerator(
                    name="train", gen_kwargs={"data_dir": data_dir, "name": "train","caption_file":caption_file}
                )
            ]

            return train_splits

    def _generate_examples(self, data_dir: str, name: str,caption_file:str):
        """Generate examples from a Crema unzipped directory."""
        key = 0
        examples = list()

        if not os.path.exists(data_dir):
            raise FileNotFoundError
        else:
            
            with open(caption_file) as data:
                meta_file=json.load(data)

                
            for uid in meta_file['captions'].keys():
                res = dict()

                image_name=meta_file['images'][uid]
                caption=meta_file['captions'][uid]

                img_path=f"{data_dir}/{image_name}"
                res["image"] = Image.open(img_path)
                res["caption"] = caption
                examples.append(res)
    
        for example in examples:
            yield key, {**example}
            key += 1
        examples = []
        