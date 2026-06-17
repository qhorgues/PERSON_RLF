import warnings
from random import Random
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from lightning.pytorch.utilities import CombinedLoader
from loguru import logger
from torch.utils.data import DataLoader

from data.augmentation.transform import build_image_aug_pool, build_text_aug_pool
from data.bases import (
    ImageDataset,
    ImageTextDataset,
    ImageTextMLMDataset,
    TextDataset,
)
from data.cuhk_10_percent_vn3k_mix import TenPercentCUHK_VN3KMIX
from data.cuhkpedes import CUHKPEDES
from data.icfgpedes import ICFGPEDES
from data.rstpreid import RSTPReid
from data.sampler import RandomIdentitySampler
from data.vn3k_en import VN3K_EN

# from data.sampler_ddp import RandomIdentitySampler_DDP
from data.vn3k_mixed import VN3K_MIXED
from data.vn3k_vi import VN3K_VI

# from utils.comm import get_world_size
from utils.tokenizer_utils import get_tokenizer

# from torch.utils.data.distributed import DistributedSampler
# Filter UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


class TBPSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        client_id: int = None,
        num_clients: int = 1,
        partition_samples: Optional[List[Tuple]] = None,
    ):
        self.client_id = client_id
        self.num_clients = num_clients
        self.partition_samples = partition_samples
        super().__init__()
        __factory = {
            "CUHK-PEDES": CUHKPEDES,
            "ICFG-PEDES": ICFGPEDES,
            "RSTPReid": RSTPReid,
            "VN3K_EN": VN3K_EN,
            "VN3K_VI": VN3K_VI,
            "VN3K_MIXED": VN3K_MIXED,
            "TEN_PERCENT_CUHK_VN3K_MIX": TenPercentCUHK_VN3KMIX,
        }
        self.config = config
        print(config.dataset.dataset_name)
        self.dataset = __factory[config.dataset.dataset_name](
            root=config.dataset_root_dir, seed=config.seed
        )
        self.tokenizer = get_tokenizer(config.tokenizer)
        self.num_classes = len(self.dataset.train_id_container)
        # Set up transforms
        self._prepare_augmentation_pool()

    def _prepare_augmentation_pool(self):
        self.image_aug_pool = build_image_aug_pool(self.config.aug.img.augment_cfg)
        self.text_aug_pool = build_text_aug_pool(self.config.aug.text.augment_cfg)
        self.image_random_k = self.config.aug.image_random_k
        self.text_random_k = self.config.aug.text_random_k
        if self.config.loss.SS:
            self.ss_aug = True
        else:
            self.ss_aug = None
        self.mean = self.config.aug.img.mean
        self.std = self.config.aug.img.std

    def setup(self, stage=None):
        if self.partition_samples is not None:
            self.dataset.train = self.partition_samples
            logger.info(
                f"Using federated partition for client {self.client_id}: "
                f"{len(self.dataset.train)} samples."
            )

        if self.config.dataset.proportion:
            if self.config.dataset.dataset_name == "TEN_PERCENT_CUHK_VN3K_MIX":
                raise NotImplementedError(
                    "This mixed dataset does not support subset sampling"
                )
            # TODO: The subset sample has to contain a specific number of PID
            # Shuffle dataset once for reproducible folds
            shuffled_train_data = self.dataset.train[:]
            Random(self.config.seed).shuffle(shuffled_train_data)

            proportion = self.config.dataset.proportion
            num_folds = int(1 / proportion)
            fold_size = int(len(shuffled_train_data) * proportion)

            # Get current fold index from config.
            # You need to add `fold_id` to your dataset config and vary it for each run.
            fold_id = self.config.dataset.get("fold_id", 0)

            if not (0 <= fold_id < num_folds):
                raise ValueError(
                    f"fold_id {fold_id} is out of bounds for {num_folds} folds. "
                    f"Valid fold_id is from 0 to {num_folds - 1}."
                )

            # Calculate start and end index for the fold
            start_idx = fold_id * fold_size
            end_idx = start_idx + fold_size

            # Select the fold. Slicing handles the case where the last fold might be smaller.
            self.dataset.train = shuffled_train_data[start_idx:end_idx]

            logger.info(
                f"Using fold {fold_id}/{num_folds - 1} of the training set, with {len(self.dataset.train)} samples."
            )

        if stage == "fit" or stage is None:
            # STN (STNReID): emit a simulated partial view per sample when enabled.
            stn_cfg = self.config.get("stn", None)
            partial_image = bool(stn_cfg.enabled) if stn_cfg else False
            partial_min = stn_cfg.partial_min if stn_cfg else 0.2
            partial_max = stn_cfg.partial_max if stn_cfg else 0.6

            if self.config.loss.MLM:
                self.train_set = ImageTextMLMDataset(
                    dataset=self.dataset.train,
                    tokenizer=self.tokenizer,
                    ss_aug=self.ss_aug,
                    image_augmentation_pool=self.image_aug_pool,
                    text_augmentation_pool=self.text_aug_pool,
                    image_random_k=self.image_random_k,
                    text_random_k=self.text_random_k,
                    truncate=True,
                    image_size=self.config.aug.img.size,
                    is_train=True,
                    mean=self.mean,
                    std=self.std,
                    partial_image=partial_image,
                    partial_min=partial_min,
                    partial_max=partial_max,
                )
            else:
                self.train_set = ImageTextDataset(
                    dataset=self.dataset.train,
                    tokenizer=self.tokenizer,
                    ss_aug=self.ss_aug,
                    image_augmentation_pool=self.image_aug_pool,
                    text_augmentation_pool=self.text_aug_pool,
                    image_random_k=self.image_random_k,
                    text_random_k=self.text_random_k,
                    truncate=True,
                    image_size=self.config.aug.img.size,
                    is_train=True,
                    mean=self.mean,
                    std=self.std,
                    partial_image=partial_image,
                    partial_min=partial_min,
                    partial_max=partial_max,
                )

            logger.info("Validation set is available")

            self.val_img_set = ImageDataset(
                dataset=self.dataset.val,
                is_train=False,
                image_size=self.config.aug.img.size,
                mean=self.mean,
                std=self.std,
            )
            self.val_txt_set = TextDataset(
                dataset=self.dataset.val,
                tokenizer=self.tokenizer,
                is_train=False,
            )

        if stage == "test" or stage is None:
            self.test_img_set = ImageDataset(
                dataset=self.dataset.test,
                image_size=self.config.aug.img.size,
                is_train=False,
                mean=self.mean,
                std=self.std,
            )
            self.test_txt_set = TextDataset(
                dataset=self.dataset.test,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self):
        if self.config.dataset.sampler == "identity":
            if self.config.distributed:
                raise NotImplementedError(
                    "Distributed sampler is not implemented yet, please use random sampler"
                )
                logger.info("using ddp random identity sampler")
                logger.info("DISTRIBUTED TRAIN START")
                # mini_batch_size = self.config.dataset.batch_size // get_world_size()
                # data_sampler = RandomIdentitySampler_DDP(
                #     self.dataset.train,
                #     self.config.dataset.batch_size,
                #     self.config.dataset.num_instance,
                # )
                # batch_sampler = torch.utils.data.sampler.BatchSampler(
                #     data_sampler, mini_batch_size, True
                # )
                # return DataLoader(
                #     self.train_set,
                #     batch_sampler=batch_sampler,
                #     num_workers=self.config.dataset.num_workers,
                #     # collate_fn=self.collate_fn,
                #     drop_last=False,
                # )
            else:
                logger.info(
                    f"using random identity sampler: batch_size: {self.config.dataset.batch_size}, id: {self.config.dataset.batch_size // self.config.dataset.num_instance}, instance: {self.config.dataset.num_instance}"
                )
                return DataLoader(
                    self.train_set,
                    batch_size=self.config.dataset.batch_size,
                    sampler=RandomIdentitySampler(
                        self.dataset.train,
                        self.config.dataset.batch_size,
                        self.config.dataset.num_instance,
                    ),
                    num_workers=self.config.dataset.num_workers,
                    # collate_fn=self.collate_fn,
                    drop_last=False,
                )
        elif self.config.dataset.sampler == "random":
            logger.info("using random sampler")
            return DataLoader(
                self.train_set,
                batch_size=self.config.dataset.batch_size,
                shuffle=True,
                num_workers=self.config.dataset.num_workers,
                # collate_fn=self.collate_fn,
                drop_last=True,
            )
        else:
            logger.error(
                "unsupported sampler! expected softmax or triplet but got {}".format(
                    self.config.dataset.sampler
                )
            )
            raise ValueError("Unsupported sampler type")

    def val_dataloader(self):
        val_img_loader = DataLoader(
            self.test_img_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        val_txt_loader = DataLoader(
            self.test_txt_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        combined_val = {
            "img": val_img_loader,
            "txt": val_txt_loader,
        }
        combined_loader = CombinedLoader(combined_val, mode="max_size")
        return combined_loader

    def test_dataloader(self):
        test_img_loader = DataLoader(
            self.test_img_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        test_txt_loader = DataLoader(
            self.test_txt_set,
            batch_size=self.config.dataset.test_batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            # collate_fn=self.collate_fn,
        )
        combined_test = {
            "img": test_img_loader,
            "txt": test_txt_loader,
        }
        combined_loader = CombinedLoader(combined_test, mode="sequential")
        return combined_loader
