from collections import OrderedDict

import torch
import torch.nn as nn
from utils.logger import log as logger

from model import objectives
from model.layers import LayerNorm, QuickGELU, Transformer
from model.stn import STNReIDModule

TASK_LIST = ["ITC", "SDM", "CMPM", "ID", "MLM", "SS", "MVS", "RITC", "CITC", "NITC"]


class TBPS(nn.Module):
    def __init__(self, config, backbone, vocab_size, pad_token_id, num_classes=11003):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        # self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.backbone = backbone
        self.vision_model = backbone.vision_model
        self.text_model = backbone.text_model
        self.embed_dim = config.backbone.embedding_dim

        if config.backbone.freeze.vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            logger.info("Freezing vision backbone")
        if config.backbone.freeze.text:
            for param in self.text_model.parameters():
                param.requires_grad = False
            logger.info("Freezing text backbone")

        self.use_sigmoid = config.backbone.use_sigmoid
        if not hasattr(self.backbone, "logit_bias"):
            self.backbone.logit_bias = 0

        task_lists = []
        for task in TASK_LIST:
            try:
                if config.loss.get(task, None):
                    task_lists.append(task)
            except AttributeError:
                pass
        logger.info(f"Tasks: {task_lists}")

        self.contain_visual_projection, self.contain_text_projection = (
            self.check_contain_projection()
        )
        if config.loss.get("SS", None):
            self.simclr_mlp = self._build_mlp(
                self.embed_dim, self.embed_dim, self.embed_dim
            )

        if config.loss.get("ID", None):
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if config.loss.get("MLM", None):
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim, self.embed_dim // 64, batch_first=True
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim,
                layers=config.loss.get("cmt_depth", 4),
                heads=self.embed_dim // 64,
            )
            scale = self.cross_modal_transformer.width**-0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict(
                    [
                        ("dense", nn.Linear(self.embed_dim, self.embed_dim)),
                        ("gelu", QuickGELU()),
                        ("ln", LayerNorm(self.embed_dim)),
                        (
                            "fc",
                            nn.Linear(self.embed_dim, self.vocab_size),
                        ),
                    ]
                )
            )
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        # STN module (STNReID, Phase 2) — gated by the top-level `stn.enabled` config.
        # The `stn*` name makes its params part of the shared "heads" subspace (FedSH).
        stn_cfg = config.get("stn", None)
        if stn_cfg is not None and stn_cfg.enabled:
            self.stn_module = STNReIDModule(self.encode_image, config)

    def _build_mlp(self, in_dim=512, mlp_dim=128, out_dim=512):
        return nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim),
        )

    def check_contain_projection(self):
        """
        Check if the backbone contains visual and text projection.
        """
        contain_visual_projection = hasattr(self.backbone, "visual_projection")
        logger.info(
            f"Contain visual projection: {contain_visual_projection}"
        ) if contain_visual_projection else None
        contain_text_projection = hasattr(self.backbone, "text_projection")
        logger.info(
            f"Contain text projection: {contain_text_projection}"
        ) if contain_text_projection else None
        return contain_visual_projection, contain_text_projection

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q), self.ln_pre_i(k), self.ln_pre_i(v), need_weights=False
        )[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image, return_last_hidden=False):
        """
        Encodes an image into a tensor.

        Args:
            image (PIL.Image): Image to be encoded.
            return_last_hidden (bool): Whether to return the last hidden state.

        Returns:
            torch.Tensor: The encoded image, use pooler_output as the image embedding."""
        x = self.vision_model(image)
        pooler_output = x.pooler_output
        last_hidden = x.last_hidden_state

        if not hasattr(self.backbone, "visual_projection"):
            # If no projection layer exists, normalize and return
            # normalized_pooler = pooler_output / pooler_output.norm(dim=1, keepdim=True)
            if return_last_hidden:
                return pooler_output, last_hidden
            return pooler_output

        # If projection layer exists, project both outputs
        pooler_output = self.backbone.visual_projection(pooler_output)
        # pooler_output = pooler_output / pooler_output.norm(dim=1, keepdim=True)

        if return_last_hidden:
            last_hidden = self.backbone.visual_projection(last_hidden)
            return pooler_output, last_hidden

        return pooler_output

    def encode_text(self, text, return_last_hidden=False):
        """
        Encodes text into a tensor.

        Args:
            text (dict): Text to be encoded, containing the keys "input_ids" and "attention_mask".
            return_last_hidden (bool): Whether to return the last hidden state.
        Returns:
            torch.Tensor: The encoded text, use pooler_output as the sentence embedding."""
        # text will be a dict that has forms {"input_ids": ..., "attention_mask": ...}
        x = self.text_model(**text)
        pooler_output = x.pooler_output
        last_hidden = x.last_hidden_state

        if not hasattr(self.backbone, "text_projection"):
            # If no projection layer exists, normalize and return
            # normalized_pooler = pooler_output / pooler_output.norm(dim=1, keepdim=True)
            if return_last_hidden:
                return pooler_output, last_hidden
            return pooler_output

        # If projection layer exists, project both outputs
        pooler_output = self.backbone.text_projection(pooler_output)
        # pooler_output = pooler_output / pooler_output.norm(dim=1, keepdim=True)

        if return_last_hidden:
            last_hidden = self.backbone.text_projection(last_hidden)
            return pooler_output, last_hidden

        return pooler_output

    def prepare_sim_targets(self, pids, use_sigmoid=True):
        """
        Prepare similarity targets for constrative learning.

        Args:
            pids (torch.Tensor): Tensor containing the person IDs.
        Returns:
            torch.Tensor: Tensor containing the similarity targets.
        """
        sim_targets = torch.eq(pids.view(-1, 1), pids.view(1, -1)).float()
        if use_sigmoid:
            sim_targets = (
                -torch.ones_like(sim_targets) + 2 * sim_targets
            )  # -1 if different, 1 if same
            return sim_targets

        return sim_targets / sim_targets.sum(
            dim=1, keepdim=True
        )  # normalize the true matching distribution

    def forward(self, batch, alpha, weights=None):
        """
        Forward pass of the model.

        Args:
            batch (dict): A dictionary containing the input data.
            Containing the following: ['pids', 'image_ids', 'images', 'aug_images', 'caption_input_ids', 'caption_attention_mask', 'ss_images1', 'ss_images2']
        """
        ret = dict()

        caption_input = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }

        images = batch["images"]

        logit_scale = self.backbone.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        if self.config.loss.get("MLM", None) or self.config.loss.get("Ring", None):
            image_pooler_output, image_last_hidden = self.encode_image(images, True)
            if self.config.loss.get("Ring", None):
                caption_pooler_output, caption_last_hidden = self.encode_text(
                    caption_input, True
                )
            else:
                caption_pooler_output = self.encode_text(caption_input)
        else:
            image_pooler_output = self.encode_image(images)
            caption_pooler_output = self.encode_text(caption_input)

        ret.update({"temperature": self.backbone.logit_scale})
        ret.update({"bias": self.backbone.logit_bias})

        # Improve data efficiency with SimCLR
        if self.config.loss.get("SS", None):
            ss_images1_embed = self.simclr_mlp(self.encode_image(batch["ss_images1"]))
            ss_images2_embed = self.simclr_mlp(self.encode_image(batch["ss_images2"]))
            ss_loss = (
                objectives.compute_simclr(
                    ss_images1_embed,
                    ss_images2_embed,
                    self.config.loss.simclr_temperature,
                )
                * self.config.loss.ss_loss_weight
            )
            ret.update({"ss_loss": ss_loss})

        # Compute NITC loss
        if self.config.loss.get("NITC", None):
            sim_targets = self.prepare_sim_targets(batch["pids"], self.use_sigmoid)
            image_pooler_output_stopped = (
                image_pooler_output.clone().detach() if alpha != 0 else None
            )
            caption_pooler_output_stopped = (
                caption_pooler_output.clone().detach() if alpha != 0 else None
            )
            nitc_loss = objectives.compute_constrative(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                image_features_stopped=image_pooler_output_stopped,
                text_features_stopped=caption_pooler_output_stopped,
                sim_targets=sim_targets,
                alpha=alpha,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                use_sigmoid=self.use_sigmoid,
                weights=weights,
            )
            if self.config.loss.get("MVS", None):
                aug_images = batch["aug_images"]
                aug_images_features = self.encode_image(aug_images)
                aug_images_features_stopped = (
                    aug_images_features.clone().detach() if alpha != 0 else None
                )
                # Compute NITC loss with augmented images
                augmented_nitc_loss = objectives.compute_constrative(
                    image_features=image_pooler_output,
                    text_features=caption_pooler_output,
                    image_features_stopped=aug_images_features_stopped,
                    text_features_stopped=caption_pooler_output_stopped,
                    sim_targets=sim_targets,
                    alpha=alpha,
                    logit_scale=logit_scale,
                    use_sigmoid=self.use_sigmoid,
                    logit_bias=self.backbone.logit_bias,
                    weights=weights,
                )
                nitc_loss = (nitc_loss + augmented_nitc_loss) / 2

            ret.update({"nitc_loss": nitc_loss * self.config.loss.nitc_loss_weight})

        # Compute CITC loss
        if self.config.loss.get("CITC", None):
            loss = objectives.compute_citc(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                inmodal_weight=self.config.loss.citc_inmodal_weight,
                intermodal_weight=self.config.loss.citc_intermodal_weight,
            )
            ret.update({"citc_loss": loss * self.config.loss.citc_loss_weight})

        # Compute RITC loss
        if self.config.loss.get("RITC", None):
            sim_targets = self.prepare_sim_targets(
                batch["pids"], use_sigmoid=self.use_sigmoid
            )
            loss = objectives.compute_ritc(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                sim_targets=sim_targets,
                use_sigmoid=self.use_sigmoid,
                eps=self.config.loss.ritc_eps,
            )
            ret.update({"ritc_loss": loss * self.config.loss.ritc_loss_weight})

        # Compute ITC loss
        if self.config.loss.get("ITC", None):
            ret.update(
                {
                    "itc_loss": (
                        objectives.compute_itc(
                            image_pooler_output, caption_pooler_output, logit_scale
                        )
                        * self.config.loss.itc_loss_weight
                    )
                }
            )

        # Compute SDM loss
        if self.config.loss.get("SDM", None):
            # if self.config.backbone.use_sigmoid:
            #     raise NotImplementedError("SDM loss does not support sigmoid.")
            ret.update(
                {
                    "sdm_loss": (
                        objectives.compute_sdm(
                            image_pooler_output,
                            caption_pooler_output,
                            batch["pids"],
                            logit_scale,
                            self.backbone.logit_bias,
                            weights=weights,
                        )
                        * self.config.loss.sdm_loss_weight
                    )
                }
            )

        # Compute CMPM loss
        if self.config.loss.get("CMPM", None):
            ret.update(
                {
                    "cmpm_loss": (
                        objectives.compute_cmpm(
                            image_pooler_output, caption_pooler_output, batch["pids"]
                        )
                        * self.config.loss.cmpm_loss_weight
                    )
                }
            )

        # Compute Ring (Align and Unify) loss
        if self.config.loss.get("Ring", None):
            lam = 0  # A weight factor for align loss
            image_logits = self.classifier(image_pooler_output).float()
            text_logits = self.classifier(caption_pooler_output).float()

        # Compute ID loss
        if self.config.loss.get("ID", None):
            image_logits = self.classifier(image_pooler_output).float()
            text_logits = self.classifier(caption_pooler_output).float()
            ret.update(
                {
                    "id_loss": (
                        objectives.compute_id(
                            image_logits, text_logits, batch["pids"], weights=weights
                        )
                        * self.config.loss.id_loss_weight
                    )
                }
            )

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch["pids"]).float().mean()
            text_precision = (text_pred == batch["pids"]).float().mean()
            ret.update({"img_acc": image_precision})
            ret.update({"txt_acc": text_precision})

        # Compute MLM loss
        if self.config.loss.get("MLM", None):
            mlm_input = {
                "input_ids": batch["mlm_input_ids"],
                "attention_mask": batch["mlm_attention_mask"],
            }
            mlm_labels = batch["mlm_labels"]

            _, mlm_last_hidden = self.encode_text(mlm_input, return_last_hidden=True)

            x = self.cross_former(mlm_last_hidden, image_last_hidden, image_last_hidden)

            x = self.mlm_head(x)

            scores = x.reshape(-1, self.vocab_size)
            mlm_labels = mlm_labels.reshape(-1)  # [batch_size * text_len]
            ret.update(
                {
                    "mlm_loss": objectives.compute_mlm(
                        scores, mlm_labels, self.pad_token_id
                    )
                    * self.config.loss.mlm_loss_weight
                }
            )

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels != self.pad_token_id)

            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({"mlm_acc": acc})

        # Compute STN loss (STNReID eq. 5): align partial and affined ReID features.
        if getattr(self, "stn_module", None) is not None and "partial_images" in batch:
            partial = batch["partial_images"]
            affined = self.stn_module.transform(images, partial)
            f_p = self.encode_image(partial)
            f_a = self.encode_image(affined)
            stn_loss = self.stn_module.compute_stn_loss(f_p, f_a)
            ret.update({"stn_loss": stn_loss * self.config.stn.stn_loss_weight})

        return ret

    def forward2(self, batch):
        caption_input = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }

        images = batch["images"]

        with torch.no_grad():
            image_feats = self.encode_image(images)
            text_feats = self.encode_text(caption_input)

        return image_feats, text_feats
