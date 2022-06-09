import torch
import torch.nn as nn
import pytorch_lightning as pl
from . import vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # if config["loss_names"]["mlm"] > 0:
        self.mlm_score = heads.MLMHead(bert_config)
        self.mlm_score.apply(objectives.init_weights)

        # if config["loss_names"]["itm"] > 0:
        self.itm_score = heads.ITMHead(config["hidden_size"])
        self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)
        print("initialization middle")

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else "" # ""
        text_ids = batch[f"text_ids{do_mlm}"] # text_ids: torch.Size([64, 40])
        text_labels = batch[f"text_labels{do_mlm}"] # text_labels: torch.Size([64, 40])
        text_masks = batch[f"text_masks"] # text_masks: torch.Size([64, 40]) , text_embeds: torch.Size([64, 40, 768])
        text_embeds = self.text_embeddings(text_ids) # 4. Text embedding is extracted 

        
        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0] # # 5. Image embedding is extracted 
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

    
        text_embeds, image_embeds = ( # 6. Both embeddings are summed with token type embeddings
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1) # 7. Concatenate the text embeddings and image embeddings
        co_masks = torch.cat([text_masks, image_masks], dim=1) # 

        x = co_embeds # x -> torch.Size([64, 257, 768]), text_embeds -> torch.Size([64, 40, 768]), image_embeds -> torch.Size([64, 217, 768]), co_masks -> torch.Size([64, 257])
        
        for i, blk in enumerate(self.transformer.blocks): # 8. ViT encoder blocks iteratively processes the input embedding
            x, _attn = blk(x, mask=co_masks)
            print(f"x:{x.size()},_attn:{_attn.size()}") # x:torch.Size([64, 257, 768]),_attn:torch.Size([64, 12, 257, 257]) -> tensor size is maintained 
            

        x = self.transformer.norm(x) # 9. transformer output ,  x : torch.Size([64, 257, 768]) 
        text_feats, image_feats = ( # 10. divides output into two subsets
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x) # 11. apply linear projection & hyperbolic tangent upon the first index of transformer output
        # import pdb; pdb.set_trace()
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }
        print("training_infer_end")
        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks: # 12.Masked language modeling (during pretraing process) : predict ground truth labels of masked text tokens
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks: # 14. Image Text Matching with Word Patch Alignment 
            ret.update(objectives.compute_itm_wpa(self, batch))

        ################################## 19. various downstream tasks #########################
        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))
            #ret.update(objectives.compute_irtr(self, batch))s
            print("training_forward_end")
        ##############################################################################################
        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        print("training_training_step_end")
        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
        print("training_training_epoch_end_end")

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        print("training_validation_step_end")

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
        print("training_validation_epoch_end_end")

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))
        print("training_test_step_end")
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)
        print("training_test_epoch_end_end")

    def configure_optimizers(self):
        print("training_configure_optimizer_end")
        return vilt_utils.set_schedule(self)
