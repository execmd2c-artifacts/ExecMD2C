# Consolidated benchmark answer key for PASA.
# Source model components copied from models.py.

import torch
import torch.nn.functional as F

from torch import nn
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel
from transformers.modeling_outputs import NextSentencePredictorOutput


def entropy(log_qy, batch_size=None, unit_average=False):
    """
    -qy log(qy)
    """
    if log_qy.dim() > 2:
        log_qy = log_qy.squeeze()
    qy = torch.exp(log_qy)
    h_q = torch.sum(-1 * log_qy * qy, dim=1)
    #if unit_average:
    return torch.mean(h_q)
    #else:
        #return torch.sum(h_q) / batch_size

from typing import Optional, List

class LatentModelOutput(NextSentencePredictorOutput):
    mi: Optional[torch.tensor] = None
    actions: Optional[List[torch.tensor]] = None
    states: Optional[torch.tensor] = None
    z: Optional[torch.tensor] = None
    cls: Optional[torch.tensor] = None


class IntentModelForPretraining(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        self.num_codes = 5 # num codes per latent
        self.intent_type = "regex"

        self.cls = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.Linear(config.dim, self.num_codes)
        )
        self.proj = nn.Linear(config.dim + self.num_codes, config.dim)

        self.post_init()

    @staticmethod
    def unroll(ids, sizes):
        new_ids = []
        for i, size in enumerate(sizes):
            new_ids.append(ids[i, :size, :])
        
        return torch.cat(new_ids)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        act_input_ids, 
        act_attention_mask,  
        cand_input_ids=None, 
        cand_attention_mask=None, 
        labels=None,
        intents=None,
        persona_ids=None
    ) -> LatentModelOutput:

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Unroll actions and next observations
        if cand_input_ids is not None:
            act_input_ids = torch.cat([act_input_ids.unsqueeze(1), cand_input_ids], dim=1)
            act_input_ids = act_input_ids.reshape([-1, act_input_ids.shape[-1]])
            act_attention_mask = torch.cat([act_attention_mask.unsqueeze(1), cand_attention_mask], dim=1)
            act_attention_mask = act_attention_mask.reshape([-1, act_attention_mask.shape[-1]])

        act_embed = self.distilbert(act_input_ids, attention_mask=act_attention_mask).last_hidden_state[:, 0, :]

        obs_embed = self.distilbert(
            input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        kld_loss = 0.0

        if cand_input_ids is None:
            cls = F.softmax(self.cls(torch.cat([obs_embed, act_embed], dim=-1)), dim=-1)
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1)  # actions x codes

            obs_embed = self.proj(torch.cat([obs_embed, one_hot], dim=-1))
            logits = torch.matmul(obs_embed, act_embed.T)
        else:
            act_embed = act_embed.reshape([bsz, -1, act_embed.shape[-1]])
            cls = F.softmax(self.cls(torch.cat([obs_embed, act_embed[:, 0, :]], dim=-1)), dim=-1)
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1, hard=True)
 
            obs_embed = self.proj(torch.cat([obs_embed, one_hot], dim=-1)).unsqueeze(1)
            logits = (obs_embed * act_embed).sum(-1)  # bsz x actions

        if cand_input_ids is not None:# not self.training:
            labels = torch.zeros(bsz).long().to(device)
        else:
            labels = torch.arange(bsz).to(device)

        loss = nn.CrossEntropyLoss()(logits, labels) + kld_loss

        b_pr = cls.mean(0, keepdim=True)
        mi = entropy(torch.log(b_pr)) - entropy(torch.log(cls))

        if self.intent_type == "regex":
            z_loss = -1 * (F.one_hot(intents, self.num_codes).to(device) * torch.log(cls)).sum(-1).mean(0)
            loss += z_loss
        elif self.intent_type == "persona":
            z_loss = -1 * (F.one_hot(persona_ids, 375).to(device) * torch.log(cls)).sum(-1).mean(0)
        
        elif self.intent_type == "latent":
            prior = torch.ones([1, self.num_codes]).float().div(self.num_codes).to(device)
            kld_loss = nn.KLDivLoss(reduction="batchmean")(torch.log(b_pr), prior)
            loss += kld_loss

        outputs = LatentModelOutput(loss=loss, logits=logits)
        outputs.mi = mi

        return outputs


class IntentModel(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        self.num_codes = 5 # num codes per latent

        self.intent_type = "regex"

        self.cls = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.Linear(config.dim, self.num_codes)
        )
        self.proj = nn.Linear(config.dim + self.num_codes, config.dim)
        

        self.post_init()

    @staticmethod
    def unroll(ids, sizes):
        new_ids = []
        for i, size in enumerate(sizes):
            new_ids.append(ids[i, :size, :])
        
        return torch.cat(new_ids)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        act_input_ids, 
        act_attention_mask, 
        act_sizes, 
        intents=None, 
        labels=None,
        **kwargs
    ):

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Encode observations
        obs_embed = self.distilbert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        # Unroll actions
        flat_act_input_ids, flat_act_attention_mask = [], []
        for i, size in enumerate(act_sizes):
            flat_act_input_ids.append(act_input_ids[i, :size, :])
            flat_act_attention_mask.append(act_attention_mask[i, :size, :])

        flat_act_input_ids = torch.cat(flat_act_input_ids)
        flat_act_attention_mask = torch.cat(flat_act_attention_mask)
        
        act_embed = self.distilbert(flat_act_input_ids, attention_mask=flat_act_attention_mask).last_hidden_state[:, 0, :]

        # Re-roll actions
        actions = []
        scores = []
        states = []
        z = []
        all_cls = []
        cum_idx = 0
        ce_loss = 0
        kld_loss = 0
        intent_loss = 0
        for i, size in enumerate(act_sizes):
            act_embeds = act_embed[cum_idx:cum_idx + size, :]
            actions.append(act_embeds)
            cls = F.softmax(self.cls(torch.cat([obs_embed[i], act_embeds[0, :]], dim=-1).unsqueeze(0)), dim=-1)
            all_cls.append(cls)
            #if labels is not None:
                #one_hot = F.one_hot(labels[i], cls.shape[-1]).to(device).unsqueeze(0)
            #elif not self.intent_type == "prior":
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1)  # 1 x codes
            #else:
                #one_hot = F.gumbel_softmax(torch.log(p_z[0].unsqueeze(0)), dim=-1, tau=0.1, hard=True)
            z.append(one_hot.argmax())
            o = self.proj(torch.cat([obs_embed[i].unsqueeze(0), one_hot], dim=-1)) # 1 x dim
            states.append(o)
            logits = (o * act_embeds).sum(-1)
            scores.append(logits)
            ce_loss += -1 * torch.log_softmax(logits, dim=-1)[0]

            if self.intent_type == "regex" and self.training:
                intent_loss += -1 * torch.log(cls)[0, intents[i]]

            cum_idx += size

        if self.intent_type == "latent" and self.training:
            cls = torch.cat(all_cls, dim=0)
            b_pr = cls.mean(0, keepdim=True)
            prior = torch.ones([1, self.num_codes]).float().div(self.num_codes).to(device)
            kld_loss += nn.KLDivLoss(reduction="batchmean")(torch.log(b_pr), prior)

        ce_loss /= bsz
        kld_loss /= bsz
        intent_loss /= bsz
        loss = ce_loss + intent_loss + kld_loss

        states = torch.cat(states, dim=0)

        outputs = LatentModelOutput(loss=loss, logits=scores)
        outputs.actions = actions
        outputs.states = states
        outputs.z = torch.stack(z)
        outputs.cls = torch.cat(all_cls)

        return outputs



class LatentModelForPretraining(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        self.num_codes = 8 # num codes per latent
        self.intent_type = "latent"

        self.cls = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.Linear(config.dim, self.num_codes)
        )
        self.proj = nn.Linear(config.dim + self.num_codes, config.dim)

        self.post_init()

    @staticmethod
    def unroll(ids, sizes):
        new_ids = []
        for i, size in enumerate(sizes):
            new_ids.append(ids[i, :size, :])
        
        return torch.cat(new_ids)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        act_input_ids, 
        act_attention_mask,  
        cand_input_ids=None, 
        cand_attention_mask=None, 
        labels=None,
        intents=None,
        persona_ids=None
    ) -> LatentModelOutput:

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Unroll actions and next observations
        if cand_input_ids is not None:
            act_input_ids = torch.cat([act_input_ids.unsqueeze(1), cand_input_ids], dim=1)
            act_input_ids = act_input_ids.reshape([-1, act_input_ids.shape[-1]])
            act_attention_mask = torch.cat([act_attention_mask.unsqueeze(1), cand_attention_mask], dim=1)
            act_attention_mask = act_attention_mask.reshape([-1, act_attention_mask.shape[-1]])

        act_embed = self.distilbert(act_input_ids, attention_mask=act_attention_mask).last_hidden_state[:, 0, :]

        obs_embed = self.distilbert(
            input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        kld_loss = 0.0

        if cand_input_ids is None:
            cls = F.softmax(self.cls(torch.cat([obs_embed, act_embed], dim=-1)), dim=-1)
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1)  # actions x codes

            obs_embed = self.proj(torch.cat([obs_embed, one_hot], dim=-1))
            logits = torch.matmul(obs_embed, act_embed.T)
        else:
            act_embed = act_embed.reshape([bsz, -1, act_embed.shape[-1]])
            cls = F.softmax(self.cls(torch.cat([obs_embed, act_embed[:, 0, :]], dim=-1)), dim=-1)
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1, hard=True)
 
            obs_embed = self.proj(torch.cat([obs_embed, one_hot], dim=-1)).unsqueeze(1)
            logits = (obs_embed * act_embed).sum(-1)  # bsz x actions

        if cand_input_ids is not None:# not self.training:
            labels = torch.zeros(bsz).long().to(device)
        else:
            labels = torch.arange(bsz).to(device)

        loss = nn.CrossEntropyLoss()(logits, labels) + kld_loss

        b_pr = cls.mean(0, keepdim=True)
        mi = entropy(torch.log(b_pr)) - entropy(torch.log(cls))

        if self.intent_type == "regex":
            z_loss = -1 * (F.one_hot(intents, self.num_codes).to(device) * torch.log(cls)).sum(-1).mean(0)
            loss += z_loss
        elif self.intent_type == "persona":
            z_loss = -1 * (F.one_hot(persona_ids, 375).to(device) * torch.log(cls)).sum(-1).mean(0)
        
        elif self.intent_type == "latent":
            prior = torch.ones([1, self.num_codes]).float().div(self.num_codes).to(device)
            kld_loss = nn.KLDivLoss(reduction="batchmean")(torch.log(b_pr), prior)
            loss += kld_loss

        outputs = LatentModelOutput(loss=loss, logits=logits)
        outputs.mi = mi

        return outputs


class LatentModel(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        self.num_codes = 8 # num codes per latent

        self.intent_type = "latent"

        self.cls = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.Linear(config.dim, self.num_codes)
        )
        self.proj = nn.Linear(config.dim + self.num_codes, config.dim)
        

        self.post_init()

    @staticmethod
    def unroll(ids, sizes):
        new_ids = []
        for i, size in enumerate(sizes):
            new_ids.append(ids[i, :size, :])
        
        return torch.cat(new_ids)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        act_input_ids, 
        act_attention_mask, 
        act_sizes, 
        intents=None, 
        labels=None,
        **kwargs
    ):

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Encode observations
        obs_embed = self.distilbert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        # Unroll actions
        flat_act_input_ids, flat_act_attention_mask = [], []
        for i, size in enumerate(act_sizes):
            flat_act_input_ids.append(act_input_ids[i, :size, :])
            flat_act_attention_mask.append(act_attention_mask[i, :size, :])

        flat_act_input_ids = torch.cat(flat_act_input_ids)
        flat_act_attention_mask = torch.cat(flat_act_attention_mask)
        
        act_embed = self.distilbert(flat_act_input_ids, attention_mask=flat_act_attention_mask).last_hidden_state[:, 0, :]

        # Re-roll actions
        actions = []
        scores = []
        states = []
        z = []
        all_cls = []
        cum_idx = 0
        ce_loss = 0
        kld_loss = 0
        intent_loss = 0
        for i, size in enumerate(act_sizes):
            act_embeds = act_embed[cum_idx:cum_idx + size, :]
            actions.append(act_embeds)
            cls = F.softmax(self.cls(torch.cat([obs_embed[i], act_embeds[0, :]], dim=-1).unsqueeze(0)), dim=-1)
            all_cls.append(cls)
            #if labels is not None:
                #one_hot = F.one_hot(labels[i], cls.shape[-1]).to(device).unsqueeze(0)
            #elif not self.intent_type == "prior":
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1)  # 1 x codes
            #else:
                #one_hot = F.gumbel_softmax(torch.log(p_z[0].unsqueeze(0)), dim=-1, tau=0.1, hard=True)
            z.append(one_hot.argmax())
            o = self.proj(torch.cat([obs_embed[i].unsqueeze(0), one_hot], dim=-1)) # 1 x dim
            states.append(o)
            logits = (o * act_embeds).sum(-1)
            scores.append(logits)
            ce_loss += -1 * torch.log_softmax(logits, dim=-1)[0]

            if self.intent_type == "regex":
                intent_loss += -1 * torch.log(cls)[0, intents[i]]

            cum_idx += size

        if self.intent_type == "latent":
            cls = torch.cat(all_cls, dim=0)
            b_pr = cls.mean(0, keepdim=True)
            prior = torch.ones([1, self.num_codes]).float().div(self.num_codes).to(device)
            kld_loss += nn.KLDivLoss(reduction="batchmean")(torch.log(b_pr), prior)

        ce_loss /= bsz
        kld_loss /= bsz
        intent_loss /= bsz
        loss = ce_loss + intent_loss + kld_loss

        states = torch.cat(states, dim=0)

        outputs = LatentModelOutput(loss=loss, logits=scores)
        outputs.actions = actions
        outputs.states = states
        outputs.z = torch.stack(z)
        outputs.cls = torch.cat(all_cls)

        return outputs



class PersonaModelForPretraining(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        self.num_codes = 375 # num codes per latent
        self.intent_type = "persona"

        self.cls = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.Linear(config.dim, self.num_codes)
        )
        self.proj = nn.Linear(config.dim + self.num_codes, config.dim)

        self.post_init()

    @staticmethod
    def unroll(ids, sizes):
        new_ids = []
        for i, size in enumerate(sizes):
            new_ids.append(ids[i, :size, :])
        
        return torch.cat(new_ids)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        act_input_ids, 
        act_attention_mask,  
        cand_input_ids=None, 
        cand_attention_mask=None, 
        labels=None,
        intents=None,
        persona_ids=None
    ) -> LatentModelOutput:

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Unroll actions and next observations
        if cand_input_ids is not None:
            act_input_ids = torch.cat([act_input_ids.unsqueeze(1), cand_input_ids], dim=1)
            act_input_ids = act_input_ids.reshape([-1, act_input_ids.shape[-1]])
            act_attention_mask = torch.cat([act_attention_mask.unsqueeze(1), cand_attention_mask], dim=1)
            act_attention_mask = act_attention_mask.reshape([-1, act_attention_mask.shape[-1]])

        act_embed = self.distilbert(act_input_ids, attention_mask=act_attention_mask).last_hidden_state[:, 0, :]

        obs_embed = self.distilbert(
            input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        kld_loss = 0.0

        if cand_input_ids is None:
            cls = F.softmax(self.cls(torch.cat([obs_embed, act_embed], dim=-1)), dim=-1)
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1)  # actions x codes

            obs_embed = self.proj(torch.cat([obs_embed, one_hot], dim=-1))
            logits = torch.matmul(obs_embed, act_embed.T)
        else:
            act_embed = act_embed.reshape([bsz, -1, act_embed.shape[-1]])
            cls = F.softmax(self.cls(torch.cat([obs_embed, act_embed[:, 0, :]], dim=-1)), dim=-1)
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1, hard=True)
 
            obs_embed = self.proj(torch.cat([obs_embed, one_hot], dim=-1)).unsqueeze(1)
            logits = (obs_embed * act_embed).sum(-1)  # bsz x actions

        if cand_input_ids is not None:# not self.training:
            labels = torch.zeros(bsz).long().to(device)
        else:
            labels = torch.arange(bsz).to(device)

        loss = nn.CrossEntropyLoss()(logits, labels) + kld_loss

        b_pr = cls.mean(0, keepdim=True)
        mi = entropy(torch.log(b_pr)) - entropy(torch.log(cls))

        if self.intent_type == "regex":
            z_loss = -1 * (F.one_hot(intents, self.num_codes).to(device) * torch.log(cls)).sum(-1).mean(0)
            loss += z_loss
        elif self.intent_type == "persona":
            z_loss = -1 * (F.one_hot(persona_ids, 375).to(device) * torch.log(cls)).sum(-1).mean(0)
        
        elif self.intent_type == "latent":
            prior = torch.ones([1, self.num_codes]).float().div(self.num_codes).to(device)
            kld_loss = nn.KLDivLoss(reduction="batchmean")(torch.log(b_pr), prior)
            loss += kld_loss

        outputs = LatentModelOutput(loss=loss, logits=logits)
        outputs.mi = mi

        return outputs


class PersonaModel(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        self.num_codes = 375 # num codes per latent

        self.intent_type = "persona"

        self.cls = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.Linear(config.dim, self.num_codes)
        )
        self.proj = nn.Linear(config.dim + self.num_codes, config.dim)
        

        self.post_init()

    @staticmethod
    def unroll(ids, sizes):
        new_ids = []
        for i, size in enumerate(sizes):
            new_ids.append(ids[i, :size, :])
        
        return torch.cat(new_ids)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        act_input_ids, 
        act_attention_mask, 
        act_sizes, 
        intents=None, 
        labels=None,
        **kwargs
    ):

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Encode observations
        obs_embed = self.distilbert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        # Unroll actions
        flat_act_input_ids, flat_act_attention_mask = [], []
        for i, size in enumerate(act_sizes):
            flat_act_input_ids.append(act_input_ids[i, :size, :])
            flat_act_attention_mask.append(act_attention_mask[i, :size, :])

        flat_act_input_ids = torch.cat(flat_act_input_ids)
        flat_act_attention_mask = torch.cat(flat_act_attention_mask)
        
        act_embed = self.distilbert(flat_act_input_ids, attention_mask=flat_act_attention_mask).last_hidden_state[:, 0, :]

        # Re-roll actions
        actions = []
        scores = []
        states = []
        z = []
        all_cls = []
        cum_idx = 0
        ce_loss = 0
        kld_loss = 0
        intent_loss = 0
        for i, size in enumerate(act_sizes):
            act_embeds = act_embed[cum_idx:cum_idx + size, :]
            actions.append(act_embeds)
            cls = F.softmax(self.cls(torch.cat([obs_embed[i], act_embeds[0, :]], dim=-1).unsqueeze(0)), dim=-1)
            all_cls.append(cls)
            #if labels is not None:
                #one_hot = F.one_hot(labels[i], cls.shape[-1]).to(device).unsqueeze(0)
            #elif not self.intent_type == "prior":
            one_hot = F.gumbel_softmax(torch.log(cls), dim=-1, tau=0.1)  # 1 x codes
            #else:
                #one_hot = F.gumbel_softmax(torch.log(p_z[0].unsqueeze(0)), dim=-1, tau=0.1, hard=True)
            z.append(one_hot.argmax())
            o = self.proj(torch.cat([obs_embed[i].unsqueeze(0), one_hot], dim=-1)) # 1 x dim
            states.append(o)
            logits = (o * act_embeds).sum(-1)
            scores.append(logits)
            ce_loss += -1 * torch.log_softmax(logits, dim=-1)[0]

            if self.intent_type == "regex":
                intent_loss += -1 * torch.log(cls)[0, intents[i]]

            cum_idx += size

        if self.intent_type == "latent":
            cls = torch.cat(all_cls, dim=0)
            b_pr = cls.mean(0, keepdim=True)
            prior = torch.ones([1, self.num_codes]).float().div(self.num_codes).to(device)
            kld_loss += nn.KLDivLoss(reduction="batchmean")(torch.log(b_pr), prior)

        ce_loss /= bsz
        kld_loss /= bsz
        intent_loss /= bsz
        loss = ce_loss + intent_loss + kld_loss

        states = torch.cat(states, dim=0)

        outputs = LatentModelOutput(loss=loss, logits=scores)
        outputs.actions = actions
        outputs.states = states
        outputs.z = torch.stack(z)
        outputs.cls = torch.cat(all_cls)

        return outputs


class BiEncoder(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        self.negatives = "valid"

    def forward(self, input_ids, attention_mask, act_input_ids, act_attention_mask, act_sizes, labels=None, **kwargs):
        bsz = input_ids.shape[0]
        device = input_ids.device

        # Encode observations
        obs_embed = self.distilbert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        # Unroll actions
        flat_act_input_ids, flat_act_attention_mask = [], []
        for i, size in enumerate(act_sizes):
            flat_act_input_ids.append(act_input_ids[i, :size, :])
            flat_act_attention_mask.append(act_attention_mask[i, :size, :])

        flat_act_input_ids = torch.cat(flat_act_input_ids)
        flat_act_attention_mask = torch.cat(flat_act_attention_mask)
        
        act_embed = self.distilbert(flat_act_input_ids, attention_mask=flat_act_attention_mask).last_hidden_state[:, 0, :]

        # Re-roll actions
        act_embeds = []
        cum_idx = 0
        for i, size in enumerate(act_sizes):
            act_embeds.append(act_embed[cum_idx:cum_idx + size, :])
            cum_idx += size
        
        # Scores
        loss = 0.0
        logits = []
        for i in range(bsz):
            if not self.training:
                negatives = act_embeds[i]
            elif self.negatives == "valid":
                negatives = act_embeds[i]
            elif self.negatives == "batch":
                negatives = torch.stack([act_embeds[j][0, :] for j in range(bsz)])
                labels = torch.arange(bsz, device=device, dtype=torch.long)
            elif self.negatives == "batch_valid":
                negatives = torch.cat([act_embeds[i], torch.stack([act_embeds[j][0, :] for j in range(bsz) if j != i])], dim=0)
            scores = (obs_embed[i, :].unsqueeze(0) * negatives).sum(-1)
            logits.append(scores)
            loss = -1 * torch.log_softmax(scores, dim=-1)[labels[i]]
            loss += loss

        outputs = LatentModelOutput(loss=loss, logits=logits)
        outputs.actions = act_embeds
        outputs.states  =obs_embed

        return outputs



class BiEncoderForPretraining(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        self.negatives = "valid"

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        act_input_ids, 
        act_attention_mask,
        cand_input_ids = None,
        cand_attention_mask = None, 
        labels=None,
        **kwargs
    ) -> NextSentencePredictorOutput:
        bsz = input_ids.shape[0]
        device = input_ids.device

        # Encode observations
        obs_embed = self.distilbert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        
        if not self.training:
            act_input_ids = torch.cat([act_input_ids.unsqueeze(1), cand_input_ids], dim=1)
            act_input_ids = act_input_ids.reshape([-1, act_input_ids.shape[-1]])
            act_attention_mask = torch.cat([act_attention_mask.unsqueeze(1), cand_attention_mask], dim=1)
            act_attention_mask = act_attention_mask.reshape([-1, act_attention_mask.shape[-1]])
        
        act_embed = self.distilbert(act_input_ids, attention_mask=act_attention_mask).last_hidden_state[:, 0, :]

        if not self.training:
            act_embed = act_embed.reshape([bsz, -1, act_embed.shape[-1]])
            logits = (obs_embed.unsqueeze(1) * act_embed).sum(-1)
            labels = torch.zeros(bsz).long().to(device)
        else:
            logits = torch.matmul(obs_embed, act_embed.T)
            labels = torch.arange(bsz).to(device)
        loss = nn.CrossEntropyLoss()(logits, labels)

        return NextSentencePredictorOutput(
            loss=loss,
            logits=logits
        )

class Distillation(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.student = BiEncoder.from_pretrained(args.student_path)

        if args.intent_type == "regex":
            self.teacher = IntentModel.from_pretrained(args.teacher_path)
        elif args.intent_type == "latent":
            self.teacher =  LatentModel.from_pretrained(args.teacher_path)
        elif args.intent_type == "persona":
            self.teacher =  PersonaModel.from_pretrained(args.teacher_path)

    def forward(self, input_ids, attention_mask, act_input_ids, act_attention_mask, act_sizes, **kwargs):

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Obtain targets
        with torch.no_grad():
            self.teacher.eval()
            t_outputs = self.teacher(input_ids, attention_mask, act_input_ids, act_attention_mask, act_sizes)
        
        outputs = self.student(
            input_ids, attention_mask, act_input_ids, act_attention_mask, act_sizes, labels=kwargs["labels"]
        )

        ce_loss = 0.0
        tau = 20.0
        for pred, target in zip(outputs.logits, t_outputs.logits):
            warm_target, warm_pred = target / tau, pred / tau
            ce_loss += -1 * (F.softmax(warm_target, dim=-1) * torch.log_softmax(warm_pred, dim=-1)).sum() * tau * tau
            ce_loss += -1 * torch.log_softmax(pred, dim=-1)[0]
        ce_loss /= (bsz * tau * tau)

        loss = ce_loss

        out = LatentModelOutput(loss=loss, logits=outputs.logits)
        out.actions = outputs.actions
        out.z = t_outputs.z

        return out

    def save_pretrained(self, path):
        self.student.save_pretrained(path)
   
if __name__ == "__main__":
    from transformers import DistilBertConfig

    torch.manual_seed(42)

    passed = 0
    failed = 0

    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")

    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")

    def tiny_config():
        return DistilBertConfig(
            vocab_size=101,
            max_position_embeddings=32,
            n_layers=1,
            n_heads=4,
            dim=16,
            hidden_dim=32,
            dropout=0.0,
            attention_dropout=0.0,
        )

    def fake_pretraining_batch(batch_size=2, seq_len=7, cand_count=3):
        input_ids = torch.randint(0, 101, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        act_input_ids = torch.randint(0, 101, (batch_size, seq_len))
        act_attention_mask = torch.ones_like(act_input_ids)
        cand_input_ids = torch.randint(0, 101, (batch_size, cand_count, seq_len))
        cand_attention_mask = torch.ones_like(cand_input_ids)
        return input_ids, attention_mask, act_input_ids, act_attention_mask, cand_input_ids, cand_attention_mask

    def fake_jericho_batch(batch_size=2, max_actions=4, seq_len=7):
        input_ids = torch.randint(0, 101, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        act_input_ids = torch.randint(0, 101, (batch_size, max_actions, seq_len))
        act_attention_mask = torch.ones_like(act_input_ids)
        act_sizes = torch.tensor([3, 4])
        labels = torch.zeros(batch_size, dtype=torch.long)
        return input_ids, attention_mask, act_input_ids, act_attention_mask, act_sizes, labels

    def finite_list(tensors):
        return all(torch.isfinite(tensor).all().item() for tensor in tensors)

    print("=" * 70)
    print("PASA benchmark: latent action models and distillation")
    print("=" * 70)

    try:
        model = LatentModelForPretraining(tiny_config())
        model.train()
        batch = fake_pretraining_batch()
        out = model(batch[0], batch[1], batch[2], batch[3])
        check("latent pretraining output not None", out is not None)
        if out is not None:
            check("latent pretraining logits shape", out.logits.shape == (2, 2), f"got {tuple(out.logits.shape)}")
            check("latent pretraining loss finite", torch.isfinite(out.loss).item())
            check("latent pretraining mi finite", torch.isfinite(out.mi).item())
            out.loss.backward()
            grad = model.cls[0].weight.grad
            check("latent pretraining classifier gradient", grad is not None and torch.isfinite(grad).all().item())
        else:
            skip_checks(4, "LatentModelForPretraining.forward returned None")
    except Exception as exc:
        skip_checks(5, f"LatentModelForPretraining.forward raised {type(exc).__name__}: {exc}")

    try:
        model = LatentModelForPretraining(tiny_config())
        model.eval()
        batch = fake_pretraining_batch()
        with torch.no_grad():
            out = model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
        check("latent pretraining candidate output not None", out is not None)
        if out is not None:
            check("latent pretraining candidate logits shape", out.logits.shape == (2, 4), f"got {tuple(out.logits.shape)}")
            check("latent pretraining candidate logits finite", torch.isfinite(out.logits).all().item())
            check("latent pretraining candidate loss finite", torch.isfinite(out.loss).item())
        else:
            skip_checks(3, "LatentModelForPretraining.forward candidate branch returned None")
    except Exception as exc:
        skip_checks(4, f"LatentModelForPretraining.forward candidate branch raised {type(exc).__name__}: {exc}")

    try:
        model = IntentModelForPretraining(tiny_config())
        model.train()
        batch = fake_pretraining_batch()
        intents = torch.tensor([0, 4])
        out = model(batch[0], batch[1], batch[2], batch[3], intents=intents)
        check("intent pretraining output not None", out is not None)
        if out is not None:
            check("intent pretraining logits shape", out.logits.shape == (2, 2), f"got {tuple(out.logits.shape)}")
            check("intent pretraining loss finite", torch.isfinite(out.loss).item())
            check("intent pretraining mi finite", torch.isfinite(out.mi).item())
            out.loss.backward()
            grad = model.proj.weight.grad
            check("intent pretraining projection gradient", grad is not None and torch.isfinite(grad).all().item())
        else:
            skip_checks(4, "IntentModelForPretraining.forward returned None")
    except Exception as exc:
        skip_checks(5, f"IntentModelForPretraining.forward raised {type(exc).__name__}: {exc}")

    try:
        model = PersonaModelForPretraining(tiny_config())
        model.train()
        batch = fake_pretraining_batch()
        persona_ids = torch.tensor([0, 374])
        out = model(batch[0], batch[1], batch[2], batch[3], persona_ids=persona_ids)
        check("persona pretraining output not None", out is not None)
        if out is not None:
            check("persona pretraining logits shape", out.logits.shape == (2, 2), f"got {tuple(out.logits.shape)}")
            check("persona pretraining loss finite", torch.isfinite(out.loss).item())
            check("persona pretraining mi finite", torch.isfinite(out.mi).item())
            out.loss.backward()
            grad = model.cls[0].weight.grad
            check("persona pretraining classifier gradient", grad is not None and torch.isfinite(grad).all().item())
        else:
            skip_checks(4, "PersonaModelForPretraining.forward returned None")
    except Exception as exc:
        skip_checks(5, f"PersonaModelForPretraining.forward raised {type(exc).__name__}: {exc}")

    try:
        model = LatentModel(tiny_config())
        model.train()
        batch = fake_jericho_batch()
        out = model(batch[0], batch[1], batch[2], batch[3], batch[4])
        check("latent jericho output not None", out is not None)
        if out is not None:
            check("latent jericho logits list length", len(out.logits) == 2, f"got {len(out.logits)}")
            check("latent jericho per-action shapes", [tuple(x.shape) for x in out.logits] == [(3,), (4,)])
            check("latent jericho states shape", out.states.shape == (2, 16), f"got {tuple(out.states.shape)}")
            check("latent jericho cls shape", out.cls.shape == (2, 8), f"got {tuple(out.cls.shape)}")
            check("latent jericho finite", torch.isfinite(out.loss).item() and finite_list(out.logits))
            out.loss.backward()
            grad = model.proj.weight.grad
            check("latent jericho projection gradient", grad is not None and torch.isfinite(grad).all().item())
        else:
            skip_checks(6, "LatentModel.forward returned None")
    except Exception as exc:
        skip_checks(7, f"LatentModel.forward raised {type(exc).__name__}: {exc}")

    try:
        model = IntentModel(tiny_config())
        model.train()
        batch = fake_jericho_batch()
        intents = torch.tensor([0, 4])
        out = model(batch[0], batch[1], batch[2], batch[3], batch[4], intents=intents)
        check("intent jericho output not None", out is not None)
        if out is not None:
            check("intent jericho logits list length", len(out.logits) == 2, f"got {len(out.logits)}")
            check("intent jericho per-action shapes", [tuple(x.shape) for x in out.logits] == [(3,), (4,)])
            check("intent jericho states shape", out.states.shape == (2, 16), f"got {tuple(out.states.shape)}")
            check("intent jericho cls shape", out.cls.shape == (2, 5), f"got {tuple(out.cls.shape)}")
            check("intent jericho finite", torch.isfinite(out.loss).item() and finite_list(out.logits))
            out.loss.backward()
            grad = model.cls[0].weight.grad
            check("intent jericho classifier gradient", grad is not None and torch.isfinite(grad).all().item())
        else:
            skip_checks(6, "IntentModel.forward returned None")
    except Exception as exc:
        skip_checks(7, f"IntentModel.forward raised {type(exc).__name__}: {exc}")

    try:
        model = PersonaModel(tiny_config())
        model.train()
        batch = fake_jericho_batch()
        out = model(batch[0], batch[1], batch[2], batch[3], batch[4])
        check("persona jericho output not None", out is not None)
        if out is not None:
            check("persona jericho logits list length", len(out.logits) == 2, f"got {len(out.logits)}")
            check("persona jericho per-action shapes", [tuple(x.shape) for x in out.logits] == [(3,), (4,)])
            check("persona jericho states shape", out.states.shape == (2, 16), f"got {tuple(out.states.shape)}")
            check("persona jericho cls shape", out.cls.shape == (2, 375), f"got {tuple(out.cls.shape)}")
            check("persona jericho finite", torch.isfinite(out.loss).item() and finite_list(out.logits))
            out.loss.backward()
            grad = model.proj.weight.grad
            check("persona jericho projection gradient", grad is not None and torch.isfinite(grad).all().item())
        else:
            skip_checks(6, "PersonaModel.forward returned None")
    except Exception as exc:
        skip_checks(7, f"PersonaModel.forward raised {type(exc).__name__}: {exc}")

    try:
        class Args:
            ...

        args = Args()
        args.intent_type = "latent"
        distill = Distillation.__new__(Distillation)
        nn.Module.__init__(distill)
        distill.args = args
        distill.student = BiEncoder(tiny_config())
        distill.teacher = LatentModel(tiny_config())
        distill.train()
        batch = fake_jericho_batch()
        out = distill(batch[0], batch[1], batch[2], batch[3], batch[4], labels=batch[5])
        check("distillation output not None", out is not None)
        if out is not None:
            check("distillation logits list length", len(out.logits) == 2, f"got {len(out.logits)}")
            check("distillation z shape", out.z.shape == (2,), f"got {tuple(out.z.shape)}")
            check("distillation loss finite", torch.isfinite(out.loss).item())
            check("distillation action shapes", [tuple(x.shape) for x in out.actions] == [(3, 16), (4, 16)])
        else:
            skip_checks(4, "Distillation.forward returned None")
    except Exception as exc:
        skip_checks(5, f"Distillation.forward raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        print(f"{failed} check(s) FAILED")
        raise SystemExit(1)
    print("All tests PASSED")
