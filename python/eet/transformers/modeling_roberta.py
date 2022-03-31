#
# Created by zsd on 2022/02/22.
#
"""EET transformers roberta model. """

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from transformers import RobertaModel
from eet.transformers.modeling_transformer import *
from eet.transformers.modeling_bert import EETBertEmbedding
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding


class EETRobertaModel():
    def __init__(self, config, embedding, encoder):
        self.embedding = embedding
        self.encoder = encoder
        self.pre_padding_len = torch.empty(0).long()
        self.padding_idx = config.pad_token_id

    def __call__(
            self,
            input_ids,
            position_ids=None,
            token_type_ids=None,
            attention_mask=None,
    ):
        '''
        attention_mask:attention_padding_mask(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input.)
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        '''
        input_shape = input_ids.size()

        # Same as BertEmbeddings with a tiny tweak for positional embeddings indexing
        mask = input_ids.ne(self.padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + self.padding_idx

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len = torch.sum(1 - attention_mask, 1).long().cuda()

        embedding_out = self.embedding(input_ids, position_ids, token_type_ids)

        encoder_out = self.encoder(embedding_out,
                                   pre_padding_len=pre_padding_len,
                                   normalize_before=False)

        return encoder_out

    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = RobertaModel.from_pretrained(model_id_or_path)
        model_name = type(torch_model).__name__
        cfg = torch_model.config

        for k, v in torch_model.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # group by 'layer.n'
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cuda:0"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers,
                           cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                           activation_fn)

        embedding = EETBertEmbedding.from_torch(config, embedding_dict, data_type)
        # embedding = None
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        eet_model = EETRobertaModel(cfg, embedding, encoder)
        return eet_model
