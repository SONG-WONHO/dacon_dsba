#!/usr/bin/env python
# coding: utf-8
MAX_SIZE = 5000

import random
import os
import gc
import sys
import ast
import copy
import json
import math
import warnings
import argparse
from tqdm import tqdm
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from transformers import BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from tokenizer import BertTokenizer


warnings.filterwarnings("ignore")


""" Commonly used functions
"""


def get_exp_id(path="./model/", prefix="exp_"):
    """Get new experiement ID

    Args:
        path: Where the "model/" or "log/" used in the project is stored.
        prefix: Experiment ID ahead.

    Returns:
        Experiment ID
    """
    files = set([int(d.replace(prefix, "")) for d in os.listdir(path) if prefix in d])
    if len(files):
        return min(set(range(0, max(files) + 2)) - files)
    else:
        return 0


def seed_everything(seed=42):
    """Seed All

    Args:
        seed: seed number
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    """Get device type

    Returns: device, "cpu" if cuda is available else "cuda"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_data(config):
    train_df = pd.read_csv(os.path.join(config.root_path, "train.csv"))
    test_ext_df = pd.read_csv(
        os.path.join(config.root_path, "test_extractive.csv"))
    test_abs_df = pd.read_csv(
        os.path.join(config.root_path, "test_abstractive.csv"))
    ss_ext_df = pd.read_csv(
        os.path.join(config.root_path, "sample_submission_extractive.csv"))
    ss_abs_df = pd.read_csv(
        os.path.join(config.root_path, "sample_submission_abstractive.csv"))

    return train_df, test_ext_df, test_abs_df, ss_ext_df, ss_abs_df


def preprocess(df, test=False):
    df['article_original'] = df['article_original'].apply(lambda v: ast.literal_eval(v))
    if not test:
        df['extractive'] = df['extractive'].apply(lambda v: ast.literal_eval(v))


class DSBADataset(Dataset):
    def __init__(self, config, df, tokenizer, augment=False, test=False):
        """ Init Dataset Class

        Args:
            config: CFG information
            df: Dataframe
            tokenizer: Tokenizer Object
            test: test or net
        """
        self.config = config
        self.df = df
        self.tokenizer = tokenizer
        self.items = df[
            ['id', 'media', 'article_original', 'extractive', 'abstractive']].values
        self.augment = augment
        self.test = test

    def __len__(self):
        """Len

        Returns: Dataset size
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Get Item

        Args:
            idx: index

        Returns: item
        """
        t_id, media, txt, label, txt_tgt = self.items[idx]

        src = []
        segs = []
        clss = []
        mask_src = []
        mask_cls = []
        num_tokens = []

        num = 0
        for i, sent in enumerate(txt):
            # sent = self.tokenizer.encode(sent)
            tokens = tokenizer.tokenize(sent)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            sent = tokenizer.convert_tokens_to_ids(tokens)

            src += sent
            segs += [i % 2] * len(sent)
            clss.append(num)
            mask_src += [1] * len(sent)
            mask_cls.append(1)
            num += len(sent)
            num_tokens.append(num)

        labels = torch.zeros(len(txt))
        if not self.test: labels[label] = 1
        labels = labels.numpy().tolist()

        tgt = [0]
        mask_tgt = [1]
        if not self.test:
            # tgt = self.tokenizer.encode(txt_tgt)
            tokens = tokenizer.tokenize(txt_tgt)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tgt = tokenizer.convert_tokens_to_ids(tokens)
            if len(tgt) > 150:
                tgt = tgt[:150]
                tgt[-1] = 3

            mask_tgt = [1] * len(tgt)

        # augmentation
        if self.augment:
            # get candidate
            cands = []
            clss, num_tokens = np.asarray(clss), np.asarray(num_tokens)
            for i, n in enumerate(clss):
                j = np.where(num_tokens - n <= 512)[0][-1] + 1
                cands.append((i, j, sum(labels[i:j])))
            max_num = max([c[-1] for c in cands])
            cands = [c for c in cands if c[-1] == max_num]
            # select one
            cand = random.choice(cands)

            # update
            src = src[clss[cand[0]]: num_tokens[:cand[1]][-1]]
            mask_src = mask_src[clss[cand[0]]: num_tokens[:cand[1]][-1]]
            segs = segs[clss[cand[0]]: num_tokens[:cand[1]][-1]]
            clss = clss[cand[0]: cand[1]]
            clss -= clss[0]
            clss = clss.tolist()
            labels = labels[cand[0]: cand[1]]
            mask_cls = mask_cls[cand[0]: cand[1]]

        return src, segs, clss, mask_src, mask_cls, labels, tgt, mask_tgt


def collate_fn(batch):
    """ collate functions

    Args:
        batch: batch

    Returns: src, segs, clss, mask_src, mask_cls, labels
    """

    # max length
    max_len = min(max([len(b[0]) for b in batch]), CFG.max_len)
    # max_len cls token
    max_len_cls = max([len(b[2]) for b in batch])
    # max length target
    max_len_tgt = max([len(b[7]) for b in batch])

    # encoded
    src = torch.LongTensor([b[0] + [0] * (max_len - len(b[0])) for b in batch])
    segs = torch.LongTensor([b[1] + [0] * (max_len - len(b[1])) for b in batch])
    clss = torch.LongTensor(
        [b[2] + [0] * (max_len_cls - len(b[2])) for b in batch])
    mask_src = torch.LongTensor(
        [b[3] + [0] * (max_len - len(b[3])) for b in batch])
    mask_cls = torch.LongTensor(
        [b[4] + [0] * (max_len_cls - len(b[4])) for b in batch])
    labels = torch.FloatTensor(
        [b[5] + [0] * (max_len_cls - len(b[5])) for b in batch])
    tgt = torch.LongTensor(
        [b[6] + [0] * (max_len_tgt - len(b[6])) for b in batch])
    mask_tgt = torch.LongTensor(
        [b[7] + [0] * (max_len_tgt - len(b[7])) for b in batch])

    return src, segs, clss, mask_src, mask_cls, labels, tgt, mask_tgt


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in
               arguments), "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, attn_type="dot"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None,
                memory_masks=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_masks is not None:
            memory_masks = memory_masks.transpose(0, 1)
            memory_masks = memory_masks.transpose(1, 2)
            align.masked_fill_(1 - memory_masks.byte(), -float('inf'))

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        align_vectors = F.softmax(align.view(batch * target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)


        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, align_vectors


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1,
                                                                              2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                       head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), self.linear_keys(
                    query), self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), self.linear_values(
                            value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], layer_cache[
                            "memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), self.linear_values(
                        value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)

            scores = scores.masked_fill(mask.type(torch.bool), -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (
                        torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """

    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x,
                                          1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, all_input
        # return output

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)


        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None, cache=None,memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return output, state

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)


class BaseModel2(nn.Module):
    def __init__(self, config):
        super(BaseModel2, self).__init__()

        self.config = config

        # bert encoder
        print(f"... {config.etri_path}")
        bert_config = BertConfig.from_json_file(os.path.join(config.etri_path, "config.json"))
        bert_config.attention_probs_dropout_prob = config.dropout
        bert_config.hidden_dropout_prob = config.dropout
        self.bert = BertModel.from_pretrained(None, config=bert_config, state_dict=torch.load(os.path.join(config.etri_path, "pytorch_model.bin")))

        self.vocab_size = self.bert.config.vocab_size

        tgt_emb = nn.Embedding(
            self.vocab_size, self.bert.config.hidden_size,
            padding_idx=self.bert.config.pad_token_id)
        tgt_emb.weight = copy.deepcopy(
            self.bert.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            8, 768, 8, d_ff=2048, dropout=0.2, embeddings=tgt_emb)

        generator = nn.Sequential(
            nn.Linear(768, self.vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        self.generator = generator.to(self.config.device)
        self.generator[0].weight = self.decoder.embeddings.weight

        # init decoder weight
        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        for p in self.generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()

        tgt_emb = nn.Embedding(
            self.vocab_size, self.bert.config.hidden_size,
            padding_idx=self.bert.config.pad_token_id)
        tgt_emb.weight = copy.deepcopy(
            self.bert.embeddings.word_embeddings.weight)

        self.decoder.embeddings = tgt_emb
        self.generator[0].weight = self.decoder.embeddings.weight

        # out
        self.ext_layer = ExtTransformerEncoder(self.bert.config.hidden_size,
                                               2048, 8, 0.2, 1)

        self.ext_layer = Classifier(self.bert.config.hidden_size)

        if (config.max_len > 512):
            my_pos_embeddings = nn.Embedding(config.max_len, self.bert.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][None, :].repeat(config.max_len - 512, 1)
            self.bert.embeddings.position_embeddings = my_pos_embeddings
            self.bert.embeddings.position_ids = torch.arange(config.max_len).expand((1, -1))

    def forward(self, src, mask_src, segs, clss, mask_cls, tgt):
        # (last_hidden_state, pooler_output, hidden_states, attentions)
        top_vec, _ = self.bert(src, mask_src, segs)

        # abstract
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        decoder_outputs = self.generator(decoder_outputs)

        # extract
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls)

        return sent_scores, mask_cls, decoder_outputs


def get_model(config):
    try:
        f = globals().get(f"{config.model_name}")
        return f(config)

    except TypeError:
        raise NotImplementedError("model name not matched")


# average meter
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# loss function
def loss_func(pred, target):
    return nn.BCELoss(reduction='none')(pred, target)


class Learner(object):
    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.logger = None
        self.name = "model"

    def train(self, trn_data, val_data, model, optimizer, scheduler):

        # dataloader
        train_loader = DataLoader(
            trn_data,
            batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.workers, pin_memory=True,
            collate_fn=collate_fn
        )

        valid_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=self.config.workers, pin_memory=True,
            collate_fn=collate_fn
        )

        # loger
        logger = self._create_logger()

        # training
        best_loss = 1e8
        for epoch in range(self.config.num_epochs):
            tr_loss, tr_abs_loss = self._train_one_epoch(train_loader, model, optimizer,
                                            scheduler)
            vl_loss, vl_abs_loss = self._valid_one_epoch(valid_loader, model)

            # logging
            logger.loc[epoch] = [
                np.round(tr_loss, 4),
                np.round(tr_abs_loss, 4),
                np.round(vl_loss, 4),
                np.round(vl_abs_loss, 4),
                optimizer.param_groups[0]['lr']]

            logger.to_csv(os.path.join(self.config.log_path,
                                       f'log.{self.name.split(".")[-1]}.csv'))

            # save model
            if best_loss >= logger.loc[epoch, 'val_abs_loss']:
                print(
                    f"... From {best_loss:.4f} To {logger.loc[epoch, 'val_abs_loss']:.4f}")
                best_loss = logger.loc[epoch, 'val_abs_loss']
                self.best_model = copy.deepcopy(model)
                name = self.name
                self.name = f"{name}.epoch_{epoch}"
                self.save()
                self.name = f"{name}.best"
                self.save()
                self.name = name

        self.logger = logger

    def predict(self, tst_data):
        model = self.best_model

        test_loader = DataLoader(
            tst_data,
            batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=False
        )

        pred_final = []

        model.eval()

        test_loader = tqdm(test_loader, leave=False)

        for X_batch, _ in test_loader:
            X_batch = X_batch.to(self.config.device)

            with torch.no_grad():
                preds = model(X_batch)

            preds = preds.cpu().detach()

            pred_final.append(preds)

        pred_final = torch.cat(pred_final, dim=0)

        return pred_final

    def save(self):
        if self.best_model is None:
            print("Must Train before save !")
            return

        torch.save({
            "logger": self.logger,
            "model_state_dict": self.best_model.cpu().state_dict(),
        }, f"{os.path.join(self.config.model_path, self.name)}.pt")

    def load(self, path, name=None):
        ckpt = torch.load(path)
        self.logger = ckpt['logger']
        model_state_dict = ckpt[name]
        model = get_model(self.config)
        try:
            model.load_state_dict(model_state_dict)
            print("... Single GPU (Train)")
        except:
            def strip_module_str(v):
                if v.startswith('module.'):
                    return v[len('module.'):]

            model_state_dict = {strip_module_str(k): v for k, v in
                                model_state_dict.items()}
            model.load_state_dict(model_state_dict)
            print("... Multi GPU (Train)")

        self.best_model = model.to(self.config.device)
        print("... Model Loaded!")

    def _train_one_epoch(self, train_loader, model, optimizer, scheduler):
        losses = AverageMeter()
        losses_abs = AverageMeter()

        model.train()
        train_iterator = tqdm(train_loader, leave=False)
        for src, segs, clss, mask_src, mask_cls, labels, tgt, mask_tgt in train_iterator:
            src = src.to(self.config.device)
            segs = segs.to(self.config.device)
            clss = clss.to(self.config.device)
            mask_src = mask_src.to(self.config.device)
            mask_cls = mask_cls.to(self.config.device)
            labels = labels.to(self.config.device)
            tgt = tgt.to(self.config.device)
            mask_tgt = mask_tgt.to(self.config.device)

            batch_size = src.size(0)

            preds, _, outputs = model(src, mask_src, segs, clss, mask_cls, tgt)

            # ext
            loss = loss_func(preds, labels)
            loss = (loss * mask_cls.float()).mean()
            losses.update(loss.item(), batch_size)

            # abs
            tgt = tgt[:, 1:]

            loss_abs = nn.NLLLoss(ignore_index=self.config.pad_token_id, reduction='none')(
                outputs.view(-1, self.config.vocab_size), tgt.contiguous().view(-1)).view(batch_size, -1)
            loss_abs = (loss_abs.sum(-1) / mask_tgt.sum(-1)).mean()
            losses_abs.update(loss_abs.item(), batch_size)

            optimizer.zero_grad()
            loss_abs.backward()
            optimizer.step()
            scheduler.step()

            train_iterator.set_description(
                f"train ext:{losses.avg:.4f} abs {losses_abs.avg:.4f}, lr:{optimizer.param_groups[0]['lr']:.6f}, lr2:{optimizer.param_groups[1]['lr']:.6f}")

        return losses.avg, losses_abs.avg

    def _valid_one_epoch(self, valid_loader, model):
        losses = AverageMeter()
        losses_abs = AverageMeter()

        model.eval()

        valid_loader = tqdm(valid_loader, leave=False)
        for i, (src, segs, clss, mask_src, mask_cls, labels, tgt, mask_tgt) in enumerate(
                valid_loader):
            src = src.to(self.config.device)
            segs = segs.to(self.config.device)
            clss = clss.to(self.config.device)
            mask_src = mask_src.to(self.config.device)
            mask_cls = mask_cls.to(self.config.device)
            labels = labels.to(self.config.device)
            tgt = tgt.to(self.config.device)
            mask_tgt = mask_tgt.to(self.config.device)

            batch_size = src.size(0)

            with torch.no_grad():
                preds, _, outputs = model(src, mask_src, segs, clss, mask_cls, tgt)

                # ext
                loss = loss_func(preds, labels)
                loss = (loss * mask_cls.float()).mean()
                losses.update(loss.item(), batch_size)

                # abs
                tgt = tgt[:, 1:]
                loss_abs = nn.NLLLoss(ignore_index=self.config.pad_token_id,
                                      reduction='none')(
                    outputs.view(-1, self.config.vocab_size),
                    tgt.contiguous().view(-1)).view(batch_size, -1)
                loss_abs = (loss_abs.sum(-1) / mask_tgt.sum(-1)).mean()
                losses_abs.update(loss_abs.item(), batch_size)

            losses_abs.update(loss_abs.item(), batch_size)

            valid_loader.set_description(f"valid ext:{losses.avg:.4f} abs {losses_abs.avg:.4f}")

        return losses.avg, losses_abs.avg

    def _create_logger(self):
        log_cols = ['tr_loss', 'tr_abs_loss', 'val_loss', 'val_abs_loss', 'lr']
        return pd.DataFrame(index=range(self.config.num_epochs),
                            columns=log_cols)


class CFG:
    # path
    root_path = "./input/data/"
    log_path = "./log/"
    model_path = "./model/"
    etri_path = "./input/etri/"

    # preprocess
    max_len = 1536

    # model
    model_name = "BaseModel2"
    pretrained_name = "bert-base-uncased"
    dropout = 0.3

    # train
    batch_size = 16
    learning_rate = 1e-5
    num_epochs = 12
    start_epoch = 0
    warmup_steps = 300

    # etc
    seed = 42
    workers = 8
    num_targets = 2
    val_fold = 0
    n_splits = 5

    # ext
    ext_version = 5
    ext_exp_id = 1


# get version
_, version, _ = sys.argv[0].split('/')
CFG.version = version
# get device
CFG.device = get_device()

# update log path
CFG.log_path = os.path.join(CFG.log_path, CFG.version)
os.makedirs(CFG.log_path, exist_ok=True)
CFG.log_path = os.path.join(CFG.log_path, f'exp_{get_exp_id(CFG.log_path, prefix="exp_")}')
os.makedirs(CFG.log_path, exist_ok=True)

# update model path
CFG.model_path = os.path.join(CFG.model_path, CFG.version)
os.makedirs(CFG.model_path, exist_ok=True)
CFG.model_path = os.path.join(CFG.model_path, f'exp_{get_exp_id(CFG.model_path, prefix="exp_")}')
os.makedirs(CFG.model_path, exist_ok=True)

pprint({k: v for k, v in dict(CFG.__dict__).items() if '__' not in k})
json.dump(
    {k: v for k, v in dict(CFG.__dict__).items() if '__' not in k},
    open(os.path.join(CFG.log_path, 'CFG.json'), "w"))

### seed all
seed_everything(CFG.seed)

train_df, test_ext_df, test_abs_df, ss_ext_df, ss_abs_df = load_data(CFG)

preprocess(train_df)
preprocess(test_ext_df, True)
preprocess(test_abs_df, True)

train_df['fold'] = -1
folds = StratifiedKFold(CFG.n_splits, shuffle=True, random_state=CFG.seed)
for fold, (tr_idx, vl_idx) in enumerate(folds.split(train_df, pd.qcut(
    train_df['article_original'].apply(lambda v: len(" ".join(v).split(" "))), 
    np.arange(0, 1.01, 0.1), labels=False))):
    train_df.loc[vl_idx, 'fold'] = fold

tokenizer = BertTokenizer.from_pretrained(os.path.join(CFG.etri_path, "vocab.korean.rawtext.list"), do_lower_case=False)
CFG.pad_token_id = tokenizer.vocab["[PAD]"]
CFG.vocab_size = len(tokenizer.vocab)

trn_dataset = DSBADataset(
    CFG, train_df[train_df['fold'] != CFG.val_fold], tokenizer, augment=True, test=False)
val_dataset = DSBADataset(
    CFG, train_df[train_df['fold'] == CFG.val_fold], tokenizer, augment=False, test=False)

# samples
loader = DataLoader(trn_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
for d in loader: break
print("Sample: ", tokenizer.convert_ids_to_tokens(d[0][1].numpy()))

# get model
print(f"Get Model: {CFG.model_name}")
model = get_model(CFG)

# load ext weight
print("load bert weight")
state_dict = torch.load(os.path.join("model", f"v{CFG.ext_version}", f"exp_{CFG.ext_exp_id}", f"model.fold_{CFG.val_fold}.best.pt"))['model_state_dict']

model.bert.load_state_dict(
    {k.replace("bert.", ""): v
     for k, v
     in state_dict.items()
     if k.startswith("bert.")})

optimizer = AdamW([
                {'params': model.bert.parameters(), 'lr': CFG.learning_rate},
                {'params': model.decoder.parameters(), 'lr': CFG.learning_rate * 10},
            ])

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(CFG.device)

assert (model.bert.embeddings.word_embeddings.weight == state_dict['bert.embeddings.word_embeddings.weight'].to(CFG.device)).all()

# get scheduler
num_training_steps = int(len(trn_dataset) / CFG.batch_size) * (CFG.num_epochs)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=CFG.warmup_steps, num_training_steps=num_training_steps)

### Train related logic
# get learner
learner = Learner(CFG)
learner.name = f"model.fold_{CFG.val_fold}"
learner.train(trn_dataset, val_dataset, model, optimizer, scheduler)
