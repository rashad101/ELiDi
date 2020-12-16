import torch
from torch import nn
import torch.nn.functional as F
from models.relation_det import RelationPrediction
from models.entity_det import EntityDetection
from utils.util import threshold
from torch.autograd import Variable
from itertools import groupby
from utils.args import get_args
import math

args = get_args()
class Gumbel():
    def __init__(self, gpu):
        self.gpu = gpu

    def sample_gumbel(self, shape, eps=1e-20):
        if self.gpu:
            U = torch.rand(shape).cuda()
        else:
            U = torch.rand(shape)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self, threshold=-7.6):
        super(Binarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(self.threshold)] = 0
        outputs[inputs.gt(self.threshold)] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class E2E_entity_linker(nn.Module):
    def __init__(self, num_words, emb_dim, kg_emb_dim, hidden_size, num_layers, ent_cand_size, rel_size,
                 dropout=0.5, emb_dropout=0.2, pretrained_emb=None, train_embed=False, pretrained_ent=None,
                 batch_first=False, pretrained_rel=None, use_cuda=False):
        super(E2E_entity_linker, self).__init__()
        self.hidden_size = hidden_size
        self.num_words = num_words
        self.use_cuda = use_cuda
        self.emb_dim = emb_dim
        self.rel_size = rel_size
        self.num_layer = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        #self.binarizer = Binarizer()
        self.embed = nn.Embedding(self.num_words, self.emb_dim, padding_idx=0)
        # self.kg_ent = nn.Embedding(total_ent, kg_emb_dim)
        self.kg_rel = nn.Embedding(rel_size, kg_emb_dim, padding_idx=0)
        self.gumbel = Gumbel(use_cuda)
        self.cosine_ent_l = nn.CosineSimilarity(dim=-1)
        self.cosine_kb = nn.CosineSimilarity(dim=-1)
        self.eucli = nn.PairwiseDistance(keepdim=True)
        self.all_reln = torch.Tensor(range(0, 6701)).long()
        if use_cuda:
            self.all_reln = self.all_reln.cuda()
        #self.pos_emb = PositionalEncoder(max_seq_len=36, d_model=self.emb_dim)
        # print (pretrained_ent.size())
        if pretrained_emb is not None:
            self.embed.weight.data.copy_(pretrained_emb)
        #if pretrained_ent is not None:
        #    self.kg_ent.weight.data.copy_(pretrained_ent)
        if pretrained_rel is not None:
            self.kg_rel.weight.data.copy_(pretrained_rel)
        if train_embed == False:
            #self.embed.weight.requires_grad = False
            self.kg_rel.weight.requires_grad = False
            #self.kg_ent.weight.requires_grad = False
        self.soft_gate = nn.Linear(ent_cand_size, 1, bias=False)
        #self.soft_gate_bn = nn.BatchNorm1d(ent_cand_size)
        #self.ent_emb_w = nn.Linear(ent_cand_size, ent_cand_size)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.rel_det = RelationPrediction(n_words=self.num_words, emb_dim=self.emb_dim, h_dim=self.hidden_size, target_size=self.rel_size,
                                     lstm_drop=self.dropout, dropout=self.dropout)
        self.ent_det = EntityDetection(num_words=self.num_words, emb_dim=self.emb_dim, hidden_size=self.hidden_size,
                                       num_layers=self.num_layer, batch_f=self.batch_first, dropout=self.dropout)


        # self.ent_linker = nn.Linear(1000, 1000)
        torch.nn.init.xavier_uniform_(self.soft_gate.weight)
        self.stoi, vectors, dim = torch.load(args.vector_cache)
        self.stoi['<unk>'] = len(self.stoi)
        self.stoi['<pad>'] = 0  # add padding index and remove comma to another index
        self.stoi[','] = len(self.stoi)

    def forward(self, text, e_label, t_mask, e_candidates, cand_labels, relations, reln_mask, reln_1hop, reln_1hop_mask,
                cand_rank, c_wiki, train=True):
        maxpool = nn.MaxPool1d(reln_1hop.size(2))
        #avgpool = nn.AvgPool1d(reln_1hop.size(2))
        if self.batch_first:
            text, e_label, t_mask = text.transpose(0, 1), e_label.transpose(0, 1), t_mask.transpose(0, 1)
        # Get embedding for input
        # x_text = self.emb_drop(self.embed(text))  # S X B X EMB
        # x_pos = pemb.transpose(0, 1) # S X B X E

        x_text = self.embed(text)  # S X B X EMB

        # Pass through entity span detector
        '''
        Detecting Entity spans
        '''
        ent_pred_out = self.ent_det(x_text, t_mask)  # S * B X 1
        # Get sigmoid out from span detector
        _ent_o = torch.sigmoid(ent_pred_out).view(x_text.size(0), x_text.size(1), -1)  # S x B X 1
        # _ent_o = (ent_pred_out).view(x_text.size(0), x_text.size(1), -1)  # S x B X 1
        # Multiply span detector output with original embedding
        if train:
            _avg_x = x_text * _ent_o  # S X B X EMB
        else:
            _ent_o = (_ent_o > threshold).float()
            _avg_x = x_text * _ent_o

        # Get average embedding for detected spans
        _avg_x = _avg_x.transpose(0, 1).mean(1)  # B X EMB

        # Get embedding for all entitity labels and average them
        ent_c_emb = self.embed(cand_labels).mean(2).transpose(0, 1)# B X E_CAND X EMB ==> E_CAND X B X EMB
        # Get cosine similarities for original entity with candidates for word embeddings
        # ent_avg_score = [self.cosine_ent_l(_avg_x, ent_c_emb[i].squeeze()) for i in range(ent_c_emb.size(0))]
        _avg_x = _avg_x.expand(ent_c_emb.size(0), ent_c_emb.size(1), ent_c_emb.size(2)) # E_CAND X B X EMB
        cos_l = self.cosine_ent_l(_avg_x, ent_c_emb).transpose(0, 1).contiguous()  # B X E_CAND
        # cos_l = cos_l.view(-1).unsqueeze(1).view(e_candidates.size(0), e_candidates.size(1))
        # ent_avg_score = torch.cat(ent_avg_score, 0).unsqueeze(1)  # B * E_CAND X 1
        # print (ent_avg_score.size())
        ''''
        Get relation predictions
        '''
        reln_emb = self.kg_rel(self.all_reln.expand(relations.size(0), -1))  # B X N_R X R_EMB

        rel_out = self.rel_det(x_text, t_mask)  # B X NUM_R

        rel_soft = self.gumbel.gumbel_softmax(rel_out, 0.8)

        rel_avg = (reln_emb * rel_soft.unsqueeze(-1)).sum(1)  # B X R_EMB

        ent_1hop_emb = self.kg_rel(reln_1hop).transpose(0, 1)  # E_CAND X B X 1HOP_RELN X R_EMB_S
        rel_avg = rel_avg.unsqueeze(0).unsqueeze(2).expand(e_candidates.size(1), rel_avg.size(0), ent_1hop_emb.size(2), -1)  # E_CAND X B X R_EMB_S
        # cos_kb = [self.cosine_kb(rel_avg, ent_c_kgemb[j]) for j in range(ent_c_kgemb.size(0))]  # B X E_CAND
        cos_kb = self.cosine_kb(ent_1hop_emb, rel_avg) * reln_1hop_mask.transpose(0, 1)  # E_CAND X B X 1HOP_RELN

        cos_pooled = maxpool(cos_kb)  # E_CAND X B X 1
        cos_pooled = cos_pooled.squeeze(2).transpose(0, 1)  # B X E_CAND

        sa_gate = self.emb_drop(self.soft_gate(cos_l))
        #print (pooled_emb_avg.size())
        amb_decision = (torch.sigmoid(sa_gate))
        cos_cat = amb_decision*((cos_l+cos_pooled)/2) + (1-amb_decision)*cos_l

        #cos_cat = (self.cos_w(F.softmax(cos_pooled, dim=-1)) + self.ent_emb_w(F.softmax(cos_l, dim=-1))) # .view(e_candidates.size(0), e_candidates.size(1))
        # print (cos_cat.size())

        cos_avg = torch.sigmoid(cos_cat)
        # ent_pred = torch.sigmoid(self.ent_linker(cos_cat)).view(e_candidates.size(0), e_candidates.size(1))
        #ent_pred = torch.tanh(self.ent_linker(cos_avg))
        # print (ent_pred.size())
        ent_linker = F.log_softmax(cos_avg, dim=-1)  # B X E_CAND

        l2_soft_loss = (self.ent_det.hidden2tag[0].weight - self.rel_det.hidden2tag[0].weight).pow(2).sum()

        if train:
            return ent_pred_out, rel_out, ent_linker, sa_gate, l2_soft_loss
        else:
            return ent_pred_out, rel_out, ent_linker, amb_decision, cos_l.view(e_candidates.size(0), e_candidates.size(1)), \
                   cos_pooled.view(e_candidates.size(0), e_candidates.size(1))

    def get_w2id(self,word):
        # get word2ids
        try:
            return int(self.stoi[word])
        except KeyError:
            return int(self.stoi['<unk>'])

    def infer(self, q, text, t_mask, t2id, e2id, topk, e1hop, get_candidates):
        """
        module to use during inference
        """
        predicted_entities = []
        maxpool = nn.MaxPool1d(8)
        text = text.transpose(0, 1).to(next(self.parameters()).device)  # B X S ==> S X B
        t_mask = t_mask.transpose(0, 1).to(next(self.parameters()).device)  # B X S ==> S X B
        x_text = self.embed(text)  # S X B X EMB Embedding the text
        ent_pred_out = self.ent_det(x_text, t_mask)  # S * B X 1
        e = ''
        # pass thorugh a sigmoid
        _ent_o = torch.sigmoid(ent_pred_out).view(x_text.size(0), x_text.size(1), -1)  # S x B X 1
        _ent_o_s = (_ent_o > threshold).float()

        entities = self.get_entities(q, _ent_o_s)  # Get the entities
        # while not entities:
        #      _ent_o_s = (_ent_o > (threshold-0.1)).float()
        #      entities = self.get_entities(q, _ent_o_s)  # Get the entities
        # print (entities)
        # Get the relations
        rel_out = self.rel_det(x_text, t_mask)  # B X NUM_R
        relation_out = torch.argmax(rel_out)  # pred_relation

        for e in entities:
            ent_w = torch.LongTensor([self.get_w2id(w) for w in e.split()]).to(next(self.parameters()).device)
            ent_w_emb = self.embed(ent_w)
            _avg_x = ent_w_emb.mean(0)  # B X EMB

            cand_l, cand_id, can_reln = get_candidates(e, self.get_w2id, topk, e2id, e1hop, text.device)
            # Get embedding for all entitity labels and average them
            ent_c_emb = self.embed(cand_l).mean(2).transpose(0, 1)  # B X E_CAND X EMB ==> E_CAND X B X EMB
            # Get cosine similarities for original entity with candidates for word embeddings
            # ent_avg_score = [self.cosine_ent_l(_avg_x, ent_c_emb[i].squeeze()) for i in range(ent_c_emb.size(0))]
            _avg_x = _avg_x.expand(ent_c_emb.size(0), ent_c_emb.size(1), ent_c_emb.size(2))  # E_CAND X B X EMB
            cos_l = self.cosine_ent_l(_avg_x, ent_c_emb).transpose(0, 1).contiguous()  # B X E_CAND
            # cand = torch.LongTensor(cand_l)
            reln_emb = self.kg_rel(self.all_reln.expand(1, -1))  # B X N_R X R_EMB

            # rel_out = self.rel_det(x_text, t_mask)  # B X NUM_R

            rel_soft = self.gumbel.gumbel_softmax(rel_out, 0.8)

            rel_avg = (reln_emb * rel_soft.unsqueeze(-1)).sum(1)  # B X R_EMB

            ent_1hop_emb = self.kg_rel(can_reln).transpose(0, 1)  # E_CAND X B X 1HOP_RELN X R_EMB_S
            rel_avg = rel_avg.unsqueeze(0).unsqueeze(2).expand(ent_c_emb.size(1), rel_avg.size(0),
                                                               ent_1hop_emb.size(2), -1)  # E_CAND X B X R_EMB_S
            # cos_kb = [self.cosine_kb(rel_avg, ent_c_kgemb[j]) for j in range(ent_c_kgemb.size(0))]  # B X E_CAND
            cos_kb = self.cosine_kb(ent_1hop_emb, rel_avg) # E_CAND X B X 1HOP_RELN

            cos_pooled = maxpool(cos_kb).squeeze(2).transpose(0, 1)  # B X E_CAND

            sa_gate = self.emb_drop(self.soft_gate(cos_l))
            # print (pooled_emb_avg.size())
            amb_decision = (torch.sigmoid(sa_gate))
            cos_cat = amb_decision * ((cos_l + cos_pooled) / 2) + (1 - amb_decision) * cos_l

            # print (cos_cat.size())

            cos_avg = torch.sigmoid(cos_cat)
            # ent_pred = torch.sigmoid(self.ent_linker(cos_cat)).view(e_candidates.size(0), e_candidates.size(1))
            # ent_pred = torch.tanh(self.ent_linker(cos_avg))
            # print (ent_pred.size())
            ent_linker = F.log_softmax(cos_avg, dim=-1)  # B X E_CAND
            pred_ent = torch.argmax(ent_linker, dim=-1)
            ent_linker_out = [l.item() for l in ent_linker.squeeze()]
            ent_prob = [(e, ent_linker_out[j]) for j, e in enumerate(cand_id)]  # sort based on scores
            cand_sorted = sorted(ent_prob, key=lambda x: x[1], reverse=True)
            # print (pred_ent)
            predicted_entities.append(cand_sorted)

            # print (cands)

        return predicted_entities, relation_out, e

    def get_entities(self, query, predicted_ent):
        """
        Get entity spans from query and predicted entity span
        query: question text
        predicted_ent: predicted entity span has value in (0, 1) size = equal to query
        """
        predicted_ent = predicted_ent.transpose(0, 1)  # B X S
        predicted_ent = predicted_ent.tolist()[:len(query)]
        # predicted_ent = predicted_ent[0][0]
        predicted_ent = [p[0] for p in predicted_ent[0]]
        group_adjacent_ent = [list(y) for x, y in groupby(predicted_ent)]  # group adjacent entities predicted as 1
        token_count = 0.0
        entities = []
        for e in group_adjacent_ent:
            if 1.0 not in e:
                token_count += len(e)
            else:
                ent = query[int(token_count):int(token_count + len(e))]
                token_count += len(e)
                entities.append(' '.join(t for t in ent))
        return entities








