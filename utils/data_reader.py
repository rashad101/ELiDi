# imports
import pickle
import numpy as np
from collections import defaultdict
import os
import torch
import unidecode
import json
import random
import math
from utils.args import get_args
import pickle
import re
args = get_args()


class SimpleQABatcher:
    """
    Wrapper for batching the SimpleQuestions dataset
    """
    def __init__(self, gpu=True, max_sent_len=36, entity_cand_size=100, kg_dim=50):
        self.batch_size = args.batch_size
        # self.use_mask = use_mask
        self.gpu = gpu
        self.max_sent_len = max_sent_len
        self.ent_cand_s = entity_cand_size
        #
        # # load (e|r)2idx and (e|r) vectors
        # if os.path.isfile(args.kg_embed):
        #     self.r2id, r_vectors, self.e2id, e_vectors = np.load(args.kg_embed)
        #     self.r2id = self.r2id.item() # ndarray to dict
        #     self.e2id = self.e2id.item() # ndarray to dict
        #     del(r_vectors, e_vectors)

        with open(args.rel2id_f, 'rb') as f:
            self.r2id = json.load(f)
        with open(args.entity2id_f, 'rb') as f:
            self.e2id = json.load(f)
        with open(args.entity_fb2w_map, 'rb') as f:
            self.fb2w = pickle.load(f)
        # self.r_weights = torch.load('data/weights.pt')
        self.fb2w = set([int(e) for e in self.fb2w])
        if os.path.isfile(args.vector_cache):
            self.stoi, self.vectors, self.dim = torch.load(args.vector_cache)
        self.stoi['<unk>'] = len(self.stoi)
        self.stoi['<pad>'] = 0 # add padding index and remove comma to another index
        self.stoi[','] = len(self.stoi)

        # concatenate random vector for <unq> token and extra vocab word
        self.vectors = torch.cat([self.vectors, torch.FloatTensor(self.dim).uniform_(-0.25, 0.25).unsqueeze(0)], 0)
        self.vectors = torch.cat([self.vectors, self.vectors[0].unsqueeze(0)], 0)
        self.vectors = torch.cat([self.vectors, self.vectors[0].unsqueeze(0)], 0)
        self.vectors[0] = torch.zeros(self.dim)
        # fields in the file
        self.fields = ['id', 'sub', 'entity', 'relation', 'obj', 'text', 'ed', 'entity_cand']
        self.tag2id = {'O': 0, 'I': 1}
        # load train, test and dev
        self.train = self.read_data('train100.txt')
        self.test = self.read_data('test100.txt')
        self.valid = self.read_data('valid100.txt')
        self.n_train = len(self.train['id'])
        self.n_test = len(self.test['id'])
        self.n_valid= len(self.valid['id'])
        self.n_words = len(self.vectors)
        self.n_rel = len(self.r2id)
        self.n_ent = int(np.max(list(self.e2id.values()))) + 1
        self.all_r = list(self.r2id.values())
        self.itos = {v: k for k, v in self.stoi.items()} # id2word dictionary
        self.train_reln = list(set([r for r in self.train['relation']]))
        # Get all kg vectors
        self.rel_emb = torch.from_numpy(np.loadtxt(args.rel_kg_vec))
        self.e_1hop = np.load(args.entity_1hop, allow_pickle=True).item()
        # Reverse labels
        self.id2r = {v: k for k, v in self.r2id.items()}
        self.id2e = {v: k for k, v in self.e2id.items()}

    def read_data(self, filename):
        """
        return output data as dictionary
        :param filename: file names ex: train.txt
        :return:
        """
        out = {}
        with open(args.data_dir+filename, 'r', encoding='utf-8') as f:
            data = f.readlines()
        data = [t.replace('\n','').split('\t') for t in data]
        for i, f in enumerate(self.fields):
            if i != 7:
                out[f] = [d[i] for d in data]

        out['entity_cand'] = [d[-(self.ent_cand_s):] for d in data]
        del (data)

        out['text'] = ([[self.get_w2id(w) for w in unidecode.unidecode(sent).split()] for sent in out['text']])
        #out['pos_emb'] = ([self.get_positional_encoding(s) for s in out['text']])
        out['ed'] = ([[self.get_t2i(id) for id in l.split()] for l in out['ed']])
        out['entity'] = [[self.get_w2id(w) for w in e.split()] for e in out['entity']]
        out['sub'] = np.array([self.get_e2id(e) for e in out['sub']])
        out['relation'] = np.array([int(self.get_r2id(r)) for r in out['relation']])
        #out['cand_labels'] = [[self.get_e2label(e) for e in es.split()] for es in out['entity_cand']]
        out['cand_labels'] = [[unidecode.unidecode(e.split('~')[1]) for e in es] for es in out['entity_cand']]
        out['entity_cand'] = [[self.get_e2id(e.split('~')[0]) for e in es] for es in out['entity_cand']]

        #out['cand_labels'] = [es.split('~')[1] for es in out['entity_cand']]
        #out['entity_cand'] = [self.get_e2id(es.split('~')[0]) for es in out['entity_cand']]
        out['cand_labels'] = [[[self.get_w2id(w) for w in label.split()] for label in labels] for labels in out['cand_labels']]
        #out['cand_labels'] = [[e.split('~')[1] for e in es.split()] for es in out['entity_cand']]
        return out

    def get_e2id(self, entity):
        #try:
        return int(self.e2id[entity])
        #except KeyError:
        #    self.ent404.write(entity+'\n')
        #    return 0

    # def get_e2label(self, entity):
    #     return self.e2label[entity]

    def get_r2id(self, relation):
        try:
            return int(self.r2id[relation])
        except KeyError:
            return int(self.r2id['oov_r'])

    def get_w2id(self, word):
        # get word2ids
        try:
            return int(self.stoi[word])
        except KeyError:
            return int(self.stoi['<unk>'])

    def get_t2i(self, l):
        # get tag to words
        try:
            return self.tag2id[l]
        except Exception:
            return self.tag2id[self.clean_str(l)]


    def clean_str(self, string):
        # clean error tags
        string = re.sub(r"\. \. \.", "\.", string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
        return string.strip()[1]

    def get_iter(self, dataset='train'):
        # get iterations.
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'valid':
            dataset = self.valid
        else:
            dataset = self.test

        for i in range(0, len(dataset['id']), self.batch_size):
            text = dataset['text'][i:i+self.batch_size]
            #pos_emb = dataset['pos_emb'][i:i+self.batch_size]
            label = dataset['ed'][i:i+self.batch_size]
            sub = dataset['sub'][i:i + self.batch_size]
            sub_l = dataset['entity'][i:i + self.batch_size]
            relation = dataset['relation'][i:i + self.batch_size]
            ent_cand = dataset['entity_cand'][i:i + self.batch_size]
            cand_labels = dataset['cand_labels'][i:i + self.batch_size]
            #cand_labels = dataset['cand_labels'][i:i + self.batch_size]
            #shuffle entity candidates
            #for c in ent_cand:
            #    random.shuffle(c)
            t, l, m, g, s, c_s, r, r_m, e_c, c_l, c1h, c1hm, c_r, c_w = self._load_batch(text, label, sub, sub_l, relation, ent_cand, cand_labels, self.batch_size)

            yield t, l, m, g, s, c_s, r, r_m, e_c, c_l, c1h, c1hm, c_r, c_w

    def _load_batch(self, text, label, sub, sub_l, rel, cand, cand_l, b_s):
        b_s = min(b_s, len(text))

        max_r_1hop = np.max([np.max([len(self.e_1hop[e]) for e in e_c]) for e_c in cand])
        #except KeyError:
            
        #    max_r_1hop = 10
        max_len = np.max([len(sent) for sent in text])
        max_len = (max_len + 1) if max_len < self.max_sent_len else self.max_sent_len
        #max_len_candl = np.max([len(l) for l in cand_l])
        text_o = np.zeros([max_len, b_s], np.int)
        #pos_emb = torch.zeros([b_s, max_len, self.dim])
        label_o = np.zeros([max_len, b_s], np.int)
        #cand_o = np.zeros([self.ent_cand_s, b_s], np.int)
        sent_mask = np.zeros([max_len, b_s], np.int)
        #cand_l = cand_l.astype(np.int32, copy=False)
        #cand_l = np.reshape(cand_l, (cand.shape[1] * b_s, ))
        cand_l_o = np.zeros([b_s, self.ent_cand_s, 8], np.int)
        correct_s = np.zeros([b_s], np.int)
        cand_1hop_r = np.zeros([b_s, self.ent_cand_s, max_r_1hop], np.int)
        cand_1hop_r_mask = np.zeros([b_s, self.ent_cand_s, max_r_1hop], np.int)
        cand_rank = np.zeros([b_s, self.ent_cand_s], np.float32)
        cand_wiki = np.zeros([b_s, self.ent_cand_s], np.float32)

        sa_gate = torch.zeros(b_s)

        reln_mask = torch.zeros([b_s, len(self.r2id)])

        #cand_one_hot = np.zeros([b_s, self.ent_cand_s], np.int)

        for j, (row_t, row_l) in enumerate(zip(text, label)):
            row_t = row_t[:max_len]
            row_l = row_l[:max_len]
            # print (row_t, len(row_t))
            text_o[:len(row_t), j] = row_t
            label_o[:len(row_l), j] = row_l
            sent_mask[:len(row_t), j] = 1
            #pos_emb[j, :len(row_t)] = pemb[j]
            #reln_mask[j][self.e_1hop[sub[j]]] = 1
            #train_reln = [r for r in self.train['relation']]
            reln_mask[j][self.train_reln] = 1

        for j, c in enumerate(cand_l):
            for k, b in enumerate(c):
                l = b[:8]
                cand_l_o[j][k][:len(l)] = l

        for j, c in enumerate(cand):
            max_len_bat = float(np.max([len(self.e_1hop[b_c]) for b_c in c]))
            probable_disambiguate = 0.0
            if sub[j] in cand[j]:
                for k, s_l in enumerate(cand_l[j]):
                    e_l = sub_l[j]
                    if s_l == e_l and cand[j][k] == sub[j]:
                        correct_s[j] = k
                    can_1hop_rs = self.e_1hop[cand[j][k]]
                    r = len(can_1hop_rs)/max_len_bat
                    cand_rank[j][k] = r
                    if cand[j][k] in self.fb2w:
                        cand_wiki[j][k] = 1.0
                    for l, c in enumerate(can_1hop_rs):
                        cand_1hop_r[j][k][l] = c
                        cand_1hop_r_mask[j][k][l] = 1.0
                    #else:
                        # can_1hop_rs = self.e_1hop[s]
                        # for l, c in enumerate(can_1hop_rs):
                        #     cand_1hop_r[j][k][l] = c
                        #     cand_1hop_r_mask[j][k][l] = 1
                        #cand_one_hot[j][k] = 1
            else:
                max_len_bat = float(np.max([len(self.e_1hop[b_c]) for b_c in c]))
                correct_s[j] = np.random.random_integers(0, (self.ent_cand_s-1))
                for k, s_l in enumerate(cand_l[j]):
                    can_1hop_rs = self.e_1hop[cand[j][k]]
                    cand_rank[j][k] = len(can_1hop_rs) / max_len_bat
                    if cand[j][k] in self.fb2w:
                        cand_wiki[j][k] = 1*0.001
                    for l, c in enumerate(can_1hop_rs):
                        cand_1hop_r[j][k][l] = c
                        cand_1hop_r_mask[j][k][l] = 1.0
            for c in cand_l[j]:
                if sub_l[j] == c:
                    probable_disambiguate += 1.0
            if probable_disambiguate > 1.0:
                sa_gate[j] = 1.0


        text_o = torch.from_numpy(text_o)
        label_o = torch.from_numpy(label_o).type(torch.FloatTensor)
        sent_mask = torch.from_numpy(sent_mask).type(torch.FloatTensor)
        cand_o = torch.LongTensor(cand)
        cand_1hop_r = torch.from_numpy(cand_1hop_r).type(torch.LongTensor)
        cand_1hop_r_mask = torch.from_numpy(cand_1hop_r_mask).type(torch.FloatTensor)
        cand_l_o = torch.from_numpy(cand_l_o).type(torch.LongTensor)
        correct_s_o = torch.from_numpy(correct_s).type(torch.LongTensor)
        cand_rank = torch.from_numpy(cand_rank).type(torch.FloatTensor)
        cand_wiki = torch.from_numpy(cand_wiki).type(torch.FloatTensor)
        subject = torch.from_numpy(sub).type(torch.LongTensor)
        relation = torch.from_numpy(rel).type(torch.LongTensor)

        if self.gpu:
            text_o, label_o, sent_mask, sa_gate = text_o.cuda(), label_o.cuda(), sent_mask.cuda(), sa_gate.cuda()
            subject, relation, reln_mask, correct_s_o, cand_o, cand_l_o, cand_1hop_r, cand_1hop_r_mask, \
                                                                             cand_rank, cand_wiki = subject.cuda(), relation.cuda(),\
                                                                             reln_mask.cuda(), correct_s_o.cuda(), cand_o.cuda(), \
                                                                             cand_l_o.cuda(), cand_1hop_r.cuda(), \
                                                                             cand_1hop_r_mask.cuda(), cand_rank.cuda(), cand_wiki.cuda()

        return text_o.long(), label_o, sent_mask, sa_gate, subject, correct_s_o, relation, reln_mask, cand_o, cand_l_o, cand_1hop_r, \
               cand_1hop_r_mask, cand_rank, cand_wiki #, cand_one_hot.cuda()
        #return text_o, label_o, sent_mask, subject, correct_s_o, relation, cand_o, cand_l_o, cand_one_hot.cuda()


if __name__ == '__main__':
    qa = SimpleQABatcher(gpu=False)
    train_iter = enumerate(qa.get_iter('train'))
    for it, mb in train_iter:
        t, l, m, s, c_s, r, r_m, c, cl, c1h, c1hm, c_rank, c_wiki = mb

    #if not args.no_tqdm:
    #    train_iter = tqdm(train_iter)
    #    train_iter.set_description_str('Training')
    #    train_iter.total = qa.n_train // qa.batch_size
