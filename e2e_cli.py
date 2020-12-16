import torch
import numpy as np
from argparse import ArgumentParser
import json
import pickle
from models.e2e_entity_linking import E2E_entity_linker
from utils.args import get_args
from utils.entity_cand_gen import get_candidates
from utils.get_wikidata_mapping import get_wikidata_mapping, fetch_entity
from utils.util import load_model
import warnings
warnings.filterwarnings("ignore")

args = get_args()

def generate_ngrams(s, n=[1, 2, 3, 4]):
    words_list = s.split()
    ngrams_list = []

    for num in range(0, len(words_list)):
        for l in n:
            ngram = ' '.join(words_list[num:num + l])
            ngrams_list.append(ngram)
    ngrams_list.sort(key=lambda x: len(x),reverse=True)
    return ngrams_list

def get_w2id(word,stoi):
    # get word2ids
    try:
        return int(stoi[word])
    except KeyError:
        return int(stoi['<unk>'])


def infer(text, model, e2id,e_1hop, stoi, top_k_ent=1):
    """
    Infer for single text
    """
    text_w = torch.LongTensor([get_w2id(w,stoi) for w in text.split()]).resize(1, len(text.split()))
    text_m = torch.ones(1, len(text.split())).long()
    pred_ent, pred_rel, e_label_pred = model.infer(text.split(), text_w, text_m, get_w2id, e2id, 100, e_1hop, get_candidates)
    if e_label_pred:
        pred_ent_id = [p[0] for p in pred_ent[0]][:top_k_ent]  # sorted entities based on scores
        # print(pred_ent_id)
        pred_fb_ent = pred_ent_id[0].replace('fb:', '').replace('.', '/')
        wikidata_ent_pred = get_wikidata_mapping(pred_fb_ent)
        # print(wikidata_ent_pred)
        if wikidata_ent_pred:  # check in wikidata fb mapping
            wikidata_ent_pred = wikidata_ent_pred.split('/')[-1]
        else:  # check as wikidata
            wikidata_ent_pred = fetch_entity(e_label_pred)
            if wikidata_ent_pred:
                wikidata_ent_pred = wikidata_ent_pred.split('/')[-1]
            else:
                ngrams = generate_ngrams(e_label_pred)
                found = False
                for agram in ngrams:
                    result = fetch_entity(agram)
                    if result:
                        wikidata_ent_pred = result.split('/')[-1]
                        found = True
                        break
                if not found:
                    wikidata_ent_pred = 'Not found'
        return wikidata_ent_pred, e_label_pred, pred_fb_ent
    else:
        return 'Not found', 'Not found', 'Not found'


def interact(model, e2id=None, e_1hop=None, stoi=None):
    question = input("Please type your question (type q to quit):  ")
    if question!="q":
        if question!="":
            wikiid,elabel,predfb = infer(question,model, e2id=e2id, e_1hop=e_1hop, stoi=stoi)
            return wikiid, elabel
        else:
            print("You have entered nothing !!")
            return "",""
    else:
        return "q",""


if __name__=="__main__":
    # Set random seed
    np.random.seed(args.randseed)
    torch.manual_seed(args.randseed)
    if args.gpu:
        torch.cuda.manual_seed(args.randseed)

    model_name = 'E2E_SQA_graph'

    # load dataset
    print("Loading model... ")

    # loading vectors and variables
    with open(args.rel2id_f, 'rb') as f:
        r2id = json.load(f)
    with open(args.entity2id_f, 'rb') as f:
        e2id = json.load(f)
    with open(args.entity_fb2w_map, 'rb') as f:
        fb2w = pickle.load(f)

    stoi, vectors, dim = torch.load(args.vector_cache)
    stoi['<unk>'] = len(stoi)
    stoi['<pad>'] = 0  # add padding index and remove comma to another index
    stoi[','] = len(stoi)

    e_1hop = np.load(args.entity_1hop, allow_pickle=True).item()

    vectors = torch.cat([vectors, torch.FloatTensor(dim).uniform_(-0.25, 0.25).unsqueeze(0)], 0)
    vectors = torch.cat([vectors, vectors[0].unsqueeze(0)], 0)
    vectors = torch.cat([vectors, vectors[0].unsqueeze(0)], 0)
    vectors[0] = torch.zeros(dim)

    n_words = len(vectors)
    n_rel = len(r2id)
    rel_emb = torch.from_numpy(np.loadtxt(args.rel_kg_vec))
    ent_cand_s = 100


    model = E2E_entity_linker(num_words=n_words, emb_dim=dim, hidden_size=args.hidden_size, num_layers=args.num_layer,
                            emb_dropout=args.emb_drop, pretrained_emb=vectors, train_embed=False, kg_emb_dim=50,
                            rel_size=n_rel, ent_cand_size=ent_cand_s, pretrained_rel=rel_emb, dropout=args.rnn_dropout,
                            use_cuda=args.gpu)

    model = load_model(model, model_name, gpu=args.gpu)
    model.eval()
    print("Done !!")
    while True:
        wiki_id, ent_label = interact(model, e2id=e2id, e_1hop=e_1hop, stoi=stoi)
        if wiki_id=="q":
            exit()
        elif wiki_id=="":
            continue
        else:
            print(f"Wiki entity ID: {wiki_id}\nWiki entity label: {ent_label}\n")




