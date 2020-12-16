# <snippet_imports>
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from tqdm import tqdm
import json
import re
import torch
# </snippet_imports>
# Created by Debanjan Chaudhuri at 12/12/2020

websqp_question_f = 'data/processed_simplequestions_dataset/webqsp/webqsp_wd-test.json'  # webqsp dataset
sq_wikidata_f = 'data/processed_simplequestions_dataset/sq_wikidata.txt'
fb2m_entities_f = 'data/freebase/names.trimmed.2M.txt'  # names of 2M entities

# load the question
with open(websqp_question_f, 'r', encoding='utf-8') as f:
    websqp_questions = json.load(f)
with open(sq_wikidata_f, 'r', encoding='utf-8') as f:
    sq_w_dat = f.readlines()
questions = []
sq_wiki_test = []


# def _load_questions(data='webqsp'):

for q in websqp_questions:
    questions.append(q['utterance'].replace('\'', ' \'').replace('?', '').lower().split())
for d in sq_w_dat:
    d = d.replace('\n', '').split('\t')
    questions.append(d[3].replace('\'', ' \'').replace('?', '').lower().split())
    sq_wiki_test.append(d)


vocab_dict = Dictionary(questions)  # vocab dictionary


def load_data():
    """
    creating entity mapping
    """
    # logger.info("Creating entity to list")
    ent2name_list = list()
    with open(fb2m_entities_f, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin, desc="2M names"):
            line = line.strip().split('\t')
            ent2name_list.append((line[0].replace('<', '').replace('>', ''), re.sub('[^a-zA-Z0-9\n\.]', ' ', line[2])))
    return ent2name_list


fb2m_entities = load_data()  # load a 2M entity dataset

corpus = [vocab_dict.doc2bow(item[1].split()) for item in tqdm(fb2m_entities, desc="building corpus")]
tfidf = TfidfModel(corpus)  # tfidf from corpus
entity_matrix = SparseMatrixSimilarity(corpus=tfidf[corpus],
                                       num_features=len(vocab_dict.items()))  # create sparse matrix


def get_candidates(entity_span, t2id, topk, e2id, e1hop, device, max_cand_l=5, max_cand_r=8):
    """
    Return candidates based on entity span
    entity_span: Detected entity span
    max_cand_l: maximum candidate label length
    """

    # id2t = {v: k for k, v in t2id.items()}
    ent_cand_wid = torch.zeros(1, topk, max_cand_l).long().to(device)
    ent_1hop_reln = torch.zeros(1, topk, max_cand_r).long().to(device)
    # entity_span_w = [id2t[w.item()] for w in entity_span]  # convert to word
    q_entity_tfidf = tfidf[vocab_dict.doc2bow(entity_span.split())]  # question tfidf vec
    sims = enumerate(entity_matrix.get_similarities(q_entity_tfidf))  # Get the entity similarity
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:topk]
    ent_cand = ['~'.join(fb2m_entities[c[0]]) for c in sims]
    ent_cand_text = [e.split('~')[1] for e in ent_cand]  # get the candidate label
    ent_cand_id = [e.split('~')[0] for e in ent_cand]  # get the candidate label
    # ent_cand_wid = [[t2id(w) for w in e_t.split()] for e_t in ent_cand_text]  # entity candidate
    for j, e_t in enumerate(ent_cand_text):
        cand_text = [t2id(w) for w in e_t.split()[:max_cand_l]]
        ent_cand_wid[0][j][:len(cand_text)] = torch.LongTensor(cand_text[:max_cand_l])
        cand_reln = e1hop[e2id[ent_cand_id[j]]][:max_cand_r]
        ent_1hop_reln[0][j][:len(cand_reln)] = torch.LongTensor(cand_reln)
    # ent_1hop_reln = [e1hop[e2id[e]] for e in ent_cand_id]
    return ent_cand_wid, ent_cand_id, ent_1hop_reln

