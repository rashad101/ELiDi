import torch
import torch.nn.functional as F
import numpy as np
from utils.data_reader import SimpleQABatcher
from utils.args import get_args
from utils.util import save_model, clip_gradient_threshold, load_model, get_f1, threshold, get_f1_mit_span
from models.e2e_entity_linking import E2E_entity_linker
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from utils.entity_cand_gen import get_candidates, sq_wiki_test
from utils.get_wikidata_mapping import get_wikidata_mapping, fetch_entity
import time
from utils.locs import *
args = get_args()
# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
if args.gpu:
    torch.cuda.manual_seed(args.randseed)

model_name = 'E2E_SQA_graph'
id2name = {}
# load dataset
qa = SimpleQABatcher(args.gpu, entity_cand_size=100)
model = E2E_entity_linker(num_words=qa.n_words, emb_dim=qa.dim, hidden_size=args.hidden_size, num_layers=args.num_layer,
                        emb_dropout=args.emb_drop, pretrained_emb=qa.vectors, train_embed=False, kg_emb_dim=50,
                        rel_size=qa.n_rel, ent_cand_size=qa.ent_cand_s, pretrained_rel=qa.rel_emb, dropout=args.rnn_dropout,
                        use_cuda=args.gpu)

parameter = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

# convert to cuda for gpu
if args.gpu:
    model.cuda()
    #r_weight = qa.r_weights.cuda()


def create_id2ent():
    with open("data/freebase/names.trimmed.2M.txt") as f:
        for line in tqdm(f,desc="buiding entity dictionary"):
            info = line.strip().split("\t")
            id2name[info[0]] = info[2]


def run():
    # scores for storing best model
    e_linking_val = 0.0
    span_f1 = 0.0
    reln_acc = 0.0
    for epoch in range(args.epochs):
        # validation scores
        ent_linking = []
        f1_span = []
        relation_acc = []
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')
        model.train();  optimizer.zero_grad()

        train_iter = enumerate(qa.get_iter('train'))

        if not args.no_tqdm:
            train_iter = tqdm(train_iter)
            train_iter.set_description_str('Training')
            train_iter.total = qa.n_train // qa.batch_size

        for it, mb in train_iter:
            t, l, m, g, s, c_s, r, r_m, c, cl, c1h, c1hm, c_rank, c_wiki = mb
            ent_s_p, reln_p, ent_c_p, sa_g_out, l1_soft_loss = model(t, l, m, c, cl, r, r_m, c1h, c1hm, c_rank, c_wiki)

            e_loss = F.binary_cross_entropy_with_logits(ent_s_p, l.view(-1, 1))
            r_loss = F.cross_entropy(reln_p, r)
            l_loss = F.nll_loss(ent_c_p, c_s)
            sa_loss = F.binary_cross_entropy_with_logits(sa_g_out.squeeze(), g)

            loss_tot = e_loss + l_loss + sa_loss + r_loss + l1_soft_loss
            loss_tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()

        # validation
        val_iter = enumerate(qa.get_iter('valid'))
        if not args.no_tqdm:
            val_iter = tqdm(val_iter)
            val_iter.set_description_str('Valid')
            val_iter.total = qa.n_valid // qa.batch_size
        model.eval()
        for it, mb in val_iter:
            t, l, m, g, s, c_s, r, r_m, c, cl, c1h, c1hm, c_rank, c_wiki = mb
            ent_s_p, reln_p, ent_c_p, g_amb, ent_avg_, kg_emb_cosine = model(t, l, m, c, cl, r, r_m, c1h, c1hm,
                                                                               c_rank, c_wiki, train=False)
            ent_c_p_argm = torch.argmax(ent_c_p, dim=-1)
            reln_p = torch.argmax(reln_p, dim=-1)  # Predicted relations
            if args.gpu:
                ent_link, c_s = ent_c_p_argm.data.cpu().numpy(), c_s.data.cpu().numpy()
                pred_rel, c_r = reln_p.data.cpu().numpy(), r.data.cpu().numpy()
                ent_c_p_sorted = torch.sort(ent_c_p, descending=True)[1].data.cpu().numpy()
            else:
                ent_link, c_s = ent_c_p_argm.data.numpy(), c_s.data.numpy()
                pred_rel, c_r = reln_p.data.numpy(), r.data.numpy()
                ent_c_p_sorted = torch.sort(ent_c_p, descending=True)[1].data.numpy()
            for j in range(len(c_s)):
                ent_linking.append([qa.id2e[(c[j][c_s[j].item()]).item()], qa.id2e[(c[j][ent_c_p_argm[j].item()]).item()]])
            relation_acc.append(accuracy_score(c_r, pred_rel))
            preds_mb_spans = (torch.sigmoid(ent_s_p) > threshold).view(t.size(0), t.size(1))
            p, r, f = get_f1(l, preds_mb_spans, t, qa)
            f1_span.append(f)
        e_link_acc = entity_linking_acc(ent_linking)
        if e_link_acc > e_linking_val:
            e_linking_val = e_link_acc
            span_f1 = np.average(f1_span)
            reln_acc = np.average(relation_acc)
            print('Saving model with E-linking accuracy score:' + str(e_linking_val))
            print('Entity span detection f1:' + str(span_f1))
            print('Relation detection accuracy:' + str(reln_acc))
            save_model(model, model_name)
        else:
            print('Not saving, entity linking accuracy so far:' + str(e_linking_val))
            print('Entity span detection f1:' + str(span_f1))
            print('Relation detection accuracy:' + str(reln_acc))

    #test(model)


def test(model):
    model = load_model(model, model_name, gpu=args.gpu)
    model.eval()
    ent_linking = []
    f1_span = []
    relation_acc = []
    test_iter = enumerate(qa.get_iter('test'))
    if not args.no_tqdm:
        test_iter = tqdm(test_iter)
        test_iter.set_description_str('test')
        test_iter.total = qa.n_test // qa.batch_size
    for it, mb in test_iter:
        t, l, m, g, s, c_s, r, r_m, c, cl, c1h, c1hm, c_rank, c_wiki = mb
        ent_s_p, reln_p, ent_c_p, g_amb, ent_avg_, kg_emb_cosine = model(t, l, m, c, cl, r, r_m, c1h, c1hm, c_rank, c_wiki,
                                                                           train=False)
        ent_c_p_argm = torch.argmax(ent_c_p, dim=-1)
        _, pr_idx = torch.sort(F.log_softmax(reln_p, dim=-1), descending=True)
        pr_top3 = pr_idx[:, :5]
        pr_score = _[:, :5]
        reln_p = torch.argmax(reln_p, dim=-1)
        c_rank = c_rank * c_wiki
        # print (c_s[0])
        if args.gpu:
            ent_link, c_s, c_l = ent_c_p_argm.data.cpu().numpy(), c_s.data.cpu().numpy(), cl.data.cpu().numpy()
            pred_rel, c_r = reln_p.data.cpu().numpy(), r.data.cpu().numpy()
            ent_avg_, kg_emb_cosine = ent_avg_.data.cpu().numpy(), kg_emb_cosine.data.cpu().numpy()

        else:
            ent_link, c_s, c_l = ent_c_p_argm.data.numpy(), c_s.data.numpy(), cl.data.numpy()
            pred_rel, c_r = reln_p.data.numpy(), r.data.numpy()
            ent_avg_, kg_emb_cosine = ent_avg_.data.numpy(), kg_emb_cosine.data.numpy()

        for j in range(len(c_s)):
            ent_linker_out.write(qa.id2e[(c[j][c_s[j].item()]).item()] + ', ' +
                                 qa.id2e[(c[j][ent_c_p_argm[j].item()]).item()] + '\n')
            ent_linking.append([qa.id2e[(c[j][c_s[j].item()]).item()], qa.id2e[(c[j][ent_c_p_argm[j].item()]).item()]])
        for j in range(len(c_s)):
            predicted_reln.write(str(r[j].item()) + ', ' + str(pred_rel[j]) + '\n')
        for j in range(len(c_s)):
            g_amb_f.write(str(g_amb[j].item())+'\n')
        for j in range(len(c_s)):
            ent_emb_avg_out.write(str(ent_avg_[j][c_s[j].item()]) + ', ' + str(ent_avg_[j][ent_c_p_argm[j].item()]) + '\n')
        for j in range(len(c_s)):
            cosine_distance_out.write(str(kg_emb_cosine[j][c_s[j].item()]) + ', ' + str(kg_emb_cosine[j][ent_c_p_argm[j].item()]) + '\n')
        for j in range(len(c_s)):
            for k in range(len(ent_c_p[j])):
                all_ent_score_out.write(str(ent_c_p[j][k].item())+', ')
            all_ent_score_out.write('\n')
        for j in range(len(c_s)):
            for l1, r3 in enumerate(pr_top3[j]):
                predicted_reln_top3.write(str(r3.item())+':'+str(pr_score[j][l1].item())+',')
            predicted_reln_top3.write('\n')

        for j in range(len(c_s)):
            for k in range(len(ent_avg_[j])):
                all_ent_emb_avg.write(str(ent_avg_[j][k].item()) + ', ')
            all_ent_emb_avg.write('\n')

        for j in range(len(c_s)):
            for k in range(len(kg_emb_cosine[j])):
                all_kg_cos.write(str(kg_emb_cosine[j][k].item()) + ', ')
            all_kg_cos.write('\n')
        for j in range(len(c_s)):
            for k in range(len(c_rank[j])):
                cand_rank.write(str(c_rank[j][k].item()) + ', ')
            cand_rank.write('\n')
        relation_acc.append(accuracy_score(c_r, pred_rel))
        preds_mb_spans = (torch.sigmoid(ent_s_p) > threshold).view(t.size(0), t.size(1))
        p, r, f, p_span = get_f1_mit_span(l, preds_mb_spans, t, qa)
        f1_span.append(f)
        for s in p_span:
            predicted_e_spans.write(s+'\n')
    print('Entity span F1 on test iteration = ' + str(np.average(f1_span)))
    print('Relation accuracy on test iteration = ' + str(np.average(relation_acc)))
    print('Entity linking accuracy on test iteration for top1= ' + str(entity_linking_acc(ent_linking)))


def infer(text, model, top_k_ent=1):
    """
    Infer for single text
    """
    print("Query:", text)
    text_w = torch.LongTensor([qa.get_w2id(w) for w in text.split()]).resize(1, len(text.split()))
    text_m = torch.ones(1, len(text.split())).long()
    pred_ent, pred_rel, e_label_pred = model.infer(text.split(), text_w, text_m, qa.get_w2id, qa.e2id, 100, qa.e_1hop, get_candidates)
    if e_label_pred:
        pred_ent_id = [p[0] for p in pred_ent[0]][:top_k_ent]  # sorted entities based on scores
        # print(pred_ent_id)
        pred_fb_ent = pred_ent_id[0].replace('fb:', '').replace('.', '/')
        print(pred_fb_ent, e_label_pred)
        wikidata_ent_pred = get_wikidata_mapping(pred_fb_ent)
        # print(wikidata_ent_pred)
        if wikidata_ent_pred:  # check in wikidata fb mapping
            wikidata_ent_pred = wikidata_ent_pred.split('/')[-1]
        else:  # check as wikidata
            wikidata_ent_pred = fetch_entity(e_label_pred)
            if wikidata_ent_pred:
                wikidata_ent_pred = wikidata_ent_pred.split('/')[-1]
            else:
                wikidata_ent_pred = pred_fb_ent
        print (wikidata_ent_pred)
        return wikidata_ent_pred, e_label_pred, pred_fb_ent
    else:
        return '', '', ''


def entity_linking_score(correct_ent, candidates, cand_pred, topk=1):
    batch_s = correct_ent.shape[0]
    top_val = []
    for i, e in enumerate(correct_ent):
        pred_e_cand = [candidates[i][n].item() for n  in cand_pred[i]]
        pred_e_cand = get_unique(pred_e_cand)
        if e.item() in pred_e_cand[:topk]:
            top_val.append(e.item())
    return len(top_val)/float(batch_s)


def entity_linking_acc(batch_cand):
    tpentity = 0
    fpentity = 0
    fnentity = 0
    totalentchunks = 0
    count=0
    for i,b in enumerate(batch_cand):
        corr, pred = b
        try:
            corr,pred = id2name[corr].split(),id2name[pred].split()
            for goldentity in corr:
                totalentchunks += 1
                if goldentity in pred:
                    tpentity += 1
                else:
                    fnentity += 1
            for queryentity in pred:
                if queryentity not in corr:
                    fpentity += 1
            count+=1
        except:
            print(corr ," or ", pred, " not found")
            continue

    precisionentity = tpentity/float(tpentity+fpentity)
    recallentity = tpentity/float(tpentity+fnentity)
    f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
    print("precision entity = ",precisionentity)
    print("recall entity = ",recallentity)
    print("f1 entity = ",f1entity)

    return f1entity


def get_unique(pred_cands):
    used = set()
    unique = [x for x in pred_cands if x not in used and (used.add(x) or True)]
    return unique


if __name__ == '__main__':
    # run()
    # create_id2ent()
    # test(model)

    model = load_model(model, model_name, gpu=args.gpu)
    model.eval()
    data = 'sq'
    print ('Loaded the model.....')
    no_ent = []
    sq_pred = []
    # infer('what is a short-lived british sitcom series', model)
    start_time = time.time()  # Get timing
    if data == 'sq':
        # evaluate sq wikidata
        for d in sq_wiki_test:
            g_sub = d[0]
            query = d[3].replace('\'', ' \'').replace('?', '').lower()
            pred_sub, e_span, fb_ent = infer(query, model)
            if e_span:
                if pred_sub == g_sub:
                    sq_pred.append(1)
                else:
                    sq_pred.append(0)
            else:
                no_ent.append(query)
                sq_pred.append(0)
            ent_linker_out.write(g_sub + '\t' + query + '\t' + e_span + '\t' + pred_sub + '\n')
    else:
        # evaluate on wikidata
        for d in sq_wiki_test:
            g_sub = d['main_entity']
            query = d['utterance'].replace('\'', ' \'').replace('?', '').lower()
            pred_sub, e_span, fb_ent = infer(query, model)
            if e_span:
                if pred_sub == g_sub:
                    sq_pred.append(1)
                else:
                    sq_pred.append(0)
            else:
                no_ent.append(query)
                sq_pred.append(0)
            ent_linker_out.write(g_sub + '\t' + query + '\t' + e_span + '\t' + pred_sub + '\n')


    # Close all files
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Total size:', str(len(sq_wiki_test)))
    print (np.average(sq_pred))
    ent_linker_out.close()
    ent_emb_avg_out.close()
    cosine_distance_out.close()
    all_ent_emb_avg.close()
    all_kg_cos.close()
    cand_rank.close()
    predicted_reln_top3.close()
    all_ent_score_out.close()
    predicted_e_spans.close()
