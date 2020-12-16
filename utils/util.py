import os
import torch
from utils.args import get_args
args = get_args()

threshold = torch.tensor([args.threshold])
if args.gpu:
    threshold = threshold.cuda()
def save_model(model, name):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/{}.bin'.format(name))


def load_model(model, name, gpu=True):
    if gpu:
        model.load_state_dict(torch.load('models/{}.bin'.format(name)))
    else:
        model.load_state_dict(torch.load('models/{}.bin'.format(name), map_location=lambda storage, loc: storage))

    return model


def clip_gradient_threshold(model, min, max):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(min, max)


def get_span(batch_t, batch_l, qa):
    # get spans from labels and original text
    batch_t = torch.t(batch_t)
    batch_l = torch.t(batch_l)
    spans = []
    for i, b in enumerate(batch_l):
        if args.gpu:
            ent = batch_t[i] * b.type(torch.LongTensor).cuda()
        else:
            ent = batch_t[i] * b.type(torch.LongTensor)
        ent = [qa.itos[int(t.data.cpu().numpy())] for t in ent if t != 0]
        spans.append(' '.join(e for e in ent))
    return spans

def get_char_span(batch_char, batch_tag, batcher_obj):
    # get spans from labels and original text
    batch_char = torch.t(batch_char)
    batch_tag = torch.t(batch_tag)
    spans = []
    for i, example in enumerate(batch_tag):
        ent = batch_char[i] * example.type(batch_char.type())
        ent = [batcher_obj.itos[t.item()] for t in ent if t != 0]
        spans.append(''.join(ent))
    return spans

def get_char_f1(gold_batch, pred_batch, text_b, qa):
    #get f1 scores
    right = 0
    gold_span = get_char_span(text_b, gold_batch, qa)
    pred_span = get_char_span(text_b, pred_batch, qa)

    #print('PREDS: {}|\n|GOLD: {}|\n|TAGS: {}'.format(pred_span, gold_span, torch.t(gold_batch)[0]))

    total_en = len(gold_span)
    predicted = len(pred_span)
    for item in pred_span:
        if item in gold_span:
            right += 1
    if predicted == 0:
        precision = 0
    else:
        precision = right / predicted
    if total_en == 0:
        recall = 0
    else:
        recall = right / total_en
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    #fout.flush()
    #fout.close()
    return precision, recall, f1

def get_f1(gold_batch, pred_batch, text_b, qa):
    #get f1 scores
    right = 0
    gold_span = get_span(text_b, gold_batch, qa)
    pred_span = get_span(text_b, pred_batch, qa)

    total_en = len(gold_span)
    predicted = len(pred_span)
    for item in pred_span:
        if item in gold_span:
            right += 1
    if predicted == 0:
        precision = 0
    else:
        precision = right / predicted
    if total_en == 0:
        recall = 0
    else:
        recall = right / total_en
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    #fout.flush()
    #fout.close()
    return precision, recall, f1

def get_f1_mit_span(gold_batch, pred_batch, text_b, qa):
    #get f1 scores
    right = 0
    gold_span = get_span(text_b, gold_batch, qa)
    pred_span = get_span(text_b, pred_batch, qa)

    total_en = len(gold_span)
    predicted = len(pred_span)
    for item in pred_span:
        if item in gold_span:
            right += 1
    if predicted == 0:
        precision = 0
    else:
        precision = right / predicted
    if total_en == 0:
        recall = 0
    else:
        recall = right / total_en
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    #fout.flush()
    #fout.close()
    return precision, recall, f1, pred_span
