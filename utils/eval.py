import json
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Evaluation")
    parser.add_argument('--filepath', type=str, required=True)

def eval_dataset(filepath):
    data = json.load(open(filepath))
    tpentity = 0
    fpentity = 0
    fnentity = 0
    for d in data:
        queryentities = [ent for ent in [d["pred_ent"]] if ent is not None]
        if None in set(d['gold_ents']):
            print('skip none')
            continue
        for goldentity in set(d['gold_ents']):
            if goldentity == None:
                print("skip none")
                continue
            if goldentity in queryentities:
                tpentity += 1
            else:
                fnentity += 1
        for queryentity in set(queryentities):
            if queryentity not in d['gold_ents']:
                fpentity += 1


    precisionentity = tpentity/float(tpentity+fpentity)
    recallentity = tpentity/float(tpentity+fnentity)
    f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
    print("precision entity = ",precisionentity)
    print("recall entity = ",recallentity)
    print("f1 entity = ",f1entity)
    return f1entity, precisionentity, recallentity

if __name__=="__main__":
    args = get_args()
    f1,p,r = eval_dataset(args.file_path)