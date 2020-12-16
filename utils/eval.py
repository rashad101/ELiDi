import json

data = json.load(open("predictions/entity_linker_sq-wd.json"))

tpentity = 0
fpentity = 0
fnentity = 0
for d in data:
    queryentities = [d["pred_ent"]]
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