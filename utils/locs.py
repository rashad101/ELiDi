ent_linker_out = open('predictions/entity_linker.txt', 'w', encoding='utf-8')
ent_emb_avg_out = open('predictions/entity_emb_avg.txt', 'w')
cosine_distance_out = open('predictions/cos_out.txt', 'w')
predicted_reln = open('predictions/pred_relations.txt', 'w')
predicted_reln_top3 = open('predictions/pred_relations_top3.txt', 'w')
all_ent_emb_avg = open('predictions/ent_avg_all.txt', 'w')
all_kg_cos = open('predictions/kg_cosine_all.txt', 'w')
cand_rank = open('predictions/cand_rank.txt', 'w')
g_amb_f = open('predictions/g_amb.txt', 'w')
all_ent_score_out = open('predictions/all_ent_cand_out.txt', 'w')
predicted_e_spans = open('predictions/predicted_spans.txt', 'w')