from dataloader import *
from utils import *
from metrics import *
import numpy as np

# hyperparameter settings
alpha = (3, 3, 2)
beta = 2
gamma = 2
iteration = 50

# Data processing
records, groundtruths, hierarchy = get_heritage()
src_info, obj_info = records_processing(records)
ancestors, descendants = hierarchy_processing(hierarchy)

# Get gold standard from groundtruths
gold_standard = dict()
for obj in obj_info.keys():
    candidate = obj_info[obj]['Vo']
    groundtruth = groundtruths[obj]
    gold_standard[obj] = None
    for truth in groundtruth:
        if truth in candidate:
            gold_standard[obj] = truth
            break

# Initialize parameters
phi, psi, mu = dict(), dict(), dict()
for src in src_info.keys():
    phi[src] = np.random.dirichlet(alpha=alpha)
for obj in obj_info.keys():
    mu[obj] = dict()
    dist = np.random.dirichlet(alpha=[gamma]*len(obj_info[obj]['Vo']))
    for i, v in enumerate(obj_info[obj]['Vo']):
        mu[obj][v] = dist[i]

# Hidden Variables: f_os^v, f_ow^v, g_os^t, g_ow^t
f_os, f_ow, g_os, g_ow = dict(), dict(), dict(), dict()

# EM algorithm
for i in range(iteration):
    # E Step
    f_os = get_f_os(phi, mu, obj_info, src_info, ancestors)
    g_os = get_g_os(phi, mu, obj_info, src_info, ancestors, descendants)
    # M Step
    mu = get_mu(f_os, f_ow, gamma, obj_info)
    phi = get_phi(g_os, alpha, src_info)
# inference
ans = inference(obj_info, mu)
print(f_os)
print(g_os)
print(mu)
print(phi)

# print answer
print(ans)
goldstandards = dict()
for obj in groundtruths:
    goldstandards[obj] = groundtruths[obj][0]
print(goldstandards)

# Accuracy evaluation
acc = accuracy(ans, gold_standard)
gen_acc = gen_accuracy(ans, groundtruths)
avg_dist = avg_distance(ans, gold_standard, groundtruths, ancestors, descendants)
print(acc, gen_acc, avg_dist)

# Baselines: VOTE
vote_obj = dict()
for record in records:
    if record['obj'] not in vote_obj.keys():
        vote_obj[record['obj']] = dict()
    if record['value'] not in vote_obj[record['obj']].keys():
        vote_obj[record['obj']][record['value']] = 0
    vote_obj[record['obj']][record['value']] += 1
vote_ans = dict()
for obj in vote_obj.keys():
    max_val = None
    for val in vote_obj[obj].keys():
        if max_val == None or (max_val in vote_obj[obj].keys() and vote_obj[obj][max_val] < vote_obj[obj][val]):
            max_val = val
    vote_ans[obj] = max_val

print('========== VOTE ==========')
print(vote_ans)
print(accuracy(vote_ans, gold_standard))
