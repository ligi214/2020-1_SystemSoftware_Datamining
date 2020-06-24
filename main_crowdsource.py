from dataloader import *
from utils import *
from metrics import *
import numpy as np

# hyperparameter settings
alpha = (3, 3, 2)
beta = (2, 2, 2)
gamma = 2
iteration = 50
num_workers = 10
k = 5  # the number of questions to be asked in each round
pi = 0.75

# Data processing
records, groundtruths, hierarchy = get_heritage()
src_info, obj_info = records_processing(records)
ancestors, descendants = hierarchy_processing(hierarchy)
workers = np.arange(0, 10)
worker_info = dict()

# Get gold standard from groundtruths
gold_standards = dict()
for obj in obj_info.keys():
    candidate = obj_info[obj]['Vo']
    groundtruth = groundtruths[obj]
    gold_standards[obj] = None
    for truth in groundtruth:
        if truth in candidate:
            gold_standards[obj] = truth
            break

# Modify groundtruths
for obj in groundtruths.keys():
    try:
        idx = groundtruths[obj].index(gold_standards[obj])
    except ValueError:
        idx = 0
    groundtruths[obj] = groundtruths[obj][idx:]

# Initialize parameters
phi, psi, mu = dict(), dict(), dict()
for src in src_info.keys():
    phi[src] = np.random.dirichlet(alpha=alpha)
for i in range(num_workers):
    psi[i] = np.random.dirichlet(alpha=beta)
    worker_info[i] = dict()
for obj in obj_info.keys():
    mu[obj] = dict()
    dist = np.random.dirichlet(alpha=[gamma]*len(obj_info[obj]['Vo']))
    for i, v in enumerate(obj_info[obj]['Vo']):
        mu[obj][v] = dist[i]
worker_correct_prob = [np.random.uniform(pi-0.05, pi+0.05) for _ in range(num_workers)]

# Hidden Variables: f_os^v, f_ow^v, g_os^t, g_ow^t
f_os, f_ow, g_os, g_ow = dict(), dict(), dict(), dict()

# EM algorithm
for i in range(iteration):
    print('Step {}'.format(i))
    # E Step
    f_os = get_f_os(phi, mu, obj_info, src_info, ancestors)
    f_ow = get_f_ow(psi, mu, obj_info, worker_info, src_info, ancestors)
    g_os = get_g_os(phi, mu, obj_info, src_info, ancestors)
    g_ow = get_g_ow(psi, mu, obj_info, worker_info, src_info, ancestors)
    # M Step
    mu = get_mu(f_os, f_ow, gamma, obj_info)
    phi = get_phi(g_os, alpha, src_info)
    if i > 0:
        psi = get_psi(g_ow, beta, worker_info)
    # Task assignment at every round
    U_EAI = get_U_EAI(mu, obj_info, gamma)
    tasks = task_assignment(U_EAI, psi, k, mu, f_os, f_ow, obj_info, src_info, ancestors, gamma)
    worker_answer(tasks, gold_standards, groundtruths, obj_info, worker_info, worker_correct_prob)

# inference
ans = inference(obj_info, mu)
print(f_os)
print(f_ow)
print(g_os)
print(g_ow)
print(mu)
print(psi)
print(phi)
print(worker_info)

# print answer
print(ans)

# Accuracy evaluation
acc = accuracy(ans, gold_standards)
gen_acc = gen_accuracy(ans, groundtruths)
avg_dist = avg_distance(ans, groundtruths, ancestors, descendants)
print(acc, gen_acc, avg_dist)


exit(0)

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
print(accuracy(vote_ans, gold_standards))
