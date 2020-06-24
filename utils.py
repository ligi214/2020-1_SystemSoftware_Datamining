import heapq
import random


def get_prob_vos(vos, vo_star, phi, candidate, ancestors):
    # phi : 3-dim vector
    # candidate : vo
    go_vo_star = []
    ancestor = ancestors[vo_star]
    for v in candidate:
        if v in ancestor:
            go_vo_star.append(v)

    flag_h = False
    for v in candidate:
        for w in candidate:
            if w in ancestors[v]:
                flag_h = True

    if flag_h:
        if vos == vo_star:
            return phi[0]
        elif vos in ancestor:
            return phi[1] / len(go_vo_star)
        else:
            return phi[2] / (len(candidate) - len(go_vo_star) - 1)
    else:
        if vos == vo_star:
            return phi[0] + phi[1]
        else:
            return phi[2] / (len(candidate) - 1)


def get_prob_vow(vow, vo_star, psi, candidate, obj, so, src_info, ancestors):
    go_vo_star = []
    ancestor = ancestors[vo_star]
    for v in candidate:
        if v in ancestor:
            go_vo_star.append(v)

    flag_h = False
    for v in candidate:
        for w in candidate:
            if w in ancestors[v]:
                flag_h = True

    if flag_h:
        if vow == vo_star:
            return psi[0]
        elif vow in ancestor:
            denominator = 0
            numerator = 0
            for src in so:
                if src_info[src][obj] in go_vo_star:
                    denominator += 1
                if src_info[src][obj] == vow:
                    numerator += 1
            return psi[1] * numerator / denominator
        else:
            denominator = 0
            numerator = 0
            for src in so:
                if src_info[src][obj] not in go_vo_star and src_info[src][obj] != vo_star:
                    denominator += 1
                if src_info[src][obj] == vow:
                    numerator += 1
            return psi[2] * numerator / denominator
    else:
        if vow == vo_star:
            return psi[0] + psi[1]
        else:
            denominator = 0
            numerator = 0
            for src in so:
                if src_info[src][obj] not in go_vo_star and src_info[src][obj] != vo_star:
                    denominator += 1
                if src_info[src][obj] == vow:
                    numerator += 1
            return psi[2] * numerator / denominator


def get_f_os(phi, mu, obj_info, src_info, ancestor):
    f_os = dict()
    for obj in obj_info.keys():
        f_os[obj] = dict()
        for src in obj_info[obj]['So']:
            vos = src_info[src][obj]
            f_os[obj][src] = dict()

            # evaluate denominator
            denominator = 0
            for v in obj_info[obj]['Vo']:
                denominator = denominator + get_prob_vos(vos, v, phi[src], obj_info[obj]['Vo'], ancestor) * mu[obj][v]

            # final calculation
            for v in obj_info[obj]['Vo']:
                numerator = get_prob_vos(vos, v, phi[src], obj_info[obj]['Vo'], ancestor) * mu[obj][v]
                f_os[obj][src][v] = numerator / denominator
    return f_os


def get_f_ow(psi, mu, obj_info, worker_info, src_info, ancestor):
    f_ow = dict()
    for obj in obj_info.keys():
        f_ow[obj] = dict()
        for worker in obj_info[obj]['Wo']:
            vow = worker_info[worker][obj]
            f_ow[obj][worker] = dict()

            # evaluate denominator
            denominator = 0
            for v in obj_info[obj]['Vo']:
                denominator = denominator + get_prob_vow(vow, v, psi[worker], obj_info[obj]['Vo'], obj,
                                                         obj_info[obj]['So'], src_info, ancestor) * mu[obj][v]
            for v in obj_info[obj]['Vo']:
                numerator = get_prob_vow(vow, v, psi[worker], obj_info[obj]['Vo'], obj,
                                         obj_info[obj]['So'], src_info, ancestor) * mu[obj][v]
                f_ow[obj][worker][v] = numerator / denominator
    return f_ow


def get_g_os(phi, mu, obj_info, src_info, ancestor):
    g_os = dict()
    for obj in obj_info.keys():
        g_os[obj] = dict()
        for src in obj_info[obj]['So']:
            g_os[obj][src] = [0, 0, 0]
            vos = src_info[src][obj]
            candidate = obj_info[obj]['Vo']
            denominator = 0

            do_vos = []
            for v in candidate:
                if vos in ancestor[v]:
                    do_vos.append(v)

            flag_h = False
            for v in candidate:
                for w in candidate:
                    if v in ancestor[w]:
                        flag_h = True

            for v in candidate:
                denominator += (get_prob_vos(vos, v, phi[src], candidate, ancestor) * mu[obj][v])

            # 1
            g_os[obj][src][0] = phi[src][0] * mu[obj][vos] / denominator

            # 2
            if flag_h:
                numerator = 0
                for v in do_vos:
                    numerator += (get_prob_vos(vos, v, phi[src], candidate, ancestor) * mu[obj][v])
                g_os[obj][src][1] = numerator / denominator
            else:
                g_os[obj][src][1] = phi[src][1] * mu[obj][vos] / denominator

            # 3
            numerator = 0
            for v in candidate:
                if v != vos and v not in do_vos:
                    numerator += (get_prob_vos(vos, v, phi[src], candidate, ancestor) * mu[obj][v])
            g_os[obj][src][2] = numerator / denominator

            if 0.999 < sum(g_os[obj][src]) < 1.001:
                continue
            else:
                # for debugging
                print('obj:{}, src:{}, vos:{}, flag:{}'.format(obj, src, vos, flag_h))
    return g_os


def get_g_ow(psi, mu, obj_info, worker_info, src_info, ancestor):
    g_ow = dict()
    for obj in obj_info.keys():
        g_ow[obj] = dict()
        for worker in obj_info[obj]['Wo']:
            g_ow[obj][worker] = [0, 0, 0]
            vow = worker_info[worker][obj]
            candidate = obj_info[obj]['Vo']
            denominator = 0

            do_vow = []
            for v in candidate:
                if vow in ancestor[v]:
                    do_vow.append(v)

            flag_h = False
            for v in candidate:
                for w in candidate:
                    if v in ancestor[w]:
                        flag_h = True

            for v in candidate:
                denominator += (get_prob_vow(vow, v, psi[worker], candidate, obj, obj_info[obj]['So'], src_info, ancestor) * mu[obj][v])

            # 1
            g_ow[obj][worker][0] = psi[worker][0] * mu[obj][vow] / denominator
            #print('1: ', psi[worker][0], mu[obj][vow])
            #print()

            # 2
            if flag_h:
                numerator = 0
                for v in do_vow:
                    numerator += (get_prob_vow(vow, v, psi[worker], candidate, obj, obj_info[obj]['So'], src_info, ancestor) * mu[obj][v])
                g_ow[obj][worker][1] = numerator / denominator
            else:
                g_ow[obj][worker][1] = psi[worker][1] * mu[obj][vow] / denominator

            # 3
            numerator = 0
            for v in candidate:
                if v != vow and v not in do_vow:
                    numerator += (get_prob_vow(vow, v, psi[worker], candidate, obj, obj_info[obj]['So'], src_info, ancestor) * mu[obj][v])
            g_ow[obj][worker][2] = numerator / denominator

            if 0.999 < sum(g_ow[obj][worker]) < 1.001:
                continue
            else:
                # for debugging
                print('obj:{}, worker:{}, vow:{}, flag:{}'.format(obj, worker, vow, flag_h))
    return g_ow


def get_mu(f_os, f_ow, gamma, obj_info):
    mu = dict()
    for obj in obj_info.keys():
        mu[obj] = dict()
        val_sum = 0
        for _ in obj_info[obj]['Vo']:
            val_sum += (gamma - 1)
        for val in obj_info[obj]['Vo']:
            numerator = 0
            for src in obj_info[obj]['So']:
                numerator += f_os[obj][src][val]
            for worker in obj_info[obj]['Wo']:
                numerator += f_ow[obj][worker][val]
            numerator = numerator + gamma - 1

            denominator = len(obj_info[obj]['So']) + len(obj_info[obj]['Wo']) + val_sum
            mu[obj][val] = numerator / denominator
    return mu


def get_phi(g_os, alpha, src_info):
    phi = dict()
    for src in src_info.keys():
        phi[src] = [None] * 3
        denominator = len(src_info[src]) + alpha[0] + alpha[1] + alpha[2] - 3
        for t in range(3):
            numerator = 0
            for obj in src_info[src].keys():
                numerator += g_os[obj][src][t]
            numerator = numerator + alpha[t] - 1
            phi[src][t] = numerator / denominator
    return phi


def get_psi(g_ow, beta, worker_info):
    psi = dict()
    for worker in worker_info.keys():
        psi[worker] = [None] * 3
        denominator = len(worker_info[worker]) + sum(beta) - 3
        for t in range(3):
            numerator = 0
            for obj in worker_info[worker].keys():
                numerator += g_ow[obj][worker][t]
            numerator = numerator + beta[t] - 1
            psi[worker][t] = numerator / denominator
    return psi


def get_U_EAI(mu, obj_info, gamma):
    U_EAI = dict()
    for obj in obj_info.keys():
        denominator = len(obj_info[obj]['So']) + len(obj_info[obj]['Wo'])
        for _ in obj_info[obj]['Vo']:
            denominator += (gamma - 1)
        denominator = len(obj_info.keys()) * (denominator + 1)

        max_mu = max(mu[obj].values())
        U_EAI[obj] = (1 - max_mu) / denominator
    return U_EAI


def inference(obj_info, mu):
    inferred_val = dict()
    for obj in obj_info.keys():
        ans_val = obj_info[obj]['Vo'][0]
        for v in obj_info[obj]['Vo']:
            if mu[obj][ans_val] < mu[obj][v]:
                ans_val = v
        inferred_val[obj] = ans_val
    return inferred_val


def task_assignment(U_EAI, psi, k, mu, f_os, f_ow, obj_info, src_info, ancestor, gamma):
    workers = psi.keys()
    num_workers = len(workers)
    # Build max heap
    h_UB = [(-U_EAI[obj], obj) for obj in U_EAI.keys()]
    heapq.heapify(h_UB)  # hold -EAI for max_heap

    # Sort workers based on psi[:,0]
    sorted_workers = sorted(workers, key=lambda kk: -psi[kk][0])

    h_EAI = dict()
    for worker in workers:
        h_EAI[worker] = []
        heapq.heapify(h_EAI[worker])

    w = 0
    while True:
        # extract max from max heap and set as <U_EAI, o>
        try:
            _, o = heapq.heappop(h_UB)
        except IndexError:
            print('index error')
            break

        if len(h_EAI[num_workers - 1]) == k and len(h_EAI) > 0 and all(h_EAI[w][0][0] > U_EAI[o] for w in sorted_workers):
            break
        for worker in sorted_workers:
            w = worker
            if worker in obj_info[o]['Wo'] or (len(h_EAI[worker]) > 0 and h_EAI[worker][0][0] > U_EAI[o]):
                continue
            # compute EAI(w,o) --> how? EQ14
            eai_w_o = get_eai_w_o(worker, o, mu, psi[w], f_os[o], f_ow[o], obj_info, src_info, ancestor, gamma)
            '''
            if eai_w_o < 1e-18:
                print('object: {}, worker: {}, eai_w_o: {}, mu: {}, psi: {}, f_os: {}, f_ow: {}'.format(o, worker, eai_w_o, mu[o], psi[w], f_os[o], f_ow[o]))
                '''
            heapq.heappush(h_EAI[worker], (eai_w_o, o))
            if len(h_EAI[worker]) <= k:
                break
            _, o = heapq.heappop(h_EAI[worker])
    print(h_EAI)

    tasks = dict()
    for worker in workers:
        tasks[worker] = [t[1] for t in h_EAI[worker]]
    return tasks


def get_eai_w_o(w, o, mu, psi, f_os, f_ow, obj_info, src_info, ancestor, gamma):
    expectation = 0
    prob_sum = 0
    for v_ in obj_info[o]['Vo']:
        prob = 0
        max_mu = 0
        for v in obj_info[o]['Vo']:
            prob += get_prob_vow(v_, v, psi, obj_info[o]['Vo'], o, obj_info[o]['So'], src_info, ancestor) * mu[o][v]
            cond_mu = get_mu_o_v_cond(o, v, v_, mu, psi, f_os, f_ow, obj_info[o]['Vo'], obj_info[o]['So'],
                                      obj_info[o]['Wo'], src_info, ancestor, gamma)
            if max_mu < cond_mu:
                max_mu = cond_mu
        # get mu o v vow conditional and get the max of it and multiply the two of them
        expectation += prob * max_mu  # change 1 to max of mu
        prob_sum += prob
    expectation = expectation / prob_sum
    max_mu_o_v = 0
    for v in obj_info[o]['Vo']:
        max_mu_o_v = max(max_mu_o_v, mu[o][v])
    # subtract max v ,u o v and divide by |O|
    eai = (expectation - max_mu_o_v) / len(obj_info.keys())
    return eai


def get_mu_o_v_cond(o, v, v_, mu, psi, f_os, f_ow, candidate, so, wo, src_info, ancestor, gamma):
    numerator = get_prob_vow(v_, v, psi, candidate, o, so, src_info, ancestor) * mu[o][v]
    denominator = 0
    for v__ in candidate:
        denominator += (get_prob_vow(v_, v__, psi, candidate, o, so, src_info, ancestor) * mu[o][v__])
    f = numerator / denominator

    val_sum = 0
    for _ in candidate:
        val_sum += (gamma - 1)
    denominator = len(so) + len(wo) + val_sum + 1

    numerator = 0
    for src in so:
        numerator += f_os[src][v]
    for worker in wo:
        numerator += f_ow[worker][v]
    numerator += f
    numerator += (gamma - 1)

    assert 0 <= numerator / denominator <= 1

    return numerator / denominator


def worker_answer(tasks, gold_standards, groundtruths, obj_info, worker_info, worker_correct_prob):
    workers = worker_info.keys()
    for worker in workers:
        for obj in tasks[worker]:
            rand_num = random.random()
            if rand_num <= worker_correct_prob[worker] or len(obj_info[obj]['Vo']) == 1:
                ans = gold_standards[obj]
            else:
                assert gold_standards[obj] in obj_info[obj]['Vo']
                ans_list = obj_info[obj]['Vo'].copy()
                ans_list.remove(gold_standards[obj])
                ans = random.sample(ans_list, 1)[0]
            obj_info[obj]['Wo'].append(worker)
            worker_info[worker][obj] = ans
