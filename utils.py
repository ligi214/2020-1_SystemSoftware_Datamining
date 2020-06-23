import numpy as np


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


def get_g_os(phi, mu, obj_info, src_info, ancestor, temp):
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
                print('obj:{}, src:{}, vos:{}, flag:{}'.format(obj, src, vos, flag_h))
    return g_os


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
            # should add Worker's
            numerator = numerator + gamma - 1

            denominator = len(obj_info[obj]['So'])
            # should add |Wo|
            denominator = denominator + val_sum
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


def inference(obj_info, mu):
    inferred_val = dict()
    for obj in obj_info.keys():
        ans_val = obj_info[obj]['Vo'][0]
        for v in obj_info[obj]['Vo']:
            if mu[obj][ans_val] < mu[obj][v]:
                ans_val = v
        inferred_val[obj] = ans_val
    return inferred_val

