def accuracy(ans, gold_standard):
    # change groundtruths if the gold standard is not in candidate
    num = 0
    for obj in ans.keys():
        inferred_ans = ans[obj]
        true_ans = gold_standard[obj]
        if inferred_ans == true_ans:
            num += 1
    acc = num / len(ans.keys())
    return acc


def gen_accuracy(ans, groundtruths):
    num = 0
    for obj in ans.keys():
        inferred_ans = ans[obj]
        if inferred_ans in groundtruths[obj]:
            num += 1
    acc = num / len(ans.keys())
    return acc


def avg_distance(ans, gold_standard, groundtruths, ancestors, descendants):
    dist = 0
    for obj in ans.keys():
        inferred_ans = ans[obj]
        if inferred_ans in groundtruths[obj]:
            # inferred ans == true or ancestor of goldstandard
            dist += groundtruths[obj].index(inferred_ans)
        elif groundtruths[obj][0] in ancestors[inferred_ans]:
            # inferred ans == descendant of goldstandard
            dist += ancestors[inferred_ans].index(groundtruths[obj][0])
        else:
            # absolutely wrong answer
            d = 0
            v = groundtruths[obj][0]
            while v not in descendants.keys() or inferred_ans not in descendants[v]:
                v = ancestors[v][0]
                d += 1
            d += (ancestors[inferred_ans].index(v) + 1)
            dist += d
    acc = dist / len(ans.keys())
    return acc
