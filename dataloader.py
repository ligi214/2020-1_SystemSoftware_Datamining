import os
import csv
import pandas as pd

heritage_path = './../dataset/heritage'
birthplace_path = './../dataset/birthplace'


def get_heritage():
    claims, groundtruths, hierarchy = [], dict(), []
    claims_path = os.path.join(heritage_path, 'claims.csv')
    groundtruths_path = os.path.join(heritage_path, 'groundtruths.csv')
    hierarchy_path = os.path.join(heritage_path, 'hierarchy.csv')

    df = pd.read_csv(claims_path)
    index = df.keys()
    for row in df.values:
        claims.append({index[0]: row[0], index[1]: row[1], index[2]: row[2]})

    df = pd.read_csv(groundtruths_path)
    for row in df.values:
        groundtruths[row[0]] = row[1].split(',')

    df = pd.read_csv(hierarchy_path)
    index = df.keys()
    for row in df.values:
        hierarchy.append({index[0]: row[0], index[1]: row[1], index[2]: row[-1]})

    return claims, groundtruths, hierarchy


def get_birthplace():
    pass


def records_processing(records):
    src_info, obj_info = dict(), dict()
    for record in records:
        if not record['src'] in src_info.keys():
            src_info[record['src']] = dict()
        src_info[record['src']][record['obj']] = record['value']

        if not record['obj'] in obj_info.keys():
            obj_info[record['obj']] = {'Vo': [], 'So': []}
        if not record['value'] in obj_info[record['obj']]['Vo']:
            obj_info[record['obj']]['Vo'].append(record['value'])
        obj_info[record['obj']]['So'].append(record['src'])
    return src_info, obj_info


def hierarchy_processing(hierarchy):
    ancestors, descendants = dict(), dict()
    for i, node in enumerate(hierarchy):
        assert i == node['ID']
        name = node['name']
        if name in ancestors.keys():
            continue
        ancestors[name] = []
        id = node['parentID']
        while id > -1:
            curr_node = hierarchy[id]
            if curr_node['name'] in ancestors.keys():
                ancestors[name] = ancestors[name] + [curr_node['name']] + ancestors[curr_node['name']]
                break
            else:
                ancestors[name].append(curr_node['name'])
                id = curr_node['parentID']
    for name in ancestors:
        for anc in ancestors[name]:
            if anc not in descendants.keys():
                descendants[anc] = []
            descendants[anc].append(name)
    return ancestors, descendants


if __name__ == '__main__':
    claims, groundtruths, hierarchy = get_heritage()
    print(claims)
    print(groundtruths)
    print(hierarchy)
    src_list, obj_info = records_processing(claims)
    anc, des = hierarchy_processing(hierarchy)
    print(anc)
    print(anc['New York'])
