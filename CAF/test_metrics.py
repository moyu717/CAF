#coding:utf-8
import numpy as np
import multiprocessing
import tensorflow as tf
import seaborn as sn
import logging
import scipy
from collections import defaultdict
user_num, item_num = 1000, 2000
sn.set()
cores = multiprocessing.cpu_count()
Items = set(range(item_num))
workdir = '../data/'

user_pos_train = {}
with open(workdir + 'train_data.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        lid = int(line[1])
        r = float(line[2])
        if r > 0:
            if uid in user_pos_train:
                user_pos_train[uid].append(lid)
            else:
                user_pos_train[uid] = [lid]

user_pos_test = {}
with open(workdir + 'test_data.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        lid = int(line[1])
        r = float(line[2])
        if r > 0:
            if uid in user_pos_test:
                user_pos_test[uid].append(lid)
            else:
                user_pos_test[uid] = [lid]

all_users = list(user_pos_train.keys())
all_users.sort()


def load_UV_model(pmf_path):
    if pmf_path is not None:
        logging.info("Loading pmf data from " + pmf_path)
        data = scipy.io.loadmat(pmf_path)
        m_U = data["m_U"]
        m_V = data["m_V"]
        m_theta = data["m_theta"]
    return m_U, m_V, m_theta

def simple_test_one_user(x):
    score = x[0]
    u = x[1]

    test_pois = list(Items - set(user_pos_train[u]))
    poi_score = []
    for i in test_pois:
        poi_score.append((i, score[i]))

    poi_score = sorted(poi_score, key=lambda x: x[1], reverse=True)
    poi_sort = [x[0] for x in poi_score]

    r = []
    for i in poi_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)


    precision_5 = np.mean(r[:5])
    precision_10 = np.mean(r[:10])
    precision_20 = np.mean(r[:20])
    precision_50 = np.mean(r[:50])


    result = np.array(
        [precision_5, precision_10, precision_20, precision_50])
    print u, result
    return result

m_U, m_V, m_theta = load_UV_model(
    pmf_path="./model/caf_pmf.mat")
m_V = np.array(m_V)
n5_list, n10_list, n20_list, n50_list, n100_list, r5_list, \
r10_list, r20_list, r50_list = [], [], [], [], [], [], [], [], []

test_users = list(user_pos_test.keys())
test_user_num = len(test_users)
N_test = user_num
idxlist_test = range(N_test)
batch_size_test = 128
with tf.Session() as sess:
    result = np.array([0.] * 4)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = list(user_pos_test.keys())
    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        user_batch = test_users[st_idx:end_idx]
        u_tmp = np.array(m_U[st_idx:end_idx])
        user_batch_rating = np.dot(u_tmp, m_V.T)
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re
    pool.close()
    result = result / test_user_num
    result = list(result)
    print result


