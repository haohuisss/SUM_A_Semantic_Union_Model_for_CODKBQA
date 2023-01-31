import pandas as pd
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from run_pm_bert import BertSim

sim = BertSim()
sim.set_mode(tf.estimator.ModeKeys.PREDICT)
data_path = './Data/SIM_Data/test_no_mask.txt'
top1_xq_num = 0
top2_xq_num = 0
top3_xq_num = 0
flag = 1
query_num = 0
result_rel = []
err = []
really_rel_list = []
rea_r_flag = 0
with open(data_path, 'r', encoding='utf-8') as fd:
    # que = '你知道entity这本书的作者是谁吗？'
    que = '你知道计算机应用基础这本书的作者是谁吗？'
    que = que.strip()
    for line in fd:
        line = line.replace(' ', '')
        label = line.split('\t')[3].replace('\n', '')
        que2 = line.split('\t')[1].strip().replace(' ', '')
        if str(que) != str(que2):
            query_num += 1
            test_score = []
            for j in range(len(result_rel)):
                test_score.append(sim.predict(que, result_rel[j])[0][1])
            max_idx = test_score.index(max(test_score))
            print('相似度最高的关系为：', result_rel[max_idx])
            C = len(test_score)
            flag = 1
            # 开始top1 正确
            if result_rel[max_idx] in really_rel_list:
                top1_xq_num += 1
            else:
                err.append(que + '\t' + really_rel + '\t' + result_rel[max_idx] + '\n')
            if C >= 2:
                # 开始top2 正确
                del test_score[max_idx]
                del result_rel[max_idx]
                max_idx = test_score.index(max(test_score))
                if result_rel[max_idx] in really_rel_list:
                    top2_xq_num += 1
            if C >= 3:
                # 开始top3 正确
                del test_score[max_idx]
                del result_rel[max_idx]
                max_idx = test_score.index(max(test_score))
                if result_rel[max_idx] in really_rel_list:
                    top3_xq_num += 1
            result_rel = []
            really_rel_list = []
            rea_r_flag = 0
        if label == '1':
            que = line.split('\t')[1]
            really_rel = line.split('\t')[2]
            really_rel_list.append(really_rel)
            result_rel.append(really_rel)
        if label == '0':
            que = line.split('\t')[1]
            err_rel = line.split('\t')[2]
            result_rel.append(err_rel)
        que = line.split('\t')[1].strip()
print(str(top1_xq_num), str(top2_xq_num), str(top3_xq_num), str(query_num))


