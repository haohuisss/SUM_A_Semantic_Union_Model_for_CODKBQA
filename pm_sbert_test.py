#! -*- coding:utf-8 -*-

import json
import numpy as np
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf
import sys

class Sber:

    def __init__(self, maxlen=64, batch_size=8, epochs=5, task_name_pm='SIM_Data'):
        # 基本参数
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.epochs = epochs
        self.task_name_pm = task_name_pm

    def load_data(filename):
        """加载数据（带标签）
        单条格式：(文本1, 文本2, 标签)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 4:
                    D.append((l[1], l[2], int(l[3])))
        return D

    def get_tokenizer(self, dict_path):
        # 建立分词器
        tokenizer = Tokenizer(dict_path, do_lower_case=True)
        return tokenizer

    def merge(self, inputs):
        """向量合并：a、b、|a-b|拼接
        """
        a, b = inputs[::2], inputs[1::2]
        o = K.concatenate([a, b, K.abs(a - b)], axis=1)
        return K.repeat_elements(o, 2, 0)

    def get_model(self, config_path,checkpoint_path):
        # 构建模型
        base = build_transformer_model(config_path,checkpoint_path)
        output = keras.layers.Lambda(lambda x: x[:, 0])(base.output)
        # output = keras.layers.GlobalAveragePooling1D()(base.output)
        encoder = keras.models.Model(base.inputs, output)

        output = keras.layers.Lambda(self.merge)(output)
        output = keras.layers.Dense(units=2, activation='softmax')(output)
        model = keras.models.Model(base.inputs, output)

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(1e-5),
            metrics=['accuracy']
        )
        return encoder, model

    def l2_normalize(self, vecs):
        """l2标准化
        """
        norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
        return vecs / np.clip(norms, 1e-8, np.inf)



class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, tokenizer, data, batch_size=32, buffer_size=None):
        super().__init__(data, batch_size, buffer_size)
        self.tokenizer = tokenizer

        pass
    def __iter__(self, random=False):

        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            label = int(
                label > 2.5
            ) if random and sb.task_name_pm == 'STS-B' else label
            for text in [text1, text2]:
                token_ids, segment_ids = self.tokenizer.encode(text, maxlen=64)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []




class Evaluator(keras.callbacks.Callback):
    """保存验证集分数最好的模型
    """
    def __init__(self):

        self.best_val_score = 0.

    def evaluate(self, data, encoder, l2_normalize):
        Y_true, Y_pred = [], []
        for x_true, y_true in data:
            Y_true.extend(y_true[::2, 0])
            x_vecs = encoder.predict(x_true)
            x_vecs = l2_normalize(x_vecs)
            y_pred = (x_vecs[::2] * x_vecs[1::2]).sum(1)
            Y_pred.extend(y_pred)
        return Y_pred


if __name__ == '__main__':
    # 模型路径
    config_path = './ModelParams/chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = './ModelParams/chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = './ModelParams/chinese_L-12_H-768_A-12/vocab.txt'
    sb = Sber()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    evaluator = Evaluator()
    encoder, model = sb.get_model(config_path,checkpoint_path)
    model.load_weights('./Output/SIM/%s.sbert_pm_no_mask_1e-5_30_1.weights' % sb.task_name_pm)
    tokenizer = sb.get_tokenizer(dict_path)
    #
    top1_xq_num = 0
    top2_xq_num = 0
    top3_xq_num = 0
    flag = 1
    query_num = 0
    result_rel = []
    err = []
    really_rel_list = []
    rea_r_flag = 0
    with open('./Data/SIM_Data/test_no_mask.txt', 'r', encoding='utf-8') as fd:
        # que = '你知道entity这本书的作者是谁吗？'
        que ='你知道计算机应用基础这本书的作者是谁吗？'
        que = que.strip()
        for line in fd:
            line = line.replace(' ', '')
            label = line.split('\t')[3].replace('\n', '')
            que2 = line.split('\t')[1].strip().replace(' ', '')
            if str(que) != str(que2):
                query_num += 1
                sim_score = []
                if rea_r_flag != 1:
                    err.append('无正确答案' + '\t' + que + '\n')
                    query_num -= 1
                test_generator = data_generator(tokenizer,result_rel, len(result_rel))
                test_score = evaluator.evaluate(test_generator, encoder, sb.l2_normalize)
                # for j in range(len(result_rel)):
                #     sim_score.append(sim.predict(que, result_rel[j])[0][1])
                # for j in range(len(test_score)):
                #     print(result_rel[j][1], f'similarity：{test_score[j]}')
                max_idx = test_score.index(max(test_score))
                print('相似度最高的关系为：', result_rel[max_idx][1])
                C = len(test_score)
                flag = 1
                #开始top1 正确
                if result_rel[max_idx][1] in really_rel_list:
                    top1_xq_num += 1
                else:
                    err.append(que + '\t' + really_rel + '\t' + result_rel[max_idx][1] + '\n')
                if C >=2:
                    #开始top2 正确
                    del test_score[max_idx]
                    del result_rel[max_idx]
                    max_idx = test_score.index(max(test_score))
                    if result_rel[max_idx][1] in really_rel_list:
                        top2_xq_num += 1
                if C >= 3:
                    # 开始top3 正确
                    del test_score[max_idx]
                    del result_rel[max_idx]
                    max_idx = test_score.index(max(test_score))
                    if result_rel[max_idx][1] in really_rel_list:
                        top3_xq_num += 1
                result_rel = []
                really_rel_list = []
                rea_r_flag = 0
            if label == '1':
                que = line.split('\t')[1]
                really_rel = line.split('\t')[2]
                really_rel_list.append(really_rel)
                data = (que, really_rel, 1)
                result_rel.append(data)
                rea_r_flag = 1
            if label == '0':
                que = line.split('\t')[1]
                err_rel = line.split('\t')[2]
                data = (que, err_rel, 0)
                result_rel.append(data)
            que = line.split('\t')[1].strip()
    print(str(top1_xq_num), str(top2_xq_num),str(top3_xq_num),str(query_num))
