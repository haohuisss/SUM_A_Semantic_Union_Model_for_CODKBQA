# encoding=utf-8
import pandas as pd
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
import copy

from datetime import time, timedelta, datetime
from tqdm import tqdm

import re

# import neo4jchaxun
# import neo4jchaxun_xqpm
from run_ner import create_model, InputFeatures, InputExample
from bert import tokenization
from bert import modeling

from neo4j import GraphDatabase     #yueuu
#使用bert
# from run_pm_bert import BertSim  #yueuu
# from run_xq_bert import BertXQ
#使用sbert
# from xq_sbert_test import Sber
# from xq_sbert_test import Evaluator
# from xq_sbert_test import data_generator
#使用cosent
from xq_cosent_test import Csent
from xq_cosent_test import Evaluator
from xq_cosent_test import data_generator

import time

# from neo4jchaxun_xqpm import neo4j_ca

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from sklearn.metrics._classification import f1_score, accuracy_score, precision_score, recall_score

def clsentity(entity):
    result_he = ''.join((str(x) for x in entity))
    if "UNK" in result_he:
        result = result_he.replace('[UNK]','')
    else:
        result = result_he
    if bool(re.search('[0-9]',result)):
        return '2'
    else:
        if bool(re.search('[a-zA-Z]',result)):
            return '1'
        else:
            return '0'
#关系中检索不了的字符



# driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j1234"))
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "do_predict_outline", False,
    "Whether to do predict outline."
)
flags.DEFINE_bool(
    "do_predict_online", False,
    "Whether to do predict online."
)

# init mode and session
# move something codes outside of function, so that this code will run only once during online prediction when predict_online is invoked.
is_training=False
use_one_hot_embeddings=False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
# gpu_config.gpu_options.allow_growth = False
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None
print(FLAGS.output_dir)
print('checkpoint path:{}'.format(os.path.join(FLAGS.output_dir, "checkpoint")))
if not os.path.exists(os.path.join(FLAGS.output_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_mask")
    label_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="label_ids")
    segment_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="segment_ids")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config, is_training, input_ids_p, input_mask_p, segment_ids_p,
        label_ids_p, num_labels, use_one_hot_embeddings)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

def predict_outline():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, FLAGS.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, FLAGS.max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        # start = datetime.now()
        # nlpcc_test_data = pd.read_csv("./Data/NER_Data/q_t_a_df_testing.csv",sep='^')
        nlpcc_test_data = pd.read_csv("./Data/NER_Data/triple_data_ans2.csv",sep='^')
        # correct = 0
        # num = 0
        # correct_v2 = 0
        # test_size = nlpcc_test_data.shape[0]
        # nlpcc_test_result = []

        #使用BERT作为ED和PM的模型
        # sim = BertSim()
        # sim.set_mode(tf.estimator.ModeKeys.PREDICT)
        # xq = BertXQ()
        # xq.set_mode(tf.estimator.ModeKeys.PREDICT)
        # errentity = []

        #使用SBERT作为ED和PM的模型,的基本配置
        config_path = './ModelParams/chinese_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = './ModelParams/chinese_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = './ModelParams/chinese_L-12_H-768_A-12/vocab.txt'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # 使用SBERT作为XQPM的模型
        # sb = Sber()
        # evaluator = Evaluator()
        # encoder_xqpm, model_xqpm = sb.get_model(config_path, checkpoint_path)
        # model_xqpm.load_weights('./Output/XQPM/XQPM_Data.cosent_xqpm_mask_1e-5_30_1.weights')

        # tokenizer_sb = sb.get_tokenizer(dict_path)

        #使用COSENT作为ED和PM的模型
        ct = Csent()
        evaluator = Evaluator()
        encoder_xqpm, model_xqpm = ct.get_model(config_path, checkpoint_path)
        model_xqpm.load_weights('./Output/XQPM/XQPM_Data.cosent_xqpm_mask_1e-5_30_1.weights')
        tokenizer_sb = ct.get_tokenizer(dict_path)



        #读取el词典
        f_read1 = open('ment2ent.pkl', 'rb')
        dic = pickle.load(f_read1)
        print(dic)
        f_read2 = open('neo4jdata_xqpm_2.pkl', 'rb')
        neo4j_dict_xqpm = pickle.load(f_read2)
        print(neo4j_dict_xqpm)

        NER_num = 0
        REL_num = 0
        ANS_num = 0
        xq_num = 0
        # ccc = []

        err_ans = []
        # neo4j_dict_xqpm = {}

        #计时开始
        start_time = time.perf_counter()
        for row in nlpcc_test_data.index[0:]:
            print(row,nlpcc_test_data.index)
            question = nlpcc_test_data.loc[row,"q_str"]
            sentence_l = question
            # entity = nlpcc_test_data.loc[row,"t_str"].split("|||")[0].split(">")[1].strip()
            entity = nlpcc_test_data.loc[row,"t_str"].split("|||")[0].strip()
            attribute = nlpcc_test_data.loc[row, "t_str"].split("|||")[1].strip()
            answer1 = nlpcc_test_data.loc[row, "t_str"].split("|||")[2].strip().replace('**@','^')
            answer = nlpcc_test_data.loc[row, "a_str"].replace('**@','^')
            if answer == 'no':
                answer = answer1.replace('(', '@').replace(')', '').replace(',','&')
            #answer = answer.replace(' ','')
            sentence = str(question)
            if len(sentence) < 2:
                print(sentence)
                continue
            sentence = tokenizer.tokenize(sentence)
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p:segment_ids,
                         label_ids_p:label_ids}
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            # print(pred_label_result)
            #todo: 组合策略
            result = strage_combined_link_org_loc(sentence, pred_label_result[0], True)
            # 加入《》的先验方式
            if len(result) ==1 and len(result[0]) <= 8 and ('《' in result[0]) and ('》' not in result[0]):
                result = [result[0]+'》']

            #当识别实体为空时，跳过本次循环
            if result ==[]:
                # errentity.append(str(row) + "\t" + question + "\t" + entity + "\t" + ','.join(result))
                continue
            ment1 = result[0].replace('@', '').replace('，', '&').replace(',', '&').replace('#','').lower()
            ment1 = ment1.replace('"', '@@').replace('（', '').replace('）', '').replace(
                '·', '').replace('“', '').replace('”', '').replace('(', '@').replace(')', '')


            has_el = dic.get(ment1,None)
            if has_el != None:
                houxuan_entity_list = copy.deepcopy(has_el[0])
            else:
                if '》' in ment1:
                    ment1 = ment1.replace('《','').replace('》','')
                else:
                    ment1 = '《' + ment1 + '》'
                has_el = dic.get(ment1)
                if has_el != None:
                    # NER_num += 1
                    houxuan_entity_list = copy.deepcopy(has_el[0])
                else:
                    continue
                #有些带《》的实体还是有问题，没有通过dict来解决件errentity_5
                #errentity.append(str(row) + "\t" + question + "\t" + entity + "\t" + ment1)

            result_ra = neo4j_dict_xqpm.get(row,None)

            if result_ra == None or result_ra == []:
                continue

            #可以从这里获取出度
            out_degree_dict = {}
            for i in range(len(result_ra)):
                entity_4_degree = result_ra[i][0]
                if entity_4_degree in out_degree_dict:
                    out_degree_dict[entity_4_degree] += 1
                else:
                    out_degree_dict[entity_4_degree] = 1
            max_out_degree = max(out_degree_dict.values())
            #使用SBERT时
            # f_xq_data = []
            # for i in range(len(houxuan_entity_list)):
            #     data_xq = (question,houxuan_entity_list[i],0)
            #     f_xq_data.append(data_xq)
            #用于预测xq分数
            # xq_score = []
            # test_generator_sb = data_generator(tokenizer_sb, f_xq_data, len(f_xq_data))
            # xq_score = evaluator.evaluate(test_generator_sb, encoder_xq, ct.l2_normalize)

            #使用bert时
            # for j in range(len(houxuan_entity_list)):
            #     xq_score.append(xq.predict(sentence_l,houxuan_entity_list[j])[0][1])

            # s_xq_dict = {}
            # for n in range(3):
            #     if xq_score == []:
            #         break
            #     else:
            #         s_xq = max(xq_score)
            #         max_idx = xq_score.index(s_xq)
            #         fin_entity = houxuan_entity_list[max_idx]
            #         s_xq_dict[fin_entity] = s_xq
            #         del xq_score[max_idx]
            #         del houxuan_entity_list[max_idx]
            # s_xq_list = list(s_xq_dict.keys())
            # result_ra = neo4jchaxun.neo4j_ca(driver,ent_list= s_xq_list)

            # ment3 = ment1.replace('《','').replace('》','')
            # result_ra =neo4jchaxun_xqpm.neo4j_ca(driver, ment3,houxuan_entity_list)
            #
            # xqpm_neo4j_data = []
            # for i in range(len(result_ra)):
            #     if result_ra[i][0] in houxuan_entity_list:
            #         xqpm_neo4j_data.append(result_ra[i])
            #
            # neo4j_dict_xqpm[row] = xqpm_neo4j_data



            really_flag = 0
            for i in range(len(result_ra)):
                if entity == result_ra[i][0] and answer == result_ra[i][2]:
                    really_flag = 1
            if really_flag == 1:
                NER_num += 1

            # ans_score = []
            sentence_l2 = sentence_l.replace(result[0].replace('##',''),'entity')
            # #使用SBERT时
            f_xqpm_data = []
            for j in range(len(result_ra)):
                ent_rel = result_ra[j][0].replace(ment1,'entity')+result_ra[j][1]
                data = (sentence_l2,ent_rel,1)
                f_xqpm_data.append(data)
            test_generator_sb = data_generator(tokenizer_sb, f_xqpm_data, len(f_xqpm_data))
            xqpm_score = evaluator.evaluate(test_generator_sb, encoder_xqpm, ct.l2_normalize)
            for i in range(len(xqpm_score)):
                if result_ra[i][1] in sentence_l2 :
                    xqpm_score[i] += 0.1
                #在这里直接将出度的数量在在分数上（出度决定一切）
                xqpm_score[i] = xqpm_score[i] * 1 + (out_degree_dict[result_ra[i][0]] /max_out_degree )* 0.3

                #考虑在这里加入出度的分数 ， 使用出度对候选实体进行分数补强

            # for i in range(len(pm_score)):
            #     ent_xq_score = s_xq_dict.get(result_ra[i][0])
            #     sim_score=pm_score[i]
            #     fin_score = (ent_xq_score*0.5 + sim_score*0.5)
            #     ans_score.append(fin_score)


            #使用bert时
            # for j in range(len(result_ra)):
            #     ent_xq_score = s_xq_dict.get(result_ra[j][0])
            #     sim_score=sim.predict(sentence_l2,result_ra[j][1])[0][1]
            #     fin_score = (ent_xq_score*0.5 + sim_score*0.5)
            #     ans_score.append(fin_score)

            max_idx = xqpm_score.index(max(xqpm_score))
            #to_do 如果第二名答案的分数和第一名的分数相差小于X就选用2个
            # if result_ra[max_idx][0] == attribute:
            #     REL_num +=1

            #验证答案
            answer_pre = result_ra[max_idx][2]
            if answer_pre == answer:
                ANS_num+=1
            else:
                print(str(ANS_num), row)
                err_ans.append( str(row) + '\t' + question + '\t'+ ment1 +'\t' + answer + '\t' + answer_pre)
                continue
            print(str(ANS_num),row)
            # if len(neo4j_dict_xqpm) % 500 == 0:
            #     f_save = open('neo4jdata_xqpm.pkl', 'wb')
            #     pickle.dump(neo4j_dict_xqpm, f_save)
            #     f_save.close()
            #     print(neo4j_dict_xqpm)

        print(str(ANS_num),str(NER_num))
        end_time = time.perf_counter()
        print(str(end_time-start_time))
        #字典保存
        # f_save = open('nerment1_xqpm.pkl', 'wb')
        # pickle.dump(ner_dict, f_save)
        # f_save.close()
        #
        # NNN = 8465
        # print("accuracy: {}%, nlpccqa_correct: {}, total: {}".format(ANS_num * 100.0 / float(NNN), ANS_num, NNN))
        # print('Time used: {} sec'.format((datetime.now() - start).seconds))


def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


def strage_combined_link_org_loc(tokens, tags, flag):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """
    def print_output(data, type):
        line = []
        for i in data:
            line.append(i.word)
        print('{}: {}'.format(type, ', '.join(line)))

    def string_output(data):
        line = []
        for i in data:
            line.append(i.word)
        return line

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    person, loc, org = eval.get_result(tokens, tags)
    if flag:
        if len(loc) != 0:
            print_output(loc, 'LOC')
        if len(person) != 0:
            print_output(person, 'PER')
        if len(org) != 0:
            print_output(org, 'ORG')
    person_list = string_output(person)
    person_list.extend(string_output(loc))
    person_list.extend(string_output(org))
    return person_list


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []
    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))


if __name__ == "__main__":
    if FLAGS.do_predict_outline:
        predict_outline()



