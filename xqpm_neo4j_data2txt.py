import pandas as pd
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
import copy

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j1234"))

# 读取el词典
f_read1 = open('ment2ent_2.pkl', 'rb')
dic = pickle.load(f_read1)
print(dic)

f_read2 = open('neo4jdata_xqpm_3.pkl', 'rb')
neo4j_dict_xqpm = pickle.load(f_read2)
print(neo4j_dict_xqpm)
f_read_ner = open('nerment2_xqpm.pkl','rb')
ner_dic = pickle.load(f_read_ner)
print(ner_dic)

nlpcc_test_data = pd.read_csv("./Data/NER_Data/triple_data_ans2.csv", sep='^')

test_data_no_mask = []
real_num = 0
result_ra_num = 0
NER_num = 0
err_ans = 0

err_ans_list = []
with driver.session() as session:
    for row in nlpcc_test_data.index[0:]:
        question = nlpcc_test_data.loc[row, "q_str"]
        sentence_l = question
        # entity = nlpcc_test_data.loc[row,"t_str"].split("|||")[0].split(">")[1].strip()
        entity = nlpcc_test_data.loc[row, "t_str"].split("|||")[0].strip()
        attribute = nlpcc_test_data.loc[row, "t_str"].split("|||")[1].strip()
        answer1 = nlpcc_test_data.loc[row, "t_str"].split("|||")[2].strip().replace('**@', '^')
        answer = nlpcc_test_data.loc[row, "a_str"].replace('**@', '^')
        answer2 = copy.deepcopy(answer)
        if answer == 'no':
            err_ans += 1
            answer = answer1.replace('(', '@').replace(')', '').replace(',', '&')

        flag = 0
        result_ra = neo4j_dict_xqpm.get(row,None)
        ment1 = ner_dic.get(row, None)
        if ment1 == None:
            continue
        has_el = dic.get(ment1, None)
        if has_el != None:
            houxuan_entity_list = copy.deepcopy(has_el[0])
            NER_num += 1
        else:
            if '》' in ment1:
                ment1 = ment1.replace('《', '').replace('》', '')
            else:
                ment1 = '《' + ment1 + '》'
            has_el = dic.get(ment1)
            if has_el != None:
                NER_num += 1
                houxuan_entity_list = copy.deepcopy(has_el[0])

        if result_ra != None:
            result_ra_num +=1

        linshi = []
        for i in range(len(result_ra)):
            if entity == result_ra[i][0] and result_ra[i][2] == answer :
                data = question + '\t' + result_ra[i][0] +'%%%' + result_ra[i][1] + '\t' + '1'
                flag = 1
            else:
                data = question + '\t' + result_ra[i][0] + '%%%' + result_ra[i][1] + '\t' + '0'
            linshi.append(data)
        if flag == 1:
            real_num += 1
            for i in range(len(linshi)):
                test_data_no_mask.append(linshi[i])
        else:
            # ment3 = ment1.replace('《', '').replace('》', '')
            # ment3 = list(ment3)
            # entity1 = ment3[0]
            # entity2 = ment3[-1]
            # cypher_statement = 'MATCH (name1)-[r]->(name2) WHERE (name1.name =~ ".*{1}") OR (name1.name =~ "{0}.*") RETURN [name1.name,r.name,name2.name]'.format(
            #     entity1, entity2)
            # result22 = session.run(cypher_statement).value()
            #
            # xqpm_neo4j_data = []
            # for i in range(len(result22)):
            #     if result22[i][0] in houxuan_entity_list:
            #         xqpm_neo4j_data.append(result22[i])
            #
            # result_ra.clear()
            # result_ra.extend(xqpm_neo4j_data)
            continue


# #字典保存
# f_save = open('neo4jdata_xqpm_3.pkl', 'wb')
# pickle.dump(neo4j_dict_xqpm, f_save)
# f_save.close()


print(str(real_num),str(result_ra_num),str(NER_num),str(err_ans))
with open("test_data_no_mask_2.txt", "w",encoding='utf-8') as f:
    f.write("\n".join(test_data_no_mask))