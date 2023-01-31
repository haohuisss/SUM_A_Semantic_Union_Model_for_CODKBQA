from neo4jchaxun import neo4j_ca
import pandas as pd
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
import copy

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j1234"))
nlpcc_test_data = pd.read_csv("./Data/NER_Data/triple_data_ans2.csv",sep='^')

# 读取el词典
f_read1 = open('ment2ent_2.pkl', 'rb')
dic = pickle.load(f_read1)
print(dic)
#读取NER字典
f_read_ner = open('nerment1_xqpm.pkl','rb')
ner_dic = pickle.load(f_read_ner)
print(ner_dic)

f_read2 = open('neo4jdata_xqpm.pkl', 'rb')
neo4j_dict_xqpm = pickle.load(f_read2)
print(neo4j_dict_xqpm)

f_3_num = 0
rea_f_3_num = 0

f_d3_num = 0
rea_f_d3_num = 0
rea_num = 0
NER_num = 0

a = []
with driver.session() as session:
    for row in nlpcc_test_data.index[0:]:
        really_flag = 0
        print(row, nlpcc_test_data.index)
        question = nlpcc_test_data.loc[row, "q_str"]
        sentence_l = question
        # entity = nlpcc_test_data.loc[row,"t_str"].split("|||")[0].split(">")[1].strip()
        entity = nlpcc_test_data.loc[row, "t_str"].split("|||")[0].strip()
        answer1 = nlpcc_test_data.loc[row, "t_str"].split("|||")[2].strip().replace('**@', '^')
        answer = nlpcc_test_data.loc[row, "a_str"].replace('**@', '^')
        if answer == 'no':
            answer = answer1.replace(',','&').replace('(','@').replace(')','')
        ment1 = ner_dic.get(row,None)
        if ment1 == None:
            continue

        # ment1 = ment1.replace('《','').replace('》','')
        # has_el1 = dic.get(ment1,None)
        #
        # ment2 = '《' + ment1 + '》'
        # has_el2 = dic.get(ment2, None)
        # if has_el1 != None and has_el2 != None:
        #     data1 = copy.deepcopy(has_el1[0])
        #     data2 = copy.deepcopy(has_el2[0])
        #     if set(data1) != set(data2):
        #         has_el1[0].extend(data2)
        #         has_el2[0].extend(data1)






        has_el = dic.get(ment1,None)
        if has_el != None:
            NER_num += 1
            houxuan_entity_list = copy.deepcopy(has_el[0])
        else:
            if '》' in ment1:
                ment2 = ment1.replace('《','').replace('》','')
            else:
                ment2 = '《' + ment1 + '》'
            has_el = dic.get(ment2)
            if has_el != None:
                NER_num += 1
                houxuan_entity_list = copy.deepcopy(has_el[0])
            else:
                continue

        # if entity in houxuan_entity_list:
        #     rea_num += 1
        #     flag2 = 1

        houxuan_entity_list = list(set(houxuan_entity_list))

        old_result_ra = neo4j_dict_xqpm.get(row, None)
        for i in range(len(old_result_ra)):
            if entity == old_result_ra[i][0] and answer == old_result_ra[i][2]:
                really_flag = 1
        if really_flag != 1:
            if len(houxuan_entity_list) <= 3:
                xqpm_neo4j_data = neo4j_ca(driver, houxuan_entity_list)
            else:
                ment3 = ment1.replace('《','').replace('》','')
                cypher_statement = 'MATCH (name1)-[r]->(name2) WHERE (name1.name CONTAINS "{0}") RETURN [name1.name,r.name,name2.name]'.format(
                    ment3)
                result_ra = session.run(cypher_statement).value()
                # result_ra =neo4j_ca(driver, ment3,houxuan_entity_list)
                xqpm_neo4j_data = []
                for i in range(len(result_ra)):
                    if result_ra[i][0] in houxuan_entity_list:
                        xqpm_neo4j_data.append(result_ra[i])
            old_result_ra.clear()
            old_result_ra.extend(xqpm_neo4j_data)




        # if entity in b:
        #     rea_num += 1

# print(str(rea_num),str(NER_num))


        # xqpm_neo4j_data = neo4j_dict_xqpm.get(row,None)
        # if len(houxuan_entity_list) <= 3:
        #     f_3_num +=1


        # if len(houxuan_entity_list) <= 3:
        #     xqpm_neo4j_data = neo4j_ca(driver, houxuan_entity_list)
        # else:
        #     ment3 = ment1.replace('《','').replace('》','')
        #     cypher_statement = 'MATCH (name1)-[r]->(name2) WHERE (name1.name CONTAINS "{0}") RETURN [name1.name,r.name,name2.name]'.format(
        #         ment3)
        #     result_ra = session.run(cypher_statement).value()
        #     # result_ra =neo4j_ca(driver, ment3,houxuan_entity_list)
        #     xqpm_neo4j_data = []
        #     for i in range(len(result_ra)):
        #         if result_ra[i][0] in houxuan_entity_list:
        #             xqpm_neo4j_data.append(result_ra[i])
        #
        # neo4j_dict_xqpm[row] = xqpm_neo4j_data
#         if len(neo4j_dict_xqpm) % 200 == 0:
#             f_save = open('neo4jdata_xqpm.pkl', 'wb')
#             pickle.dump(neo4j_dict_xqpm, f_save)
#             f_save.close()
#             print(neo4j_dict_xqpm)
#字典保存
f_save = open('neo4jdata_xqpm_2.pkl', 'wb')
pickle.dump(neo4j_dict_xqpm, f_save)
f_save.close()
