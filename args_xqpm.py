import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

file_path = os.path.dirname(__file__)

model_dir = os.path.join(file_path, 'ModelParams/chinese_L-12_H-768_A-12/')
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
output_dir = os.path.join(file_path, 'Output/XQPM/bert_result_nomask/')
vocab_file = os.path.join(model_dir, 'vocab.txt')
data_dir = os.path.join(file_path, 'Data/XQPM_Data/')

num_train_epochs = 6
batch_size = 32
learning_rate = 0.00004

# gpu使用率
gpu_memory_fraction = 1

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 80

# 预训练模型
train = True

# 测试模型
test = False
