import os
import numpy as np



# raw_data path
train_data_path = './raw_data/train.txt'
valid_data_path = './raw_data/valid.txt'
test_data_path = './raw_data/test.txt'


PUNCTUATION_VOCABULARY = [",COMMA", ".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON", "-DASH"]
EOS_TOKENS = [".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"]


# 构造字典
class buildVocabSet():
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 1
    
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
        
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1


# 获取样本和label
def get_samples(path):
    """
    得到path路径下得数据样本inputs和labels, 分别存储到一个list中

    输出:  inputs:['that is the general context', ...]
           labels: [[0, 0, 0, 0, 2], ...]
    """
    with open(path, encoding='utf-8') as f:
        inputs, targets = [], []
        for i, line in enumerate(f):
            # line = line.rstrip().split(' ')    # 分词 ['that', 'is', 'the', 'general', 'context', '.PERIOD']
            # if len(line) > max_num_words:
            #     line = line[:max_num_words]
            # line = ' '.join(line)    # 合并 'that is the general context .PERIOD'
            if len(line.replace(' ', '')):    # 这一行不是空的
                # if not line.endswith(tuple(EOS_TOKENS)):    # 如果不是以句号，问号，感叹号结尾
                #     line = line + " <nobound>"
                split = line.split(' ')    # 分词 ['that', 'is', 'the', 'general', 'context', '.PERIOD']
                label = []
                
                for idx, word in enumerate(split):
                    if word in PUNCTUATION_VOCABULARY:    # 如果当前是标点符号，则跳过
                        continue
                    else:
                        try:
                            if split[idx + 1] in PUNCTUATION_VOCABULARY:    # 当前不是标点，但是后面是标点符号
                                class_num = PUNCTUATION_VOCABULARY.index(split[idx + 1]) + 1
                                label.append(class_num)
                            else:
                                label.append(0)
                        except IndexError:
                            continue
                for punc in PUNCTUATION_VOCABULARY:    
                    line = line.replace(punc, '')
                line = ' '.join(line.split())
                if not len(line):
                    continue
                # if line.endswith(" <nobound>"):
                #     line = line.replace(" <nobound>", "")
                inputs.append(line)
                targets.append(label)
    return inputs, targets

# 存储train, valid, test到分别得路径下面
def save_all_data_samples(save_path_prefix):
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    
    print("Building dictionary...")
    vocabSet = load_dict('./processed_data/dicts/word2index.npy')
    
    # saving training samples
    data_process_and_save(vocabSet, train_data_path, save_path_prefix, padding_mode='post', padding_value=0, name='train')
 
    # get validation samples
    data_process_and_save(vocabSet, valid_data_path, save_path_prefix, padding_mode='post', padding_value=0, name='valid')

    # get validation samples
    data_process_and_save(vocabSet, test_data_path, save_path_prefix, padding_mode='post', padding_value=0, name='test')


# 在处处之前对数据进行处理
# 1. 加载样本
# 2. word2index
# 3. 对于长度小于最大长度的进行padding
def data_process_and_save(vocabSet, data_path, save_path_prefix, padding_mode, padding_value, name='train'):
    print("Loading {} samples...".format(name))
    inputs, labels = get_samples(data_path)
    print("{} samples word to index...".format(name))
    inputs = [[int(vocabSet[word]) for word in sentence.split(' ')] for sentence in inputs]
    print("{} padding {} samples with value {}...".format(padding_mode, name, padding_value))
    samples = padding_data(inputs, labels, padding_mode='post', padding_value=padding_value)
    np.save(os.path.join(save_path_prefix, '{}.npy'.format(name)), samples)

# padding数据的函数
def padding_data(inputs, labels, padding_mode='post', padding_value=0):
    max_len = max([len(item) for item in inputs])
    if padding_mode == 'post':
        inputs = [item + [padding_value] * (max_len - len(item)) for item in inputs]
        labels = [item + [padding_value] * (max_len - len(item)) for item in labels]
    if padding_mode == 'prev':
        inputs = [[padding_value] * (max_len - len(item)) + item for item in inputs]
        labels = [[padding_value] * (max_len - len(item)) + item for item in labels]
    return (inputs, labels)

# 用于生成和保存word2index和index2word词典的函数
def save_word_index_dict(dict_path_prefix):
    if not os.path.exists(dict_path_prefix):
        os.makedirs(dict_path_prefix)
    

    print("Getting samples...")
    train_inputs, _ = get_samples(train_data_path)
    valid_inputs, _ = get_samples(valid_data_path)
    test_inputs, _ = get_samples(test_data_path)

    print("Building dictionary...")
    vocabSet = buildVocabSet()

    for sentence in train_inputs:
        vocabSet.addSentence(sentence)
    for sentence in valid_inputs:
        vocabSet.addSentence(sentence)
    for sentence in test_inputs:
        vocabSet.addSentence(sentence)

    word2index = vocabSet.word2index
    index2word = vocabSet.index2word

    print("Saving dictionary...")
    word2index_path = os.path.join(dict_path_prefix, 'word2index.npy')
    index2word_path = os.path.join(dict_path_prefix, 'index2word.npy')

    save_dict(word2index, word2index_path)
    save_dict(index2word, index2word_path)

# 保存词典到npy
def save_dict(dictionary, dict_path):
    np.save(dict_path, dictionary)

# 加载词典到内存
def load_dict(dict_path):
    return np.load(dict_path, allow_pickle=True).item()


if __name__ == "__main__":
    save_word_index_dict("./processed_data/dicts")
    save_all_data_samples("./processed_data")




