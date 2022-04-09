import pandas as pd
import re
import jieba


def data_process(file_path='./RawData/message.csv', sample_num=100, stop_words_path='./RawData/stopword.txt'):
    # 读取数据
    data = pd.read_csv(file_path, header=None, index_col=0)

    # 对列命名，方便后面操作
    data.columns = ['label', 'message']

    # 分别随机抽取sample_num条垃圾短信和非垃圾短信
    spam = data[data['label'] == 1].sample(sample_num)
    normal = data[data['label'] == 0].sample(sample_num)

    # 将以上抽取的数据组合程新的DataFrame
    data_new = pd.concat([spam, normal], axis=0)

    # 数据清洗
    # 去重处理
    data_after_dup = data_new['message'].drop_duplicates()

    # 处理“X”,英文以及数字字符串
    data_after_x = data_after_dup.apply(lambda x: re.sub('[A-Za-z0-9]', '', x))

    # 分词
    data_after_cut = data_after_x.apply(lambda x: jieba.lcut(x))

    # 去除停用词
    # 导入公共停用词
    stop_words = pd.read_csv(stop_words_path, encoding='GB18030', sep='hahaha')
    stop_words = ['≮', '≯', '≠', '≮', ' ', '会', '月', '日', '–'] + list(stop_words.iloc[:, 0])

    data_after_stop = data_after_cut.apply(lambda x: [i for i in x if i not in stop_words])

    labels = data_new.loc[data_after_stop.index, 'label']
    clean_data = data_after_stop.apply(lambda x: ' '.join(x))

    return clean_data, labels, data_after_stop
