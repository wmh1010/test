import numpy as np
import re
import random

sum_of_error = 0
def string2list(string):
    """
    接收一个字符串并将其解析为字符串列表
    :param string: 字符串
    :return: 字符串列表
    """
    list = re.split(r'\W+', string)  # 将特殊符号(非字母、非数字）作为切分标志进行字符串切分
    return [word.lower() for word in list if len(word) > 2]  # 除单个字母外其他单词变成小写

def createVocabList(data_set):
    """
    将切分的实验样本词条整理成不重复的词条列表，即词汇表
    :param data_set: 样本数据集
    :return: vocab_set    词汇表
    """
    vocab_set = set([])  # 创建一个空的词汇表
    for document in data_set:
        vocab_set = vocab_set | set(document)  # 取并集
    return list(vocab_set)


def words_set2vec(vocab_list, input_set):
    """
    根据vocab_list词汇表构建词集模型，将input_set向量化，向量的每个元素为1或0
    :param vocab_list: createVocabList返回的列表
    :param input_set: 切分的词条列表
    :return: vec   文档向量，词集模型
    """
    vec = [0] * len(vocab_list)  # 创建一个所含元素均为0的向量
    for word in input_set:  # 遍历每个词条
        if word in vocab_list:  # 词条若存在于词汇表中则置1
            vec[vocab_list.index(word)] = 1
        else:
            print('The word "%s" is not in the vocabulary!' % word)
    return vec


def words_bag2vec(vocab_list, input_set):
    """
    根据vocab_list词汇表，构建词袋模型
    :param vocab_list: createVocabList返回的列表
    :param input_set: 切分的词条列表
    :return: vec    文档向量，词袋模型
    """
    vec = [0] * len(vocab_list)
    for word in input_set:  # 遍历每个词条
        if word in vocab_list:  # 词条若存在于词汇表中则计数＋1
            vec[vocab_list.index(word)] += 1
    return vec


def trainNB(train_matrix, train_category):
    """
    朴素贝叶斯垃圾邮件过滤器训练函数
    :param train_matrix:训练文档矩阵，即words_bag2vec返回的vec构成的矩阵
    :param train_category:训练类别标签向量
    :returns:p0_vec  正常邮件类的条件概率数组
            p1_vec  垃圾邮件类的条件概率数组
            p_spam  文档属于垃圾邮件类的概率
    """
    train_docs_num = len(train_matrix)  # 训练的文档数目
    words_num = len(train_matrix[0])  # 每篇文档的词条数
    p_spam = sum(train_category) / float(train_docs_num)  # 文档属于垃圾邮件类的概率
    p0_num = np.ones(words_num)
    p1_num = np.ones(words_num)  # 创建numpy.ones数组，词条出现初始化为1，Laplace平滑
    p0_denom = 2.0  # 分母初始化为2，Laplace平滑，防止概率乘积为0
    p1_denom = 2.0
    for i in range(train_docs_num):
        if train_category[i] == 1:  # 统计属于垃圾邮件类的条件概率所需的数据，即P（w0|1),P(w1|1)...
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:  # 统计属于正常邮件类的条件概率所需的数据，即P（w0|0),P(w1|0)...
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p0_vec = np.log(p0_num / p0_denom)  # 取对数防止向下溢出
    p1_vec = np.log(p1_num / p1_denom)
    return p0_vec, p1_vec, p_spam


def classifyNB(vec, p0_vec, p1_vec, p_spam):
    """
    朴素贝叶斯垃圾邮件过滤器分类函数
    :param vec: 待分类的词条数组
    :param p0_vec: 正常邮件类的条件概率数组
    :param p1_vec: 垃圾邮件类的条件概率数组
    :param p_spam: 文档属于垃圾邮件类的概率
    :returns: 0-属于正常邮件类
              1-属于垃圾邮件类
    """
    p1 = sum(vec * p1_vec) + np.log(p_spam)
    p0 = sum(vec * p0_vec) + np.log(1.0-p_spam)
    if p1 > p0:
        return 1
    else:
        return 0


def testNB():
    """
    测试朴素贝叶斯垃圾邮件过滤器，使用朴素贝叶斯进行交叉验证
    :return:
    """
    doc_list = []
    full_text = []
    class_list = []
    for i in range(1, 26):  # 遍历25个txt文件
        word_list = string2list(open('email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(1)  # 标记垃圾邮件为1
        word_list = string2list(open('email/ham/%d.txt' % i, 'r').read())  # 读取每个正常邮件
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)  # 标记正常邮件为0
    vocab_list = createVocabList(doc_list)  # 创建词汇表
    train_set = list(range(50))
    test_set = []  # 创建存储训练集和测试集的索引值的列表
    for i in range(10):  # 从50封邮件中随机选40封作为训练集，10封作为测试集
        index = int(random.uniform(0, len(train_set)))  # 随机选取索引值
        test_set.append(train_set[index])  # 添加测试集的索引值
        del (train_set[index])  # 在训练集列表中删除已添加到测试集的索引值
    # print(train_set, test_set)
    train_mat = []  # 创建训练集矩阵
    train_class = []  # 创建训练集类别标签系向量
    for t in train_set:  # 遍历训练集
        train_mat.append(words_set2vec(vocab_list, doc_list[t]))  # 将生成的词集
        train_class.append(class_list[t])  # 将类别添加到训练集类别标签系向量中
    p0_vec, p1_vec, p_spam = trainNB(np.array(train_mat), np.array(train_class))  # 训练朴素贝叶斯
    error_count = 0  # 错误分类计数器
    for t in test_set:
        word_vector = words_set2vec(vocab_list, doc_list[t])  # 测试集的词集模型
        if classifyNB(np.array(word_vector), p0_vec, p1_vec, p_spam) != class_list[t]:
            error_count += 1  # 错误计数＋1
            print('分类错误的测试集：', doc_list[t])
    print('错误率：%.2f%%' % (float(error_count) / len(test_set) * 100))
    global sum_of_error
    sum_of_error += (float(error_count) / len(test_set) * 100)


if __name__ == '__main__':
    for i in range(1, 101):
        testNB()
        i += 1
    print('平均错误率:', sum_of_error / 100, '%')

