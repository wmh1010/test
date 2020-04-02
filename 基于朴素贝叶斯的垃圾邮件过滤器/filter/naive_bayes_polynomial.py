# 信息内容安全作业-朴素贝叶斯邮件过滤（多项式模型）
# SC011701-2017302232-罗倩倩

import os
import jieba
import random

from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix




# 提取 50 封邮件并标注好类型（正常邮件为1）
def get_email():
    global email_list
    global email_num

    # 提取正常邮件，并标注类型为1
    for root, dirs, files in os.walk(ham_path):
        for fn in files:
            email_list.append([fn,1])
            email_num += 1

    # 提取垃圾邮件，并标注类型为0
    for root, dirs, files in os.walk(spam_path):
        for fn in files:
            email_list.append([fn,0])
            email_num += 1


# 从 spam 和 pam 集中随机取 10 封邮件作为测试集，剩下 40 封作为训练集
# （并标注类型、记录各类型邮件数量）
def get_training_set(email_list):
    global training_set
    global test_set
    global ham_training_num
    global spam_training_num
    global email_training_num
    training_set = email_list[:]
    email_training_num = email_num

    for i in range(10):
        s = int( random.uniform(0, len(training_set)) )
        if training_set[s][1] == 1:
            ham_training_num -= 1
        else:
            spam_training_num -= 1
        email_training_num -= 1
        test_set.append(training_set[s])
        del (training_set[s])

    print("测试集：")
    print(test_set)


# 计算先验概率 P(y)
def get_priori_possibility():
    global py1
    global py2

    py1 = round( (ham_training_num + a) / (email_training_num + 2 * a), 2)
    py2 = round( (spam_training_num + a) / (email_training_num + 2 * a), 2)


# 判断字符是否为英文字母（是则返回 True）
def is_alphabet(char):
    if (char >= u'\u0041' and char <= u'\u005a') or (char >= u'\u0061' and char <= u'\u007a'):
        return True
    else:
        return False


# 标准化处理（除去无关字符）
def standardize(text):
    st_text = ''
    for char in text:
        if is_alphabet(char) or char == ' ' or char == '\t' or char == '\n' or char == '\r':
            st_text = st_text + char

    '''
    with open(st, 'a') as s:
        s.write("\n原语料内容：\n%s" % text)
        s.write("\n-------------------------------------------------------------------")
        s.write("\n标准化处理结果：\n%s" % st_text)
        s.close()
    '''

    return st_text


# jieba 分词
def divide(st_text):
    di_list = jieba.lcut(st_text, cut_all = False)       # 精准模式分词

    '''
    with open(st, 'a') as s:
        s.write("\n-------------------------------------------------------------------")
        s.write("\n分词结果：\n")
        s.write('/'.join(di_list))
        s.close()
    '''

    return di_list


# 使用停用词表进行过滤
def delete_stopword(di_list):
    de_list = []
    # global essay_word
    with open(stopword_list, 'r', encoding = 'utf-8') as c:
        stopwords = c.read()
        c.close()
    for word in di_list:
        # essay_word += 1
        if word in stopwords:
            continue
        else:
            de_list.append(word)

    '''
    with open(st, 'a') as s:
        s.write("\n-------------------------------------------------------------")
        s.write("\n过滤结果：\n")
        s.write('/'.join(de_list))
        s.close()
    '''

    return de_list


# 统计出现各特征（词）的正常/垃圾邮件数
def get_Nya():
    global ham_word
    global spam_word
    global ham_word_num
    global spam_word_num
    ham_word_num = 0
    spam_word_num = 0

    # 按类获取邮件内容
    for i in training_set:
        if i[1] == 1:
            #print(i[0],i[1])
            with open(ham_path+'/'+i[0], 'r', errors = 'ignore') as f:
                text = f.read()
                #text = text.lower()
                f.close()
            #print(text)
        else:
            with open(spam_path+'/'+i[0], 'r', errors = 'ignore') as f:
                text = f.read()
                #text = text.lower()
                f.close()

        # 对邮件内容进行预处理及分词
        st_text = standardize(text)
        di_list = divide(st_text)
        de_list = delete_stopword(di_list)

        # 统计单词出现次数
        erase = []
        if i[1] == 1:
            for d in range( len(de_list) ):
                if de_list[d] in erase:
                    continue
                else:
                    erase.append(de_list[d])
                    temp = ham_word.get(de_list[d])
                    t0 = spam_word.get(de_list[d])
                    if temp:
                        ham_word[de_list[d]] += 1
                    else:
                        ham_word[de_list[d]] = 1
                        ham_word_num += 1
                    if t0:
                        pass
                    else:
                        spam_word[de_list[d]] = 0

        else:
            for d in range( len(de_list) ):
                if de_list[d] in erase:
                    continue
                else:
                    temp = spam_word.get(de_list[d])
                    t0 = ham_word.get(de_list[d])
                    if temp:
                        spam_word[de_list[d]] += 1
                    else:
                        spam_word[de_list[d]] = 1
                        spam_word_num += 1
                    if t0:
                        pass
                    else:
                        ham_word[de_list[d]] = 0


# 测试邮件过滤
def filter():
    wrong = 0
    #print(spam_word)
    #print(ham_word)

    for i in test_set:
        pxy1 = 1.0000000
        pxy2 = 1.0000000

        # 获取测试邮件内容
        if i[1] == 1:
            with open(ham_path + '/' + i[0], 'r', errors='ignore') as f:
                text = f.read()
                #text = text.lower()
                f.close()
        else:
            with open(spam_path + '/' + i[0], 'r', errors='ignore') as f:
                text = f.read()
                #text = text.lower()
                f.close()

        # 处理邮件内容
        st_text = standardize(text)
        di_list = divide(st_text)
        de_list = delete_stopword(di_list)
        print("------------------------------------------")
        print("de_list：")
        print(de_list)

        # 对每个单词（即特征）计算 pyx
        for d in range( len(de_list) ):
            t1 = ham_word.get(de_list[d])
            t2 = spam_word.get(de_list[d])

            if t1 or t1 == 0:
                print(de_list[d])
                nya = ham_word[de_list[d]]
                temp = round( (nya + a) / (ham_word_num + 2 * a), 7)
                pxy1 = pxy1 * temp
                print("t1：", nya)
                print("pay：", temp)
                print("pxy1：", pxy1)
            if t2 or t2 == 0:
                nya = spam_word[de_list[d]]
                temp = round( (nya + a) / (spam_word_num + 2 * a), 7)
                pxy2 = pxy2 * temp
                print("t2：", nya)
                print("pay：", temp)
                print("pxy2：", pxy2)

        # 根据 p(x|yi) 计算 p(yi|x)（后验概率）
        pyx1 = pxy1 * py1
        pyx2 = pxy2 * py2
        print("pxy1,pxy2：", pxy1, pxy2)
        print("pyx1,pyx2：", pyx1, pyx2)

        # 判断邮件类别（注意不漏掉重要邮件）
        if pyx1 >= pyx2:
            flag = 1
        else:
            flag = 0

        # 验证判断是否正确
        if flag != i[1]:
            # 判断错误
            wrong += 1

        print("处理项：", i[0])
        print("判断结果：", flag)
        print("实际类别：", i[1])

    wrong_rate = wrong / (email_num - email_training_num)

    print("本次运行错误率为：")
    print(wrong_rate)

'''
## SVM 算法
# 提取文件内容并创建词典
def get_dict():
    global svm_dict
    all_words = []

    for i in training_set:
        if i[1] == 1:
            with open(ham_path+'/'+i[0], 'r', errors = 'ignore') as f:
                for line in f:
                    words = line.split()
                    all_words += words
                f.close()
        else:
            with open(spam_path+'/'+i[0], 'r', errors = 'ignore') as f:
                for line in f:
                    words = line.split()
                    all_words += words
                f.close()

    svm_dict = Counter(all_words)
    
    # 删去无关字符
    
    useless = svm_dict.keys()
    for i in useless:
        if i.isalpha() == False:
            del svm_dict[i]
        elif len(i) == 1:
            del svm_dict[i]
    

    svm_dict = svm_dict.most_common(3000)

    return svm_dict

# 特征提取
def get_feature():
    files = [os.path.join(email_path,fi) for fi in os.listdir(email_path)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0

    for i in training_set:
        if i[1] == 1:
            with open(ham_path+'/'+i[0], 'r', errors = 'ignore') as fi:
                for line in fi:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(svm_dict):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
                docID = docID + 1
                fi.close()
        else:
            with open(spam_path+'/'+i[0], 'r', errors = 'ignore') as fi:
                for line in fi:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(svm_dict):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
                docID = docID + 1
                fi.close()

    return features_matrix
'''



if __name__ == '__main__':
    # 邮件及停用词表文件
    stopword_list = 'baidu_stopwords.txt'
    # st = '处理结果.txt'
    email_path = '../filter/email'
    ham_path = '../filter/email/ham'
    spam_path = '../filter/email/spam'
    email_num = 0               # 邮件总数
    email_list = []             # 全部邮件
    training_set = []           # 训练集
    test_set = []               # 测试集
    ham_training_num = 25       # 训练的正常邮件数（预设25）
    spam_training_num = 25      # 训练的垃圾邮件数（预设25）
    email_training_num = 0      # 训练的总邮件数
    ham_word = {}               # 特征（词）在正常邮件中的出现次数
    spam_word = {}              # 特征（词）在垃圾邮件中的出现次数
    ham_word_num = 0
    spam_word_num = 0
    py1 = 0
    py2 = 0
    a = 1                       # 拉普拉斯平滑


    get_email()
    get_training_set(email_list)
    get_priori_possibility()
    get_Nya()
    filter()

    '''
    # svm 算法
    get_email()
    get_training_set(email_list)
    svm_dict = get_dict()
    train_labels = np.zeros(702)
    train_labels[351:701] = 1
    train_matrix = get_feature()

    model1 = MultinomialNB()
    model2 = LinearSVC()
    model1.fit(train_matrix, train_labels)
    model2.fit(train_matrix, train_labels)

    test_dir = '../filter/email/ham'
    test_matrix = get_feature(test_dir)
    test_labels = np.zeros(260)
    test_labels[130:260] = 1
    result1 = model1.predict(test_matrix)
    result2 = model2.predict(test_matrix)
    print(confusion_matrix(test_labels, result1))
    print(confusion_matrix(test_labels, result2))
    '''


