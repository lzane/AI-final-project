import math
import multiprocessing
import random
from autocorrect import spell
from copy import deepcopy

# parameters
train_set_ratio = 0.95

emotion_cnt = 1
emotion_map = dict()
cnt_emotion_map = dict()
stop_words_list = list()


def import_data(path):
    with open(path, 'rt', encoding='utf-8') as file:
        lines = file.read().splitlines()
        return lines

def data_preprocessing(lines):
    texts = list()
    for line in lines:
        temp = Text()
        line = line.split(',')
        temp.emotions.append(get_emotion_num(line[0]))
        line = line[1]
        line = line.split()

        for word in line:
            corrected_word = spell(word)  # correct words
            # corrected_word = word
            if corrected_word in stop_words_list:  # remove stop words
                continue
            temp.ori_words.append(corrected_word)
            if corrected_word in temp.words:
                temp.words[corrected_word] += 1
            else:
                temp.words[corrected_word] = 1

        texts.append(temp)
    return texts


def random_generate_data(texts):
    new_texts = deepcopy(texts)
    train_set_texts_list = random.sample(new_texts, round(len(new_texts) * train_set_ratio))
    for text in train_set_texts_list:
        new_texts.remove(text)
    return train_set_texts_list, new_texts


def get_emotion_num(emotion):
    global emotion_cnt, emotion_map, cnt_emotion_map
    if emotion not in emotion_map:
        cnt_emotion_map[emotion_cnt] = emotion
        emotion_map[emotion] = emotion_cnt
        emotion_cnt += 1

    return emotion_map[emotion]


def get_emotion_label(num):
    global cnt_emotion_map
    return cnt_emotion_map[num]


class Text:
    def __init__(self):
        self.ori_words = list()
        self.words = dict()
        self.emotions = list()
        self.predicts = list()


class TextSet:
    def __init__(self, set_type):
        self.texts = list()
        self.lineCnt = None
        self.type = set_type
        self.total_words = list()  # words within according emotions
        self.no_repetitive_words = list()  # unique words within according emotions
        self.emotion_text_cnt = list()  # texts count of emotions
        self.word_set = set()

    def read_data(self, texts):
        for text in texts:
            for word in text.words:
                self.word_set.add(word)
            self.texts.append(text)
        self.lineCnt = len(self.texts)

    def calculate_words_and_emotions(self):
        self.total_words = [0] * (emotion_cnt + 1)
        # total_words_set_list = list()
        self.emotion_text_cnt = [0] * (emotion_cnt + 1)
        self.no_repetitive_words = [0] * (emotion_cnt + 1)
        # for i in range(1, total_emotion_cnt+1):
        # total_words_set_list.append(set())

        for text in self.texts:
            e = text.emotions[0]
            self.emotion_text_cnt[e] += 1
            self.total_words[e] += len(text.ori_words)
            # total_words_set_list[e].update(text.oriwords)

            # for i in range(1, total_emotion_cnt + 1):
            # self.no_repetitive_words[i] = len(total_words_set_list[i])
            # print(self.total_words)
            # print(self.no_repetitive_words)


class Classification:
    def __init__(self, train_set, test_set):
        self.train = train_set
        self.test = test_set

    def judge(self, text):
        emotion_list = [0] * (emotion_cnt + 1)
        for e in range(1, emotion_cnt):
            res = 1
            for test_word in text.ori_words:
                word_appear_cnt = 0
                for train_text in self.train.texts:
                    if train_text.emotions[0] == e:
                        if test_word in train_text.words:
                            word_appear_cnt += train_text.words[test_word]
                res = res * (word_appear_cnt + 1) / \
                      (self.train.total_words[e] + len(self.train.word_set))

                # res = res * (word_appear_cnt + 1) / \
                #       (self.train.total_words[e] + self.train.no_repetitive_words[e])

            res *= (self.train.emotion_text_cnt[e] / self.train.lineCnt)
            emotion_list[e] = res

        max_value = max(emotion_list)
        max_index = emotion_list.index(max_value)
        if max_index == 0:
            print("!")
            max_index = 1
        text.predicts.append(max_index)
        return max_index

    def classify(self):
        result = multiprocessing.Pool().map(self.judge, [text for text in self.test.texts])
        result_index = 0
        for text in self.test.texts:
            text.predicts.append(result[result_index])
            result_index += 1

    def check_accuracy(self):
        cnt = 0
        cnt2 = 0
        cnt3 = 0
        for text in self.test.texts:
            if text.emotions[0] == text.predicts[0]:
                cnt += 1
            elif text.predicts[0] == -1:
                cnt3 += 1
            else:
                cnt2 += 1

        accuracy_rate = float(cnt) / len(self.test.texts)
        print('accuracy rate:  ' + str(accuracy_rate))

        return accuracy_rate


def write_result_to_file(test_set):
    with open('./results/' + str(id(test_set)) + '.txt', 'w+') as f:
        for text in test_set.texts:
            f.write(get_emotion_label(text.predicts[0]))
            f.write('\n')
    f.close()


def get_final_result():
    raw_data = import_data('train.txt')
    preprocessed_data = data_preprocessing(raw_data)

    trainSet = TextSet('TrainSet')
    trainSet.read_data(preprocessed_data)
    trainSet.calculate_words_and_emotions()
    testSet = TextSet('TestSet')
    test_set_data = import_data('test.txt')
    preprocessed_test_set_data = data_preprocessing(test_set_data)
    testSet.read_data(preprocessed_test_set_data)

    classify = Classification(trainSet, testSet)
    classify.classify()
    write_result_to_file(testSet)


def test():
    print('------start testing model--------')
    raw_data = import_data('train.txt')
    print('Start pre-processing!')
    preprocessed_data = data_preprocessing(raw_data)
    print('---Finish Read data and pre-processing!---')
    for i in range(8):
        test_unit(preprocessed_data)


def test_unit(preprocessed_data):
    # Classify:
    train_set_data, verify_set_data = random_generate_data(preprocessed_data)

    trainSet = TextSet('TrainSet')
    trainSet.read_data(train_set_data)
    trainSet.calculate_words_and_emotions()
    verifySet = TextSet('VerifySet')
    verifySet.read_data(verify_set_data)

    classify = Classification(trainSet, verifySet)
    classify.classify()
    classify.check_accuracy()


def init():
    # mystoplist.txt
    stop_words_data = import_data('mystoplist.txt')
    for word in stop_words_data:
        stop_words_list.append(word)


if __name__ == '__main__':
    # init()
    test()
    # get_final_result()
    print("------finished------")
