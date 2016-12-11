import math

emotion_cnt = 1
emotion_map = dict()
cnt_emotion_map = dict()


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

    def read_file(self, path):
        with open(path) as file:
            lines = file.read().splitlines()
            # lines = lines[1:]  # skip the first sentence
            for line in lines:
                temp = Text()
                line = line.split(',')
                temp.emotions.append(get_emotion_num(line[0]))
                line = line[1]
                if self.type == 'Classification':
                    line = line.split()

                for word in line:
                    temp.ori_words.append(word)
                    self.word_set.add(word)
                    if word in temp.words:
                        temp.words[word] += 1
                    else:
                        temp.words[word] = 1
                self.texts.append(temp)

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

                # print("wordset")
                # print(len(self.train.wordset))
                # print("no_repetitive")
                # print(self.train.no_repetitive_words[e])

            res *= (self.train.emotion_text_cnt[e] / self.train.lineCnt)
            emotion_list[e] = res

        max_value = max(emotion_list)
        max_index = emotion_list.index(max_value)
        text.predicts.append(max_index)

    def classify(self):
        for text in self.test.texts:
            self.judge(text)

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
        print('rate:  ' + str(accuracy_rate))

        return accuracy_rate


# Classify:
trainSet = TextSet('Classification')
trainSet.read_file('train copy.txt')
trainSet.calculate_words_and_emotions()
testSet = TextSet('Classification')
testSet.read_file('train copy.txt')
classify = Classification(trainSet, testSet)
classify.classify()
for text in testSet.texts:
    print(get_emotion_label(text.predicts[0]))
classify.check_accuracy()
