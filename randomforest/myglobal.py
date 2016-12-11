# Algorithm
import random
from copy import deepcopy

ID3 = 1
C4_5 = 2
CART = 3

# modify in order to support text attribute
string2intMap = dict()
word_cnt = 0


def printcnt():
    print(word_cnt)


def isnumber(value):
    # return isinstance(value, int) or isinstance(value, float)
    return value.isnumeric()


def getWordLabel(word):
    if word in string2intMap:
        return string2intMap[word]
    else:
        global word_cnt
        word_cnt += 1
        string2intMap[word] = word_cnt
        return word_cnt


def getLabel(word):
    if isnumber(word):
        return int(word)
    else:
        return getWordLabel(word)


# ----------------------------------

class Row:
    def __init__(self, my_set, label):
        self.attr = my_set
        self.label = label
        self.predict = None
        self.mid_predict = [0, 0]


class DataSet:
    def __init__(self, lines, file_type):
        self.positive_threshold = 0.4
        self.type = file_type
        self.rows = []
        self.remove_index = []

        # random delete some attribute
        if file_type == "TrainSet":
            # remove_cnt = random.randint(2, 3)
            remove_cnt = 2
            remove_up = 4
            for i in range(remove_cnt):
                temp_remove_index = random.randint(0, len(lines[0].split(',')) - remove_up)
                remove_up += 1
                self.remove_index.append(temp_remove_index)

                # temp_remove_index1 = random.randint(0, len(lines[0].split(',')) - 4)
                # temp_remove_index2 = random.randint(0, len(lines[0].split(',')) - 5)
                # temp_remove_index3 = random.randint(0, len(lines[0].split(',')) - 6)
                # self.remove_index.append(temp_remove_index1)
                # self.remove_index.append(temp_remove_index2)
                # self.remove_index.append(temp_remove_index3)

        for line in lines:
            line = line.split(',')
            del line[2:4]
            line = [getLabel(word) for word in line]
            if file_type == "TestSet":
                row = Row(line[:-1], None)
            elif file_type == "TrainSet":
                for index in self.remove_index:
                    del line[index]  # delete some attribute
                row = Row(line[:-1], line[-1])
            else:
                row = Row(line[:-1], line[-1])

            self.rows.append(row)

    def start_classify(self, tree_root_list, trees_dataset):
        for row in self.rows:
            # for tree_root in tree_root_list:
            for i in range(len(tree_root_list)):
                temp_row = deepcopy(row)
                tree_root = tree_root_list[i]
                remove_index = trees_dataset[i]
                for index in remove_index:
                    del temp_row.attr[index]
                row.mid_predict[classify(temp_row, tree_root)] += 1

            positive_ratio = float(row.mid_predict[1]) / (row.mid_predict[0] + row.mid_predict[1])
            if positive_ratio > self.positive_threshold:
                row.predict = 1
            else:
                row.predict = 0

    def print_out_result(self):
        for row in self.rows:
            print(row.predict)


def classify(row, tree_root):
    """
    classify the given row by the given decision tree

    :param row: Row object
    :param tree_root: the decision tree root node
    :return: which class it should be
    """
    my_row = row.attr
    node = tree_root
    while node.label is None:
        category_index = get_category_index(node.separate_value, my_row[node.col])
        node = node.child[category_index]

    return node.label


def get_category_index(separate_value, value):
    """
    find out which range index the value lay within the separate_value list

    :param separate_value:
    :param value: target value
    :return: the index
    """
    SUBSET_NUM = len(separate_value) + 1
    category_index = -1
    for i in range(SUBSET_NUM - 1):
        if value < separate_value[i]:
            category_index = i
            break

    if category_index == -1:
        category_index = SUBSET_NUM - 1

    return category_index
