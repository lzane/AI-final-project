import operator
from evaluate import Evaluate
from collections import deque
from math import log

# Algorithm
ID3 = 1
C4_5 = 2
CART = 3

# modify in order to support text attribute
string2intMap = dict()
word_cnt = 0


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


class DataSet:
    def __init__(self, path, file_type, left, right):
        self.type = file_type
        self.rows = []
        with open(path) as file:
            for line in file.read().splitlines()[left:right]:
                line = line.split(',')
                line = [getLabel(word) for word in line]
                # line = list(map(int, line))  # Parse to int
                if file_type == "TestSet":
                    row = Row(line[:], None)
                else:
                    row = Row(line[:-1], line[-1])
                self.rows.append(row)

    def start_classify(self, tree_root):
        for row in self.rows:
            row.predict = classify(row, tree_root)

    def check_accuracy(self):
        """
        :return: first the accuracy
                 second the correct count
        """
        cnt = 0
        for row in self.rows:
            if row.predict == row.label:
                cnt += 1

        res = float(cnt) / len(self.rows)
        return res, cnt

    def print_out_result(self):
        for row in self.rows:
            print(row.predict)


def get_separate_value(rows, col):
    """
    get separate value list order by increasing

    :param rows: [Row]
    :param col: target column
    :return: separated value array(list)
    """
    result_set = set()
    for row in rows:
        result_set.add(row.attr[col])

    result_list = sorted(result_set)
    separate_list = list()
    for i in range(len(result_list) - 1):
        separate_list.append((result_list[i] + result_list[i + 1]) / 2)
    return separate_list


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


def divide_set(rows, col, separate_value):
    """
    divide rows into several subset based on the col value

    :param rows: [Row]
    :param col: attribute considered
    :param separate_value: [Int] a list contains separate values
    :return: [[Row]] list of all subset
    """

    SUBSET_NUM = len(separate_value) + 1
    result_list = [[] for i in range(SUBSET_NUM)]

    for row in rows:
        category_index = get_category_index(separate_value, row.attr[col])
        result_list[category_index].append(row)

    return result_list


def unique_result_counts(rows):
    """
    calculate the count of possible results

    :param rows: [Row]
    :return: a dict contains all results and its count
    """
    results = {}
    for row in rows:
        if row.label not in results:
            results[row.label] = 0
        results[row.label] += 1

    return results


def unique_column_attribute_counts(rows, col):
    """
    calculate the count of possible results according to given column
    used for C4.5 algorithm.

    :param rows:
    :param col:
    :return:
    """
    results = {}
    for row in rows:
        if row.attr[col] not in results:
            results[row.attr[col]] = 0
        results[row.attr[col]] += 1

    return results


def log2(x):
    return log(x) / log(2)


def get_entropy(rows):
    """
    calculate entropy of given rows

    :param rows: [Row]
    :return: entropy
    """

    results = unique_result_counts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)

    return ent


def get_CART(rows):
    """
    calculate CART of given rows

    :param rows: [Row]
    :return: cart
    """

    results = unique_result_counts(rows)
    ent = 1
    for r in results.keys():
        p = (float(results[r]) / len(rows)) ** 2
        ent -= p

    return ent


def get_column_entropy(rows, col):
    """
    Use for C4.5 to calculate the column entropy

    :param rows:
    :param col:
    :return :
    """
    results = unique_column_attribute_counts(rows, col)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)

    return ent


class Node:
    def __init__(self, col=-1, separate_value={}):
        self.col = col
        self.separate_value = separate_value
        self.child = []
        self.label = None


def build_tree(rows, scoref=get_entropy, type=ID3):
    """
    Build a decision tree.

    :param type: which algorithm
    :param rows: [Row]
    :param scoref: entropy calculate function
    :return: the root Node of the decision tree
    """

    # choose a column with biggest gain
    original_entropy = scoref(rows)
    column_count = len(rows[0].attr)
    best_gain = 0
    best_gain_col = -1
    best_gain_divide = []

    for col in range(column_count):
        separate_value = get_separate_value(rows, col)
        divided = divide_set(rows, col, separate_value)

        new_entropy = 0.0
        for new_rows in divided:
            temp_entropy = scoref(new_rows)
            p = float(len(new_rows)) / len(rows)
            new_entropy += temp_entropy * p

        # calculate the gain based on the algorithm
        if type == ID3:
            gain = original_entropy - new_entropy
        elif type == C4_5:  # C4.5  divided by SplitInfo
            gain = (original_entropy - new_entropy) / get_column_entropy(rows, col)
        elif type == CART:  # CART
            gain = 1 - new_entropy

        if gain > best_gain:
            best_gain = gain
            best_gain_col = col
            best_gain_divide = divided
            best_gain_separate = separate_value

    # check whether reach the leaf node
    if best_gain != 0:
        # recursively build tree
        node = Node(best_gain_col, best_gain_separate)
        for rows in best_gain_divide:
            node.child.append(build_tree(rows))
    else:
        # already reach the leaf
        results = unique_result_counts(rows)
        node = Node()
        sorted_results = sorted(results.items(), key=operator.itemgetter(1))
        node.label = sorted_results[-1][0]  # majority voted principle

    # print(node.child)
    return node


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


def calculate_tree_error_rate(tree_root, beta, data_set):
    """
    calculate the error rate of decision tree

    :param tree_root: decision tree root
    :param beta: punishment value
    :param data_set: the train set
    :return:
    """
    data_set.start_classify(tree_root)
    total_count = len(data_set.rows)
    acc, correct_count = data_set.check_accuracy()
    leaf_node_count = calculate_leaf_node_count(tree_root)
    error = (total_count - correct_count + leaf_node_count * beta) / total_count
    return error


def calculate_leaf_node_count(tree_root):
    """
    calculate leaf node count of a decision tree
    use BFS

    :param tree_root:
    :return:
    """
    count = 0
    queue = deque([])
    queue.append(tree_root)
    while queue:
        temp_node = queue.popleft()
        if temp_node.label is not None:
            count += 1
        for child_node in temp_node.child:
            queue.append(child_node)

    return count


if __name__ == "__main__":
    print('readfile start')
    train_set = DataSet("../数据集/二元分类（30162：15060）：成年人收入/train.csv", "TrainSet", 1, 25000)
    test_set = DataSet("../数据集/二元分类（30162：15060）：成年人收入/train.csv", "TrainSet", 25000, 30163)
    my_rows = train_set.rows
    print('readfile finish')

    # my_type = ID3
    my_type = C4_5
    # my_type = CART

    print('build_tree start')
    root = build_tree(my_rows, get_entropy, my_type)
    print('build_tree finish')
    print('classify start')
    test_set.start_classify(root)
    print('classify finish')

    evaluate = Evaluate(test_set)
    evaluate.print()
    # res = test_set.check_accuracy()
    # print("(accuracy, correct count) = " + str(res))
