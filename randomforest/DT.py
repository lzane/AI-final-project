import operator
from collections import deque
from math import log
from myglobal import *
from evaluate import *


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
    train_set = DataSet("./train.csv", "TrainSet", 1, 1000)
    test_set = DataSet("./train.csv", "TrainSet", 30000, 30163)
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
