from DT import *
import multiprocessing

# parameters
my_type = C4_5
train_set_ratio = 0.95
tree_subset_data_ratio = 0.1
best_model_threshold = 0.7
tree_num = 1000
bestF1 = 0


def import_data_set(path):
    with open(path) as file:
        lines = file.read().splitlines()[1:]
        train_set_data = random.sample(lines, round(len(lines) * train_set_ratio))
        for line in train_set_data:
            lines.remove(line)
        return train_set_data, lines


def import_test_set(path):
    with open(path) as file:
        lines = file.read().splitlines()[1:]
        return lines


def generate_dataSet(lines, ratio):
    selected_lines = random.sample(lines, round(len(lines) * ratio))
    return DataSet(selected_lines, 'TrainSet')


class RandomForest:
    def __init__(self):
        self.trees = list()
        self.trees_dataset = list()

    def generate_trees(self, lines, num, ratio):
        for i in range(num):
            sub_dataset = generate_dataSet(lines, ratio)
            sub_tree = build_tree(sub_dataset.rows, get_entropy, my_type)
            self.trees.append(sub_tree)
            self.trees_dataset.append(sub_dataset.remove_index)
            # print("sub tree #"+str(i)+" finish")


def write_result_to_file(evaluate, test_set):
    with open('./results/' + 'trees-' + str(tree_num) + '_ratio-' + str(tree_subset_data_ratio) + '_f1-' + str(
            round(evaluate.F1, 3)) + '.txt', 'w+') as f:
        f.write("##########################" + '\n')
        f.write("trainset ratio: " + str(train_set_ratio) + '\n')
        f.write("accuracy: " + str(evaluate.accuracy) + '\n')
        f.write("precision: " + str(evaluate.precision) + '\n')
        f.write("recall: " + str(evaluate.recall) + '\n')
        f.write("F1: " + str(evaluate.F1) + '\n')
        f.write("TP:" + str(evaluate.TP) + " FN:" + str(evaluate.FN) + " FP:" + str(evaluate.FP) + " TN:" + str(
            evaluate.TN) + '\n')
        f.write("ALL:" + str(evaluate.TP + evaluate.FN + evaluate.FP + evaluate.TN) + '\n')
        f.write("##########################" + '\n')
        f.write('\n')
        for row in test_set.rows:
            f.write(str(row.predict))
            f.write('\n')
    f.close()


def unit(test_set_data_inner):
    global bestF1
    trains_set_data_inner, verify_set_data_inner = import_data_set('./train.csv')
    # build some trees using different date
    # print("Build forest start")
    forest = RandomForest()
    forest.generate_trees(trains_set_data_inner, tree_num, tree_subset_data_ratio)
    # print("Build forest finish")

    # classify using these trees & majority voting to get results
    # print("classify start")
    verify_set = DataSet(verify_set_data_inner, 'VerifySet')
    verify_set.start_classify(forest.trees, forest.trees_dataset)
    # print("classify finish")

    # print('my_type: ' + str(my_type) + ' train_set_ratio: ' + str(train_set_ratio)
    #       + ' tree_subset_data_ratio: ' + str(tree_subset_data_ratio) + ' tree_num: ' + str(tree_num))
    evaluate_inner = Evaluate(verify_set)
    evaluate_inner.calculte()
    print(evaluate_inner.F1)

    if evaluate_inner.F1 > bestF1:
        bestF1 = evaluate_inner.F1

    if evaluate_inner.F1 > best_model_threshold:
        test_set_inner = DataSet(test_set_data_inner, 'TestSet')
        test_set_inner.start_classify(forest.trees, forest.trees_dataset)
        write_result_to_file(evaluate_inner, test_set_inner)


if __name__ == "__main__":
    mode = 0

    if mode == 0:
        # particular generate
        train_set_ratio = 0.9
        tree_subset_data_ratio = 0.15
        best_model_threshold = 0.7
        tree_num = 100

        test_set_data = import_test_set('./test.csv')
        print('my_type: ' + str(my_type) + ' train_set_ratio: ' + str(train_set_ratio)
              + ' tree_subset_data_ratio: ' + str(tree_subset_data_ratio) + ' tree_num: ' + str(tree_num))
        process_list = []
        for i in range(1, 8):
            t = multiprocessing.Process(target=unit, args=(test_set_data,))
            # t = multiprocessing.Process()
            process_list.append(t)

        for process in process_list:
            process.start()

        for process in process_list:
            process.join()

    else:
        # multiprocess random generate
        while train_set_ratio > 0.7:
            train_set_ratio -= 0.05
            test_set_data = import_test_set('./test.csv')
            # init
            tree_num = 1500
            while tree_num <= 3000:
                tree_subset_data_ratio = 0.05
                while tree_subset_data_ratio <= 0.1:
                    tree_subset_data_ratio += 0.05
                    print('my_type: ' + str(my_type) + ' train_set_ratio: ' + str(train_set_ratio)
                          + ' tree_subset_data_ratio: ' + str(tree_subset_data_ratio) + ' tree_num: ' + str(tree_num))
                    process_list = []
                    for i in range(0, 8):
                        t = multiprocessing.Process(target=unit, args=test_set_data)
                        # t = multiprocessing.Process()
                        process_list.append(t)

                    for process in process_list:
                        process.start()

                    for process in process_list:
                        process.join()
                tree_num += 500
