from DT import *

# parameters
my_type = C4_5
train_set_ratio = 0.9

tree_subset_data_ratio = 0.4
tree_num = 30


def import_data_set(path):
    with open(path) as file:
        lines = file.read().splitlines()[1:]
        train_set_data = random.sample(lines, round(len(lines) * train_set_ratio))
        for line in train_set_data:
            lines.remove(line)
        return train_set_data, lines


def generate_dataSet(lines, ratio):
    selected_lines = random.sample(lines, round(len(lines) * ratio))
    return DataSet(selected_lines, 'TrainSet')


class RandomForest:
    def __init__(self):
        self.trees = list()

    def generate_trees(self, lines, num, ratio):
        for i in range(num):
            sub_dataset = generate_dataSet(lines, ratio)
            sub_tree = build_tree(sub_dataset.rows, get_entropy, my_type)
            self.trees.append(sub_tree)
            print("sub tree #"+str(i)+" finish")


if __name__ == "__main__":
    # build some trees using different date
    print("Build forest start")
    trains_set_data, verify_set_data = import_data_set('./train.csv')
    forest = RandomForest()
    forest.generate_trees(trains_set_data, tree_num, tree_subset_data_ratio)
    print("Build forest finish")

    # classify using these trees & majority voting to get results
    print("classify start")
    verify_set = DataSet(verify_set_data, 'VerifySet')
    verify_set.start_classify(forest.trees)
    print("classify finish")

    evaluate = Evaluate(verify_set)
    evaluate.print()
