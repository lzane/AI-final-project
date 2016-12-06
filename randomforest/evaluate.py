class Evaluate:
    def __init__(self, data_set):
        self.dataSet = data_set
        self.accuracy = None
        self.recall = None
        self.precision = None
        self.F1 = None
        self.TP = None
        self.FN = None
        self.FP = None
        self.TN = None

    def calculte(self):
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for sample in self.dataSet.rows:
            if sample.label == 1:
                if sample.predict == 1:
                    TP += 1
                elif sample.predict == 0:
                    FN += 1
            elif sample.label == 0:
                if sample.predict == 1:
                    FP += 1
                elif sample.predict == 0:
                    TN += 1

        self.accuracy = (TP + TN) / (TP + FP + TN + FN)
        self.recall = TP / (TP + FN)
        self.precision = TP / (FP + TP)
        self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
        self.TP = TP
        self.FN = FN
        self.FP = FP
        self.TN = TN

    def print(self):
        self.calculte()
        print("##########################")
        print("accuracy: " + str(self.accuracy))
        print("precision: " + str(self.precision))
        print("recall: " + str(self.recall))
        print("F1: " + str(self.F1))
        print("TP:" + str(self.TP) + " FN:" + str(self.FN) + " FP:" + str(self.FP) + " TN:" + str(self.TN))
        print("ALL:" + str(self.TP+self.FN+self.FP+self.TN))
        print("##########################")
