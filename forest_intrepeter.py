from conjunction import *
from utils import *
import xgboost
import numpy as np

SKLEARN_PROBAS_FORESTS = [RandomForestClassifier,ExtraTreesClassifier]
SKLEARN_LOGIT_FORESTS = [GradientBoostingClassifier]

class ForestLocalInterpreter():
    def __init__(self,classes,feature_names):
        self.classes = classes
        self.feature_names = feature_names

    def fit(self,model):
        self.model = model
        if type(self.model) in SKLEARN_PROBAS_FORESTS:
            self.fit_sklearn_probas_forest()
        elif type(self.model) in SKLEARN_LOGIT_FORESTS:
            self.fit_sklearn_logit_forest()
        elif type(self.model) == xgboost.sklearn.XGBClassifier:
            self.fit_xgboost()

    def get_prediction_rules(self,X):
        output = []
        for inst in X:
            conj = Conjunction(self.feature_names, self.classes, leaf_index=[], class_values=np.zeros(len(self.classes)),
                               output_type=self.output_type)  # Define the conjunction
            for tree_index, tree in enumerate(self.trees_conjunctions):
                for leaf_index, leaf in enumerate(tree):
                    if leaf.containsInstance(inst):
                        conj = conj.merge(leaf)
            output.append(conj.toString())

        return output
    def fit_sklearn_probas_forest(self):
        print('aaaaaaa')
        self.trees_conjunctions = extract_trees_from_sklearn_probas_ensemble(self.model, self.classes, self.feature_names)
        self.output_type = 'probas'
    def fit_sklearn_logit_forest(self):
        self.trees_conjunctions = extract_trees_from_sklearn_logit_ensemble(self.model, self.classes, self.feature_names)
        self.output_type = 'logit'
    def fit_xgboost(self):
        self.trees_conjunctions = extract_trees_from_xgb(self.model, self.classes, self.feature_names)
        self.output_type = 'logit'
