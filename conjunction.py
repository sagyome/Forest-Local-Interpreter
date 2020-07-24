"""
This module contains the conjunction class

"""
import numpy as np
from utils import *

class Conjunction():
    """
    A conjunction is a combination of feature bounds mapped into a class probability vector
    """
    def __init__(self,feature_names,classes,leaf_index=None,class_values=None, output_type='probas'):
        """
        :param feature_names: list of strings. Also determine the dimensionality
        :param label_names: list of labels. Determines the number of labels too
        :param leaf_index: This feature is optional. Can be relevant if we'd like to document the leaves that were used from the input forest
        :param label_probas: also optional. Relevant if we'd like to determine the class probabilities within the constructor
        """
        self.feature_names = feature_names
        self.number_of_features = len(feature_names)
        self.classes_= classes
        self.output_type = output_type

        # upper and lower bounds of the feature for each rule
        self.features_upper = [np.inf] * len(feature_names)
        self.features_lower = [-np.inf] * len(feature_names)

        self.class_values = np.array(class_values)
        self.leaf_index = leaf_index

        #The following dict is used for excluding irrelevant merges of different dummy variables that come from the same categorical feature
        self.categorical_features_dict={}

    def addCondition(self, feature, threshold, bound):
        """
        This method adds a condition to the conjunction if relevant (rule isn't already contained in the conjunction)

        :param feature: relevant feature
        :param threshold: upper\lower bound
        :param bound: bound direction

        """
        #Check if the rule isn't already contained in the conjunction
        if bound == 'lower':
            if self.features_lower[feature] < threshold:
                self.features_lower[feature] = threshold
        else:
            if self.features_upper[feature] > threshold:
                self.features_upper[feature] = threshold

        #Address categorical features:
        if '=' in self.feature_names[feature] and threshold >= 1 and bound == 'lower':
            splitted = self.feature_names[feature].split('=')
            self.categorical_features_dict[splitted[0]] = splitted[1]

    def merge(self, other):
        """
        :param other: conjunction
        :return: new_conjunction - a merge of the self conjunction with other
        """
        new_conjunction = Conjunction(self.feature_names, self.classes_,
                                      self.leaf_index + other.leaf_index, self.class_values + other.class_values,output_type=self.output_type)
        new_conjunction.features_upper = [min(i,j) for i,j in zip(self.features_upper,other.features_upper)]
        new_conjunction.features_lower = [max(i, j) for i, j in zip(self.features_lower, other.features_lower)]
        new_conjunction.categorical_features_dict = self.categorical_features_dict
        new_conjunction.categorical_features_dict.update(other.categorical_features_dict)
        return new_conjunction

    def containsInstance(self,inst):
        """
        Checks whether the input instance falls under the conjunction

        :param inst:
        :return: True if
        """
        for i, lower, upper in zip(range(len(inst)), self.features_lower, self.features_upper):
            if inst[i] >= upper or inst[i] < lower:
                return False
        return True

    def toString(self):
        """
        This function creates a string representation of the conjunction (only for demonstration purposes)
        """
        s = ""
        #save lower bounds
        for feature, threshold in enumerate(self.features_lower):
            if threshold != (-np.inf):
                s +=  self.feature_names[feature] + ' >= ' + str(np.round(threshold,3)) + ", "
        #save upper bounds
        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                s +=  self.feature_names[feature] + ' < ' + str(np.round(threshold,3)) + ", "
        #save labels
        s += 'labels: ['
        for c,v in zip(self.classes_,self.get_output()):
            s+=c+' : '+str(v)+', '
        s + ' ]'
        return s

    def get_output(self):
        x = np.array(self.class_values)
        if self.output_type == 'probas':
            return list(x/np.sum(x))
        if self.output_type == 'logit':
            return [np.exp(x) / np.sum(np.exp(x))]

