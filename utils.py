from conjunction import *
import numpy as np
EPSILON=0.00001
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
import xgboost
import re
feature_regex = re.compile('\D+(?P<node_index>\d+):\[(?P<feature>[^<]+)<(?P<value>[^\]]+)\D+(?P<left>\d+)\D+(?P<right>\d+)\D+(?P<missing>\d+)')
leaf_regex = re.compile('\D+(?P<node_index>\d+)[^\=]+=(?P<prediction>.+)')


PROBABILITY_OUTPUT_TREES = [RandomForestClassifier]
LOGIT_OUTPUT_TREES = [xgboost.sklearn.XGBClassifier,GradientBoostingClassifier]


def get_conjunction_from_sklearn_leaf(tree_, leaf_index, features, classes, tree_index, output_type,class_index=None):
    if output_type=='probas':
        class_values = tree_.value[leaf_index][0]
        class_values = list(class_values/class_values.sum())
    elif output_type=='logit':
        class_values = [0 if i!=class_index else tree_.value[leaf_index][0][0] for i in range(len(classes))]
    new_branch=Conjunction(features,classes,leaf_index=[str(tree_index)+'_'+str(leaf_index)],class_values=class_values,output_type=output_type)#initialize branch
    node_id = leaf_index
    while node_id: #iterate over all nodes in branch
        ancesor_index = np.where(tree_.children_right == node_id)[0] #assuming left is the default for efficiency purposes
        bound='lower'
        if len(ancesor_index) == 0:
            bound = 'upper'
            ancesor_index = np.where(tree_.children_left == node_id)[0]
        new_branch.addCondition(tree_.feature[ancesor_index[0]], tree_.threshold[ancesor_index[0]]-EPSILON, bound)
        node_id=ancesor_index[0]
    return new_branch


def parse_xgboost_dump(model):
    """
    Extract decision trees from XGBoost.

    :param model: XGBoost model
    :param feature_dict: {feature_name: feature_index}
    :return: trees: List of trees where trees represented as lists of dictionaries. Each dictionary represents a node within the corresponding tree
    """
    trees= []
    for tree_string in model._Booster.get_dump():
        nodes = [feature_regex.search('t' + node).groupdict() if '[' in node else leaf_regex.search('t' +node).groupdict() for node in tree_string.split('\n')[:-1]]
        trees.append(nodes)
    return trees

def extract_class_values(tree, leaf_index, label_names, class_index):
    """
    This function takes a leaf index and convert the class logit into a probability

    :param tree: dictionary that represents a decision tree
    :param leaf_index: leaf index - integer
    :param label_names: list of strings - labels
    :param class_index: index of the addressed class
    :return: class probabilitis
    """
    pred = float(tree[leaf_index]['prediction'])
    if len(label_names)>2:
        return np.array([pred if i == class_index else 0 for i in range(len(label_names))])
    else:
        return [0,pred]


def extract_conjunction_from_xgboost_base_tree(tree, tree_index, leaf_index, feature_dict, label_names, class_index, output_type):
    """
    Covert the leaves of a tree into a set of conjunctions

    :param tree: list of dictionaries where each dictionary represents a node within a tree
    :param leaf_index: index of the currently processed node
    :param feature_dict: {feature name: feature index} - for converting xgboost feature names to conjunction feature indices
    :param label_names: possible class values
    :param class_index: currently addressed class - since each model is basically a binary classification of tree of a single class it's impoertant to know the relevant class
    :return: A set of conjunctions
    """
    if 'prediction' in tree[leaf_index]:
        class_values = extract_class_values(tree, leaf_index, label_names, class_index)
        return [Conjunction(list(feature_dict.keys()),label_names,
                            leaf_index=[str(tree_index)+'_'+str(leaf_index)],class_values=class_values,output_type=output_type)]
    l_conjunctions = extract_conjunction_from_xgboost_base_tree(tree, tree_index, int(tree[leaf_index]['left']), feature_dict, label_names, class_index, output_type)
    r_conjunctions = extract_conjunction_from_xgboost_base_tree(tree, tree_index, int(tree[leaf_index]['right']), feature_dict, label_names, class_index, output_type)
    for c in l_conjunctions:
        c.addCondition(feature_dict[tree[leaf_index]['feature']],float(tree[leaf_index]['value']),'upper')
    for c in r_conjunctions:
        c.addCondition(feature_dict[tree[leaf_index]['feature']],float(tree[leaf_index]['value']),'lower')
    return l_conjunctions + r_conjunctions

def extract_trees_from_xgb(model, unique_labels, features):
    """
    This function takes XGBoost model and returns a list of trees where each tree is represented as a list of conjunctions.
    Each of the tree conjunctions stands for a single decision path

    :param model: XGBoost model
    :param unique_labels: label names
    :param features: feature names
    :return: a list of conjunctions
    """
    trees = parse_xgboost_dump(model)
    num_of_labels = len(unique_labels)
    feature_dict = {v:k for k,v in enumerate(features)}
    conjunction_sets = {}
    for i,t in enumerate(trees): #i stands for the corresponding class index
        indexed_tree = {int(v['node_index']): v for v in t}
        conjunction_sets[i] = extract_conjunction_from_xgboost_base_tree(indexed_tree, i, 0, feature_dict, unique_labels, i % num_of_labels, output_type='logit')
    return list(conjunction_sets.values())

def extract_trees_from_sklearn_probas_ensemble(model, classes, feature_cols):
    conjunction_sets = []
    for tree_index, base_model in enumerate(model.estimators_):
        tree_ = base_model.tree_
        leaf_indexes = [i for i in range(tree_.node_count) if
                        tree_.children_left[i] == -1 and tree_.children_right[i] == -1]
        conjunction_sets.append([get_conjunction_from_sklearn_leaf(tree_, i, feature_cols, classes, tree_index, output_type='probas') for i in leaf_indexes])
    return conjunction_sets

def extract_trees_from_sklearn_logit_ensemble(model, classes, feature_cols):
    conjunction_sets = []
    for tree_index, base_model in enumerate(model.estimators_):
        for class_index,base_tree in enumerate(base_model):
            tree_ = base_tree.tree_
            leaf_indexes = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1 and tree_.children_right[i] == -1]
            conjunction_sets.append([get_conjunction_from_sklearn_leaf(tree_, i, feature_cols, classes, tree_index, output_type='logit',class_index=class_index) for i in leaf_indexes])
    return conjunction_sets

