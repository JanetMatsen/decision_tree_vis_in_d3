import json
import sklearn 

# from http://planspace.org/20151129-see_sklearn_trees_with_d3/

def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node


def save_tree_as_dict(clf, feature_names, label_names, save_path, node_index=0):
    """
    Wrapper for rules, that includes saving to a path and 
    more descriptive argument names
    """
    tree_dict = rules(clf, feature_names, label_names, node_index)
    # save the dict as a text file

    with open(save_path, 'w') as f:
        f.write(json.dumps(tree_dict))


