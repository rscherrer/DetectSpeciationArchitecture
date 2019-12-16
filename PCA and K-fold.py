import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as p
from matplotlib import cm
from sklearn.manifold import TSNE
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from Plots import *
from ensemble import make_prediction
#import graphviz 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# data
data = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=0)[:179]
test_data = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=0)[180:]
# read labels
labels = np.loadtxt(open("labels.csv", "rb"), delimiter=",", skiprows=0)

labeled_data = np.append(data, labels[:,None], axis=1)
color_array = []
for i in range(0,len(labels)):
    if labels[i] == 1:
        color_array.append([1,0,0])
    else: #sick
        color_array.append([0,1,0])


def pca_reduced_data(n):
    pca = PCA(n_components=n)
    # subtract mean to center data around origin
    centered_data = data - data.mean(0)
    return pca.fit_transform(centered_data)
   
def get_anova_best_features(n_features):
    feature_scores = f_classif(data, labels)[0]
    X_indices = np.arange(data.shape[-1])
    best_features = sorted(zip(feature_scores, X_indices), reverse=True)[:n_features]
    return list(map(lambda bf: bf[1], best_features))

# Applies nfold cross validation on knn with a given k and fold count.
# Returns: average accuracy
def nfold_cross_knn(data,labels,fold_count,k):
    kf = KFold(n_splits=fold_count)
    accuracy_sum = 0
    mat = [[0, 0],[ 0, 0]]
    for train_indices, test_indices in kf.split(data):
        # train the model
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(data[train_indices,:], labels[train_indices])
        # test the model
        score = neigh.score(data[test_indices,:], labels[test_indices])
        labels_pred = neigh.predict(data[test_indices,:])
        conf_mat = confusion_matrix(labels[test_indices], labels_pred)
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        mat += conf_mat
        accuracy_sum = accuracy_sum + score

    return (accuracy_sum/fold_count, mat/fold_count)

def evaluate_knn():
    #1 base data, not preprocessed
    scores = []
    ks = []
    cm0 = []
    cm1 = []

    for k in [k for k in range(3,20) if k %2]:
        ks.append(k)
        (score, cm) = nfold_cross_knn(data,labels,10,k)
        scores.append(score)
        cm0.append(cm[0][0])
        cm1.append(cm[1][1])

    p.figure(1)
    p.plot(ks,scores, label="unfiltered data")

    p.figure(2)
    p.plot(ks, cm0, label="unfiltered data")

    p.figure(3)
    p.plot(ks, cm1, label="unfiltered data")
    
    #2 use only top anova features
    scores = []
    ks = []
    cm0 = []
    cm1 = []

    for k in [k for k in range(3,20) if k %2]:
        ks.append(k)
        (score, cm) = nfold_cross_knn(data[:,get_anova_best_features(10)],labels,10,k)
        scores.append(score)
        cm0.append(cm[0][0])
        cm1.append(cm[1][1])

    p.figure(1)
    p.plot(ks,scores,label="ANOVA top 10 features")

    p.figure(2)
    p.plot(ks, cm0, label="ANOVA top 10 features")

    p.figure(3)
    p.plot(ks, cm1, label="ANOVA top 10 features")
    
    #3 use 5 pca features
    scores = []
    ks = []
    cm0 = []
    cm1 = []

    for k in [k for k in range(3,20) if k %2]:
        ks.append(k)
        (score, cm) = nfold_cross_knn(pca_reduced_data(5),labels,10,k)
        scores.append(score)
        cm0.append(cm[0][0])
        cm1.append(cm[1][1])

    p.figure(1)
    p.plot(ks,scores,label="PCA 5 PC")

    p.figure(2)
    p.plot(ks, cm0, label="PCA 5 PC")

    p.figure(3)
    p.plot(ks, cm1, label="PCA 5 PC")
    
    #4 use 6 pca features
    scores = []
    ks = []
    cm0 = []
    cm1 = []

    for k in [k for k in range(3,20) if k %2]:
        ks.append(k)
        (score, cm) = nfold_cross_knn(pca_reduced_data(6),labels,10,k)
        scores.append(score)
        cm0.append(cm[0][0])
        cm1.append(cm[1][1])
    
    p.figure(1)
    p.plot(ks,scores,label="PCA 6 PC")

    p.figure(2)
    p.plot(ks, cm0, label="PCA 6 PC")

    p.figure(3)
    p.plot(ks, cm1, label="PCA 6 PC")
    
    #5 use z transformations
    scores = []
    ks = []
    cm0 = []
    cm1 = []

    for k in [k for k in range(3,20) if k %2]:
        ks.append(k)
        (score, cm) = nfold_cross_knn(stats.zscore(data, axis=1, ddof=1),labels,10,k)
        scores.append(score)
        cm0.append(cm[0][0])
        cm1.append(cm[1][1])
    
    p.figure(1)
    p.plot(ks,scores,label="z transformed data")

    p.figure(2)
    p.plot(ks, cm0, label="z transformed data")

    p.figure(3)
    p.plot(ks, cm1, label="z transformed data")

    #6 z transformed + top anova features
    scores = []
    ks = []
    cm0 = []
    cm1 = []

    for k in [k for k in range(3,20) if k %2]:
        ks.append(k)
        (score, cm) = nfold_cross_knn(stats.zscore(data, axis=1, ddof=1)[:,get_anova_best_features(10)],labels,10,k)
        scores.append(score)
        cm0.append(cm[0][0])
        cm1.append(cm[1][1])
    
    p.figure(1)
    p.plot(ks,scores,label="z transformed ANOVA top 10 features")

    p.figure(2)
    p.plot(ks, cm0, label="z transformed ANOVA top 10 features")

    p.figure(3)
    p.plot(ks, cm1, label="z transformed ANOVA top 10 features")

    #7 top 50 anova features
    scores = []
    ks = []
    cm0 = []
    cm1 = []

    for k in [k for k in range(3,20) if k %2]:
        ks.append(k)
        (score, cm) = nfold_cross_knn(stats.zscore(data, axis=1, ddof=1)[:,get_anova_best_features(50)],labels,10,k)
        scores.append(score)
        cm0.append(cm[0][0])
        cm1.append(cm[1][1])

    p.figure(1)
    p.plot(ks,scores,label="ANOVA top 50 features")

    p.figure(2)
    p.plot(ks, cm0, label="ANOVA top 50 features")

    p.figure(3)
    p.plot(ks, cm1, label="ANOVA top 50 features")

    # show combined plot
    p.figure(1)
    p.xlabel("k amount of neighbours")
    p.ylabel("accuracy (%)")
    p.legend()
    
    p.figure(2)
    p.xlabel("k neighbours")
    p.ylabel("classwise accuracy class 0 (%)")
    p.legend()

    p.figure(3)
    p.xlabel("k neighbours")
    p.ylabel("classwise accuracy class 1 (%)")
    p.legend()
    p.show()

def evaluate_tree_splits():
    n = 10 # enforce 10-fold cross validation
    bar_scores = []
    bar_labels = []
    #determine random seed for classifier and also max depth of the tree
    clf = tree.DecisionTreeClassifier(max_depth=50,random_state=1234)
    clf_entropy = tree.DecisionTreeClassifier(criterion='entropy',max_depth=50,random_state=1234)
    # number for writing to file

    #1 unprocessed data gini
    bar_scores.append(np.mean(cross_val_score(clf, data, labels, cv=n)))
    bar_labels.append("all, gini")

    #2 top 50 anova features gini
    bar_scores.append(np.mean(cross_val_score(clf, data[:,get_anova_best_features(50)],labels, cv=n)))
    bar_labels.append("top 50 ANOVA, gini")

    #3 unprocessed data entropy
    bar_scores.append(np.mean(cross_val_score(clf_entropy, data, labels, cv=n)))
    bar_labels.append("all, entropy")

    #4 top 50 anova features entropy
    bar_scores.append(np.mean(cross_val_score(clf_entropy, data[:,get_anova_best_features(50)],labels, cv=n)))
    bar_labels.append("top 50 ANOVA, entropy")

    #5 unprocessed data entropy
    bar_scores.append(np.mean(cross_val_score(clf, pca_reduced_data(10), labels, cv=n)))
    bar_labels.append("top 10 PCA, gini")

    #6 pca to 10 entropy
    bar_scores.append(np.mean(cross_val_score(clf_entropy, pca_reduced_data(10),labels, cv=n)))
    bar_labels.append("top 10 PCA, entropy")

    # show combined plot
    p.bar(bar_labels,bar_scores)
    p.xlabel("Features, split type")
    p.ylabel("accuracy (%)")
    p.ylim(0.85,1.0)
    p.legend()
    p.show()

# do multiple n-fold cross validations for decision trees using different preprocessing
# then fit a final version on all data and plot decision tree graph
def evaluate_tree():
    #determine random seed for classifier and also max depth of the tree
    clf = tree.DecisionTreeClassifier(max_depth=3,random_state=1234)

    # number for writing to file
    number = 1
    #1 unprocessed data
    scores = []
    ns = []
    for n in range(2,11):
        ns.append(n)
        scores.append(np.mean(cross_val_score(clf, data, labels, cv=n)))

    p.plot(ns,scores,label="unprocessed")

    # now fit a final version on all data and generate a graph and write to file (Requires graphviz)
    #fit = clf.fit(data, labels)
    #dot_data = tree.export_graphviz(fit, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("leukemia" + str(number)) 
    #number += 1
    
    #2 use only top anova features
    scores = []
    ns = []
    for n in range(2,11):
        ns.append(n)
        scores.append(np.mean(cross_val_score(clf, data[:,get_anova_best_features(10)],labels, cv=n)))
    
    p.plot(ns,scores,label="ANOVA top 10 features")

    #(Requires graphviz)
    #fit = clf.fit(data[:,get_anova_best_features(10)], labels)
    #dot_data = tree.export_graphviz(fit, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("leukemia" + str(number)) 
    #number += 1

    #3 use 5 pca features
    scores = []
    ns = []
    for n in range(2,11):
        ns.append(n)
        scores.append(np.mean(cross_val_score(clf, pca_reduced_data(5),labels, cv=n)))
    
    p.plot(ns,scores,label="PCA top 5")

    #(Requires graphviz)
    #fit = clf.fit(pca_reduced_data(5), labels)
    #dot_data = tree.export_graphviz(fit, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("leukemia" + str(number)) 
    #number += 1

    #4 use 10 pca features
    scores = []
    ns = []
    for n in range(2,11):
        ns.append(n)
        scores.append(np.mean(cross_val_score(clf, pca_reduced_data(10),labels, cv=n)))
    
    p.plot(ns,scores,label="PCA top 10")
    #(Requires graphviz)
    #fit = clf.fit(pca_reduced_data(10), labels)
    #dot_data = tree.export_graphviz(fit, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("leukemia" + str(number)) 
    #number += 1

    #5 use z transformations
    scores = []
    ns = []
    for n in range(2,11):
        ns.append(n)
        scores.append(np.mean(cross_val_score(clf, stats.zscore(data, axis=1, ddof=1),labels, cv=n)))
    
    p.plot(ns,scores,label="Z-transform")
    #(Requires graphviz)
    #fit = clf.fit(stats.zscore(data, axis=1, ddof=1), labels)
    #dot_data = tree.export_graphviz(fit, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("leukemia" + str(number)) 
    #number += 1

    #6 z transformed + top anova features
    scores = []
    ns = []
    for n in range(2,11):
        ns.append(n)
        scores.append(np.mean(cross_val_score(clf, stats.zscore(data, axis=1, ddof=1)[:,get_anova_best_features(10)],labels, cv=n)))
    
    p.plot(ns,scores,label="Z-transform and top 10 ANOVA")
    #(Requires graphviz)
    #fit = clf.fit(stats.zscore(data, axis=1, ddof=1)[:,get_anova_best_features(10)], labels)
    #dot_data = tree.export_graphviz(fit, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("leukemia" + str(number)) 
    #number += 1

    #7 top 50 anova features
    scores = []
    ns = []
    for n in range(2,11):
        ns.append(n)
        scores.append(np.mean(cross_val_score(clf, stats.zscore(data, axis=1, ddof=1)[:,get_anova_best_features(50)],labels, cv=n)))

    p.plot(ns,scores,label="Z-transform and top 50 ANOVA")

    #8 top 50 anova features
    scores = []
    ns = []
    for n in range(2,11):
        ns.append(n)
        scores.append(np.mean(cross_val_score(clf, data[:,get_anova_best_features(50)],labels, cv=n)))

    p.plot(ns,scores,label="top 50 ANOVA")
    #(Requires graphviz)   
    #fit = clf.fit(stats.zscore(data, axis=1, ddof=1)[:,get_anova_best_features(50)], labels)
    #dot_data = tree.export_graphviz(fit, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("leukemia" + str(number)) 
    #number += 1
    
    

    # show combined plot
    p.xlabel("n folds")
    p.ylabel("accuracy (%)")
    p.legend()
    p.show()

#holisticAndExploratory()
#evaluate_knn()
#evaluate_tree()
#evaluate_tree_splits()
#team1_ensemble()
make_prediction()