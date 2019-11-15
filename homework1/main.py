import libs
import sklearn as sk
from sklearn.datasets import load_wine
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, y_train, X_validate, y_validate, X_test, y_test = libs.prepare_data()

#--------------------------------------<K-NEIGHBORS>---------------------------------------#

# K-neighbors parameters
kn_values = [1,3,5,7]

print("\n### KNN ###\n------------------------------------------")
best_knn, scores = libs.do_knn(kn_values, X_train, y_train, X_validate, y_validate)
test_accuray = libs.get_score(best_knn, X_test, y_test)
print("------------------------------------------")
print("Accuracy on test: % 2f, k: % 2d" %(test_accuray, best_knn.n_neighbors) )

#-------------------------------------------<SVM>-------------------------------------------#

C=[0.001, 0.01, 0.1, 1, 10, 100,1000]

print("\n### SVM LINEAR ANALYSIS ###\n------------------------------------------")
best_lin_svm, scores = libs.do_svm("linear", C, X_train, y_train, X_validate, y_validate)
test_accuray = libs.get_score(best_lin_svm, X_test, y_test)
print("------------------------------------------")
print("Accuracy on test: % 2f, C: % 2f" %(test_accuray, best_lin_svm.C) )

#-----------------------------------------<SVM RBF>-----------------------------------------#

print("\n### SVM RBF ANALYSIS ###\n------------------------------------------")
best_rbf_svm, scores = libs.do_svm("rbf", C, X_train, y_train, X_validate, y_validate)
test_accuray = libs.get_score(best_rbf_svm, X_test, y_test)
print("------------------------------------------")
print("Accuracy on test: % 2f, C: % 2f" %(test_accuray, best_rbf_svm.C) )

#-------------------------------------- <MANUAL-TUNING>-------------------------------------#

print("\n### RBF ANALYSIS POINT 15 ###\n------------------------------------------")

gamma=[1e-3, 1e-4]
best_rbf_svm= libs.do_svm_grid("rbf", C, X_train, y_train, X_validate, y_validate, gamma)
test_accuray = libs.get_score(best_rbf_svm, X_test, y_test)
print("------------------------------------------")
print("Accuracy on test: % 2f, C: % 2f, g: %2f" %(test_accuray, best_rbf_svm.C, best_rbf_svm.gamma ) )

#-------------------------------------- <K-FOLD>-------------------------------------#

print("\n### SVM ANALYSIS K-FOLD ###\n------------------------------------------")
best_fold_svm = libs.do_svm_fold(X_train, y_train)


