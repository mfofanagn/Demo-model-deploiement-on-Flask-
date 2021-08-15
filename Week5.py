import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,\
    confusion_matrix,plot_confusion_matrix,classification_report, precision_recall_curve, roc_curve

from sklearn.metrics import plot_roc_curve
import pickle

""""
The breast cancer dataset is a classic and binary classification dataset.

Data Set Information:

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 
They describe characteristics of the cell nuclei present in the image.

Attribute Information:

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)

Our aim is to make a binary classification (Diagnosis (1 = malignant, 0 = benign)) 
based on aforementionned information

In order to reduce complexity of the study, (demo purpose only) we decided to retain only features: 
mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness

"""

class breast_cancer:

 #def __init__(self):

 def predict(self, mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness):

        data = load_breast_cancer(as_frame=False)
        X = pd.DataFrame(data.data, columns=data.feature_names)
        XX = X.iloc[:,0:6]
        yy = data.target
        y = yy.reshape(-1, 1)
        print(y.shape)
        print(XX.info())
        print(XX.head())

        X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.33, random_state=42)

        print(" X_train shape is {0}".format(X_train.shape))
        print(" y_train shape is {0}".format(y_train.shape))

        mod = LogisticRegression(max_iter=10000)
        mod.fit(X_train, np.ravel(y_train))

        entry2 = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness]])
        entry3 = entry2.reshape(1, -1)

        pred = mod.predict(entry3)

        print(" pred shape is {0} and value is {1}".format(pred.shape, pred[0]))

        """""
        print(" accuracy_score is {}", accuracy_score(y_test, pred))
        print(" precision_score is {}", precision_score(y_test, pred))
        print(" recall_score is {}", recall_score(y_test, pred))
        print(" f1_score is {}", f1_score(y_test, pred))
        print(" roc_auc_score is {}", roc_auc_score(y_test, pred))
        #print(confusion_matrix(y_test, pred))
        #print(classification_report(y_test, pred))

        roc_auc = roc_auc_score(y_test, pred)
        fpr, tpr, thresholds = roc_curve(y_test, pred)

        #precision_recall_curve(y_test, pred)
        #plot_confusion_matrix(mod, X_test, y_test)
        #plt.show()
        
        """

        # print(y.shape)
        # Plot of a ROC curve for a specific class
        #ax = plt.gca()
        #rfc_disp = plot_roc_curve(mod, X_test, y_test, ax=ax, alpha=0.8)
        #rfc_disp.plot(ax=ax, alpha=0.8)

        return (pred[0])

# Object serialization

o_breast_cancer = breast_cancer()
with open("breast_cancer.pkl", "wb") as f:
    pickle.dump(o_breast_cancer, f)

#model = pickle.load(open("breast_cancer.pkl","rb"))

#model.predict()



