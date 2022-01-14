"""
=================================
1. Load module
=================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef,auc, roc_curve,plot_roc_curve, plot_precision_recall_curve,classification_report, confusion_matrix,average_precision_score, precision_recall_curve
from pandas.core.frame import DataFrame
from numpy import mean
from collections import Counter
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras import activations
import keras
from keras.layers import Dense,Dropout, Flatten, Conv1D, MaxPooling1D,GlobalAveragePooling1D,Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import BatchNormalization
import tensorflow as tf
from keras import optimizers
## imbalance
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN


### F1 score, precision, recall and accuracy metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

"""
=====================================================
2. CIP procssing 
=====================================================
"""
gi_data = pd.read_csv("/cip_multi_data.csv",index_col=0) ### change
gi_pheno = pd.read_csv("/cip_ctx_ctz_gen_pheno.csv",index_col=0)    ### change
gi_data.shape,gi_pheno.shape
gi_data2 = gi_data.values
cip_pheno = gi_pheno[["CIP"]]
cip_pheno2 = cip_pheno.values
cip_pheno3 = cip_pheno2.reshape(809,)
X = gi_data2
y = cip_pheno3
X.shape, y.shape
#((809, 14972), (809,))

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
x_train.shape,y_train.shape,x_test.shape,y_test.shape

inputs = x_train.reshape(647, 14972,1)
inputs = inputs.astype('float32')
targets = to_categorical(y_train)

x_test = x_test.reshape(162, 14972,1)
x_test = x_test.astype('float32')
y_test2 = to_categorical(y_test)

############################# Step3: Model: basic model training (from previous my model)#####################

"""
=====================================================
2. CIP basic model training, save model weights 
=====================================================
"""


batch_size = 8
no_classes = 2
no_epochs = 50
verbosity = 1
num_folds = 5

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)
# K-fold Cross Validation model evaluation
fold_no = 1
model_history=[]
for train, test in kfold.split(inputs, targets):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu',input_shape=(14972,1)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=8, kernel_size=3, padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Conv1D(filters=16, kernel_size=3,  padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=3,  padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc',f1_m,precision_m, recall_m])
    # Generate a print
    print('--------------------------------')
    print(f'Training for fold {fold_no} ...')
    ## checkpoint for saving model
    filepath="CIP_basic_best_{epoch:03d}-{acc:03f}-{val_acc:03f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    # Fit data to model
    train_model = model.fit(inputs[train], targets[train],batch_size=batch_size,epochs=no_epochs,callbacks=[checkpoint],verbose=verbosity,validation_data=(inputs[test], targets[test]))
    model_history.append(train_model.history)
    # Increase fold number
    fold_no = fold_no + 1

## save model
model.save_weights('CIP_basic_cnn.h5', save_format='h5')
# save model history
model_out = DataFrame(model_history)
model_out.to_csv("CIP_basic_model_history_out.csv",index=False)


"""
=========================================================================
3. Plot the accuracy and loss of training and validation dataset 
=========================================================================
"""
## plot
plt.figure()
plt.title('Train Accuracy vs Val Accuracy')
plt.plot(model_history[0]['acc'], color='blue',label=r'T_Acc Fold 1: (best = %0.3f)' % (np.max(model_history[0]['acc'])))
plt.plot(model_history[0]['val_acc'], color='red',label=r'V_Acc Fold 1: (best = %0.3f)' % (np.max(model_history[0]['val_acc'])))
plt.plot(model_history[1]['acc'], color='blue',label=r'T_Acc Fold 2: (best = %0.3f)' % (np.max(model_history[1]['acc'])))
plt.plot(model_history[1]['val_acc'], color='red',label=r'V_Acc Fold 2: (best = %0.3f)' % (np.max(model_history[1]['val_acc'])))
plt.plot(model_history[2]['acc'], color='blue',label=r'T_Acc Fold 3: (best = %0.3f)' % (np.max(model_history[2]['acc'])))
plt.plot(model_history[2]['val_acc'], color='red',label=r'V_Acc Fold 3: (best = %0.3f)' % (np.max(model_history[2]['val_acc'])))
plt.plot(model_history[3]['acc'], color='blue',label=r'T_Acc Fold 4: (best = %0.3f)' % (np.max(model_history[3]['acc'])))
plt.plot(model_history[3]['val_acc'], color='red',label=r'V_Acc Fold 4: (best = %0.3f)' % (np.max(model_history[3]['val_acc'])))
plt.plot(model_history[4]['acc'], color='blue',label=r'T_Acc Fold 5: (best = %0.3f)' % (np.max(model_history[4]['acc'])))
plt.plot(model_history[4]['val_acc'], color='red',label=r'V_Acc Fold 5: (best = %0.3f)' % (np.max(model_history[4]['val_acc'])))
#plt.ylim((0.7, 1.03))
plt.legend(prop={'size':6}) 
plt.savefig('CIP_basic_acc_eva.pdf')
plt.show()

## plot
plt.figure()
plt.title('Train Loss vs Val Loss')
plt.plot(model_history[0]['loss'], color='blue',label=r'T_Loss Fold 1: (best = %0.3f)' % (np.max(model_history[0]['loss'])))
plt.plot(model_history[0]['val_loss'], color='red',label=r'V_Loss Fold 1: (best = %0.3f)' % (np.max(model_history[0]['val_loss'])))
plt.plot(model_history[1]['loss'], color='blue',label=r'T_Loss Fold 2: (best = %0.3f)' % (np.max(model_history[1]['loss'])))
plt.plot(model_history[1]['val_loss'], color='red',label=r'V_Loss Fold 2: (best = %0.3f)' % (np.max(model_history[1]['val_loss'])))
plt.plot(model_history[2]['loss'], color='blue',label=r'T_Loss Fold 3: (best = %0.3f)' % (np.max(model_history[2]['loss'])))
plt.plot(model_history[2]['val_loss'], color='red',label=r'V_Loss Fold 3: (best = %0.3f)' % (np.max(model_history[2]['val_loss'])))
plt.plot(model_history[3]['loss'], color='blue',label=r'T_Loss Fold 4: (best = %0.3f)' % (np.max(model_history[3]['loss'])))
plt.plot(model_history[3]['val_loss'], color='red',label=r'V_Loss Fold 4: (best = %0.3f)' % (np.max(model_history[3]['val_loss'])))
plt.plot(model_history[4]['loss'], color='blue',label=r'T_Loss Fold 5: (best = %0.3f)' % (np.max(model_history[4]['loss'])))
plt.plot(model_history[4]['val_loss'], color='red',label=r'V_Loss Fold 5: (best = %0.3f)' % (np.max(model_history[4]['val_loss'])))
#plt.ylim((0.7, 1.03))
plt.legend(prop={'size':6}) 
plt.savefig('CIP_basic_loss_eva.pdf')
plt.show()


"""
=========================================================================
4. Evaluation on testing set 
=========================================================================
"""

y_pred_proba = model.predict(x_test)
y_pred_classes = y_pred_proba.argmax(axis=-1)
f1_matrix = confusion_matrix(y_test,y_pred_classes)
f1_report = classification_report(y_test,y_pred_classes)
MCC = matthews_corrcoef(y_test,y_pred_classes)


np.save("CIP_basic_cnn_prob_out.npy",y_pred_proba)
np.save("CIP_basic_cnn_cls_out.npy",y_pred_classes)   
np.save("CIP_basic_cnn_real_out.npy",y_test)
np.save("CIP_basic_cnn_x_test.npy",x_test)
file = open("/CIP_basic_cnn_confusion_matrix.csv","w")
file.write(str(MCC))
file.write(str(f1_report))
file.write(str(f1_matrix))
file.close() 

### ROC
fpr_keras, tpr_keras,thresholds_keras = roc_curve(y_test,y_pred_proba[:,1])
auc_keras = auc(fpr_keras,tpr_keras)
plt.figure(figsize=(18 , 13))
plt.plot([0,1],[0,1],linestyle="--",lw=3,color='k',alpha=.8)
plt.plot(fpr_keras,tpr_keras,label= 'ROC (AUC = {:.3f} )'.format(auc_keras))
plt.xlabel('False Postive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.legend(loc='best')
plt.savefig("CIP_basic_CNN_ROC_curve.pdf",bbox_inches='tight')
np.savetxt("CIP_basic_CNN_fpr.csv",fpr_keras,delimiter=",")
np.savetxt("CIP_basic_CNN_tpr.csv",tpr_keras,delimiter=",")


###### PR curve
precision,recall,thresholds = precision_recall_curve(y_test,y_pred_proba[:,1])
pr_auc = auc(recall,precision)
plt.figure(figsize=(18 , 13))
plt.plot([0,1],[1,0],linestyle="--",lw=3,color='k',alpha=.8)
plt.plot(recall,precision,label= '(AUCPR = {:.3f})'.format(pr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.savefig("CIP_basic_CNN_P-R_curve.pdf")
np.savetxt("CIP_basic_CNN_precision.csv",precision,delimiter=",")
np.savetxt("CIP_basic_CNN_recall.csv",recall,delimiter=",")
