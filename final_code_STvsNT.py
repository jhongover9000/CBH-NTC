
# import numpy as np # linear algebra
# import keras
# import mat73
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add
# from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import SpatialDropout2D, Reshape
# from tensorflow.keras.regularizers import l1_l2
# from tensorflow.keras.layers import Input, Flatten
# from tensorflow.keras.constraints import max_norm
# from tensorflow.keras import backend as K
# import tensorflow as tf
# from datetime import datetime

# from sklearn.utils import class_weight
# from sklearn.preprocessing import StandardScaler, scale
# from tensorflow.keras.layers import Dropout, MultiHeadAttention, LayerNormalization, Reshape
# from keras.layers import Dropout, Add, Lambda, DepthwiseConv2D, Input, Permute, Concatenate, Flatten, Reshape
# from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
# import shap
# import tensorflow as tf
# import gc
# import keras
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as plt
# import shap
# import tensorflow as tf
# import tensorflow.keras.backend as K

import numpy as np
import mat73
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D, 
    AveragePooling2D, Add, SeparableConv2D, DepthwiseConv2D, 
    BatchNormalization, SpatialDropout2D, Reshape, Input, 
    Flatten, MultiHeadAttention, LayerNormalization, Conv1D, 
    Concatenate, Lambda, GlobalAveragePooling2D, multiply
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import shap
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import keras
def proposed(n_timesteps, n_features, n_outputs):
    
    input_1 = Input(shape=(1, n_features, n_timesteps))  # TensorShape([None, 1, 22, 1125])

    block0       = Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', data_format="channels_first")(input_1)
    block0       = LayerNormalization()(block0)
    block0       = Activation(activation='elu')(block0)

    block1       = Conv2D(filters=8, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first")(input_1)
    block1       = LayerNormalization()(block1)
    block1       = Activation(activation='elu')(block1)
    
  
    block2       = Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first")(input_1)
    block2       = LayerNormalization()(block2)
    block2       = Activation(activation='elu')(block2)


    block2 = Concatenate(axis=1)([block0, block1, block2])
    block2       = se_block(block2, 8)
    
    
    
    block3       = DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first")(block2)
    block3       = LayerNormalization()(block3)
    block3       = Activation(activation='elu')(block3)

    block3       = AveragePooling2D(pool_size=(1, 64), padding='same', data_format="channels_first")(block3)

    block5  = Flatten() (block3)
    block5       = Dense(n_outputs, kernel_constraint=max_norm(0.25))(block5)
    block5       = Activation(activation='softmax')(block5)

    return Model(inputs=input_1, outputs=block5)


def se_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 #if K.image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D(data_format="channels_first")(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    #if K.image_data_format() == 'channels_first':
    se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def plot_history(train_acc,val_acc):
    plt.plot(train_acc, color ='blue', label='train')
    plt.plot(val_acc, color ='red', label = 'test')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc="upper left")
    plt.show()
    fig_file = f"curves/AccuracyCurve_{timestamp}.pdf"
    plt.savefig(fig_file)

# Creating Arrays of Data
nSub = 20
img_dict = mat73.loadmat('./dataset/SF_Img_MLdata.mat')
img_list = img_dict['data']

import numpy as np

img_tr_con = []
y_img = []



for i in range(len(img_list)):
    img_arr = img_list[i]

    img_temp = np.concatenate((np.array(img_arr[0]), np.array(img_arr[1]), np.array(img_arr[2]), np.array(img_arr[3])),
                              axis=2)

    img_tr_con.append(img_temp)

    # extract number of trials for each condition for subject i
    NF_fx = np.shape(img_arr[0])
    NF_ex = np.shape(img_arr[1])
    SF_fx = np.shape(img_arr[2])
    SF_ex = np.shape(img_arr[3])

    y_temp = np.concatenate(
        (np.zeros((1, NF_fx[2])), np.zeros((1, NF_ex[2])), np.ones((1, SF_fx[2])), np.ones((1, SF_ex[2]))), axis=1)
    y_temp = np.squeeze(y_temp)  # turns array into 1D
    y_img.append(y_temp)


#Removing bad subjects
indexes = [0,5,11,19]
# del_bad =  [12,3,14,13,6,18,17,10]
# del_good = [16,15,7,8,9,2,4,1]

# indexes = indexes + del_bad
for index in sorted(indexes, reverse=True):
    del y_img[index]
    del img_tr_con[index]



# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()

nSub = 20 - len(indexes)  # number of subjects
bs_t = 16  # batch size
epochs = 30
lr = 0.00005
scores_atc = []
scores_dcn = []
scores_soft = []
nb_classes = 2
chans = 56
samples = 400
w_decay = 0.01
confx = np.zeros((nSub, nb_classes, nb_classes))

shap_values_all = []
y_test_all = []
y_pred_all = []


print(np.shape(img_tr_con[0]))


# channels_to_focus = [33,30,26,4,3]#[54, 50, 47, 29, 28, 26, 21, 19, 1]
# channels_remained = [i for i in range(56) if i not in channels_to_focus]
# chans = len(channels_to_focus)

# for i in range(len(img_tr_con)):
#     img_tr_con[i] = np.delete(img_tr_con[i],channels_remained,axis=0)
history_list=[]
print('Comment: Test')

model = proposed(samples, chans, nb_classes)
print(model.summary())

# model = proposed(samples, chans, nb_classes)
# print(model.summary())



for sub in range(nSub):
    print(f'Subject: #{sub}')
    # split train and test - time
    X_test_t = img_tr_con[sub]
    x_train_t = img_tr_con[:sub] + img_tr_con[sub + 1:]  # all other subjects as train data
    X_train_t = np.dstack(x_train_t)

    Y_test_t = np.squeeze(y_img[sub])
    y_train_t = y_img[:sub] + y_img[sub + 1:]
    Y_train_t = np.concatenate(y_train_t)

    X_test_t = np.expand_dims(X_test_t, axis=3)
    X_train_t = np.expand_dims(X_train_t, axis=3)

    # add kernels dimension to X_train_t & X_test_t => np.newaxis / np.expand_dims
    X_train_t = np.swapaxes(X_train_t, 0, 2)
    X_train_t = np.swapaxes(X_train_t, 1, 3)

    X_test_t = np.swapaxes(X_test_t, 0, 2)
    X_test_t = np.swapaxes(X_test_t, 1, 3)

    Y_train_t = to_categorical(Y_train_t)
    Y_test_t = to_categorical(Y_test_t)

    # Downsampling through skipping
    X_train_t = X_train_t[:, :, :, 4::5]
    X_test_t = X_test_t[:, :, :, 4::5]

    print('X Test: ', np.shape(X_test_t))
    print('X Train: ', np.shape(X_train_t))

    print('Y Test: ', np.shape(Y_test_t), '\n')
    print('Y Train: ', np.shape(Y_train_t), '\n')


    model_atc = proposed(samples, chans, nb_classes)

    opt_atc = keras.optimizers.Adam(learning_rate=lr ,weight_decay=w_decay)

    model_atc.compile(loss='categorical_crossentropy', optimizer=opt_atc,
                          metrics=['accuracy'])



    history_atc = model_atc.fit(X_train_t, Y_train_t, validation_data = (X_test_t,Y_test_t), batch_size=bs_t, epochs=epochs, verbose=0)
    probs_atc = model_atc.predict(X_test_t)
    preds_atc = probs_atc.argmax(axis=-1)
    acc_atc = np.mean(preds_atc == Y_test_t.argmax(axis=-1))
    print(f'ATC:{acc_atc} %')
    history_list.append(history_atc)


    # confx[sub,:,:] = confusion_matrix(Y_test_t.argmax(axis = -1), preds_atc)

    scores_atc.append(acc_atc)


    ##SHAP STUFF
    # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough  # this solves the "shap_ADDV2" problem but another one will appear
    # shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.


    # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough  
    # shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
    # shap.explainers._deep.deep_tf.op_handlers["DepthwiseConv2dNative"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  

    # shap.explainers._deep.deep_tf.op_handlers["BatchToSpaceND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
    # shap.explainers._deep.deep_tf.op_handlers["SpaceToBatchND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
    # shap.explainers._deep.deep_tf.op_handlers["Einsum"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  

    
    
    # background = X_train_t[np.random.choice(X_train_t.shape[0], 300, replace=False)]
    # e = shap.DeepExplainer(model_atc, background)

    # print(e)
    # print(np.shape(X_test_t))
    # shap_values = e.shap_values(X_test_t,check_additivity= False)
    # shap_values_all.append(shap_values)
    # y_test_all.append(Y_test_t.argmax(axis=-1))
    # y_pred_all.append(preds_atc)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

#plot accuracy history
train_acc=np.zeros((epochs,1))
val_acc=np.zeros((epochs,1))
for sub in range(nSub-4):
    train_acc=train_acc.flatten() + np.array(history_list[sub].history['accuracy']).flatten()
    val_acc=val_acc.flatten()+ np.array(history_list[sub].history['val_accuracy']).flatten()
train_acc=train_acc/(nSub-4)
val_acc= val_acc/(nSub-4)
plot_history(train_acc,val_acc)

## SAVING SHAP STUFF
# import pickle

# with open("shaps_values_all_haneen_full", "wb") as fp:  # Pickling
#     pickle.dump(shap_values_all, fp)

# with open("y_test_all_haneen_full", "wb") as fp:  # Pickling
#     pickle.dump(y_test_all, fp)

# with open("y_pred_all_haneen_full", "wb") as fp:  # Pickling
#     pickle.dump(y_pred_all, fp)


print(f'Avg Accuracy ATC:{np.mean(scores_atc)} %')
print(f'All Accuracy ATC:{scores_atc} ')

