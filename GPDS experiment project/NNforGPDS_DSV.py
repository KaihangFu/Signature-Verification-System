import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


#Build network & train & test
def relu_cross_net(NumUnit,LearnRate,Momentum):
    model = Sequential()
    model.add(GaussianNoise(0.02, input_shape=(56,)))
    model.add(Dense(NumUnit, input_dim=56, W_regularizer=l2(0.02), init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, W_regularizer=l2(0.02), init='glorot_normal'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=LearnRate, momentum=Momentum, decay=0.005, nesterov=True)
    adam = Adam(lr=LearnRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005)
    model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
    model.fit(X_train, Y_train, batch_size=100, nb_epoch=1000, shuffle=True, verbose=0, callbacks=[earlyStopping], validation_split=0.2)

    Y_train_pred = model.predict_classes(X_train, verbose=0)
    train_acc = np.sum(train_label == Y_train_pred, axis=0) / float(X_train.shape[0])
    Y_test_pred = model.predict_classes(X_test, verbose=0)
    test_acc = np.sum(test_label == Y_test_pred, axis=0) / float(X_test.shape[0])

    with open('ExperimentResult.txt','a') as f:
        print('\n\n'
              'Test results of writer',i,':', '\n'
              'Training accuracy: %.4f%%' % (train_acc * 100), '\n'
              'TestF1 predictions: ', Y_test_pred[0:22], '\n'
              'TestF2 predictions: ', Y_test_pred[22:44], '\n'
              'TestG1 predictions: ', Y_test_pred[44:66], '\n'
              'TestG2 predictions: ', Y_test_pred[66:88], '\n'
              'Test overall accuracy: %.4f%%' % (test_acc * 100), file = f)

    print('TestF1 predictions: ', Y_test_pred[0:22])
    print('TestF2 predictions: ', Y_test_pred[22:44])
    print('TestG1 predictions: ', Y_test_pred[44:66])
    print('TestG2 predictions: ', Y_test_pred[66:88])
    print('Test overall accuracy: %.4f%%' % (test_acc * 100))


for i in range(51,101):
    #Load data
    train_data = np.loadtxt(open("writer"+str(i)+"traindata.csv","rb"),delimiter=",",skiprows=0)
    test_data = np.loadtxt(open("writer"+str(i)+"testdata.csv","rb"),delimiter=",",skiprows=0)
    X_train,Y_train = np.hsplit(train_data,(56,))
    train_label = np.mat(Y_train)
    train_label = train_label.getA1()
    X_test,Y_test = np.hsplit(test_data,(56,))
    test_label = np.mat(Y_test)
    test_label = test_label.getA1()
    Y_train = np_utils.to_categorical(Y_train, 2)
    Y_test = np_utils.to_categorical(Y_test, 2)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #Build network & train & test
    relu_cross_net(256, 0.002, 0.9)