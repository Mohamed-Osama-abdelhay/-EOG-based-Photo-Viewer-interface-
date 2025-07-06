import numpy as np
from pickle import dump
from pywt import wavedec
from save import save_data
from sklearn.svm import SVC
from random import shuffle, seed
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt, decimate
from statsmodels.tsa.ar_model import AutoReg

def read_file(file_path):
    values = []

    with open(file_path) as file:
        lines = file.readlines()

    for line in lines:
        values.append(int(line))

    return values

def read_signals():
    file_path = 'D:/FCIS - ASU/Y4S2/Human Computer Interface/Project/classes'
    classes = {"yukari":"Up", "asagi":"Down", "sag":"Right", "sol":"Left", "kirp":"Blink"}

    train_h = []; train_v = []; test_h = []; test_v = []
    
    keys = classes.keys()
    for key in keys:
        for i in range(1, 21):
            signal_h = read_file(f'{file_path}/{classes[key]}/{key}{i}h.txt')
            signal_v = read_file(f'{file_path}/{classes[key]}/{key}{i}v.txt')
            if i < 16:
                train_h.append(signal_h)
                train_v.append(signal_v)
            else:
                test_h.append(signal_h)
                test_v.append(signal_v)

    save_data(train_h, train_v, test_h, test_v, "before_preprocessing")
    return np.array(train_h), np.array(train_v), np.array(test_h), np.array(test_v)

def bandpass_filter(signal, Low_Cutoff = 1, High_Cutoff = 20, SamplingRate = 176, order = 2):
    nyq = SamplingRate / 2
    low = Low_Cutoff / nyq
    high = High_Cutoff / nyq

    b, a = butter(order, [low, high], btype = 'band', analog = False, fs = None)
    Filtered_Data = filtfilt(b, a, signal)

    return Filtered_Data

# Down Sampling
def resampling(signal):
    return decimate(signal, 5)

# 0 -> 1
def normalization(signal):
    max = np.max(signal)
    min = np.min(signal)
    signal = np.array(signal)
    normalized_signal = (signal - min) / (max - min)
    return normalized_signal

def dc_removal(signal):
    signal = np.array(signal)
    signal = signal - np.mean(signal)
    return signal

def calc_wavelet(signal):
    coeffs = wavedec(signal, 'db4', level=2)
    return coeffs[0]

def calc_auto_regressive(signal):
    model = AutoReg(signal, lags=19)
    model_fit = model.fit()
    return model_fit.params

def model(signal_train, signal_test):
    global svm
    x_train = signal_train[:, :-1]
    y_train = signal_train[:, -1]

    x_test = signal_test[:, :-1]
    y_test = signal_test[:, -1]

    svm = SVC(kernel='rbf', C=1, gamma="scale")
    svm.fit(x_train, y_train)

    y_train_pred = svm.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred) * 100

    y_test_pred = svm.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred) * 100

    return train_acc, test_acc

def preprocessing(train_h, train_v, test_h, test_v):
    wavelet_train_h = [];  wavelet_train_v = []; wavelet_test_h = []; wavelet_test_v = []; wavelet_train = []; wavelet_test = []
    ar_train_h = []; ar_train_v = []; ar_test_h = []; ar_test_v = []; ar_train = []; ar_test = []
    tmp1_h = []; tmp1_v = []; tmp2_h = []; tmp2_v = []

    train_h = bandpass_filter(train_h)
    train_v = bandpass_filter(train_v)
    test_h = bandpass_filter(test_h)
    test_v = bandpass_filter(test_v)

    for i in range(len(train_h)):
        tmp1_h.append(resampling(train_h[i]))
        tmp1_v.append(resampling(train_v[i]))

        wavelet_train_h.append(calc_wavelet(tmp1_h[i]))
        wavelet_train_v.append(calc_wavelet(tmp1_v[i]))

        ar_train_h.append(calc_auto_regressive(tmp1_h[i]))
        ar_train_v.append(calc_auto_regressive(tmp1_v[i]))

        if i < len(test_h):
            tmp2_h.append(resampling(test_h[i]))
            tmp2_v.append(resampling(test_v[i]))

            wavelet_test_h.append(calc_wavelet(tmp2_h[i]))
            wavelet_test_v.append(calc_wavelet(tmp2_v[i]))

            ar_test_h.append(calc_auto_regressive(tmp2_h[i]))
            ar_test_v.append(calc_auto_regressive(tmp2_v[i]))
        
    save_data(tmp1_h, tmp1_v, tmp2_h, tmp2_v, "after_preprocessing")

    wavelet_train = np.concatenate((np.array(wavelet_train_h), np.array(wavelet_train_v)), axis=1)
    wavelet_test = np.concatenate((np.array(wavelet_test_h), np.array(wavelet_test_v)), axis=1)
    ar_train = np.concatenate((np.array(ar_train_h), np.array(ar_train_h)), axis=1)
    ar_test = np.concatenate((np.array(ar_test_h), np.array(ar_test_v)), axis=1)
    
    save_data(wavelet_train, wavelet_test, ar_train, ar_test, "after_feature_extraction")

    # Adding labels
    c1, c2 = 0, 0
    label1, label2 = 1, 1
    for i in range(len(wavelet_train)):
        wavelet_train[i][-1] = label1
        ar_train[i][-1] = label1

        c1 += 1
        if c1 % 15 == 0:
            c1 = 0
            label1 += 1
        
        if i < len(wavelet_test):
            wavelet_test[i][-1] = label2
            ar_test[i][-1] = label2

            c2 += 1
            if c2 % 5 == 0:
                c2 = 0
                label2 += 1

    return wavelet_train, wavelet_test, ar_train, ar_test

def main():

    # Read & Split Signals
    train_h, train_v, test_h, test_v = read_signals()

    # Preprocessing & Feature Extraction
    wavelet_train, wavelet_test, ar_train, ar_test = preprocessing(train_h, train_v, test_h, test_v)

    # Shuffling
    seed(999)
    shuffle(wavelet_train)
    shuffle(ar_train)

    train_acc, test_acc = model(np.array(wavelet_train), np.array(wavelet_test))
    print(f"\nWavelet Train Accuracy = {round(train_acc)}%")
    print(f"Wavelet Test Accuracy = {round(test_acc)}%\n")

    with open('D:/FCIS - ASU/Y4S2/Human Computer Interface/Project/svm_model.pkl', 'wb') as file:
        dump(svm, file)

    train_acc, test_acc = model(np.array(ar_train), np.array(ar_test))
    print(f"Auto Regression Train Accuracy = {round(train_acc)}%")
    print(f"Auto Regression Test Accuracy = {round(test_acc)}%")

if __name__ == "__main__":
    main()