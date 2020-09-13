import time

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras



class AI:

    def __init__(self, EPOCH=100, LEARNING_RATE=0.3, hidden_size=120, seed=1):
        self.isCreated = False

        mnist = keras.datasets.mnist
        (x_train,t_train),(x_test,t_test) = mnist.load_data()

        x_train = x_train.reshape((x_train.shape[0]),-1)
        t_train = keras.utils.to_categorical(t_train)
        self.x_test = x_test.reshape((x_test.shape[0]),-1)
        self.t_test = keras.utils.to_categorical(t_test)


        self.DATA_SIZE = np.size(x_train, 0)
        self.TEST_SIZE = np.size(self.x_test, 0)
        self.VALIDATION_SIZE = 10000
        self.TRAIN_SIZE = self.DATA_SIZE - self.VALIDATION_SIZE
        self.FEATURES = 784
        self.LEARNING_RATE = LEARNING_RATE
        self.EPOCH = EPOCH

        #np.random.seed(seed)
        np.random.seed()
        train_idx = np.random.choice(self.DATA_SIZE, self.TRAIN_SIZE, replace=False)
        test_idx = np.setdiff1d(range(self.DATA_SIZE), train_idx)

        self.train_x = x_train[train_idx, :] / 255
        self.train_t = t_train[train_idx]

        self.valid_x = x_train[test_idx, :] / 255
        self.valid_t = t_train[test_idx]

        self.L1_size = 784
        self.L1_weight = np.random.randn(self.L1_size, 1)
        self.L1_result = np.zeros((self.L1_size, 1), dtype=float)

        self.L2_size = hidden_size
        self.L2_weight = np.random.randn(self.L2_size, self.L1_size)
        self.L2_result = np.zeros((self.L2_size, 1), dtype=float)

        self.L3_size = 10
        self.L3_weight = np.random.randn(self.L3_size, self.L2_size)
        self.L3_result = np.zeros((self.L3_size, 1), dtype=float)
        self.G_L3 = np.zeros((self.L3_size, 1), dtype=float)

        self.ERR = np.zeros((self.EPOCH, 1), dtype=float)
        self.ERR_entire = np.zeros((self.EPOCH * self.TEST_SIZE, 1), dtype=float)

    def __sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def __ReLU(self, x):
        return max(0, x)

    def __G_ReLU(self, x):
        if (x <= 0):
            return 0
        else:
            return 1

    def __softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def Run(self,dist=0):
        print('*********************************************')
        print('*******************  RUN  *******************\n')


        self.best = 0
        self.start_time = time.time()
        self.local = 0

        for epoch in range(self.EPOCH):

            for index in range(self.TRAIN_SIZE):
                self.L1_drop, self.L1_mask = self.__DropOut(self.L1_size)
                self.L2_drop, self.L2_mask = self.__DropOut(self.L2_size)
                #self.L1_weight, L1_droped = self.__DropOut(self.L1_weight)
                #self.L2_weight, L2_droped = self.__DropOut(self.L2_weight)
                #print(np.around(self.L2_weight[:10,0]),np.around(L2_droped[:10,0]))
                # self.L3_weight,L3_droped = self.__DropOut(self.L3_weight)

                droped_data_L1, droped_data_L2 = self.__Forward(index)
                self.__Backward(index)

                self.L1_weight += droped_data_L1
                self.L2_weight += droped_data_L2

                #self.L1_weight = self.L1_weight + L1_droped
                #self.L2_weight = self.L2_weight + L2_droped
                # self.L3_weight = self.L3_weight + L3_droped

            self.__Validate(epoch, dist)

        if self.local > self.best:
            self.isCreated = True
            self.best = self.local
            self.best_w1 = self.temp_w1
            self.best_w2 = self.temp_w2
            self.best_w3 = self.temp_w3

        self.__Testing()


    def __Forward(self, index):

        # 가중치 값에 드랍 아웃 및 데이터 저장
        L1_droped = self.L1_weight * self.L1_mask
        self.L1_weight = self.L1_weight * self.L1_drop
        L2_droped = self.L2_weight * self.L2_mask
        self.L2_weight = self.L2_weight * self.L2_drop

        x = self.train_x[index]
        w = self.L1_weight.T
        y = w * x
        z = self.__sigmoid(y)
        self.L1_result = z.T
        # 결과 값에도 드랍 아웃 적용
        self.L1_result = self.L1_result * self.L1_drop

        x = self.L1_result
        w = self.L2_weight
        y = np.dot(w, x)
        z = self.__sigmoid(y)
        self.L2_result = z
        self.L2_result = self.L2_result * self.L2_drop

        x = self.L2_result
        w = self.L3_weight
        y = np.dot(w, x)
        z = self.__sigmoid(y)
        self.L3_result = z

        return L1_droped, L2_droped

        # for i in range(L3_size):
        # z = ReLU(y[i])
        # z_ = G_ReLU(y[i])

        # L3_result[i] = z
        # G_L3[i] = z_

    def __Backward(self, index):
        e = self.train_t[index] - self.L3_result.T

        # delta_3 = (e*G_L3.T).T
        delta_3 = e.T * self.L3_result * (1 - self.L3_result)
        delta_w = self.LEARNING_RATE * delta_3 * self.L2_result.T
        self.L3_weight = self.L3_weight + delta_w

        delta_2 = np.dot(self.L3_weight.T, delta_3) * self.L2_result * (1 - self.L2_result)
        delta_w = self.LEARNING_RATE * delta_2 * self.L1_result.T
        self.L2_weight = self.L2_weight + delta_w * self.L2_drop

        delta_1 = np.dot(self.L2_weight.T, delta_2) * self.L1_result * (1 - self.L1_result)
        delta_w = self.LEARNING_RATE * delta_1.T * self.train_x[index]
        self.L1_weight = self.L1_weight + delta_w.T * self.L1_drop

    def __Validate(self, epoch, dist):
        local = 0
        for index in range(self.VALIDATION_SIZE):

            L1_result = self.__sigmoid(self.L1_weight.T * self.valid_x[index]).T
            L2_result = self.__sigmoid(np.dot(self.L2_weight, L1_result))
            L3_result = self.__sigmoid(np.dot(self.L3_weight, L2_result))

            # for i in range(y.size):
            #    L3_result[i] = ReLU(np.dot(w,L2_result)[i])
            # z = softmax(L2_result)

            e = self.valid_t[index] - L3_result.T
            e_ = (e * e) / 2

            self.ERR_entire[index + (epoch * self.VALIDATION_SIZE)] = e_.sum()
            self.ERR[epoch] += e_.sum()

            if (np.argmax(L3_result) == np.argmax(self.valid_t[index])):
                local = local + 1

        if local > self.local:
            self.local = local
            self.temp_w1 = self.L1_weight
            self.temp_w2 = self.L2_weight
            self.temp_w3 = self.L3_weight
        elif local < self.local:
            self.LEARNING_RATE = round(self.LEARNING_RATE * 0.8, 5)

        if epoch % dist == 0:
            print('epoch count    : {}/{} '.format(epoch, self.EPOCH))
            print('learning rate  : ', self.LEARNING_RATE)
            print('start to time  : ', round(time.time() - self.start_time, 2))
            print('Accuracy count : ', local)
            print('Accuracy rate  : ', (local / self.VALIDATION_SIZE) * 100)
            print('-------------------------------------------------')

        # TP[np.argmax(valid_t[index])] += 1
        # TN += 1
        # TN[np.argmax(valid_t[index])] -=1

        # else:
        # FP[np.argmax(L3_result)] += 1
        # TN += 1

        # TN[np.argmax(valid_t[index])] -=1
        # TN[np.argmax(L3_result)] -= 1

        # FN[np.argmax(valid_t[index])] += 1

        # TP ; T -> 맞
        # TN : F -> 틀
        # FP : F -> 맞
        # FN : T -> 틀

        # elif(pre_acry == final_acry):
        #    print(pre_acry,final_acry)
        #    ERR = ERR[:epoch]
        #    ERR_entire =ERR_entire[:epoch]
        #    print('학습 종료')
        #    break

        # L1_weight = best_w1
        # L2_weight = best_w2
        # L3_weight = best_w3

        # print('***************RESULT***************')
        # print('final time time   : ', round(time.time() - real_start, 2))
        # print('final epoch count : ', epoch)
        # print('final accuracy    : ', pre_acry / TEST_SIZE)

    def __Testing(self):
        final_e = 0
        e1 = 0
        e2 = 0

        self.TP = np.zeros((20, 10), dtype=float)
        self.TN = np.zeros((20, 10), dtype=float)
        self.FP = np.zeros((20, 10), dtype=float)
        self.FN = np.zeros((20, 10), dtype=float)

        acry = 0

        for index in range(self.TEST_SIZE):

            L1_result = self.__sigmoid(self.best_w1.T * self.x_test[index]).T
            L2_result = self.__sigmoid(np.dot(self.best_w2, L1_result))
            L3_result = self.__sigmoid(np.dot(self.best_w3, L2_result))

            e = self.t_test[index] - L3_result.T
            e_ = (e * e) / 2

            final_e += e_.sum()

            if (np.argmax(L3_result) == np.argmax(self.t_test[index])):
                acry = acry + 1
                e1 += e_.sum()
                self.TP[index % 20][np.argmax(self.t_test[index])] += 1
                self.TN[index % 20] += 1
                self.TN[index % 20][np.argmax(self.t_test[index])] -= 1

            else:
                e2 += e_.sum()
                self.FP[index % 20][np.argmax(L3_result)] += 1
                self.TN[index % 20] += 1

                self.TN[index % 20][np.argmax(self.t_test[index])] -= 1
                self.TN[index % 20][np.argmax(L3_result)] -= 1

                self.FN[index % 20][np.argmax(self.t_test[index])] += 1

        print('***************RESULT***************')
        print('model accuracy        : ', round((acry / self.TEST_SIZE * 100), 5))
        print('model accuracy        : ', acry)
        print('model error is        : ', final_e)
        print('model correct error   : ', e1)
        print('model incorrect error : ', e2)

    def __DropOut(self, size):
        np.random.seed()
        drop = np.random.randn(size, 1)
        mask = np.zeros((size, 1))
        num  =0

        for i, data in enumerate(drop[:, 0]):
            if (data > 1.5):
                num+=1
                drop[i, 0] = 0
                mask[i, 0] = 1
            else:
                drop[i, 0] = 1

        return drop, mask

    def Err_table(self):
        Sensitivity = np.around((self.TN / (self.TN + self.FN)), 2)
        Specificity = np.around((self.TN / (self.TN + self.FN)), 2)
        print('test-------------------------------------------')
        print("TP     : ", self.TP[1])
        print('TN     : ', self.TN[1])
        print('FP     : ', self.FP[1])
        print('FN     : ', self.FN[1])
        print('민감도 : ', Sensitivity[1])
        print('특이도 : ', Specificity[1])
        print('good---------------------------------------------')

    def Err_Graph(self):
        plt.subplot(2, 1, 1)
        x1 = np.arange(0, np.size(self.ERR))
        plt.plot(x1, self.ERR)

        plt.subplot(2, 1, 2)
        x1 = np.arange(0, np.size(self.ERR_entire))
        plt.plot(x1, self.ERR_entire)

        plt.show()

# EPOCH=100, LEARNING_RATE=0.3, hidden_size=120, seed=1
a = AI(20) # epoch 10 짜리 모델 생성
a.Run(1)   # 출력 epoch 간격
a.Err_Graph()

