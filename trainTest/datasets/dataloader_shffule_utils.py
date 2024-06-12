import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

'''
    类initDataset的作用：
        initIntroSubjectDataset
        initInterSubjectDataset
    初始化时需传递的参数及其含义参数：
        initIntroSubjectDataset:
            path                 : 受试者npz文件路径
            raw_data_list        : 模态信号名称列表，以npz文件的字典规则为依据
            label_name           : 用作label的npz文件字段名 ，应该是'..._encoded'
            total_exp_time       : 设计的总重复试验次数，步态相位识别中不起作用
            说明：
                initInroSubjectDataset等于给出了针对某一确定受试者提取train，valid，test DataSet的方法。当gait_or_motion参数为'motion'时，使用sklearn中的StratifiedShuffleSplit方法划分成
                total_exp_time组（train valid test）。 

        initInterSubjectDataset:


            说明：
        getDataLoader:
            exp_time             : 当前为第几次重复试验 ：1~total_exp_time
            train_batch          : 训练集batchSize
            test_batch           : 测试集batchSize
            valid_batch          : 验证集batchSize
'''


class initDatasetShffule:
    # 针对intra任务，读取单个受试者的npz文件，划分训练，测试，验证样本，供getDataLoader生成用于pytorch模型训练的dataLoader
    def initIntraSubjectDataset(self, path, label_name, total_exp_time):
        # 函数返回 train valid test DataSet数据，同时也直接保存在self中
        data_list = ['sub_emg_sample']

        test_ratio = 0.2
        valid_ratio = 0.1 / (1 - test_ratio)

        data = np.load(path)
        raw_data_container = []
        if len(data_list) == 0:
            raise Exception(f"you dont contain raw_time_domain_data,this is not right, check!")
        # 先将各模态信号在通道层拼接，可以通过raw_data_list选择信号模态
        for name in data_list:
            raw_data_container.append(data[name])
        raw_data = np.concatenate(raw_data_container, axis=1)
        self.raw_data_time_step = raw_data.shape[2]  # 记录原始时域信号的样本点（时间步_time_step）
        self.total_data = raw_data
        self.total_label = data[label_name]
        # 程序运行到这里，total_data,total_label就已经形成了
        self.total_data_shape = self.total_data.shape  # 记录一下shape
        # self.total_data = self.total_data.reshape(self.total_data_shape[0], -1)
        # self.total_label_shape = self.total_label.shape  # 记录一下shape
        # print(self.total_data.shape, self.total_label.shape)
        # 使用sklearn将数据集按照比例7：1：2划分成train：valid：test。
        self.X_train = []
        self.X_valid = []
        self.X_test = []
        self.y_train = []
        self.y_valid = []
        self.y_test = []
        sss = StratifiedShuffleSplit(n_splits=total_exp_time, test_size=test_ratio, random_state=42)
        sssForValid = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=42)
        for train_index, test_index in sss.split(self.total_data, self.total_label):
            X_trainAndValid, X_test = self.total_data[train_index], self.total_data[test_index]
            y_trainAndValid, y_test = self.total_label[train_index], self.total_label[test_index]
            self.X_test.append(X_test.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
            self.y_test.append(y_test)
            # 再将X_trainAndValid和Y_trainAndValid种划分出train和valid
            for train_index_, valid_index in sssForValid.split(X_trainAndValid, y_trainAndValid):
                X_train, X_valid = X_trainAndValid[train_index_], X_trainAndValid[valid_index]
                y_train, y_valid = y_trainAndValid[train_index_], y_trainAndValid[valid_index]
            self.X_train.append(X_train.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
            self.X_valid.append(X_valid.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
            self.y_train.append(y_train)
            self.y_valid.append(y_valid)
        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test

    # 作用：返回三种DataSet：trainSet validSet testSet
    def getDataLoader_intra(self, exp_time, train_batch, valid_batch, test_batch):
        train_set = myDataset(data=self.X_train[exp_time - 1], label=self.y_train[exp_time - 1],
                              time_step=self.raw_data_time_step)
        valid_set = myDataset(data=self.X_valid[exp_time - 1], label=self.y_valid[exp_time - 1],
                              time_step=self.raw_data_time_step)
        test_set = myDataset(data=self.X_test[exp_time - 1], label=self.y_test[exp_time - 1],
                             time_step=self.raw_data_time_step)
        train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=valid_batch, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False)
        return train_loader, valid_loader, test_loader

    # 针对inter任务，形成源域和目标域的train valid test DataSet数据，保存到self中
    def getDataSetFromDomain(self, domain_path_list, label_name, total_exp_time):
        X_train_result = [[] for i in range(total_exp_time)]
        y_train_result = [[] for i in range(total_exp_time)]
        X_valid_result = [[] for i in range(total_exp_time)]
        y_valid_result = [[] for i in range(total_exp_time)]
        X_test_result = [[] for i in range(total_exp_time)]
        y_test_result = [[] for i in range(total_exp_time)]
        for path in domain_path_list:
            # 得到源域中每一个受试者的train valid test （每个人都有total_exp_time组试验）
            X_train, y_train, X_valid, y_valid, X_test, y_test = self.initIntraSubjectDataset(path, label_name,
                                                                                              total_exp_time)
            for i in range(total_exp_time):
                X_train_result[i].append(X_train[i])
                y_train_result[i].append(y_train[i])
                X_valid_result[i].append(X_valid[i])
                y_valid_result[i].append(y_valid[i])
                X_test_result[i].append(X_test[i])
                y_test_result[i].append(y_test[i])
        # 下面形成源域train valid test，一共total_exp_time组实验
        for i in range(total_exp_time):
            X_train_result[i] = np.concatenate(X_train_result[i], axis=0)
            y_train_result[i] = np.concatenate(y_train_result[i])
            X_valid_result[i] = np.concatenate(X_valid_result[i], axis=0)
            y_valid_result[i] = np.concatenate(y_valid_result[i])
            X_test_result[i] = np.concatenate(X_test_result[i], axis=0)
            y_test_result[i] = np.concatenate(y_test_result[i])
        # self.X_train,self.y_train,self.X_valid,self.y_valid,self.X_test,self.y_test = X_train_result,y_train_result,X_valid_result,y_valid_result,X_test_result,y_test_result
        return X_train_result, y_train_result, X_valid_result, y_valid_result, X_test_result, y_test_result

    def initInterSubjectDataset(self, source_path_list, target_path_list, label_name, total_exp_time):
        # 按照源域，目标域路径列表分别获取源域目标域的train，valid，test DataSet
        source_X_train, source_y_train, source_X_valid, source_y_valid, source_X_test, source_y_test = self.getDataSetFromDomain(
            source_path_list, label_name, total_exp_time)
        target_X_train, target_y_train, target_X_valid, target_y_valid, target_X_test, target_y_test = self.getDataSetFromDomain(
            target_path_list, label_name, total_exp_time)

        self.source_X_train, self.source_y_train, self.source_X_valid, self.source_y_valid, self.source_X_test, self.source_y_test = source_X_train, source_y_train, source_X_valid, source_y_valid, source_X_test, source_y_test
        self.target_X_train, self.target_y_train, self.target_X_valid, self.target_y_valid, self.target_X_test, self.target_y_test = target_X_train, target_y_train, target_X_valid, target_y_valid, target_X_test, target_y_test

    # 为inter-subject任务设计的获取数据加载器方法
    def getDataLoader_inter(self, exp_time, source_train_batch, source_valid_batch, source_test_batch,
                            target_train_batch, target_valid_batch, target_test_batch):
        # 将源域DataSet封装成DataLoader
        source_train_set = myDataset(data=self.source_X_train[exp_time - 1],
                                     label=self.source_y_train[exp_time - 1], time_step=self.raw_data_time_step)
        source_valid_set = myDataset(data=self.source_X_valid[exp_time - 1],
                                     label=self.source_y_valid[exp_time - 1], time_step=self.raw_data_time_step)
        source_test_set = myDataset(data=self.source_X_test[exp_time - 1], label=self.source_y_test[exp_time - 1],
                                    time_step=self.raw_data_time_step)
        source_train_loader = DataLoader(source_train_set, batch_size=source_train_batch, shuffle=True,
                                         drop_last=True)
        source_valid_loader = DataLoader(source_valid_set, batch_size=source_valid_batch, shuffle=False)
        source_test_loader = DataLoader(source_test_set, batch_size=source_test_batch, shuffle=False)
        # 将目标域DataSet封装成DataLoader
        target_train_set = myDataset(data=self.target_X_train[exp_time - 1],
                                     label=self.target_y_train[exp_time - 1], time_step=self.raw_data_time_step)
        target_valid_set = myDataset(data=self.target_X_valid[exp_time - 1],
                                     label=self.target_y_valid[exp_time - 1], time_step=self.raw_data_time_step)
        target_test_set = myDataset(data=self.target_X_test[exp_time - 1], label=self.target_y_test[exp_time - 1],
                                    time_step=self.raw_data_time_step)
        target_train_loader = DataLoader(target_train_set, batch_size=target_train_batch, shuffle=True,
                                         drop_last=True)
        target_valid_loader = DataLoader(target_valid_set, batch_size=target_valid_batch, shuffle=False)
        target_test_loader = DataLoader(target_test_set, batch_size=target_test_batch, shuffle=False)

        return source_train_loader, source_valid_loader, source_test_loader, target_train_loader, target_valid_loader, target_test_loader



class myDataset(Dataset):
    def __init__(self, data, label, time_step):
        self.data = data
        self.label = label
        self.time_step = time_step

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        singleData = torch.from_numpy(self.data[item, :, :]).unsqueeze(0)
        singleLabel = torch.from_numpy(self.label[item:item + 1]).to(dtype=torch.long).view(-1)
        return singleData, singleLabel


'''
    RepeatDataLoaderIterator:
        

'''


class RepeatDataLoaderIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.dataset = data_loader.dataset

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)

        return batch
