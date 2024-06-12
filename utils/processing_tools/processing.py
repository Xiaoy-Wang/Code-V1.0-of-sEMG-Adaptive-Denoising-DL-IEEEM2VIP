import math
import numpy as np
import librosa
from utils.processing_tools.wavelet_denoising import WaveletDenoising
from utils.processing_tools.wavelet_packet_denoising import WaveletPacketDenoising
from utils.processing_tools.iceemdan_pe_denoising import ICEEMDANPEDenoising
from scipy.signal import butter, lfilter, iirfilter
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

"""信号重采样"""


def signal2d_resampling(rawdata, raw_fs, tar_fs):
    channel = rawdata.shape[1]
    x = []
    for i in range(channel):
        temp = librosa.resample(rawdata[:, i], orig_sr=raw_fs, target_sr=tar_fs, res_type='kaiser_best',
                                fix=True, scale=False)
        x.append(temp)
    x = np.array(x).transpose()
    return x


"""emg滤波器：陷波滤波、带通滤波、低通滤波"""


class emg_filtering:
    def __init__(self, fs, lowcut, highcut, imf_band, imf_freq):
        self.fs = fs
        # butterWorth带通滤波器
        self.lowcut, self.highcut = lowcut, highcut
        # 50 Hz陷波滤波器
        self.imf_band, self.imf_freq = imf_band, imf_freq
        # 低通滤波
        self.cutoff = 15

    def Implement_Notch_Filter(self, data, order=3, filter_type='butter'):
        # Required input defintions are as follows;
        # time:   Time between samples
        # band:   The bandwidth around the centerline freqency that you wish to filter
        # freq:   The centerline frequency to be filtered
        # ripple: The maximum passband ripple that is allowed in db
        # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
        #         IIR filters are best suited for high values of order.  This algorithm
        #         is hard coded to FIR filters
        # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
        # data:         the data to be filtered
        fs = self.fs
        nyq = fs / 2.0
        freq, band = self.imf_freq, self.imf_band
        low = freq - band / 2.0
        high = freq + band / 2.0
        low = low / nyq
        high = high / nyq
        b, a = iirfilter(order, [low, high], btype='bandstop', analog=False, ftype=filter_type)
        filtered_data = lfilter(b, a, data)

        return filtered_data

    def butter_bandpass(self, order=7):
        lowcut, highcut, fs = self.lowcut, self.highcut, self.fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        return b, a

    def butter_bandpass_filter(self, data):
        b, a = self.butter_bandpass()
        y = lfilter(b, a, data)

        return y

    def butter_lowpass(self, order=5):
        cutoff, fs = self.cutoff, self.fs
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        return b, a

    def butter_lowpass_filter(self, data):
        b, a = self.butter_lowpass()
        y = lfilter(b, a, data)

        return y


"""emg去噪"""


class Signal2dDenoise:
    def __init__(self, data, denoise_method):
        self.data = data
        self.denoise_method = denoise_method
        assert self.denoise_method in ['rawdata', 'WD-GT', 'WPD-GT', 'EMD-PE-GT', 'EMD-PE-SVD',
                                       'EEMD-PE-GT', 'EEMD-PE-SVD', 'ICEEMDAN-PE-GT', 'ICEEMDAN-PE-SVD']
        print('emg denoising method: %s' % self.denoise_method)
        if self.denoise_method == 'WD-GT':
            # pyyawt.denoising.wden(*args)
            #  [XD,CXD,LXD] = wden(X,TPTR,SORH,SCAL,N,wname) or [XD,CXD,LXD] = wden(C,L,TPTR,SORH,SCAL,N,wname)
            # X: 输入; C/L：稀疏矩阵； TPTR: str阈值选择规则（‘rigrsure’运用了斯坦因的无偏风险原则；` heursure `是第一个选项的启发式变体；` sqtwlog `用于通用阈值` minimaxi `用于minimax阈值化）
            # ‘visushrink’ = delta*sqrt(2lnN), delta = MAD/0.6745, MAD是所有高频子带小波系数幅度的中值。（这个是全局的思路，如果MAD是各个分解层的各高频子带系数幅度的中值，那就是局部的思路）
            # delta = np.median(np.abs(coeffs[-1])) / 0.6745
            # sqtwlog：lamda = sqrt(2lnN)（相较于VisuShrink阈值少了噪声的标准差，相当于是针对标准差(小波域)为1的高斯白噪声而言的）;  heursure: ‘rigrsure’和sqtwlog的综合形式，
            # SORH: str；(' s '或' h ')软阈值或硬阈值

            # thr_mode= {'soft', 'hard', 'garrote', 'greater', 'less'}(``garrote``介于“硬”和“软”阈值之间。It behaves like soft thresholding for small data values and approaches hard thresholding for large data values.)
            # method = {'universal'(即visushrink)， ‘sqtwolog’， ‘energy’（energy_perc=0.90），‘stein’（即‘rigrsure’），‘heurstein’}
            self.wd = WaveletDenoising(normalize=False, wavelet='db5', level=3, thr_mode='garrote',
                                       selected_level=None, method="universal", energy_perc=0.90)
        elif self.denoise_method == 'WPD-GT':
            self.wpd = WaveletPacketDenoising(normalize=False, wavelet='db5', level=3, thr_mode='garrote',
                                              method="universal", energy_perc=0.90)
        elif self.denoise_method == 'EMD-PE-GT':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='emd',
                                                             denoise_method='garrote_threshold')
        elif self.denoise_method == 'EMD-PE-SVD':
            # 奇异值阈值设置：奇异值突变的值，奇异值的平均值；svd_threshold_type = {'mutation_value', 'mean_value'}
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='emd',
                                                             denoise_method='svd', svd_threshold_type='mutation_value')
        elif self.denoise_method == 'EEMD-PE-GT':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='eemd',
                                                             denoise_method='garrote_threshold')
        elif self.denoise_method == 'EEMD-PE-SVD':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='eemd',
                                                             denoise_method='svd', svd_threshold_type='mutation_value')
        elif self.denoise_method == 'ICEEMDAN-PE-GT':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='iceemdan',
                                                             denoise_method='garrote_threshold')
        else:
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='iceemdan',
                                                             denoise_method='svd', svd_threshold_type='mutation_value')
        self.denoised_data = None

    def forward(self):
        print('Signal Length: %d' % len(self.data))
        if self.denoise_method == 'rawdata':
            self.denoised_data = self.data
        elif self.denoise_method == 'WD-GT':
            temp = []
            for i in range(self.data.shape[1]):
                print('Prosessing Signal Channel: %d' % (i + 1))
                temp.append(self.wd.fit(self.data[:, i]))
            self.denoised_data = np.array(temp).T
        elif self.denoise_method == 'WPD-GT':
            temp = []
            for i in range(self.data.shape[1]):
                print('Prosessing Signal Channel: %d' % (i + 1))
                temp.append(self.wpd.fit(self.data[:, i]))
            self.denoised_data = np.array(temp).T
        else:
            temp = []
            for i in range(self.data.shape[1]):
                print('Prosessing Signal Channel: %d' % (i + 1))
                denoised_signal = self.iceemdan_pe_denoising.fit(self.data[:, i])
                temp.append(denoised_signal)
            self.denoised_data = np.array(temp).T

        return self.denoised_data


"""多模态多通道数据归一化方法，其中支持归一化方法：'min-max'、'max-abs'、'positive_negative_one；归一化层面：'matrix'、'rows'"""


def data_nomalize(data, normalize_method, normalize_level):
    if normalize_level == 'matrix':
        if normalize_method == 'min-max':
            # 实例化 MinMaxScaler 并设置归一化范围为 [0, 1]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 来进行整体归一化
            normalized_data = (data - np.min(scaler.data_min_)) / (np.max(scaler.data_max_) - np.min(scaler.data_min_))
        elif normalize_method == 'positive_negative_one':
            # 实例化 MinMaxScaler 并设置归一化范围为 [-1, 1]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 来进行整体归一化
            # print(np.min(scaler.data_min_),np.max(scaler.data_max_))
            normalized_data = ((data - np.min(scaler.data_min_)) / (
                    np.max(scaler.data_max_) - np.min(scaler.data_min_))) * 2 - 1
        elif normalize_method == 'max-abs':
            # 实例化 MaxAbsScaler，并拟合数据以计算每列的最大值和最小值的绝对值
            scaler = MaxAbsScaler()
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 将数据整体缩放到 [-1, 1] 范围内
            normalized_data = (data / np.maximum(np.abs(np.max(scaler.data_max_)),
                                                 np.abs(np.min(scaler.data_min_)))) * scaler.scale_
        else:
            print('Error: 未识别的normalize_method！')
    elif normalize_level == 'rows':
        if normalize_method == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'positive_negative_one':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'max-abs':
            scaler = MaxAbsScaler()
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        else:
            print('Error: 未识别的normalize_method！')
    else:
        print('Error: 未识别的normalize_level！')

    return normalized_data


### 对运动模式分类任务

"""基于滑动重叠窗口采样的样本集分割：重叠窗长window、步进长度step"""


def overlapping_windowing_movement_classification(emg_data_act, movement, window, step):
    length = math.floor((np.array(emg_data_act).shape[0] - window) / step)
    emg_sample, movement_label = [], []
    for j in range(length):
        sub_emg_sample = emg_data_act[step * j:(window + step * j), :]
        emg_sample.append(sub_emg_sample)
        movement_label.append(movement)

    return np.array(emg_sample), np.array(movement_label)


"""活动段提取和重叠窗口分割"""


def movement_classification_sample_segmentation(movement, emg_data_act, window, step):
    print('       进行重叠窗分割...')
    emg_sample, movement_label = overlapping_windowing_movement_classification(emg_data_act, movement, window, step)
    print('       emg_sample.shape: ', emg_sample.shape, ', movement_label.shape: ', movement_label.shape)

    return emg_sample, movement_label


def get_emg_act_signal(movement, emg_raw_data, status_label):
    if movement in ['WAK', 'UPS', 'DNS']:
        print('       运动类型为: ', movement, '，无需处理活动段...')
        emg_data_act = emg_raw_data
    else:
        print('       运动类型为: ', movement, '，开始处理活动段...')
        indices_a = np.where(status_label == 'A')
        emg_data_act = emg_raw_data[indices_a[0], :]

    return emg_data_act
