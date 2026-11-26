import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .auxiliary_funcs import autocor
import os


def plot_with_period(df_data_aggregated, diffs, other_tolerances, best_tolerance, lag_list, r_list, p_list, corfunc, model_data, norm_diff_between_singal_and_model, plot_tolerances, level_of_significance_for_pearson, consider_only_significant_correlation, minimum_number_of_datapoints_for_correlation_test):
    '''
    This is the regular plotting function in case a period is detected, comparing the real data with the model.
    It plots a comparison of the real data with the one from the model, 
    the norm difference between the unshifted and shifted autocorrelation function (optional: plot possible tolerances and the one with the best performance), 
    the autocorrelation function of the data,
    the difference between data and model (noise) and
    the autocorrelation function of the real data minus the autocorrelation function of the model.
    :param df_data_aggregated: pd.DataFrame  原始信号时间序列（含 time、value 列）-->绘制原始数据与模型拟合曲线
    :param diffs: list of positive floats    shift autocorrelation 差异序列  -->绘制差异函数曲线以展示周期检测趋势
    :param other_tolerances: list of positive floats   其他容差参数-->标出其他容忍度下的检测结果
    :param best_tolerance: positive float     当前最佳容差参数-->标出最终使用的容差
    :param lag_list: list of positive integers   滞后列表（自相关横坐标）-->绘制自相关函数横轴
    :param r_list: list of floats     各滞后的相关系数 (Pearson r)-->绘制自相关函数主图
    :param p_list: list of floats     各滞后的显著性 p 值-->可在图中标出显著滞后位置
    :param corfunc: list of floats between -1 and  1   自相关函数值列表-->绘制完整自相关曲线（ACF）
    :param model_data: list of 1-D lists of floats    模型拟合出的周期性信号-->绘制模型拟合与原信号的对比图
    :param norm_diff_between_singal_and_model: positive float   信号与模型的平均差值-->用于图表标题或评估拟合优劣
    :param plot_tolerances: Boolean   是否显示容差线-->控制在图中标注容忍区间
    :param level_of_significance_for_pearson: positive float  皮尔逊显著性水平-->标出显著与非显著滞后点
    :param consider_only_significant_correlation: Boolean   是否仅考虑显著相关-->图中是否只显示显著自相关峰
    :param  minimum_number_of_datapoints_for_correlation_test: positive integer  自相关检验最小数据点数-->决定绘图范围与有效滞后区间
    '''
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

    signal_data=df_data_aggregated["value"].to_numpy().reshape(df_data_aggregated["value"].size, 1)
    norm_model = sum(abs(model_data))[0] / model_data.size  #模型信号的平均“能量强度”（类似归一化振幅）
    norm_signal = sum(abs(signal_data))[0] / signal_data.size
    signal_subtracted_model = signal_data - model_data
    #计算模型拟合误差（即原信号与预测信号的差）,如果模型捕捉了主要周期成分，那么：残差信号会更“随机”（周期性被削弱）；残差信号会更“随机”（周期性被削弱）；
    df_data_difference_signal_model = pd.DataFrame(data=signal_subtracted_model, columns=["value"])
    r_list_diff, p_list_diff, corfunc_diff, lag_list_diff = autocor(df_data_difference_signal_model["value"], list(range(0, int((df_data_difference_signal_model["value"].size) - minimum_number_of_datapoints_for_correlation_test))),level_of_significance_for_pearson,consider_only_significant_correlation)
    print('Norm of difference between signal and model: ' + str(norm_diff_between_singal_and_model))  #残差信号的平均幅度,越小表示模型越好（周期性被解释掉）
    print('Norm model: ' + str(norm_model))  #原始信号的平均幅度-->表示信号整体强度
    print('Norm signal: ' + str(norm_signal)) # 模型信号的平均幅度-->模型解释的能量


    plt.plot(df_data_aggregated["value"], label="data")
    plt.plot(model_data, label="model")
    plt.legend()
    plt.title("Data and fitted model")
    plt.xlabel('Time in time points')
    plt.savefig(os.path.join(SAVE_DIR, 'data_model.png'), format='png')
    plt.show()


    #衡量自相关函数右移 i 位后与原函数的差异（L¹ 距离）。低谷（local minima）的位置表示可能的周期点。（因为此时“移位前后的自相关函数差异最小” → 周期重合。）
    plt.plot(list(np.array(diffs)))
    plt.title('Norm of difference between unshifted and shifted autocorrelation function')  #非移位与移位自相关函数之间的差异；
    plt.xlabel('Shift in time points')   #时间滞后；
    plt.ylabel('Function values of d')   #差异函数的值（越小越周期）。
    if plot_tolerances == 1:
        plt.hlines(y=other_tolerances, xmin=0, xmax=len(diffs), colors='green')
        plt.hlines(y=[best_tolerance], xmin=0, xmax=len(diffs), colors='red')
    plt.savefig(os.path.join(SAVE_DIR, 'shift_acf.png'), format='png')
    plt.show()

    #原始信号的自相关函数
    plt.plot(lag_list, r_list, label="r")
    plt.plot(lag_list, p_list, label="p")
    plt.plot(lag_list, corfunc, label="acf")
    plt.legend()
    plt.title("Autocorrelation function of data")
    plt.xlabel('Shift of the data in time points')
    plt.savefig(os.path.join(SAVE_DIR, 'acf.png'), format='png')
    plt.show()

    plt.plot(df_data_difference_signal_model["value"])
    plt.title("Difference of data and model")
    plt.savefig(os.path.join(SAVE_DIR, 'Difference of data and model.png'), format='png')
    plt.show()

    plt.plot(lag_list_diff, r_list_diff, label="r")
    plt.plot(lag_list_diff, p_list_diff, label="p")
    plt.plot(lag_list_diff, corfunc_diff, label="acf")
    plt.legend()
    plt.title("Autocorrelation function of data minus model")
    plt.xlabel('Shift of the data minus model in time points')
    plt.savefig(os.path.join(SAVE_DIR, 'acf_data_model.png'), format='png')
    plt.show()

def plot_without_period(df_data_aggregated, diffs, lag_list, r_list, p_list, corfunc):
    '''
    This function plots the results when no period was suggested.
    It shows the time series data,
    the norm difference between unshited and shifted autocorrelation function and 
    the autocorrelation function of the time series.
    :param df_data_aggregated: pd.DataFrame
    :param diffs: list of positive floats
    :param lag_list: list of positive integers
    :param r_list: list of floats
    :param p_list: list of floats
    :param corfunc: list of floats between -1 and  1
    '''
    plt.plot(df_data_aggregated["value"], label="data")
    plt.legend()
    plt.title("time series data")
    plt.show()

    plt.plot(list(np.array(diffs)))
    plt.title('Norm of difference between unshifted and shifted autocorrelation function')
    plt.xlabel('Shift in time points')
    plt.show()

    plt.plot(lag_list, r_list, label="r")
    plt.plot(lag_list, p_list, label="p")
    plt.xlabel('Time in data points')
    plt.legend()
    plt.title("Autocorrelation function of data")
    plt.show()
