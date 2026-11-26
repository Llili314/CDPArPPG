import numpy as np
import functools
from scipy.signal import find_peaks
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



def calculate_autocorrelation(x,i,level_of_significance_for_pearson,consider_only_significant_correlation):
    '''
    This function calculates the autocorrelation function for a vector x and a shift given as index i. 
    The parameters consider_only_significant_correlation and level_of_significance_for_pearson decide if and which results are considered depending on their significance p.
    :param x: list
    :param i: positive integer
    :param level_of_significance_for_pearson: float between 0 and 1
    :param consider_only_significant_correlation: Boolean or interger in {0,1}
    :return: float between -1 and 1, float between 0 and 1, float between -1 and 1, positive integer
    '''
    if i == 0:
        return 1,0,1,0   #当滞后量 i=0 时，信号与自身完全对齐，返回，r：完全相关 p：显著性（无需检验）corfunc：有效自相关值 当前滞后（lag）
    else:
        r, p = stats.pearsonr(x[i:], x[:-i])
        if np.isnan(r):
            return 0,0,0,i
        if p > level_of_significance_for_pearson and consider_only_significant_correlation:
            return r,p,0,i
        else:
            return r,p,r,i

def autocor(x,list_of_lags,level_of_significance_for_pearson,consider_only_significant_correlation):
    #对输入信号 x 计算不同滞后量下的 Pearson 自相关系数、p 值，并根据显著性筛选后输出自相关曲线。
    #r_list，各滞后下的 Pearson 相关系数
    #p_list，对应的 p 值（显著性）
    #corfunc，筛选后得到的自相关函数曲线（p>0.01 的位置置 0）
    #lag_list，滞后值（lags）对应索引
    '''
    This function calculates the autocorrelation function for a time series given as vector x and the shifts given as indices in list_of_lags.
    The parameters consider_only_significant_correlation and level_of_significance_for_pearson decide if and which results are considered depending on their significance p.
    It returns the correlation coefficients as r_list, their significance as p_list, the autocorrelation function as func and the shift indices as lag_list
    :param x: list
    :param list_of_lags: list of positive integers
    :param level_of_significance_for_pearson: float between 0 and 1
    :param consider_only_significant_correlation: Boolean or interger in {0,1}
    :return: list of floats, list of floats, list of floats, list of positive integers
    '''
    list_of_results=list(zip(*list(map(functools.partial(calculate_autocorrelation,x,level_of_significance_for_pearson=level_of_significance_for_pearson,consider_only_significant_correlation=consider_only_significant_correlation),list_of_lags))))
    r_list=list(list_of_results[0])
    p_list=list(list_of_results[1])
    func=list(list_of_results[2])
    lag_list=list(list_of_results[3])
    return r_list, p_list, func, lag_list

def shift_diff(i, corfunc):
    '''
    This function calculates the L^1 norm of the difference of the original autocorrelation function and the shifted version. 
    The values of the autocorrelation function results are given in corfunc and the shift index is given as i.
    :param i: positive integer
    :param corfunc: list of floats between -1 and  1
    :return: positive float
    '''
    if i==0:
        return 0 
    else:
        w1 = np.array(corfunc)[i:]
        w2 = np.array(corfunc)[:-i]
        return float(sum(abs(w1 - w2)) / w1.size)

def fit_model(df_data_aggregated):
    '''
    This function uses the phase of a date in a suggested period as input in order to fit a model. 
    The model will later also test how well besaid period fits the original time series.
    Disclaimer: Here the sklearn.ensemble.RandomForestRegressor() is used, but any model can be used instead.
    :param df_data_aggregated: pd.DataFrame
    :return: list of 1-D lists of floats, RandomForestRegressor object
    '''
    # The routine to fit a model based on the periodic information/phase of a date concerning the period to the original time series to test the hypothesis how well the suggested period fits the original time series
    X = df_data_aggregated["date_modulo"].to_numpy().reshape(df_data_aggregated["date_modulo"].size, 1)
    y = df_data_aggregated["value"].to_numpy().reshape(df_data_aggregated["value"].size)
   
    mlp = RandomForestRegressor()

    mlp.fit(X, y)
    y_model = mlp.predict(X).reshape(X.size, 1)

    return y_model, mlp  #y_model模型预测的周期性信号，mlp拟合器对象，可用于后续预测新点

def get_relevant_diffs(diffs):
    '''
    This function detects the local minima in the differences diffs between the original autocorrelation function and its shifted versions.
    It returns a list of the peaks, their indices and a Boolean indicating whether further calculation can be aboarted do to a lack of local minima.
    :param diffs: list of positive floats
    :return: list of positive floats, list of positive integers, Boolean
    '''
    #任务是找出差异曲线（diffs）中的“局部最小值”位置（即可能的周期），并返回：
    #对应这些最小值的差异值（relevant_diffs）,它们的索引位置（peaks）,是否需要提前终止（stop_calculation）
    stop_calculation=0
    peaks, _ = find_peaks(-np.array(diffs))
    if peaks.size <= 0:
        print('No local minima in the inner domain of the function mapping the L^1 norm of the difference between the autocorrelation function and its shifted versions against the shifts! No period detection possible!')
        stop_calculation = 1
    # For shift=0, two identical functions are subtracted resulting in a local minimum at shift 0 in the function mapping the L^1 norm of the difference between the autocorrelation function and its shifted versions against the shifts.
    # Needs to be inserted separately since find_peaks searches only in the inne of the domain, consequently no boundary values are considered
    peaks = np.insert(peaks, 0, 0)
    peaks = peaks.astype(int)  #确保 peaks 的类型为整数（索引需要是 int 类型）。
    return np.array(diffs)[peaks],peaks,stop_calculation

def sum_shifted_function(i, corfunc):
    '''
    This function calculates the sum of the L^1 norm of the correlation function and its shifted version (given as corfunc) for a given shift i.
    This is used to see if their sum is already smaller than our criterion (for further detail see paper Algorithm 1 step 5))
    :param i: positive integer
    :param corfunc: list of floats between -1 and  1
    :return: positive float
    '''
    if i==0:
        return 2*sum(abs(np.array(corfunc))) / np.array(corfunc).size
    else:
        w1 = np.array(corfunc[i:])
        w2 = np.array(corfunc[:-i])
        return sum(abs(w1))/w1.size + sum(abs(w2))/w2.size
