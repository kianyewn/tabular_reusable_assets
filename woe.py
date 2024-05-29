from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

class WOE:

    """Weight of Evidence transformer to preprocess data
    
    Args:
        missing_values: list of values that we want to consider as missing values, eg ['empty']
        categorical_features: list of columns to consider as categorical features
        numerical_features: list of columns to consider as numerical features
    """
    def __init__(self,
                 missing_values: List[str],
                 categorical_features: List[str], 
                 numerical_features: List[str],
                 label: str = 'label',
                 nbins=5,
                **kwargs):
        
        self.missing_values = missing_values
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.label = label
        self.nbins = nbins
        self.min_bin_rate = kwargs.get('min_bin_rate', 0.02)
        self.min_bin_size = kwargs.get('min_bin_size', 50)
        self.min_bin_adjusted_rate = self.min_bin_rate
        
    def is_missing_value(self, x: Any) -> bool:
        """Checks if the single value is a missing value"""
        if pd.isna(x) or x in self.missing_values:
            return True
        else:
            return False
    @classmethod
    def test(self):
        return 'hello'
    def split_data_by_missing(self, df: pd.DataFrame, c:str) -> Tuple[pd.DataFrame]:
        """Splits a dataframe into two dataframe. 
        
        - df_missing contain a subset of the original dataframe such that the value in c is missing
        - df_data contains the other subset of the original dataframe such that the value in c is not missing
        """
        df_missing = df.loc[df[c].apply(lambda x: self.is_missing_value(x))]
        df_data = df.loc[df[c].apply(lambda x: not self.is_missing_value(x))]
        
        if c in self.categorical_features:
            df_missing.loc[:,c] = df_missing.loc[:, c].fillna('NA')
        else:
            df_missing.loc[:, c] = df_missing.loc[:, c].fillna('990000')
            
        return df_missing, df_data
    
    def get_bin_stats(self, df: pd.DataFrame, c:str, label:str) -> pd.DataFrame:
        """Get the prior probability of the labels according to the buckets in the column
        """
        # it is possible that there are no values in one of the labels
        stat = df.groupby(c)[label].value_counts().unstack().reset_index()
        stat = stat.sort_values(by=c)
        stat = stat.rename(columns={0:'good', 1:'bad', c:'bin'}).fillna(0)
        stat['var'] = c
        stat.columns.name = None
        if 'good' not in stat.columns:
            stat['good'] = 0
        if 'bad' not in stat.columns:
            stat['bad'] = 0
        return stat
    
    def rebin_cat_stat_missing(self, stat_missing, total_good_data, total_bad_data):
        """Rebin stat missing according to whether or not len(stat_missing) > 0"""
        
        if len(stat_missing) > 0:
            # this is no longer required because we got the total_good and bad_data here
#             good_missing = stat_missing['good'].sum()
#             bad_missing = stat_missing['bad'].sum()

            bin_missing = self.rebin_stat(stat_missing,
                                          stat_missing.index,
                                          'cat_missing',
                                          total_good_data, 
                                          total_bad_data)
        else:
            # if there are no missing values, create an empty dataframe with the required dataframe
            # ValueError: If using all scalar values, you must pass an index. 
            # wrap dictionary with list: https://stackoverflow.com/questions/17839973/constructing-pandas-dataframe-from-values-in-variables-gives-valueerror-if-usi
            bin_missing = pd.DataFrame([{'var':stat_missing['var'][0], 
                                       'type': 'cat_missing',
                                       'bin': 'NA',
                                       'min': 'NA',
                                       'max': 'NA',
                                       'woe': -np.inf,
                                       'iv':0,
                                       'total':0,
                                       'ratio':0,
                                       'bad':0,
                                       'bad_rate':0}])
        return bin_missing
    
    def rebin_cat_stat_data(self, stat_data, total_good_data, total_bad_data):
        """ rebin normal categorical data
        """
        bin_data = self.rebin_stat(stat_data, stat_data.index, 'cat_normal', total_good_data, total_bad_data)
        return bin_data
    
    def rebin_num_stat_missing(self, stat_missing, total_good_data, total_bad_data):
        if len(stat_missing) > 0:
            bin_missing = self.rebin_stat(stat_missing,
                                          stat_missing.index,
                                          'num_missing',
                                          total_good_data,
                                          total_bad_data)
        else:
            bin_missing = pd.DataFrame([{'var':stat_missing['var'][0],
                                       'type':'num_missing',
                                       'bin': -99000, # keep numerical
                                       'min': -99000,
                                       'max': -99000,
                                       'woe': -np.inf,
                                       'iv':0,
                                       'total':0,
                                       'ratio':0,
                                       'bad':0,
                                       'bad_rate':0}])
        return bin_missing
    
    def rebin_num_stat_data(self, stat_data, knots, total_good_data, total_bad_data):
        """ rebin normal numerical data
        """
        bin_data = self.rebin_stat(stat_data, knots, 'num_normal', total_good_data, total_bad_data)
        return bin_data
    
    def find_index_helper(self, lst_val, target):
#         return [lst[i] for i in range(len(lst)) if lst[i] == val]
        return list(filter(lambda v: lst_val[v] == target, range(0, len(lst_val))))
    
    def get_best_knot_by_ks(self, stat_bin, total, left, right):
#         stat = stat_bin.loc[left:right]
#         cum_good = stat['good'].cumsum() / stat['good'].sum()
#         cum_bad = stat['bad'].cumsum() / stat['bad'].sum()
#         ks = abs(cum_good - cum_bad)
#         max_ks_index = self.find_index_helper(ks, max(val))
        
        stat = stat_bin.loc[left:right]
        total_cur = stat['good'].sum() + stat['bad'].sum()
        left_add = sum(np.cumsum(stat['good'] + stat['bad']) < self.min_bin_adjusted_rate * total)
        right_add = sum(np.cumsum(stat['good'] + stat['bad']) <= total_cur - self.min_bin_adjusted_rate * total)
    
        left_adjust = left + left_add
        right_adjust = left + right_add - 1 # similar to len(lst) - 1
        
        if right_adjust >= left_adjust:
            if (stat['bad'].sum() != 0) and (stat['good'].sum() != 0):
                cdf_bad = np.cumsum(stat['bad']) / stat['bad'].sum()
                cdf_good = np.cumsum(stat['good']) / stat['good'].sum()
                # note that this cumulative sum is before the left and right adjusted
                # this is to make sure that the first knot can have minimum of (self.min_bin_adjusted_rate * total) samples
                # if we do not cumsum before the left and right adjusted, we are making the minimum samples in the bin larger
                cdf_abs_diff = abs(cdf_bad - cdf_good)
                ks = max(cdf_abs_diff.loc[left_adjust: right_adjust])
                idx = self.find_index_helper(list(cdf_abs_diff), ks)
                return stat.index[max(idx)]
            else:
                return None
        else:
            return None
        
    def get_best_knots_helper(self, stat_bin, total, max_iter, left, right, cur_iter):
        # termination condition
        stat = stat_bin.loc[left:right]
        total_cur = stat['good'].sum() + stat['bad'].sum()
        if total_cur < self.min_bin_adjusted_rate * total * 2 or cur_iter >= max_iter:
            return []
        
        best_knot = self.get_best_knot_by_ks(stat_bin, total, left, right)
        
        if best_knot is not None:
            left_knots = self.get_best_knots_helper(stat_bin, total, max_iter, left, best_knot, cur_iter+1)
            right_knots = self.get_best_knots_helper(stat_bin, total, max_iter, best_knot+1, right, cur_iter+1)
        else:
            left_knots = []
            right_knots = []
        return left_knots + [best_knot] + right_knots
        
    def init_knots(self, stat_bin, total, max_iter):
        """Initialize a new set of knots"""
        knots = self.get_best_knots_helper(stat_bin=stat_bin, 
                                           total=total, 
                                           max_iter=max_iter, 
                                           left=0, 
                                           right=len(stat_bin), 
                                           cur_iter=0)
        
        knots = list(filter(lambda x: x is not None, knots))
        knots.sort()
        return knots
    
    def eval_iv_mono(self, stat, knots):
        lst_df = []
        for i in range(1, len(knots)):
            if i == 1:
                lst_df.append(stat.loc[knots[i-1]: knots[i]])
            else:
                lst_df.append(stat.loc[knots[i-1]+1: knots[i]])
                
        total_good = stat['good'].sum()
        total_bad = stat['bad'].sum()
        
        ratio_good = pd.Series(list(map(lambda x: float(sum(x['good'])) / float(total_good), lst_df)))
        ratio_bad = pd.Series(list(map(lambda x: float(sum(x['bad'])) / float(total_good), lst_df)))
        
        # monotonous property
        lst_woe = list(np.log(ratio_good / ratio_bad))
        if sorted(lst_woe) != lst_woe and sorted(lst_woe, reverse=True) != lst_woe:
            return None
        lst_iv = (ratio_good - ratio_bad) * np.log(ratio_good / ratio_bad)
        if np.inf in list(lst_iv) or -np.inf in list(lst_iv):
            return None
        else:
            return sum(lst_iv)
        
    def combine_helper(self, stat, nbins, knots):
        from itertools import combinations
        lst_knots = list(combinations(knots, nbins-1))
        
        knots_list = list(map(lambda x: sorted(x + (0, len(stat)-1)), lst_knots))
        lst_iv = list(map(lambda x: self.eval_iv_mono(stat,x), knots_list))
        
        lst_iv_filtered = list(filter(lambda x: x is not None, lst_iv))
        if len(lst_iv_filtered) == 0:
            return None
        else:
            if len(self.find_index_helper(lst_iv, max(lst_iv_filtered))) > 0:
                target_index = self.find_index_helper(lst_iv, max(lst_iv_filtered))[0]
                return knots_list[target_index]
            else:
                return None
        
    def combine_bins(self, stat, max_nbins, knots):
        max_nbins = min(max_nbins, len(knots)+1)
        if max_nbins == 1:
            return [0, len(stat) - 1]
        for cur_nbins in sorted(range(2, max_nbins+1), reverse=True):
            new_knots = self.combine_helper(stat, cur_nbins, knots)
            if new_knots is not None:
                return new_knots
        print("No available bins with mono constrain")
        return [0, len(stat) - 1]
        
    def combine_stats(self, stat_data: pd.DataFrame, stat_missing: pd.DataFrame, c:str) -> pd.DataFrame:
        """stat_bin_info proxy
        
        output: combined stats from all the numerical features, categorical features and missing values
        Returns:
            (pd.DataFrame) Columns include
                - var: var name
                - type: type of data
                - bin: the bin corresponding to the value, if categorical then just the value
                - min: minimum value in the bin, if categorical then just the value
                - max: maximum value in the bin, if categorical then just the value
                - woe: weight of evidence for the bin
                - iv: information value for the bin
                - total: total number of examples in the bin
                - ratio: ratio of the number of examples in the bin to the total number of examples
                - bad: number of `bad` data in the bin
                - bad_rate: percentage of `bad` data across the total number of examples
        """
        
        good_data = stat_data['good'].sum()
        bad_data = stat_data['bad'].sum()
        
        total_good_data = stat_data['good'].sum() + stat_missing['good'].sum()
        total_bad_data = stat_data['bad'].sum() + stat_missing['bad'].sum()
        
        if c in self.categorical_features:
            # get new bins for stat_missing
            bin_missing = self.rebin_cat_stat_missing(stat_missing, total_good_data, total_bad_data)
            # get new bins for stat_data
            bin_data = self.rebin_cat_stat_data(stat_data, total_good_data, total_bad_data)
        
        else:
            bin_missing = self.rebin_num_stat_missing(stat_missing, total_good_data, total_bad_data)
            # for numerical stat_data, we want to rebin according to constraint optimization
            total = total_good_data + total_bad_data
            knots = self.init_knots(stat_data, total, self.nbins)
            knots = self.combine_bins(stat_data, self.nbins, knots)
            print(f'new knots: {knots}')
            bin_data = self.rebin_num_stat_data(stat_data, knots, total_good_data, total_bad_data)
            
        bin_total = pd.concat([bin_data, bin_missing], axis=0).reset_index(drop=True)
        
        woe_max = bin_data['woe'].max()
        woe_min = bin_data['woe'].min()
        
        bin_total['bin'] = bin_total['bin'].apply(lambda x: str(x))
        for idx, row in bin_total.iterrows():
            if row['type'] == 'num_normal':
                v = f"{idx:02}.{row['bin']}"
            else:
                v = f"{idx:02}.{{{row['bin']}}}"
            bin_total.at[idx, 'bin'] = v
        bin_total['woe_max'] = woe_max
        bin_total['woe_min'] = woe_min
        return bin_total
        
    def rebin_stat(self,
                   stat: pd.DataFrame, 
                   knots: List,
                   bin_type:str,
                   total_good_data:int,
                   total_bad_data:int):
        """Performing binning on the stat data"""
        var = stat['var'][0]
#         ## this is no longer required because we got the total_good and bad_data here
#         total_good = stat['good'].sum() + good
#         total_bad = stat['bad'].sum() + bad
        
        lst_df, lst_bin, lst_min, lst_max = list(), list(), list(), list()
        if bin_type in ['cat_normal', 'cat_missing', 'num_missing']:
            for i in knots:
                lst_df.append(stat.loc[i:i]) # slicing to return pd.DataFrame
                lst_bin.append(stat.loc[i]['bin']) # no slicing to get raw values from pd.Series()
                lst_min.append(stat.loc[i]['bin'])
                lst_max.append(stat.loc[i]['bin'])
        else:
            # numerical
            if len(knots) == 2:
                lst_df.append(stat.loc[knots[0]: knots[1]])
                lst_bin.append(pd.Interval(left=-np.inf, right=np.inf))
                lst_min.append(-np.inf)
                lst_max.append(np.inf)
            else:
                for i in range(1, len(knots)):
                    if i == 1:
                        lst_df.append(stat.loc[knots[i-1]:knots[i]])
                        val_right = float(stat['bin'].loc[knots[i]])
                        lst_bin.append(pd.Interval(left=-np.inf, right = val_right))
                        lst_min.append(-np.inf)
                        lst_max.append(val_right)
                    else:
                        lst_df.append(stat.loc[knots[i-1]+1: knots[i]])
                        if i == len(knots)-1:
                            val_left = float(stat['bin'].loc[knots[i-1]])
                            lst_bin.append(pd.Interval(left= val_left, right = np.inf))
                            lst_min.append(val_left)
                            lst_max.append(val_right)
                        else:
                            val_left = float(stat['bin'].loc[knots[i-1]])
                            val_right = float(stat['bin'].loc[knots[i]])
                            lst_bin.append(pd.Interval(left=val_left, right=val_right))
                            lst_min.append(val_left)
                            lst_max.append(val_right)
        df_bin = self.get_new_bin_stat(var, bin_type, lst_df, lst_bin, lst_min, lst_max, total_good_data, total_bad_data) 
        return df_bin
    
    def get_new_bin_stat(self, var, bin_type, lst_df, lst_bin, lst_min, lst_max, total_good, total_bad):
        # get ratio of good data in the bin to total good data
        ratio_good = pd.Series(list(map(lambda x: float(x['good'].sum() + 0.5) / float(total_good + 0.5), lst_df)))
        # get ratio of bad data in the bin to total bad data
        ratio_bad = pd.Series(list(map(lambda x: float(x['bad'].sum() + 0.5) / float(total_bad + 0.5), lst_df)))
        # get total number of examples in the bin
        lst_total = list(map(lambda x: x['good'].sum() + x['bad'].sum(), lst_df))
        # ratio of the examples in the bin to the total number of examples
        lst_ratio = list(map(lambda x: x / (total_good + total_bad), lst_total))
        # number of bad data in the bin
        lst_bad = list(map(lambda x: x['bad'].sum(), lst_df))
        # percentage of bad data in the bin over total number of examples in the bin
        lst_rate = list(map(lambda x: x['bad'].sum() / (x['bad'].sum() + x['good'].sum()), lst_df))
        # get the woe for the bin
        lst_woe = list(np.log(ratio_good / ratio_bad))
        # get the iv for the bin
        lst_iv = list((ratio_good - ratio_bad) * lst_woe)
        
        df_bin = pd.DataFrame({'var':var,
                              'type': bin_type,
                              'bin': lst_bin,
                              'min': lst_min,
                              'max':lst_max,
                              'woe':lst_woe,
                              'iv':lst_iv,
                              'total': lst_total,
                              'ratio': lst_ratio,
                              'bad': lst_bad,
                              'rate': lst_rate})
        return df_bin
        
        
    def fit_numerical(self, df, c):
        """Obtain the trend statistics for column `c`
        """
        df_missing, df_data = self.split_data_by_missing(df, c)
        df_data.loc[:, c] = df_data.loc[:, c].apply(lambda x: round(x,4))
        stat_missing = self.get_bin_stats(df_missing, c, self.label)
        stat_data = self.get_bin_stats(df_data, c, self.label)
        
        df_bin = self.combine_stats(stat_data=stat_data, stat_missing=stat_missing, c=c)
        return df_bin
        

    def fit_categorical(self, df, c):
        """
        

        Args:
            df (pd.DataFrame): Original pandas dataframe.
            c (str): name of the col
        """
        
        df_missing, df_data = self.split_data_by_missing(df, c)
        
        stat_missing = self.get_bin_stats(df_missing, c, self.label)
        stat_data = self.get_bin_stats(df_data, c, self.label)
        
        df_bin = self.combine_stats(stat_data=stat_data, stat_missing=stat_missing, c=c)
        return df_bin