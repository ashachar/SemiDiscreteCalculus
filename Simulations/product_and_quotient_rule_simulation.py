import itertools
import numpy as np
import pandas as pd

class DetachmentBasedRuleSimulator:
    """
    An abstract class used to represent the simulation of a detachment based rule

    Attributes
    ----------
    default_val : float
        The number with respect to whom vlues are calculated in each iteration of the simulation
    delta : float
        The gap between two consecutive numbers in the simulation
    rule_name : str
        The name of the rule at stake, e.g. 'product'
    operator_str : str
        A representation the rule for the target field name, e.g. '*'
    operator_exponent : int
        An indication of the exponent of the function g in the rule, e.g. +1
    should_include_zero_in_g: boolean
        Should the value 0 be part of the simulation for the function g

    Methods
    -------
    get_detachment(neighborhood_val, static_val)
        Calculates the numeric detachment of a function f at a point x, where f(x)=static_val
        and there exists a neighborhood of x where the values of f are neighborhood val
    get_function_inherent_sign_continuity(detachment, sign)
        Determines whether the sign of a function whose detachment and sign at a point are given,
        when considered as a function, is continuous there just by the given assumptions
    get_neighboring_function_vals(val, detachment, isc, should_include_zero)
        Simulates a function's neighboring values given its value at a point, the detachment,
        the inherent sign continuity property, and whether zero should be included
    apply_operator_to_values(val1, val2)
        Calculate the result of the operator applied to the values of f and g
    calculate_operator_detachment(f_detachment, g_detachment, f_sign, g_sign, f_isc, g_isc)
        Calculates the operator detachments assuming different combinations of sign continuities
        of f and g, while filtering only the legit ones given the inherent sign continuity of each function
    add_match_fields():
        Adding fields that check, for each of the engineered features, whether the feature's value is
        in accordance with the target field
    add_conditions():
        Calculates the conditions for each of the sub-formulas of the rule
    simulate():
        The main function that conducts the rule simulation and saves the results
    engineer_features():
        Add combinations of the original fields that aid in building the rules' formulas and conditions
    get_target_feature_name():
        Getting the column name of the target feature that the rule attempts to predict
    """
    def __init__(self, rule_name, operator_str, operator_exponent, should_include_zero_in_g):
        self.default_val = 10
        self.delta = self.default_val/3
        self.rule_name = rule_name
        self.operator_str = operator_str
        self.operator_exponent = operator_exponent
        self.should_include_zero_in_g = should_include_zero_in_g

    def get_detachment(self, neighborhood_val, static_val):
        if neighborhood_val > static_val:
            return +1
        elif neighborhood_val < static_val:
            return -1
        else:
            return 0

    def get_function_inherent_sign_continuity(self, detachment, sign):
        if detachment * sign == +1 or detachment == 0:
            return 1
        elif detachment != 0 and sign == 0:
            return -1
        else:
            return 0

    def get_neighboring_function_vals(self, val, detachment, isc, should_include_zero=True):
        vals = {'1': [], '-1': []}
        if isc >= 0:
            vals['1'] = [val + detachment * self.delta, val + 2 * detachment * self.delta]
        if isc <= 0:
            if val != 0:
                vals['-1'] = [-val, -(val + self.delta)]
                if should_include_zero:
                    vals['-1'].append(0)
            else:
                vals['-1'] = [self.default_val * detachment, (self.default_val + self.delta)*detachment]
        return vals

    def apply_operator_to_values(self, val1, val2):
        return val1 * (val2 ** self.operator_exponent)

    def calculate_operator_detachment(self, f_detachment, g_detachment, f_sign, g_sign, f_isc, g_isc):
        f_val = self.default_val * f_sign
        g_val = self.default_val * g_sign
        f_bar_vals = self.get_neighboring_function_vals(f_val, f_detachment, f_isc)
        g_bar_vals = self.get_neighboring_function_vals(g_val, g_detachment, g_isc, should_include_zero=self.should_include_zero_in_g)
        raw_fg_val = self.apply_operator_to_values(f_val, g_val)
        for f_sign_continuity in ['1','-1']:
            for g_sign_continuity in ['1','-1']:
                cur_f_bar_vals = f_bar_vals[f_sign_continuity]
                cur_g_bar_vals = g_bar_vals[g_sign_continuity]
                permutations = list(itertools.product(cur_f_bar_vals, cur_g_bar_vals))
                attempts = [self.get_detachment(self.apply_operator_to_values(cur_f_val, cur_g_val), raw_fg_val) for (cur_f_val, cur_g_val) in permutations]
                detachment_val = attempts[0] if (len(attempts) and attempts.count(attempts[0]) == len(attempts)) else None
                if (f_isc * int(f_sign_continuity) == -1 or g_isc * int(g_sign_continuity) == -1):
                    continue  # Impossible combination
                self.df = self.df.append(pd.DataFrame({'f;': [f_detachment], 'g;': [g_detachment], 'sgn(f)': [f_sign], 'sgn(g)': [g_sign], 'f_isc': [f_isc], 'g_isc': [g_isc], 'f_sc': [int(f_sign_continuity)], 'g_sc': [int(g_sign_continuity)], self.get_target_feature_name(): detachment_val}))

    def add_match_fields(self):
        for col in self.features.keys():
            self.df[f'{col}_Match'] = (self.df[col] == self.df[self.get_target_feature_name()]).astype(int)

    def add_conditions(self):
        pass # Stub

    def simulate(self):
        self.df = pd.DataFrame(columns=['f;', 'g;', 'sgn(f)', 'sgn(g)', 'f_sc', 'g_sc', 'f_isc', 'g_isc', self.get_target_feature_name()])
        signs_indicators = [-1, 0, +1]
        for i, (f_detachment, g_detachment, f_sign, g_sign) in enumerate(itertools.product(signs_indicators, signs_indicators, signs_indicators, signs_indicators if self.should_include_zero_in_g else list(set(signs_indicators).difference([0])))):
            f_isc = self.get_function_inherent_sign_continuity(f_detachment, f_sign)
            g_isc = self.get_function_inherent_sign_continuity(g_detachment, g_sign)
            self.calculate_operator_detachment(f_detachment, g_detachment, f_sign, g_sign, f_isc, g_isc)
        self.engineer_features()
        self.add_conditions()
        self.add_match_fields()
        self.df.to_csv(f'{self.rule_name}_rule_simulation_results.csv', index=False)
        return self.df

    def engineer_features(self):
        self.features = {
            'sgn-sgn': np.sign(self.df['f;'] * self.df['sgn(g)'] - self.df['g;'] * self.df['sgn(f)']),
            'sgn+sgn': np.sign(self.df['f;'] * self.df['sgn(g)'] + self.df['g;'] * self.df['sgn(f)']),
            'f;g;': self.df['f;'] * self.df['g;'],
            'f;g;sgn(fg)': self.df['f;'] * self.df['g;'] * self.df['sgn(f)'] * self.df['sgn(g)'],
        }
        for feature_name, feature in self.features.items():
            self.df[feature_name] = feature

    def get_target_feature_name(self):
        return f'(f{self.operator_str}g);'

class ProductRuleSimulator(DetachmentBasedRuleSimulator):
    def __init__(self):
        super().__init__(rule_name='product', operator_str='*', operator_exponent=+1, should_include_zero_in_g=True)

    def add_conditions(self):
        self.df['Overall_Condition1'] = ((self.df['f;g;sgn(fg)'] >= 0) & (((self.df['f_sc'] == 1) | (self.df['g_sc'] == 1)) | ((self.df['sgn(f)']==0) & (self.df['sgn(g)']==0)))).astype(int)
        self.df['Overall_Condition2'] = ((self.df['f;g;sgn(fg)'] < 0) & (((self.df['f_sc'] == -1) | (self.df['g_sc'] == -1)))).astype(int)
        self.df['Formula1_Condition'] = ((self.df['f;g;sgn(fg)'] >= 0) & ((self.df['f_sc'] == 1) | (self.df['g_sc'] == 1))).astype(int)

class QuotientRuleSimulator(DetachmentBasedRuleSimulator):
    def __init__(self):
        super().__init__(rule_name='quotient', operator_str='/', operator_exponent=-1, should_include_zero_in_g=False)

    def add_conditions(self):
        self.df['Overall_Condition1'] = (((self.df['f;g;sgn(fg)'] <= 0) & (((self.df['f_sc'] == 1) | (self.df['g_sc'] == 1)) | (self.df['sgn(f)'] == 0))).astype(int))
        self.df['Overall_Condition2'] = ((self.df['f;g;sgn(fg)'] > 0) & (((self.df['f_sc'] == -1) ^ (self.df['g_sc'] == -1)))).astype(int)
        self.df['Formula1_Condition'] = ((self.df['g_isc'] > 0) | ((self.df['f_sc'] > 0) & (self.df['g_sc'] > 0))).astype(int)
        self.df['Formula2_Condition'] = ((self.df['Formula1_Condition'] == 0) & ((self.df['f;g;sgn(fg)'] >= 0) & ((self.df['f_sc'] == 1) | (self.df['g_sc'] == 1)))).astype(int)

if __name__ == "__main__":
    ProductRuleSimulator().simulate()
    QuotientRuleSimulator().simulate()