import random
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift

class FunctionalPowerRuleSimulator():
    """
    A class used to represent the simulation of the detachment functional power rule

    Attributes
    ----------
    grid_size : int
        The number of ticks
    x_range : float
        The functions' domains is (-x_range,x_range)
    num_coeffs : int
        The functions f,g are polynomials of degree num_coefficients - 1
    num_simulations : int
        The number of the simulations that should be conducted

    Methods
    -------
    is_sign_continuous(ys):
        Returns a vector indicating whether the function whose values are ys is sign-continuous at each one of the points

    detachment(ys):
        Returns the vector of (right-) detachments for the function whose values are ys

    calc_func_vals(func):
        Calculates the function func to the precalculated vector xs

    simulate():
        The main function that conducts the simulations and prints a report on the accuracy and conditions per each of them
    """
    def __init__(self, grid_size=10000, x_range=10, num_coeffs=8, num_simulations=10):
        self.grid_size = grid_size
        self.x_range = x_range
        self.num_coeffs = num_coeffs
        self.num_simulations = num_simulations
        self.xs = np.linspace(-self.x_range, self.x_range, self.grid_size)
        self.simulation_df = pd.DataFrame(columns=['NumPoints', 'Condition1', 'Condition2', 'Discrepancies'])

    def is_sign_continuous(self, ys):
        return shift([np.sign(next_y) == np.sign(cur_y) for cur_y, next_y in zip(shift(ys, 1, cval=ys[0], order=0), ys)], -1, cval=np.nan, order=0)

    def detachment(self, ys):
        return shift([np.sign(next_y - cur_y) for cur_y, next_y in zip(shift(ys, 1, cval=ys[0], order=0), ys)], -1, cval=np.nan, order=0)

    def f(self, x):
        return np.dot(self.f_coeffs, [x**i for i in range(len(self.f_coeffs))])

    def g(self, x):
        return np.dot(self.g_coeffs, [x**i for i in range(len(self.g_coeffs))])

    def g_ln_f(self, x):
        return self.g(x) * np.log(self.f(x))

    def f_minus1(self, x):
        return self.f(x) - 1

    def f_minus1_g(self, x):
        return self.f_minus1(x) * self.g(x)

    def calc_func_vals(self, func):
        return list(map(func, self.xs))

    def simulate(self):
        for sim in range(self.num_simulations):
            print(f'Conducting simulation #{sim+1}/{self.num_simulations}')
            discrepancies = 0
            first_condition_met = 0
            second_condition_met = 0
            self.f_coeffs = random.sample(range(-10, 10), self.num_coeffs)
            self.g_coeffs = random.sample(range(-10, 10), self.num_coeffs)
            f_minus1_ys = self.calc_func_vals(self.f_minus1)
            g_ys = self.calc_func_vals(self.g)
            g_lnf_ys = self.calc_func_vals(self.g_ln_f)
            f_minus1_g_ys = self.calc_func_vals(self.f_minus1_g)
            f_detachments = self.detachment(f_minus1_ys)
            g_detachments = self.detachment(g_ys)
            g_lnf_detachments = self.detachment(g_lnf_ys)
            f_minus1_g_detachments = self.detachment(f_minus1_g_ys)
            g_sc = self.is_sign_continuous(g_ys)
            f_minus1_sc = self.is_sign_continuous(f_minus1_ys)

            # TODO: replace the following loop with vector operations
            for i, x in enumerate(self.xs):
                actual_detachment = g_lnf_detachments[i]
                if not np.isnan(actual_detachment):
                    product = g_ys[i] * g_detachments[i] * f_minus1_ys[i] * f_detachments[i]
                    first_condition = product >= 0 and (f_minus1_sc[i] or g_sc[i] or (f_minus1_ys[i] == 0 and g_ys[i] == 0))
                    second_condition = product < 0 and (not f_minus1_sc[i] or not g_sc[i])
                    if first_condition or second_condition:
                        if first_condition:
                            first_condition_met += 1
                        if second_condition:
                            second_condition_met += 1
                        if product >= 0 and (f_minus1_sc[i] or g_sc[i]):
                            expected_detachment = f_minus1_g_detachments[i]
                        else:
                            expected_detachment = f_detachments[i] * g_detachments[i]
                        if expected_detachment != actual_detachment:
                            discrepancies += 1
            self.simulation_df = self.simulation_df.append(pd.DataFrame({'NumPoints': [len(self.xs)], 'Condition1': [first_condition_met], 'Condition2': [second_condition_met], 'Discrepancies': [discrepancies]}))
        print(self.simulation_df)

if __name__ == "__main__":
    FunctionalPowerRuleSimulator().simulate()