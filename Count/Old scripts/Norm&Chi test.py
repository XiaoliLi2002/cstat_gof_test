import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import statistics
from scipy.stats import poisson
from scipy.stats import chi2
import pylatex
import scipy.optimize as opt
import scipy
np.set_printoptions(threshold=np.inf)

def p_value_norm(x,mu,sigma):
    z=abs(x-mu)/sigma
    return 2*min(scipy.stats.norm.sf(abs(z)),1-scipy.stats.norm.sf(abs(z)))

def p_value_chi(x,df):
    return 2*min(scipy.stats.chi2.sf(abs(x),df),1-scipy.stats.chi2.sf(abs(x),df))

print(p_value_norm(4.109309848,9.724845646,4.473195924))
print(p_value_chi(103.160783030592,98))
