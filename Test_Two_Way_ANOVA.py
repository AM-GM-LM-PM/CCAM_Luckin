# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:29:14 2019

@author: Administrator
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats

datafile = "ToothGrowth.csv"
data = pd.read_csv(datafile)

fig = interaction_plot(data.dose, data.supp, data.len, colors = ['red', 'blue'], markers = ['D', '^'], ms = 10)
# x, category, y

N = len(data.len)
df_a = len(data.supp.unique()) - 1
df_b = len(data.dose.unique()) - 1
df_axb = df_a * df_b
df_w = N - (df_a+1)*(df_b+1)

grand_mean = data['len'].mean()
ssq_a = sum([(data[data.supp == l].len.mean() - grand_mean)**2 for l in data.supp])
ssq_b = sum([(data[data.dose == l].len.mean() - grand_mean)**2 for l in data.dose])
ssq_t = sum((data.len - grand_mean)**2)

vc = data[data.supp == 'VC']
oj = data[data.supp == 'OJ']
vc_dose_means = [vc[vc.dose == d].len.mean() for d in vc.dose]
oj_dose_means = [oj[oj.dose == d].len.mean() for d in oj.dose]
ssq_w = sum((oj.len - oj_dose_means)**2) + sum((vc.len - vc_dose_means)**2)
ssq_axb = ssq_t - ssq_a - ssq_b - ssq_w

### Mean Square
ms_a = ssq_a / df_a
ms_b = ssq_b / df_b
ms_axb = ssq_axb / df_axb
ms_w = ssq_w / df_w

### F-ratio
f_a = ms_a / ms_w
f_b = ms_b / ms_w
f_axb = ms_axb / ms_w

p_a = stats.f.sf(f_a, df_a, df_w)
p_b = stats.f.sf(f_b, df_b, df_w)
p_axb = stats.f.sf(f_axb, df_axb, df_w)

formula = 'len ~ C(supp) + C(dose) + C(supp):C(dose)'
model = ols(formula, data).fit()
aov_table = anova_lm(model, typ=2)
# eta_squared(aov_table)
# omega_squared(aov_table)
print(aov_table.round(4))

## QQplot
res = model.resid
fig = sm.qqplot(res, line='s')
plt.show()