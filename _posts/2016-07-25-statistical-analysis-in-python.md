---
title: "Statistical Analysis Tools in Python"
date: 2016-07-25 09:00:00
categories: Tutorial
tags: [Statistics, Python, NumPy, SciPy, Matplotlib]
---

In the world of data analysis it is often very useful to have a variety of tools at hand to make life a little more convenient. In this tutorial a few tools for statistical analysis will be constructed. One may find these tools useful during an undergraduate statistical analysis course.

## 2-Way ANOVA
In statistics, the two-way analysis of variance (ANOVA) is an extension of the one-way ANOVA that examines the influence of two different categorical independent variables on one continuous dependent variable.

One can construct a function to perform this analysis as follows:

```python
def ANOVA_2(IJ_List):
    '''
    Ouputs values of interest from factor
    ANOVA analysis with 1 observation per
    treatment.
    IJ_List is a (I,J) shaped numpy array:
    | x_i,j    x_i,j+1    ...  x_i,J   |
    | x_i+1,j  x_i+1,j+1  ...  x_i+1,J |
    | :        :          ...  :       |
    | x_I,j    x_I,j+1    ...  x_I,J   |
    '''
    IJ_List = IJ_List.astype(np.float64)
    I       = np.shape(IJ_List)[0]
    J       = np.shape(IJ_List)[1]
    dfi     = I - 1
    dfj     = J - 1
    dfe     = dfi * dfj
    dft     = I * J -1
    xibar   = [np.mean(IJ_List[i,:])
            for i in range(I)]
    xjbar   = [np.mean(IJ_List[:,j])
            for j in range(J)]
    xbar    = np.mean(IJ_List)
    SSA     = J * np.sum([(xibar[i] - xbar)**2
            for i in range(I)])
    SSB     = I * np.sum([(xjbar[j] - xbar)**2
            for j in range(J)])
    SSE     = np.sum([np.sum([
            (IJ_List[i,j] - xibar[i] - xjbar[j] + xbar)**2
            for j in range(J)])
            for i in range(I)])
    SST     = np.sum([np.sum([
            (IJ_List[i,j] - xbar)**2
            for j in range(J)])
            for i in range(I)])
    MSA     = SSA / dfi
    MSB     = SSB / dfj
    MSE     = SSE / dfe
    fA      = MSA / MSE
    fB      = MSB / MSE
    results = (dfi  , dfj  , dfe, dft,
               SSA  , SSB  , SSE, SST,
               MSA  , MSB  , MSE,
               fA   , fB   ,
               xibar, xjbar, xbar)
    return results
```

## 2-Way ANOVA with K number of observations
One can easily augment the previous generic 2-way ANOVA function. Only slight modification is required.

```python
def ANOVA_IJK(IJK_List):
    '''
    Outputs values of interst from a 2-factor
    ANOVA analysis with K number of observations
    per treatment.
    I       = Number of factor A levels
    J       = Number of factor B levels
    K       = Number of observations at level (i,j)
    IJK_List is a numpy array of shape (I,J,K)
    '''
    shape   = np.shape(IJK_List)
    I       = shape[0]
    J       = shape[1]
    K       = shape[2]
    dfi     = I - 1
    dfj     = J - 1
    dfij    = dfi * dfj
    dfe     = I * J * (K - 1)
    dft     = I * J * K - 1
    xibar   = np.array([np.mean(IJK_List[i,:,:])
            for i in range(I)])
    xjbar   = np.array([np.mean(IJK_List[:,j,:])
            for j in range(J)])
    xijbar  = np.asarray([[np.mean(IJK_List[i,j,:])
            for j in range(J)]
            for i in range(I)])
    xbar    = np.mean(IJK_List)
    SSA     = np.sum([np.sum([np.sum([
            (xibar[i] - xbar)**2
            for k in range(K)])
            for j in range(J)])
            for i in range(I)])
    SSB     = np.sum([np.sum([np.sum([
            (xjbar[j] - xbar)**2
            for k in range(K)])
            for j in range(J)])
            for i in range(I)])
    SSAB    = np.sum([np.sum([np.sum([
            (xijbar[i,j] - xibar[i] - xjbar[j] + xbar)**2
            for k in range(K)])
            for j in range(J)])
            for i in range(I)])
    SSE     = np.sum([np.sum([np.sum([
            (IJK_List[i,j,k] - xijbar[i,j])**2
            for k in range(K)])
            for j in range(J)])
            for i in range(I)])
    SST     = np.sum([np.sum([np.sum([
            (IJK_List[i,j,k] - xbar)**2
            for k in range(K)])
            for j in range(J)])
            for i in range(I)])
    MSA     =  SSA  / float(dfi)
    MSB     =  SSB  / float(dfj)
    MSAB    =  SSAB / float(dfij)
    MSE     =  SSE  / float(dfe)
    fA      =  MSA  / MSE
    fB      =  MSB  / MSE
    fAB     =  MSAB / MSE
    results = (dfi  , dfj  , dfij,  dfe,  dft,
               SSA  , SSB  , SSAB,  SSE,  SST,
               MSA  , MSB  , MSAB,  MSE,
               fA   , fB   , fAB ,
               xibar, xjbar, xbar)
    return results
```

## Linear Regression
In statistics, linear regression is an approach for modeling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X. The case of one explanatory variable is called simple linear regression.

A function returning the primary values of interest in the context of a linear regression analysis can be constructed using 'scipy.stats' as follows:

```python
from scipy import stats

def Linear_Regression(x_list, y_list, verbose = False):
    '''
    x_list and y_list are cooresponding arrays
    of 1 dimensional shape and common length N.
    x_list = [x_n, x_n+1, ..., x_N]
    y_list = [y_n, y_n+1, ..., y_N]
    '''
    (slope , intercept,
    r_value, p_value,
    std_err) = stats.linregress(x_list, y_list)
    cor_coef = stats.pearsonr(x_list, y_list)[0]
    if verbose == False:
        return slope, intercept, r_value, cor_coef
    else:
        print('y = ' + str(intercept) + ' + ' + str(slope) + 'x')
```

It may be easily plotted with the use of 'matplotlib.pyplot':

```python
import matplotlib.pyplot as plt

def Linear_Regression_Plot(x_list, y_list):
    plt.plot(x_list, y_list, 'o')
    (slope, intercept,
    r_value, cor_coef) = Linear_Regression(x_list, y_list)
    x = np.array([min(x_list), max(x_list)])
    y = intercept + slope * x
    plt.plot(x, y)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$y=%f+%fx$\n$R^2=%f$ | $Correlation=%f$'
               %(intercept, slope, r_value**2, cor_coef))
    plt.show()
```

With intuitively named functions, the linear regression analysis can be plotted for a given set of data:

```python
x_list = np.array([46  , 48  , 55  , 57  , 60  , 72  , 81  , 85  , 94  ,
                   109 , 121 , 132 , 137 , 148 , 149 , 184 , 185 , 187 ])
y_list = np.array([2.18, 2.10, 2.13, 2.28, 2.34, 2.53, 2.28, 2.62, 2.63,
                   2.50, 2.66, 2.79, 2.80, 3.01, 2.98, 3.34, 3.49, 3.26])

Linear_Regression_Plot(x_list, y_list)
```

<img src="https://cdn.rawgit.com/CISprague/Statistical_Analysis/master/figure_1.png" >

## Chi-Squared 2x2 Contingency Statistic (Advanced)
A chi square statistic is used to investigate whether distributions of categorical variables differ from one another. A test of goodness of fit establishes whether or not an observed frequency distribution differs from a theoretical distribution. A test of independence assesses whether unpaired observations on two variables, expressed in a contingency table, are independent of each other (e.g. polling responses from people of different nationalities to see if one's nationality is related to the response).

We can compute this statistic in the following way:

```python
def Two_Way_Contingency_Chi2_Stat(two_way_table):
    '''
    Tests whether the null hypothesis of homogeneity
    is true.
    two_way_table is a numpy array of shape (I,J):
    [n_i,j   n_i,j+1   ... n_i,J   ]
    [n_i+1,j n_i+1,j+1 ... n_i+1, J]
    [:       :         ... :       ]
    [n_I,j   n_I,j+1   ... n_I,J   ]
    '''
    nij  = two_way_table.astype(np.float64)
    I, J = np.shape(nij)
    ni   = [np.sum(nij[i,:]) for i in range(I)]
    nj   = [np.sum(nij[:,j]) for j in range(J)]
    n    = np.sum(nij)
    eij  = np.array([[(ni[i] * nj[j]) / n
        for j in range(J)]
        for i in range(I)])
    chi2 = np.sum([np.sum([
        (nij[i,j] - eij[i,j])**2 / eij[i,j]
        for j in range(J)])
        for i in range(I)])
    return chi2
```

## Wilcoxon Test Statistic
The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test used when comparing two related samples, matched samples, or repeated measurements on a single sample to assess whether their population mean ranks differ (i.e. it is a paired difference test). It can be used as an alternative to the paired Student's t-test, t-test for matched pairs, or the t-test for dependent samples when the population cannot be assumed to be normally distributed.

One can compute this statistic as follows:

```python
def Wilcoxon_Test_Stat(data, mu0):
    '''
    Returns the Wilcoxon test statistic,
    the sum of ranks associated with
    positive (x_i - mu0), where x_i is
    an individual data item and mu0 is
    the hypothesised true average.
    '''
    data       = data - mu0
    abs_mag    = np.sort(np.absolute(data))
    data       = np.ndarray.tolist(data)
    abs_mag    = np.ndarray.tolist(abs_mag)
    s          = 0
    for d in data:
        if d  >= 0.0:
            s += abs_mag.index(d) + 1
    return s
```

## Want to see the whole package?
These statistical analysis tools proved to be useful during my statistics course. These functions in their entirety can be seen [here](https://github.com/CISprague/Statistical_Analysis/blob/master/Functions.py)
