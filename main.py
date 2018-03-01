
# coding: utf-8

# # Project 1

# ### Import data

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import math
import numpy as np # importing this way allows us to refer to numpy as np

df = pd.read_excel('DataSet/university data.xlsx', encoding = 'utf8').drop([49])


# In[2]:

# In[3]:

def to_precision(x,p):
    """
    Code credit for this function: http://randlet.com/blog/python-significant-figures-format/
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)


# #Q1 Compute for each variable ((CS Score, Research Overhead, Admin Base Pay, Tuition)) its sample mean, variance and standard deviation. Related variables: mu1, mu2, mu3, mu4, var1, var2, var3, var4, sigma1, sigma2, sigma3, sigma4 

# In[4]:

def sampleMean(stat):
    n = 0
    total = 0
    for i in range (0,len(df.index)):
        n += 1
        total += df.at[i,stat]
    sampleMean = total / n
    return sampleMean
def sampleMeanNumpy(stat): ##CHECK FORMULA IS CORRECT
    return np.mean(df[stat])

def variance(stat):
    n = 1
    sumSqs = 0
    mean = sampleMean(stat)
    for i in range (0,len(df.index)):
        n += 1
        sumSqs += (df.at[i,stat] - mean) ** 2
    variance = sumSqs / (n - 1)
    return variance
def varianceNumpy(stat): ##CHECK FORMULA IS CORRECT
    return np.var(df[stat])

def standardDeviation(stat):
    var = variance(stat)
    stddev = var ** (1/2.0)
    return stddev
def standardDeviationNumpy(stat): ##CHECK FORMULA IS CORRECT
    return np.std(df[stat])


# In[5]:

mu1 = sampleMean('CS Score (USNews)')
#mu1np = sampleMeanNumpy('CS Score (USNews)') ##CHECK FORMULA IS CORRECT
mu2 = sampleMean('Research Overhead %')
mu3 = sampleMean('Admin Base Pay$')
mu4 = sampleMean('Tuition(out-state)$')
var1 = variance('CS Score (USNews)')
#var1np = varianceNumpy('CS Score (USNews)') ##CHECK FORMULA IS CORRECT
var2 = variance('Research Overhead %')
var3 = variance('Admin Base Pay$')
var4 = variance('Tuition(out-state)$')
sigma1 = standardDeviation('CS Score (USNews)')
#sigma1np = standardDeviationNumpy('CS Score (USNews)') ##CHECK FORMULA IS CORRECT
sigma2 = standardDeviation('Research Overhead %')
sigma3 = standardDeviation('Admin Base Pay$')
sigma4 = standardDeviation('Tuition(out-state)$')


# In[6]:

print"mu1 =",to_precision(mu1,3)
#print"mu1np =",mu1np
print"mu2 =",to_precision(mu2,3)
print"mu3 =",to_precision(mu3,3)
print"mu4 =",to_precision(mu4,3)
print"var1 =",to_precision(var1,3)
#print"var1np =",var1np
print"var2 =",to_precision(var2,3)
print"var3 =",to_precision(var3,3)
print"var4 =",to_precision(var4,3)
print"sigma1 =",to_precision(sigma1,3)
#print"sigma1np =",sigma1np
print"sigma2 =",to_precision(sigma2,3)
print"sigma3 =",to_precision(sigma3,3)
print"sigma4 =",to_precision(sigma4,3)


# #Q2 Compute for each pair of variables their covariance and correlation.
# 
# Show the results in the form of covariance and correlation matrices. 
# 
# Also make a plot of the pairwise data showing the label associated with each data point. 
# 
# Which are the most correlated and least correlated variable pair? 
# 
# Related variables: covarianceMat, correlationMat 

# In[7]:

covarianceMat = np.cov([df['CS Score (USNews)'],df['Research Overhead %'],df['Admin Base Pay$'],df['Tuition(out-state)$']])
correlationMat = np.corrcoef([df['CS Score (USNews)'],df['Research Overhead %'],df['Admin Base Pay$'],df['Tuition(out-state)$']])
def covarXMat(features):
    cov = df[features].cov().as_matrix()
    return cov

def corrXMat(features):
    corr = df[features].corr().as_matrix()
    return corr


# In[8]:

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print "covarianceMat =",covarianceMat
print "correlationMat =",correlationMat


# Also make a plot of the pairwise data showing the label associated with each data point.

# In[9]:

def scatterplot(feature1, feature2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = df[feature1].as_matrix()
    y = df[feature2].as_matrix()
    ax.scatter(x, y)
    fit = np.polyfit(x, y, deg=1)
    ax.plot(x, fit[0] * x + fit[1], color='red')
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    plt.tight_layout()
    plt.show()
    name = feature1 + feature2 + "1.png"
    fig.savefig(name)


# In[10]:

scatterplot('CS Score (USNews)', 'Admin Base Pay$')
scatterplot('CS Score (USNews)', 'Research Overhead %')
scatterplot('CS Score (USNews)', 'Tuition(out-state)$')
scatterplot('Research Overhead %', 'Admin Base Pay$')
scatterplot('Research Overhead %', 'Tuition(out-state)$')
scatterplot('Admin Base Pay$', 'Tuition(out-state)$')


# #Q3 Assuming that each variable is normally distributed and that they are independent of each other, determine the log-likelihood of the data 
# 
# (Use the means and variances computed earlier to determine the likelihood of each data value.) 
# 
# Related variables: logLikelihood 

# In[ ]:




# Calculate the log likelihood for a single variable value (along one dimension)

# In[11]:

def pOfX(x,feature): 
    sigma = standardDeviation(feature)
    mean = sampleMean(feature)
    prob = 1 / (sigma*(2 * math.pi) ** (1/2.0)) * math.e ** (-1 / 2.0 * ((x - mean)/sigma) ** 2.0)
    return prob


# In[12]:

#CALCULATE P(x) FOR A PARTICULAR FEATURE WITH norm.pdf
def pOfXpdf(x,feature):
    sigma = standardDeviation(feature)
    mean = sampleMean(feature)
    prob = norm.pdf(x, loc=mean, scale=sigma)
    return prob


# In[13]:

#ICALCULATE P(x1,x2,...,xn) WITH multivariate_normal.pdf
def pOfXMultivariatePDF(row,features):
    dpoint = np.zeros((len(features)))
    meanArr = np.zeros((len(features)))
    i = 0
    for f in features:
        dpoint[i] = df.loc[row,f]
        meanArr[i] = sampleMean(f)
        i = i + 1
    covr = covarXMat(features)
    return multivariate_normal.pdf(dpoint,mean=meanArr, cov=covr, allow_singular=True)


# Note that by independence, Pn(x)=P(x0)* P(x1) * P(x2)...P(xn)
# 
# To calculate the log likelihood for a single value along all dimensions

# In[14]:

def logLikelihoodMultivariatePDF(features):
    sumlh = 0
    for i in range (0,len(df.index)):
        prob = pOfXMultivariatePDF(i,features)
        sumlh += math.log(prob)
    return sumlh
def logLikelihoodIndependentPDF(features):
    sumlh = 0
    for i in range (0,len(df.index)):
        rowprob = 1
        for f in features:
            x = df.loc[i,f]
            prob = pOfX(x,f)
            rowprob = rowprob*prob
        sumlh += math.log(rowprob)
    return sumlh


# Sum over all data points to get log likelihood of the data

# In[20]:

logLikelihood = logLikelihoodMultivariatePDF(['CS Score (USNews)','Research Overhead %','Admin Base Pay$','Tuition(out-state)$']) #PRINT LOG LIKELIHOOD
logLikelihood2 = logLikelihoodIndependentPDF(['CS Score (USNews)','Research Overhead %','Admin Base Pay$','Tuition(out-state)$'])
print"logLikelihood with Multivariate PDF = ",logLikelihood
print"logLikelihood with independent PDFs = ",logLikelihood2


# #Q4 Using the correlation values construct a Bayesian network which results in a higher log-likelihood than in 3. 
# 
# Related variables: BNgraph, BNlogLikelihood 

# Loop over bnStructure, figure out what each variable is conditioned on 
# e.g.
# [[0 1 0 0]
#  [0 0 0 1]
#  [1 0 0 1]
#  [0 0 0 0]]
#  P(v1,v2,v3,v4) = P(v4|v2,v3)P(v2|v1)P(v1|v3)P(v3)

# In[16]:

#Creates a list of tuples, tuple contains conditioned item and what it is conditioned on, based on bnStructure
vDict = {0:'CS Score (USNews)', 1:'Research Overhead %', 2:'Admin Base Pay$', 3:'Tuition(out-state)$'}
def calcCondDist (bnStruc):
    condDist = []
    i = 0
    while i<4:
        condOn = []
        j = 0
        #print("Checking Child " + vDict[i])
        while j<4:
            #print("Checking Parent " + vDict[j])
            #print("i,j is :")
            #print(bnStruc[i,j])
            if bnStruc[i,j] == 1:
                #print(vDict[j])
                condOn.append(vDict[j])
            j=j+1
        term = (vDict[i],condOn)
        condDist.append(term)
        i=i+1
    return condDist


# In[17]:

def logLikelihoodLinearAlgSolution(data, bnStructure):
    #print(bnStructure)
    condDists = calcCondDist(bnStructure)
    #print(condDists)
    totalLogLike = 0
    for dist in condDists:
        (var1, conditionedvars) = dist
        n = len(data.index)
        numVars = len(conditionedvars)
    #termslhs = np.zeros((numVars+1,numVars+1))#create matrix of summation terms
    #termsrhs = np.zeros((numVars+1,1))
        dfSubset = np.ones((numVars+1,n))
        i = 1
        for f in conditionedvars:
            dfSubset[i] = np.transpose(data[f])
            i = i+1
        termslhs = np.dot(dfSubset,np.transpose(dfSubset))
        termsrhs = np.dot(dfSubset,data[var1])
        params = np.linalg.solve(termslhs, termsrhs)
    #print(params)
        sigmaSquared = np.sum(((np.dot(params,dfSubset) - data[var1])**2))/n
    #print(-1/2*n*math.log(2*math.pi*sigmaSquared))
        loglikelihood = -1*n*math.log(2*math.pi*sigmaSquared)/2 - np.sum(((np.dot(params,dfSubset) - data[var1])**2)/(2*sigmaSquared))
        #print("loglikelihood ",dist,"is",to_precision(loglikelihood,3))
        totalLogLike = totalLogLike + loglikelihood
    return totalLogLike


# In[18]:

bnStructure1 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
bnStructure2 = np.array([[0,1,0,0],[0,0,0,1],[1,0,0,1],[0,0,0,0]])
bnStructure3 = np.array([[0,0,0,1],[1,0,0,0],[0,0,0,0],[0,0,1,0]])
bnStructure4 = np.array([[0,0,0,1],[1,0,1,0],[0,0,0,1],[0,0,0,0]])
bnStructure5 = np.array([[0,0,0,1],[1,0,0,1],[0,0,0,1],[0,0,0,0]])
bnStructure6 = np.array([[0,0,0,1],[1,0,0,1],[0,0,0,0],[0,0,1,0]])
bnStructure7 = np.array([[0,0,0,1],[1,0,1,1],[0,0,0,1],[0,0,0,0]])


print "BNgraph ="
print bnStructure7
print "BNlogLikelihood =",logLikelihoodLinearAlgSolution(df, bnStructure7)


# In[ ]:




# In[ ]:



