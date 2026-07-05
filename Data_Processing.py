# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:35:08 2023

@author: AB

This file contains all functions for data processing
"""

#%% Import packages
import numpy as np
import pandas as pd
from scipy import integrate
import copy
import matplotlib.pyplot as plt

#%%
def Qpfn(mu, sigma, c, logfactorial): 
    y = lambda x: np.exp((c - 1)*np.log(x) - (x + (((np.log(x) - mu)**2) / (2*(sigma ** 2)))) - logfactorial[int(c)])
    lamb, err = integrate.quad(y,0,np.inf)
    return (lamb / sigma)

def Qptot(cVec, nstep, logfactorial):
    # Create array for probability
    p = np.zeros((nstep, nstep))
    # Replace 0s in cVec with 0.25
    lc = np.array(cVec).astype(float)
    lc[lc == 0] = 0.25
    # Average of log(c)
    muTmp = np.average(np.log(lc))
    # Standard deviation of log(c)
    sigTmp = np.std(np.log(lc))
    # Plugging muTmp and sigTmp into the Lognormal Functions to get the proper Mean and Standard Deviation.
    Mean = np.exp(muTmp + ((sigTmp ** 2) / 2))
    Std = np.sqrt((np.exp(sigTmp**2) - 1) * np.exp((2*muTmp) + (sigTmp ** 2)))
    # Mu values to iterate through
    mu = np.linspace(Mean - (3.5*Std), Mean + (2*Std), num = nstep)
    # Sigma values to iterate through
    sigma = np.linspace(Std/200, Std, num = nstep)
    # Calculate the probability of cVec given mu and sigma
    for a in range(nstep):
        for b in range(nstep):
            Qs1 = Qpfn(mu[a], sigma[b], cVec[0], logfactorial)
            Qs2 = Qpfn(mu[a], sigma[b], cVec[1], logfactorial)
            Qs3 = Qpfn(mu[a], sigma[b], cVec[2], logfactorial)
            p[a, b] = max(Qs1 * Qs2 * Qs3, 0) # Ensures positive values
            
    # plt.contourf(p) # !!!
    # Normalize to 1
    p = p / np.sum(p)
    
    mus = np.tile(mu, (nstep, 1)).T
    EMu = np.sum(p * mus) # Expected Mu
    CMu = EMu / np.mean(cVec) # Estimated Mean Intensity over Average of Actual
    
    # Compute 5th and 95th percentiles for mu
    pfl = p.flatten()
    mufl = mus.flatten()
    ix = np.argsort(mufl)
    muflsort = mufl[ix]
    pflsort = pfl[ix]
    cdf = np.cumsum(pflsort)
    Mpct5 = muflsort[np.sum(cdf < 0.05)]
    Mpct95 = muflsort[np.sum(cdf < 0.95)]
    MuConfint = Mpct95 - Mpct5 # Confidence interval for Mu
    MuUncert = MuConfint / EMu # Regional Mean Intensity Confidence Interval over Estimated Mean Intensity
    
    # Return Expected Sigma
    sigmas = np.tile(sigma, (nstep, 1))
    ESigma = np.sum(p * sigmas)
    
    SigVsMu = ESigma / EMu # Estimated Standard Deviation over Estimated Mean Intensity
    
    # Compute 5th and 95th percentiles for Standard Deviation
    Sigfl = sigmas.flatten()
    ix = np.argsort(Sigfl)
    Sigflsort = Sigfl[ix]
    pflsort = pfl[ix]
    cdf = np.cumsum(pflsort)
    Spct5 = Sigflsort[np.sum(cdf < 0.05)]
    Spct95 = Sigflsort[np.sum(cdf < 0.95)]
    SigConfint = Spct95 - Spct5 # Confidence interval for the Standard Deviation
    SigUncert = SigConfint / EMu # Uncertainty in Standard Deviation over Estimated Mean Intensity
    
    return CMu, MuUncert, SigVsMu, SigUncert, Mpct5, Mpct95

def Qfn(a, b, c, logfactorial):
    y = lambda x: np.exp(c*np.log(x) - x - logfactorial[int(c)])
    lamb, err = integrate.quad(y,a,b)
    return ((1/(b-a)) * lamb)
    
def Qtot(m, M, cVec, nstep, logfactorial, ERangeCase):
    # Note a and b refer to q1 and q2 in the documentation.
    # Generate [a,b] pairs
    # Vector of values
    steps = np.linspace(m,M,num=nstep)
    stepscol = np.reshape(steps, (1,-1))
    
    # Grid of interval widths (b-a)
    bminusa = stepscol - stepscol.T / np.sqrt(12)
    # Grid of mean values mean(a,b)
    bamean = (stepscol + stepscol.T) / 2
    # In case 0/0, make the ratio 0
    bratioa = bminusa / (bamean + (bamean == 0))
    
    # Create array for probability
    p = np.zeros((nstep,nstep))
    amax = 0
    for a in range(nstep-1):
        for b in range(a+1,nstep):
            Qs1 = Qfn(steps[a],steps[b],cVec[0],logfactorial)
            Qs2 = Qfn(steps[a],steps[b],cVec[1],logfactorial)
            Qs3 = Qfn(steps[a],steps[b],cVec[2],logfactorial)
            p[a,b] = Qs1 * Qs2 * Qs3
        # Keep running tally of max value (in order to stop)
        if (max(p[a,:]) > amax):
            amax = max(p[a,:])
        else:
            # Terminate if probability becomes too small compared to max
            # @@@ CPT changed to .005, see if there's a difference
            if max(p[a,:]) < (.005*amax):
                 break
    # Normalize to 1
    p = p / np.sum(p)
    
    # compute expected values of midpoint, Std, and Std/mean
    Emean = np.sum(p * bamean)
    EStd = np.sum(p * bminusa)
    if ERangeCase: # If I just need the ERange, I return with that.
        return EStd
    EmeanRatio = np.sum(p * bratioa) # Average of the ratio
    
    Cmean = Emean / np.mean(cVec) # Estimated Mean Intensity over Average of Actual
    Crange = EStd / Emean # Estimated Standard Deviation over Estimated Mean Intensity
    
    # Creating the distribution, xDist, for a random value x.
    xDist = np.zeros(nstep)
    pPrime = p / (bminusa + (bminusa == 0))
    for js in range(nstep):
        xDist[js] = np.sum(pPrime[:js+1,js:])
    xDist = xDist / np.sum(xDist)
    cdfX = np.cumsum(xDist)
    xD05 = steps[np.sum(cdfX < 0.05)] 
    xD95 = steps[np.sum(cdfX < 0.95)] 
    
    # Compute 5th and 95th percentiles for midpoint
    pfl = p.flatten()
    bafl = bamean.flatten()
    ix = np.argsort(bafl)
    baflsort = bafl[ix]
    pflsort = pfl[ix]
    cdf = np.cumsum(pflsort)
    pct5 = baflsort[np.sum(cdf < 0.05)]
    pct95 = baflsort[np.sum(cdf < 0.95)]
    MidConfint = pct95 - pct5 # Confidence interval for the midpoint
    MidUncert = (MidConfint / Emean) # Regional Mean Intensity Confidence Interval over Estimated Mean Intensity
    
    # Compute 5th and 95th percentiles for Standard Deviation
    bminusafl = bminusa.flatten()
    ix = np.argsort(bminusafl)
    bminusaflsort = bminusafl[ix]
    pflsort = pfl[ix]
    cdf = np.cumsum(pflsort)
    pct5r = bminusaflsort[np.sum(cdf < 0.05)]
    pct95r = bminusaflsort[np.sum(cdf < 0.95)]
    StdConfint = pct95r - pct5r # Confidence interval for the Standard Deviation
    StdUncert = StdConfint / Emean # Uncertainty in Standard Deviation over Estimated Mean Intensity
    

    return Crange, Cmean, MidUncert, StdUncert, xD05, xD95, pct5, pct95, Emean

def Theo_Gen(PlotRanges, nstep, kScale, logfactorial, IncludeLognormal): # Data processing function for theoretical models
    Data = {} # All of the data needed for the models will be put into a dictionary
    Data["T_c1ratio"] = np.linspace(0,1,nstep)
    Data["T_c2ratio"] = Data["T_c1ratio"]
    for i in range(4):
        PlotNum = "Bin_" + str(i+1) # Used for naming the inner dictionary
        ThisPlot = {} # The inner dictionary mentioned above.
        ThisPlot["cscale"] = (PlotRanges[i] + PlotRanges[i+1]) / 2
        ThisPlot["T_Cmeans"] = np.zeros((nstep, nstep))
        ThisPlot["T_Crange"] = np.zeros((nstep, nstep))
        ThisPlot["T_MidUncert"] = np.zeros((nstep, nstep))
        ThisPlot["T_RUncert"] = np.zeros((nstep, nstep))
        for j in range(nstep): # These loop through every cell in the theoretical models to calculate what their individual values are.
            for k in range(nstep):
                print(i+1, j+1, k+1)
                c = ThisPlot["cscale"] * np.array([Data["T_c1ratio"][j]*Data["T_c2ratio"][k], Data["T_c2ratio"][k], 1])
                Crange, Cmean, MidUncert, RUncert, xD05, xD95, pct5, pct95, Emean = Qtot(c[0]/kScale, c[2]*kScale, c, nstep, logfactorial, False)
                ThisPlot["T_Cmeans"][j,k] = Cmean
                ThisPlot["T_Crange"][j,k] = Crange
                ThisPlot["T_MidUncert"][j,k] = MidUncert
                ThisPlot["T_RUncert"][j,k] = RUncert
                
        if IncludeLognormal: 
            ThisPlot["T_CMu"] = np.zeros((nstep, nstep))
            ThisPlot["T_MuUncert"] = np.zeros((nstep, nstep))
            ThisPlot["T_SigVsMu"] = np.zeros((nstep, nstep))
            ThisPlot["T_SigUncert"] = np.zeros((nstep, nstep))
            for j in range(nstep): # These loop through every cell in the theoretical models to calculate what their individual values are.
                for k in range(nstep):
                    print(i+1, j+1, k+1)
                    if j >= nstep - 2 and k >= nstep - 2: # In the case of [j,k] = [24,25], [25,24], and [25,25], we replace all values with that of the previous entry to prevent errors.   
                        ThisPlot["T_CMu"][j,k] = CMu
                        ThisPlot["T_MuUncert"][j,k] = MuUncert
                        ThisPlot["T_SigVsMu"][j,k] = SigVsMu
                        ThisPlot["T_SigUncert"][j,k] = SigUncert
                    else:
                        c = ThisPlot["cscale"] * np.array([Data["T_c1ratio"][j]*Data["T_c2ratio"][k], Data["T_c2ratio"][k], 1])
                        CMu, MuUncert, SigVsMu, SigUncert, Mpct5, Mpct95 = Qptot(c, nstep, logfactorial)
                        ThisPlot["T_CMu"][j,k] = CMu
                        ThisPlot["T_MuUncert"][j,k] = MuUncert
                        ThisPlot["T_SigVsMu"][j,k] = SigVsMu
                        ThisPlot["T_SigUncert"][j,k] = SigUncert
        
        Data[PlotNum] = ThisPlot
        
    # The following lines generate the minimums and maximums between all points for each of the values for the global colorbar.
    Data['T_Cmeans_max'] = max(np.max(Data['Bin_1']['T_Cmeans']), np.max(Data['Bin_2']['T_Cmeans']), \
                               np.max(Data['Bin_3']['T_Cmeans']), np.max(Data['Bin_4']['T_Cmeans']))
    Data['T_Cmeans_min'] = min(np.min(Data['Bin_1']['T_Cmeans']), np.min(Data['Bin_2']['T_Cmeans']), \
                               np.min(Data['Bin_3']['T_Cmeans']), np.min(Data['Bin_4']['T_Cmeans']))

    Data['T_Crange_max'] = max(np.max(Data['Bin_1']['T_Crange']), np.max(Data['Bin_2']['T_Crange']), \
                               np.max(Data['Bin_3']['T_Crange']), np.max(Data['Bin_4']['T_Crange']))
    Data['T_Crange_min'] = min(np.min(Data['Bin_1']['T_Crange']), np.min(Data['Bin_2']['T_Crange']), \
                               np.min(Data['Bin_3']['T_Crange']), np.min(Data['Bin_4']['T_Crange']))

    Data['T_MidUncert_max'] = max(np.max(Data['Bin_1']['T_MidUncert']), np.max(Data['Bin_2']['T_MidUncert']), \
                                  np.max(Data['Bin_3']['T_MidUncert']), np.max(Data['Bin_4']['T_MidUncert']))
    Data['T_MidUncert_min'] = min(np.min(Data['Bin_1']['T_MidUncert']), np.min(Data['Bin_2']['T_MidUncert']), \
                                  np.min(Data['Bin_3']['T_MidUncert']), np.min(Data['Bin_4']['T_MidUncert']))

    Data['T_RUncert_max'] = max(np.max(Data['Bin_1']['T_RUncert']), np.max(Data['Bin_2']['T_RUncert']), \
                                np.max(Data['Bin_3']['T_RUncert']), np.max(Data['Bin_4']['T_RUncert']))
    Data['T_RUncert_min'] = min(np.min(Data['Bin_1']['T_RUncert']), np.min(Data['Bin_2']['T_RUncert']), \
                                np.min(Data['Bin_3']['T_RUncert']), np.min(Data['Bin_4']['T_RUncert']))
    if IncludeLognormal:
        Data['T_CMu_max'] = max(np.max(Data['Bin_1']['T_CMu']), np.max(Data['Bin_2']['T_CMu']), \
                                np.max(Data['Bin_3']['T_CMu']), np.max(Data['Bin_4']['T_CMu']))
        Data['T_CMu_min'] = min(np.min(Data['Bin_1']['T_CMu']), np.min(Data['Bin_2']['T_CMu']), \
                                np.min(Data['Bin_3']['T_CMu']), np.min(Data['Bin_4']['T_CMu']))
        
        Data['T_MuUncert_max'] = max(np.max(Data['Bin_1']['T_MuUncert']), np.max(Data['Bin_2']['T_MuUncert']), \
                                     np.max(Data['Bin_3']['T_MuUncert']), np.max(Data['Bin_4']['T_MuUncert']))
        Data['T_MuUncert_min'] = min(np.min(Data['Bin_1']['T_MuUncert']), np.min(Data['Bin_2']['T_MuUncert']), \
                                     np.min(Data['Bin_3']['T_MuUncert']), np.min(Data['Bin_4']['T_MuUncert']))
        
        Data['T_SigVsMu_max'] = max(np.max(Data['Bin_1']['T_SigVsMu']), np.max(Data['Bin_2']['T_SigVsMu']), \
                                    np.max(Data['Bin_3']['T_SigVsMu']), np.max(Data['Bin_4']['T_SigVsMu']))
        Data['T_SigVsMu_min'] = min(np.min(Data['Bin_1']['T_SigVsMu']), np.min(Data['Bin_2']['T_SigVsMu']), \
                                    np.min(Data['Bin_3']['T_SigVsMu']), np.min(Data['Bin_4']['T_SigVsMu']))
        
        Data['T_SigUncert_max'] = max(np.max(Data['Bin_1']['T_SigUncert']), np.max(Data['Bin_2']['T_SigUncert']), \
                                      np.max(Data['Bin_3']['T_SigUncert']), np.max(Data['Bin_4']['T_SigUncert']))
        Data['T_SigUncert_min'] = min(np.min(Data['Bin_1']['T_SigUncert']), np.min(Data['Bin_2']['T_SigUncert']), \
                                      np.min(Data['Bin_3']['T_SigUncert']), np.min(Data['Bin_4']['T_SigUncert']))
    
    return Data

def Data_Gen(Regions, Data, kScale, nstep, logfactorial, IncludeLognormal): # Data processing function for collected data
    E_Conf90XMax = 0 # Highest y value in range plot
    CIMaxCount = 0 # Highest x value in range plot
    
    E_Conf90MuMax = 0
    CIMuMaxCount = 0
    
    for reg in Regions:
        # These all have to be lists, because the amount within them are different for each region.
        E_Conf90M = [] # Confidence interval for midpoint
        E_Conf90X = [] # Confidence interval for random x value
        E_EMidpoint = [] # Expected Midpoint
        
        E_c1ratio1 = [] # Ratio of c1 and c2
        E_c2ratio1 = [] # Ratio of c2 and c3
        E_Crange1 = []
        E_Cmeans1 = []
        E_MidUncert1 = []
        E_RUncert1 = []
        
        E_c1ratio2 = []
        E_c2ratio2 = []
        E_Crange2 = []
        E_Cmeans2 = []
        E_MidUncert2 = []
        E_RUncert2 = []
        
        E_c1ratio3 = []
        E_c2ratio3 = []
        E_Crange3 = []
        E_Cmeans3 = []
        E_MidUncert3 = []
        E_RUncert3 = []
        
        E_c1ratio4 = []
        E_c2ratio4 = []
        E_Crange4 = []
        E_Cmeans4 = []
        E_MidUncert4 = []
        E_RUncert4 = []
        
        if IncludeLognormal:
            E_Conf90Mu = [] # Confidence interval for Mu
            E_EMu = [] # Expected Mu
            
            E_CMu1 = []
            E_MuUncert1 = []
            E_SigVsMu1 = []
            E_SigUncert1 = []
            
            E_CMu2 = []
            E_MuUncert2 = []
            E_SigVsMu2 = []
            E_SigUncert2 = []
            
            E_CMu3 = []
            E_MuUncert3 = []
            E_SigVsMu3 = []
            E_SigUncert3 = []
            
            E_CMu4 = []
            E_MuUncert4 = []
            E_SigVsMu4 = []
            E_SigUncert4 = []
        # This function has to run data processing for every bird-plant interaction in each bin.
        for c in Regions[reg]["Bin 1 cs"]:
            print(reg, c)
            Crange, Cmean, MidUncert, RUncert, xD05, xD95, pct5, pct95, Emean = Qtot(c[0]/kScale, c[2]*kScale, c, nstep, logfactorial, False)
            E_MidUncert1.append(MidUncert)
            E_Crange1.append(Crange)
            E_Cmeans1.append(Cmean)
            E_RUncert1.append(RUncert)
            E_Conf90M.append([pct5, pct95])
            E_Conf90X.append([xD05, xD95])
            E_EMidpoint.append(Emean)
            
            if c[1] == 0:
                E_c1ratio1.append(np.random.rand())
            else:
                E_c1ratio1.append(c[0]/c[1])
            if c[2] == 0:
                E_c2ratio1.append(1)
            else:
                E_c2ratio1.append(c[1]/c[2])
                
            if IncludeLognormal:
                CMu, MuUncert, SigVsMu, SigUncert, Mpct5, Mpct95 = Qptot(c, nstep, logfactorial)
                E_CMu1.append(CMu)
                E_MuUncert1.append(MuUncert)
                E_SigVsMu1.append(SigVsMu)
                E_SigUncert1.append(SigUncert)
                E_Conf90Mu.append([Mpct5, Mpct95])
                
        Regions[reg]["Bin 1 Std / Emean"] = E_Crange1
        Regions[reg]["Bin 1 Emean / c"] = E_Cmeans1
        Regions[reg]["Bin 1 Midpoint Uncertainty"] = E_MidUncert1
        Regions[reg]["Bin 1 Std Uncertainty"] = E_RUncert1
        Regions[reg]["Bin 1 d1s"] = E_c1ratio1
        Regions[reg]["Bin 1 d2s"] = E_c2ratio1

        for c in Regions[reg]["Bin 2 cs"]:
            print(reg, c)
            Crange, Cmean, MidUncert, RUncert, xD05, xD95, pct5, pct95, Emean = Qtot(c[0]/kScale, c[2]*kScale, c, nstep, logfactorial, False)
            E_MidUncert2.append(MidUncert)
            E_Crange2.append(Crange)
            E_Cmeans2.append(Cmean)
            E_RUncert2.append(RUncert)
            E_Conf90M.append([pct5, pct95])
            E_Conf90X.append([xD05, xD95])
            E_EMidpoint.append(Emean)

            if c[1] == 0:
                E_c1ratio2.append(np.random.rand())
            else:
                E_c1ratio2.append(c[0]/c[1])
            if c[2] == 0:
                E_c2ratio2.append(1)
            else:
                E_c2ratio2.append(c[1]/c[2])
                
            if IncludeLognormal:      
                CMu, MuUncert, SigVsMu, SigUncert, Mpct5, Mpct95 = Qptot(c, nstep, logfactorial)
                E_CMu2.append(CMu)
                E_MuUncert2.append(MuUncert)
                E_SigVsMu2.append(SigVsMu)
                E_SigUncert2.append(SigUncert)
                E_Conf90Mu.append([Mpct5, Mpct95])
                
        Regions[reg]["Bin 2 Std / Emean"] = E_Crange2
        Regions[reg]["Bin 2 Emean / c"] = E_Cmeans2
        Regions[reg]["Bin 2 Midpoint Uncertainty"] = E_MidUncert2
        Regions[reg]["Bin 2 Std Uncertainty"] = E_RUncert2
        Regions[reg]["Bin 2 d1s"] = E_c1ratio2
        Regions[reg]["Bin 2 d2s"] = E_c2ratio2
        
        for c in Regions[reg]["Bin 3 cs"]:
            print(reg, c)
            Crange, Cmean, MidUncert, RUncert, xD05, xD95, pct5, pct95, Emean = Qtot(c[0]/kScale, c[2]*kScale, c, nstep, logfactorial, False)
            E_MidUncert3.append(MidUncert)
            E_Crange3.append(Crange)
            E_Cmeans3.append(Cmean)
            E_RUncert3.append(RUncert)
            E_Conf90M.append([pct5, pct95])
            E_Conf90X.append([xD05, xD95])

            E_EMidpoint.append(Emean)
            
            if c[1] == 0:
                E_c1ratio3.append(np.random.rand())
            else:
                E_c1ratio3.append(c[0]/c[1])
            if c[2] == 0:
                E_c2ratio3.append(1)
            else:
                E_c2ratio3.append(c[1]/c[2])
                             
            if IncludeLognormal:  
                CMu, MuUncert, SigVsMu, SigUncert, Mpct5, Mpct95 = Qptot(c, nstep, logfactorial)
                E_CMu3.append(CMu)
                E_MuUncert3.append(MuUncert)
                E_SigVsMu3.append(SigVsMu)
                E_SigUncert3.append(SigUncert)
                E_Conf90Mu.append([Mpct5, Mpct95])
            
        Regions[reg]["Bin 3 Std / Emean"] = E_Crange3
        Regions[reg]["Bin 3 Emean / c"] = E_Cmeans3
        Regions[reg]["Bin 3 Midpoint Uncertainty"] = E_MidUncert3
        Regions[reg]["Bin 3 Std Uncertainty"] = E_RUncert3
        Regions[reg]["Bin 3 d1s"] = E_c1ratio3
        Regions[reg]["Bin 3 d2s"] = E_c2ratio3

        for c in Regions[reg]["Bin 4 cs"]:
            print(reg, c)
            Crange, Cmean, MidUncert, RUncert, xD05, xD95, pct5, pct95, Emean = Qtot(c[0]/kScale, c[2]*kScale, c, nstep, logfactorial, False)
            E_MidUncert4.append(MidUncert)
            E_Crange4.append(Crange)
            E_Cmeans4.append(Cmean)
            E_RUncert4.append(RUncert)
            E_Conf90M.append([pct5, pct95])
            E_Conf90X.append([xD05, xD95])
            E_EMidpoint.append(Emean)
            
            if c[1] == 0:
                E_c1ratio4.append(np.random.rand())
            else:
                E_c1ratio4.append(c[0]/c[1])
            if c[2] == 0:
                E_c2ratio4.append(1)
            else:
                E_c2ratio4.append(c[1]/c[2])
                            
            if IncludeLognormal:    
                CMu, MuUncert, SigVsMu, SigUncert, Mpct5, Mpct95 = Qptot(c, nstep, logfactorial)
                E_CMu4.append(CMu)
                E_MuUncert4.append(MuUncert)
                E_SigVsMu4.append(SigVsMu)
                E_SigUncert4.append(SigUncert)
                E_Conf90Mu.append([Mpct5, Mpct95])
                
        Regions[reg]["Bin 4 Std / Emean"] = E_Crange4
        Regions[reg]["Bin 4 Emean / c"] = E_Cmeans4
        Regions[reg]["Bin 4 Midpoint Uncertainty"] = E_MidUncert4
        Regions[reg]["Bin 4 Std Uncertainty"] = E_RUncert4
        Regions[reg]["Bin 4 d1s"] = E_c1ratio4
        Regions[reg]["Bin 4 d2s"] = E_c2ratio4
        
        Regions[reg]["Confidence Interval for Midpoint"] = E_Conf90M
        Regions[reg]["Confidence Interval for X"] = E_Conf90X
        Regions[reg]["Expected Midpoint"] = E_EMidpoint
        
        if np.size(E_Conf90X) != 0: # If either the count or max value is exceeded, update.
            if np.max(E_Conf90X) > E_Conf90XMax:
                E_Conf90XMax = np.max(E_Conf90X)
            if np.shape(E_Conf90X)[0] > CIMaxCount:
                CIMaxCount = np.shape(E_Conf90X)[0]
             
        # These values are needed to define the x and y limits for the range plot.
        Data['E_ConIntM_max'] = E_Conf90XMax
        Data['CIMaxCount'] = CIMaxCount
             
        if IncludeLognormal:
            Regions[reg]["Bin 1 EMu / c"] = E_CMu1
            Regions[reg]["Bin 1 Mu Uncertainty"] = E_MuUncert1
            Regions[reg]["Bin 1 ESigma / Emu"] = E_SigVsMu1
            Regions[reg]["Bin 1 Sigma Uncertainty"] = E_SigUncert1
            
            Regions[reg]["Bin 2 EMu / c"] = E_CMu2
            Regions[reg]["Bin 2 Mu Uncertainty"] = E_MuUncert2
            Regions[reg]["Bin 2 ESigma / Emu"] = E_SigVsMu2
            Regions[reg]["Bin 2 Sigma Uncertainty"] = E_SigUncert2
            
            Regions[reg]["Bin 3 EMu / c"] = E_CMu3
            Regions[reg]["Bin 3 Mu Uncertainty"] = E_MuUncert3
            Regions[reg]["Bin 3 ESigma / Emu"] = E_SigVsMu3
            Regions[reg]["Bin 3 Sigma Uncertainty"] = E_SigUncert3
            
            Regions[reg]["Bin 4 EMu / c"] = E_CMu4
            Regions[reg]["Bin 4 Mu Uncertainty"] = E_MuUncert4
            Regions[reg]["Bin 4 ESigma / Emu"] = E_SigVsMu4
            Regions[reg]["Bin 4 Sigma Uncertainty"] = E_SigUncert4
            
            Regions[reg]["Confidence Interval for Mu"] = E_Conf90Mu
            Regions[reg]["Expected Mu"] = E_EMu
        
            if np.size(E_Conf90Mu) != 0: # If either the count or max value is exceeded, update.
                if np.max(E_Conf90Mu) > E_Conf90MuMax:
                    E_Conf90MuMax = np.max(E_Conf90Mu)
                if np.shape(E_Conf90Mu)[0] > CIMuMaxCount:
                    CIMuMaxCount = np.shape(E_Conf90Mu)[0]
                    
            # These values are needed to define the x and y limits for the range plot.  
            Data["E_Conf90MuMax"] = E_Conf90MuMax
            Data["CIMuMaxCount"] = CIMuMaxCount
            
    return Regions, Data

def ERange_Processing(Regions_Genus, kScale, nstep, logfactorial, ERangeCap): # Selective data processing function for Emean comparison.
    for reg in Regions_Genus: # I run this algorithm once for Species-species interactions and once more for the grouped interactions.
        PlotInfo = {} # Dictionary to store values for the variance and bar plots by genus.    
        I = np.shape(Regions_Genus[reg]["Uncombined Variance Arrays"][0])[0]
        J = np.shape(Regions_Genus[reg]["Uncombined Variance Arrays"][0])[1]
        # For interactions greater than ERangeCap, the ERange is entered into the array, otherwise it keeps it at 0.
        for i in range(I):
            for j in range(J):
                if max(Regions_Genus[reg]["Uncombined Variance Arrays"][0][i,j], Regions_Genus[reg]["Uncombined Variance Arrays"][1][i,j], Regions_Genus[reg]["Uncombined Variance Arrays"][2][i,j]) < ERangeCap:
                    continue
                else:
                    GenusPairName = str(Regions_Genus[reg]["Genus Names"][i,j]) # Genus pair for the chosen interaction. Used as the dictionary key.
                    c = np.sort([int(Regions_Genus[reg]["Uncombined Variance Arrays"][0][i,j]), int(Regions_Genus[reg]["Uncombined Variance Arrays"][1][i,j]), int(Regions_Genus[reg]["Uncombined Variance Arrays"][2][i,j])]) # Sorts the c values for the interaction
                    ERange = Qtot(c[0]/kScale, c[2]*kScale, c, nstep, logfactorial, True)
                    if GenusPairName in PlotInfo:
                        PlotInfo[GenusPairName][0].append([int(Regions_Genus[reg]["Uncombined Variance Arrays"][0][i,j]), int(Regions_Genus[reg]["Uncombined Variance Arrays"][1][i,j]), int(Regions_Genus[reg]["Uncombined Variance Arrays"][2][i,j])])
                        PlotInfo[GenusPairName][1] += (ERange ** 2)
                    else:
                        PlotInfo[GenusPairName] = [[[int(Regions_Genus[reg]["Uncombined Variance Arrays"][0][i,j]), int(Regions_Genus[reg]["Uncombined Variance Arrays"][1][i,j]), int(Regions_Genus[reg]["Uncombined Variance Arrays"][2][i,j])]]]
                        PlotInfo[GenusPairName].append(ERange ** 2)
        
        L = np.shape(Regions_Genus[reg]["Combined Variance Arrays"][0])[0]
        M = np.shape(Regions_Genus[reg]["Combined Variance Arrays"][0])[1]
        # For interactions greater than ERangeCap, the ERange is entered into the array, otherwise it keeps it at 0.
        for l in range(L):
            for m in range(M):
                if max(Regions_Genus[reg]["Combined Variance Arrays"][0][l,m], Regions_Genus[reg]["Combined Variance Arrays"][1][l,m], Regions_Genus[reg]["Combined Variance Arrays"][2][l,m]) < ERangeCap:
                    continue
                else:
                    GenusPairName = str(Regions_Genus[reg]["Genus Names Combined"][l,m])
                    c = np.sort([int(Regions_Genus[reg]["Combined Variance Arrays"][0][l,m]), int(Regions_Genus[reg]["Combined Variance Arrays"][1][l,m]), int(Regions_Genus[reg]["Combined Variance Arrays"][2][l,m])]) # Sorts the c values for the interaction
                    PlotInfo[GenusPairName].append((Qtot(c[0]/kScale, c[2]*kScale, c, nstep, logfactorial, True)) ** 2)
                  
            Regions_Genus[reg]["Combined Variance Plot Data"] = PlotInfo
    return Regions_Genus
        
        
        
        
        
        
        
        