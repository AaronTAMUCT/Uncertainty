"""
Created on Sat Oct  8 09:43:49 2022

author: AB
"""
#%% Notes for Aaron (also in Overleaf)

#%% Notes for Thron.
""" 9 Sep 2023
    Turn in code to GitHub
"""
#%% Parameters and packages

# Import packages
import numpy as np
import time # Used for optimizing functions
import pickle
import sys

# Import functions from files
from Data_Collection import *
from Data_Processing import *
from Data_Plotting import *

ColumnFile = "totalBirds.csv" # Data separated by columns in interaction files (Birds in the case of the study)
RowFile = "totalPlants.csv" # Data separated by rows in interaction files (Plants in the case of the study)
fileDict = {"Bombuscaro": "bo_net.csv", "Bellavista": "be_net.csv", "Copalinga": "co_net.csv", "Cajanuma": "ca_net.csv", "ECSF": "sf_net.csv", "Finca": "fi_net.csv"} # Dictionary of each file to each location
# Locations in each site are separated in the following way: The total data is in "bo_net.csv", the data for location 1 is in "bo1_net.csv", the data for location 2 is in "bo2_net.csv", and the data for location 3 is in "bo3_net.csv", 
siteNames = np.array([['Bombuscaro', 'Copalinga'], ['ECSF', 'Finca'], ['Cajanuma', 'Bellavista']]) # Array of site names in particular order
Group = "Genus" # 'X' for species, 'Genus' for genus, or 'Family' for family
GroupLabel = "Genus" # 'Species' for species, 'Genus' for genus, or 'Family' for family

nstep = 25 # Square root of number of Qtots run
kScale = 2 # Scalar operated with m and M
ERangeCap = 10 # Used to restrict the values calculated when performing the ERange comparison.

IncludeLognormal = False # Toggle to include Lognormal distribution in calculation.
IncludeSmallValues = True # Toggle to include values below PlotRanges[0]
Points = True # Whether to include data points on plots
LoadData = False # Toggle to load pre-calculated data for plotting/reviewing
SaveData = False # Toggle to save loaded or calculated data to files
Plotting = False # Toggle to plot calculated/loaded data

PlotRanges = [10,20,30,40,50]
xylims = np.array([-0.05,1.05]) # X and y ranges in contour plots.

""" Because the function Qfn is run well over 1,000,000 times, 
    logfactorial values are precalculated so we only need to 
    select them instead of performing this calculation each time. """
logfactorial = np.cumsum(np.log(1 + np.arange(1000)))
logfactorial = np.insert(logfactorial, 1, 0)

#%% Run functions to collect and generate relevant data from scratch
if LoadData:
    try:    
        Data = pickle.load(open("Data_"+GroupLabel+".pkl", "rb"))
        Regions = pickle.load(open("Regions_"+GroupLabel+".pkl", "rb"))
    except FileNotFoundError: # !!! I have never really done exceptions well before. I may not be doing this right.
        sys.exit("File not in directory. Consider calculating and saving the file.")
else:    
    Data = Theo_Gen(PlotRanges, nstep, kScale, logfactorial, IncludeLognormal)
    Regions = bp_data_processing_Proj_2(ColumnFile, RowFile, fileDict, siteNames, Group, PlotRanges, IncludeSmallValues, ERangeCap)
    Regions, Data = Data_Gen(Regions, Data, kScale, nstep, logfactorial, IncludeLognormal)
    if SaveData:
        pickle.dump(Data, open("Data_"+GroupLabel+".pkl", "wb"))
        pickle.dump(Regions, open("Regions_"+GroupLabel+".pkl", "wb"))
        print("Data saved.")
#%% Plotting       
if Plotting:
    QPlot("(Estimated Mean intensity) / (Average of Actual Obs. Values)", "T_Cmeans", Points, Regions, Data, PlotRanges, xylims)
    QPlot("Estimated Standard Deviation / Estimated Mean intensity", "T_Crange", Points, Regions, Data, PlotRanges, xylims)
    QPlot("Regional Mean Intensity Confidence Interval / Estimated Mean Intensity", "T_MidUncert", Points, Regions, Data, PlotRanges, xylims)
    QPlot("Uncertainty in Estimated Standard Deviation / Estimated Mean Intensity", "T_RUncert", Points, Regions, Data, PlotRanges, xylims)
    RangeGraphs(Regions, Data, siteNames)
    cmax_Hist(Regions, siteNames, Group)
    try:
        Regions = ERange_Processing(Regions, kScale, nstep, logfactorial, ERangeCap)
        ERange_Plot(Regions, siteNames)
    except FileNotFoundError:
        sys.exit("Load or create the Genus file.")
