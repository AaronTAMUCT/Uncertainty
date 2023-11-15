# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:38:58 2023

@author: AB

This file contains all functions for collecting important data from the csvs
"""

#%% Import packages
import numpy as np
import pandas as pd
import copy

#%%
def Get_Genus_Labels(cx, dataB, dataP, name, Region, combined): # This function creates an array that contains the genus names for the plant and bird in each interaction in each cell.
    cNew = copy.deepcopy(cx)
    if not combined:
        cNew = cx.rename(columns = dict(zip(dataB['X'], dataB["Genus"]))) # Renames each column header to the bird's respective genus name.
        cNew.index = dataP["Genus"].values # Renames each row index name to the plant's respective genus name.
    for b in range(len(cNew.columns)):
        cNew[cNew.columns[b]] = cNew[cNew.columns[b]].astype(object) # Changes the type of the values in cNew to object for the next step.
        for a in range(len(cNew.index)):
            cNew.iat[a,b] = [cNew.columns[b], cNew.index[a]] # Change each cell to the bird/plant ggenuses associated with the interaction.
    Region[name] = cNew.to_numpy() # Saves this array as a numpy array to remove the column names and index for easier calling later.

def bp_data_processing_Proj_2(ColumnFile, RowFile, fileDict, siteNames, Group, PlotRanges, IncludeSmallValues, ERangeCap): # This function collects relevant data from the csvs.
    Regions = {} # Creates the Regions dictionary.
    for j in range(2): # This iterates through each of the 6 regions.
        for i in range(3):
            Region = {} # Creates a dictionary for the singular region.
            Reg = fileDict[siteNames[i,j]]
            
            # Read raw data
            dataB = pd.read_csv(ColumnFile)
            dataP = pd.read_csv(RowFile)
    
            # Select data from single site
            dataB = dataB[dataB['Site'] == siteNames[i,j]]
            dataP = dataP[dataP['Site'] == siteNames[i,j]]
            
            """ Pulling dataframes from each CSV and removing duplicate columns and rows. 
                There are a couple of extra steps in here, because there is a column with
                a space in the name and a single duplicate column. """ 
            
            c1 = pd.read_csv(Reg[0:2]+"1"+Reg[2:], index_col = 0) # Pulls the interaction numbers from location 1.
            c1.columns = c1.columns.str.replace(' ', '') # Removes any spaces from the column names
            if (sum(c1.columns.duplicated()) != 0): # If there are any duplicate columns
                print("Duplicate column")
                c1 = c1.loc[:,~c1.columns.duplicated()] # Removes the duplicate column
            c1.index = c1.index.str.replace(' ', '') # Removes any spaces from the index names.
            if (sum(c1.index.duplicated()) != 0): # If there are any duplicate rows
                print("Duplicate row")
                c1 = c1.loc[~c1.index.duplicated(), :] # Removes the duplicate row
            
            c2 = pd.read_csv(Reg[0:2]+"2"+Reg[2:], index_col = 0) # Pulls the interaction numbers from location 2.
            c2.columns = c2.columns.str.replace(' ', '')
            if (sum(c2.columns.duplicated()) != 0):
                print("Duplicate column")
                c2 = c2.loc[:,~c2.columns.duplicated()]
            c2.index = c2.index.str.replace(' ', '')
            if (sum(c2.index.duplicated()) != 0):
                print("Duplicate row")
                c2 = c2.loc[~c2.index.duplicated(), :]
            
            c3 = pd.read_csv(Reg[0:2]+"3"+Reg[2:], index_col = 0) # Pulls the interaction numbers from location 3.
            c3.columns = c3.columns.str.replace(' ', '')
            if (sum(c3.columns.duplicated()) != 0):
                print("Duplicate column")
                c3 = c3.loc[:, ~c3.columns.duplicated()]
            c3.index = c3.index.str.replace(' ', '')
            if (sum(c3.index.duplicated()) != 0):
                print("Duplicate row")
                c3 = c3.loc[~c3.index.duplicated(), :]
        
            """ I pull the total values and set all values to zero for the lists of all 
                bird and plant names, once again removing any duplicate rows or columns
                if present. """
            
            ct = pd.read_csv(Reg, index_col = 0)
            ct.columns = ct.columns.str.replace(' ', '')
            if (sum(ct.columns.duplicated()) != 0):
                print("Duplicate column")
                ct = ct.loc[:, ~ct.columns.duplicated()]
            ct.index = ct.index.str.replace(' ', '')
            if (sum(ct.index.duplicated()) != 0):
                print("Duplicate row")
                ct = ct.loc[~ct.index.duplicated(), :]
            for col in ct.columns:
                ct[col].values[:] = 0 # Sets all values to zero.
        
            """ Next, we concatenate each subset of the interactions with the total to set 
                all 3 sets of data to the same size. This is necessary to have c-values for
                all bird/plant interactions, where 0 fills the empty spaces. """
    
            c12 = pd.concat([c1, ct], sort = False).groupby(level = 0).sum() # This adds in any birds or plants in the other two locations and sets their interaction levels to 0.
            c12 = c12.reindex(sorted(c12.columns), axis=1) # Sorts the columns
            c12 = c12.reindex(sorted(c12.index, key=lambda x: x.lower())) # Sorts the rows
            c22 = pd.concat([c2, ct], sort = False).groupby(level = 0).sum()
            c22 = c22.reindex(sorted(c22.columns), axis=1)
            c22 = c22.reindex(sorted(c22.index, key=lambda x: x.lower()))
            c32 = pd.concat([c3, ct], sort = False).groupby(level = 0).sum()
            c32 = c32.reindex(sorted(c32.columns), axis=1)
            c32 = c32.reindex(sorted(c32.index, key=lambda x: x.lower()))
            
            """ First, we need to keep only the interactions that have at least one location value above the set minimum. """
            
            if not IncludeSmallValues:
                
                c12 = c12.where((c12 >= PlotRanges[0]) | (c22 >= PlotRanges[0]) | (c32 >= PlotRanges[0]), other = 0)
                c22 = c22.where((c12 >= PlotRanges[0]) | (c22 >= PlotRanges[0]) | (c32 >= PlotRanges[0]), other = 0)
                c32 = c32.where((c12 >= PlotRanges[0]) | (c22 >= PlotRanges[0]) | (c32 >= PlotRanges[0]), other = 0)
            
            """ Here, if the grouping is by genus, we need this separate 
                process to gather the modified data for the variance plot. """
            
            if Group == "Genus":
                
                """ We need to guarantee that only the interactions that have at least one location value above the set minimum are kept, regardless of IncludeSmallValues. """
                 
                c12_Removed = copy.copy(c12)
                c22_Removed = copy.copy(c22)
                c32_Removed = copy.copy(c32)
                
                c12_Removed = c12_Removed.where((c12 >= PlotRanges[0]) | (c22 >= PlotRanges[0]) | (c32 >= PlotRanges[0]), other = 0)
                c22_Removed = c22_Removed.where((c12 >= PlotRanges[0]) | (c22 >= PlotRanges[0]) | (c32 >= PlotRanges[0]), other = 0)
                c32_Removed = c32_Removed.where((c12 >= PlotRanges[0]) | (c22 >= PlotRanges[0]) | (c32 >= PlotRanges[0]), other = 0)
                
                """ Next, we store the kept values. """
                
                Region["Uncombined Variance Arrays"] = [c12_Removed.to_numpy(), c22_Removed.to_numpy(), c32_Removed.to_numpy()]
                
                """ Now, we combine the rows and columns of each subset on the basis of the grouping. """
                
                c12_Removed = c12_Removed.rename(columns = dict(zip(dataB['X'], dataB[Group]))) # Renames each column header to the bird's respective group to be analyzed.
                c12_Removed.index = dataP[Group].values # Renames each row index name to the plant's respective group to be analyzed.
                c12_Removed = c12_Removed.groupby(by = c12_Removed.columns, axis = 1).sum() # Adds together the columns of birds of the same group.
                c12_Removed = c12_Removed.groupby(by = c12_Removed.index, sort = False, axis = 0).sum() # Adds together the rows of plants of the same group.
                
                c22_Removed = c22_Removed.rename(columns=dict(zip(dataB['X'], dataB[Group])))
                c22_Removed.index = dataP[Group].values
                c22_Removed = c22_Removed.groupby(by = c22_Removed.columns, axis = 1).sum()
                c22_Removed = c22_Removed.groupby(by = c22_Removed.index, sort = False, axis = 0).sum()

                c32_Removed = c32_Removed.rename(columns=dict(zip(dataB['X'], dataB[Group])))
                c32_Removed.index = dataP[Group].values
                c32_Removed = c32_Removed.groupby(by = c32_Removed.columns, axis = 1).sum()
                c32_Removed = c32_Removed.groupby(by = c32_Removed.index, sort = False, axis = 0).sum()
                
                """ Once again, we store the combined kept values. """
                
                Region["Combined Variance Arrays"] = [c12_Removed.to_numpy(), c22_Removed.to_numpy(), c32_Removed.to_numpy()]
                
                """ Saves the genus names for each bird-plant interaction for the variance plot. """
            
                Get_Genus_Labels(c12, dataB, dataP, "Genus Names", Region, False) # Between c12, c22, c32 and any of the variants with removed values, they all are interchangeable for this process.
            
            """ Now, we combine the rows and columns of each subset on the basis of the grouping. """
            
            c12 = c12.rename(columns = dict(zip(dataB['X'], dataB[Group]))) # Renames each column header to the bird's respective group to be analyzed.
            c12.index = dataP[Group].values # Renames each row index name to the plant's respective group to be analyzed.
            c12 = c12.groupby(by = c12.columns, axis = 1).sum() # Adds together the columns of birds of the same group.
            c12 = c12.groupby(by = c12.index, sort = False, axis = 0).sum() # Adds together the rows of plants of the same group.
            
            c22 = c22.rename(columns=dict(zip(dataB['X'], dataB[Group])))
            c22.index = dataP[Group].values
            c22 = c22.groupby(by = c22.columns, axis = 1).sum()
            c22 = c22.groupby(by = c22.index, sort = False, axis = 0).sum()

            c32 = c32.rename(columns=dict(zip(dataB['X'], dataB[Group])))
            c32.index = dataP[Group].values
            c32 = c32.groupby(by = c32.columns, axis = 1).sum()
            c32 = c32.groupby(by = c32.index, sort = False, axis = 0).sum()
            
            """ We also need the genus names after combining. """
            
            if Group == "Genus":
                Get_Genus_Labels(c12, dataB, dataP, "Genus Names Combined", Region, True)
            
            """ This step removes bird/plant names. They are no longer needed. """
            
            Region["c1s"] = c12.to_numpy()
            Region["c2s"] = c22.to_numpy()
            Region["c3s"] = c32.to_numpy()
            
            cmax = np.zeros(np.shape(c12)) # Like before, the one chosen here between c12, c22, and c32 is arbitrary.
            for k in range(np.shape(c12)[0]):
                for l in range(np.shape(c12)[1]):
                    cmax[k,l] = max(Region["c1s"][k,l], Region["c2s"][k,l], Region["c3s"][k,l])
            Region["cmax"] = cmax # This array contains the highest value for each interaction between the 3 locations.
            
            if IncludeSmallValues:
                cmax_10above = np.where(cmax < PlotRanges[0], 0, cmax)
                cmax_loc = np.transpose(np.nonzero(cmax_10above)) # Gives the matrix locations for all values >= feedmin
            else:
                cmax_loc = np.transpose(np.nonzero(cmax)) # Gives the matrix locations for all values >= feedmin
            cmax_size = np.shape(cmax_loc)[0] # Gives the number of interactions >= feedmin
            Bin1 = []
            Bin2 = []
            Bin3 = []
            Bin4 = []
            # This function sorts all bird-plant interactions by their amount, sorting them into their respective bins.
            for z in range(cmax_size):
                if (cmax[cmax_loc[z][0], cmax_loc[z][1]] < PlotRanges[1]): # If the cmax value is less than the first bin cap, 
                    Bin1.append(cmax_loc[z]) # add the interaction to bin 1.
                elif (cmax[cmax_loc[z][0], cmax_loc[z][1]] < PlotRanges[2]): # If the cmax value is higher than the first bin cap, but below the second,
                    Bin2.append(cmax_loc[z]) # add the interattion to bin 2.
                elif (cmax[cmax_loc[z][0], cmax_loc[z][1]] < PlotRanges[3]): # If the cmax value is higher than the second bin cap, but below the third,
                    Bin3.append(cmax_loc[z]) # add the interaction to bin 3.
                else:
                    Bin4.append(cmax_loc[z]) # If it is higher than the 3 caps, add the interaction to bin 4.
            
            Region["Bin 1"] = np.matrix(Bin1)
            Region["Bin 2"] = np.matrix(Bin2)
            Region["Bin 3"] = np.matrix(Bin3)
            Region["Bin 4"] = np.matrix(Bin4)
            
            # Note that the bins just contain the matrix location of the bird-plant combination.
            # These 4 if-else statements return the sorted c values from each location.
            if np.shape(Region["Bin 1"])[1] > 0: 
                Region["Bin 1 cs"] = [np.sort([Region["c1s"][Region["Bin 1"][x,0], Region["Bin 1"][x,1]], \
                                               Region["c2s"][Region["Bin 1"][x,0], Region["Bin 1"][x,1]], \
                                                   Region["c3s"][Region["Bin 1"][x,0], Region["Bin 1"][x,1]]]) for x in range(np.shape(Region["Bin 1"])[0])]
            else:
                Region["Bin 1 cs"] = np.array([])
                
            if np.shape(Region["Bin 2"])[1] > 0:
                Region["Bin 2 cs"] = [np.sort([Region["c1s"][Region["Bin 2"][x,0], Region["Bin 2"][x,1]], \
                                               Region["c2s"][Region["Bin 2"][x,0], Region["Bin 2"][x,1]], \
                                                   Region["c3s"][Region["Bin 2"][x,0], Region["Bin 2"][x,1]]]) for x in range(np.shape(Region["Bin 2"])[0])]
            else:
                Region["Bin 2 cs"] = np.array([])
            
            if np.shape(Region["Bin 3"])[1] > 0:
                Region["Bin 3 cs"] = [np.sort([Region["c1s"][Region["Bin 3"][x,0], Region["Bin 3"][x,1]], \
                                               Region["c2s"][Region["Bin 3"][x,0], Region["Bin 3"][x,1]], \
                                                   Region["c3s"][Region["Bin 3"][x,0], Region["Bin 3"][x,1]]]) for x in range(np.shape(Region["Bin 3"])[0])]
            else:
                Region["Bin 3 cs"] = np.array([])
                
            if np.shape(Region["Bin 4"])[1] > 0:
                Region["Bin 4 cs"] = [np.sort([Region["c1s"][Region["Bin 4"][x,0], Region["Bin 4"][x,1]], \
                                               Region["c2s"][Region["Bin 4"][x,0], Region["Bin 4"][x,1]], \
                                                   Region["c3s"][Region["Bin 4"][x,0], Region["Bin 4"][x,1]]]) for x in range(np.shape(Region["Bin 4"])[0])]
            else:
                Region["Bin 4 cs"] = np.array([])
            
            Regions[siteNames[i,j]] = Region
            
    # Defining each regions marker colors and shapes
    Regions["Bombuscaro"]["Color"] = "Black" # 1000 + Nat
    Regions["Copalinga"]["Color"] = "Black" # 1000 + Dis
    Regions["ECSF"]["Color"] = "Red" # 2000 + Nat
    Regions["Finca"]["Color"] = "Red" # 2000 + Dis
    Regions["Cajanuma"]["Color"] = "Blue" # 3000 + Nat
    Regions["Bellavista"]["Color"] = "Blue" # 3000 + Dis

    Regions["Bombuscaro"]["Shape"] = "o" # Dot
    Regions["Copalinga"]["Shape"] = "," # Square
    Regions["ECSF"]["Shape"] = "o"
    Regions["Finca"]["Shape"] = ","
    Regions["Cajanuma"]["Shape"] = "o"
    Regions["Bellavista"]["Shape"] = ","
    
    return Regions
