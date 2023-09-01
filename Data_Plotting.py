# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:46:14 2023

@author: AB

This file contains all graphing functions
"""
#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%%
def QPlot(Title, Set, Points, Regions, Data, PlotRanges, xylims): # This function generates the 2x2 plots containing the information gathered in Theo_Gen.
    fig, ax = plt.subplots(2,2)
    fig.suptitle(Title)
    fig.set_size_inches(6, 6)
    
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("C1 / C2")
    plt.ylabel("C2 / C3")
    
    # This plots the data from Theo_Gen and changes the title of each plot depending on whether the interaction points are present. # ??? Should I change the plot titles, because both titles are technically true?
    ax[0,0].contourf(Data["T_c1ratio"], Data["T_c2ratio"], Data['Bin_1'][Set].T, levels = 10, vmin = Data[Set + '_min'], vmax = Data[Set + '_max'])
    if Points == True:
        ax[0,0].set_title(f"{PlotRanges[0]} <= cmax <= {PlotRanges[1]}")
    else:
        ax[0,0].set_title(f"CScale = {int((PlotRanges[0] + PlotRanges[1]) / 2)}")
    ax[0,1].contourf(Data["T_c1ratio"], Data["T_c2ratio"], Data['Bin_2'][Set].T, levels = 10, vmin = Data[Set + '_min'], vmax = Data[Set + '_max'])
    if Points == True:
        ax[0,1].set_title(f"{PlotRanges[1]} <= cmax <= {PlotRanges[2]}")
    else:
        ax[0,1].set_title(f"CScale = {int((PlotRanges[1] + PlotRanges[2]) / 2)}")
    ax[1,0].contourf(Data["T_c1ratio"], Data["T_c2ratio"], Data['Bin_3'][Set].T, levels = 10, vmin = Data[Set + '_min'], vmax = Data[Set + '_max'])
    if Points == True:
        ax[1,0].set_title(f"{PlotRanges[2]} <= cmax <= {PlotRanges[3]}")
    else:
        ax[1,0].set_title(f"CScale = {int((PlotRanges[2] + PlotRanges[3]) / 2)}")
    ax[1,1].contourf(Data["T_c1ratio"], Data["T_c2ratio"], Data['Bin_4'][Set].T, levels = 10, vmin = Data[Set + '_min'], vmax = Data[Set + '_max'])
    if Points == True:
        ax[1,1].set_title(f"cmax > {PlotRanges[3]}")
    else:
        ax[1,1].set_title(f"CScale = {int((PlotRanges[3] + PlotRanges[4]) / 2)}")
    
    if Points == True:
        # Plotting bird values
        for reg in Regions:
            ax[0,0].scatter(Regions[reg]["Bin 1 d1s"], Regions[reg]["Bin 1 d2s"], c = Regions[reg]["Color"], marker = Regions[reg]["Shape"], label = reg)
            ax[0,1].scatter(Regions[reg]["Bin 2 d1s"], Regions[reg]["Bin 2 d2s"], c = Regions[reg]["Color"], marker = Regions[reg]["Shape"], label = reg)
            ax[1,0].scatter(Regions[reg]["Bin 3 d1s"], Regions[reg]["Bin 3 d2s"], c = Regions[reg]["Color"], marker = Regions[reg]["Shape"], label = reg)
            ax[1,1].scatter(Regions[reg]["Bin 4 d1s"], Regions[reg]["Bin 4 d2s"], c = Regions[reg]["Color"], marker = Regions[reg]["Shape"], label = reg)
            
        # Global legends
        handles, labels = ax[1,1].get_legend_handles_labels()
        labels, index = np.unique(labels, return_index=True)
        handles = [handles[i] for i in index]
        fig.legend(handles, labels, loc = 'lower right')
    
        # Adjust x and y limits
        ax[0,0].set_xlim(xylims)
        ax[0,0].set_ylim(xylims)   
        ax[0,1].set_xlim(xylims)
        ax[0,1].set_ylim(xylims)   
        ax[1,0].set_xlim(xylims)
        ax[1,0].set_ylim(xylims)   
        ax[1,1].set_xlim(xylims)
        ax[1,1].set_ylim(xylims)
        
    fig.tight_layout()

    # Generate the global colorbar
    sm = plt.cm.ScalarMappable(cmap = "viridis", norm = plt.Normalize(vmin = Data[Set + '_min'], vmax = Data[Set + '_max']))
    fig.subplots_adjust(left = 0.1, right = 0.765)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(sm, cax=cbar_ax)
    
    plt.show()
    
def RangeGraphs(Regions, Data, siteNames): # This function generates the range plot for bird-plant interactions.
    Width = 0.40
    fig, ax = plt.subplots(3,2)
    fig.suptitle('Range Plot of Variance by Region')
    fig.set_size_inches(12, 9)
    plt.xlabel("Bird/Plant Interactions")
    plt.ylabel("Variance")
    
    for j in range(2):
        for i in range(3): # Iterates through each site
            Reg = siteNames[i,j]
            ax[i,j].set_title(Reg)
            ax[i,j].set_ylim([1, Data['E_ConIntM_max'] + 5]) # Sets the x and y limits based on the largest in each direction
            ax[i,j].set_xlim([-2, Data['CIMaxCount']]) 
            if len(Regions[Reg]["Confidence Interval for Midpoint"]) == 0: # These lines are made for Bellavista, which tends to have nothing to offer here.
                continue
            E_Conf90M = Regions[Reg]["Confidence Interval for Midpoint"] # Shown in blue
            E_Conf90X = Regions[Reg]["Confidence Interval for X"]   # Shown in red
            E_EMidpoint = Regions[Reg]["Expected Midpoint"] # Shown in black
            
            ix = np.argsort(E_Conf90M, axis = 0)[:,0].T # Sorts all intervals by the 5% value.
            
            E_Conf90MArr = np.array(E_Conf90M)
            E_Conf90XArr = np.array(E_Conf90X)
            E_EMidpointArr = np.array(E_EMidpoint)
            E_Conf90M_2 = E_Conf90MArr[ix]
            E_Conf90X_2 = E_Conf90XArr[ix]
            E_EMidpoint_2 = E_EMidpointArr[ix]
            
            xspan = len(E_Conf90M_2)
        
            for k in range(xspan): # Here each range is plotted individually as 3 rectangles.
                lowerlim = max(E_Conf90X_2[k][0],0.9)
                ax[i,j].add_patch(patches.Rectangle((k, lowerlim), Width, E_Conf90X_2[k][1] - E_Conf90X_2[k][0], color = 'red'))
                ax[i,j].add_patch(patches.Rectangle((k, E_Conf90M_2[k][0]), Width, E_Conf90M_2[k][1] - E_Conf90M_2[k][0], color = 'blue'))
                ax[i,j].add_patch(patches.Rectangle((k-.1, E_EMidpoint_2[k]), 0.45, Width, color = 'black'))

            ax[i,j].plot((E_Conf90M_2[0][0],E_Conf90M_2[xspan-1][1]), 'r-', alpha = 0.01)
            ax[i,j].set_yscale("log")
            ax[i,j].set_title(Reg)
            
    plt.tight_layout()
    plt.show()
    
def cmax_Hist(Regions, siteNames, Group):
    fig, ax = plt.subplots(3,2)
    fig.suptitle('Maximum interaction counts')
    fig.set_size_inches(12, 9)
    plt.rcParams.update({'font.size': 8})

    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Maximum number of interactions per location")

    for j in range(2):
        for i in range(3):
            ax[i,j].hist(np.log10(Regions[siteNames[i,j]]["cmax"][Regions[siteNames[i,j]]["cmax"] != 0]), bins = np.linspace(0,3,13))
            ax[i,j].set_title(siteNames[i,j])
            ax[i,j].set_xlim([0,3])
            ax[i,j].set_xticks(range(4), ['0','10','100','1000'])
            if Group == 'X':
                ax[i,j].set_ylim([0,70])
            else:
                ax[i,j].set_ylim([0,15])
                
def ERange_Plot(Regions, siteNames):
    fig, ax = plt.subplots(2,2)
    fig.suptitle('Inter-location Interaction Strength Variance by Genus, Species-Combined and Species-Separated')
    fig.set_size_inches(12, 9)
    plt.rcParams.update({'font.size': 8})

    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Variance")
    
    for j in range(2):
        for i in range(2):
            Data = Regions[siteNames[i,j]]["Combined Variance Plot Data"]
            Data = Data[Data[:,1] != Data[:,2]]
            xRange = np.shape(Data)[0]
            
            ax[i,j].plot(np.arange(xRange), [np.log10(D) for D in Data[:,1]], label = 'Species-Separated')
            ax[i,j].plot(np.arange(xRange), [np.log10(D) for D in Data[:,2]], label = 'Species-Combined')
            
            
            ax[i,j].set_title(siteNames[i,j])
            ax[i,j].set_xticks(range(xRange), Data[:,0])
            ax[i,j].tick_params(labelrotation = 90)  
            ax[i,j].set_yscale("log")
    
    # Global legends
    handles, labels = ax[1,1].get_legend_handles_labels()
    labels, index = np.unique(labels, return_index=True)
    handles = [handles[i] for i in index]
    fig.legend(handles, labels, loc = 'lower right')
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    