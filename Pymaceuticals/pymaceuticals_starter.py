#!/usr/bin/env python
# coding: utf-8

# ## Observations and Insights 

# 

# In[359]:


# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import pylab as pl
import sklearn
from scipy.stats import linregress
from sklearn import datasets

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)


# Combine the data into a single dataset
mouse_df =pd.merge(mouse_metadata,study_results, on="Mouse ID")


# Display the data table for preview

mouse_df = mouse_df.sort_values(["Mouse ID","Timepoint"], ascending = [True,True])
mouse_df = mouse_df.reset_index()


mouse_df


# In[360]:


# Checking the number of mice from original data.
number_of_mice_original = mouse_df.count()
number_of_mice_original


# In[361]:


# Finding how many pairs of unique mouse ID  and how many Timepoints per each mouse
mouse_df["Mouse ID"].value_counts()


# In[362]:


# View data only ID# g989
only_g989 = mouse_df.loc[mouse_df["Mouse ID"] == "g989", :]
sorted_only_g989 = only_g989.sort_values(["Timepoint"], ascending = [True]) 
sorted_only_g989


# In[363]:


# check types of variable
mouse_df.dtypes


# In[364]:


# Change "Tumor Volume (mm3)" from float64 to interger
mouse_df['Tumor Volume (mm3)'] = mouse_df['Tumor Volume (mm3)'].astype('int')
mouse_df.dtypes


# In[365]:



# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 

#cleaned_mouse_df = mouse_df.drop_duplicates(subset="Mouse ID", keep = "first")

#cleaned_mouse_df

cleaned_mouse_df = mouse_df.drop_duplicates(subset="Mouse ID", keep = "last")
cleaned_mouse_df = cleaned_mouse_df.sort_values(["Mouse ID","Timepoint"], ascending = [True, True]) 

del cleaned_mouse_df["index"]

cleaned_mouse_df


# In[366]:


cleaned_mouse_df["Mouse ID"].value_counts()


# In[367]:


only_g989 = cleaned_mouse_df.loc[cleaned_mouse_df["Mouse ID"] == "g989", :]
print(only_g989)
sorted_only_g989 = only_g989.sort_values(["Timepoint"], ascending = [True]) 
sorted_only_g989


# In[368]:


# Checking number of mice after removing duplicates
cleaned_mouse_df.describe()


# In[369]:


# Optional: Get all the data for the duplicate mouse ID. 

duplicate_mouse_ID = mouse_df["Mouse ID"].duplicated() 
  
# displaying data 

mouse_df[duplicate_mouse_ID]


# In[370]:


# Checking the number of mice in the clean DataFrame.
cleaned_mouse_df.count()


# ## Summary Statistics

# In[371]:


# Generate a summary statistics table of mean, median, variance, standard deviation, 
    #and SEM of the tumor volume for each regimen
# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
# Assemble the resulting series into a single summary dataframe.


# In[372]:


# create a group by object based on "Drug Regimen"

grouped_by_drug = cleaned_mouse_df.groupby(["Drug Regimen"])

grouped_by_drug.count()


# In[373]:


tumor_volume_mm3_mean = grouped_by_drug ["Tumor Volume (mm3)"].mean()

tumor_volume_mm3_mean

#grouped_by_drug.mean()


# In[374]:


tumor_volume_mm3_median = grouped_by_drug ["Tumor Volume (mm3)"].median()
tumor_volume_mm3_median
#grouped_by_drug.median()


# In[375]:


tumor_volume_mm3_var = grouped_by_drug ["Tumor Volume (mm3)"].var()
tumor_volume_mm3_var
#grouped_by_drug.var()


# In[376]:


tumor_volume_mm3_std = grouped_by_drug ["Tumor Volume (mm3)"].std()
tumor_volume_mm3_std
#grouped_by_drug.std()


# In[377]:


tumor_volume_mm3_sem = grouped_by_drug ["Tumor Volume (mm3)"].sem()
tumor_volume_mm3_sem
#grouped_by_drug.sem()


# In[378]:


# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen
# Using the aggregation method, produce the same summary statistics in a single line

summary_statistics_by_drug_regimen = pd.DataFrame({"Tumor Volume (mm3) Mean":tumor_volume_mm3_mean,
                                                   "Tumor Volume (mm3) Median":tumor_volume_mm3_median,
                                                   "Tumor Volume (mm3) Variance":tumor_volume_mm3_var,
                                                   "Tumor Volume (mm3) Standard deviation":tumor_volume_mm3_std,
                                                   "Tumor Volume (mm3) Standard SEM":tumor_volume_mm3_sem})


summary_statistics_by_drug_regimen


# ## Bar and Pie Charts

# In[379]:


# Generate a bar plot showing the total number of unique mice tested on each drug regimen using pandas.
cleaned_mouse_df


# In[380]:


# Generate a bar plot showing the total number of unique mice tested on each drug regimen using pandas.

unique_mice_tested  = grouped_by_drug['Drug Regimen'].count()
unique_mice_tested


# In[381]:


counts  = mouse_df['Drug Regimen'].value_counts()
counts


# In[382]:


# Generate a bar plot showing the total number of unique mice tested on each drug regimen using pyplot.
unique_mice_tested_df = pd.DataFrame({"unique mice tested": unique_mice_tested})

unique_mice_tested_df


# In[383]:


counts_df = pd.DataFrame({"mice counts": counts})

counts_df


# In[384]:


#option1: unique mice
unique_mice_tested_chart = unique_mice_tested.plot(kind="bar", title="unique mice tested by drug regimen")
unique_mice_tested_chart.set_xlabel("Drug Regimen")
unique_mice_tested_chart.set_ylabel("Number of unique mice tested")

plt.show()
plt.tight_layout()


# In[385]:


#Option2 : 
counts_chart = counts.plot(kind="bar", title=" Number of tests by drug regimen")
counts_chart.set_xlabel("Drug Regimen")
counts_chart.set_ylabel("Number of tests")

plt.show()
plt.tight_layout()


# In[386]:


##Option1: Generate a bar plot showing the total number of unique mice tested on each drug regimen using pyplot.
# Set x axis and tick locations
drugs = ["Capomulin","Ceftamin", "Infubinol", "Ketapril", "Naftisol", "Placebo", "Propriva", "Ramicane", "Stelasyn", "Zoniferol"]
x_axis = np.arange(len(unique_mice_tested_df))
tick_locations = [value for value in x_axis]

# Create a list indicating where to write x labels and set figure size to adjust for space
plt.figure(figsize=(20,3))
#plt.bar(x_axis,unique_mice_tested_df, color='r', alpha=0.5, align="center")
plt.bar(x_axis,unique_mice_tested_df["unique mice tested"], color='r', alpha=0.5, align="center")
#plt.xticks(tick_locations, grouped_by_drug["Drug Regimen"], rotation="vertical")

plt.xticks(tick_locations, drugs, rotation="vertical")


# In[387]:


##Option 2 : Generate a bar plot showing the total number of unique mice tested on each drug regimen using pyplot.
# Set x axis and tick locations
drugs = ["Capomulin","Ceftamin", "Infubinol", "Ketapril", "Naftisol", "Placebo", "Propriva", "Ramicane", "Stelasyn", "Zoniferol"]
x_axis = np.arange(len(counts_df))
tick_locations = [value for value in x_axis]

# Create a list indicating where to write x labels and set figure size to adjust for space
plt.figure(figsize=(20,3))
#plt.bar(x_axis,unique_mice_tested_df, color='r', alpha=0.5, align="center")
plt.bar(x_axis,counts_df["mice counts"], color='r', alpha=0.5, align="center")
#plt.xticks(tick_locations, grouped_by_drug["Drug Regimen"], rotation="vertical")

plt.xticks(tick_locations, drugs, rotation="vertical")


# In[388]:


# Generate a pie plot showing the distribution of female versus male mice using pandas
# gropu
grouped_by_sex = cleaned_mouse_df.groupby(["Sex"])
grouped_by_sex.count()


# In[389]:


## Generate a pie plot showing the distribution of female versus male mice using pandas


# Create a new variable that holds the count of Female and Male groups
count_it_up = grouped_by_sex.count()
count_it_up.head(12)
explode = (0.1, 0)

mice_pie = count_it_up.plot(kind="pie",y='Mouse ID', title=("unique mice tested by drug regimen "),explode=explode, colors=colors,
        autopct="%1.1f%%", shadow=True, startangle=120)
mice_pie.set_ylabel("index")

plt.show()


# In[390]:


## Generate a pie plot showing the distribution of female versus male mice using pyplot

# Labels for the sections of our pie chart
labels = ["Female", "Male"]

# The values of each section of the pie chart
sizes = [124, 125]

# The colors of each section of the pie chart
colors = ["red", "orange"]

# Tells matplotlib to seperate the "Humans" section from the others
explode = (0.1, 0)


# In[391]:


# Creates the pie chart based upon the values above
# Automatically finds the percentages of each part of the pie chart
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", shadow=True, startangle=140)


# ## Quartiles, Outliers and Boxplots

# In[392]:


# Calculate the final tumor volume of each mouse across four of the treatment regimens:  
# Capomulin, Ramicane, Infubinol, and Ceftamin

# Start by getting the last (greatest) timepoint for each mouse


# Merge this group df with the original dataframe to get the tumor volume at the last timepoint


# In[393]:


# Start by getting the last (greatest) timepoint for each mouse


last_timepoint_cleaned_mouse_df = cleaned_mouse_df.sort_values(["Mouse ID", "Timepoint"],ascending = [False, False])
last_timepoint_cleaned_mouse_df = cleaned_mouse_df.drop_duplicates(subset='Mouse ID', keep='last')


last_timepoint_cleaned_mouse_df




# In[394]:


# Merge this group df with the original dataframe to get the tumor volume at the last timepoint
only_capomulins = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Capomulin", :]
(only_capomulins["Tumor Volume (mm3)"]).describe()
capomulin = only_capomulins["Tumor Volume (mm3)"]

fig1, ax1 = plt.subplots()
ax1.set_title('The effect of Capomulin to tumor size')
ax1.set_ylabel('Tumor Volume (mm3)')
ax1.boxplot(capomulin)
plt.show()


# In[395]:


capomulin.describe()


# In[396]:


only_ramicanes = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Ramicane", :]
(only_ramicanes["Tumor Volume (mm3)"]).describe()
ramicane = only_ramicanes["Tumor Volume (mm3)"]

fig2, ax2 = plt.subplots()
ax2.set_title('The effect of Ramicane to tumor size')
ax2.set_ylabel('Tumor Volume (mm3)')
ax2.boxplot(ramicane)
plt.show()


# In[397]:


ramicane.describe()


# In[424]:


only_infubinols = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Infubinol", :]
(only_infubinols["Tumor Volume (mm3)"]).describe()
infubinol = only_infubinols["Tumor Volume (mm3)"]

fig2, ax2 = plt.subplots()
ax2.set_title('The effect of Infubinol to tumor size')
ax2.set_ylabel('Tumor Volume (mm3)')
ax2.boxplot(ramicane)
plt.show()


# In[399]:


infubinol.describe()


# In[400]:


only_ceftamins = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Ceftamin", :]
(only_ceftamins["Tumor Volume (mm3)"]).describe()
ceftamin = only_ceftamins["Tumor Volume (mm3)"]

fig2, ax2 = plt.subplots()
ax2.set_title('The effect of Ceftamin to tumor size')
ax2.set_ylabel('Tumor Volume (mm3)')
ax2.boxplot(ceftamin)
plt.show()


# In[401]:


ceftamin.describe()


# In[402]:



#plot tumor data from 4 drug regimen together on the same charge
data = [capomulin,ramicane,infubinol, ceftamin]
##labeling
fig5, ax5 = plt.subplots()
ax5.set_title('The effect of drug regimen to tumor size')
ax5.set_ylabel('Tumor Volume (mm3)')
ax5.boxplot(data)
plt.xticks([1, 2, 3,4], ['capomulin', 'ramicane', 'infubinol','ceftamin'])
plt.show()


# In[ ]:





# In[403]:


# Put treatments into a list for for loop (and later for plot labels)
treatment_types = mouse_df["Drug Regimen"].unique()
treatments = treatment_types.tolist()
treatments


# Create empty list to fill with tumor vol data (for plotting)

#tumor_vol_data = []

# Calculate the IQR and quantitatively determine if there are any potential outliers. 

    
    # Locate the rows which contain mice on each drug and get the tumor volumes
    
    
    # add subset 
    
    
    # Determine outliers using upper and lower bounds
    


# In[404]:


# Create empty list to fill with tumor vol data (for plotting)

tumor_vol_data = []
tumor_vol_data

#tumor_vol_data = []
#for x in treatments:
   # tumor_vol_data.append(x * x + np.random.randint(0, np.ceil(max(x_axis))))


# In[405]:


# Calculate the IQR and quantitatively determine if there are any potential outliers. 


# In[422]:


# capomulin-IQR-potential outliers
capomulin_quartiles = capomulin.quantile([.25,.5,.75])
lowerq = capomulin_quartiles[0.25]
upperq = capomulin_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {capomulin_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[423]:


# ramicane-IQR-potential outliers
ramicane_quartiles = ramicane.quantile([.25,.5,.75])
lowerq = ramicane_quartiles[0.25]
upperq = ramicane_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {ramicane_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[425]:


# infubinol-IQR-potential outliers
infubinol_quartiles = infubinol.quantile([.25,.5,.75])
lowerq = infubinol_quartiles[0.25]
upperq = infubinol_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {infubinol_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[426]:


# ceftamin-IQR-potential outliers
ceftamin_quartiles = ceftamin.quantile([.25,.5,.75])
lowerq = ceftamin_quartiles[0.25]
upperq = ceftamin_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {ceftamin_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[427]:


# Placebo-IQR-potential outliers
only_placebos = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Placebo", :]
(only_placebos["Tumor Volume (mm3)"]).describe()
placebo = only_placebos["Tumor Volume (mm3)"]

placebo_quartiles = placebo.quantile([.25,.5,.75])
lowerq = placebo_quartiles[0.25]
upperq = placebo_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {placebo_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[ ]:


# Stelasyn-IQR-potential outliers
only_stelasyns = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Stelasyn", :]
(only_stelasyns["Tumor Volume (mm3)"]).describe()
stelasyn = only_stelasyns["Tumor Volume (mm3)"]

stelasyn_quartiles = stelasyn.quantile([.25,.5,.75])
lowerq = stelasyn_quartiles[0.25]
upperq = stelasyn_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {stelasyn_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[433]:


# Zoniferol-IQR-potential outliers
only_zoniferols = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Zoniferol", :]
(only_zoniferols["Tumor Volume (mm3)"]).describe()
zoniferol = only_zoniferols["Tumor Volume (mm3)"]

zoniferol_quartiles = zoniferol.quantile([.25,.5,.75])
lowerq = zoniferol_quartiles[0.25]
upperq = zoniferol_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {zoniferol_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[432]:


# Ketapril-IQR-potential outliers
only_ketaprils = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Ketapril", :]
(only_ketaprils["Tumor Volume (mm3)"]).describe()
ketapril = only_ketaprils["Tumor Volume (mm3)"]

ketapril_quartiles = ketapril.quantile([.25,.5,.75])
lowerq = ketapril_quartiles[0.25]
upperq = ketapril_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {ketapril_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[430]:


# Propriva-IQR-potential outliers
only_proprivas = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Propriva", :]
(only_proprivas["Tumor Volume (mm3)"]).describe()
propriva = only_proprivas["Tumor Volume (mm3)"]

propriva_quartiles = propriva.quantile([.25,.5,.75])
lowerq = propriva_quartiles[0.25]
upperq = propriva_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {propriva_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[429]:


# Naftisol-IQR-potential outliers
only_naftisols = last_timepoint_cleaned_mouse_df.loc[last_timepoint_cleaned_mouse_df["Drug Regimen"] == "Naftisol", :]
(only_naftisols["Tumor Volume (mm3)"]).describe()
naftisol = only_naftisols["Tumor Volume (mm3)"]

naftisol_quartiles = naftisol.quantile([.25,.5,.75])
lowerq = naftisol_quartiles[0.25]
upperq = naftisol_quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor_vol_data is: {lowerq}")
print(f"The upper quartile of tumor_vol_data is: {upperq}")
print(f"The interquartile range of tumor_vol_data is: {iqr}")
print(f"The the median of tumor_vol_data is: {naftisol_quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


# In[420]:


# Generate a box plot of the final tumor volume of each mouse across four regimens of interest

#plot tumor data from 4 drug regimen together on the same charge
data = [capomulin,ramicane,infubinol, ceftamin]
##labeling
fig5, ax5 = plt.subplots()
ax5.set_title('The effect of drug regimen to tumor size')
ax5.set_ylabel('Tumor Volume (mm3)')
ax5.boxplot(data)
plt.xticks([1, 2, 3,4], ['capomulin', 'ramicane', 'infubinol','ceftamin'])
plt.show()


# ## Line and Scatter Plots

# In[407]:


# Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin
only_capomulins

x_axis = only_capomulins["Tumor Volume (mm3)"]
y_axis = only_capomulins["Timepoint"]

plt.title("Tumor volume vs. time point for a mouse treated with Capomulin")
plt.xlabel("Tumor Volume (mm3)")
plt.ylabel("Timepoint")

plt.plot(X_axis, y_axis,  marker="s", color="Red", linewidth=1, label="Capomulin" )
plt.show()


# In[408]:


# Generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen
only_capomulins

#grouped_by_drug = cleaned_mouse_df.groupby(["Drug Regimen"])
capomulin_data_groupby_id = only_capomulins.groupby(["Mouse ID"])
capomulin_data_groupby_id_mean = capomulin_data_groupby_id.mean()
capomulin_data_groupby_id_mean
#df.rename(columns={"A": "a", "B": "c"})
#capomulin_data_groupby_id_mean.rename(columns={"Age_months": "Average Age_months", "Weight (g)": "Average Weight (g)",
                                              # "Timepoint": "Average Timepoint", "Tumor Volume (mm3)": "Average Tumor Volume (mm3)",
                                              # "Metastatic Sites": "Average Metastatic Sites", })


# In[409]:


capomulin_data_groupby_id_mean_rename = capomulin_data_groupby_id_mean.rename(columns={"Age_months": "Average Age_months", "Weight (g)": "Average Weight (g)",
                                               "Timepoint": "Average Timepoint", "Tumor Volume (mm3)": "Average Tumor Volume (mm3)",
                                               "Metastatic Sites": "Average Metastatic Sites", })
capomulin_data_groupby_id_mean_rename


# In[410]:



x_axis = capomulin_data_groupby_id_mean_rename["Average Weight (g)"]
y_axis = capomulin_data_groupby_id_mean_rename["Average Tumor Volume (mm3)"]

plt.title("Average tumor volume vs. mouse weight for the Capomulin regimen")
plt.xlabel("Average Tumor Volume (mm3)")
plt.ylabel("Average Weight (g)")

plt.scatter(x_axis, y_axis, marker="o", facecolors="red", edgecolors="black",
           s=x_axis, alpha=0.75, label="Capomulin" )
plt.show()


# ## Correlation and Regression

# In[411]:


# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen


# In[419]:


#y = mx + c
x_axis = capomulin_data_groupby_id_mean_rename["Average Weight (g)"]
y_axis = capomulin_data_groupby_id_mean_rename["Average Tumor Volume (mm3)"]

(slope, intercept, rvalue, pvalue, stderr) = linregress(x_axis, y_axis)

line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))

regress_values = x_axis * slope + intercept


plt.scatter(x_axis,y_axis)
plt.plot(x_axis,regress_values,"r-")

plt.annotate(line_eq,(20,30),fontsize=15,color="red")


plt.title("Average tumor volume vs. mouse weight for the Capomulin regimen")
plt.xlabel("Average Tumor Volume (mm3)")
plt.ylabel("Average Weight (g)")

print(f"The r-squared is: {rvalue**2}")
plt.show()


# In[ ]:





# In[ ]:




