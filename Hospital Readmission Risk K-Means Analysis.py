''' Hospital Readmission Risk Segmentation (K-Means)
Date: April 2026
Name: Emily Quick-Cole

Background: The United States healthcare system combines public and private, for-profit and nonprofit insurers and healthcare
providers. According to KFF, healthcare costs in the United States generally grow faster than inflation and represent
a much larger share of the economy in the U.S. than in peer nations. Despite this substantial spending, elevate
healthcare expenditure in the U.S. does not consistently translate into superior health outcomes. Rising health care
costs instead contribute to Americans facing difficulty affording medical care and drugs, even among those with
insurance.

Over the years, multiple efforts have been made by the federal government to reduce one of the contributors of
rising healthcare costs--hospital readmissions rates. Readmission rates contribute to high healthcare costs by
generating an estimated $15-$20 billion in annual spending due to added resource use and penalties incurred under
federal programs. For example, the Hospital Readmission Reduction Program (HRRP) was designed as a Medicare
value-based purchasing program that decreases payments to hospitals that have disproportionately high readmissions.

According to policymakers, hospitals have two incentives to decrease readmission rates:

Transparency through public reporting provides an incentive to reduce readmission rates, as hospitals with high
readmission rates might deter future patients from choosing them. Reputation for quality has been discussed as a
driver for profits through its effect on increased market share and hospitals have the incentive to decrease
readmission rates to avoid developing a bad reputation.
Hospitals get penalized under the CMS HRRP for having excess readmission rates. According to a study, readmission
penalties in 2017 exceed half a billion dollars alone.
The question is then, what characteristics of a patient indicate whether they have a high-likelihood of being readmitted?

The HRRP program mentioned above focuses on 30-day readmissions for Acute Myocardial Infarction (AMI), Chronic
Obstructive Pulmonary Disease (COPD), Heart Failure (HF), Pneumonia, Coronary Artery Bypass Graft (CABG) surgery,
and Elective Primary Total Hip/Knee Arthroplasty (THA/TKA).

Key Question: Are there are other factors about a patient that hospitals can consider when assessing the
likelihood of a patient being readmitted?

Analytical Model: K-means clustering is an unsupervised machine learning technique. We can use this analysis
tool to segment patients into distinct groups based on similar characteristics--such as age, comorbidities,
length of hospital stay, and medication use--to identify those at high risk for hospital readmission.

Data Source: https://www.kaggle.com/datasets/dubradave/hospital-readmissions?resource=download

Date of Data Download: April 2026

Results:
Feature selection: out of 7 continuous variables, lab procedure count and medication count were chosen because they
had the highest standard deviation (spread), meaning they carry the most discriminating power for clustering.

Cluster Selection: K=3 was selected based on the elbow plot below (figure 2) which showed diminishing WCSS returns
after 3 clusters, and 3 maps intuitively to a high/medium/low risk framework relevant to clinical decision-making.

Clinical Implication: patients with more complex care needs during a hospital stay (more tests, more medications)
cluster together and may warrant targeted discharge planning or follow-up to reduce readmission likelihood.
'''
### Global imports
import seaborn as sns
import pandas as pd
from scipy.cluster import vq
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import boto3
from io import StringIO


# We'll be using a dataset from Kaggle, which we've loaded and displayed in this notebook using the codeblock below.
# Each row represents a patient who stayed in a hospital.

#Load and display the data
#data_df = pd.read_csv('hospital_readmissions.csv')
#data_df.head()

#Alternate: Load and display the data from Amazon S3
#Create S3 client
s3 = boto3.client('s3')

#Get the file
response = s3.get_object(
    Bucket = 'emily-cloud-practice',
    Key = 'hospital_readmissions.csv'
)

#Read into pandas
data_df = pd.read_csv(response['Body'])
print(data_df.head)


# As you can see, our dataset has numerous variables that we can use to conduct a K-means clustering analysis and
# predict whether a patient will be at high or low-risk for readmission.
# 
# 1. "age" - age bracket of the patient
# 2. "time_in_hospital" - days (from 1 to 14)
# 3. "n_procedures" - number of procedures performed during the hospital stay
# 4. "n_lab_procedures" - number of laboratory procedures performed during the hospital stay
# 5. "n_medications" - number of medications administered during the hospital stay
# 6. "n_outpatient" - number of outpatient visits in the year before a hospital stay
# 7. "n_inpatient" - number of inpatient visits in the year before the hospital stay
# 8. "n_emergency" - number of visits to the emergency room in the year before the hospital stay
# 9. "medical_specialty" - the specialty of the admitting physician
# 10. "diag_1" - primary diagnosis (Circulatory, Respiratory, Digestive, etc.)
# 11. "diag_2" - secondary diagnosis
# 12. "diag_3" - additional secondary diagnosis
# 13. "glucose_test" - whether the glucose serum came out as high (> 200), normal, or not performed
# 14. "A1Ctest" - whether the A1C level of the patient came out as high (> 7%), normal, or not performed
# 15. "change" - whether there was a change in the diabetes medication ('yes' or 'no')
# 16. "diabetes_med" - whether a diabetes medication was prescribed ('yes' or 'no')
# 17. "readmitted" - if the patient was readmitted at the hospital ('yes' or 'no')
# 
# 
# We will focus on the following continuous variables to determine whether they can point us three distinct groups:
# - time in hospital,
# - n_lab_procedures,
# - n_procedures, 
# - n_medications, 
# - n_outpatient,
# - n_inpatient,
# - n_emergency.

# # Data Processing and Exploring

# First, we need to clean up the initial data so that we can visualize it. Let's create a new dataframe that only
# includes the continuous variables of interest.

# ## Subsetting to Continuous Variables

#Subset the dataframe to keep only continuous variables
df = data_df.copy(deep = True)
cont_df = df[['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']]


# ## Renaming the Columns

# Now, let's rename the columns so that they're more intuitive.

#Store the current column names in a list
old_names = cont_df.columns

#Create a list of new column names
new_names = ['total_days', 'lab_procedure_count', 'procedure_count', 'med_count', 'op_visits', 'ip_visits', 'ed_visits']

#Use the rename function and zip the old_names and new_names lists together to reassign column names
cont_df = cont_df.rename(columns = dict(zip(old_names,new_names)))

#Ensure all datatypes are floats 
for col in cont_df.columns:
    cont_df[col] = cont_df[col].astype(float)

#Display the head of the dataframe with renamed columns
cont_df.head()



# ## Exploratory Analysis

# Let's take a deeper dive into our data by counting the rows, presence of NaN values, and using descriptive statistics. 



print("=== Health Data Info ===")
print(f"\nThere are", cont_df.shape[0], "rows and", cont_df.shape[-1], "columns in this dataset.")
print(f"\nThis is how many Nan values are in our dataset:\n", cont_df.isna().sum())

print("\n=== Health Data Statistics ===")
print(round(cont_df.describe(), 2))


# From this cleaning and exploratory data analysis we can conclude: 
# - We have 25,000 rows in our dataset, and each variable has a count of 25,000 (This, along with 0 NaN values
# confirms all our columns and rows have a value).
# - The means of these variables range from 0.19 (ed_visits) to 43.24 (lab_procedure_count). Generally speaking, the
# means for each of these variables are distinct.
# - We can see from the standard deviation row that lab_procedure_count and med_count have the highest standard
# deviation (or spread) and median values of 44 and 15, respectively. We'll continue our analysis using these variables.



#Create a copy of the dataframe with continuous variables and subset it to only include lab procedure count and
#med_count
km_df = cont_df[['lab_procedure_count', 'med_count']].copy(deep = True)
km_df.head()

#This will be the dataframe we continue to work with, write back to Amazon S3 and save this file
# Convert to CSV string — this line defines csv_buffer
csv_buffer = StringIO()
km_df.to_csv(csv_buffer, index=False)

#Write back to s3
s3.put_object(
    Bucket = 'emily-cloud-practice',
    Key = 'outputs/hospital_readmissions_for_analysis.csv',
    Body = csv_buffer.getvalue() #df.to_csv(index = False)
)


# ## Visualize the Data

# Now that we know what columns we'd like to work with, let's take a moment to visualize our data using a scatterplot.

# In[9]:


#1. Set the style for the plots
sns.set(style="darkgrid")
sns.set_palette('bright')
plt.figure(figsize = (4,3))

#2. Generate the scatterplot
sns.scatterplot(data = km_df, x = 'lab_procedure_count', y = 'med_count')
    
#3. Set the title and labels
plt.title(f'Figure 1: Scatterplot of Lab Procedure Count and Medication Count', weight = 'bold')
plt.xlabel('Lab Procedure Count')
plt.ylabel('Medication Count')
    
#4. Show the plot grid
plt.grid(True)

#5. Display the plot
#plt.show()
fig1_path = '/Users/emilyquick-cole/Documents/Python/kmeansscatterplot.png'
plt.savefig(fig1_path, bbox_inches="tight")
plt.close()


# ## Scaling the Data

# Now we need to standardize our dataset. Standardizing values for K-means clustering is essential because the
# algorithm uses Euclidean distance to measure similarity. Features with larger numerical ranges will disproportionately
# influence the distance calculation, effectively ignoring smaller-scale features. Once we create these standardized
# columns, we'll add them to our dataframe.


# 1. Initialize the scaler
scaler = StandardScaler()

# 2. Fit and transform the numerical columns
# Standardizes values to mean=0 and std_dev=1T
km_df[['lab_procedure_count_T', 'med_count_T']] = scaler.fit_transform(km_df[['lab_procedure_count', 'med_count']])
km_df.head()


# ## Determining How Many K-Clusters

# To determine how many K-clusters we should select, we can use the Elbow Method. The Elbow Method helps us
# determine how many clusters we should select by plotting the Within-Cluster Sum of Squares (WCSS) against
# increasing k values and looking for a point where the improvement slows down, this point is called the "elbow."

#Let's write a function that will take our data and produce an elbow plot
def optimise_k_means(data, max_k):
    means = []
    inertias = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(data)
        
        means.append(k)
        inertias.append(kmeans.inertia_)
    #Generate the elbow plot
    fig = plt.subplots(figsize = (10,5))
    plt.plot(means, inertias, 'o-')
    plt.title(f'Figure 2: Elbow Plot to Determine Cluster Count', weight = 'bold')
    plt.xlabel('Lab Procedure Count')
    plt.ylabel('Medication Count')
    plt.text(3, 25000, "Use K=3 to create high-, low-, and medium-risk clusters.", color = 'red')
    plt.grid(True)
    #plt.show()
    fig2_path = '/Users/emilyquick-cole/Documents/Python/elbowplot.png'
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()


# Now run the function using the scaled values of lab_procedure_count and med_count.

optimise_k_means(km_df[['lab_procedure_count_T', 'med_count_T']], 10)


# From the plot above, we can see that the "elbow" occurs around cluster count 3 and 4. Because we're interested in
# categorizing and determining who is at high-risk for readmission, we'll continue our analysis using k=3 to create
# high, medium, and low risk category.

# # Performing the K-Means Clustering Analysis

# Assign variables to hold the lab_procedure_count_T, med_count_T points, the dimensions of the data, and our
# target number of clusters.

#Assign variables to hold our points, dimensions of our data, and the number of clusters we want
points = km_df[['lab_procedure_count_T', 'med_count_T']].values
n, d = points.shape
k = 3


# Now run the k-means clustering algorithm.

# `distortion` below is the similar to WCSS.
# It is called distortion in the Scipy documentation
# since clustering can be used in compression.
centers_vq, distortion_vq = vq.kmeans(points, k)

# vq return the clustering (assignment of group for each point)
# based on the centers obtained by the kmeans function.
# _ here means ignore the second return value
clustering_vq, _ = vq.vq(points, centers_vq)

#assign the clustering labels to our dataframe as a new column
km_df['clustering_vq'] = clustering_vq


# Now let's visualize the scatterplot, color-coding by the k-means generated cluster labels.

#Create the plot size
plt.figure(figsize = (16,10))

#Generate the scatterplot
# Set a color palette
#sns.set_palette(clustering_vq)
sns.scatterplot(data = km_df, x = 'lab_procedure_count', y = 'med_count', hue = 'clustering_vq')

    
#Set the title and labels
plt.title(f'Figure 3: Scatterplot of Lab Procedure Count and Medication Count with K-Means Cluster Labels', weight = 'bold', fontsize = 16)
plt.xlabel('Lab Procedure Count')
plt.ylabel('Medication Count')
plt.legend(fontsize=20, markerscale=2.0)
    
#Show the plot grid
plt.grid(True)
    
#Display the plot
fig3_path = '/Users/emilyquick-cole/Documents/Python/Kmeans.png'
plt.savefig(fig3_path, bbox_inches="tight")
plt.close()


# # Conclusions

# From the figure above, we can see that we have three clearly defined categories of patients based on the number of
# medications they received during their hospital stay and their total number of lab procedures. Patients who received
# about 20 or more medications and lab procedures appear to have similarities.

# # Sources for information about hospital readmission

# - Source: https://www.kaggle.com/datasets/dubradave/hospital-readmissions?resource=download 
# https://www.commonwealthfund.org/international-health-policy-center/countries/united-states
# - https://www.kff.org/health-costs/health-policy-101-health-care-costs-and-affordability/?entry=table-of-contents-how-has-u-s-health-care-spending-changed-over-time
# - https://pmc.ncbi.nlm.nih.gov/articles/PMC6614936/
