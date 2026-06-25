# Hospital-Readmission-Risk-Segmentation
Engineered a K-means clustering pipeline in Python to segment 25,000 hospital patients by readmission risk — applying StandardScaler normalization, elbow method optimization, and SciPy/scikit-learn clustering to identify high, medium, and low-risk patient profiles based on lab procedure and medication utilization patterns.

## Description 
The United States healthcare system combines public and private, for-profit and nonprofit insurers and healthcare providers. According to KFF, healthcare costs in the United States generally grow faster than inflation and represent a much larger share of the economy in the U.S. than in peer nations. Despite this substantial spending, elevate healthcare expenditure in the U.S. does not consistently translate into superior health outcomes. Rising health care costs instead contribute to Americans facing difficulty affording medical care and drugs, even among those with insurance.

Over the years, multiple efforts have been made by the federal government to reduce one of the contributors of rising healthcare costs--hospital readmissions rates. Readmission rates contribute to high healthcare costs by generating an estimated $15-$20 billion in annual spending due to added resource use and penalties incurred under federal programs. For example, the Hospital Readmission Reduction Program (HRRP) was designed as a Medicare value-based purchasing program that decreases payments to hospitals that have disproportionately high readmissions.

According to policymakers, hospitals have two incentives to decrease readmission rates: 

1. Transparency through public reporting provides an incentive to reduce readmission rates, as hospitals with high readmission rates might deter future patients from choosing them. Reputation for quality has been discussed as a driver for profits through its effect on increased market share and hospitals have the incentive to decrease readmission rates to avoid developing a bad reputation.
2. Hospitals get penalized under the CMS HRRP for having excess readmission rates. According to a study, readmission penalties in 2017 exceed half a billion dollars alone.

The question is then, what characteristics of a patient indicate whether they have a high-likelihood of being readmitted?

The HRRP program mentioned above focuses on 30-day readmissions for Acute Myocardial Infarction (AMI), Chronic Obstructive Pulmonary Disease (COPD), Heart Failure (HF), Pneumonia, Coronary Artery Bypass Graft (CABG) surgery, and Elective Primary Total Hip/Knee Arthroplasty (THA/TKA).

Our question for today is: Are there are other factors about a patient that hospitals can consider when assessing the likelihood of a patient being readmitted?

K-means clustering is an unsupervised machine learning technique. We can use this analysis tool to segment patients into distinct groups based on similar characteristics--such as age, comorbidities, length of hospital stay, and medication use--to identify those at high risk for hospital readmission.

The following steps were taken within this analysis: 
1. Import relevant packages.
2. Explore the data, and identify potential continuous variables that we can use for this analysis .
3. Evaluate the mean and standard deviation of each continouous variable.
4. Produce a scatterplot to visualize the data (figure 1). 
5. Standardize the dataset of our selected variable to perform the K-means analysis.
6. Produce an elbow plot to determine how many K-clusters should be selected (figure 2).
7. Run the k-means cluster analysis on the selected variables with the selected number of clusters.
8. Generate the scatterplot to visualize the k-means analysis (figure 3).

## Conclusions
**Feature selection**: out of 7 continuous variables, lab procedure count and medication count were chosen because they had the highest standard deviation (spread), meaning they carry the most discriminating power for clustering.
**Cluster Selection**: K=3 was selected based on the elbow plot below (figure 2) which showed diminishing WCSS returns after 3 clusters, and 3 maps intuitively to a high/medium/low risk framework relevant to clinical decision-making.

**Clinical Implication**: patients with more complex care needs during a hospital stay (more tests, more medications) cluster together and may warrant targeted discharge planning or follow-up to reduce readmission likelihood.

## Getting Started 

### Dependencies
Dataset was downloaded from the following site: https://www.kaggle.com/datasets/dubradave/hospital-readmissions?resource=download 

Use "Hospital_Readmissions_Kmeans.py" to see data cleaning and analysis.

**Relevant Packages:** 
import seaborn as sns
import pandas as pd 
from scipy.cluster import vq
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

Modify file path names for uploading the dataset and output files. Install sklearn in your terminal using:
pip install scikit-learn

### Useful Resources
Sources for information about hospital readmission
- https://www.kaggle.com/datasets/dubradave/hospital-readmissions?resource=download 
 https://www.commonwealthfund.org/international-health-policy-center/countries/united-states
- https://www.kff.org/health-costs/health-policy-101-health-care-costs-and-affordability/?entry=table-of-contents-how-has-u-s-health-care-spending-changed-over-time
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6614936/

