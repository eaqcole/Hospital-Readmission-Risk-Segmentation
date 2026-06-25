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

## Getting Started

## Useful Resources

