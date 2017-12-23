'''
Created on 1 Apr 2017

@author: Gary, Terence, Ho Fai, Benjamin
'''

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

# function to calculate true positives and true negatives

def truevalues(prediction, actual):
    count = 0 # initialise the count for correct predictions
    if len(prediction) != len(actual): # checks whether the two input lists are of the same length
        return "Error"
    for i in range(len(prediction)): 
        if prediction[i] == actual[i]:
            count+=1 # count increases if the prediction matches the actual diagnosis
    return count # returns the total number of correct predictions



# accuracy function

def accuracy(true_count, total_observations):
    return float(true_count)/total_observations # returns accuracy as a float value

# precision function

def precision(prediction, actual):
    true_positive_count = 0
    false_positive_count = 0
    
    if len(prediction) != len(actual): # checks whether the two input lists are of the same length
        return "Error"
    for i in range(len(prediction)): 
        if prediction[i] == 1 and actual[i]==1: # check for true positives
            true_positive_count+=1 
        elif prediction[i] == 1 and actual[i] == 0: # check for false positives
            false_positive_count +=1
    return float(true_positive_count)/(true_positive_count+false_positive_count)

# recall function

def recall(prediction, actual):
    true_positive_count = 0
    false_negative_count = 0
    
    if len(prediction) != len(actual): # checks whether the two input lists are of the same length
        return "Error"
    for i in range(len(prediction)): 
        if prediction[i] == 1 and actual[i]==1: # check for true positives
            true_positive_count+=1 
        elif prediction[i] == 0 and actual[i] == 1: # check for false negatives
            false_negative_count +=1
    return float(true_positive_count)/(true_positive_count+false_negative_count)

# f1 function

def f1(precision, recall):
    return 2*((precision*recall)/(precision+recall))

# specificity function

def specificity(prediction, actual):
    
    true_negative_count = 0
    false_positive_count = 0
    
    if len(prediction) != len(actual): # checks whether the two input lists are of the same length
        return "Error"
    for i in range(len(prediction)): 
        if prediction[i] == 0 and actual[i]==0: # check for true negatives
            true_negative_count+=1 
        elif prediction[i] == 1 and actual[i] == 0: # check for false positives
            false_positive_count +=1
    return float(true_negative_count)/(true_negative_count+false_positive_count)

# reads dataset into a dataframe

path = 'c:\\sem8\pdpa_data.csv'

pdpa_data = pd.read_csv(path, skiprows=1, \
                        names = ['index', 'url', 'new_website', 'sg_domain',\
                                'dpo', 'purpose', 'dpp1', 'dpp2', \
                                'data_type_collected', 'amendments_notification',\
                                'existence', 'sale', 'access', 'retention', \
                                'location', 'last_amended', 'operating_security', \
                                'tech_security', '3P_sharing', '3P_collect', \
                                '3P_collect_own', 'comprehensiveness', \
                                'total_global_visits', 'total_sg_visits', \
                                'percentage_sg_visits', 'SG_office', 'listed',\
                                'year_last_amended', 'lifestyle', 'travel',\
                                'fashion', 'beauty', 'IT', 'gifts', 'food', \
                                'financial_services', 'others', 'DPP_length', \
                                'cookies', 'https', 'EU_law', 'pdpa', 'other_law',\
                                'clicks', 'sections', 'simplified', 'flesch1', 'flesch2', \
                                'notice_account', 'notice_transaction', 'contact_info', 'com_info', \
                                'financial_info', 'interactive_info', 'content_info',\
                                'sensitive_info', '3P_identify', '3P_dpp', 'last_amended_raw', 'no_of_days_last_amended']) # skips first row of headers in CSV





def log_it(x):  # log function
    return np.log(x)

def divide_by_hundred(x): # division by 100 function
    return x/100.0

pdpa_data['dpo'].replace(to_replace=[2, 1], value=[1, 0], inplace=True)  

pdpa_data['dpp1'].replace(to_replace=[2, 1], value=[1, 0], inplace=True)  



# interruption: graph of global traffic against dpp before global traffic is manipulated (slide 28)


graph4 = pdpa_data.plot(x='total_global_visits', y='dpp1', kind='scatter') # plots total global traffic against presence of DPP
plt.ylabel('Whether there is a DPP') # label of y-axis
plt.xlabel('Amount of global traffic') # label of x-axis
graph4.set_yticks([0,1])
graph4.set_yticklabels(['No', 'Yes'])
plt.show() # shows the graph




# 

pdpa_data['dpp2'].replace(to_replace=[2, 1], value=[1, 0], inplace=True)  

pdpa_data['purpose'].replace(to_replace=[2, 1], value=[1, 0], inplace=True)  

pdpa_data['total_global_visits'] = pdpa_data['total_global_visits'].map(log_it)  

pdpa_data['total_sg_visits'] = pdpa_data['total_sg_visits'].map(log_it) 

pdpa_data['percentage_sg_visits'] = pdpa_data['percentage_sg_visits'].map(divide_by_hundred)


pdpa_data['legal_requirements'] = np.nan

for index, rows in pdpa_data.iterrows():
    
    if rows['dpp1'] == 1 and rows['dpo'] == 1 and rows['purpose'] == 1:
        pdpa_data.set_value(index, 'legal_requirements', 1)
    
    else:
        pdpa_data.set_value(index, 'legal_requirements', 0)

pdpa_data2 = (pdpa_data.loc[pdpa_data['dpp1'] == 1]).copy(deep=True) # creates a new dataframe of websites with a DPP (website or email); copy needed to avoid errors empty values
pdpa_data2.reset_index(inplace=True) # resets the index of the new dataframe

pdpa_data2['DPP_length'] = pdpa_data2['DPP_length'].map(log_it)
pdpa_data2['year_last_amended'] = pdpa_data2['year_last_amended'].astype('category')


plt.style.use('seaborn-deep') # choice of colour for graphs

# visitorship variables: total_global_visits, total_sg_visits, percentage_sg_visits


print pdpa_data.describe()['total_global_visits']
print pdpa_data.describe()['total_sg_visits']
print pdpa_data.describe()['percentage_sg_visits']


# graph of total_global_visits
#SLIDE 16 histogram for total_global_visits


plt.hist(pdpa_data['total_global_visits'])
plt.title('Total visits from Global Visitors over three months')
plt.xlabel('log(number of visits)')
plt.ylabel('Number of Organisations')
plt.show()


# graph of total_sg_visits
#SLIDE 17 histogram for total_sg_visits


plt.hist(pdpa_data['total_sg_visits'])
plt.title('Total visits from Singapore over three months')
plt.xlabel('log(number of visits)')
plt.ylabel('Number of Organisations')
plt.show()


# graph of percentage_sg_visits
#SLIDE 18 histogram for percentage_sg_visits


plt.hist(pdpa_data['percentage_sg_visits'])
plt.title('Percentage of visits from Singapore over three months')
plt.xlabel('(% of SG visits)/100')
plt.ylabel('Number of Organisations')
plt.show()


# graph of total length of DPP
#SLIDE 19 histogram for DPP_length


plt.hist(pdpa_data2['DPP_length'])
plt.title('Total Length of the DPP')
plt.xlabel('Length of DPP')
plt.ylabel('Number of Organisations')
plt.show()


# graphs of websites with DPP, DPO and purpose


graph_legal_req = pdpa_data['legal_requirements'].value_counts().sort_index().plot(kind='bar') # plots value counts of legal requirements
plt.ylabel('Number of observations') # labels y-axis
plt.xlabel('Whether the website complies with the requirements of PDPA') # labels x-axis
graph_legal_req.set_xticklabels(['No: 209', 'Yes: 88'], rotation = 0) # values for x-axis

plt.show()

graph_dpp = pdpa_data['dpp1'].value_counts().sort_index().plot(kind='bar') # plots value counts of dpp1
plt.ylabel('Number of observations') # labels y-axis
plt.xlabel('Whether the website has a DPP') # labels x-axis
graph_dpp.set_xticklabels(['No: 29', 'Yes: 268'], rotation=0) # values for x-axis

plt.show()

graph_dpo = pdpa_data['dpo'].value_counts().sort_index().plot(kind='bar') # plots value counts of dpp1
plt.ylabel('Number of observations') # labels y-axis
plt.xlabel('Whether the website states purpose of data collection') # labels x-axis
graph_dpo.set_xticklabels(['No: 206', 'Yes: 91'], rotation = 0) # values for x-axis

plt.show()

graph_purpose = pdpa_data['purpose'].value_counts().sort_index().plot(kind='bar') # plots value counts of dpp1
plt.ylabel('Number of observations') # labels y-axis
plt.xlabel('Whether the website states purpose of data collection') # labels x-axis
graph_purpose.set_xticklabels(['No: 37', 'Yes: 260'], rotation = 0) # values for x-axis

plt.show()


#histogram for legal requirements


graph0 = pdpa_data['legal_requirements'].value_counts(sort=False).plot(kind='bar')
plt.ylabel('Number of observations')
plt.xlabel('Whether the website complies with the requirements of PDPA')
graph0.set_xticklabels(['No: 209', 'Yes: 88'], rotation = 0)

plt.show()




# value counts for table of SG_office, sg_domain, pdpa vs. legal_requirements, dpp1, dpo, purpose



print (pdpa_data[pdpa_data['SG_office'] == 2])['legal_requirements'].value_counts()
print (pdpa_data[pdpa_data['SG_office'] == 2])['dpp1'].value_counts()
print (pdpa_data[pdpa_data['SG_office'] == 2])['dpo'].value_counts()
print (pdpa_data[pdpa_data['SG_office'] == 2])['purpose'].value_counts()

print (pdpa_data[pdpa_data['sg_domain'] == 1])['legal_requirements'].value_counts()
print (pdpa_data[pdpa_data['sg_domain'] == 1])['dpp1'].value_counts()
print (pdpa_data[pdpa_data['sg_domain'] == 1])['dpo'].value_counts()
print (pdpa_data[pdpa_data['sg_domain'] == 1])['purpose'].value_counts()

print (pdpa_data2[pdpa_data2['pdpa'] == 2])['legal_requirements'].value_counts()
print (pdpa_data2[pdpa_data2['pdpa'] == 2])['dpp1'].value_counts()
print (pdpa_data2[pdpa_data2['pdpa'] == 2])['dpo'].value_counts()
print (pdpa_data2[pdpa_data2['pdpa'] == 2])['purpose'].value_counts()





# logit model for SG websites vs. legal requirements


logitmodel_legal_requirements_sg = smf.logit(formula = 'legal_requirements ~ total_sg_visits + SG_office + percentage_sg_visits + sg_domain \
                                        + pdpa', data=pdpa_data2).fit() # pdpa_data2 used because websites without DPP do not have a value for pdpa

print logitmodel_legal_requirements_sg.summary()

pdpa_data2['legal_requirements_predictions'] = np.nan # creates empty column to store predictions

for index, rows in pdpa_data2.iterrows(): # iterates through the pdpa_data2 DataFrame
    x1 = rows['total_sg_visits']
    x2 = rows['SG_office']
    x3 = rows['pdpa']
    x4 = rows['percentage_sg_visits']
    x6 = rows['sg_domain']


    prediction = {'total_sg_visits': x1, 'SG_office': x2, 'pdpa': x3, 'percentage_sg_visits': x4, \
                   'sg_domain': x6} # creates a prediction dictionary using the stored values for the current row
    predicted_result = logitmodel_legal_requirements_sg.predict(prediction) # stores the raw result of the prediction 
    
    if predicted_result < 0.5: # determines if the prediction is 0 or 1 and stores the value in the column
        pdpa_data2.set_value(index, 'legal_requirements_predictions', 0)
    elif predicted_result >= 0.5:
        pdpa_data2.set_value(index, 'legal_requirements_predictions', 1)
        
print ("Accuracy of model: ").rjust(19) + str(accuracy((truevalues(pdpa_data2['legal_requirements_predictions'], \
                            pdpa_data2['legal_requirements'])), len(pdpa_data2['legal_requirements_predictions']))) # prints the accuracy of the logit model
a = precision(pdpa_data2['legal_requirements_predictions'], pdpa_data2['legal_requirements'])
b = recall(pdpa_data2['legal_requirements_predictions'], pdpa_data2['legal_requirements'])
print ("Precision: ").rjust(19) + str(a)
print ("Recall: ").rjust(19) + str(b)
print ("F1: ").rjust(19) +str(f1(a,b))
print ("Specificity: ").rjust(19) + str(specificity(pdpa_data2['legal_requirements_predictions'], pdpa_data2['legal_requirements']))




# logit model for SG websites vs. DPO


logitmodel_pdo_sg = smf.logit(formula = 'dpo ~ total_sg_visits + SG_office + percentage_sg_visits + sg_domain \
                                        + pdpa', data=pdpa_data2).fit() # pdpa_data2 used because websites without DPP do not have a value for pdpa
                                        
print logitmodel_pdo_sg.summary()

pdpa_data2['dpo_predictions'] = np.nan # creates empty column to store predictions

for index, rows in pdpa_data2.iterrows(): # iterates through the pdpa_data2 DataFrame
    x1 = rows['total_sg_visits']
    x2 = rows['SG_office']
    x3 = rows['pdpa']
    x4 = rows['percentage_sg_visits']
    x6 = rows['sg_domain']
 
    prediction = {'total_sg_visits': x1, 'SG_office': x2, 'pdpa': x3, 'percentage_sg_visits': x4, \
                   'sg_domain': x6} # creates a prediction dictionary using the stored values for the current row
    
    predicted_result = logitmodel_pdo_sg.predict(prediction) # stores the raw result of the prediction 
    
    if predicted_result < 0.5: # determines if the prediction is 0 or 1 and stores the value in the column
        pdpa_data2.set_value(index, 'dpo_predictions', 0)
    elif predicted_result >= 0.5:
        pdpa_data2.set_value(index, 'dpo_predictions', 1)
        
print ("Accuracy of model: ").rjust(19) + str(accuracy((truevalues(pdpa_data2['dpo_predictions'], pdpa_data2['dpo'])), \
                                        len(pdpa_data2['dpo_predictions']))) # prints the accuracy of the logit model

a = precision(pdpa_data2['dpo_predictions'], pdpa_data2['dpo'])
b = recall(pdpa_data2['dpo_predictions'], pdpa_data2['dpo'])
print ("Precision: ").rjust(19) + str(a)
print ("Recall: ").rjust(19) + str(b)
print ("F1: ").rjust(19) +str(f1(a,b))
print ("Specificity: ").rjust(19) + str(specificity(pdpa_data2['dpo_predictions'], pdpa_data2['dpo']))




# logit model for SG websites vs. purpose


logitmodel_purpose_sg = smf.logit(formula = 'purpose ~ total_sg_visits + SG_office + percentage_sg_visits + sg_domain \
                                        + pdpa', data=pdpa_data2).fit() # pdpa_data2 used because websites without DPP do not have a value for pdpa
                                                            
print logitmodel_purpose_sg.summary()

pdpa_data2['purpose_predictions'] = np.nan # creates empty column to store predictions

for index, rows in pdpa_data2.iterrows(): # iterates through the pdpa_data2 DataFrame
    x1 = rows['total_sg_visits']
    x2 = rows['SG_office']
    x3 = rows['pdpa']
    x4 = rows['percentage_sg_visits']
    x6 = rows['sg_domain']
 
    prediction = {'total_sg_visits': x1, 'SG_office': x2, 'pdpa': x3, 'percentage_sg_visits': x4, \
                   'sg_domain': x6} # creates a prediction dictionary using the stored values for the current row
    
    predicted_result = logitmodel_purpose_sg.predict(prediction) # stores the raw result of the prediction 
    
    if predicted_result < 0.5: # determines if the prediction is 0 or 1 and stores the value in the column
        pdpa_data2.set_value(index, 'purpose_predictions', 0)
    elif predicted_result >= 0.5:
        pdpa_data2.set_value(index, 'purpose_predictions', 1)
        
print ("Accuracy of model: ").rjust(19) + str(accuracy((truevalues(pdpa_data2['purpose_predictions'], pdpa_data2['purpose'])), \
                                        len(pdpa_data2['purpose_predictions']))) # prints the accuracy of the logit model

a = precision(pdpa_data2['purpose_predictions'], pdpa_data2['purpose'])
b = recall(pdpa_data2['purpose_predictions'], pdpa_data2['purpose'])
print ("Precision: ").rjust(19) + str(a)
print ("Recall: ").rjust(19) + str(b)
print ("F1: ").rjust(19) +str(f1(a,b))
print ("Specificity: ").rjust(19) + str(specificity(pdpa_data2['purpose_predictions'], pdpa_data2['purpose']))




# legal requirements across industries




print (pdpa_data[pdpa_data['lifestyle'] == 1])['dpp1'].value_counts() # value counts of dpp1 for lifestyle websites
print (pdpa_data[pdpa_data['lifestyle'] == 1])['dpo'].value_counts() # value counts of dpo for lifestyle websites
print (pdpa_data[pdpa_data['lifestyle'] == 1])['purpose'].value_counts() # value counts of purpose for lifestyle websites

print (pdpa_data[pdpa_data['travel'] == 1])['dpp1'].value_counts() # value counts of dpp1 for travel websites
print (pdpa_data[pdpa_data['travel'] == 1])['dpo'].value_counts() # value counts of dpo for travel websites
print (pdpa_data[pdpa_data['travel'] == 1])['purpose'].value_counts() # value counts of purpose for travel websites

print (pdpa_data[pdpa_data['fashion'] == 1])['dpp1'].value_counts() # value counts of dpp1 for fashion websites
print (pdpa_data[pdpa_data['fashion'] == 1])['dpo'].value_counts() # value counts of dpo for fashion websites
print (pdpa_data[pdpa_data['fashion'] == 1])['purpose'].value_counts() # value counts of purpose for fashion websites

print (pdpa_data[pdpa_data['beauty'] == 1])['dpp1'].value_counts() # value counts of dpp1 for beauty websites
print (pdpa_data[pdpa_data['beauty'] == 1])['dpo'].value_counts() # value counts of dpo for beauty websites
print (pdpa_data[pdpa_data['beauty'] == 1])['purpose'].value_counts() # value counts of purpose for beauty websites

print (pdpa_data[pdpa_data['IT'] == 1])['dpp1'].value_counts() # value counts of dpp1 for IT websites
print (pdpa_data[pdpa_data['IT'] == 1])['dpo'].value_counts() # value counts of dpo for IT websites
print (pdpa_data[pdpa_data['IT'] == 1])['purpose'].value_counts() # value counts of purpose for IT websites

print (pdpa_data[pdpa_data['food'] == 1])['dpp1'].value_counts() # value counts of dpp1 for food websites
print (pdpa_data[pdpa_data['food'] == 1])['dpo'].value_counts() # value counts of dpo for food websites
print (pdpa_data[pdpa_data['food'] == 1])['purpose'].value_counts() # value counts of purpose for food websites

print (pdpa_data[pdpa_data['financial_services'] == 1])['dpp1'].value_counts() # value counts of dpp1 for financial services websites
print (pdpa_data[pdpa_data['financial_services'] == 1])['dpo'].value_counts() # value counts of dpo for financial services websites
print (pdpa_data[pdpa_data['financial_services'] == 1])['purpose'].value_counts() # value counts of purpose for financial services websites

print (pdpa_data[pdpa_data['others'] == 1])['dpp1'].value_counts() # value counts of dpp1 for other websites
print (pdpa_data[pdpa_data['others'] == 1])['dpo'].value_counts() # value counts of dpo for other websites
print (pdpa_data[pdpa_data['others'] == 1])['purpose'].value_counts() # value counts of purpose for other websites




# types of data purported to be collected
#SLIDE 34 description of types 

types = ['contact_info', 'com_info', 'financial_info', 'interactive_info',\
         'content_info', 'sensitive_info']
for i in types: 
    print pdpa_data2[i].describe(), pdpa_data2[i].value_counts(sort=False)



# additional findings on 3P practices
#SLIDE 35 descriptive third party variables
#creates a third dataframe for websites that allows 3P to collect PII or engage 3P to collect PII or share data with 3P

pdpa_data3 = (pdpa_data2.loc[(pdpa_data2['3P_sharing'] == 2) | (pdpa_data2['3P_collect'] == 2) | (pdpa_data2['3P_collect_own'] == 2)]).copy(deep=True) 

pdpa_data3['year_last_amended'] = pdpa_data3['year_last_amended'].astype('category')

print pdpa_data3['3P_identify'].describe(), pdpa_data3['3P_identify'].value_counts()
print pdpa_data3['3P_dpp'].describe(), pdpa_data3['3P_dpp'].value_counts()




# function to bin comprehensiveness scores

def comprehensiveness_bins(x): # function to classify comprehensiveness into 3 categories
    if x < 4.0:
        return 1
    if x < 10.0:
        return 2
    else: 
        return 3
    
pdpa_data2['comprehensiveness_actual_bin'] = pdpa_data2['comprehensiveness'].map(comprehensiveness_bins) # bins the actual comprehensiveness scores

# linear model & graph: whether more popular websites are likely to have more comprehensive privacy policies
#Slide 37

pdpa_data2.plot(x='total_global_visits', y='comprehensiveness', kind='scatter') # plots length of DPP against comprehensiveness of DPP
plt.ylabel('Comprehensiveness') # label of y-axis
plt.xlabel('Total global traffic (log)') # label of x-axis
plt.show() # shows the graph

linearmodel_comprehensiveness_global = smf.ols(formula = 'comprehensiveness ~ total_global_visits', data = pdpa_data2).fit()
print linearmodel_comprehensiveness_global.summary()

pdpa_data2['comprehensiveness_predicted_raw00'] = np.nan

for index, rows in pdpa_data2.iterrows(): # iterates through the pdpa_data2 DataFrame
    
    x1 = rows['total_global_visits']
    
    prediction = {'total_global_visits': x1} # creates a prediction dictionary using the stored values for the current row
                  
    predicted_result = linearmodel_comprehensiveness_global.predict(prediction) # stores the raw result of the prediction 

    pdpa_data2.set_value(index, 'comprehensiveness_predicted_raw00', predicted_result)

pdpa_data2['comprehensiveness_predicted_bin00'] = pdpa_data2['comprehensiveness_predicted_raw00'].map(comprehensiveness_bins) # bins the predicted scores

print "Accuracy of linear model: " + str(accuracy(truevalues(pdpa_data2['comprehensiveness_predicted_bin00'], pdpa_data2['comprehensiveness_actual_bin']), \
                                                    len(pdpa_data2['comprehensiveness_predicted_bin00']))) # prints the accuracy of the logit model




# linear model & graph: whether longer DPPs are more comprehensive
#Slide 38

pdpa_data2.plot(x='DPP_length', y='comprehensiveness', kind='scatter') # plots length of DPP against comprehensiveness of DPP
plt.ylabel('Comprehensiveness') # label of y-axis
plt.xlabel('Length of DPP (log)') # label of x-axis
plt.show() # shows the graph

linearmodel_comprehensiveness_length = smf.ols(formula = 'comprehensiveness ~ DPP_length', data = pdpa_data2).fit()
print linearmodel_comprehensiveness_length.summary()

pdpa_data2['comprehensiveness_predicted_raw0'] = np.nan

for index, rows in pdpa_data2.iterrows(): # iterates through the pdpa_data2 DataFrame
    
    x1 = rows['DPP_length']
    
    prediction = {'DPP_length': x1} # creates a prediction dictionary using the stored values for the current row
                  
    predicted_result = linearmodel_comprehensiveness_length.predict(prediction) # stores the raw result of the prediction 

    pdpa_data2.set_value(index, 'comprehensiveness_predicted_raw0', predicted_result)

pdpa_data2['comprehensiveness_predicted_bin0'] = pdpa_data2['comprehensiveness_predicted_raw0'].map(comprehensiveness_bins) # bins the predicted scores

print "Accuracy of linear model: " + str(accuracy(truevalues(pdpa_data2['comprehensiveness_predicted_bin0'], pdpa_data2['comprehensiveness_actual_bin']), \
                                                    len(pdpa_data2['comprehensiveness_predicted_bin0']))) # prints the accuracy of the logit model




# linear model: whether SG websites are more comprehensive
#Slide 39

linearmodel_comprehensiveness_sg = smf.ols(formula = 'comprehensiveness ~ SG_office + total_sg_visits +\
                                                        percentage_sg_visits + sg_domain + pdpa', \
                                                        data = pdpa_data2).fit()

print linearmodel_comprehensiveness_sg.summary()

pdpa_data2['comprehensiveness_predicted_raw1'] = np.nan

for index, rows in pdpa_data2.iterrows(): # iterates through the pdpa_data2 DataFrame
    
    x1 = rows['SG_office']
    x2 = rows['total_sg_visits']
    x5 = rows['percentage_sg_visits']
    x9 = rows['sg_domain']
    x15 = rows['pdpa']

    prediction = {'SG_office': x1, 'total_sg_visits': x2, 'percentage_sg_visits': x5, 'sg_domain': x9, \
                  'pdpa': x15} # creates a prediction dictionary using the stored values for the current row
                  
    predicted_result = linearmodel_comprehensiveness_sg.predict(prediction) # stores the raw result of the prediction 

    pdpa_data2.set_value(index, 'comprehensiveness_predicted_raw1', predicted_result)

pdpa_data2['comprehensiveness_predicted_bin1'] = pdpa_data2['comprehensiveness_predicted_raw1'].map(comprehensiveness_bins) # bins the predicted scores

print "Accuracy of linear model: " + str(accuracy(truevalues(pdpa_data2['comprehensiveness_predicted_bin1'], pdpa_data2['comprehensiveness_actual_bin']), \
                                                    len(pdpa_data2['comprehensiveness_predicted_bin1']))) # prints the accuracy of the logit model




# linear model: comprehensiveness against various factors
#Slide 40-41

linearmodel_comprehensiveness_various = smf.ols(formula = 'comprehensiveness ~ DPP_length + year_last_amended + \
                                        cookies + https + gifts + financial_services + other_law', \
                                         data = pdpa_data2).fit()
                                                                                  
print linearmodel_comprehensiveness_various.summary()

for index, rows in pdpa_data2.iterrows(): # iterates through the pdpa_data2 DataFrame
    
    x1 = rows['DPP_length']
    x2 = rows['year_last_amended']
    x5 = rows['cookies']
    x9 = rows['https']
    x15 = rows['gifts']
    x17 = rows['financial_services']
    x21 = rows['other_law']
    
    prediction = {'DPP_length': x1, 'year_last_amended': x2, 'cookies': x5, 'https': x9, \
                  'gifts': x15, 'financial_services': x17, 'other_law': x21} # creates a prediction dictionary using the stored values for the current row
                  
    predicted_result = linearmodel_comprehensiveness_various.predict(prediction) # stores the raw result of the prediction 

    pdpa_data2.set_value(index, 'comprehensiveness_predicted_raw2', predicted_result)

pdpa_data2['comprehensiveness_predicted_bin2'] = pdpa_data2['comprehensiveness_predicted_raw2'].map(comprehensiveness_bins) # bins the predicted scores

print "Accuracy of linear model 3: " + str(accuracy(truevalues(pdpa_data2['comprehensiveness_predicted_bin2'], pdpa_data2['comprehensiveness_actual_bin']), \
                                                    len(pdpa_data2['comprehensiveness_predicted_bin2']))) # prints the accuracy of the logit model




# additional models after project presentation (added in all the industries; recalibrated last amended to number of days)



linearmodel_comprehensiveness_various2 = smf.ols(formula = 'comprehensiveness ~ DPP_length + total_global_visits +\
                                        cookies + https + lifestyle + travel + fashion + beauty + \
                                        IT + gifts + food + others +\
                                        financial_services + other_law + EU_law + pdpa', \
                                         data = pdpa_data2).fit()
                
print linearmodel_comprehensiveness_various2.summary()


linearmodel_comprehensiveness_various3 = smf.ols(formula = 'comprehensiveness ~ DPP_length + total_global_visits + no_of_days_last_amended + \
                                        cookies + https + lifestyle + travel + fashion + beauty + \
                                        IT + gifts + food + others + financial_services + other_law + EU_law + pdpa', \
                                         data = pdpa_data2, missing='drop').fit()

                                                                                  
print linearmodel_comprehensiveness_various3.summary()



# comprehensiveness across industries (descriptive + graphs)
#Slide 42
categories = ['lifestyle', 'travel',\
  'fashion', 'beauty', 'IT', 'gifts', 'food', \
  'financial_services', 'others']


print pdpa_data2["comprehensiveness"].describe()
for i in categories:
    print i
    c = i
    i = (pdpa_data2.loc[pdpa_data2[i] == 1]).copy(deep=True) #create new df for each category 
    print i["comprehensiveness"].describe()
    y = len(set(i["comprehensiveness"]))
    plt.hist(i["comprehensiveness"])
    plt.title("Comprehensiveness in "+c+" industry companies")
    plt.xlabel("Comprehensiveness in "+c+" industry companies")
    plt.ylabel('Number of Organisations')
    plt.show() 

# flesch score (descriptive + graph)

#Number of policies under 45#
#Slide 46

print pdpa_data['flesch2'].describe()
fleschFailscore = 0
for index, rows in pdpa_data.iterrows():
    if rows['flesch2'] < 45:
        fleschFailscore +=1
print fleschFailscore


#Slide 46

print pdpa_data2.describe()['flesch2']
for i in categories:
    print i
    c = i
    i = (pdpa_data2.loc[pdpa_data2[i] == 1]).copy(deep=True)
    print i["flesch2"].describe()

#  word count statistics
#Slide 48-50

print pdpa_data2.describe()['DPP_length']
for i in categories:
    print i
    c = i
    i = (pdpa_data2.loc[pdpa_data2[i] == 1]).copy(deep=True)
    print i["DPP_length"].describe()

# graph for simplified english
#Slide 51

print pdpa_data2['simplified'].describe()
print pdpa_data2['simplified'].value_counts()
simplified_list = pdpa_data2[pdpa_data2['simplified']==2].url.tolist() # list of websites with simplified english
print simplified_list


graph = pdpa_data2['simplified'].value_counts().sort_index().plot(kind='bar')
plt.ylabel('Number of websites')
plt.xlabel('Whether there is simplified English')
graph.set_xticklabels(['No: 258', 'Yes: 10'], rotation = 0)

plt.show()


# number of clicks 
#Slide 53



print pdpa_data['clicks'].value_counts()
pdpa_data['clicks'].value_counts().plot(kind='bar').set_xticklabels(['1 (245)', '2 (19)','3 (2)','4 (1)'],rotation = 0)
plt.ylabel('Number of Websites')
plt.xlabel('Number of Clicks')
plt.show()

# graph for notification on consent and TestGBQConnectorServiceAccountKeyContentsIntegration
#Slide 54

graph = pdpa_data2['notice_account'].value_counts().sort_index().plot(kind='bar')
plt.ylabel('Number of websites')
plt.xlabel('Whether there is notification of DPP when creating account')
graph.set_xticklabels(['NA: 29', 'No: 90', 'Yes: 149'], rotation = 0)

plt.show()

graph = pdpa_data2['notice_transaction'].value_counts().sort_index().plot(kind='bar')
plt.ylabel('Number of websites')
plt.xlabel('Whether there is notification of DPP when entering into a transaction')
graph.set_xticklabels(['NA: 49', 'No: 123', 'Yes: 96'], rotation = 0)

plt.show()

# presence of https 
#Slide 59

for i in categories:
    print i
    c = i
    i = (pdpa_data2.loc[pdpa_data2[i] == 1]).copy(deep=True)
    print i["https"].value_counts()

# values for cookies across Categories
#Slide 60

print pdpa_data.describe()['cookies']

print pdpa_data[pdpa_data['lifestyle'] == 1].describe()['cookies']
print pdpa_data[pdpa_data['travel'] == 1].describe()['cookies']
print pdpa_data[pdpa_data['fashion'] == 1].describe()['cookies']

print pdpa_data[pdpa_data['beauty'] == 1].describe()['cookies']
print pdpa_data[pdpa_data['IT'] == 1].describe()['cookies']
print pdpa_data[pdpa_data['gifts'] == 1].describe()['cookies']

print pdpa_data[pdpa_data['food'] == 1].describe()['cookies']
print pdpa_data[pdpa_data['financial_services'] == 1].describe()['cookies']
print pdpa_data[pdpa_data['others'] == 1].describe()['cookies']


# emails 

pathEmail = 'C:\\sem8\\email_data.csv'
email_data = pd.read_csv(pathEmail, skiprows = 1, names = ['index', 'url', 'first', 'elapsed', 'reply', \
                                                                   'daysreply', 'quality', 'automated', 'invalid'])

# count of websites with/without dpo that replied

email_data['dpo'] = np.nan

for index, rows in email_data.iterrows():
    website = rows['url']
    dpo_index = (pdpa_data[pdpa_data['url'] == website]).index
    dpo_value = pdpa_data['dpo'][dpo_index]
    
    
    email_data.set_value(index, 'dpo', dpo_value)

print (email_data[email_data['daysreply'] >= 0])['dpo'].value_counts(dropna=False)
print email_data['dpo'].value_counts()

# number of dpo emails that were invalid
print (email_data[email_data['dpo'] == 1])['invalid'].value_counts(dropna=False)



repliesOnly = (email_data.loc[email_data['daysreply'] >= 0]).copy(deep=True) #create a new df for only those that have replied
repliesOnly.reset_index(inplace=True)
 
print email_data.describe()

print repliesOnly['daysreply'].value_counts()
print repliesOnly['quality'].value_counts()




# graph for quality of replies

print email_data['quality'].value_counts(bins=4)
graph_email_quality = email_data['quality'].value_counts(bins=4).sort_index().plot(kind='bar') # plots value counts of Email Quality
plt.title("Quality of replies (Count = 112)")
plt.ylabel('Number of Organisations') # labels y-axis
plt.xlabel('Quality of Email Score (Max 10)') # labels x-axis
graph_email_quality.set_xticklabels(['0-2.5 (15)', '2.5-4.5 (47)','5-7.0 (40)', '7.5-10 (10)'], rotation = 0) # values for x-axis
plt.show()




# graph for time taken to reply


plt.hist(repliesOnly["daysreply"], bins=12)
plt.title("Graph of days by organizations to reply (Count = 112)")
plt.xlabel("Days taken to reply")
plt.xticks(range(0,13))
plt.ylabel('Number of Organisations')
plt.show()


for i in range(0,len(email_data)): #identify websites with invalid emails
    if email_data["invalid"][i] == 1:
        print email_data["url"][i]



# count of websites with/without dpo that replied
print (email_data[email_data['daysreply'] >= 0])['dpo'].value_counts(dropna=False)
print (email_data[email_data['daysreply'] >= 1])['dpo'].value_counts(dropna=False)



# number of dpo emails that were invalid
print (email_data[email_data['dpo'] == 1])['invalid'].value_counts(dropna=False)
print (email_data[email_data['dpo'] == 0])['invalid'].value_counts(dropna=False)
print email_data['invalid'].value_counts()