#TITANIC DATA ANALYSIS
#We are going to try to predict who dies and who survives the Titanic shipwrek
#I have imported pandas as pd, numpy as np, and matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

original_df = pd.read_csv("titanic.csv")
# print("# of passengers in original data:" + str(len(original_df.index)) + '\n') #We're working with 887
# print(original_df.head())
# print(original_df.isnull().sum()) #None in our own file/

#DATA WRANGLING: For missing values in an colunm of the dataset
# age_wwrangled_df = original_df[pd.notnull(original_df['Age'])]
# print('# of passengers in age wrangled data:' + str(len(age_wwrangled_df.index)) + "\n")
# embark_wrangled_df = age_wwrangled_df[pd.notnull(age_wwrangled_df['Embarked'])] #But we do not have Embarked on our file

#Effect of Gender

gender_data = original_df.groupby('Sex', as_index = False)
gender_mean_data = gender_data.mean()
# print('Total Survival Rate: ' + str(original_df['Survived'].mean()))
# print('\nMean Data by Gender')
# print(gender_mean_data[['Sex', 'Survived', 'Age', 'Pclass', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']])

#Further Investigation on Gender
total_df = gender_data['Name'].count() #My file does not have passenger id so i used name
# print(total_df)

total_df.columns = ['Sex', 'Total']
# print(total_df)

gender_list = total_df['Sex'] #Save 'Sex' column in list for input in future plot
del(total_df['Sex'])
# print(total_df)
# print (gender_data)

gender_survived_df = gender_data['Survived'].sum()
# print(gender_survived_df)
del(gender_survived_df['Sex'])

combined_df = total_df.add(gender_survived_df, fill_value = 0)
# print(combined_df)

combined_df.plot.bar(color=['limegreen', 'dodgerblue'])
plt.title('Effect of Gender on Survival')
plt.xlabel('Gender')
plt.ylabel('# of People')
plt.xticks(range(len(gender_list)), gender_list)
# plt.show()

survival_gender_list = [combined_df.loc[0]['Survived'], combined_df.loc[1]['Survived']]
total_gender_list = [combined_df.loc[0]['Total'], combined_df.loc[1]['Total']]

#Define Function to create labels on the plot
def create_value_labels(list_data, decimals, x_adjust, y_adjust):
    for x, y in enumerate(list_data):
        plt.text(x + x_adjust, y + y_adjust, round(list_data[x], decimals), color='m', fontweight='bold')

create_value_labels(survival_gender_list, 1, -0.2, -50)
create_value_labels(total_gender_list, 1, 0.05, -50)
# plt.show()

#What is the effect of age on survival rate?
#WHat is the effect of company(traveling with others) on survival rate?
#What is the effect of socio-economic status on survival rate?

survivor_data = original_df.groupby('Survived', as_index=False)
survivor_mean_data = survivor_data.mean()
# print(survivor_mean_data)

#Spliting the data into children and adults
children_data = original_df[original_df['Age'] <= 18]
adult_data = original_df[original_df['Age'] > 18]

#Counting number of total & survived children and adults
children_count = children_data['Name'].count()
adult_count = adult_data['Name'].count()

# print(children_count)
# print(adult_count)

survived_children_count = children_data['Survived'].sum()
survived_adult_count = adult_data['Survived'].sum()

#Putting into a list
children_list = [survived_children_count, children_count]
adult_list = [survived_adult_count, adult_count]
total_list = [children_count, adult_count]

# print(children_list)
# print(adult_list)
# print(total_list)

survived_list = [survived_children_count, survived_adult_count]
#Creating a pd dataframe for counts above
CvsA_df = pd.DataFrame([children_list, adult_list], columns=['Survival', 'Total'], index=['Children', 'Adult'])
# print(CvsA_df)

#Creating a plot of the data
CvsA_df.plot.bar(color=['limegreen', 'dodgerblue'])
plt.title('Number of Survivals between Children and Adults')
plt.ylabel('# of People')
plt.xticks(range(len(CvsA_df.index)), CvsA_df.index)
# Create and Adding Value labels
create_value_labels(survived_list, 1, -0.2, -50)
create_value_labels(total_list, 1, 0.05, -50)
# plt.show()

#Creating a list with survival rates for children and adults
survival_rate_CvsA = [children_data.mean()['Survived'], adult_data.mean()['Survived']]

plt.bar(range(len(survival_rate_CvsA)), survival_rate_CvsA, align="center", color=['dodgerblue', 'limegreen'])
plt.title('Survival Rate between Children and Adults')
plt.ylabel('Survival Rate')
plt.xticks(range(len(survival_rate_CvsA)), ['Children', 'Adult'])

#Adding value labels for each category
create_value_labels(survival_rate_CvsA, 4, -0.1, -0.1)
# plt.show() #This more children survived than adults, whic seem very logical sha

#Looking at the age distribution of the passengers
original_df['Age'].plot.hist(bins=range(100), color='dodgerblue')
plt.title('Age Distribution of All Passengers')
plt.xlabel('Age')
plt.ylabel('# of passengers')
plt.show()

#Survivors Age Distribution
# survivor_data['Age'].hist(bins=range(100), color='limegreen', label='Survived')
# plt.xlabel('Age')
# plt.ylabel('# of Passengers')
# plt.title('Survivor Age Distribution')
# plt.show()
survived_stats = survivor_data['Age'].describe()
# print(survived_stats)

#Group data by age
age_data = original_df.groupby('Age', as_index=False)
age_mean_data = age_data.mean()

age_list = age_mean_data['Age'].tolist()

#Determining number of passengers in age group
num_passengers_in_age = age_data.count()['Name']

#plot survival rates by age on scatter plot
# scatter_plot1 = plt.scatter(age_mean_data['Age'], age_mean_data['Survived'], s=30, \
#                             alpha=0.9, c=num_passengers_in_age, cmap='RdYlGn', edgecolors='none', vmin=0, vmax=30)
# plt.title('Effect of Age on Survial Rate')
# plt.colorbar(scatter_plot1, label="# of passengers")
# plt.ylabel('Survival Rate')
# plt.xlabel('Age')
# plt.show()

#More Data Analysis oooo(Ike agwula mua self)
count_age = age_data['Name'].count()
#So we'd get df of ages that have greater than 5 passengers
count_age_gt5 = count_age[count_age['Name']>5]
#Create list that stores all ages of passengers that have more than 5 passengers
age_gt5_list = count_age_gt5['Age'].values.tolist()
#Keep data only with ages that have greater than 5 passengers
age_gt5_df = original_df[original_df['Age'].isin(age_gt5_list)]
# print(age_gt5_df['Name'].count())

# age_gt5_df['Age'].plot.hist(bins=range(100), color='mediumorchid')
# plt.title('Age Distribution of All Passengers [Cleansed Data]')
# plt.xlabel('Age')
# plt.ylabel('# of Passengers')
# plt.show()

#Group data by age for another scatter plot
age_gt5_data = age_gt5_df.groupby('Age', as_index = False)
age_gt5_mean_data = age_gt5_data.mean()

num_passengers_in_age_gt5 = age_gt5_data.count()['Name']
#Replot survival rates by age on scatter plot
scatter_plot2 = plt.scatter(age_gt5_list, age_gt5_mean_data['Survived'], s = 30, \
                            alpha=0.9, c = num_passengers_in_age_gt5, cmap='RdYlGn', edgecolors='none', vmin=0, vmax=30)
plt.title('Effect of Age on Survival Rate [Cleaned Data]')
plt.colorbar(scatter_plot2, label = '# of Passengers')
plt.ylabel('Survival Rate')
plt.xlabel('Age')
plt.show()
