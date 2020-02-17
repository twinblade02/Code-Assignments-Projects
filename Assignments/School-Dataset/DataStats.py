# Assignment header
print("DATA-51100, Spring 2020")
print("Lionel Dsilva")
print("Programming Assignment 5 \n")

# Import and get data
import pandas as pd
import re 

dataset = pd.read_csv('C:/Users/ldmag/Downloads/cps.csv')
updated_data = dataset[['School_ID', 'Short_Name', 'Is_High_School', 'Zip',
                        'Student_Count_Total', 'College_Enrollment_Rate_School',
                        'Grades_Offered_All','School_Hours']].sort_index()

# grades
updated_data['lowest_grade'] = updated_data.apply(lambda x: x['Grades_Offered_All'][0:2],1).str.replace(",",'')
updated_data['highest_grade'] = updated_data.apply(lambda x: x['Grades_Offered_All'][-2:],1).str.replace(",",'')

# All missing data, replace with mean [generic algorithm to replace int column nan]
for num in updated_data.select_dtypes(['int64', 'float64']).columns:
    updated_data[num].fillna(updated_data[num].mean(), inplace=True)
    
# hours, find and replace missing data
def start_time(x):
    if str(x[0]) == 'nan':
        return 0
    else:
        return int(re.findall(r'[1-9]', x[0])[0])
    
time = updated_data[['School_Hours']].apply(start_time, axis = 1)
updated_data = updated_data.assign(Starting_Hour = time)

# Drop unnecessary columns, print dataframe 10 columns
updated_data = updated_data.drop(['Grades_Offered_All', 'School_Hours'], axis = 1)
print(updated_data.head(10))

# Mean and standard deviation of college enrollment for high schools only
mean_college = updated_data.groupby('Is_High_School')['College_Enrollment_Rate_School'].mean()
std_college = updated_data.groupby('Is_High_School')['College_Enrollment_Rate_School'].std()
print("College Enrollment Rate for High Schools = ", mean_college[1].round(2), "(sd=", std_college[1].round(2),")")
print("\n")

# Mean and sd of student count for non high schools
mean_count = updated_data.groupby('Is_High_School')['Student_Count_Total'].mean()
std_count = updated_data.groupby('Is_High_School')['Student_Count_Total'].std()
print("Total Student Count for non-High Schools = ", mean_count[0].round(2), "(sd=", std_count[0].round(2),")")
print("\n")

# Distribution for start hours
hour_7 = []
hour_8 = []
hour_9 = []

for n in updated_data['Starting_Hour']:
    if n == 7:
        hour_7.append(n)
    if n == 8:
        hour_8.append(n)
    if n == 9:
        hour_9.append(n)

print("Distribution of Starting Hours \n")
print("8am: ", len(hour_8))
print("7am: ", len(hour_7))
print("9am: ", len(hour_9))
print("\n")

# zip code search + count
arr = [60601,60602,60603,60604,60605,60606,60607,60616]
out_loop = []
for zcode in updated_data['Zip']:
    if zcode not in arr:
        out_loop.append(zcode)

print("Number of schools outside the loop: ", len(out_loop))