import pandas as pd
import re 

dataset = pd.read_csv('C:/Users/ldmag/Downloads/cps.csv')
updated_data = dataset[['School_ID', 'Short_Name', 'Is_High_School', 'Zip',
                        'Student_Count_Total', 'College_Enrollment_Rate_School',
                        'Grades_Offered_All','School_Hours']].sort_index()

# grades
updated_data['lowest_grade'] = updated_data.apply(lambda x: x['Grades_Offered_All'][0:2],1).str.replace(",",'')
updated_data['highest_grade'] = updated_data.apply(lambda x: x['Grades_Offered_All'][-2:],1).str.replace(",",'')

# All missing data, replace with mean
updated_data_copy = updated_data.copy()
for num in updated_data_copy.select_dtypes(['int64', 'float64']).columns:
    updated_data_copy[num].fillna(updated_data_copy[num].mean(), inplace=True)
    
# hours, find and replace missing data
def start_time(x):
    if str(x[0]) == 'nan':
        return 0
    else:
        return int(re.findall(r'[1-9]', x[0])[0])
    
time = updated_data_copy[['School_Hours']].apply(start_time, axis = 1)
updated_data_copy = updated_data_copy.assign(Starting_Hour = time)

# Drop unnecessary columns
updated_data_copy = updated_data_copy.drop(['Grades_Offered_All', 'School_Hours'], axis = 1)


# Mean and standard deviation of college enrollment for high schools only
mean_college = updated_data_copy.groupby('Is_High_School')['College_Enrollment_Rate_School'].mean()
std_college = updated_data_copy.groupby('Is_High_School')['College_Enrollment_Rate_School'].std()
print("College Enrollment Rate for High Schools = ", mean_college[1].round(2), "(sd=", std_college[1].round(2),")")

# Mean and sd of student count for non high schools
mean_count = updated_data_copy.groupby('Is_High_School')['Student_Count_Total'].mean()
std_count = updated_data_copy.groupby('Is_High_School')['Student_Count_Total'].std()
print("Total Student Count for non-High Schools = ", mean_count[0].round(2), "(sd=", std_count[0].round(2),")")
 
# Distribution for start hours


# zip code search + count
arr = [60601,60602,60603,60604,60605,60606,60607,60616]
out_loop = []
for zcode in updated_data_copy['Zip']:
    if zcode not in arr:
        out_loop.append(zcode)

print("Number of schools outside the loop: ", len(out_loop))