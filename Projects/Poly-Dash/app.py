import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output


# ---------------------------- # ----------------------------------# ----------------------------- #
app = dash.Dash(__name__)
fall_data = pd.read_excel('FA23-standardized.xlsx')
fall_data = fall_data.query('Department != "Naval Science" and Department != "Military Science" and Department != "Aerospace Studies"')
spring_data = pd.read_excel('SP23 Standardized.xlsx')
spring_data = spring_data.query('Department != "Naval Science" and Department != "Military Science" and Department != "Aerospace Studies"')
clean_fall = pd.read_csv('clean_fall.csv')
clean_spring = pd.read_csv('clean_sping.csv')
# ---------------------------- # --------------------------------- # ---------------------------- #

app.layout = html.Div([
    html.H1('Dashboard', style={'tex_align': 'center'}),
    dcc.Dropdown(id='file-dropdown', options=[{'label': 'Spring', 'value':'spring'},
                                              {'label': 'Fall', 'value': 'fall'}],
                                              value='spring', multi=False),

    dcc.Dropdown(id = 'instructor_dropdown', options=[{'label':i, 'value': i} for i in spring_data.Instructor.unique()],
                 value=fall_data.Instructor.iloc[0], multi=False),
    html.Div([
        dcc.Graph(id='enrollment-capacity-chart'),
        dcc.Graph(id='mean-courses'),
        dcc.Graph(id='instructor-courses-count'),
        dcc.Graph(id='instructor-courses-sum-department'),
        #dcc.Graph(id='course-count-chart'),
        dcc.Graph(id='department-course-count'),
        dcc.Graph(id='enrollment-by-day'),

        html.Div([
            dcc.Markdown("### Outlier Table", style={'fontSize': 20}),
            dash_table.DataTable(id='outlier-table', columns=[{'name': col, 'id': col} for col in spring_data.columns],
                                 style_table={'height': '300px', 'overflowY': 'auto'},
                                 )
                ])
        ])
])


@app.callback(
        Output('instructor_dropdown', 'options'),
        [Input('file-dropdown', 'value')]
)
def update_instructor_dropdown(value):
    if value == 'spring':
        options = [{'label': i, 'value': i} for i in spring_data['Instructor'].unique()]
    elif value == 'fall':
        options = [{'label': i, 'value': i} for i in fall_data['Instructor'].unique()]
    else:
        options = [{'label': i, 'value': i} for i in spring_data['Instructor'].unique()]
    return options


@app.callback(
        [Output('enrollment-capacity-chart', 'figure'),
         Output('mean-courses', 'figure'),
         Output('instructor-courses-count', 'figure'),
         Output('instructor-courses-sum-department', 'figure'),
         #Output('course-count-chart', 'figure'),
         Output('department-course-count', 'figure'),
         Output('enrollment-by-day', 'figure'),
         Output('outlier-table', 'data')
         ],
        [Input('file-dropdown', 'value'),
         Input('instructor_dropdown', 'value')]
)
def update_chart(selected_file, value):
    if selected_file == 'spring':
        df = spring_data
        df2 = clean_spring
    elif selected_file == 'fall':
        df = fall_data
        df2 = clean_fall
    
    filtered_df = df[df['Instructor'] == value]
    filtered_df['Enrollment %'] = filtered_df['Sum_Enrollment'] / filtered_df['Sum_Capacity'] * 100

    mean_courseDF = df.groupby(['Department', 'Instructor_Title']).agg({'Course_count': 'mean'}).reset_index()

    instructorTitle_courses = df.groupby(['Instructor_Title'], as_index=False)['Course_count'].sum()
    instructorTitle_Deptcourses = df.groupby(['Instructor_Title', 'Department'], as_index=False)['Course_count'].sum()

    Q1 = df['Course_count'].quantile(0.25)
    Q3 = df['Course_count'].quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR
    outliers = df[(df['Course_count'] < lower_threshold) | (df['Course_count'] > upper_threshold)]

    department = df.groupby('Department', as_index=False)['Sum_Enrollment', 'Sum_Capacity'].sum()
    #department['Enrollment %'] = department['Sum_Enrollment'] / department['Sum_Capacity'] * 100
    day_avail = df2.groupby(['Department', 'Days'], as_index=False)['Enrl*'].sum()
    #top_courses = df2.groupby(['Subject','Crs .', 'Title'], as_index=False).agg({'Enrl*': 'sum'}).nlargest(10, 'Enrl*')

    fmt_cap_txt = filtered_df['Sum_Capacity'].astype(int).astype(str)
    fmt_ccount_txt = filtered_df['Course_count'].astype(int).astype(str)

    enrollment_capacity_figure = {
        'data': [
            {'x': filtered_df['Courses'], 'y': filtered_df['Enrollment %'], 'type': 'bar', 'name': 'Enrollment'},
            {'x': filtered_df['Courses'], 'y': 100 - filtered_df['Enrollment %'], 'type': 'bar', 'name': 'Capacity', 'text': fmt_cap_txt}
        ],
        'layout': {
            'title': f'Enrollment and Capacity',
            'barmode': 'stack'
        }
    }

    #course_count_figure = {
    #    'data': [
    #        {'x': filtered_df['Department'] + '-' + filtered_df['Courses'], 'y': filtered_df['Course_count'], 'type':'bar', 'name':'Course count', 'text': fmt_ccount_txt}
    #    ],
    #    'layout': {
    #        'title': f'Course count'
    #    }
    #}

    mean_courses_chart = px.bar(mean_courseDF, x='Department', y='Course_count', color='Instructor_Title',
                                title=f'Mean number of courses by instructor title and department ({selected_file.capitalize()})',
                                labels={'Course_count': 'Mean Courses'}, height=400)
    
    instructor_course_assignment = {
        'data': [
            {'x': instructorTitle_courses['Instructor_Title'], 'y': instructorTitle_courses['Course_count'], 'type': 'bar', 'name': 'Titles'}
        ],
        'layout': {
            'title': f'Sum of Instructor titles teaching all possible courses'
        }
    }

    instructor_department_courseSum = px.bar(instructorTitle_Deptcourses, x='Instructor_Title', y='Course_count', color='Department',
                                             title=f'Sum of instructor titles by department', barmode='stack', labels={'Course_count': 'Sum of Courses'}) 

    department_summary = {
        'data': [
            {'x': department['Department'], 'y': department['Sum_Enrollment'], 'type': 'bar', 'name': 'Enrollment'},
            {'x': department['Department'], 'y': department['Sum_Capacity'], 'type': 'bar', 'name': 'Capacity'}
        ],
        'layout': {
            'title': f'Department Enrollment'
        }
    }

    enrollment_day_summary = px.bar(day_avail, x='Days', y='Enrl*', color='Department',
                                    title=f'Enrollment by day and department', barmode='stack')
    
    outlier_table = outliers.to_dict('records')

    return enrollment_capacity_figure, mean_courses_chart, instructor_course_assignment, instructor_department_courseSum, department_summary, enrollment_day_summary, outlier_table

if __name__ == '__main__':
    app.run(debug=True)