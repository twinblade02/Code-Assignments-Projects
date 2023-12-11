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
fall_data = pd.read_csv('FA23.csv')
spring_data = pd.read_csv('SP23.csv')
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
        dcc.Graph(id='course-count-chart'),
        dcc.Graph(id='department-course-count'),
        dcc.Graph(id='enrollment-by-day'),
        dash_table.DataTable(id='popular-courses-table', columns=[{'name': 'Subject', 'id': 'Subject'},
                                                              {'name': 'Course', 'id': 'Crs .'},
                                                              {'name': 'Title', 'id': 'Title'},
                                                              {'name': 'Sum_Enrolled', 'id': 'Enrl*'}
                                                              ], style_table={'height':'300px', 'overflowY':'auto'})
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
         Output('course-count-chart', 'figure'),
         Output('department-course-count', 'figure'),
         Output('enrollment-by-day', 'figure'),
         Output('popular-courses-table', 'data')],
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

    department = df.groupby('Department', as_index=False)['Sum_Enrollment', 'Sum_Capacity'].sum()
    #department['Enrollment %'] = department['Sum_Enrollment'] / department['Sum_Capacity'] * 100
    day_avail = df2.groupby(['Department', 'Days'], as_index=False)['Enrl*'].sum()
    top_courses = df2.groupby(['Subject','Crs .', 'Title'], as_index=False).agg({'Enrl*': 'sum'}).nlargest(10, 'Enrl*')

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

    course_count_figure = {
        'data': [
            {'x': filtered_df['Department'] + '-' + filtered_df['Courses'], 'y': filtered_df['Course_count'], 'type':'bar', 'name':'Course count', 'text': fmt_ccount_txt}
        ],
        'layout': {
            'title': f'Course count'
        }
    }

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
    
    popular_courses = top_courses.to_dict('records')

    return enrollment_capacity_figure, course_count_figure, department_summary, enrollment_day_summary, popular_courses

if __name__ == '__main__':
    app.run(debug=True)