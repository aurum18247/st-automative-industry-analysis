import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



st.title('Automobile Industry Analysis')

df = pd.read_csv(r'cars.csv')





df1 = df.copy()




drop_list = ["Make","Model","Variant"]
df1 = df1.drop(df1.columns.difference(drop_list), axis=1)
df1 = df1.dropna()
df1 =df1.sort_values("Make")

df1

Price = st.slider('Budget', 236447, 21215539, 600000)



#Price = int(Price)
selected_makes = st.multiselect('Make',df['Make'].unique() )

if not selected_makes:
    df2=df.copy()
else:
    df2=df[df['Make'].isin(selected_makes)]

selected_models = st.multiselect('Model',df2['Model'].unique())

if not selected_models:
    df3=df2.copy()
else:
    df3=df2[df2['Model'].isin(selected_models)]


selected_fuel_types= st.multiselect('Fuel Type',df3['Fuel_Type'].unique())

if not selected_fuel_types:
    df4=df3.copy()
else:
    df4=df3[df3['Fuel_Type'].isin(selected_fuel_types)]

selected_body_types = st.multiselect('Body Type',df4['Body_Type'].unique())

if not selected_body_types :
    df5=df4.copy()
else:
    df5=df4[df4['Body_Type'].isin(selected_body_types )]

selected_types = st.multiselect('Type',df5['Type'].unique())



filtered_df = df[
    (df['Make'].isin(selected_makes)) &
    (df['Model'].isin(selected_models)) &
    (df['Fuel_Type'].isin(selected_fuel_types)) &
    (df['Body_Type'].isin(selected_body_types)) &
    (df['Type'].isin(selected_types))&
     (df['Ex-Showroom_Price'] <= Price)
]


number_of_choices = len(filtered_df)



st.write(f"Number of choices after applying filters: {number_of_choices}")

st.dataframe(filtered_df)


#st.dataframe(filtered_df, columns=['Make', 'Model', 'Ex-Showroom_Price','Variant','Displacement','Fuel_Type','City_Mileage','Highway_Mileage'])
#filtered_df['Make', 'Model', 'Ex-Showroom_Price','Variant','Displacement','Fuel_Type','City_Mileage','Highway_Mileage']


numeric_columns = filtered_df.select_dtypes(include=['int64', 'float64'])
st.dataframe(numeric_columns)


# Group by Model and calculate the average price
grouped_df = filtered_df.groupby('Model')['Ex-Showroom_Price'].mean().reset_index()

# Plot the grouped bar plot
fig = px.bar(grouped_df, x='Model', y='Ex-Showroom_Price', title='Average Ex-Showroom Price by Model',
             labels={'Ex-Showroom_Price': 'Average Price'},
             color='Model',
             height=600)

# Customize layout
fig.update_layout(xaxis_title='Model', yaxis_title='Average Ex-Showroom Price',
                  xaxis_tickangle=-45, barmode='group')

# Display the plot
st.plotly_chart(fig)


# Group by Model and calculate the average Fuel Tank Capacity
grouped_capacity_df = filtered_df.groupby('Model')['Fuel_Tank_Capacity'].mean().reset_index()

# Plot the grouped bar plot for Fuel Tank Capacity
fig_capacity = px.bar(grouped_capacity_df, x='Model', y='Fuel_Tank_Capacity', 
                      title='Average Fuel Tank Capacity by Model',
                      labels={'Fuel_Tank_Capacity': 'Average Capacity'},
                      color='Model',
                      height=600)

# Customize layout
fig_capacity.update_layout(xaxis_title='Model', yaxis_title='Average Fuel Tank Capacity',
                           xaxis_tickangle=-45, barmode='group')

# Display the plot for Fuel Tank Capacity
st.plotly_chart(fig_capacity)


#df_melted = df.melt(id_vars='Model', var_name='Mileage_Type', value_name='Mileage')

# Create a bar chart with City Mileage and Highway Mileage on the y-axis and Model on the x-axis
# fig = px.bar(filtered_df, x='Model', y=['City_Mileage', 'Highway_Mileage'],
#              title='City Mileage and Highway Mileage for Filtered Car Models',
#              labels={'value': 'Mileage', 'variable': 'Mileage Type'},
#              height=400)

# # Display the plot
# st.plotly_chart(fig)



# Count the number of customers who purchased each model
customer_count_per_model = df['Make'].value_counts().reset_index()
customer_count_per_model.columns = ['Make', 'Number_of_Customers_Purchased']

# Display the result
st.write(customer_count_per_model)







# Assuming you have a DataFrame named 'customer_count_per_model' with 'Make' and 'Number_of_Customers_Purchased' columns
# Replace this with your actual dataset
# customer_count_per_model = pd.read_csv("your_dataset.csv")

# Select relevant columns for clustering
customer_segmentation = {}

# Calculate the total number of customers
total_customers = len(df)

# Loop through each unique model in the dataframe
for model, model_data in df.groupby('Make'):
    # Count the number of customers who purchased the model
    number_of_customers_purchased = len(model_data)

    # Calculate the percentage of customers who purchased the model
    purchase_percentage = (number_of_customers_purchased / total_customers) * 100

    # Add the data to the dictionary
    customer_segmentation[model] = {
        "Number_of_Customers_Purchased": number_of_customers_purchased,
        "Percentage": purchase_percentage
    }

# Create a pie chart of the customer segmentation data
pie_chart = px.pie(
    customer_segmentation.values(),
    names=customer_segmentation.keys(),
    values='Number_of_Customers_Purchased',
    title='Customer Segmentation by Model'
)

# Show the pie chart on the Streamlit app
st.plotly_chart(pie_chart)
