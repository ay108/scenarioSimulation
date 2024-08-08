import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import numpy as np
import pickle
# Load SARIMA, SARIMAX, and LSTM models
model_path = 'sarimax_model.pkl'

@st.cache_data
def load_sarimax_model():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

sarimax_model = load_sarimax_model()

# Path to your data file
file_path = 'quarter_data.xlsx'

# Load the data
data = pd.read_excel(file_path, index_col='date', parse_dates=True)

# Streamlit app
st.title("Electricity Price Scenario Simulation in Fairbanks North Star Borough, Alaska")
st.write("This project aims to forecast how factors such as changes to customer base, changes in consumer use patterns, or changes to the cost of fuel impact the delivered price of electricity for residents in the Fairbanks North Star Borough. This tool is intended as a high-level predictive tool to help policy makers understand how potential future scenarios could impact pricing positively or negatively. Alaskans have some of the highest energy prices in the country and the cost of power is a major concern to residents. Expected shortages of Cook Inlet gas, and recent sharp increases to as high as 30 cents/kWh have elevated this issue for both residents and policy makers.")
st.write("The data is from the GVEA website (Residential Rates, Eva Creek Wind Farm data), the EIA-923 website, as well as the EIA.gov website.")

# Sidebar
st.sidebar.title("What Scenario?")

# Sidebar options
scenarios = ['Baseline Model', 'Generation Doubled', 'Natural Gas Prices Doubled', 'Mines Close']

# Sidebar selection
scenario = st.sidebar.selectbox("Choose a scenario", scenarios)

# Scenario simulation
if scenario == 'Baseline Model':
    st.header("Baseline Model (No Scenario)")
    st.write("The model predicts the electricity prices for the next 12 months.")
    # Predict the electricity prices for the next 12 months
    model = sarimax_model

        # Prepare exogenous variables for the forecast
    exog = pd.DataFrame({
        'net_gen': data['net_gen'],
        'nat_gas_prices': data['nat_gas_real'],
        'crude_prod': data['crude_oil_prod'], 
        'crude_prices_real': data['crude_prices_real'],
        'coal': data['coal'],
        'wind': data['wnd'],
        'oil': data['oil']
    })
            
    # Forecasting
    window_size = 4
    n_steps = 4
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='Q')[1:]
    net_gen_adding = np.mean(data['net_gen'].values[-window_size:])

    future_exog = {
        'net_gen': np.mean(data['net_gen'].values[-window_size:]) * np.ones(n_steps),
        'nat_gas_prices': np.mean(data['nat_gas_real'].values[-window_size:]*1.1) * np.ones(n_steps),
        'crude_prod': np.mean(data['crude_oil_prod'].values[-window_size:]) * np.ones(n_steps),
        'crude_prices_real': np.mean(data['crude_prices_real'].values[-window_size:]*1.059) * np.ones(n_steps),
        'coal': np.mean(data['coal'].values[-window_size:]) * np.ones(n_steps),
        'wind': np.mean(data['wnd'].values[-window_size:]) * np.ones(n_steps),
        'oil': np.mean(data['oil'].values[-window_size:]) * np.ones(n_steps)
    }
    future_exog_df = pd.DataFrame(future_exog, index=future_dates)
            
    forecast = model.get_forecast(steps=n_steps, exog=future_exog_df)
    forecast_series = forecast.predicted_mean
    forecast_index = future_dates

    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['effective_real'], label='Observed', color='blue')
    plt.plot(forecast_index, forecast_series, label='Forecast', color='red')
    plt.fill_between(forecast_index, forecast.conf_int()['lower effective_real'], forecast.conf_int()['upper effective_real'], color='pink')
    plt.xlabel('Date')
    plt.ylabel('Effective Rate Rates')
    plt.title('Effective Rate Forecast')
    plt.legend()
    st.pyplot(plt)
    
    # Handling the future warnings for pandas
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

    utility_last = data["utility_real"].iloc[-1]
    last_cpi = data['CPI_new'].iloc[-1]    
    utility_add = (utility_last*last_cpi)/100

    # Handling the future warnings for pandas
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

    # Updating the forecast series calculation
    for i in range(len(forecast_series)):
        forecast_series.iloc[i] = (forecast_series.iloc[i] * last_cpi) / 100

    # Remove the dates and round the values
    forecast_series_values = forecast_series.round(3).values

    #get the first value of the forecasted series
    forecast_series_value = forecast_series_values[0]
    #convert to string
    forecast_series_value_str = str(forecast_series_value)

    # Convert forecast_series values to a string before concatenating
    forecast_series_str = ', '.join(map(str, forecast_series_values))
    st.write("Nominal Forecast Values: " + forecast_series_str)
    st.subheader("Predicted Effective Rates : " + "$"+ forecast_series_value_str)


    # # Calculate nominal effective rates and round to 3 decimal places
    # nom_effective = []
    # for i in range(len(forecast_series)):
    #     nom_effective.append(round(forecast_series.iloc[i] + utility_add, 3))

    # # Convert nom_effective list to a string before concatenating
    # nom_effective_str = ', '.join(map(str, nom_effective))
    # st.write("Nominal Effective Rates Forecast: " + nom_effective_str)

    # # If there is any other Series that needs to be displayed, handle it similarly
    # forecast_series_str = ', '.join(map(str, forecast_series_values))
    # st.write("Forecasted Real Fuel Rates: " + forecast_series_str)


elif scenario == 'Generation Doubled':
    st.header("In this scenario, the generation is doubled.")
    st.write("The model predicts the electricity prices for the next 12 months.")
    # Predict the electricity prices for the next 12 months
    model = sarimax_model
            
    # Prepare exogenous variables for the forecast
    exog = pd.DataFrame({
        'net_gen': data['net_gen'],
        'nat_gas_prices': data['nat_gas_real'],
        'crude_prod': data['crude_oil_prod'], 
        'crude_prices_real': data['crude_prices_real'],
        'coal': data['coal'],
        'wind': data['wnd'],
        'oil': data['oil']
    })
            
    # Forecasting
    window_size = 4
    n_steps = 4
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]
    net_gen_adding = np.mean(data['net_gen'].values[-window_size:])

    future_exog = {
        'net_gen': np.mean(data['net_gen'].values[-window_size:]*2) * np.ones(n_steps),
        'nat_gas_prices': np.mean(data['nat_gas_real'].values[-window_size:]) * np.ones(n_steps),
        'crude_prod': np.mean(data['crude_oil_prod'].values[-window_size:]) * np.ones(n_steps),
        'crude_prices_real': np.mean(data['crude_prices_real'].values[-window_size:]) * np.ones(n_steps),
        'coal': np.mean(data['coal'].values[-window_size:]) * np.ones(n_steps),
        'wind': np.mean(data['wnd'].values[-window_size:]) * np.ones(n_steps),
        'oil': np.mean(data['oil'].values[-window_size:]+net_gen_adding) * np.ones(n_steps)
    }
    future_exog_df = pd.DataFrame(future_exog, index=future_dates)
            
    forecast = model.get_forecast(steps=n_steps, exog=future_exog_df)
    forecast_series = forecast.predicted_mean
    forecast_index = future_dates

    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['effective_real'], label='Observed', color='blue')
    plt.plot(forecast_index, forecast_series, label='Forecast', color='red')
    plt.fill_between(forecast_index, forecast.conf_int()['lower effective_real'], forecast.conf_int()['upper effective_real'], color='pink')
    plt.xlabel('Date')
    plt.ylabel('Effective Real Rates')
    plt.title('Effective Rate Forecast')
    plt.legend()
    st.pyplot(plt)
    
    # Handling the future warnings for pandas
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

    utility_last = data["utility_real"].iloc[-1]
    last_cpi = data['CPI_new'].iloc[-1]    
    utility_add = (utility_last*last_cpi)/100

    # Handling the future warnings for pandas
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

    # Updating the forecast series calculation
    for i in range(len(forecast_series)):
        forecast_series.iloc[i] = (forecast_series.iloc[i] * last_cpi) / 100

    # Remove the dates and round the values
    forecast_series_values = forecast_series.round(3).values
    
    #get the first value of the forecasted series
    forecast_series_value = forecast_series_values[0]
    #convert to string
    forecast_series_value_str = str(forecast_series_value)

    # Convert forecast_series values to a string before concatenating
    forecast_series_str = ', '.join(map(str, forecast_series_values))
    st.write("Nominal Forecast Values: " + forecast_series_str)
    st.subheader("Predicted Effective Rates : " + "$"+ forecast_series_value_str)


    # # Calculate nominal effective rates and round to 3 decimal places
    # nom_effective = []
    # for i in range(len(forecast_series)):
    #     nom_effective.append(round(forecast_series.iloc[i] + utility_add, 3))

    # # Convert nom_effective list to a string before concatenating
    # nom_effective_str = ', '.join(map(str, nom_effective))
    # st.write("Nominal Effective Rates Forecast: " + nom_effective_str)

    # # If there is any other Series that needs to be displayed, handle it similarly
    # forecast_series_str = ', '.join(map(str, forecast_series_values))
    # st.write("Forecasted Real Fuel Rates: " + forecast_series_str)







elif scenario == 'Natural Gas Prices Doubled':
    st.header("In this scenario, the natural gas prices are doubled.")
    st.write("The model predicts the electricity prices for the next 12 months.")
    # Predict the electricity prices for the next 12 months
    model = sarimax_model
            
    # Prepare exogenous variables for the forecast
    exog = pd.DataFrame({
        'net_gen': data['net_gen'],
        'nat_gas_prices': data['nat_gas_real'],
        'crude_prod': data['crude_oil_prod'], 
        'crude_prices_real': data['crude_prices_real'],
        'coal': data['coal'],
        'wind': data['wnd'],
        'oil': data['oil']
    })
            
    # Forecasting
    window_size = 4
    n_steps = 4
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]
    net_gen_adding = np.mean(data['net_gen'].values[-window_size:])

    future_exog = {
        'net_gen': np.mean(data['net_gen'].values[-window_size:]) * np.ones(n_steps),
        'nat_gas_prices': np.mean(data['nat_gas_real'].values[-window_size:]*2) * np.ones(n_steps),
        'crude_prod': np.mean(data['crude_oil_prod'].values[-window_size:]) * np.ones(n_steps),
        'crude_prices_real': np.mean(data['crude_prices_real'].values[-window_size:]*1.059) * np.ones(n_steps),
        'coal': np.mean(data['coal'].values[-window_size:]+net_gen_adding) * np.ones(n_steps),
        'wind': np.mean(data['wnd'].values[-window_size:]) * np.ones(n_steps),
        'oil': np.mean(data['oil'].values[-window_size:]) * np.ones(n_steps)
    }
    future_exog_df = pd.DataFrame(future_exog, index=future_dates)
            
    forecast = model.get_forecast(steps=n_steps, exog=future_exog_df)
    forecast_series = forecast.predicted_mean
    forecast_index = future_dates
    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['effective_real'], label='Observed', color='blue')
    plt.plot(forecast_index, forecast_series, label='Forecast', color='red')
    plt.fill_between(forecast_index, forecast.conf_int()['lower effective_real'], forecast.conf_int()['upper effective_real'], color='pink')
    plt.xlabel('Date')
    plt.ylabel('Effective Real Rates')
    plt.title('Effective Rate Forecast')
    plt.legend()
    st.pyplot(plt)
    
    # Handling the future warnings for pandas
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

    utility_last = data["utility_real"].iloc[-1]
    last_cpi = data['CPI_new'].iloc[-1]    
    utility_add = (utility_last*last_cpi)/100

    # Handling the future warnings for pandas
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

    # Updating the forecast series calculation
    for i in range(len(forecast_series)):
        forecast_series.iloc[i] = (forecast_series.iloc[i] * last_cpi) / 100

    # Remove the dates and round the values
    forecast_series_values = forecast_series.round(3).values

    #get the first value of the forecasted series
    forecast_series_value = forecast_series_values[0]
    #convert to string
    forecast_series_value_str = str(forecast_series_value)

    # Convert forecast_series values to a string before concatenating
    forecast_series_str = ', '.join(map(str, forecast_series_values))
    st.write("Nominal Forecast Values: " + forecast_series_str)
    st.subheader("Predicted Effective Rates : " + "$"+ forecast_series_value_str)


    # # Calculate nominal effective rates and round to 3 decimal places
    # nom_effective = []
    # for i in range(len(forecast_series)):
    #     nom_effective.append(round(forecast_series.iloc[i] + utility_add, 3))

    # # Convert nom_effective list to a string before concatenating
    # nom_effective_str = ', '.join(map(str, nom_effective))
    # st.write("Nominal Effective Rates Forecast: " + nom_effective_str)

    # # If there is any other Series that needs to be displayed, handle it similarly
    # forecast_series_str = ', '.join(map(str, forecast_series_values))
    # st.write("Forecasted Real Fuel Rates: " + forecast_series_str)



# elif scenario == 'Generation Doubled with Wind':
#     st.write("In this scenario, generation is doubled and met with wind")
#     st.write("The model predicts the electricity prices for the next 12 months.")
#     # Predict the electricity prices for the next 12 months
#     model = sarimax_model
            
#     # Prepare exogenous variables for the forecast
#     exog = pd.DataFrame({
#         'net_gen': data['net_gen'],
#         'nat_gas_prices': data['nat_gas_real'],
#         'crude_prod': data['crude_oil_prod'], 
#         'crude_prices_real': data['crude_prices_real'],
#         'coal': data['coal'],
#         'wind': data['wnd'],
#         'oil': data['oil']
#     })
            
#     # Forecasting
#     window_size = 4
#     n_steps = 4
#     future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]
#     net_gen_adding = np.mean(data['net_gen'].values[-window_size:])

#     future_exog = {
#         'net_gen': (np.mean(data['net_gen'].values[-window_size:]*2)) * np.ones(n_steps),
#         'nat_gas_prices': np.mean(data['nat_gas_real'].values[-window_size:]) * np.ones(n_steps),
#         'crude_prod': np.mean(data['crude_oil_prod'].values[-window_size:]) * np.ones(n_steps),
#         'crude_prices_real': np.mean(data['crude_prices_real'].values[-window_size:]) * np.ones(n_steps),
#         'coal': np.mean(data['coal'].values[-window_size:]) * np.ones(n_steps),
#         'wind': np.mean(data['wnd'].values[-window_size:]+ net_gen_adding) * np.ones(n_steps),
#         'oil': np.mean(data['oil'].values[-window_size:]) * np.ones(n_steps)
#         }
#     future_exog_df = pd.DataFrame(future_exog, index=future_dates)
            
#     forecast = model.get_forecast(steps=n_steps, exog=future_exog_df)
#     forecast_series = forecast.predicted_mean
#     forecast_index = future_dates
#     # Plot the forecast
#     plt.figure(figsize=(10, 6))
#     plt.plot(data.index, data['effective_real'], label='Observed', color='blue')
#     plt.plot(forecast_index, forecast_series, label='Forecast', color='red')
#     plt.fill_between(forecast_index, forecast.conf_int()['lower effective_real'], forecast.conf_int()['upper effective_real'], color='pink')
#     plt.xlabel('Date')
#     plt.ylabel('Effective Rate (Real) Price')
#     plt.title('Effective Rate Forecast')
#     plt.legend()
#     st.pyplot(plt)
    
#     # Handling the future warnings for pandas
#     future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

#     utility_last = data["utility_real"].iloc[-1]
#     last_cpi = data['CPI_new'].iloc[-1]    
#     utility_add = (utility_last*last_cpi)/100

#     # Handling the future warnings for pandas
#     future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

#     # Updating the forecast series calculation
#     for i in range(len(forecast_series)):
#         forecast_series.iloc[i] = (forecast_series.iloc[i] * last_cpi) / 100

#     # Remove the dates and round the values
#     forecast_series_values = forecast_series.round(3).values

#     # Convert forecast_series values to a string before concatenating
#     forecast_series_str = ', '.join(map(str, forecast_series_values))
#     st.write("Nominal Forecast Values: " + forecast_series_str)

#     # # Calculate nominal effective rates and round to 3 decimal places
#     # nom_effective = []
#     # for i in range(len(forecast_series)):
#     #     nom_effective.append(round(forecast_series.iloc[i] + utility_add, 3))

#     # # Convert nom_effective list to a string before concatenating
#     # nom_effective_str = ', '.join(map(str, nom_effective))
#     # st.write("Nominal Effective Rates Forecast: " + nom_effective_str)

#     # # If there is any other Series that needs to be displayed, handle it similarly
#     # forecast_series_str = ', '.join(map(str, forecast_series_values))
#     # st.write("Forecasted Real Fuel Rates: " + forecast_series_str)




elif scenario == 'Mines Close':
    st.header("In this scenario, the mines have closed.")
    st.write("The model predicts the trend of electricity prices for the next 12 months.")
    # Predict the electricity prices for the next 12 months
    model = sarimax_model
            
    coal_gen = data['coal'].sum()
    perc_coal = coal_gen/data['net_gen'].sum()
    net_gen_last = data['net_gen'].iloc[-1]
    coal_part = perc_coal*net_gen_last

    # Prepare exogenous variables for the forecast
    exog = pd.DataFrame({
        'net_gen': data['net_gen'],
        'nat_gas_prices': data['nat_gas_real'],
        'crude_prod': data['crude_oil_prod'], 
        'crude_prices_real': data['crude_prices_real'],
        'coal': data['coal'],
        'wind': data['wnd'],
        'oil': data['oil']
    })
            
    # Forecasting
    window_size = 4
    n_steps = 4
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]
    net_gen_adding = np.mean(data['net_gen'].values[-window_size:])

    future_exog = {

        'net_gen': (np.mean(data['net_gen'].values[-window_size:]-coal_part) * np.ones(n_steps)),
        'nat_gas_prices': np.mean(data['nat_gas_real'].values[-window_size:]*1.1) * np.ones(n_steps),
        'crude_prod': np.mean(data['crude_oil_prod'].values[-window_size:]+coal_part) * np.ones(n_steps),
        'crude_prices_real': (np.mean(data['crude_prices_real'].values[-window_size:])*1.059) * np.ones(n_steps),
        'coal': (0) * np.ones(n_steps),
        'wind': np.mean(data['wnd'].values[-window_size:]) * np.ones(n_steps),
        'oil': np.mean((data['oil'].values[-window_size:])) * np.ones(n_steps) 

    }
    future_exog_df = pd.DataFrame(future_exog, index=future_dates)
    forecast = model.get_forecast(steps=n_steps, exog=future_exog_df)
    forecast_series = forecast.predicted_mean
    forecast_index = future_dates
    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['effective_real'], label='Observed', color='blue')
    plt.plot(forecast_index, forecast_series, label='Forecast', color='red')
    plt.fill_between(forecast_index, forecast.conf_int()['lower effective_real'], forecast.conf_int()['upper effective_real'], color='pink')
    plt.xlabel('Date')
    plt.ylabel('Effective Real Rates')
    plt.title('Effective Rate Forecast')
    plt.legend()
    st.pyplot(plt)
    
    # Handling the future warnings for pandas
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

    utility_last = data["utility_real"].iloc[-1]
    last_cpi = data['CPI_new'].iloc[-1]    
    utility_add = (utility_last*last_cpi)/100

    # Handling the future warnings for pandas
    future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

    # Updating the forecast series calculation
    for i in range(len(forecast_series)):
        forecast_series.iloc[i] = (forecast_series.iloc[i] * last_cpi) / 100

    # Remove the dates and round the values
    forecast_series_values = forecast_series.round(3).values

        #get the first value of the forecasted series
    forecast_series_value = forecast_series_values[0]
    #convert to string
    forecast_series_value_str = str(forecast_series_value)

    # Convert forecast_series values to a string before concatenating
    forecast_series_str = ', '.join(map(str, forecast_series_values))
    st.write("Nominal Forecast Values: " + forecast_series_str)
    st.subheader("Predicted Effective Rates : " + "$"+ forecast_series_value_str)

    # # Calculate nominal effective rates and round to 3 decimal places
    # nom_effective = []
    # for i in range(len(forecast_series)):
    #     nom_effective.append(round(forecast_series.iloc[i] + utility_add, 3))

    # # Convert nom_effective list to a string before concatenating
    # nom_effective_str = ', '.join(map(str, nom_effective))
    # st.write("Nominal Effective Rates Forecast: " + nom_effective_str)

    # # If there is any other Series that needs to be displayed, handle it similarly
    # forecast_series_str = ', '.join(map(str, forecast_series_values))
    # st.write("Forecasted Real Fuel Rates: " + forecast_series_str)





# else:
#     st.write("In this scenario, Elon Musk has built a data center.")
#     st.write("The model predicts the trend of electricity prices for the next 12 months.")
#     # Predict the electricity prices for the next 12 months
#     model = sarimax_model
            
#     coal_gen = data['coal'].sum()
#     perc_coal = coal_gen/data['net_gen'].sum()
#     net_gen_last = data['net_gen'].iloc[-1]
#     coal_part = perc_coal*net_gen_last

#     # Prepare exogenous variables for the forecast
#     exog = pd.DataFrame({
#         'net_gen': data['net_gen'],
#         'nat_gas_prices': data['nat_gas_real'],
#         'crude_prod': data['crude_oil_prod'], 
#         'crude_prices_real': data['crude_prices_real'],
#         'coal': data['coal'],
#         'wind': data['wnd'],
#         'oil': data['oil']
#     })
            
#     # Forecasting
#     window_size = 4
#     n_steps = 4
#     future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]
#     net_gen_adding = np.mean(data['net_gen'].values[-window_size:])

#     future_exog = {
#         'net_gen': (np.mean(data['net_gen'].values[-window_size:]*1.5)) * np.ones(n_steps),
#         'nat_gas_prices': np.mean(data['nat_gas_real'].values[-window_size:]) * np.ones(n_steps),
#         'crude_prod': np.mean(data['crude_oil_prod'].values[-window_size:]) * np.ones(n_steps),
#         'crude_prices_real': np.mean(data['crude_prices_real'].values[-window_size:]) * np.ones(n_steps),
#         'coal': np.mean(data['coal'].values[-window_size:]) * np.ones(n_steps),
#         'wind': np.mean(data['wnd'].values[-window_size:]+net_gen_adding/2) * np.ones(n_steps),
#         'oil': np.mean(data['oil'].values[-window_size:]+net_gen_adding/2) * np.ones(n_steps)

#     }
#     future_exog_df = pd.DataFrame(future_exog, index=future_dates)
            
#     forecast = model.get_forecast(steps=n_steps, exog=future_exog_df)
#     forecast_series = forecast.predicted_mean
#     forecast_index = future_dates
#     # Plot the forecast
#     plt.figure(figsize=(10, 6))
#     plt.plot(data.index, data['fpp_real'], label='Observed', color='blue')
#     plt.plot(forecast_index, forecast_series, label='Forecast', color='red')
#     plt.fill_between(forecast_index, forecast.conf_int()['lower fpp_real'], forecast.conf_int()['upper fpp_real'], color='pink')
#     plt.xlabel('Date')
#     plt.ylabel('Fuel Price')
#     plt.title('Fuel Price Forecast')
#     plt.legend()
#     st.pyplot(plt)

#     utility_last = data["utility_real"].iloc[-1]
#     last_cpi = data['CPI_new'].iloc[-1]    
#     utility_add = (utility_last*last_cpi)/100


#     # Handling the future warnings for pandas
#     future_dates = pd.date_range(start=data.index[-1], periods=n_steps+1, freq='QE')[1:]

#     # Updating the forecast series calculation
#     for i in range(len(forecast_series)):
#         forecast_series.iloc[i] = (forecast_series.iloc[i] * last_cpi) / 100

#     # Remove the dates and round the values
#     forecast_series_values = forecast_series.round(3).values

#     # Convert forecast_series values to a string before concatenating
#     forecast_series_str = ', '.join(map(str, forecast_series_values))
#     st.write("Nominal Forecast Values: " + forecast_series_str)

#     # Calculate nominal effective rates and round to 3 decimal places
#     nom_effective = []
#     for i in range(len(forecast_series)):
#         nom_effective.append(round(forecast_series.iloc[i] + utility_add, 3))

#     # Convert nom_effective list to a string before concatenating
#     nom_effective_str = ', '.join(map(str, nom_effective))
#     st.write("Nominal Effective Rates Forecast: " + nom_effective_str)

#     # If there is any other Series that needs to be displayed, handle it similarly
#     forecast_series_str = ', '.join(map(str, forecast_series_values))
#     st.write("Forecasted Real Fuel Rates: " + forecast_series_str)








