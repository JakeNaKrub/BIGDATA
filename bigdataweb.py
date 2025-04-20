import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="Stock Selector", layout="wide")
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
@st.cache_data
def get_financial_metrics(ticker_symbol,currentprice):
    # Download stock data
    stock = yf.Ticker(ticker_symbol)
    logging.info(stock)
    # Get current date and historical dates
    end_date = datetime.now().replace(tzinfo=None)
    start_date_3m = (end_date - timedelta(days=90)).replace(tzinfo=None)
    start_date_6m = (end_date - timedelta(days=180)).replace(tzinfo=None)
    
    # Get price history
    hist = stock.history(period="1y")
    hist.index = hist.index.tz_localize(None)
    #current_price =
    
    current_price= currentprice
    price_3m_ago = hist.loc[hist.index >= start_date_3m]['Close'].iloc[0]
    price_6m_ago = hist.loc[hist.index >= start_date_6m]['Close'].iloc[0]
    #print(price_3m_ago,price_6m_ago)
    # Calculate momentum
    momentum_3m = (current_price - price_3m_ago) / price_3m_ago
    momentum_6m = (current_price - price_6m_ago) / price_6m_ago
    
    # Get financial statements
    balance_sheet = stock.balance_sheet
    income_stmt = stock.income_stmt

    #print(balance_sheet)
    # Extract latest available values
    try:
        # Profitability Ratios
        roe = income_stmt.loc['Net Income'].iloc[0] / balance_sheet.loc['Stockholders Equity'].iloc[0]
        net_profit_margin = income_stmt.loc['Net Income'].iloc[0] / income_stmt.loc['Total Revenue'].iloc[0]
        
        # Valuation Ratios
        pe_ratio = current_price / income_stmt.loc['Diluted EPS'].iloc[0]
        pb_ratio = current_price / (balance_sheet.loc['Stockholders Equity'].iloc[0] / 
                                   balance_sheet.loc['Common Stock'].iloc[0])
        
        # Leverage Ratios
        info = stock.info
        
        # Create output dictionary
        metrics = {
            'ROE': roe,
            'P/E': pe_ratio,
            'P/B': pb_ratio,
            'D/E': info.get('priceToBook'),
            'Current_Ratio': info.get('currentRatio'),
            'Net_Profit_Margin': net_profit_margin,
            'EBITDA': info.get('ebitda'),
            'DilutedEPS': info.get('trailingEps'),
            '3m_momentum': float(momentum_3m),
            '6m_momentum': float(momentum_6m)
        }
        #print(metrics)
        
        return metrics,hist
    
    except Exception as e:
        print(f"Error calculating metrics for {ticker_symbol}: {str(e)}")
        return None

def predict(model):
    print("Begin Prediction Process")
    import xgboost as xgb
    import numpy as np
    from sklearn.preprocessing import LabelEncoder 
    le = LabelEncoder()
    le.fit([-1, 0, 1]) 
    import pandas as pd
    import numpy as np
    import joblib
    symbol = st.session_state.display_symbol
    # Create a dictionary with random values for each column
    active_params = {k: [v] for k, v in st.session_state.stock_params.items() if v is not None}

    # Create DataFrame
    df = pd.DataFrame(active_params)
    df = df.rename(columns={
        'Current_Ratio': 'Current Ratio',
        'DilutedEPS': 'Diluted EPS',
        'Net_Profit_Margin': 'Net Profit Margin',
        'Current Price' : 'price'
    })
    print("The Following Was used for prediction:\n",df)
    print("Current Model Selection:",model)
    if model == "XGboost":
        bst_loaded = xgb.Booster()
        bst_loaded.load_model('xgboost_model.json')  # Update path if needed

        dval = xgb.DMatrix(df)

        # Predict with the loaded model
        preds = bst_loaded.predict(dval)  # Shape: (n_samples, n_classes)

    # Convert probabilities to class labels
    preds_class = np.argmax(preds, axis=1)  # Encoded: 0 (Sell), 1 (Hold), 2 (Buy)
    preds_class_decoded = le.inverse_transform(preds_class)  # Decode to -1, 0, 1
    print(preds)
    return preds_class_decoded, preds

# Load stock database from CSV
@st.cache_data
def load_stock_db():
    data = pd.read_csv("cleaned_combined.csv")
    # Rename columns to match expected names
    data = data.rename(columns={
        'Security': 'Company Name',
        'GICS Sector': 'Sector'
    })
    data = data.rename(columns={
        'Current Ratio':'Current_Ratio',
        'Diluted EPS':'DilutedEPS',
        'Net Profit Margin':'Net_Profit_Margin',
        'price':'Current Price'
    })
    data = data.sort_values(by='Date', ascending=False)
    data = data.drop_duplicates(subset='Symbol', keep='first')
    print(data)
    return data

# Find closest stocks based on parameter values
def find_closest_stocks(df, target_params, num_results=5):
    # Extract numerical columns for comparison
    numerical_cols = ['ROE','P/E','P/B','D/E','Current_Ratio','Net_Profit_Margin','EBITDA','DilutedEPS','3m_momentum','6m_momentum']
    
    # Calculate Euclidean distance for each stock
    distances = []
    for _, row in df.iterrows():
        
        distance = 0
        for col in numerical_cols:
            # Only consider if column is in target_params AND target value is not None AND row value is not NA
            if col in target_params and target_params[col] is not None and pd.notna(row[col]):
                col_std = df[col].std()
                if col_std > 0:  # Avoid division by zero
                    distance += ((row[col] - target_params[col]) / col_std) ** 2
                else:
                    distance += (row[col] - target_params[col]) ** 2
        distances.append(np.sqrt(distance))
    # Add distances to dataframe
    df['Distance'] = distances
    
    # Sort by distance and return top matches
    closest_stocks = df.sort_values('Distance').head(num_results)
    closest_stocks = closest_stocks.loc[df['Symbol'] != st.session_state.current_symbol]
    return closest_stocks.drop(columns=['Distance'])
# Page 1: Stock Analyzer
def stock_analyzer(stock_db):
    st.title("Stock Analyzer")
    st.info("The investment risk. Prospectus carefully before investing")
    if 'signal' not in st.session_state:
        st.session_state.signal = None
    if 'option' not in st.session_state:
        st.session_state.option = None
    # Initialize session state with default None values
    if 'stock_params' not in st.session_state:
        st.session_state.signal = None
        st.session_state.stock_params = {
            "P/E": None,
            "P/B": None,
            "ROE": None,
            "3m_momentum": None,
            "6m_momentum": None,
            "Current_Ratio": None,
            "DilutedEPS":None,
            "EBITDA":None,
            "D/E":None,
            "Net_Profit_Margin":None,
            "Current Price": None
        }
    
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = ""
    
    if 'stock_info' not in st.session_state:
        st.session_state.stock_info = None
        st.session_state.display_symbol = ""

    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Stock Information")
        # Symbol input and fetch button
        symbol = st.text_input("Enter Stock Symbol", 
                             value=st.session_state.current_symbol,
                             key="symbol_input").upper()
        inputprice= st.number_input(f"Enter {symbol}'s price($)",key="price_input")
        if st.button("🔍 Fetch Data", key="fetch_button"):
            st.session_state.display_symbol = symbol
            try:
                # Find stock in database
                stock_info = stock_db[stock_db['Symbol'] == symbol].iloc[0]
                
                # Update stock info
                st.session_state.stock_info = {
                    'Company Name': stock_info.get('Company Name', symbol),
                    'Sector': stock_info.get('Sector', 'N/A'),
                    'Industry': stock_info.get('Industry', 'N/A')
                }
                stock_info,pricehis = get_financial_metrics(symbol,inputprice)
                # Update parameters with fetched values
                st.session_state.stock_params.update({
                    "P/E": stock_info.get('P/E'),
                    "P/B": stock_info.get('P/B'),
                    "ROE": stock_info.get('ROE'),
                    "3m_momentum": stock_info.get('3m_momentum'),
                    "6m_momentum": stock_info.get('6m_momentum'),
                    "Current Price": inputprice,
                    "D/E": stock_info.get('D/E'),
                    "Current_Ratio": stock_info.get('Current_Ratio'),
                    "Net_Profit_Margin": stock_info.get('Net_Profit_Margin'),
                    "EBITDA":stock_info.get('EBITDA'),
                    "DilutedEPS":stock_info.get('DilutedEPS')

                    
                })
                
                st.session_state.current_symbol = symbol
                st.session_state.real_price_data = pricehis
                st.session_state.real_price = inputprice
                
                st.success("Data loaded successfully!")
                
            except Exception as e:
                st.error(f"Error finding data for {symbol}: {str(e)}")
                st.session_state.stock_info = None
                st.session_state.real_price_data = None
                st.session_state.real_price = None
                st.session_state.stock_params.update({
                    "Current Price": inputprice,
                })
        
        # Editable parameters section - will now show current values
        st.subheader("Edit Parameters")
        
        # Create a copy of current params to edit
        edited_params = st.session_state.stock_params

        # Display editable inputs with current values
        for param in edited_params:
            current_value = edited_params[param]
            
            # Create a unique key for each parameter
            input_key = f"edit_{param}"
            
            if param == "Current Price":
                # For currency fields
                user_input = st.text_input(
                    f"{param} ($)",
                    value=str(current_value) if current_value is not None else "",
                    key=input_key
                )
            else:
                # For regular number fields
                user_input = st.text_input(
                    f"{param}",
                    value=str(current_value) if current_value is not None else "",
                    key=input_key
                )
            
            # Convert empty string to None, otherwise try to convert to float
            if user_input.strip() == "":
                edited_params[param] = None
            else:
                try:
                    edited_params[param] = float(user_input)

                except ValueError:
                    st.error(f"Please enter a valid number for {param}")
                    edited_params[param] = current_value  # Revert to previous value
        st.session_state.option = st.selectbox(
        "Which Model To Use?",
        ("XGboost"),key="Select_model"
        )
        if st.button("Submit For Predict", key="update_button"):
            st.session_state.stock_params = edited_params
            result, probs = predict(st.session_state.option)
            st.session_state.signal=  result
            print("predciction result",result)
            st.success("Finished!")
        
    # Rest of your display code...
    with col1:
        if st.session_state.display_symbol:
            st.subheader("Current Stock Info")
            display_name = st.session_state.stock_info['Company Name'] if st.session_state.stock_info else st.session_state.display_symbol
            st.write(f"**Company Name:** {display_name}")
            
            if st.session_state.stock_info:
                st.write(f"**Sector:** {st.session_state.stock_info['Sector']}")
            
            current_price = st.session_state.stock_params.get("Current Price")
            if current_price is not None:
                st.write(f"**Current Price:** ${current_price:.2f}")
            else:
                st.write("**Current Price:** Data not available")
    
    with col2:
        chart_title = f"Price Chart for {st.session_state.display_symbol}" if st.session_state.display_symbol else "Price Chart (No Stock Selected)"
        st.subheader(chart_title)
        
        if 'real_price_data' in st.session_state and st.session_state.real_price_data is not None:
            stock_data = st.session_state.real_price_data
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price'
            ))
            fig.update_layout(
                height=600,
                xaxis_rangeslider_visible=False,
                title=f"{st.session_state.display_symbol} Price History (1 Year)",
                yaxis_title="Price (USD)"
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                height=600,
                xaxis_rangeslider_visible=False,
                title=f"{st.session_state.display_symbol} - No price data available" if st.session_state.display_symbol else "No price data available",
                yaxis_title="Price (USD)"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Prediction Results")
        st.text("Click on submit to get the result")
        try:
            if result == 1:
                st.success(f"Based on our model, this stock shows {round(probs[0][2],2) *100}% BUY signal")
                st.info("The investment risk. Prospectus carefully before investing")
            elif result==0:
                st.warning(f"Based on our model, this stock shows {round(probs[0][1],2)*100}% HOLD signal")
                st.info("The investment risk. Prospectus carefully before investing")
            elif result==-1:
                st.error(f"Based on our model, this stock shows {round(probs[0][0],2)*100}% SELL signal")
                st.info("The investment risk. Prospectus carefully before investing")
            else:
                st.text("Click on submit to get the result")
        except: #Prevent message of unbound local error
            pass
        if st.session_state.stock_params and any(v is not None for v in st.session_state.stock_params.values()):
            st.subheader("Similar Stocks in Database")
            active_params = {k: v for k, v in st.session_state.stock_params.items() if v is not None}
            
            similar_stocks = find_closest_stocks(stock_db, active_params)
            display_columns = ['Date','Symbol','ROE','P/E','P/B','D/E','Current_Ratio','Net_Profit_Margin','EBITDA','DilutedEPS','3m_momentum','6m_momentum']
            if not similar_stocks.empty:
                display_columns = [col for col in display_columns if col in similar_stocks.columns]
                st.dataframe(similar_stocks[display_columns],hide_index=True)
            else:
                st.write("No similar stocks found with current parameters")

# Main app
def main():
    # Initialize stock database
    try:
        stock_db = load_stock_db()
    except Exception as e:
        st.error(f"Failed to load stock database: {str(e)}")
        return
    

    stock_analyzer(stock_db)


if __name__ == "__main__":
    main()
