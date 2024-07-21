### About the Model

- Developed a long-short term memory (LSTM) model in Python to predict stock prices for big companies such as Google, JNJ, BRK.B, based on trend, momentum, volume, and volatility factors; fine-tuned hyperparameters with Bayesian optimization and grid search such as epochs, time steps, neuron per layer, and the batch size; and achieved 92% out-of-sample accuracy.

### Run the Program File: Stock_Prediction_LIVE_ver1.3.5.py

#1. Use the following commands in the terminal to run this program file. Change directory to the file path:

```bash
cd
```


#2. Add a output folder: `mkdir output` 


#3. An example to run this program:

```bash
cd /Users/shihanq/Desktop/Version1.3.6/ # This is the directory that contains source files
python3 Stock_Prediction_LIVE_ver1.3.6.py
```




### About the Program

#1. After running the command above, a user interface will show up.

User Interface: The interface contains a search bar for stock symbol input.

Input: code for stock of interest
- For stocks in US stock market, use all-letter codes.
  for example: “AAPL” for apple, “GOOGL” for google, “AMZN” for amazon.
  
- For stocks on Hongkong stock exchange, use 4-digit code followed by ".HK"
  for example: “0700.HK” for Tencent, “2318.HK” for Ping An Insurance Company. 
  
- For stocks on Shanghai stock exchange, use 6-digit code followed by ".SS"
  for example: “601398.SS” for Industrial and Commercial Bank of China Limited, “600519.SS” for Kweichow Moutai Co..
  
- For stocks on Shenzhen stock exchange, use 6-digit code followed by ".SZ"
  for example: “002594.SZ” for BYD Company Limited, “300760.SZ” for Shenzhen Mindray Bio-Medical Electronics Co..

Output: plot of 90- days stock price data.


#2. Choose an algorithm and choose the training data size, then click Train A Model button.
Algorithm: Currently there are 3 algorithms (LSTM, CNN, BiLSTM), and we use LSTM as the default model to train and predict stock price.

Training data size: The default is 2000 previous data points, that is, stock prices in previous 2000 trading days. 

Train A Model: Click this button to train current model using stock data. 

Training Progress: After the previous step, a progress bar will show up at the bottom and you can see the training progress there. 


#3. Choose an Input Date and Click “Predict”: 

Input Date should follow the format: %y-%m-%d (eg. 2023-04-17), and weekends and holidays are invalid. 
Click “Plot Chart” to output a chart which indicates predicted trends.

Prediction can only be made for the next unknown closing price. 
Predictions except tomorrow in the future is not allowed due to poor accuracy using the current model.


#4. Adjust Percent and Click “Re-Plot”:

The percent indicates the interval given the allowed percent of deviation from the predicted price. 


#5. Click Proof to show the actual closing price, accuracy, MAPE.


