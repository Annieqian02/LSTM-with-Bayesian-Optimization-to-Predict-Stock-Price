### Version1.3.6

**Updates on Version1.3.6: 1. Bug fixes and usability improvement. 2. Packages updates. 3. Instruction on predicting stocks in different stock markets



### Part1: Set the Environment

Here are two methods to set up the environment:

- Use Anaconda (This method could manage the environment easily, but it may need more time to install the environment.)
- Use pure Python (The installation will be a bit faster than using Anaconda, but we could not use GUI to manage the environment.)

**Please choose a method above.**


## Use Anaconda

#1. Install Anaconda: 

Go to the Anaconda installation page: <https://www.anaconda.com/products/distribution#download-section>

Follow the instructions on the following link to install Anaconda: <https://docs.anaconda.com/anaconda/install/>

One reason we recommend using Anaconda is that you can create virtual environments for your requirement. The virtual environments will not affect each other.


#2. Set virtual environment (CPU only)

TensorFlow

First, open the terminal. This can be done in two ways:
- open the terminal in your computer by clicking the terminal icon on the dock
- or open Anaconda, then launch the jupyter notebook. Press the New drop-down button on the top right corner of the page and select terminal

Once the terminal is opened, run the following commands in the terminal:

```bash
conda create -n tf_cpu tensorflow python=x.x
# "python=x.x" is optional, as you could use the following command instead. **please choose only one command (not both)**

# Or use
conda create -n tf_cpu tensorflow
```


#3. Activate the environment

Please active the environment by running the following command:

```bash
conda activate tf_cpu
# If you see an error message saying "CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'," try running the following code to activate the environment.
source activate tf_cpu # try this only if "conda activate tf_cpt" does not work

# If you want to stop using the virtual environment, please use the following command. But we do not need to do this now!
conda deactivate
```


#4. Install Required Modules:

In Anaconda and your virtual environment, using “​conda install​” to install all the packages. (Modules includes: matplotlib, pandas, PIL, keras, sklearn, yfinance)

Please refer to the attachment "requirements.txt" for more version details:

Note that, the `conda` command needs more time to resolve the dependencies and we are simpily using tensorflow CPU. **We could just using `pip` command to setup packages.**

Please run command in the directory that contains `requirements.txt`. If not, please use `cd` command to go to the directory that contains this file. (`cd/path/to/directory`, where the `/path/to/directory` is the dir that contains the file)

```bash
# assume that `requirements.txt` is contains in dir `env`
# you would find some infomation as
# username@host:/path/to/file$
# or
# PS C:\path\to\env
# or
# C:\path\to\env>
# then, run the following command:

pip3 install -r requirements.txt
```



## Use Python

If you want to use pure Python, please install Python first. You could follow the documation on python.org to setup **Python 3.9**: [for windows](https://docs.python.org/3/using/windows.html), [for MacOS or Linux](https://docs.python.org/3/using/unix.html).

**Note:** If you are trying to install python on windows, please [check the tcl/tk feature](https://stackoverflow.com/a/59970646/12349560), or if you are trying to install python on MacOS or Linux, please also ([check this thread](https://stackoverflow.com/questions/25905540/importerror-no-module-named-tkinter)).

Then, please refer to the attachment "requirements.txt for more version details:

Note that, the `conda` command needs more time to resolve the dependencies and we are simpily using tensorflow CPU. **We could just using `pip` command to setup packages.**

Please run command in the directory that contains `requirements.txt`. If not, please use `cd` command to go to the directory that contains this file. (`cd/path/to/directory`, where the `/path/to/directory` is the dir that contains the file)

```bash
# assume that `requirements.txt` is contains in dir `env`
# you would find some infomation as
# username@host:/path/to/file$
# or
# PS C:\path\to\env
# or
# C:\path\to\env>
# then, run the following command:

pip3 install -r requirements.txt
```








### Part 2: Run the Program File: Stock_Prediction_LIVE_ver1.3.5.py

#1. Use the following commands in the terminal to run this program file. Change directory to the file path:

```bash
cd
```


#2. Add a output folder: `mkdir output` // IMPORTANT!!!

**Note that, if you are using Anaconda and the environment is not actived (as the command line doesn't show something like `tf_cpu`), please run command `conda activate tf_cpu` again to active this environment!!!**


#3. An example to run this program:

```bash
cd /Users/jing/Desktop/Version1.3.6/ # This is the directory that contains source files
python3 Stock_Prediction_LIVE_ver1.3.6.py
```







### Part 3: About the Program

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


