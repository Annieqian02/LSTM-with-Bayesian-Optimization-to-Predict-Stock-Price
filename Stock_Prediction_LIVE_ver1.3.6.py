from tkinter import *
import tkinter as tk
from tkinter import ttk
import json
from multiprocess import Process, Queue, Manager
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

from datetime import datetime
from datetime import date as dt
from tkcalendar import DateEntry
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Building the RNN
# from opts import parse_opts
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from ModelTrain import *

# Time Period
temp = None
time_steps = 30

# Create a window
root = tk.Tk()

####### App Name #######
root.title('AI Stock Prediction 1.3.6')

# Size of the window
root.geometry('1200x800')

####### Set up Framework #######
# Split the window into two parts
topFrame = Frame(root)
topFrame.pack(side=TOP, padx=100)
bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM, padx=10, pady=50)
# bottomFrame.pack(anchor='center', padx=10, pady=50)

topLeftFrame = Frame(root)
topLeftFrame.pack(side=LEFT, padx=150)
topRightFrame = Frame(root)
topRightFrame.pack(side=LEFT)

'''Top Frame Set'''
####### Place Title #######
title = tk.Label(topLeftFrame, text="     Stock price prediction     ")
title.config(font=("Courier", 28))
title.pack(side=TOP,pady=10)

class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder="Search for stock symbols", color='grey'):
        super().__init__(master)
        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']
        self.configure(width=25)
        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)
        self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

    def foc_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self['fg'] = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()

InputSearchEntry = EntryWithPlaceholder(topLeftFrame, placeholder='Search for stock symbols', color='grey')
InputSearchEntry.pack(side=TOP)
InputSearchEntry.config(font=("Arial", 15))

def click_outside(event):
    widget = event.widget
    if not isinstance(widget, tk.Entry) or not widget.winfo_ismapped():
        root.focus()

root.bind("<Button-1>", click_outside)

####### Place Logo #######
path = "./visionx-logo.png"
logo = Image.open(path)
logo = logo.resize((200, 200), Image.Resampling.LANCZOS)
# Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
logo = ImageTk.PhotoImage(logo)
# The Label widget is a standard Tkinter widget used to display a text or image on the screen.
panel = tk.Label(topRightFrame, image=logo)
panel.pack(side=RIGHT, pady=15)

''' Bottom Frame Set '''
####### Bottom Frame Split #######
# Split Bot Frame to two parts, for text&chart
BotLeftFrame = Frame(bottomFrame)
BotLeftFrame.pack(side=LEFT, padx=5)
BotMidFrame = Frame(bottomFrame)
BotMidFrame.pack(side=LEFT)
BotRightFrame = Frame(bottomFrame)
BotRightFrame.pack(side=LEFT)

test_close_plot, y_pred_group, y_pred_v, t = '', '', '', ''

####### show model info label ######

doneLabel = None

def remove_done_label():
    global doneLabel
    # remove the "Done" label is visible, remove it
    if doneLabel and doneLabel.winfo_exists():
        doneLabel.destroy()
        doneLabel = None

def search_button():
    global modelfile, MT, df, Stock_symbol
    remove_done_label()
    Stock_symbol = str(InputSearchEntry.get()).upper()

    try:
        def show_price_chart(end_date = None):
            if end_date is None:
                end_idx = len(df)
            else:
                # find the closest date to the end date (should be the same or the earlier date)
                tmp_df = df[df['date'] <= end_date]
                end_idx = len(tmp_df)
            start_idx = end_idx - 90 if end_idx > 90 else 0
            end_date = df['date'].iloc[end_idx-1].strftime('%Y-%m-%d')
            print("plotting...")
            ####### Draw Chart #######
            f = Figure(figsize=(7, 5), dpi=50)
            a = f.add_subplot(111)
            # a.plot(df['date'][-90:, ], df['Close'][-90:, ], color='red', label=str('Real ' + Stock_symbol + ' Stock Price'))
            a.plot(df['date'].iloc[start_idx:end_idx], df['Close'].iloc[start_idx:end_idx], color='red', label=str('Real ' + Stock_symbol + ' Stock Price'))
            a.set_title('90 Days of {} (ends {})'.format(Stock_symbol, end_date))
            a.set_xlabel('Date')
            a.set_ylabel(str(Stock_symbol + ' Close Price'))
            date_form = DateFormatter("%m-%d")
            a.xaxis.set_major_formatter(date_form)
            a.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

            canvas = FigureCanvasTkAgg(f, BotLeftFrame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0)


        try:
            with open('model_info.json', 'r') as openfile:
                model_info = json.load(openfile)
            MT = ModelTrain(Stock_symbol, 2000)
            MT.SetIndicator(model_info.get('indicator'))
            MT.SetDataSize(int(model_info.get('data size')))
            df = MT.df
        except FileNotFoundError:
            MT = ModelTrain(Stock_symbol, 2000)
            df = MT.df
        show_price_chart()


        def ShowInfo():
            global InputDate, Pred, Act, Acc, MAPE, Daily_return, daily_return,InputDateEntry, pred, act, acc, mape, Algorithm, selected_algorithm, algorithm_button, Source, selected_source, source_button, selected_size, selected_indicator
            ####### Bottom Mid Frame for Text #######
            InputDate = tk.Label(BotMidFrame, text='Input Date', fg='black')
            InputDate.grid(row=0, column=1)
            Pred = tk.Label(BotMidFrame, text='', fg='black')
            Pred.grid(row=1, column=1)
            Act = tk.Label(BotMidFrame, text='', fg='black')
            Act.grid(row=2, column=1)
            Acc = tk.Label(BotMidFrame, text='', fg='black')
            Acc.grid(row=3, column=1)
            MAPE = tk.Label(BotMidFrame, text='', fg='black')
            MAPE.grid(row=4, column=1)
            Daily_return = tk.Label(BotMidFrame, text='', fg='black')
            Daily_return.grid(row=5, column=1)

            #######Bottom Left Frame for Input/Output #######
            start = str(MT.df.date.values.astype('datetime64[D]')[0]).split('-')
            end = str(MT.df.date.values.astype('datetime64[D]')[-1]).split('-')
            InputDateEntry = DateEntry(BotMidFrame, selectmode='day', date_pattern='y-mm-dd',
                                       firstweekday='sunday', locale='en_US', showweeknumbers=False,
                                       mindate=dt(year=int(start[0]), month=int(start[1]), day=int(start[2])),
                                       maxdate=dt(year=int(end[0]), month=int(end[1]), day=int(end[2])),
                                       font='bold', normalforeground='black', weekendforeground='#d3d3cd',
                                       othermonthforeground='black', othermonthweforeground='#d3d3cd',
                                       foreground="black", background="lightgrey", selectforeground="blue",
                                       disableddayforeground="#d3d3cd", disabledbackground="lightgrey")
            InputDateEntry.grid(row=0, column=2)
            InputDateEntry.config(width=10)

            # add a callback to the date entry widget
            InputDateEntry.bind('<<DateEntrySelected>>', lambda e: show_price_chart(InputDateEntry.get()))

            pred = tk.Label(BotMidFrame, text='', fg='black')
            pred.grid(row=1, column=2)
            act = tk.Label(BotMidFrame, text='', fg='black')
            act.grid(row=2, column=2)
            acc = tk.Label(BotMidFrame, text='', fg='red')
            acc.grid(row=3, column=2)
            mape = tk.Label(BotMidFrame, text='', fg='red')
            mape.grid(row=4, column=2)
            daily_return = tk.Label(BotMidFrame, text='', fg='red')
            daily_return.grid(row=5, column=2)

            ####### algorithm button #######        
            Algorithm = tk.Label(BotMidFrame, text="Choose an Algorithm", justify=LEFT, padx=30)
            Algorithm.grid(row=0, column=0)
            Algorithm.config(width=9)

            selected_algorithm = tk.StringVar()
            selected_algorithm.set('LSTM')
            algorithm_button = OptionMenu(BotMidFrame, selected_algorithm, 'LSTM', 'CNN', "BiLSTM")
            algorithm_button.config(width=9)
            algorithm_button.grid(row=1, column=0)

            ####### data size button ######
            Data_Size = tk.Label(BotMidFrame, text="Choose Training Data Size", padx=50)
            Data_Size.grid(row=2, column=0)
            Data_Size.config(width=9)

            selected_size = tk.IntVar()
            selected_size.set(2000)
            size_button = OptionMenu(BotMidFrame, selected_size, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
            size_button.config(width=9)
            size_button.grid(row=3, column=0)

            ####### indicator button ######
            Indicator = tk.Label(BotMidFrame, text="Choose Indicators to Transfer Data", padx=70)
            Indicator.grid(row=4, column=0)
            Indicator.config(width=9)

            selected_indicator = tk.StringVar()
            selected_indicator.set('Standard Indicators')
            indicator_button = OptionMenu(BotMidFrame, selected_indicator, 'Standard Indicators',
                                          'Standard Indicators \nwith Stochastic Oscillator')
            indicator_button.config(width=15)
            indicator_button.grid(row=5, column=0)

        ShowInfo()

        ####### train Button #######
        def train_button():
            global MT, modelfile, proc, df, doneLabel
            remove_done_label()
            algorithm = selected_algorithm.get()
            data_size = selected_size.get()
            indicator = selected_indicator.get().replace('\n', '')
            print(f'Runing on {algorithm} algorithm with {str(data_size)} data size and {indicator}')
            MT.SetAlgorithm(algorithm)
            MT.SetIndicator(indicator)
            MT.SetDataSize(data_size)
            df = MT.df

            def f(MT: ModelTrain, q: Queue, return_dict):
                def progress(current_step, total_step):
                    q.put((current_step, total_step))
                modelfile, loss, val_loss = MT.GetModel(progress=progress)
                print(modelfile)
                return_dict['modelfile'] = modelfile
                return_dict['loss'] = loss
                return_dict['val_loss'] = val_loss

            model_info = {'algorithm': algorithm,
                          'data size': data_size,
                          'indicator': indicator
                          }
            with open("model_info.json", "w") as outfile:
                json.dump(model_info, outfile)

            q = Queue()
            manager = Manager()
            return_dict = manager.dict()
            proc = Process(target=f, args=[MT, q, return_dict])
            proc.start()
            total_step = 20
            current_step = 0
            pb = ttk.Progressbar(BotMidFrame, orient="horizontal", length=200, mode="determinate")
            pb.grid(column=0, row=8, columnspan=2, padx=10, pady=20)

            while current_step != total_step:
                current_step, total_step = q.get()
                # print("current_step: ", current_step, "total_step: ", total_step)
                percentage = int((current_step / total_step) * 100)# // 5
                pb["value"] = percentage
                # render the progress bar immediately
                if current_step == total_step:
                    # add a label to show "Done"
                    doneLabel = tk.Label(BotMidFrame, text="Done", fg="green")
                    doneLabel.grid(column=2, row=8)
                root.update()

            # pb.stop()
            proc.join()
            modelfile = return_dict['modelfile']

        def stop_button():
            if os.path.exists("model_info.json"):
                os.remove("model_info.json")
            proc.terminate()
            proc.join()
            root.update()

        ####### Load model #######
        def LoadModel():
            global model, MT

            try:
                # load original model
                modelfile = MT.GetOriModel()
            except ValueError as detail:
                tk.messagebox.showerror("No Model Warning",
                                        "There is no model in the folder, please train a model first!")

            filepath = str(modelfile)
            model = load_model(filepath)

        # This funcion is use for calculate Accuracy and plot
        def pre_plot():
            global test_close_plot, y_pred_group, y_pred_v, t
            test_date_index = np.where(df[date].values.astype('datetime64[D]') == temp)[0][0]
            test_date = np.array(X_t[test_date_index - 30])
            t = np.array(df[date].values.astype('datetime64[D]')[test_date_index - 30:test_date_index])
            test_info = df[test_date_index - 30:test_date_index]
            test_close_plot = test_info[cp]

            ####### Predict that day #######
            test_date = np.array(test_date)
            y_pred = model.predict(test_date[np.newaxis, ...])
            y_pred_v = y_sc.inverse_transform(y_pred[0].reshape(-1, 1))[0][0]

            ####### Predict pervious 30 days #######
            num_pred_group = len(y) - time_steps
            # shape 30*30*18
            test_pred_group = X_t[num_pred_group - 30:num_pred_group]
            y_pred_group = model.predict(test_pred_group)
            y_pred_group = y_sc.inverse_transform(y_pred_group.reshape(-1, 1))

        def Pred_button():
            global temp, X, y, X_sc, y_sc, X_t, y_t, percent

            LoadModel()
            X, y = MT.X, MT.y
            X_sc, y_sc, X_t, y_t = MinMax(X, y)

            temp = InputDateEntry.get_date()

            test_date_list = df[date].values.astype('datetime64[D]')
            print(test_date_list[-5:])
            if temp not in test_date_list:
                if (temp > test_date_list[-1]):
                    tk.messagebox.showerror("Error",
                                            "Prediction can only be made for the next unknown closing price. Predictions further into the future is not allowed due to poor accuracy using the current model.")
                else:
                    tk.messagebox.showerror("Error", "Weekend and holidays are invalid dates.")
            else:
                test_date_index = np.where(test_date_list == temp)[0][0]
                test_date = np.array(X_t[test_date_index - 30])
                test_date_value = np.array(y_t[test_date_index - 30])
                test_date = np.array(test_date)
                # 使用 np.newaxis 来扩展一个维度 (30, 18) ==> (None, 30, 18)
                test_pred = model.predict(test_date[np.newaxis, ...])
                test_pred_value = y_sc.inverse_transform(test_pred[0].reshape(-1, 1))[0][0]

                test_real = np.array(y[test_date_index])

                if (test_pred_value > test_real):
                    t = 'Increasing'
                    # v = (test_pred_value - test_real)/test_real*100
                    v = (test_pred_value - test_real)
                elif (test_pred_value < test_real):
                    t = 'Decreasing'
                    # v = (test_real - test_pred_value)/test_real*100
                    v = (test_pred_value - test_real)

                pre_plot()

                Pred.configure(text="Predicted Price")
                Pred.bind("<Enter>", lambda e: e.widget.config(text="2nd day's Price"))
                Pred.bind("<Leave>", lambda e: e.widget.config(text="Predicted Price"))
                Act.configure(text=t)
                Acc.configure(text='')
                MAPE.configure(text='')
                Daily_return.configure(text='')

                pred.configure(text=str(test_pred_value))
                # act.configure(text=str(str(round(v, 2))+"%"))
                act.configure(text=str(round(v, 2)))
                acc.configure(text='')
                mape.configure(text='')
                daily_return.configure(text='')

                Plot_button = tk.Button(BotMidFrame, text='Plot Chart', font='Arial', relief='groove', bg='white',
                                        command=plot_button)
                Plot_button.grid(row=6, column=2)

        '''
        Pred_button = tk.Button(BotMidFrame, text ='Predict',font = 'Arial', bg = 'white',activebackground = 'black', 
                                command=Pred_button)
        Pred_button.grid(row=5, column=1, pady=5)
        '''

        def Proof_button():
            global temp, X, y, X_sc, y_sc, X_t, y_t
            LoadModel()
            X, y = MT.X, MT.y
            X_sc, y_sc, X_t, y_t = MinMax(X, y)

            temp = InputDateEntry.get_date()

            test_date_list = df[date].values.astype('datetime64[D]')
            print(test_date_list[-5:])

            if temp not in test_date_list:
                tk.messagebox.showerror("Error", "Not a valid date")
            else:
                test_date_index = np.where(test_date_list == temp)[0][0]
                # tomorrow's pred value
                test_date = np.array(X_t[test_date_index - 30])
                test_date_value = np.array(y_t[test_date_index - 30])

                test_pred = model.predict(test_date[np.newaxis, ...])
                test_pred_value = y_sc.inverse_transform(test_pred[0].reshape(-1, 1))[0][0]

                # today's pred value
                test_today_date = np.array(X_t[test_date_index - 31])
                test_today_date_value = np.array(y_t[test_date_index - 30])  # ******

                test_today_pred = model.predict(test_date[np.newaxis, ...])
                test_today_pred_value = y_sc.inverse_transform(test_today_pred[0].reshape(-1, 1))[0][0]

                # today's real value
                test_real = np.array(y[test_date_index])

            pre_plot()

            Pred.configure(text="Predicted Closing Price")
            Pred.bind("<Enter>", lambda e: e.widget.config(text="Today's  Closing  Price"))
            Pred.bind("<Leave>", lambda e: e.widget.config(text="Predicted Closing Price"))
            Act.configure(text="Actual Closing Price")
            Act.bind("<Enter>", lambda e: e.widget.config(text="Today's Actual Price"))
            Act.bind("<Leave>", lambda e: e.widget.config(text="Actual Closing Price"))
            Acc.configure(text='Accuracy')
            MAPE.configure(text='MAPE')
            Daily_return.configure(text='Daily return')

            pred.configure(text=str(test_today_pred_value))
            act.configure(text=str(test_real))

            ####### Accuracy Rate #######
            a = test_real
            b = test_today_pred_value
            Accuracy = (1 - abs((a - b) / a)) * 100
            acc.configure(text=(str(round(Accuracy, 4)) + "%"))

            ####### MAPE Rate #######
            c = np.array(test_close_plot).reshape(30, 1)
            d = np.array(y_pred_group)
            MAPE_Rate = np.sum(abs(c - d) / c) / time_steps
            mape.configure(text=(str(round(MAPE_Rate, 4))))

            ####### Daily_return Rate #######
            e = test_real
            f = np.array(y[test_date_index - 1])
            Daily = (e / f - 1) * 100
            daily_return.configure(text=(str(round(Daily, 4)) + "%"))
        '''
        Proof_button = tk.Button(BotMidFrame, text ='Proof',font = 'Arial', bg = 'white',activebackground = 'black', 
                                 command=Proof_button)
        Proof_button.grid(row=6, column=1, pady = 5)
        '''

        Train_button = tk.Button(BotMidFrame, text='Train a Model', font='Arial', relief='groove', bg='white',
                                 command=train_button)
        Train_button.grid(row=6, column=0)

        Stop_button = tk.Button(BotMidFrame, text='Stop Training', font='Arial', relief='groove', bg='white',
                                command=stop_button)
        Stop_button.grid(row=7, column=0)

        Pred_button = tk.Button(BotMidFrame, text='Predict', font='Arial', bg='white', activebackground='black',
                                command=Pred_button)
        Pred_button.grid(row=6, column=1, pady=5)

        Proof_button = tk.Button(BotMidFrame, text='Proof', font='Arial', bg='white', activebackground='black',
                                 command=Proof_button)
        Proof_button.grid(row=7, column=1, pady=5)

    except KeyError as detail:
        tk.messagebox.showerror("Error", "Symbol Not Found")
    except ValueError as detail:
        tk.messagebox.showerror("Error", detail)


##############################################################################################################################
# Search_button = tk.Button(topLeftFrame, text ='Search',font = ('Arial', 20), bg = 'black',activebackground = 'white', command=search_button)
Search_button = tk.Button(topLeftFrame, text='Predict', font=('Arial', 20), relief='groove', bg='white',
                          command=search_button)

Search_button.pack(side=BOTTOM,pady=20)


##############################################################################################################################

####### Give X, y, get X_sc, y_sc, X_t, y_t ########
def MinMax(X, y):
    X_sc = MinMaxScaler(feature_range=(0, 1))
    y_sc = MinMaxScaler(feature_range=(0, 1))
    X_MinMax = X_sc.fit_transform(X)
    y_MinMax = y_sc.fit_transform(np.array(y).reshape(-1, 1))

    X_t = []
    y_t = []

    for i in range(time_steps, len(y)):
        X_t.append(X_MinMax[i - time_steps:i])
        y_t.append(y_MinMax[i])
    X_t, y_t = np.array(X_t), np.array(y_t)
    return X_sc, y_sc, X_t, y_t


# default data and cp
date = 'date'
cp = 'Close'


####### Plot Button #######
def plot_button():
    global percent, InputShadow

    shadow = tk.Label(BotRightFrame, text='Adjust Percent(3% original)', fg='black', bg='white')
    shadow.grid(row=0)

    InputShadow = tk.Entry(BotRightFrame)
    InputShadow.grid(row=1)

    percent = 3

    ####### Draw Chart #######
    percent = int(percent)
    flat_list = [item for sublist in y_pred_group for item in sublist]
    f1 = [i * (1 - int(percent) / 100) for i in flat_list]
    f2 = [i * (1 + int(percent) / 100) for i in flat_list]

    f = Figure(figsize=(7, 4), dpi=50)
    f.tight_layout()
    a = f.add_subplot(111)
    # b = f.add_subplot(111)
    ax = f.add_subplot(111)
    # clear x tick label
    ax.set_xticklabels([])
    a.set_xticklabels([])
    ax.fill_between(t, f1, f2, facecolor='yellow', interpolate=True)
    # ax.fill_between(t, test_close_plot, flat_list, where=flat_list >= test_close_plot, facecolor='yellow', interpolate=True)
    # ax.fill_between(t, test_close_plot, flat_list, where=flat_list <= test_close_plot, facecolor='yellow', interpolate=True)
    a.plot(t, test_close_plot, color='red', label=str('Real ' + Stock_symbol + ' Stock Price'))
    # clear x label
    a.plot(np.array(datetime.strptime(str(temp), '%Y-%m-%d')), np.array(y_pred_v), marker='x', color='blue',
           label='Predict Close Price')

    a.set_title(str(Stock_symbol + ' Stock Price Prediction'))
    a.set_xlabel('Time')
    a.set_ylabel(str(Stock_symbol + ' Stock Price'))
    a.legend(bbox_to_anchor=(0.05, 0.92, 1, .102), loc=3,
             ncol=2, borderaxespad=0)
    date_form = DateFormatter("%m-%d")
    a.xaxis.set_major_formatter(date_form)
    a.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    f.autofmt_xdate()
    canvas = FigureCanvasTkAgg(f, BotRightFrame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=2)

    Replot_button = tk.Button(BotMidFrame, text='Re-Plot', font='Arial', relief='groove', bg='white',
                              command=replot_button)
    Replot_button.grid(row=7, column=2)


def replot_button():
    global percent
    try:
        percent = InputShadow.get()

        flat_list = [item for sublist in y_pred_group for item in sublist]
        f1 = [i * (1 - int(percent) / 100) for i in flat_list]
        f2 = [i * (1 + int(percent) / 100) for i in flat_list]

        f = Figure(figsize=(7, 4), dpi=50)
        f.tight_layout()
        a = f.add_subplot(111)
        # b = f.add_subplot(111)
        ax = f.add_subplot(111)
        # clear x tick label
        ax.set_xticklabels([])
        a.set_xticklabels([])

        ax.fill_between(t, f1, f2, facecolor='yellow', interpolate=True)
        # ax.fill_between(t, test_close_plot, flat_list, where=flat_list >= test_close_plot, facecolor='yellow', interpolate=True)
        # ax.fill_between(t, test_close_plot, flat_list, where=flat_list <= test_close_plot, facecolor='yellow', interpolate=True)
        a.plot(t, test_close_plot, color='red', label=str('Real ' + Stock_symbol + ' Stock Price'))
        # b.plot(t, y_pred_group,color='blue', label='Predicted Intel Stock Price')
        a.plot(np.array(datetime.strptime(str(temp), '%Y-%m-%d')), np.array(y_pred_v), marker='x', color='blue',
               label='Predict Close Price')

        a.set_title(str(Stock_symbol + ' Stock Price Prediction'))
        a.set_xlabel('Date')
        a.set_ylabel(str(Stock_symbol + ' Stock Price'))
        a.legend(bbox_to_anchor=(0.05, 0.92, 1, .102), loc=3,
                 ncol=2, borderaxespad=0)
        date_form = DateFormatter("%m-%d")
        a.xaxis.set_major_formatter(date_form)
        a.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        f.autofmt_xdate()
        canvas = FigureCanvasTkAgg(f, BotRightFrame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2)

    except ValueError:
        tk.messagebox.showerror("Error", "No Adjust Percentile Found")

# only run gui on main process
if __name__ == '__main__':
    root.mainloop()
