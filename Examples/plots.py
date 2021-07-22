from matplotlib import pyplot as plt
import matplotlib.dates as md
from sippy import functionsetSIM as fsetSIM
import numpy as np
from scipy import signal
from  control import  ss, step_response, dcgain

def plot_comparison(step_test_data, model, pad_len, inputs, outputs, start_time, end_time, plt_input=False):
    """
    Plot the predicted and true output-signals.
    
    :param step_test_data: dataframe bject of loaded data.
    :param model: npz model file.
    :param pad_len: data pading legth to remove simulation artifacts.
    :param inputs: Input vectors of the model.
    :param outputs: Output vectors of the model.
    :param start_time: Starting time of prediction data.
    :param end_time: Ending time of prediction data.
    :param plt_output: Boolean whether to Input vectors.
    :param scale_plt: Boolean whether to scale ouput vector plots.
    """
    
    val_data = step_test_data.loc[start_time:end_time]
    
    Time = val_data.index
    u = val_data[inputs].to_numpy().T
    y = val_data[outputs].to_numpy().T
    # y_init = np.array([[item[:10].mean()] for item in y])
    

    # Use the model to predict the output-signals.
    mdl = np.load(model)
    X0 = mdl['X0']
    n = len(mdl['A'])
    # X0[-len(y_init):] = y_init
    # The output of the model
    # xid, yid = fsetSIM.SS_lsim_innovation_form(A=mdl['A'], B=mdl['B'], C=mdl['C'], D=mdl['D'], K=mdl['K'], y=y, u=u, x0=X0)
    xid, yid = fsetSIM.SS_lsim_process_form(A=mdl['A'], B=mdl['B'], C=mdl['C'], D=mdl['D'],u=u, x0=mdl['X0'])
    
    yid[:,:pad_len] = yid[:,pad_len+1].reshape((yid.shape[0],1))
    y_mean_val = y[:,:].mean(axis=1)
    yid_mean_val = yid[:,:].mean(axis=1)
    yid[:,:] = yid+(y_mean_val-yid_mean_val).reshape(yid.shape[0],1)
    # Make the plotting-canvas bigger.
    plt.rcParams['figure.figsize'] = [12, 4]
    # For each output-signal.
    for idx in range(0,len(outputs)):
        plt.figure(idx)
        plt.xticks(rotation=15)
        plt.plot(Time, y[idx],color='r')
        plt.plot(Time, yid[idx],color='b')
        plt.ylabel(outputs[idx])
        plt.grid()
        plt.xlabel("Time")
        plt.title('output_'+ str(idx+1))
        plt.legend(['measurment', 'prediction'])
        ax=plt.gca()
        xfmt = md.DateFormatter('%m-%d-%yy %H:%M')
        ax.xaxis.set_major_formatter(xfmt)        
        
    if plt_input == True:
        for idx in range(len(outputs), len(outputs) + len(inputs)):
            plt.figure(idx)
            plt.xticks(rotation=15)
            plt.plot(Time, u[idx-len(outputs)], color='r')
            plt.ylabel(inputs[idx-len(outputs)])
            plt.grid()
            plt.xlabel("Time")
            plt.title('input_'+ str(idx-len(outputs)+1))
            ax=plt.gca()
            xfmt = md.DateFormatter('%m-%d-%yy %H:%M')
            ax.xaxis.set_major_formatter(xfmt) 
    plt.show()

def plot_model(model, inputs, outputs, tss=90, dt=1):
    """
    Plot the model matrix.

    :param model: npz model file.
    :param inputs: Input vectors of the model.
    :param outputs: Output vectors of the model
    :Param tss: time to steady state (length of x axis of subplot).
    """
    mdl = np.load(model)
    sys = ss(mdl['A'], mdl['B'], mdl['C'], mdl['D'],dt)
    # gain_matrix = dcgain(sys).T
    num_i = len(inputs)
    num_o = len(outputs)
    fig, axs = plt.subplots(num_i,num_o, figsize=(3*len(outputs), 2*len(inputs)), facecolor='w', edgecolor='k')
    fig.suptitle('step_response: '+model.rsplit('.',1)[0])
    T = np.arange(0,tss, step=dt)
    for idx_i in range(num_i):
        for idx_o in range(num_o):
            if len(range(num_o)) < 2:
                ax = axs[idx_i]
            else:
                ax = axs[idx_i][idx_o]
            t, y_step = step_response(sys,T, input=idx_i, output=idx_o)
            gain = round(y_step[-1],4)
            ax.plot(t, y_step,color='r')
            if idx_i == 0:
                ax.set_title(outputs[idx_o], rotation='horizontal', ha='center', fontsize=10)
            if idx_o == 0:
                ax.set_ylabel(inputs[idx_i], rotation=90, fontsize=10)
            ax.grid(color='k', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='x', colors='red',size=0,labelsize=4)
            ax.tick_params(axis='y', colors='red',size=0,labelsize=4)
            ax.annotate(str(gain),xy=(.72,.8),xycoords='axes fraction')
    # fig.tight_layout()
    plt.show()
def plot_freuency_response(b):
    """
    This function plots the frequency response of FIR array.
    :param b: finite impulse response array.
    """
    w,h =signal.freqz(b)
    plt.plot(w,np.abs(h),'g')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Amplitude (db)',color='b')
    plt.xlabel('Freuency (rad/sample)')
    plt.show()