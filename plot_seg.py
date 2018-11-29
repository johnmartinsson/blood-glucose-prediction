from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt

def plot_seg(pid):
    data = np.loadtxt('seg.csv')

    fig, ax = plt.subplots()
    ax.set_title('Patient {} Surveillance Error Grid'.format(pid))
    ax.set_xlabel('Reference Concentration (mg/dl)')
    ax.set_ylabel('Predicted Concentration (mg/dl)')
    cax = ax.imshow(np.transpose(data), origin='lower', interpolation='nearest')
    cbar = fig.colorbar(cax, ticks=[0.25, 1.0, 2.0, 3.0, 3.75], orientation='vertical')
    cbar.ax.set_yticklabels(['None', 'Mild', 'Moderate', 'High', 'Extreme'],
            rotation=90, va='center')
    #cbar.ax.set_title("Risk level")

    return fig, ax

def plot_seg_with_reference_and_predictions(pid, references, predictions):
    fig, ax = plot_seg(pid)
    plt.scatter(references, predictions, s=25, facecolors='white', edgecolors='black')
    return fig, ax

def main():
    plot_seg_with_reference_and_predictions([10,10,50], [70, 15, 300])
    plt.show()

if __name__ == '__main__':
    main()
