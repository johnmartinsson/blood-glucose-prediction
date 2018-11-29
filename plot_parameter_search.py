import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set()

def main():
    artifacts_path = sys.argv[1]
    d = {}
    nb_lstm_statess = [8, 32, 96, 128, 256, 512]
    #nb_lstm_statess = [8, 32, 128, 256, 512]
    #nb_past_stepss = [12, 24]
    nb_past_stepss  = [6, 12, 24, 36]
    seeds = [0, 1, 2, 3, 4]
    #seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    results = np.zeros((len(seeds), len(nb_lstm_statess), len(nb_past_stepss)))
    for i_step, nb_past_steps in enumerate(nb_past_stepss):
        for i_state, nb_lstm_states in enumerate(nb_lstm_statess):
            for i_seed, seed in enumerate(seeds):
                experiment_name = "nb_past_steps_{}_nb_lstm_states_{}_learning_rate_1e-3_seed_{}_".format(nb_past_steps, nb_lstm_states, seed)
                experiment_path = os.path.join(artifacts_path, experiment_name)
                with open(os.path.join(experiment_path, "all_rmse.txt"), "r") as f:
                    line = f.readlines()[0]
                    rmse = float(line)
                    results[i_seed, i_state, i_step] = rmse

    # plot rmse score mean and standard deviation with respect to number of LSTM
    # units and number of past steps for five different random seeds
    for i_step, nb_past_steps in enumerate(nb_past_stepss):
        xs = nb_lstm_statess
        ys_mean = np.mean(results, axis=0)[:, i_step]
        ys_std = np.std(results, axis=0)[:, i_step]
        plt.plot(xs, ys_mean)
        plt.fill_between(xs, ys_mean+ys_std, ys_mean-ys_std, alpha=0.5,
                label='history = {} min'.format(nb_past_steps*5))

    plt.ylabel('RMSE')
    plt.xlabel('Number of LSTM units')
    plt.legend(loc = 'upper right')
    plt.savefig("parameter_search.png", dpi=300)

if __name__ == "__main__":
    main()
