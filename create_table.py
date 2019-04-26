import glob
import sys
import os
import numpy as np

def all_table(metric_name):
    artifacts_path = "artifacts/all_final_experiment"
    patient_ids = ['559', '570', '588', '563', '575', '591']
    d = {}
    nb_future_stepss = [6, 12]
    #seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    for patient_id in patient_ids:
        d[patient_id] = {
            "{}_lstm".format(6) : [],
            "{}_lstm".format(12) : [],
            "{}_t0".format(6) : [],
            "{}_t0".format(12) : []
        }
        for nb_future_steps in nb_future_stepss:
            for seed in seeds:
                experiment_name = "nb_future_steps_{}_seed_{}_".format(nb_future_steps, seed)
                experiment_path = os.path.join(artifacts_path, experiment_name)
                with open(os.path.join(experiment_path, "{}_{}.txt".format(patient_id, metric_name)), "r") as f:
                    line = f.readlines()[0]
                    if metric_name == 'seg':
                        line = line.split(',')[0][1:]
                    metric = float(line)
                    d[patient_id]["{}_lstm".format(nb_future_steps)].append(metric)
                with open(os.path.join(experiment_path, "{}_t0_{}.txt".format(patient_id, metric_name)), "r") as f:
                    line = f.readlines()[0]
                    if metric_name == 'seg':
                        line = line.split(',')[0][1:]
                    metric = float(line)
                    d[patient_id]["{}_t0".format(nb_future_steps)].append(metric)


    lstm_means_6 = []
    lstm_means_12 = []
    t0_means_6 = []
    t0_means_12 = []
    for patient_id in patient_ids:
        lstm_metric_6 = d[patient_id]["{}_lstm".format(6)]
        lstm_metric_12 = d[patient_id]["{}_lstm".format(12)]
        t0_metric_6 = d[patient_id]["{}_t0".format(6)]
        t0_metric_12 = d[patient_id]["{}_t0".format(12)]
        if not patient_id == 'all':
            lstm_means_6.append(np.mean(lstm_metric_6))
            lstm_means_12.append(np.mean(lstm_metric_12))
            t0_means_6.append(np.mean(t0_metric_6))
            t0_means_12.append(np.mean(t0_metric_12))
        print("{} & ${:.3f} \pm {:.3f}$ & ${:.3f}$ & ${:.3f} \pm {:.3f}$ & ${:.3f}$ \\\\".format(patient_id, np.mean(lstm_metric_6), np.std(lstm_metric_6), np.mean(t0_metric_6), np.mean(lstm_metric_12), np.std(lstm_metric_12), np.mean(t0_metric_12)))

    print("$\mu$ & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(np.mean(lstm_means_6), np.mean(t0_means_6), np.mean(lstm_means_12), np.mean(t0_means_12)))
    print("$\sigma$ & $\pm{:.3f}$ & $\pm{:.3f}$ & $\pm{:.3f}$ & $\pm{:.3f}$ \\\\".format(np.std(lstm_means_6), np.std(t0_means_6), np.std(lstm_means_12), np.std(t0_means_12)))

def separate_table():
    artifacts_paths = glob.glob("artifacts/*_final_experiment")
    patient_ids = [os.path.basename(artifacts_path).split("_")[0] for artifacts_path in artifacts_paths]
    d = {}
    nb_future_stepss = [6, 12]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for artifacts_path, patient_id in zip(artifacts_paths, patient_ids):
        if patient_id == 'all':
            continue
        d[patient_id] = {6 : [], 12 : []}
        for nb_future_steps in nb_future_stepss:
            for seed in seeds:
                experiment_name = "nb_future_steps_{}_seed_{}_".format(nb_future_steps, seed)
                experiment_path = os.path.join(artifacts_path, experiment_name)
                with open(os.path.join(experiment_path, "{}_rmse.txt".format(patient_id)), "r") as f:
                    line = f.readlines()[0]
                    rmse = float(line)
                    d[patient_id][nb_future_steps].append(rmse)


    means_6 = []
    means_12 = []
    for patient_id in patient_ids:
        if patient_id == 'all':
            continue
        rmses_6 = d[patient_id][6]
        rmses_12 = d[patient_id][12]
        means_6.append(np.mean(rmses_6))
        means_12.append(np.mean(rmses_12))
        print("{} | {} \pm {} | {} \pm {}".format(patient_id, np.mean(rmses_6), np.std(rmses_6), np.mean(rmses_12), np.std(rmses_12)))
    print(np.mean(means_6))
    print(np.mean(means_12))

def main():
    all_table("rmse")
    all_table("seg")
    #print("patient specific training ...")
    #separate_table()

if __name__ == "__main__":
    main()
