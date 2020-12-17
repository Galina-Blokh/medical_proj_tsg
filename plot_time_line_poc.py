import matplotlib.pyplot as plt
import numpy as np


def monotoneRanges(a, n):
    idx = [i for i, v in enumerate(a) if (not i or a[i - 1] != v) or (a[i - 1] != 1 and a[i] == 0)] + [len(a)]
    return [r for r in zip(idx, idx[1:]) if r[1] >= r[0] + n]


def plot_each_patient(patients_set, number_of_patients):

    for patient_index, indx in zip(range(number_of_patients), indices):
        res = monotoneRanges(y_patient, 2)

        fig, (ax) = plt.subplots(figsize=(12, 3))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        if res != []:
            t_end = np.array(res, dtype=int).flatten()[1]
            t_0 = np.array(res, dtype=int).flatten()[0]
            all_hours = np.linspace(0, length, t_end)
            vent_hours = np.linspace(t_0, t_end, t_end)
            plt.plot(all_hours, np.zeros(t_end), color='r', linestyle='-', linewidth=2, label='all hours')
            plt.plot(vent_hours, np.zeros(t_end), color='g', linewidth=10, label='vent hours')
            plt.xticks([t_0, t_end, length], ['t_0 ' + str(t_0), 't_end ' + str(t_end), 't_out ' + str(length)])
        else:
            t_end = 0
            t_0 = 0
            all_hours = np.linspace(0, length, length)
            vent_hours = np.linspace(0, 0, length)
            plt.plot(all_hours, np.zeros(length), color='r', linestyle='-', linewidth=2, label='all hours')
            plt.xticks([t_0, length], ['t_0 ' + str(t_0), 't_out ' + str(length)])

        plt.ylim(0, 5)
        plt.xlim(0, length)
        plt.yticks([])
        plt.legend(loc='best', fontsize=12)
        plt.xlabel("Hours")
        plt.title(f'TimeLine for patient N {indx} and predicted window')
        plt.show()
