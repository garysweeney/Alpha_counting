import numpy as np
import matplotlib.pyplot as plt

def read_alphas(file, require_fix):

    if require_fix == True:
        data = np.genfromtxt(file, dtype=str)
        channel = np.array(data[:,0])
        for i in range(len(channel)):
            channel[i] = channel[i][:-1]
        channel = np.array(channel).astype(float)
        counts = np.array(data[:,1]).astype(float)
        return channel, counts
    
    else:
        data = np.genfromtxt(file)
        channel = np.array(data[:,0])
        counts = np.array(data[:,1])
        return channel, counts

# Thin PTFE sample
thin_file1 = "./thin/P5_run1.dat"
thin_run1_energy, thin_run1_counts = read_alphas(thin_file1, False)
thin_file2 = "./thin/P5_run2.dat"
thin_run2_energy, thin_run2_counts = read_alphas(thin_file2, False)
thin_file3 = "./thin/P5_run3.dat"
thin_run3_energy, thin_run3_counts = read_alphas(thin_file3, True)
thin_file4 = "./thin/P5_run4.dat"
thin_run4_energy, thin_run4_counts = read_alphas(thin_file4, True)
thin_file5 = "./thin/P5_run5.dat"
thin_run5_energy, thin_run5_counts = read_alphas(thin_file5, True)

# Medium thickness PTFE
med_file1 = "./medium/P_med_run1.dat"
med_run1_energy, med_run1_counts = read_alphas(med_file1, True)
med_file2 = "./medium/P_med_run2.dat"
med_run2_energy, med_run2_counts = read_alphas(med_file2, True)
med_file3 = "./medium/P_med_run3.dat"
med_run3_energy, med_run3_counts = read_alphas(med_file3, True)
med_file4 = "./medium/P_med_run4.dat"
med_run4_energy, med_run4_counts = read_alphas(med_file4, True)
med_file5 = "./medium/P_med_run5.dat"
med_run5_energy, med_run5_counts = read_alphas(med_file5, True)

# Thickest PTFE sample
thick_file1 = "./thick/P_thick_run1.dat"
thick_run1_energy, thick_run1_counts = read_alphas(thick_file1, True)
thick_file2 = "./thick/P_thick_run2.dat"
thick_run2_energy, thick_run2_counts = read_alphas(thick_file2, True)
thick_file3 = "./thick/P_thick_run3.dat"
thick_run3_energy, thick_run3_counts = read_alphas(thick_file3, True)
thick_file4 = "./thick/P_thick_run4.dat"
thick_run4_energy, thick_run4_counts = read_alphas(thick_file4, True)
thick_file5 = "./thick/P_thick_run5.dat"
thick_run5_energy, thick_run5_counts = read_alphas(thick_file5, True)

day = [1., 2., 3., 4., 5.]
thin_runs=[sum(thin_run1_counts),
           sum(thin_run2_counts),
           sum(thin_run3_counts),
           sum(thin_run4_counts),
           sum(thin_run5_counts)]

med_runs=[sum(med_run1_counts),
           sum(med_run2_counts),
           sum(med_run3_counts),
           sum(med_run4_counts),
           sum(med_run5_counts)]

thick_runs = [sum(thick_run1_counts),
              sum(thick_run2_counts),
              sum(thick_run3_counts),
              sum(thick_run4_counts),
              sum(thick_run5_counts)]


# Cut the first day
thin_counts = thin_runs[1:]
thin_avg_count = np.average(thin_counts)
print(thin_avg_count)
med_counts = med_runs[1:]
med_avg_count = np.mean(med_counts)
print(med_avg_count)
thick_counts = thick_runs[1:]
thick_avg_count = np.average(thick_counts)
print(thick_avg_count)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.supxlabel("Day")
fig.supylabel("Detected Alphas")
ax1.errorbar(day, thin_runs, yerr=[np.sqrt(i) for i in thin_runs], fmt="o", capsize=3, label = "Thin Sample", color="blue")
ax1.fill_between(day, thin_avg_count-np.sqrt(thin_avg_count), thin_avg_count+np.sqrt(thin_avg_count), alpha=0.3, color="blue")
ax1.axhline(thin_avg_count, color="blue")
ax1.set_xticks([1.,2.,3.,4.,5.])
ax1.legend()
ax2.errorbar(day, med_runs, yerr=[np.sqrt(i) for i in med_runs], fmt="o", capsize=3, label = "Med Sample", color="green")
ax2.fill_between(day, med_avg_count-np.sqrt(med_avg_count), med_avg_count+np.sqrt(med_avg_count), alpha=0.3, color="green")
ax2.axhline(med_avg_count, color="green")
ax2.set_xticks([1.,2.,3.,4.,5.])
ax2.legend()
ax3.errorbar(day, thick_runs, yerr=[np.sqrt(i) for i in thick_runs], fmt="o", capsize=3, label = "Thick Sample", color="red")
ax3.fill_between(day, thick_avg_count-np.sqrt(thick_avg_count), thick_avg_count+np.sqrt(thick_avg_count), alpha=0.3, color="red")
ax3.axhline(thick_avg_count, color="red")
ax3.set_xticks([1.,2.,3.,4.,5.])
ax3.legend(loc="upper right")
fig.tight_layout()
plt.show()
