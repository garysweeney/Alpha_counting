import numpy as np
import matplotlib.pyplot as plt

def channel_to_energy(channel, counts):
    
    # Convert from channel to energy (MeV)
    energy = np.zeros(len(channel))
    for i in range(len(energy)):
        energy[i] = 0.000628 * channel[i] + 4.784
    
    # Only works for E > 4.95 MeV, so crop
    cut_energy = energy[np.where(energy > 4.95)]
    cut_counts = counts[np.where(energy > 4.95)]

    return cut_energy, cut_counts

def read_alphas(file, require_fix):

    if require_fix == True:
        data = np.genfromtxt(file, dtype=str)
        channel = np.array(data[:,0])
        for i in range(len(channel)):
            channel[i] = channel[i][:-1]
        channel = np.array(channel).astype(float)
        counts = np.array(data[:,1]).astype(float)
        energy, counts = channel_to_energy(channel, counts)
        return energy, counts
    
    else:
        data = np.genfromtxt(file)
        channel = np.array(data[:,0])
        counts = np.array(data[:,1])
        energy, counts = channel_to_energy(channel, counts)
        return energy, counts
    
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

thin_energy = np.hstack([thin_run3_energy, thin_run4_energy, thin_run5_energy])
thin_counts = np.hstack([thin_run3_counts, thin_run4_counts, thin_run5_counts])
"""
fig, (r1,r2,r3,r4,r5) = plt.subplots(5,1)
r2.plot(thin_run3_energy, thin_run3_counts)
r3.plot(thin_run4_energy, thin_run4_counts)
r4.plot(thin_run5_energy, thin_run5_counts)
plt.show()

plt.plot(thin_energy, thin_counts)
plt.show()
"""
thin_sample_livetime = 5 * 24 * 3600 # 4 days in seconds
thin_sample_area = 9.577 # cm2
thin_alpha_emissivity = np.zeros(4)

for i in range(len(thin_energy)):
    if thin_energy[i] > 4.95 and thin_energy[i] < 6.:
        thin_alpha_emissivity[0] += thin_counts[i]
    if thin_energy[i] >= 6. and thin_energy[i] < 7.:
        thin_alpha_emissivity[1] += thin_counts[i]
    if thin_energy[i] >= 7. and thin_energy[i] < 8.:
        thin_alpha_emissivity[2] += thin_counts[i]
    if thin_energy[i] >= 8. and thin_energy[i] < 9.:
        thin_alpha_emissivity[3] += thin_counts[i]

thin_alpha_emissivity /= thin_sample_livetime 
thin_alpha_emissivity /= thin_sample_area

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

med_energy = np.hstack([med_run1_energy, med_run2_energy, med_run3_energy, med_run4_energy, med_run5_energy])
med_counts = np.hstack([med_run1_counts, med_run2_counts, med_run3_counts, med_run4_counts, med_run5_counts])
"""
fig, (r1,r2,r3,r4,r5) = plt.subplots(5,1)
r1.plot(med_run1_energy, med_run1_counts)
r2.plot(med_run2_energy, med_run2_counts)
r3.plot(med_run3_energy, med_run3_counts)
r4.plot(med_run4_energy, med_run4_counts)
r5.plot(med_run5_energy, med_run5_counts)
plt.show()

plt.plot(med_energy, med_counts)
plt.show()
"""
med_sample_livetime = 5 * 24 * 3600 # 5 days in seconds
med_sample_area = 8.569 # cm2
med_alpha_emissivity = np.zeros(4)

for i in range(len(thin_energy)):
    if med_energy[i] > 4.95 and med_energy[i] < 6.:
        med_alpha_emissivity[0] += med_counts[i]
    if med_energy[i] >= 6. and med_energy[i] < 7.:
        med_alpha_emissivity[1] += med_counts[i]
    if med_energy[i] >= 7. and med_energy[i] < 8.:
        med_alpha_emissivity[2] += med_counts[i]
    if med_energy[i] >= 8. and med_energy[i] < 9.:
        med_alpha_emissivity[3] += med_counts[i]

med_alpha_emissivity /= med_sample_livetime
med_alpha_emissivity /= med_sample_area

# Thickest PTFE sample
thick_file1 = "./thick/P_thick_run1.dat"
thick_run1_energy, thick_run1_counts = read_alphas(thick_file1, True)
thick_file2 = "./thick/P_thick_run2.dat"
thick_run2_energy, thick_run2_counts = read_alphas(thick_file2, True)
thick_file3 = "./thick/P_thick_run3.dat"
thick_run3_energy, thick_run3_counts = read_alphas(thick_file3, True)
thick_file5 = "./thick/P_thick_run5.dat"
thick_run5_energy, thick_run5_counts = read_alphas(thick_file5, True)

thick_energy = np.hstack([thick_run1_energy, thick_run2_energy, thick_run3_energy, thick_run5_energy])
thick_counts = np.hstack([thick_run1_counts, thick_run2_counts, thick_run3_counts, thick_run5_counts])

thick_sample_livetime = 4 * 24 * 3600 # 5 days in seconds
thick_sample_area = 8.190 # cm2
thick_alpha_emissivity = np.zeros(4)

for i in range(len(thin_energy)):
    if thick_energy[i] > 4.95 and thick_energy[i] < 6.:
        thick_alpha_emissivity[0] += thick_counts[i]
    if thick_energy[i] >= 6. and thick_energy[i] < 7.:
        thick_alpha_emissivity[1] += thick_counts[i]
    if thick_energy[i] >= 7. and thick_energy[i] < 8.:
        thick_alpha_emissivity[2] += thick_counts[i]
    if thick_energy[i] >= 8. and thick_energy[i] < 9.:
        thick_alpha_emissivity[3] += thick_counts[i]

thick_alpha_emissivity_error_lower = np.array([17., 2., 2., 0.])
thick_alpha_emissivity_error_upper = np.array([17., 7., 7., 5.])

thick_alpha_emissivity /= thick_sample_livetime
thick_alpha_emissivity /= thick_sample_area

#thick_alpha_emissivity_error_lower /= thick_sample_livetime
#thick_alpha_emissivity_error_lower /= thick_sample_area
#thick_alpha_emissivity_error_upper /= thick_sample_livetime
#thick_alpha_emissivity_error_upper /= thick_sample_area

alpha_energy_bands = [5.5, 6.5, 7.5, 8.5]

plt.errorbar(alpha_energy_bands, [i * 1000000000 for i in thick_alpha_emissivity], xerr=0.5, capsize=3, fmt='o', label="Thick Sample")
plt.errorbar(alpha_energy_bands, [i * 1000000000 for i in med_alpha_emissivity], xerr=0.5, capsize=3, fmt='o', label="Medium Sample")
plt.errorbar(alpha_energy_bands, [i * 1000000000 for i in thin_alpha_emissivity], xerr=0.5, capsize=3, fmt='o', label = "Thin Sample")
plt.xlabel("Detected Alpha Energy (MeV)")
plt.ylabel(r"Alpha Emissivity (nBq/cm$^2$)")
plt.xticks([5.,6.,7.,8.,9.])
#plt.yscale("log")
plt.legend()
plt.show()