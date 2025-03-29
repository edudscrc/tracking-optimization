import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from my_functions import m_imshow, m_plot, load_data

data, spatial_position_start, spatial_position_end, time_start, time_end = load_data('aquisicao_40km_50ns_21_10_2024_ds1_19000m_26000m_tds1_3s_15s_fs899Hz_bandpass_30Hz_80Hz_medfilt_47_4')

print(f'{data.shape = }')

fs = 900

# O apex aqui é na verdade um referência
# apex = 17410

spatial_position_arr = np.linspace(spatial_position_start, spatial_position_end, data.shape[0])
time_arr = np.linspace(time_start, time_end, data.shape[1])
x_ticks = np.linspace(time_arr.min(), time_arr.max(), 10)
y_ticks = np.linspace(spatial_position_arr.min(), spatial_position_arr.max(), 10)
extent = [time_arr.min(), time_arr.max(), spatial_position_arr.min(), spatial_position_arr.max()]

# vmax = np.percentile(abs(data), 99.9)
# vmin = -vmax
# plt.rcParams.update({'font.size': 5})
# plt.figure(layout="constrained", dpi=300)
# plt.imshow(abs(data), aspect='auto', origin='lower', vmax=vmax, vmin=vmin)
# plt.colorbar()
# plt.show()

# print('Plotting original matrix...')
# m_imshow(data=data, title='No Beamforming', xlabel='Time [s]', ylabel='Spatial Position [m]', xticks=x_ticks, yticks=y_ticks, filename='no_beamforming.png', extent=extent, show=False)

# signal_ref = data[apex, :]
# delays = []
# for i in range(data.shape[0]):
#     signal_aux = data[i, :]
#     corr = np.correlate(signal_ref - np.mean(signal_ref), signal_aux - np.mean(signal_aux), mode='full')
#     corr_delay = np.argmax(corr) - (len(signal_ref) - 1)
#     delays.append(corr_delay)
#     print(f'Iteration {i}')
# delays = np.asarray(delays)
# np.save('./delays_px.npy', delays)

# delays = np.load('./delays_px.npy')

delays = np.load('real_simulated_delays_seconds.npy')
delays *= -1
delays *= fs

# delays_seconds = delays / fs
# m_plot(x_arr=spatial_position_arr, data=delays_seconds, title='Real Delays', xlabel='Spatial Position [m]', ylabel='Time [s]', xticks=y_ticks, filename='delays_seconds.png', show=False)

print('Applying delays to spatio-temporal matrix...')
for i in range(data.shape[0]):
    data[i] = np.roll(data[i], delays[i])

print('Plotting beamformed matrix...')
m_imshow(data=data, title='Beamforming Applied', xlabel='Time [s]', ylabel='Spatial Position [m]', xticks=x_ticks, yticks=y_ticks, filename='beamforming_applied.png', extent=extent, show=False)

# signal = np.mean(data[apex-25:apex+25, :], axis=0)

# nperseg = int(fs * 0.4)
# noverlap = int(nperseg * 0.98)

# print('Computing spectrogram...')
# f, t, Sxx = spectrogram(signal, fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=4096, scaling='spectrum')

# freq_mask = (f >= 20) & (f <= 100)
# f = f[freq_mask]
# Sxx = Sxx[freq_mask, :]

# print('Plotting spectrogram...')
# plt.rcParams.update({'font.size': 5})
# plt.figure(layout="constrained", dpi=300)
# plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=-40, vmax=-20)
# plt.colorbar(label='Spectral Density [dB]')
# plt.title(f'Spectrogram - Beamforming Applied')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.yticks(np.linspace(f.min(), f.max(), 10))
# plt.xticks(np.linspace(t.min(), t.max(), 10))
# plt.savefig('./plots/spectrogram_beamforming.png', dpi=300)
# plt.close('all')

# audio_int16 = np.int16(signal * 32767)
# write("audio_beamformed.wav", int(fs), audio_int16)
