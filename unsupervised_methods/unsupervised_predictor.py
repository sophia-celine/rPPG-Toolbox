"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import numpy as np
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
import matplotlib.pyplot as plt

# Define all unsupervised methods for combined plotting
ALL_UNSUPERVISED_METHODS = ["POS", "CHROM", "ICA", "GREEN", "LGI", "PBV", "OMIT"]
# Determine subplot grid dimensions (e.g., 2 rows, 4 columns for 7 methods)
N_ROWS_SPECTROGRAM_PLOT = (len(ALL_UNSUPERVISED_METHODS) + 3) // 4 # Ceiling division for rows
N_COLS_SPECTROGRAM_PLOT = 4 # Fixed columns for better layout


def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    SNR_all = []
    MACC_all = []
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for it, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            data_input = data_input[..., :3]
            
            [b, a] = butter(1, [0.6 / config.UNSUPERVISED.DATA.FS * 2, 3.3 / config.UNSUPERVISED.DATA.FS * 2], btype='bandpass')
            bvp_signals_for_all_methods = {}
            bvp_for_current_method_arg = None # This will store the BVP for the method_name passed to the function

             # Calculate BVP for all methods for spectrogram plotting and also identify the one for metrics
            for current_method_name_iter in ALL_UNSUPERVISED_METHODS:
                BVP = None
                if current_method_name_iter == "POS":
                    BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
                elif current_method_name_iter == "CHROM":
                    BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
                elif current_method_name_iter == "ICA":
                    BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
                elif current_method_name_iter == "GREEN":
                    BVP = GREEN(data_input)
                elif current_method_name_iter == "LGI":
                    BVP = LGI(data_input)
                elif current_method_name_iter == "PBV":
                    BVP = PBV(data_input)
                elif current_method_name_iter == "OMIT":
                    BVP = OMIT(data_input)
                else:
                    raise ValueError(f"Unsupervised method name '{current_method_name_iter}' not recognized!")
                
                BVP_filtered = scipy.signal.filtfilt(b, a, np.double(BVP))
                
                bvp_signals_for_all_methods[current_method_name_iter] = BVP_filtered

                # If this BVP corresponds to the method_name argument, store it for later metrics
                if current_method_name_iter == method_name:
                    bvp_for_current_method_arg = BVP_filtered
            
            # Check if the BVP for the specified method_name was found
            if bvp_for_current_method_arg is None:
                raise ValueError(f"BVP signal for method '{method_name}' (from function argument) was not generated. "
                                 "Ensure it's in 'ALL_UNSUPERVISED_METHODS' list.")

            # --- Combined Spectrogram Plotting ---
            fig, axes = plt.subplots(N_ROWS_SPECTROGRAM_PLOT, N_COLS_SPECTROGRAM_PLOT, figsize=(N_COLS_SPECTROGRAM_PLOT * 4, N_ROWS_SPECTROGRAM_PLOT * 3), sharex=True, sharey=True)
            axes = axes.flatten()

            for i, (method_name_key, BVP_signal_to_plot) in enumerate(bvp_signals_for_all_methods.items()):
                ax = axes[i]
                ax.specgram(BVP_signal_to_plot, Fs=config.UNSUPERVISED.DATA.FS)
                ax.set_title(f'{method_name_key}', fontsize=10)
                if i % N_COLS_SPECTROGRAM_PLOT == 0: ax.set_ylabel('Frequência (Hz)', fontsize=8)
                if i >= (N_ROWS_SPECTROGRAM_PLOT - 1) * N_COLS_SPECTROGRAM_PLOT: ax.set_xlabel('Tempo (s)', fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=7)
            for j in range(len(ALL_UNSUPERVISED_METHODS), len(axes)): fig.delaxes(axes[j]) # Hide unused subplots
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.suptitle(f'Espectrogramas BVP para todos os métodos ({it}_{idx})', fontsize=14)
            plt.savefig(f'hr_results/spectrograms_all_methods_{it}_{idx}.png')
            plt.close(fig)
            # -------------------------------------

            # Plot and save the BVP signal for the specific method requested
            plt.figure()
            plt.plot(bvp_for_current_method_arg)
            plt.title(f'BVP {method_name}')
            plt.xlabel('Amostra')
            plt.ylabel('Amplitude')
            plt.savefig(f'BVPresults/BVP_{method_name}_{it}_{idx}.png') # Isso salvará o gráfico para o método específico
            plt.close()
            np.savetxt(f'BVPresults/BVP_{method_name}_{it}_{idx}.txt', bvp_for_current_method_arg, fmt='%.7e') # Isso salvará os dados para o método específico

            # --- Plotagem do Espectro de Frequência ---
            fig_freq, axes_freq = plt.subplots(N_ROWS_SPECTROGRAM_PLOT, N_COLS_SPECTROGRAM_PLOT, 
                                              figsize=(N_COLS_SPECTROGRAM_PLOT * 4, N_ROWS_SPECTROGRAM_PLOT * 3), 
                                              sharex=True)
            axes_freq = axes_freq.flatten()
            # Definir o intervalo de frequência em BPM para o plot (equivalente ao filtro passa-banda)
            min_hr_bpm = 0.6 * 60
            max_hr_bpm = 3.3 * 60

            for i, (method_name_key, BVP_signal_to_plot) in enumerate(bvp_signals_for_all_methods.items()):
                ax = axes_freq[i]
                fs = config.UNSUPERVISED.DATA.FS
                f, Pxx = scipy.signal.periodogram(BVP_signal_to_plot, fs=fs)
                f_bpm = f * 60
                mask = (f_bpm >= min_hr_bpm) & (f_bpm <= max_hr_bpm)
                
                ax.plot(f_bpm[mask], Pxx[mask])
                ax.set_title(method_name_key, fontsize=10)
                ax.grid(True)
                if i % N_COLS_SPECTROGRAM_PLOT == 0: ax.set_ylabel('PSD', fontsize=8)
                if i >= (N_ROWS_SPECTROGRAM_PLOT - 1) * N_COLS_SPECTROGRAM_PLOT: ax.set_xlabel('Frequência (BPM)', fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=7)

            for j in range(len(ALL_UNSUPERVISED_METHODS), len(axes_freq)): fig_freq.delaxes(axes_freq[j])
            plt.tight_layout()
            plt.savefig(f'hr_results/frequency_spectrum_all_methods_{it}_{idx}.png')
            plt.close(fig_freq)
            # -------------------------------------

            # --- Plotagem das Ondas BVP Filtradas (Sinal Temporal) ---
            fig_waves, axes_waves = plt.subplots(N_ROWS_SPECTROGRAM_PLOT, N_COLS_SPECTROGRAM_PLOT, 
                                                figsize=(N_COLS_SPECTROGRAM_PLOT * 4, N_ROWS_SPECTROGRAM_PLOT * 3), 
                                                sharex=True)
            axes_waves = axes_waves.flatten()
            for i, (method_name_key, BVP_signal_to_plot) in enumerate(bvp_signals_for_all_methods.items()):
                ax = axes_waves[i]
                ax.plot(BVP_signal_to_plot, color='tab:red')
                ax.set_title(method_name_key, fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                if i % N_COLS_SPECTROGRAM_PLOT == 0: ax.set_ylabel('Amplitude', fontsize=8)
                if i >= (N_ROWS_SPECTROGRAM_PLOT - 1) * N_COLS_SPECTROGRAM_PLOT: ax.set_xlabel('Amostra', fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=7)
            for j in range(len(ALL_UNSUPERVISED_METHODS), len(axes_waves)): fig_waves.delaxes(axes_waves[j])
            plt.tight_layout()
            plt.savefig(f'hr_results/bvp_waves_all_methods_{it}_{idx}.png')
            plt.close(fig_waves)
            # -------------------------------------

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(BVP), window_frame_size):
                # Correção: Usar o sinal BVP do método atual para o cálculo das métricas,
                # em vez do último BVP calculado no loop de todos os métodos.
                BVP_window = bvp_for_current_method_arg[i:i+window_frame_size]
                label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    gt_hr, pre_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                    gt_hr_peak_all.append(gt_hr)
                    predict_hr_peak_all.append(pre_hr)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    gt_fft_hr, pre_fft_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                    gt_hr_fft_all.append(gt_fft_hr)
                    predict_hr_fft_all.append(pre_fft_hr)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                else:
                    raise ValueError("Inference evaluation method name wrong!")
    print("Used Unsupervised Method: " + method_name)

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET
    else:
        raise ValueError('unsupervised_predictor.py evaluation only supports unsupervised_method!')

    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_peak_all - gt_hr_peak_all)
                RMSE_PEAK = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        np.savetxt(f"hr_results/HR_{method_name}.txt", predict_hr_fft_all, fmt='%.7e')
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        np.savetxt(f"hr_results/GT_HR_{method_name}.txt", gt_hr_fft_all, fmt='%.7e')
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
                RMSE_FFT = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]', 
                    y_label='Average of rPPG HR and GT PPG HR [bpm]', 
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
