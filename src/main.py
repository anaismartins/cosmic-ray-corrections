import classification
import detection
import globals as g
import matplotlib.pyplot as plt
import numpy as np
import subtraction
import templates
import utils
import os
import time
from functools import partial
from scipy.optimize import curve_fit

def main(log_name=None):
    simname = "143-2a_simulations_3600_only_short"

    res = np.load(f"{g.DATA_PATH}{simname}.npy")
    sim_types = np.load(f"{g.DATA_PATH}{simname}_types.npy")
    sim_indices = np.load(f"{g.DATA_PATH}{simname}_indices.npy")
    seconds = np.linspace(0, len(res) / g.SAMPRATE, len(res))

    # for now, we are excluding the classification etc of the last min of data, so let's exclude those from the sims
    sim_types = sim_types[sim_indices < len(res) - g.SAMPRATE * 60]
    sim_indices = sim_indices[sim_indices < len(res) - g.SAMPRATE * 60]

    glitch_idx, score = detection.matched_filter(res)
    # match the detected glitches with the simulated ones
    matched_indices = np.intersect1d(glitch_idx, sim_indices)
    message = ("[Iteration 0] Detection accuracy: "
               f"{(len(matched_indices)) * 100 / len(sim_indices):.2f}%")
    utils.log_output(log_name, message)

    if g.PLOTS:
        if not os.path.exists(f"{g.FIGURES_PATH}detection/{simname}/"):
            os.makedirs(f"{g.FIGURES_PATH}detection/{simname}/")

        if len(g.GLITCH_TYPES) > 1:
            if not os.path.exists(f"{g.FIGURES_PATH}templates/{simname}/"):
                os.makedirs(f"{g.FIGURES_PATH}templates/{simname}/")

        plt.plot(seconds[:1000], score[:1000])
        visible = glitch_idx[glitch_idx < 1000]
        plt.scatter(seconds[visible], score[visible], color='red', label='Glitches')
        plt.legend()
        plt.title("Matched Filter Result - Iteration 0")
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.savefig(f"{g.FIGURES_PATH}detection/{simname}/matched_filter_result_0.png")
        plt.close()

    if len(g.GLITCH_TYPES) > 1:
        glitch_idx, glitch_labels, glitch_amps = classification.classify_glitches(glitch_idx, res,
                                                                                seconds,
                                                                                sim_name=simname,
                                                                                log_name=log_name)

        # count how many wrong classifications
        wrong_classifications = np.sum(glitch_labels[np.isin(glitch_idx, matched_indices)] != sim_types[np.isin(sim_indices, matched_indices)])
        message = ("[Iteration 0] Classification accuracy: "
                f"{100 * (1 - wrong_classifications / len(glitch_idx)):.2f}% Percentage of "
                "glitches classified as 'short': "
                f"{100 * np.sum(glitch_labels == 'short') / len(glitch_labels):.2f}%\nPercentage of "
                "glitches classified as 'long': "
                f"{100 * np.sum(glitch_labels == 'long') / len(glitch_labels):.2f}%\nPercentage of "
                "glitches classified as 'slow': "
                f"{100 * np.sum(glitch_labels == 'slow') / len(glitch_labels):.2f}%")

        utils.log_output(log_name, message)
    else:
        glitch_labels = np.repeat(g.GLITCH_TYPES[0], len(glitch_idx))
        glitch_amps = np.zeros(len(glitch_idx))
        model = partial(classification._template, glitch_type="short")
        for glitch_n, glitch_i in enumerate(glitch_idx):
            time_ax = np.arange(0, g.NSECS, 1 / g.SAMPRATE)
            popt, _ = curve_fit(model, time_ax,
                                res[glitch_i:glitch_i + int(g.NSECS * g.SAMPRATE) + 1],
                                p0=[1], bounds=(0, np.inf))
            glitch_amps[glitch_n] = popt[0]

    result, fit_amps = subtraction.subtract_glitches_from_data(glitch_idx, seconds,
                                                                  glitch_labels, glitch_amps, res)

    # calculate normalized chi2
    chi2_value = utils.chi2(result)
    message = f"[Iteration 0] Chi2: {int(chi2_value)}"
    utils.log_output(log_name, message)

    if g.PLOTS:
        if not os.path.exists(f"{g.FIGURES_PATH}subtraction/{simname}/"):
            os.makedirs(f"{g.FIGURES_PATH}subtraction/{simname}/")

        if not os.path.exists(f"{g.FIGURES_PATH}templates/{simname}/"):
            os.makedirs(f"{g.FIGURES_PATH}templates/{simname}/")

        plt.plot(seconds[:1000], res[:1000], label='Original Data')
        plt.plot(seconds[:1000], result[:1000], label='Data after Glitch Subtraction')
        plt.scatter(seconds[glitch_idx], res[glitch_idx], color='red', label='Detected Glitches')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Glitch Subtraction Result - Iteration 0')
        plt.legend()
        plt.xlim(0, seconds[1000])
        plt.savefig(f"{g.FIGURES_PATH}subtraction/{simname}/glitch_subtraction_result_0.png")
        plt.close()

    final_stack = templates.stacking(result, glitch_idx, glitch_labels, fit_amps, seconds,
                                     simname=simname)

    temp = templates._load_glitch_params()
    glitch_params = {}
    for glitch_type in g.GLITCH_TYPES:
        if final_stack[glitch_type] is None:
            message = f"No {glitch_type} glitches found, keeping previous template"
            utils.log_output(log_name, message)
            glitch_params[glitch_type] = temp[glitch_type]
            continue
        glitch_params[glitch_type] = templates.glitch_estimation(seconds[:int(g.NSECS * g.SAMPRATE)],
                                           final_stack[glitch_type], glitch_type)

        if g.PLOTS:
            if not os.path.exists(f"{g.FIGURES_PATH}templates/{simname}/"):
                os.makedirs(f"{g.FIGURES_PATH}templates/{simname}/")

            plt.plot(seconds[:int(g.NSECS*g.SAMPRATE)], final_stack[glitch_type][:int(g.NSECS*g.SAMPRATE)],
                    label="Stacked median")

            p = glitch_params[glitch_type]
            amps = [p[f'Amplitude{i}'] for i in range(1, 9)]
            taus = [p[f'Tau{i}']       for i in range(1, 9)]
            plt.plot(seconds[:int(g.NSECS*g.SAMPRATE)],
                    templates.glitch_model(seconds[:int(g.NSECS * g.SAMPRATE)], *amps,
                                           *taus)[:int(g.NSECS*g.SAMPRATE)], label="Fit Model")

            plt.title(f"Glitch Stacking and Fitting for {glitch_type} Glitches")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.savefig(f"{g.FIGURES_PATH}templates/{simname}/glitch_fitting_{glitch_type}_0.png")
            plt.close()

    for i in range(g.MAXITER):
        message = f"Iteration {i+1}/{g.MAXITER}"
        utils.log_output(log_name, message)
        prev_idx = glitch_idx.copy()
        glitch_idx, score = detection.matched_filter(res, glitch_params['short'])

        matched_indices = np.intersect1d(glitch_idx, sim_indices)
        message = (f"[Iteration {i+1}] Detection accuracy: "
                   f"{(len(matched_indices)) * 100 / len(sim_indices):.2f}%")
        utils.log_output(log_name, message)

        if g.PLOTS:
            plt.plot(seconds[:1000], score[:1000])
            visible = glitch_idx[glitch_idx < 1000]
            plt.scatter(seconds[visible], score[visible], color='red', label='Glitches')
            plt.legend()
            plt.title("Matched Filter Result - Iteration " + str(i+1))
            plt.ylabel("Amplitude")
            plt.xlabel("Time (s)")
            plt.savefig(f"{g.FIGURES_PATH}detection/{simname}/matched_filter_result_" + str(i+1) + ".png")
            plt.close()

        if len(g.GLITCH_TYPES) > 1:
            (glitch_idx, glitch_labels,
            glitch_amps) = classification.classify_glitches(glitch_idx, res, seconds,
                                                            glitch_params, iter=i+1,
                                                            prev_amps=fit_amps,
                                                            prev_labels=glitch_labels,
                                                            prev_idx=prev_idx, sim_name=simname,
                                                            log_name=log_name)

            wrong_classifications = np.sum(glitch_labels[np.isin(glitch_idx, matched_indices)] != sim_types[np.isin(sim_indices, matched_indices)])
            message = (f"[Iteration {i+1}] Classification accuracy: "
                    f"{100 * (1 - wrong_classifications / len(glitch_idx)):.2f}% Percentage of "
                "glitches classified as 'short': "
                f"{100 * np.sum(glitch_labels == 'short') / len(glitch_labels):.2f}%\nPercentage of "
                "glitches classified as 'long': "
                f"{100 * np.sum(glitch_labels == 'long') / len(glitch_labels):.2f}%\nPercentage of "
                "glitches classified as 'slow': "
                f"{100 * np.sum(glitch_labels == 'slow') / len(glitch_labels):.2f}%")
            utils.log_output(log_name, message)

        else:
            glitch_labels = np.repeat(g.GLITCH_TYPES[0], len(glitch_idx))
            glitch_amps = np.zeros(len(glitch_idx))
            model = partial(classification._template, glitch_type="short")
            for glitch_n, glitch_i in enumerate(glitch_idx):
                time_ax = np.arange(0, g.NSECS, 1 / g.SAMPRATE) 
                popt, _ = curve_fit(model, time_ax,
                                    res[glitch_i:glitch_i + int(g.NSECS * g.SAMPRATE) + 1],
                                    p0=[1], bounds=(0, np.inf))
                glitch_amps[glitch_n] = popt[0]

        # if the new glitch idx match the previous one, as well as the label, use the pewviously 
        # estimated amplitues. otherwise take from the newly estimated ones
        prev_amps = np.array([fit_amps[prev_idx == idx][0] if (idx in prev_idx and glitch_labels[prev_idx == idx][0] == label) else amp
                            for idx, label, amp in zip(glitch_idx, glitch_labels, glitch_amps)])

        result, fit_amps = subtraction.subtract_glitches_from_data(glitch_idx, seconds,
                                                                glitch_labels, prev_amps, res,
                                                                glitch_params=glitch_params)

        # calculate chi2
        chi2_value = utils.chi2(result)
        message = (f"[Iteration {i+1}] Chi2: {int(chi2_value)}")
        utils.log_output(log_name, message)

        if g.PLOTS:
            plt.plot(seconds[:1000], res[:1000], label='Original Data')
            plt.plot(seconds[:1000], result[:1000], label='Data after Glitch Subtraction')
            plt.scatter(seconds[glitch_idx], res[glitch_idx], color='red', label='Detected Glitches')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Glitch Subtraction Result - Iteration {i+1}')
            plt.legend()
            plt.xlim(0, seconds[1000])
            plt.savefig(f"{g.FIGURES_PATH}subtraction/{simname}/glitch_subtraction_result_{(i+1)}.png")
            plt.close()

        final_stack = templates.stacking(result, glitch_idx, glitch_labels, fit_amps, seconds,
                                         glitch_params=glitch_params, simname=simname)

        temp = glitch_params.copy()
        glitch_params = {}
        for glitch_type in g.GLITCH_TYPES:
            if final_stack[glitch_type] is None:
                message = f"No {glitch_type} glitches found, keeping previous template"
                utils.log_output(log_name, message)
                glitch_params[glitch_type] = temp[glitch_type]
                continue
            glitch_params[glitch_type] = templates.glitch_estimation(
                seconds[:int(g.NSECS * g.SAMPRATE)], final_stack[glitch_type], glitch_type)
    
            if g.PLOTS:
                plt.plot(seconds[:int(g.NSECS*g.SAMPRATE)],
                         final_stack[glitch_type][:int(g.NSECS*g.SAMPRATE)], label="Stacked median")
    
                p = glitch_params[glitch_type]
                amps = [p[f'Amplitude{i}'] for i in range(1, 9)]
                taus = [p[f'Tau{i}']       for i in range(1, 9)]
                plt.plot(seconds[:int(g.NSECS*g.SAMPRATE)],
                        templates.glitch_model(seconds[:int(g.NSECS * g.SAMPRATE)], *amps,
                                                *taus)[:int(g.NSECS*g.SAMPRATE)], label="Fit Model")
    
                plt.title(f"Glitch Stacking and Fitting for {glitch_type} Glitches")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.savefig(f"{g.FIGURES_PATH}templates/{simname}/glitch_fitting_{glitch_type}_{i+1}.png")
                plt.close()

if __name__ == "__main__":
    t = time.time()
    utils.log_params(f"{g.LOG_PATH}{t}.log")
    main(log_name=f"{g.LOG_PATH}{t}.log")