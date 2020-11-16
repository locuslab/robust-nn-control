import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import ticker

from constants import *

import os

tableau10 = [(31, 119, 180), (255, 127, 14),
             (44, 160, 44), (214, 39, 40),
             (148, 103, 189), (140, 86, 75),
             (227, 119, 194), (127, 127, 127),
             (188, 189, 34), (23, 190, 207)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau10)):
    r, g, b = tableau10[i]
    tableau10[i] = (r / 255., g / 255., b / 255.)

def plot(save_subs, env_names, file_name, maxes, mins):
    n = len(save_subs)
    plt.rcParams.update({'font.size': 16})

    plt.yscale('log')
    fig, axes = plt.subplots(n, 2, figsize=(9, 1.4*n + 1))
    axes = axes.reshape((n, 2))

    for j, title in enumerate(['Non-robust Methods', 'Robust Methods']):
        axes[0, j].set_title(title, y=1.05, fontweight='bold', fontsize=17)
        axes[-1, j].set_xlabel('Training epochs')

    extra_artists = []
    for i, env_name in enumerate(env_names):
        if n > 1:
            ax = axes[i, 0]
            text = ax.text(-0.32, 0.5, env_name, horizontalalignment='center',
                           verticalalignment='center', transform=ax.transAxes,
                           rotation=90, fontweight='bold', fontsize=16)
            extra_artists.append(text)

        axes[i, 0].set_ylabel('Loss')

    colors = dict(zip(['LQR', 'MBP', 'PPO', 'Robust LQR', 'RARL', 'Robust MBP$^*$', 'Robust PPO$^*$', 'Robust MPC'], tableau10))
    renaming_dict = dict(zip(['mbp', 'robust_mbp', 'ppo', 'robust_ppo', 'rarl_ppo', 'lqr', 'robust_lqr', 'robust_mpc'], 
                        ['MBP', 'Robust MBP$^*$', 'PPO', 'Robust PPO$^*$', 'RARL', 'LQR', 'Robust LQR', 'Robust MPC']))

    for i, (save_sub, ax_1, ax_2) in enumerate(zip(save_subs, axes[:, 0], axes[:, 1])):
        save = os.path.join('results', save_sub)
        
        with open(os.path.join(save, 'results.txt'), 'r') as f:
            lines = f.readlines()

        test_performances = dict([x.strip().split(': ') for x in lines])
        for key in test_performances.keys():
            test_performances[key] = float(test_performances[key].replace('[', '').replace(']', ''))

        lqr_perf = test_performances['Custom LQR']
        lqr_adv_perf = test_performances['Custom LQR-adv']
        robust_lqr_perf = test_performances['Robust LQR']
        robust_lqr_adv_perf = test_performances['Robust LQR-adv']
        nn_perf = test_performances['MBP']
        nn_adv_perf = test_performances['MBP-adv']
        robust_nn_perf = test_performances['Robust MBP']
        robust_nn_adv_perf = test_performances['Robust MBP-adv']
        ppo_perf = test_performances['PPO']
        ppo_adv_perf = test_performances['PPO-adv']
        robust_ppo_perf = test_performances['Robust PPO']
        robust_ppo_adv_perf = test_performances['Robust PPO-adv']
        rarl_perf = test_performances['RARL PPO']
        rarl_adv_perf = test_performances['RARL PPO-adv']
        mpc_perf = test_performances.get('Robust MPC', None)
        mpc_adv_perf = test_performances.get('Robust MPC-adv', None)

        print('Results for %s' % env_names[i])
        
        if mpc_perf is not None:        
            print('& O & %.4g & %.4g & %.4g & %.4g & %.4g & %.4g & %.4g & %.4g \\\\' % (
                lqr_perf, nn_perf, ppo_perf, robust_lqr_perf, mpc_perf, rarl_perf, robust_nn_perf, robust_ppo_perf))
            print('& A & %.5g & %.5g & %.5g & %.5g & %.5g & %.5g & %.5g & %.5g \\\\' % (
                lqr_adv_perf, nn_adv_perf, ppo_adv_perf, robust_lqr_adv_perf, mpc_adv_perf, rarl_adv_perf, robust_nn_adv_perf, robust_ppo_adv_perf))
        else:
            print('& O & %.4g & %.4g & %.4g & %.4g & N/A & %.4g & %.4g & %.4g \\\\' % (
                lqr_perf, nn_perf, ppo_perf, robust_lqr_perf, rarl_perf, robust_nn_perf, robust_ppo_perf))
            print('& A & %.5g & %.5g & %.5g & %.5g & N/A & %.5g & %.5g & %.5g \\\\' % (
                lqr_adv_perf, nn_adv_perf, ppo_adv_perf, robust_lqr_adv_perf, rarl_adv_perf, robust_nn_adv_perf, robust_ppo_adv_perf))

        # Separate out performances in nominal and adversarial cases
        test_perfs = {}
        test_perfs_adv = {}
        nan_to_inf = lambda x: np.where(np.isnan(x), np.inf, x)
        for key in test_performances.keys():
            red_name = key.replace('-adv', '').replace(' ', '_').lower()
            renaming = renaming_dict[red_name]
            if 'adv' in key:
                test_perfs_adv[renaming] = nan_to_inf(test_performances[key])
            else:
                test_perfs[renaming] = nan_to_inf(test_performances[key])

        # Load losses in nominal and adversarial cases
        test_losses = {}
        test_losses_adv = {}
        for sub_dir in ['mbp', 'robust_mbp', 'ppo', 'robust_ppo', 'rarl_ppo']:
            _, _, losses, losses_adv = load_results(save, sub_dir)
            renaming = renaming_dict[sub_dir]
            test_losses[renaming] = nan_to_inf(losses)
            test_losses_adv[renaming] = nan_to_inf(losses_adv)
        # Truncate MBP testing curve to appropriate length for plots
        num_test_points = len(test_losses['Robust MBP$^*$'])
        test_losses['MBP'] = test_losses['MBP'][:num_test_points]
        test_losses_adv['MBP'] = test_losses_adv['MBP'][:num_test_points]

        perf_labels = ['LQR']
        loss_labels = ['MBP', 'PPO']
        loss_markers = ['s', 'D']
        test_freqs = [20, 16]
        subplot(num_test_points, 
                perf_labels, test_perfs, test_perfs_adv,
                loss_labels, test_losses, test_losses_adv,
                maxes[i], mins[i], ax_1, colors, test_frequencies=test_freqs, markers=loss_markers)
        
        perf_labels = ['Robust LQR', 'Robust MPC']
        loss_labels = ['RARL', 'Robust MBP$^*$', 'Robust PPO$^*$']
        loss_markers = ['s', '+', 'D']
        test_freqs = [16, 20, 16]
        subplot(num_test_points, 
                perf_labels, test_perfs, test_perfs_adv,
                loss_labels, test_losses, test_losses_adv,
                maxes[i], mins[i], ax_2, colors, test_frequencies=test_freqs, markers=loss_markers)

        ax_1.label_outer()
        ax_2.label_outer()

        handles_1, labels_1 = ax_1.get_legend_handles_labels()
        handles_2, labels_2 = ax_2.get_legend_handles_labels()

    fig.tight_layout(pad=0.0, w_pad=0.2, h_pad=0.2)

    # legend = axes[-1, 0].legend(handles, labels, loc='lower center', mode='expand', bbox_to_anchor=(-0.25, -0.6, 2.6, 0.5), frameon=True, ncol=4, )
    # legend = fig.legend([handles_1[0], handles_2[0], handles_1[1], handles_2[1], handles_1[2], handles_2[2]],
    #                     [labels_1[0], labels_2[0], labels_1[1], labels_2[1], labels_1[2], labels_2[2]],
    #                     loc='lower center', mode='expand', bbox_to_anchor=(0.05, 0.0, 1.2, 0.0), ncol=6,
    #                     handletextpad=0.15)
    height = fig.bbox_inches.y1
    legend = fig.legend(handles_1 + handles_2,
                        labels_1 + labels_2,
                        loc='lower center', bbox_to_anchor=(-0.015, 0.32 / height, 1.15, 0.0), ncol=5,
                        handletextpad=0.4, fontsize=15, frameon=False)
    text_label = fig.text(0.25, -0.80 / height, 'Setting:', fontweight='bold', fontsize=15)
    legend2 = fig.legend([Line2D([0], [0], color='black', lw=1), Line2D([0], [0], color='black', lw=1, linestyle='--')],
                         ['Original', 'Adversarial'],
                         loc='lower center', mode='expand', bbox_to_anchor=(0.40, -0.08 / height, 0.45, 0.0), ncol=2,
                         handletextpad=0.4, fontsize=15, frameon=False)
    extra_artists += [text_label, legend, legend2]

    # # text_offset = -0.25
    # text_offset = -0.08
    # text = fig.text(0.0, text_offset, ' ', fontsize=14, verticalalignment='top')
    # extra_artists.append(text)

    # fig.show()
    fig.savefig('%s.pdf' % file_name, bbox_inches='tight', bbox_extra_artists=extra_artists)

    return fig


def subplot(n, perf_labels, test_perfs, test_perfs_adv, 
            loss_labels, all_losses, all_adv_losses, ma, mi, ax,
            colors, test_frequencies, markers, plot_perfs=False):
    # blowup = 1e6
    # ma = max(np.max(nn_test_losses) if np.max(nn_test_losses) < blowup else 0,
    #          np.max(robust_nn_test_losses) if np.max(robust_nn_test_losses) < blowup else 0,
    #          np.max(ppo_test_losses) if np.max(ppo_test_losses) < blowup else 0,
    #          np.max(robust_ppo_test_losses) if np.max(robust_ppo_test_losses) < blowup else 0,
    #          lqr_perf if lqr_perf < blowup else 0,
    #          robust_lqr_perf if robust_lqr_perf < blowup else 0)
    # mi = min(np.min(nn_test_losses), np.min(robust_nn_test_losses), lqr_perf, robust_lqr_perf)
    # top = 1.05 * ma
    # bottom = mi - 0.05 * ma
    top = 1.15 * ma
    bottom = mi * 0.85
    # top = ma
    # bottom = mi
    linewidth = 1

    if plot_perfs:
        for label in perf_labels:
            if label in test_perfs:
                plot_line(test_perfs[label], ma, ax, n, linewidth=1, label=label, linestyle='-', color=colors[label])
                plot_line(test_perfs_adv[label], ma, ax, n, linewidth=1, linestyle='--', color=colors[label])

    for label, test_frequency, marker in zip(loss_labels, test_frequencies, markers):
        losses = all_losses[label]
        adv_losses = all_adv_losses[label]
        if losses.shape[0] > 80:
            losses = losses[::2]
            adv_losses = adv_losses[::2]
        plot_losses(losses, ma, ax,
                    test_frequency=test_frequency, linewidth=linewidth, label=label, linestyle='-', color=colors[label], marker=marker)
        plot_losses(adv_losses, ma, ax,
                    test_frequency=test_frequency, linewidth=linewidth, linestyle='--', color=colors[label], marker=marker)

    ax.set_yscale('log')
    ax.set_ylim(bottom=bottom, top=top)
    # ax.set_yticks([x for x in [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6] if mi <= x <= ma])
    ax.set_yticks([mi, ma])
    locmin = ticker.LogLocator(base=10.0, numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    # ax.ticklabel_format(scilimits=(-2, 4))


def plot_line(loss, ma, ax, n, linewidth=3, color=None, label=None, linestyle='-', marker=None):
    ax.axhline(y=loss, linestyle=linestyle, label=label, linewidth=linewidth, color=color, marker=marker)
    if loss > ma:
        ax.scatter(-1, ma, marker='x', s=150, color=color, linewidth=3)

def plot_losses(losses, ma, ax, n=None, test_frequency=1, linewidth=3, color=None, label=None, linestyle='-', marker=None):
    i = n if n is not None else losses.shape[0]
    if (losses[:i] > ma).any():
        i = np.argmax(losses[:n] > ma) + 1
        ax.scatter((i - 1) * test_frequency, ma, marker='x', s=150, color=color, linewidth=3)
    losses = np.minimum(losses, ma)
    ax.plot(range(0, i * test_frequency, test_frequency), losses[:i], label=label, linestyle=linestyle,
            linewidth=linewidth, color=color, marker=marker, markersize=2.25, markevery=2)

def load_results(save_dir, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
    
    plt.close()
    # train_losses = np.load(os.path.join(model_save_dir, 'train_losses.npy'))
    hold_losses = np.load(os.path.join(model_save_dir, 'hold_losses.npy'))
    test_losses = np.load(os.path.join(model_save_dir, 'test_losses.npy'))
    test_losses_adv = np.load(os.path.join(model_save_dir, 'test_losses_adv.npy'))
    
    return None, hold_losses, test_losses, test_losses_adv


if __name__ == '__main__':

    # Main plot
    save_subs = [
        'random_nldi-d0+alpha0.001+gamma20+testSz50+holdSz50+trainBatch20+baselr0.001+robustlr0.0001+T2+stepTypeRK4+testStepTypeRK4+seed10+dt0.01',
        'random_nldi-dnonzero+alpha0.001+gamma20+testSz50+holdSz50+trainBatch20+baselr0.001+robustlr0.0001+T2+stepTypeRK4+testStepTypeRK4+seed10+dt0.01',
        'cartpole+alpha0.001+gamma20+testSz50+holdSz50+trainBatch20+baselr0.001+robustlr0.0001+T10.0+stepTypeRK4+testStepTypeRK4+seed10+dt0.05',
        'quadrotor+alpha0.001+gamma20+testSz50+holdSz50+trainBatch20+baselr0.001+robustlr0.0001+T4.0+stepTypeRK4+testStepTypeRK4+seed10+dt0.02',
        'microgrid+alpha0.001+gamma20+testSz50+holdSz50+trainBatch20+baselr0.001+robustlr0.0001+T2+stepTypeRK4+testStepTypeRK4+seed10+dt0.01',
    ]
    env_names = ['NLDI\n(D = 0)', 'NLDI\n(D ≠ 0)', 'Cartpole', 'Quadrotor', 'Microgrid']
    maxes = [100000, 100000, 1000, 1000, 100]
    mins = [10, 10, 1, 1, 0.1]
    plot(save_subs, env_names, 'main_results', maxes, mins)
    
    # Appendix plot
    save_subs = [
        'random_pldi_env+alpha0.001+gamma20+testSz50+holdSz50+trainBatch20+baselr0.001+robustlr0.0001+T2+stepTypeRK4+testStepTypeRK4+seed10+dt0.01',
        'random_hinf_env+alpha0.001+gamma20+testSz50+holdSz50+trainBatch20+baselr0.001+robustlr0.0001+T2+stepTypeRK4+testStepTypeRK4+seed10+dt0.01',
    ]
    env_names = ['PLDI', 'H$_\mathbf{\infty}$']
    maxes = [1500, 1500]
    mins = [10, 10]
    plot(save_subs, env_names, 'appendix_results', maxes, mins)
