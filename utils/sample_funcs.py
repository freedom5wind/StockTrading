import math
from typing import Dict, Tuple

import optuna


GAMMA = 1
TRAIN_TIME_STEP = 2000 * 500


# ------------------------------------------------
# for maskable algorithmns in single stock trading

def sample_mppo_param(trial: optuna.Trial) -> Tuple[Dict, int]:
    '''Sample hyperparameters and return them in a dictionary for model initiation.'''
    learning_rate = 3 * 10 ** (trial.suggest_int('learning_rate_3_exp', -5, -3))
    n_steps = 2 ** trial.suggest_int('n_steps_2exp', 0, 8)
    batch_size = 2 ** trial.suggest_int('batch_size_2exp', 5, 8)
    n_epochs = trial.suggest_int('n_epochs', 1, 5)
    net_arch_layers = [2 ** trial.suggest_int('net_arch_dim_2exp', 6, 10)] \
        * trial.suggest_int('net_arch_layers', 3, 5)

    gamma = GAMMA
    train_time_step = TRAIN_TIME_STEP

    return {
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'policy_kwargs': { 'net_arch': net_arch_layers }
            }, train_time_step

def sample_mdqn_param(trial: optuna.Trial) -> Tuple[Dict, int]:
    '''Sample hyperparameters and return them in a dictionary for model initiation.'''
    learning_rate = trial.suggest_float('learning_rate', 10e-6, 10e-3, log=True)
    buffer_size = 10 ** (trial.suggest_int('buffer_size_exp', 3, 6))
    learning_starts = 10 ** (trial.suggest_int('learning_starts_exp', 0, 3))
    batch_size = 2 ** trial.suggest_int('batch_size_2exp', 3, 8)
    tau = trial.suggest_float('tau', 10e-4, 1, log=True)
    train_freq = 2 ** trial.suggest_int('train_freq_2exp', 3, 9)
    gradient_steps = 2 ** trial.suggest_int('gradient_steps_2exp', 1, 8)
    target_update_interval = 10 ** trial.suggest_int('target_update_interval_exp', 1, 3)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.05, 0.2)
    net_arch_layers = [2 ** trial.suggest_int('net_arch_dim_2exp', 6, 10)] \
        * trial.suggest_int('net_arch_layers', 3, 5)

    gamma = GAMMA
    train_time_step = TRAIN_TIME_STEP

    return {
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'batch_size': batch_size,
        'tau': tau,
        'gamma': gamma,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'target_update_interval': target_update_interval,
        'exploration_fraction': exploration_fraction,
        'exploration_final_eps': exploration_final_eps,
        'policy_kwargs': { 'net_arch': net_arch_layers }
            }, train_time_step

def sample_mqrdqn_param(trial: optuna.Trial) -> Tuple[Dict, int]:
    '''Sample hyperparameters and return them in a dictionary for model initiation.'''
    learning_rate = 5 * 10 ** (trial.suggest_int('learning_rate_5_exp', -6, -2))
    buffer_size = 10 ** (trial.suggest_int('buffer_size_exp', 4, 7))
    learning_starts = 5 * 10 ** (trial.suggest_int('learning_start_5_exp', 0, 3))
    batch_size = 2 ** trial.suggest_int('batch_size_2exp', 3, 7)
    train_freq = 2 ** trial.suggest_int('train_freq_2exp', 3, 9)
    gradient_steps = 2 ** trial.suggest_int('gradient_step_2exp', 1, 8)
    target_update_interval = 10 ** trial.suggest_int('target_update_interval_exp', 1, 4)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.05, 0.2)
    n_quantiles = 2 ** trial.suggest_int('n_quantiles_2_exp', 3, 8)
    net_arch_layers = [2 ** trial.suggest_int('net_arch_dim_2exp', 6, 10)] \
        * trial.suggest_int('net_arch_layers', 3, 5)

    gamma = GAMMA
    train_time_step = TRAIN_TIME_STEP

    return {
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'batch_size': batch_size,
        'gamma': gamma,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'target_update_interval': target_update_interval,
        'exploration_fraction': exploration_fraction,
        'exploration_final_eps': exploration_final_eps,
        'policy_kwargs': {
        'n_quantiles': n_quantiles,
        'net_arch': net_arch_layers
        }
            }, train_time_step

def sample_miqn_param(trial: optuna.Trial) -> Tuple[Dict, int]:
    '''Sample hyperparameters and return them in a dictionary for model initiation.'''
    learning_rate = 5 * 10 ** (trial.suggest_int('learning_rate_5_exp', -6, -2))
    buffer_size = 10 ** (trial.suggest_int('buffer_size_exp', 4, 7))
    learning_starts = 5 * 10 ** (trial.suggest_int('learning_start_5_exp', 0, 3))
    batch_size = 2 ** trial.suggest_int('batch_size_2exp', 3, 7)
    train_freq = 2 ** trial.suggest_int('train_freq_2exp', 3, 9)
    gradient_steps = 2 ** trial.suggest_int('gradient_step_2exp', 1, 8)
    target_update_interval = 10 ** trial.suggest_int('target_update_interval_exp', 1, 4)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.05, 0.2)
    N1 = 2 ** trial.suggest_int('N1_2exp', 3, 7)
    N2 = 2 ** trial.suggest_int('N2_2exp', 3, 7)
    K = 2 ** trial.suggest_int('K_2exp', 5, 8)
    cosine_embedding_dim = 2 ** trial.suggest_int('cosine_embedding_dim_2exp', 5, 8)
    net_arch_layers = [2 ** trial.suggest_int('net_arch_dim_2exp', 6, 10)] \
        * trial.suggest_int('net_arch_layers', 3, 5)

    gamma = GAMMA
    train_time_step = TRAIN_TIME_STEP

    return {
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'batch_size': batch_size,
        'gamma': gamma,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'target_update_interval': target_update_interval,
        'exploration_fraction': exploration_fraction,
        'exploration_final_eps': exploration_final_eps,
        'N1': N1,
        'N2': N2,
        'policy_kwargs': {
        'K': K,
        'cosine_embedding_dim': cosine_embedding_dim,
        'net_arch': net_arch_layers
        }
            }, train_time_step


# ---------------------------------------------------
# for actor-critor algorithmns in multi-stock trading

def sample_a2c_param(trial: optuna.Trial) -> Tuple[Dict, int]:
    '''Sample hyperparameters and return them in a dictionary for model initiation.'''
    learning_rate = 3 * 10 ** (trial.suggest_int('learning_rate_3_exp', -5, -3))
    n_steps = 2 ** trial.suggest_int('n_steps_2exp', 0, 8)
    rms_prop_eps = 10 ** trial.suggest_int('rms_prop_eps_exp', -8, -5)
    net_arch_layers = [2 ** trial.suggest_int('net_arch_dim_2exp', 6, 10)] \
        * trial.suggest_int('net_arch_layers', 3, 5)

    gamma = GAMMA
    train_time_step = TRAIN_TIME_STEP

    return {
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'rms_prop_eps': rms_prop_eps,
        'gamma': gamma,
        'policy_kwargs': { 'net_arch': net_arch_layers }
            }, train_time_step

def sample_sac_param(trial: optuna.Trial) -> Tuple[Dict, int]:
    '''Sample hyperparameters and return them in a dictionary for model initiation.'''
    learning_rate = 3 * 10 ** (trial.suggest_int('learning_rate_3_exp', -6, -3))
    buffer_size = 10 ** (trial.suggest_int('buffer_size_exp', 4, 7))
    learning_starts = 5 * 10 ** (trial.suggest_int('learning_starts_5_exp', 0, 3))
    batch_size = 2 ** trial.suggest_int('batch_size_2exp', 3, 8)
    tau = trial.suggest_float('tau', 5e-3, 1, log=True)
    train_freq = 2 ** trial.suggest_int('train_freq_2exp', 3, 9)
    gradient_steps = 2 ** trial.suggest_int('gradient_steps_2exp', 1, 8)
    target_update_interval = 10 ** trial.suggest_int('target_update_interval_exp', 1, 4)

    net_arch_layers = [2 ** trial.suggest_int('net_arch_dim_2exp', 6, 10)] \
        * trial.suggest_int('net_arch_layers', 3, 5)

    gamma = GAMMA
    train_time_step = TRAIN_TIME_STEP

    return {
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'batch_size': batch_size,
        'tau': tau,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'target_update_interval': target_update_interval,
        'gamma': gamma,
        'policy_kwargs': {
             'net_arch': net_arch_layers,
             }
        }, train_time_step


def sample_tqc_param(trial: optuna.Trial) -> Tuple[Dict, int]:
    '''Sample hyperparameters and return them in a dictionary for model initiation.'''
    learning_rate = 3 * 10 ** (trial.suggest_int('learning_rate_3_exp', -6, -3))
    buffer_size = 10 ** (trial.suggest_int('buffer_size_exp', 4, 7))
    learning_starts = 5 * 10 ** (trial.suggest_int('learning_starts_5_exp', 0, 3))
    batch_size = 2 ** trial.suggest_int('batch_size_2exp', 3, 7)
    train_freq = 2 ** trial.suggest_int('train_freq_2exp', 3, 9)
    gradient_steps = 2 ** trial.suggest_int('gradient_steps_2exp', 1, 8)
    target_update_interval = 10 ** trial.suggest_int('target_update_interval_exp', 1, 4)
    n_quantiles = 2 ** trial.suggest_int('n_quantiles_2exp', 3, 9)
    # top_quantiles_to_drop_per_net  = math.floor(n_quantiles * trial.suggest_float('top_quantiles_drop_ratio', 0, 0.4))
    top_quantiles_to_drop_per_net = trial.suggest_int('top_quantiles_to_drop_per_net', 0, 5)
    n_critics = trial.suggest_int('n_critics', 1, 5)

    net_arch_layers = [2 ** trial.suggest_int('net_arch_dim_2exp', 6, 10)] \
        * trial.suggest_int('net_arch_layers', 3, 5)

    gamma = GAMMA
    train_time_step = TRAIN_TIME_STEP

    return {
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'batch_size': batch_size,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'target_update_interval': target_update_interval,
        'top_quantiles_to_drop_per_net': top_quantiles_to_drop_per_net,
        'gamma': gamma,
        'policy_kwargs': {
             'net_arch': net_arch_layers,
             'n_quantiles': n_quantiles,
             'n_critics': n_critics
             }
        }, train_time_step

def sample_tqci_param(trial: optuna.Trial) -> Tuple[Dict, int]:
    '''Sample hyperparameters and return them in a dictionary for model initiation.'''
    learning_rate = 3 * 10 ** (trial.suggest_int('learning_rate_3_exp', -6, -3))
    buffer_size = 10 ** (trial.suggest_int('buffer_size_exp', 4, 7))
    learning_starts = 5 * 10 ** (trial.suggest_int('learning_starts_5_exp', 0, 3))
    batch_size = 2 ** trial.suggest_int('batch_size_2exp', 3, 7)
    tau = trial.suggest_float('tau', 5e-3, 1, log=True)
    train_freq = 2 ** trial.suggest_int('train_freq_2exp', 3, 9)
    gradient_steps = 2 ** trial.suggest_int('gradient_steps_2exp', 1, 8)
    target_update_interval = 10 ** trial.suggest_int('target_update_interval_exp', 1, 4)
    n_samples_critics = 2 ** trial.suggest_int('n_samples_critics_2exp', 3, 8)
    # n_samples_target_critcs = 2 ** trial.suggest_int('n_samples_target_critcs_2exp', 3, 8)
    n_samples_target_critcs = n_samples_critics
    cos_embedding_dims = 2 ** trial.suggest_int('cos_embedding_dims_2exp', 3, 8)
    n_critics = trial.suggest_int('n_critics', 1, 5)

    net_arch_layers = [2 ** trial.suggest_int('net_arch_dim_2exp', 6, 10)] \
        * trial.suggest_int('net_arch_layers', 3, 5)

    gamma = GAMMA
    train_time_step = TRAIN_TIME_STEP

    return {
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'batch_size': batch_size,
        'tau': tau,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'target_update_interval': target_update_interval,
        'gamma': gamma,
        'policy_kwargs': {
             'net_arch': net_arch_layers,
             'n_samples_critics': n_samples_critics,
             'n_samples_target_critcs': n_samples_target_critcs,
             'cos_embedding_dims': cos_embedding_dims,
             'n_critics': n_critics
             }
        }, train_time_step
