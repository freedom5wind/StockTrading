# %% [markdown]
# # Import packages

# %%
import os
import random
from typing import Callable
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sb3_contrib.tqc import TQC
import seaborn as sns
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.sac import SAC
import torch as th

from environment.MultiStockTradingEnv import MultiStockTradingEnv
from quantile_critic.tqci import TQCI
from utils.sample_funcs import *
from utils.utils import *

# %% [markdown]
# # Config

# %%
warnings.simplefilter(action='ignore', category=FutureWarning)

TRAIN_START_DAY = '2008-01-01'
TRAIN_END_DAY = '2016-12-31'
TEST_START_DAY = '2017-01-01'
TEST_END_DAY = '2019-12-31'
TRADE_START_DAY = '2020-01-01'
TRADE_END_DAY = '2022-12-31'

N_WORKERS = 96

total_timesteps = 3000 * 100 * N_WORKERS

# Setup directories
DATA_SAVE_DIR = 'datasets'
MODEL_DIR = 'models'
MODEL_TRAINED_DIR = os.path.join(MODEL_DIR, 'trained')
TENSORBOARD_LOG_DIR = 'tensorboard_log'
PREPROCESSED_DATA_DIR = os.path.join(DATA_SAVE_DIR, 'preprocessed')
PREPROCESSED_SSE50_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'SSE50')
PREPROCESSED_HSI_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'HSI')
PREPROCESSED_DJIA_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'DJIA')
PREPROCESSED_DAX30_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'DAX30')

check_and_make_directories([
     DATA_SAVE_DIR, MODEL_DIR, MODEL_TRAINED_DIR, TENSORBOARD_LOG_DIR, \
     PREPROCESSED_DATA_DIR, \
     PREPROCESSED_SSE50_DIR, PREPROCESSED_HSI_DIR, PREPROCESSED_DJIA_DIR, PREPROCESSED_DAX30_DIR
     ])

# %% [markdown]
# # Setup environment

# %%
dataset_dir = os.path.join(PREPROCESSED_DATA_DIR, 'SSE50')
# load data

df_dict = {}
_, _, files = next(os.walk(dataset_dir))
for file in files:
    processed_file_path = os.path.join(dataset_dir, file)   
    df = pd.read_csv(processed_file_path, index_col=False)
    tic = file.replace('.csv', '')
    df_dict[tic] = df.copy()

# %%
df_dict_train, df_dict_test, df_dict_trade = split_data(df_dict, TEST_START_DAY, TRADE_START_DAY)

# %%
def get_envs(n_tickers: int = 10, seed=None) -> Tuple[MultiStockTradingEnv, MultiStockTradingEnv, MultiStockTradingEnv]:
    assert n_tickers <= len(df_dict_train)

    def env_factory():
        return Monitor(MultiStockTradingEnv(_dfs))

    env_list = list()
    random.seed(seed)
    tic_list = random.sample(df_dict_train.keys(), n_tickers)
    print('Sampled tickers: ', tic_list)
    for _df_dict in [df_dict_train, df_dict_test, df_dict_trade]:
        if len(_df_dict[tic_list[0]]) == 0:
            env_list.append(None)
        else:
            _dfs = list()
            for tic in tic_list:
                _df = _df_dict[tic].copy()
                _df['tic'] = tic
                _dfs.append(_df)
            _dfs = pd.concat(_dfs)
            # drop dates that missing data
            _dfs = _dfs.pivot_table(index=['date'], columns=['tic']).dropna().stack().reset_index()
            _dfs.sort_values(['date', 'tic'], inplace=True)
            _dfs.set_index(['date', 'tic'], inplace=True)
            env_list.append(SubprocVecEnv([env_factory] * N_WORKERS))
    
    return tuple(env_list)

# %% [markdown]
# # Exp

# %%
SEED = [114, 123, 26, 103, 233] # 89, 41, 195, 202, 197
for i in range(len(SEED)):
    SEED[i] += i
N_REPEAT = len(SEED)

# %%
SSE_path = './datasets/sse50.csv'
HSI_path = './datasets/hsi.csv'
DJI_path = './datasets/dji.csv'
DAX30_path = './datasets/dax30.csv'

# %% [markdown]
# ## SAC VS TQCR VS TQCI

# %% [markdown]
# ### Train models

# %%
# setup environment
df_dict_train, df_dict_test, df_dict_trade = split_data(df_dict, TRADE_START_DAY, TRADE_START_DAY)

model_info = []

# SAC
params = {
    'learning_rate': 3 * 10 ** -5,
    'buffer_size': 10 ** 5,
    'learning_starts': 50,
    'batch_size': 2 ** 5,
    'train_freq': 2 ** 4,
    'gradient_steps': 2 ** 3,
    'target_update_interval': 10 ** 2,
    'gamma': 1,
    'policy_kwargs': {
        'net_arch': [2 ** 7] * 5
    }
}
model_info.append((SAC, params.copy()))

# TQC
params['top_quantiles_to_drop_per_net'] = 3
params['policy_kwargs'] = {
    'net_arch': [2 ** 7] * 5,
    'n_quantiles': 2 ** 8,
    'n_critics': 3
}
model_info.append((TQC, params.copy()))

# TQCI
params.pop('top_quantiles_to_drop_per_net')
params['policy_kwargs'] = {
    'net_arch': [2 ** 7] * 5,
    'n_samples_critics': 64,
    'n_samples_target_critcs': 64,
    'cos_embedding_dims': 64,
    'n_critics': 3
}

model_info.append((TQCI, params.copy()))

# %%
for i in range(N_REPEAT):
    for model_class, params in model_info:
        model_name = model_class.__name__
        tb_log_path = os.path.join(TENSORBOARD_LOG_DIR, f'multi_{model_name}_SSE50')
        check_and_make_directories(tb_log_path)

        params['seed'] = SEED[i]

        env_train, _, _ = get_envs(seed=SEED[i])

        model = model_class(
            'MlpPolicy',
            env_train,
            **params,
            verbose=1,
            tensorboard_log=tb_log_path,
        )
        model.learn(total_timesteps=total_timesteps, tb_log_name=f'multi_train_{model_name}_{i}')
        
        model_path = os.path.join(MODEL_TRAINED_DIR, f'multi_{model_name}_{i}.pth')
        model.save(model_path)

# %% [markdown]
# ### Backtest

# %%
df_dict_train, df_dict_test, df_dict_trade = split_data(df_dict, TRADE_START_DAY, TRADE_START_DAY)

# %%
models = [SAC, TQC, TQCI]

result = {}
for model_class in models:
    model_name = model_class.__name__
    result[model_name] = []
    for i in range(N_REPEAT):
        _, _, env_trade = get_envs(seed=SEED[i])
        tics = env_trade.df.reset_index().tic.unique()
        dates = env_trade.df.reset_index().date.unique()

        model_path = os.path.join(MODEL_TRAINED_DIR, f'multi_{model_name}_{i}.pth')
        model = model_class.load(model_path)

        list_asset, actions, rewards = simulate_trading(env_trade, model)
        result[model_name].append((tics, dates, list_asset))

with open('multi_assets_SSE50.pkl', 'wb') as fout:
    pickle.dump(result, fout)

# %% [markdown]
# ### Plot

# %%
with open('multi_assets_SSE50.pkl', 'rb') as fin:
    result = pickle.load(fin)

df_sse50 = pd.read_csv(SSE_path, index_col=False)
df_sse50.date = pd.to_datetime(df_sse50.date, format='%Y-%m-%d')
df_sse50['change'] = df_sse50['close'].pct_change()
df_sse50['change'][0] = 0

df_assets = pd.DataFrame()
for model_name, tuples in result.items():
    print(model_name)
   
    for tup in tuples:
        tics, dates, asset = tup
        df_t = pd.DataFrame(data={'date': dates, 'asset': asset})
        df_t['model'] = model_name
        df_t['tics'] = ' '.join(tics)
        df_assets = pd.concat([df_assets, df_t])

    tics, dates, asset = tuples[0]
    print(tics)
    plot_asset = pd.DataFrame({'date': dates, 'asset': asset})
    plot_asset['change_'] = plot_asset['asset'].pct_change()
    plot_asset['change_'][0] = 0
    plot_asset = plot_asset.merge(df_sse50[['date', 'change']], how='left', on='date')
    plot_asset.set_index('date', inplace=True)
    plot_asset.to_csv('SAC_VS_TQC_VS_TQCI.csv')
    # backtest_plot(plot_asset['change_'], plot_asset['change'])

# %%
df_sse50['asset'] = df_sse50['close'] / df_sse50['close'].iloc[0]
df_sse50['model'] = 'SSE50'

# %%
fig, ax = plt.subplots(figsize=(16, 7))
sns.lineplot(pd.concat([df_assets, df_sse50[['date','asset', 'model']]], ignore_index=True), x='date', y='asset', hue='model', errorbar='sd', ax=ax)
ax.get_figure().savefig('SAC_VS_TQC_VS_TQCI.png')

# %% [markdown]
# ## SSE50VS HSI VS DJI VS DAX30

# %% [markdown]
# ### Train

# %%
# TQCI
params = {
    'learning_rate': 3 * 10 ** -5,
    'buffer_size': 10 ** 5,
    'learning_starts': 50,
    'batch_size': 2 ** 5,
    'train_freq': 2 ** 4,
    'gradient_steps': 2 ** 3,
    'target_update_interval': 10 ** 2,
    'gamma': 1,
    'policy_kwargs': {
        'net_arch': [2 ** 7] * 5,
        'n_samples_critics': 64,
        'n_samples_target_critcs': 64,
        'cos_embedding_dims': 64,
        'n_critics': 3
    }
}

for i in range(N_REPEAT):
    for index_name in ['DJIA']:

        # load data
        dataset_dir = os.path.join(PREPROCESSED_DATA_DIR, index_name)
        df_dict = {}
        _, _, files = next(os.walk(dataset_dir))
        for file in files:
            processed_file_path = os.path.join(dataset_dir, file)   
            df = pd.read_csv(processed_file_path, index_col=False)
            tic = file.replace('.csv', '')
            df_dict[tic] = df.copy()
        
        df_dict_train, df_dict_test, df_dict_trade = split_data(df_dict, TRADE_START_DAY, TRADE_START_DAY)

        env_train, _, _ = get_envs(seed=SEED[i])

        tb_log_path = os.path.join(TENSORBOARD_LOG_DIR, f'multi_TQCI_{index_name}')
        check_and_make_directories(tb_log_path)

        model = TQCI(
            'MlpPolicy',
            env_train,
            **params,
            verbose=1,
            tensorboard_log=tb_log_path,
            seed=SEED[i]
        )
        model.learn(total_timesteps=total_timesteps, tb_log_name=f'multi_train_tqci_on_{index_name}')
        
        model_path = os.path.join(MODEL_TRAINED_DIR, f'multi_tqci_on_{index_name}_{i}.pth')
        model.save(model_path)

# %% [markdown]
# ### Backtest

# %%
result = {}
for index_name in ['HSI', 'DJIA', 'DAX30']:
    result[index_name] = []
    
for i in range(N_REPEAT):
    for index_name in ['HSI', 'DJIA', 'DAX30']:

        # load data
        dataset_dir = os.path.join(PREPROCESSED_DATA_DIR, index_name)
        df_dict = {}
        _, _, files = next(os.walk(dataset_dir))
        for file in files:
            processed_file_path = os.path.join(dataset_dir, file)   
            df = pd.read_csv(processed_file_path, index_col=False)
            tic = file.replace('.csv', '')
            df_dict[tic] = df.copy()
        
        df_dict_train, df_dict_test, df_dict_trade = split_data(df_dict, TRADE_START_DAY, TRADE_START_DAY)

        _, _, env_trade = get_envs(seed=SEED[i])

        tics = env_trade.df.reset_index().tic.unique()
        dates = env_trade.df.reset_index().date.unique()

        if index_name == 'SSE50':
            model_path = os.path.join(MODEL_TRAINED_DIR, f'multi_TQCI_{i}.pth')
        else:
            model_path = os.path.join(MODEL_TRAINED_DIR, f'multi_tqci_on_{index_name}_{i}.pth')
        model = TQCI.load(model_path)

        list_asset, actions, rewards = simulate_trading(env_trade, model)
        result[index_name].append((tics, dates, list_asset))

with open('multi_assets_indexes.pkl', 'wb') as fout:
    pickle.dump(result, fout)

# %% [markdown]
# ### Plot

# %%
with open('multi_assets_indexes.pkl', 'rb') as fin:
    result = pickle.load(fin)

df_sse50 = pd.read_csv(SSE_path)
df_hsi = pd.read_csv(HSI_path)
df_dji = pd.read_csv(DJI_path)
df_dax30 = pd.read_csv(DAX30_path)
baselines = {
    'SSE50': df_sse50,
    'HSI': df_hsi,
    'DJIA': df_dji,
    'DAX30': df_dax30,
}
for df_baseline in baselines.values(): 
    df_baseline.date = pd.to_datetime(df_baseline.date, format='%Y-%m-%d')
    df_baseline['change'] = df_baseline['close'].pct_change()
    df_baseline['change'][0] = 0

for index_name, tuples in result.items():
    print(index_name)

    df_assets = pd.DataFrame()
    for tup in tuples:
        tics, dates, asset = tup
        df_t = pd.DataFrame(data={'date': dates, 'asset': asset})
        df_t['index_name'] = index_name
        df_t['tics'] = ' '.join(tics)
        df_assets = pd.concat([df_assets, df_t])

    tics, dates, asset = tup
    print(tics)
    plot_asset = pd.DataFrame({'date': dates, 'asset': asset})
    plot_asset['change_'] = plot_asset['asset'].pct_change()
    plot_asset['change_'][0] = 0
    plot_asset = plot_asset.merge(baselines[index_name][['date', 'change']], how='left', on='date')
    plot_asset.set_index('date', inplace=True)
    plot_asset.to_csv('SSE50_HSI_DJIA_DAX30.csv')
    # backtest_plot(plot_asset['change_'], plot_asset['change'])

# %%
fig, ax = plt.subplots(figsize=(16, 7))
sns.lineplot(df_assets[df_assets['index_name'] == 'DAX30'], x='date', y='asset', hue='tics', errorbar='sd', ax=ax)
ax.get_figure().savefig('SSE50_HSI_DJIA_DAX30.png')

# %% [markdown]
# ## Risk distortion measure

# %%
def CPW_factory(eta: float) -> Callable[[th.Tensor], th.Tensor]:
    def CPW(taus: th.Tensor) -> th.Tensor:
        taus = (taus ** eta) / ((taus ** eta + (1 - taus) ** eta) ** (1 / eta))
        return taus
    return CPW

def Wang_factory(eta: float) -> Callable[[th.Tensor], th.Tensor]:
    def Wang(taus: th.Tensor) -> th.Tensor:
        n = th.distributions.normal.Normal(th.zeros_like(taus), th.ones_like(taus))
        finfo = th.finfo(taus.dtype)
        # clamp to prevent +-inf
        taus = n.cdf(th.clamp(n.icdf(taus) + eta, min=finfo.min, max=finfo.max))
        return taus
    return Wang

def CVaR_factory(eta: float) -> Callable[[th.Tensor], th.Tensor]:
    def CVaR(taus: th.Tensor) -> th.Tensor:
        taus = taus * eta
        return taus
    return CVaR

def Norm_factory(eta: int) -> Callable[[th.Tensor], th.Tensor]:
    def Norm(taus: th.Tensor) -> th.Tensor:
        taus = taus.unsqueeze(-1).repeat_interleave(repeats=eta, dim=-1).uniform_(0, 1).mean(axis=-1)
        return taus
    return Norm

def Pow_factory(eta: float) -> Callable[[th.Tensor], th.Tensor]:
    def Pow(taus: th.Tensor) -> th.Tensor:
        if eta >= 0:
            taus = taus ** (1 / (1 + eta))
        else:
            taus = 1 - (1 - taus) ** (1 / (1 - eta))
        return taus
    return Pow

# %%
rdms = {}
rdms['CPW_0.71'] = CPW_factory(0.71)
rdms['Wang_-0.75'] = Wang_factory(-0.75)
rdms['CVaR_0.25'] = CVaR_factory(0.25)
rdms['CVaR_0.4'] = CVaR_factory(0.4)
rdms['Norm_3']= Norm_factory(3)
rdms['Pow_-2'] = Pow_factory(-2)

# %% [markdown]
# ### Train

# %%
# load data
dataset_dir = os.path.join(PREPROCESSED_DATA_DIR, 'SSE50')
df_dict = {}
_, _, files = next(os.walk(dataset_dir))
for file in files:
    processed_file_path = os.path.join(dataset_dir, file)   
    df = pd.read_csv(processed_file_path, index_col=False)
    tic = file.replace('.csv', '')
    df_dict[tic] = df.copy()

df_dict_train, df_dict_test, df_dict_trade = split_data(df_dict, TRADE_START_DAY, TRADE_START_DAY)

# %%
# TQCI
params = {
    'learning_rate': 3 * 10 ** -5,
    'buffer_size': 10 ** 5,
    'learning_starts': 50,
    'batch_size': 2 ** 5,
    'train_freq': 2 ** 4,
    'gradient_steps': 2 ** 3,
    'target_update_interval': 10 ** 2,
    'gamma': 1,
    'policy_kwargs': {
        'net_arch': [2 ** 7] * 5,
        'n_samples_critics': 64,
        'n_samples_target_critcs': 64,
        'cos_embedding_dims': 64,
        'n_critics': 3
    }
}

for i in range(1, N_REPEAT):
    for name, rdm in rdms.items():

        # load data
        dataset_dir = os.path.join(PREPROCESSED_DATA_DIR, 'SSE50')
        df_dict = {}
        _, _, files = next(os.walk(dataset_dir))
        for file in files:
            processed_file_path = os.path.join(dataset_dir, file)   
            df = pd.read_csv(processed_file_path, index_col=False)
            tic = file.replace('.csv', '')
            df_dict[tic] = df.copy()
        
        df_dict_train, df_dict_test, df_dict_trade = split_data(df_dict, TRADE_START_DAY, TRADE_START_DAY)

        env_train, _, _ = get_envs(seed=SEED[i])
        

        tb_log_path = os.path.join(TENSORBOARD_LOG_DIR, f'multi_TQCI_SSE50_{name}')
        check_and_make_directories(tb_log_path)

        params['policy_kwargs']['risk_distortion_measures'] = rdm
        model = TQCI(
            'MlpPolicy',
            env_train,
            **params,
            verbose=1,
            tensorboard_log=tb_log_path,
            seed=SEED[i]
        )
        model.learn(total_timesteps=total_timesteps, tb_log_name=f'multi_train_tqci_with_{name}')
        
        model_path = os.path.join(MODEL_TRAINED_DIR, f'multi_tqci_with_{name}_{i}.pth')
        model.save(model_path)

# %% [markdown]
# ### Backtest

# %%
result = {}
for name in rdms.keys():
    result[name] = []

for i in range(N_REPEAT):
    for name, rdm in rdms.items():

        _, _, env_trade = get_envs(seed=SEED[i])
        tics = env_trade.df.reset_index().tic.unique()
        dates = env_trade.df.reset_index().date.unique()

        model_path = os.path.join(MODEL_TRAINED_DIR, f'multi_tqci_with_{name}_{i}.pth')
        model = TQCI.load(model_path)

        list_asset, actions, rewards = simulate_trading(env_trade, model)
        result[name].append((tics, dates, list_asset))

with open('multi_assets_risk.pkl', 'wb') as fout:
    pickle.dump(result, fout)

# %% [markdown]
# ### Plot

# %%
with open('multi_assets_risk.pkl', 'rb') as fin:
    result = pickle.load(fin)

df_sse50 = pd.read_csv(SSE_path, index_col=False)
df_sse50.date = pd.to_datetime(df_sse50.date, format='%Y-%m-%d')
df_sse50['change'] = df_sse50['close'].pct_change()
df_sse50['change'][0] = 0

df_assets = pd.DataFrame()
for rdm_name, tuples in result.items():
    print(rdm_name)
   
    for tup in tuples:
        tics, dates, asset = tup
        df_t = pd.DataFrame(data={'date': dates, 'asset': asset})
        df_t['rdm_name'] = rdm_name
        df_t['tics'] = ' '.join(tics)
        df_assets = pd.concat([df_assets, df_t])

    tics, dates, asset = tuples[0]
    print(tics)
    plot_asset = pd.DataFrame({'date': dates, 'asset': asset})
    plot_asset['change_'] = plot_asset['asset'].pct_change()
    plot_asset['change_'][0] = 0
    plot_asset = plot_asset.merge(df_sse50[['date', 'change']], how='left', on='date')
    plot_asset.set_index('date', inplace=True)
    plot_asset('risk_distortion_measure.csv')
    # backtest_plot(plot_asset['change_'], plot_asset['change'])

# %%
df_sse50['asset'] = df_sse50['close'] / df_sse50['close'].iloc[0]
df_sse50['rdm_name'] = 'SSE50'
fig, ax = plt.subplots(figsize=(16, 7))
sns.lineplot(pd.concat([df_assets, df_sse50[['date','asset', 'rdm_name']]], ignore_index=True), x='date', y='asset', hue='rdm_name', errorbar='sd', ax=ax)
ax.get_figure().savefig('risk_distortion_measure.png')
