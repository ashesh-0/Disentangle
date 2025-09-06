import argparse
import os
import pickle

import wandb

import ml_collections
import yaml


def log_config(config, cur_workdir):
    # Saving config file.
    with open(os.path.join(cur_workdir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'Saved config to {cur_workdir}/config.pkl')

def add_back_config(run_id):
    """
    run = api.run("Disentanglement/yszfw23t")
    """
    api = wandb.Api()
    run = api.run(run_id)
    _ = run.file("config.yaml").download(replace=True)
    with open('./config.yaml', 'r') as file:
        data = yaml.safe_load(file)  # Parses the YAML content into data

    # data.keys() Out[12]: dict_keys(['wandb_version', '_wandb', 'data', 'datadir', 'exptname', 'git', 'hostname', 'loss', 'model', 'training', 'workdir'])
    output_dict = {}
    keys = ['data', 'datadir', 'exptname', 'git', 'hostname', 'loss', 'model', 'training', 'workdir']
    output_dict = {k: data[k]['value'] for k in keys if k in data}
    cfg_path = os.path.join(output_dict['workdir'], 'config.pkl')
    assert not os.path.exists(cfg_path), f"{cfg_path} already exists!"
    config = ml_collections.ConfigDict(output_dict)
    # freeze it 
    config = ml_collections.FrozenConfigDict(config)
    log_config(config, output_dict['workdir'])
    # clean up
    os.remove('./config.yaml')
    print(f"Config added back to {cfg_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, help='WandB run ID, e.g., "Disentanglement/yszfw23t"')
    args = parser.parse_args()
    add_back_config(args.run_id)