import os
import sys
from SplaTAM.utils.common_utils import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
from contextlib import redirect_stdout, redirect_stderr

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(version_base=None, config_path='../configs', config_name='train')
def train(config: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    # if os.path.exists(f'logs/{config.run_name}.log'):
    #     print(f'logs/{config.run_name}.log already exists, please remove it first.')
    #     return

    with open(f'logs/{config.run_name}.log', 'w') as f:
        with redirect_stdout(f), redirect_stderr(f):
            seed_everything(seed=config['seed'])
            hydra.utils.instantiate(config.backend, config, _recursive_=False)


if __name__ == "__main__":
    train()