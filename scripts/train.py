import sys
from SplaTAM.utils.common_utils import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(version_base=None, config_path='../configs', config_name='train')
def train(config: DictConfig) -> None:
    seed_everything(seed=config['seed'])
    hydra.utils.instantiate(config.backend, config, _recursive_=False)


if __name__ == "__main__":
    train()