from omegaconf import DictConfig, OmegaConf

def test_func(cfg: DictConfig):
    print("Debug test successful!")

if __name__ == "__main__":
    config = OmegaConf.create({"test": "value"})
    test_func(config)