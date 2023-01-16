import os

from main import get_parser
from utils.super_experiment import SuperExperiment

if __name__ == '__main__':
    cfg = get_parser(desc='Super Experiment Grid Search', default_file='yaml-config/dummy.yaml', add_exp_name=False)
    cfg.models.network = [p[:-5] for p in os.listdir(os.path.join('yaml-config', 'models'))]
    exp = SuperExperiment(cfg, model_tweak=True)
    exp.run_all()
