import os

from main import get_parser
from utils.config import DotDict
from utils.super_experiment import SuperExperiment

if __name__ == '__main__':
    cfg = get_parser(desc='Super Experiment Grid Search', default_file='yaml-config/inf-batch.yaml', add_exp_name=False)
    weights_dir = 'weights'
    postfix = '#cityscapes.pth'
    cfg['train.frozen.network'] = [{'model': os.path.join('yaml-config', 'models', p),
                                    'checkpoint': {'weights': os.path.join(weights_dir, p[:-5] + postfix)}}
                                   for p in os.listdir(os.path.join('yaml-config', 'models'))
                                   if os.path.exists(os.path.join(weights_dir, p[:-5] + postfix))]
    exp = SuperExperiment(cfg, model_tweak=False)
    exp.run_all()
