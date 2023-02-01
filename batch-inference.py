import copy
import glob
import os

from main import get_parser
from utils.config import DotDict
from utils.super_experiment import SuperExperiment, decode_tag

if __name__ == '__main__':
    cfg = get_parser(desc='Super Experiment Grid Search', default_file='yaml-config/inf-batch.yaml', add_exp_name=False)
    # weights_dir = 'weights'
    # postfix = '#cityscapes.pth'
    # cfg['train.frozen.network'] = [{'model': os.path.join('yaml-config', 'models', p),
    #                                 'checkpoint': {'weights': os.path.join(weights_dir, p[:-5] + postfix)}}
    #                                for p in os.listdir(os.path.join('yaml-config', 'models'))
    #                                if os.path.exists(os.path.join(weights_dir, p[:-5] + postfix))]

    res_dir = 'results/super-experiments'
    for d1 in os.listdir(res_dir):
        for d2 in os.listdir(os.path.join(res_dir, d1)):
            p = os.path.join(res_dir, d1, d2, 'models')
            for file in glob.glob('*.pth', root_dir=p):
                filename = file[:-12]
                config = decode_tag(filename)
                new_cfg = copy.deepcopy(cfg)
                new_cfg.data = [config.data]
                new_cfg['train.frozen.network'] = [{'model': os.path.join('yaml-config', 'models', config.network),
                                                    'checkpoint': {'weights': os.path.join(p, file)}}]
                new_cfg['image_size'] = [config.image_size]
                exp = SuperExperiment(new_cfg, model_tweak=False)
                print(d1, d2, filename)
                exp.run_all()

    # exp = SuperExperiment(cfg, model_tweak=False)
    # exp.run_all()
