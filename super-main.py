from main import get_parser
from utils.super_experiment import SuperExperiment

if __name__ == '__main__':
    cfg = get_parser(desc='Super Experiment Grid Search', default_file='yaml-config/lit_review.yaml', add_exp_name=False)
    exp = SuperExperiment(cfg)
    exp.run_all()
