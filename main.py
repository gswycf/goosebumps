import torch
import argparse
import yaml
def get_parser():

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='The pytorch implementation for Visual Alignment Constraint '
                    'for Continuous Sign Language Recognition.')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='./baseline.yaml',
        help='path to the configuration file')


    parser.add_argument(
        '--save-interval',
        type=int,
        default=200,
        help='the interval for storing models (#epochs)')
    parser.add_argument(
        '--random-seed',
        type=int,
        default=0,
        help='the default value for random seed.')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')

    parser.add_argument(
        '--feeder-args',
        default=dict(),
        help='the arguments of Dataprocessing loader')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=20,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='the interval for evaluating models (#epochs)')
    default_optimizer_dict = {
        "base_lr": 1e-2,
        "optimizer": "SGD",
        "nesterov": False,
        "step": [5, 10],
        "weight_decay": 0.00005,
        "start_epoch": 1,
    }
    parser.add_argument(
        '--optimizer-args',
        default=default_optimizer_dict,
        help='the arguments of optimizer')


    parser.add_argument(
        '--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=8, help='test batch size')
    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.data_loader = {}
        self.model, self.optimizer = self.loading()

    def loading(self):
        from model import GooseModel
        from optimizer import Optimizer
        net =GooseModel()
        optimizer = Optimizer(net, self.arg.optimizer_args)
        print("Loading model finished.")
        return net, optimizer

    def load_data(self):
        print("Loading Dataprocessing")
        from dataloader import feeder
        self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
        dataset_list = zip(["train", "dev", "test"], [True, False, False])

        for idx, (mode, train_flag) in enumerate(dataset_list):
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading Dataprocessing finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=10,
            pin_memory=True,
        )

    def start(self):
        if self.arg.phase == 'train':
            best_acc = 0.0
            best_epoch = 0
            total_time = 0
            print('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(0,20):

                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch)
                if eval_model:
                    acc = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    acc = seq_eval(self.arg, self.data_loader['test'], self.model, self.device,
                                       'test', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    print("Dev WER: {:05.2f}% Test WER {:05.2f}%".format(acc, acc))


if __name__ == '__main__':
    sparser = get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    print(args)
    processor = Processor(args)
    processor.start()

