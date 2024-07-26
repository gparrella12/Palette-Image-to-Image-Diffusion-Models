import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from data.dataset import ReconstructionDatasetTest

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    dataset = ReconstructionDatasetTest(args.path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(os.listdir(os.path.join(args.path))), shuffle=False)

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    
    model.test_dataloader(data_loader, args.out_dir)
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/user/gparrella/Palette-Image-to-Image-Diffusion-Models/config/reconstruction2_test.json', help='JSON file for configuration')
    parser.add_argument('-p', '--path', type=str, default='/user/gparrella/cpa_enhanced/datasets/new_reconstructions/x', help='Path to folder that contains images to restore.')
    parser.add_argument('-ph', '--phase', type=str, choices=['train','test'], help='Run train or test', default='test')
    parser.add_argument('-o', '--out_dir', type=str, default='/user/gparrella/Palette-Image-to-Image-Diffusion-Models/to_test', help='Output directory in which save images.')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    opt['world_size'] = 1 
    main_worker(0, 1, opt)