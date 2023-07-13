from badmintoncleaner import prepare_dataset_cross
from utils import draw_loss_cross
import argparse
import os
import torch
import torch.nn as nn
from evaluation import StrokeEvaluator
from generator_model3 import generate_test, generate
import torch.distributed
import torch.utils.data.distributed
import torch.multiprocessing
from tqdm import tqdm
from utils import save_fold
from loss import SupervisedContrastiveLoss
from sklearn.model_selection import KFold


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--model_type",
                        type=str,
                        default='ShuttleNet',
                        help="model type")
    opt.add_argument("--output_folder_name",
                        type=str,
                        help="path to save model",
                        default='test_2')
    opt.add_argument("--seed_value",
                        type=int,
                        default=42,
                        help="seed value")
    opt.add_argument("--max_ball_round",
                        type=int,
                        default=70,
                        help="max of ball round (hard code in this sample code)")
    opt.add_argument("--encode_length",
                        type=int,
                        default=4,
                        help="given encode length")
    opt.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="batch size")
    opt.add_argument("--lr",
                        type=int,
                        default=3e-4,
                        help="learning rate")
    opt.add_argument("--epochs",
                        type=int,
                        default=300,
                        help="epochs")
    opt.add_argument("--n_layers",
                        type=int,
                        default=1,
                        help="number of layers")
    opt.add_argument("--shot_dim",
                        type=int,
                        default=64,
                        help="dimension of shot")
    opt.add_argument("--area_num",
                        type=int,
                        default=10,
                        help="mux, muy, sx, sy, corr")
    opt.add_argument("--area_dim",
                        type=int,
                        default=64,
                        help="dimension of area")
    opt.add_argument("--player_dim",
                        type=int,
                        default=64,
                        help="dimension of player")
    opt.add_argument("--encode_dim",
                        type=int,
                        default=64,
                        help="dimension of hidden")
    opt.add_argument("--num_directions",
                        type=int,
                        default=1,
                        help="number of LSTM directions")
    opt.add_argument("--K",
                        type=int,
                        default=5,
                        help="Number of fold for dataset")
    opt.add_argument("--sample",
                        type=int,
                        default=10,
                        help="Number of samples for evaluation")
    opt.add_argument("--gpu_num",
                        type=int,
                        default=0,
                        help="Selected GPU number")
    opt.add_argument("--alpha",
                        type=float,
                        default=1,
                        help="Selected GPU number")
    opt.add_argument("--dim",
                        type=int,
                        default=64,
                        help="Selected GPU number")
    config = vars(opt.parse_args())
    return config


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


if __name__ == "__main__":

    import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


    # lr_list = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
    # batchsize_list = [32, 64, 128]
    # dim_list = [32, 64, 128, 256]
    # alpha_list = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    lr_list = [0.0001]
    batchsize_list = [32]
    dim_list = [32]
    layer_list = [3]
    alpha_list = [0.4]
    
    for lr_value in lr_list:
        for batch_value in batchsize_list:
            for dim_value in dim_list:
                for alpha in alpha_list:
                    for layer in layer_list:
                        config = get_argument()
                        config['lr'] = lr_value
                        config['batch_size'] = batch_value
                        config['area_dim'] = dim_value
                        config['shot_dim'] = dim_value
                        config['player_dim'] = dim_value
                        config['encode_dim'] = dim_value
                        config['n_layers'] = layer
                        config['alpha'] = alpha

                        # alpha = config['alpha']

                        hyper = 'lr' + str(lr_value) + 'bat' + str(batch_value) + 'dim' + str(dim_value) + 'alpha' + str(alpha) + 'layer' + str(layer) + 'epoch' + str(config['epochs'])
                        print('')
                        print(hyper)
                        config['output_folder_name'] = os.path.join("./model_3_crossvalidation_IJCAI", hyper)
                        config['data_folder'] = './dataset/'
                        config['model_folder'] = './model/'
                        model_type = config['model_type']
                        set_seed(config['seed_value'])

                        # Clean data and Prepare dataset
                        config, train_dataloader, test_dataloader, train_dataset, test_dataset, train_matches, test_matches = prepare_dataset_cross(config)

                        device = torch.device(f"cuda:{config['gpu_num']}" if torch.cuda.is_available() else "cpu")
                        # device = "cpu"
                        print("Model path: {}".format(config['output_folder_name']))
                        if not os.path.exists(config['output_folder_name']):
                            os.makedirs(config['output_folder_name'])
                        else:
                            for file in os.listdir(config['output_folder_name']):
                                os.remove(config['output_folder_name'] + "/" + file)

                        # read model
                        from ShuttleNet.ShuttleNet import ShotGenEncoder_model3, ShotGenPredictor_model3, ShotGenEncoder, ShotGenPredictor
                        from ShuttleNet.ShuttleNet_runner import shotGen_trainer, shotGen_train_epoch_model3, shotGen_validate_epoch_model3
                        kfold = KFold(n_splits=5, shuffle=True)
                        
                        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
                            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) # index 생성
                            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
                            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_subsampler) # 해당하는 index 추출
                            valloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], sampler=val_subsampler)
    
                            encoder = ShotGenEncoder_model3(config)
                            decoder = ShotGenPredictor_model3(config)
                            # encoder = ShotGenEncoder2(config)
                            # decoder = ShotGenPredictor2(config)
                            encoder.area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
                            encoder.shot_embedding.weight = decoder.shotgen_decoder.shot_embedding.weight
                            encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
                            decoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight

                            encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config['lr'])
                            decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config['lr'])

                            # if torch.cuda.is_available():
                            #     if torch.cuda.device_count() > 1:
                            #         print("multigpu")
                            #         encoder = torch.nn.DataParallel(encoder)
                            #         decoder = torch.nn.DataParallel(decoder)
                            # encoder.cuda()
                            # decoder.cuda()
                            encoder.to(device), decoder.to(device)
                            

                            criterion = {
                                'entropy': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
                                # 'entropy': nn.NLLLoss(ignore_index=0, reduction='sum'),
                                'mae': nn.L1Loss(reduction='sum')
                            }

                            
                            for key, value in criterion.items():
                                criterion[key].to(device)

                            record_train_loss = {
                                'total': [],
                                'shot': [],
                                'area': [],
                                'height': [],
                                'aroundhead': [],
                                'backhand': [],
                                'playerloc': [],
                                'opponentloc': []
                            }
                            record_validation_loss = {
                                'total': [],
                                'shot': [],
                                'area': [],
                                'height': [],
                                'aroundhead': [],
                                'backhand': [],
                                'playerloc': [],
                                'opponentloc': []
                            }

                        
                            train_loss = 1000000000
                            val_loss = 1000000000
                            for epoch in tqdm(range(config['epochs']), desc='Epoch: '):
                                new_train_loss = shotGen_train_epoch_model3(data_loader=trainloader, encoder=encoder, decoder=decoder, criterion=criterion, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, config=config, record_loss = record_train_loss, device=device)
                                new_val_loss = shotGen_validate_epoch_model3(data_loader=valloader, encoder=encoder, decoder=decoder, criterion=criterion, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, config=config, record_loss = record_validation_loss, device=device)

                                if new_val_loss < val_loss:
                                    val_loss = new_val_loss
                                    save_fold(encoder, decoder, config, new_val_loss, fold, epoch=None)
                                
                            
                            draw_loss_cross(record_train_loss, record_validation_loss, config, fold)
                            generate(config['output_folder_name'], fold)
                            stroke_evaluator = StrokeEvaluator(path=config['output_folder_name'], fold=fold)