from badmintoncleaner import prepare_dataset_cross
from badmintoncleaner import prepare_dataset
from utils import draw_loss_cross
import argparse
import torch
import torch.nn as nn
from evaluation import StrokeEvaluator
import torch.distributed
import torch.utils.data.distributed
import torch.multiprocessing
from tqdm import tqdm
from utils import save_fold
from sklearn.model_selection import KFold
import ast

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
                        default=3,
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
                        default=2,
                        help="Number of fold for dataset3")
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


def generate(model_path, fold=None):
    SAMPLES = 6  # set to 6 to meet the requirement of this challenge

    # model_path = './lr0.0001bat32dim256' # sys.argv[1]
    if fold is not None:
        config = ast.literal_eval(open(f"{model_path}/config{fold}").readline())
    else:
        config = ast.literal_eval(open(f"{model_path}/config").readline())

    set_seed(config['seed_value'])

    # Prepare Dataset
    config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches = prepare_dataset(
        config)
    device = torch.device(f"cuda:{config['gpu_num']}" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # load model
    from MuLMINet.MuLMINet import ShotGenEncoder_MuLMINet_Variant1, ShotGenPredictor_MuLMINet_Variant1
    from MuLMINet.MuLMINet_runner import shotgen_generator_MuLMINet
    encoder = ShotGenEncoder_MuLMINet_Variant1(config)
    decoder = ShotGenPredictor_MuLMINet_Variant1(config)

    encoder.to(device), decoder.to(device)

    current_model_path = model_path + '/'
    if fold is not None:
        encoder_path = current_model_path + 'encoder' + str(fold)
        decoder_path = current_model_path + 'decoder' + str(fold)
    else:
        encoder_path = current_model_path + 'encoder'
        decoder_path = current_model_path + 'decoder'

    # print(f"loading {encoder_path} and {decoder_path}")
    saved_encoder_checkpoint = torch.load(encoder_path, map_location=device)
    saved_decoder_checkpoint = torch.load(decoder_path, map_location=device)
    encoder.load_state_dict(saved_encoder_checkpoint)
    decoder.load_state_dict(saved_decoder_checkpoint)

    encode_length = config['encode_length']

    performance_log = open(f"{current_model_path}prediction{fold}.csv", "a")
    performance_log.write(
        'rally_id,sample_id,ball_round,landing_x,landing_y,short service,net shot,lob,clear,drop,push/rush,smash,defensive shot,drive,long service')
    performance_log.write('\n')

    # get all testing rallies
    testing_rallies = test_matches['rally_id'].unique()

    for rally_id in tqdm(testing_rallies):
        # read data
        selected_matches = test_matches.loc[(test_matches['rally_id'] == rally_id)][
            ['rally_id', 'set', 'rally_length', 'ball_round', 'type', 'landing_x', 'landing_y', 'landing_height',
             'aroundhead', 'backhand', 'player', 'player_location_area', 'opponent_location_area']].reset_index(
            drop=True)

        generated_length = selected_matches['rally_length'][0]  # fetch the length of the current rally
        players = [selected_matches['player'][0], selected_matches['player'][1]]
        target_players = torch.tensor([players[shot_index % 2] for shot_index in
                                       range(generated_length - len(selected_matches))])  # get the predicted players

        given_seq = {
            'given_player': torch.tensor(selected_matches['player'].values).to(device),
            'given_shot': torch.tensor(selected_matches['type'].values).to(device),
            'given_x': torch.tensor(selected_matches['landing_x'].values).to(device),
            'given_y': torch.tensor(selected_matches['landing_y'].values).to(device),
            'given_ball_height': torch.tensor(selected_matches['landing_height'].values).to(device),
            'given_ball_aroundhead': torch.tensor(selected_matches['aroundhead'].values).to(device),
            'given_ball_backhand': torch.tensor(selected_matches['backhand'].values).to(device),
            'given_ball_player_area': torch.tensor(selected_matches['player_location_area'].values).to(device),
            'given_ball_opponent_area': torch.tensor(selected_matches['opponent_location_area'].values).to(device),
            'target_player': target_players.to(device),
            'rally_length': generated_length
        }

        # feed into the model
        generated_shot, generated_area = shotgen_generator_MuLMINet(given_seq=given_seq, encoder=encoder,
                                                                    decoder=decoder, config=config, samples=SAMPLES,
                                                                    device=device)

        # print('length of generated area : ', len(generated_area))

        # store the prediction results
        for sample_id in range(len(generated_area)):
            for ball_round in range(len(generated_area[0])):
                # print('ball_round : ', ball_round + config['encode_length'] + 1)
                performance_log.write(
                    f"{rally_id},{sample_id},{ball_round + config['encode_length'] + 1},{generated_area[sample_id][ball_round][0]},{generated_area[sample_id][ball_round][1]},")
                for shot_id, shot_type_logits in enumerate(generated_shot[sample_id][ball_round]):
                    performance_log.write(f"{shot_type_logits}")
                    if shot_id != len(generated_shot[sample_id][ball_round]) - 1:
                        performance_log.write(",")
                performance_log.write("\n")

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars

if __name__ == "__main__":

    import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    lr_list = [0.0001]
    batchsize_list = [32]
    dim_list = [32]
    layer_list = [3]
    alpha_list = [0.4, 0.5]
    
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

                        hyper = 'Variant1lr' + str(lr_value) + 'bat' + str(batch_value) + 'dim' + str(dim_value) + 'alpha' + str(alpha) + 'layer' + str(layer) + 'epoch' + str(config['epochs'])
                        print('')
                        print(hyper)
                        config['output_folder_name'] = os.path.join("./MuLMINET_IJCAI", hyper)
                        config['data_folder'] = './dataset3/'
                        config['model_folder'] = './model/'
                        model_type = config['model_type']
                        set_seed(config['seed_value'])

                        # Clean data and Prepare dataset3
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
                        from MuLMINet.MuLMINet import ShotGenEncoder_MuLMINet_Variant1, ShotGenPredictor_MuLMINet_Variant1
                        from MuLMINet.MuLMINet_runner import shotGen_train_epoch_MuLMINet, shotGen_validate_epoch_MuLMINet
                        kfold = KFold(n_splits=2, shuffle=True)
                        
                        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
                            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) # index 생성
                            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
                            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_subsampler) # 해당하는 index 추출
                            valloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], sampler=val_subsampler)
    
                            encoder = ShotGenEncoder_MuLMINet_Variant1(config)
                            decoder = ShotGenPredictor_MuLMINet_Variant1(config)
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

                            train_loss = 1000000000.000
                            val_loss = 1000000000.000
                            for epoch in tqdm(range(config['epochs']), desc='Epoch: '):
                                new_train_loss = shotGen_train_epoch_MuLMINet(data_loader=trainloader, encoder=encoder, decoder=decoder, criterion=criterion, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, config=config, record_loss = record_train_loss, device=device)
                                new_val_loss = shotGen_validate_epoch_MuLMINet(data_loader=valloader, encoder=encoder, decoder=decoder, criterion=criterion, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, config=config, record_loss = record_validation_loss, device=device)

                                if new_val_loss < val_loss:
                                    val_loss = new_val_loss
                                    save_fold(encoder, decoder, config, new_val_loss, fold, epoch=None)

                            draw_loss_cross(record_train_loss, record_validation_loss, config, fold)

                            # generated the csv file
                            generate(config['output_folder_name'], fold)

                            # evaluate the score
                            stroke_evaluator = StrokeEvaluator(path=config['output_folder_name'], fold=fold)