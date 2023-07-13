from badmintoncleaner import prepare_dataset
import ast
import sys
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars

def generate(model_path, fold=None):
    SAMPLES = 6      # set to 6 to meet the requirement of this challenge

    # model_path = './lr0.0001bat32dim256' # sys.argv[1]
    if fold is not None:
        config = ast.literal_eval(open(f"{model_path}/config{fold}").readline())
    else:
        config = ast.literal_eval(open(f"{model_path}/config").readline())

    set_seed(config['seed_value'])

    # Prepare Dataset
    config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches = prepare_dataset(config)
    device = torch.device(f"cuda:{config['gpu_num']}" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # load model
    from ShuttleNet.ShuttleNet import ShotGenEncoder_model3, ShotGenPredictor_model3
    from ShuttleNet.ShuttleNet_runner import shotgen_generator_model3
    encoder = ShotGenEncoder_model3(config)
    decoder = ShotGenPredictor_model3(config)

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
    performance_log.write('rally_id,sample_id,ball_round,landing_x,landing_y,short service,net shot,lob,clear,drop,push/rush,smash,defensive shot,drive,long service')
    performance_log.write('\n')

    # get all testing rallies
    testing_rallies = test_matches['rally_id'].unique()

    for rally_id in tqdm(testing_rallies):
        # read data
        selected_matches = test_matches.loc[(test_matches['rally_id'] == rally_id)][['rally_id', 'set', 'rally_length', 'ball_round', 'type', 'landing_x', 'landing_y', 'landing_height', 'aroundhead', 'backhand', 'player', 'player_location_area', 'opponent_location_area']].reset_index(drop=True)
        
        generated_length = selected_matches['rally_length'][0]      # fetch the length of the current rally
        players = [selected_matches['player'][0], selected_matches['player'][1]]
        target_players = torch.tensor([players[shot_index%2] for shot_index in range(generated_length-len(selected_matches))])  # get the predicted players
        
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
        generated_shot, generated_area = shotgen_generator_model3(given_seq=given_seq, encoder=encoder, decoder=decoder, config=config, samples=SAMPLES, device=device)
        
        # print('length of generated area : ', len(generated_area))

        # store the prediction results
        for sample_id in range(len(generated_area)):
            for ball_round in range(len(generated_area[0])):
                # print('ball_round : ', ball_round + config['encode_length'] + 1)
                performance_log.write(f"{rally_id},{sample_id},{ball_round+config['encode_length']+1},{generated_area[sample_id][ball_round][0]},{generated_area[sample_id][ball_round][1]},")
                for shot_id, shot_type_logits in enumerate(generated_shot[sample_id][ball_round]):
                    performance_log.write(f"{shot_type_logits}")
                    if shot_id != len(generated_shot[sample_id][ball_round]) - 1:
                        performance_log.write(",")
                performance_log.write("\n")



def generate_test(model_path):
    SAMPLES = 6      # set to 6 to meet the requirement of this challenge

    # model_path = './lr0.0001bat32dim256' # sys.argv[1]
    config = ast.literal_eval(open(f"{model_path}/config").readline())
    set_seed(config['seed_value'])

    # Prepare Dataset
    config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches = prepare_dataset(config)
    device = torch.device(f"cuda:{config['gpu_num']}" if torch.cuda.is_available() else "cpu")

    # load model
    from ShuttleNet.ShuttleNet import ShotGenEncoder_model3, ShotGenPredictor_model3
    from ShuttleNet.ShuttleNet_runner import shotgen_generator_model3
    encoder = ShotGenEncoder_model3(config)
    decoder = ShotGenPredictor_model3(config)

    encoder.to(device), decoder.to(device)

    current_model_path = model_path + '/'
    encoder_path = current_model_path + 'encoder'
    decoder_path = current_model_path + 'decoder'

    # print(f"loading {encoder_path} and {decoder_path}")
    saved_encoder_checkpoint = torch.load(encoder_path, map_location=device)
    saved_decoder_checkpoint = torch.load(decoder_path, map_location=device)
    encoder.load_state_dict(saved_encoder_checkpoint)
    decoder.load_state_dict(saved_decoder_checkpoint)

    encode_length = config['encode_length']

    performance_log = open(f"{current_model_path}prediction_test.csv", "a")
    performance_log.write('rally_id,sample_id,ball_round,landing_x,landing_y,short service,net shot,lob,clear,drop,push/rush,smash,defensive shot,drive,long service')
    performance_log.write('\n')

    # get all testing rallies
    testing_rallies = test_matches['rally_id'].unique()

    for rally_id in tqdm(testing_rallies):
        # read data
        selected_matches = test_matches.loc[(test_matches['rally_id'] == rally_id)][['rally_id', 'set', 'rally_length', 'ball_round', 'type', 'landing_x', 'landing_y', 'landing_height', 'aroundhead', 'backhand', 'player', 'player_location_area', 'opponent_location_area']].reset_index(drop=True)
        
        generated_length = selected_matches['rally_length'][0]      # fetch the length of the current rally
        players = [selected_matches['player'][0], selected_matches['player'][1]]
        target_players = torch.tensor([players[shot_index%2] for shot_index in range(generated_length-len(selected_matches))])  # get the predicted players
        
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
        generated_shot, generated_area = shotgen_generator_model3(given_seq=given_seq, encoder=encoder, decoder=decoder, config=config, samples=SAMPLES, device=device)
        
        # print('length of generated area : ', len(generated_area))

        # store the prediction results
        for sample_id in range(len(generated_area)):
            for ball_round in range(len(generated_area[0])):
                # print('ball_round : ', ball_round + config['encode_length'] + 1)
                performance_log.write(f"{rally_id},{sample_id},{ball_round+config['encode_length']+1},{generated_area[sample_id][ball_round][0]},{generated_area[sample_id][ball_round][1]},")
                for shot_id, shot_type_logits in enumerate(generated_shot[sample_id][ball_round]):
                    performance_log.write(f"{shot_type_logits}")
                    if shot_id != len(generated_shot[sample_id][ball_round]) - 1:
                        performance_log.write(",")
                performance_log.write("\n")

if __name__ == "__main__":
    model_path = '/home/hcis/Desktop/Track 2_ Stroke Forecasting_final_revision/src/model_3_crossvalidation/lr0.0001bat32dim32alpha0.4layer1epoch300'
    generate(model_path, fold=1)
