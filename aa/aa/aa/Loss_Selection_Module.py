import pandas as pd
from badmintoncleaner import prepare_dataset
import ast
import torch
from tqdm import tqdm
import numpy as np
import os

def calculate_average_column_vectors(csv_files):
    # Create a list to store the column vectors
    column_vectors = []

    # Iterate over each CSV file
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        last_row = df.iloc[-1, 1:].values
        column_vectors.append(last_row.tolist())

    # Calculate the average for each column vector
    averages = np.mean(column_vectors, axis=0)

    return averages
def save_column_vector_to_csv(column_vector, headers, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(column_vector)

    print(f"Average column vector saved to '{output_file}'.")

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # gpu vars

def generate_final_csv(area_model_path, shot_model_path, fold=None):
    SAMPLES = 6  # set to 6 to meet the requirement of this challenge

    # Model for area prediction
    if fold is not None:
        config_area = ast.literal_eval(open(f"{area_model_path}/config{fold}").readline())
    else:
        config_area = ast.literal_eval(open(f"{area_model_path}/config").readline())

    # Model for shot type prediction
    if fold is not None:
        config_shot = ast.literal_eval(open(f"{shot_model_path}/config{fold}").readline())
    else:
        config_shot = ast.literal_eval(open(f"{shot_model_path}/config").readline())

    set_seed(config_area['seed_value'])

    # Prepare Dataset
    config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches = prepare_dataset(
        config_area)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(device)
    # load model
    from MuLMINet.MuLMINet import ShotGenEncoder_MuLMINet_Variant1, ShotGenPredictor_MuLMINet_Variant1, \
        ShotGenEncoder_MuLMINet_Variant2, ShotGenPredictor_MuLMINet_Variant2
    from MuLMINet.MuLMINet_runner import shotgen_generator_MuLMINet
    encoder_area = ShotGenEncoder_MuLMINet_Variant1(config_area)
    decoder_area = ShotGenPredictor_MuLMINet_Variant1(config_area)

    encoder_shot = ShotGenEncoder_MuLMINet_Variant1(config_shot)
    decoder_shot = ShotGenPredictor_MuLMINet_Variant1(config_shot)

    encoder_area.to(device), decoder_area.to(device)
    encoder_shot.to(device), decoder_shot.to(device)

    current_model_path_area = area_model_path + '/'
    current_model_path_shot = shot_model_path + '/'
    if fold is not None:
        encoder_path_area = current_model_path_area + 'encoder' + str(fold)
        decoder_path_area = current_model_path_area + 'decoder' + str(fold)
    else:
        encoder_path_area = current_model_path_area + 'encoder'
        decoder_path_area = current_model_path_area + 'decoder'

    if fold is not None:
        encoder_path_shot = current_model_path_shot + 'encoder' + str(fold)
        decoder_path_shot = current_model_path_shot + 'decoder' + str(fold)
    else:
        encoder_path_shot = current_model_path_shot + 'encoder'
        decoder_path_shot = current_model_path_shot + 'decoder'

    saved_encoder_checkpoint_area = torch.load(encoder_path_area, map_location=device)
    saved_decoder_checkpoint_area = torch.load(decoder_path_area, map_location=device)
    encoder_area.load_state_dict(saved_encoder_checkpoint_area)
    decoder_area.load_state_dict(saved_decoder_checkpoint_area)

    saved_encoder_checkpoint_shot = torch.load(encoder_path_shot, map_location=device)
    saved_decoder_checkpoint_shot = torch.load(encoder_path_shot, map_location=device)
    encoder_shot.load_state_dict(saved_encoder_checkpoint_shot)
    decoder_shot.load_state_dict(saved_decoder_checkpoint_shot)

    encode_length = config['encode_length']
    saved_model_path = '../../../MulMINET_IJCAI'
    performance_log = open(f"{saved_model_path}prediction{fold}.csv", "a")

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

        # feed into the model for shot prediction
        generated_shot_final, generated_area = shotgen_generator_MuLMINet(given_seq=given_seq, encoder=encoder_shot,
                                                                    decoder=decoder_shot, config=config_shot, samples=SAMPLES,
                                                                    device=device)
        # feed into the model for area prediction
        generated_shot, generated_area_final = shotgen_generator_MuLMINet(given_seq=given_seq, encoder=encoder_area,
                                                                    decoder=decoder_area, config=config_area, samples=SAMPLES,
                                                                    device=device)

        # print('length of generated area : ', len(generated_area))

        # store the prediction results
        for sample_id in range(len(generated_area_final)):
            for ball_round in range(len(generated_area_final[0])):
                # print('ball_round : ', ball_round + config['encode_length'] + 1)
                performance_log.write(
                    f"{rally_id},{sample_id},{ball_round + config['encode_length'] + 1},{generated_area_final[sample_id][ball_round][0]},{generated_area_final[sample_id][ball_round][1]},")
                for shot_id, shot_type_logits in enumerate(generated_shot_final[sample_id][ball_round]):
                    performance_log.write(f"{shot_type_logits}")
                    if shot_id != len(generated_shot_final[sample_id][ball_round]) - 1:
                        performance_log.write(",")
                performance_log.write("\n")


if __name__ == "__main__":
    model_path = '../../../MulMINET_IJCAI/'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import os
    optimal_parameter_area = {}
    optimal_parameter_shot = {}
    for model_type in os.listdir(model_path):
        record_score_files = []

        for file_name in os.listdir(model_path + model_type + '/'):
            if "record_score" in file_name:
                record_score_files.append(model_path + model_type + '/' + file_name)

        average_vector = calculate_average_column_vectors(record_score_files)
        # Save the parameter and average value with 5-fold cross validation
        optimal_parameter_area.update({model_type : average_vector[1]})
        optimal_parameter_shot.update({model_type: average_vector[0]})

    key_with_min_value_area = min(optimal_parameter_area, key=lambda x: optimal_parameter_area[x])
    key_with_min_value_shot = min(optimal_parameter_shot, key=lambda x: optimal_parameter_shot[x])
    print('Optimal paramter for predicting area distribution')
    print(key_with_min_value_area)
    area_model_path = '../../../MulMINET_IJCAI/' + key_with_min_value_area

    print('Optimal paramter for predicting shot type')
    print(key_with_min_value_shot)
    shot_model_path = '../../../MulMINET_IJCAI/' + key_with_min_value_shot

    generate_final_csv(area_model_path, shot_model_path, fold=1)
