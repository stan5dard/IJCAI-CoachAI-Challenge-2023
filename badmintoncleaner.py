from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

PAD = 0
class BadmintonDataset(Dataset):
    def __init__(self, matches, config):
        super().__init__()
        self.max_ball_round = config['max_ball_round'] # 70
        group = matches[['rally_id', 'set', 'ball_round', 'roundscore_A', 'roundscore_B', 'type', 'landing_x', 'landing_y', 'landing_height', 'aroundhead', 'backhand', 'player', 'player_location_area', 'opponent_location_area']].groupby('rally_id').apply(lambda r: (r['set'].values, r['ball_round'].values,  r['roundscore_A'].values,  r['roundscore_B'].values, r['type'].values, r['landing_x'].values, r['landing_y'].values, r['landing_height'].values, r['aroundhead'].values, r['backhand'].values, r['player'].values, r['player_location_area'].values, r['opponent_location_area'].values))

        self.sequences, self.rally_ids = {}, []
        for i, rally_id in enumerate(group.index):
            sets, ball_round, roundscore_A, roundscore_B, shot_type, landing_x, landing_y, landing_height, aroundhead, backhand, player, player_location_area, opponent_location_area = group[rally_id]
            self.sequences[rally_id] = (sets, ball_round, roundscore_A, roundscore_B, shot_type, landing_x, landing_y, landing_height, aroundhead, backhand, player, player_location_area, opponent_location_area)
            self.rally_ids.append(rally_id)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        rally_id = self.rally_ids[index]
        sets, ball_round, roundscore_A, roundscore_B, shot_type, landing_x, landing_y, landing_height, aroundhead, backhand, player, player_location_area, opponent_location_area = self.sequences[rally_id]

        pad_input_roundscore_A = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_roundscore_B = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_shot_type = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_landing_x = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_input_landing_y = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_input_landing_height = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_aroundhead = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_backhand = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_player = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_player_location_area = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_opponent_location_area = np.full(self.max_ball_round, fill_value=PAD, dtype=int)

        pad_output_roundscore_A = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_roundscore_B = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_shot_type = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_landing_x = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_output_landing_y = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_output_landing_height = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_aroundhead = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_backhand = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_player = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_player_location_area = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_opponent_location_area = np.full(self.max_ball_round, fill_value=PAD, dtype=int)

        # pad or trim based on the max ball round
        if len(ball_round) > self.max_ball_round:
            rally_len = self.max_ball_round

            pad_input_roundscore_A[:] = roundscore_A[0:-1:1][:rally_len]  
            pad_input_roundscore_B[:] = roundscore_B[0:-1:1][:rally_len]  
            pad_input_shot_type[:] = shot_type[0:-1:1][:rally_len]                                   # 0, 1, ..., max_ball_round-1
            pad_input_landing_x[:] = landing_x[0:-1:1][:rally_len]
            pad_input_landing_y[:] = landing_y[0:-1:1][:rally_len]
            pad_input_landing_height[:] = landing_height[0:-1:1][:rally_len]
            pad_input_aroundhead [:] = aroundhead[0:-1:1][:rally_len]
            pad_input_backhand [:] = backhand[0:-1:1][:rally_len]
            pad_input_player[:] = player[0:-1:1][:rally_len]
            pad_input_player_location_area[:] = player_location_area[0:-1:1][:rally_len]
            pad_input_opponent_location_area[:] = opponent_location_area[0:-1:1][:rally_len]

            pad_output_roundscore_A[:] = roundscore_A[1::1][:rally_len]  
            pad_output_roundscore_B[:] = roundscore_B[1::1][:rally_len]  
            pad_output_shot_type[:] = shot_type[1::1][:rally_len]  # 0, 1, ..., max_ball_round-1
            pad_output_landing_x[:] = landing_x[1::1][:rally_len]
            pad_output_landing_y[:] = landing_y[1::1][:rally_len]
            pad_output_landing_height[:] = landing_height[1::1][:rally_len]
            pad_output_aroundhead[:] = aroundhead[1::1][:rally_len]
            pad_output_backhand[:] = backhand[1::1][:rally_len]
            pad_output_player[:] = player[1::1][:rally_len]
            pad_output_player_location_area[:] = player_location_area[1::1][:rally_len]
            pad_output_opponent_location_area[:] = opponent_location_area[1::1][:rally_len]

        else:
            rally_len = len(ball_round) - 1                                                     # 0 ~ (n-2)

            pad_input_roundscore_A[:rally_len] = roundscore_A[0:-1:1][:rally_len]  
            pad_input_roundscore_B[:rally_len] = roundscore_B[0:-1:1][:rally_len] 
            pad_input_shot_type[:rally_len] = shot_type[0:-1:1][:rally_len]  # 0, 1, ..., max_ball_round-1
            pad_input_landing_x[:rally_len] = landing_x[0:-1:1][:rally_len]
            pad_input_landing_y[:rally_len] = landing_y[0:-1:1][:rally_len]
            pad_input_landing_height[:rally_len] = landing_height[0:-1:1][:rally_len]
            pad_input_aroundhead[:rally_len] = aroundhead[0:-1:1][:rally_len]
            pad_input_backhand[:rally_len] = backhand[0:-1:1][:rally_len]
            pad_input_player[:rally_len] = player[0:-1:1][:rally_len]
            pad_input_player_location_area[:rally_len] = player_location_area[0:-1:1][:rally_len]
            pad_input_opponent_location_area[:rally_len] = opponent_location_area[0:-1:1][:rally_len]

            pad_output_roundscore_A[:rally_len] = roundscore_A[1::1][:rally_len]  
            pad_output_roundscore_B[:rally_len] = roundscore_B[1::1][:rally_len]  
            pad_output_shot_type[:rally_len] = shot_type[1::1][:rally_len]  # 0, 1, ..., max_ball_round-1
            pad_output_landing_x[:rally_len] = landing_x[1::1][:rally_len]
            pad_output_landing_y[:rally_len] = landing_y[1::1][:rally_len]
            pad_output_landing_height[:rally_len] = landing_height[1::1][:rally_len]
            pad_output_aroundhead[:rally_len] = aroundhead[1::1][:rally_len]
            pad_output_backhand[:rally_len] = backhand[1::1][:rally_len]
            pad_output_player[:rally_len] = player[1::1][:rally_len]
            pad_output_player_location_area[:rally_len] = player_location_area[1::1][:rally_len]
            pad_output_opponent_location_area[:rally_len] = opponent_location_area[1::1][:rally_len]

        return (pad_input_shot_type, pad_input_landing_x, pad_input_landing_y, pad_input_landing_height, pad_input_aroundhead, pad_input_backhand, pad_input_player, pad_input_player_location_area, pad_input_opponent_location_area,
                pad_output_shot_type, pad_output_landing_x, pad_output_landing_y, pad_output_landing_height, pad_output_aroundhead, pad_output_backhand, pad_output_player, pad_output_player_location_area, pad_output_opponent_location_area, rally_len, sets[0])

def prepare_dataset2(config, train_proportion=0.8, val_proportion=0.1, test_proportion=0.1):
    train_matches = pd.read_csv(f"{config['data_folder']}train.csv")
    val_matches = pd.read_csv(f"{config['data_folder']}val_given.csv")
    test_matches = pd.read_csv(f"{config['data_folder']}test_given.csv")

    # encode shot type
    codes_type, uniques_type = pd.factorize(train_matches['type'])
    # print(codes_type, uniques_type)
    train_matches['type'] = codes_type + 1                                # Reserve code 0 for paddings
    val_matches['type'] = val_matches['type'].apply(lambda x: list(uniques_type).index(x) + 1)
    test_matches['type'] = test_matches['type'].apply(lambda x: list(uniques_type).index(x) + 1)
    config['uniques_type'] = uniques_type.to_list()
    config['shot_num'] = len(uniques_type) + 1                            # Add padding

    # encode player
    train_matches['player'] = train_matches['player'].apply(lambda x: x+1)
    val_matches['player'] = val_matches['player'].apply(lambda x: x+1)
    test_matches['player'] = test_matches['player'].apply(lambda x: x+1)
    config['player_num'] = 35 + 1                                         # Add padding

    # encode aroundhead
    train_matches['aroundhead'] = train_matches['aroundhead'].apply(lambda x: x+1)
    val_matches['aroundhead'] = val_matches['aroundhead'].apply(lambda x: x+1)
    test_matches['aroundhead'] = test_matches['aroundhead'].apply(lambda x: x+1)

    # encode backhand
    train_matches['backhand'] = train_matches['backhand'].apply(lambda x: x+1)
    val_matches['backhand'] = val_matches['backhand'].apply(lambda x: x+1)
    test_matches['backhand'] = test_matches['backhand'].apply(lambda x: x+1)

    # encode score
    # train_matches['roundscore_A'] = train_matches['roundscore_A'].apply(lambda x: x+1)
    # val_matches['roundscore_A'] = val_matches['roundscore_A'].apply(lambda x: x+1)
    # test_matches['roundscore_A'] = test_matches['roundscore_A'].apply(lambda x: x+1)
    # train_matches['roundscore_B'] = train_matches['roundscore_B'].apply(lambda x: x+1)
    # val_matches['roundscore_B'] = val_matches['roundscore_B'].apply(lambda x: x+1)
    # test_matches['roundscore_B'] = test_matches['roundscore_B'].apply(lambda x: x+1)

    total_dataset = BadmintonDataset(train_matches, config)
    train_len = int(len(total_dataset) * train_proportion)
    validate_len = int(len(total_dataset) * val_proportion)
    test_len = len(total_dataset) - train_len - validate_len

    print(f"split train: {train_len}     val: {validate_len}     test: {test_len}")
    train_dataset, val_dataset, test_dataset = random_split(total_dataset, [train_len, validate_len, test_len])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # val_dataset = BadmintonDataset(val_matches, config)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # test_dataset = BadmintonDataset(test_matches, config)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches

def prepare_dataset(config):
    train_matches = pd.read_csv(f"{config['data_folder']}train_0.85.csv")
    val_matches = pd.read_csv(f"{config['data_folder']}validation_0.15.csv")
    test_matches = pd.read_csv(f"{config['data_folder']}test_question_0.15.csv")

    # encode shot type
    codes_type, uniques_type = pd.factorize(train_matches['type'])
    # print(codes_type, uniques_type)
    train_matches['type'] = codes_type + 1                                # Reserve code 0 for paddings
    val_matches['type'] = val_matches['type'].apply(lambda x: list(uniques_type).index(x) + 1)
    test_matches['type'] = test_matches['type'].apply(lambda x: list(uniques_type).index(x) + 1)
    config['uniques_type'] = uniques_type.to_list()
    config['shot_num'] = len(uniques_type) + 1                            # Add padding

    # encode player
    train_matches['player'] = train_matches['player'].apply(lambda x: x+1)
    val_matches['player'] = val_matches['player'].apply(lambda x: x+1)
    test_matches['player'] = test_matches['player'].apply(lambda x: x+1)
    config['player_num'] = 35 + 1                                         # Add padding

    # encode aroundhead
    train_matches['aroundhead'] = train_matches['aroundhead'].apply(lambda x: x+1)
    val_matches['aroundhead'] = val_matches['aroundhead'].apply(lambda x: x+1)
    test_matches['aroundhead'] = test_matches['aroundhead'].apply(lambda x: x+1)

    # encode backhand
    train_matches['backhand'] = train_matches['backhand'].apply(lambda x: x+1)
    val_matches['backhand'] = val_matches['backhand'].apply(lambda x: x+1)
    test_matches['backhand'] = test_matches['backhand'].apply(lambda x: x+1)

    # encode score
    # train_matches['roundscore_A'] = train_matches['roundscore_A'].apply(lambda x: x+1)
    # val_matches['roundscore_A'] = val_matches['roundscore_A'].apply(lambda x: x+1)
    # test_matches['roundscore_A'] = test_matches['roundscore_A'].apply(lambda x: x+1)
    # train_matches['roundscore_B'] = train_matches['roundscore_B'].apply(lambda x: x+1)
    # val_matches['roundscore_B'] = val_matches['roundscore_B'].apply(lambda x: x+1)
    # test_matches['roundscore_B'] = test_matches['roundscore_B'].apply(lambda x: x+1)

    train_dataset = BadmintonDataset(train_matches, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    val_dataset = BadmintonDataset(val_matches, config)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    test_dataset = BadmintonDataset(test_matches, config)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches

def prepare_dataset_cross(config):
    train_matches = pd.read_csv(f"{config['data_folder']}train_0.85.csv")
    val_matches = pd.read_csv(f"{config['data_folder']}validation_0.15.csv")
    test_matches = pd.read_csv(f"{config['data_folder']}test_question_0.15.csv")

    # encode shot type
    codes_type, uniques_type = pd.factorize(train_matches['type'])
    # print(codes_type, uniques_type)
    train_matches['type'] = codes_type + 1                                # Reserve code 0 for paddings
    # val_matches['type'] = val_matches['type'].apply(lambda x: list(uniques_type).index(x) + 1)
    test_matches['type'] = test_matches['type'].apply(lambda x: list(uniques_type).index(x) + 1)
    config['uniques_type'] = uniques_type.to_list()
    config['shot_num'] = len(uniques_type) + 1                            # Add padding

    # encode player
    train_matches['player'] = train_matches['player'].apply(lambda x: x+1)
    # val_matches['player'] = val_matches['player'].apply(lambda x: x+1)
    test_matches['player'] = test_matches['player'].apply(lambda x: x+1)
    config['player_num'] = 35 + 1                                         # Add padding

    # encode aroundhead
    train_matches['aroundhead'] = train_matches['aroundhead'].apply(lambda x: x+1)
    # val_matches['aroundhead'] = val_matches['aroundhead'].apply(lambda x: x+1)
    test_matches['aroundhead'] = test_matches['aroundhead'].apply(lambda x: x+1)

    # encode backhand
    train_matches['backhand'] = train_matches['backhand'].apply(lambda x: x+1)
    # val_matches['backhand'] = val_matches['backhand'].apply(lambda x: x+1)
    test_matches['backhand'] = test_matches['backhand'].apply(lambda x: x+1)

    # encode score
    # train_matches['roundscore_A'] = train_matches['roundscore_A'].apply(lambda x: x+1)
    # val_matches['roundscore_A'] = val_matches['roundscore_A'].apply(lambda x: x+1)
    # test_matches['roundscore_A'] = test_matches['roundscore_A'].apply(lambda x: x+1)
    # train_matches['roundscore_B'] = train_matches['roundscore_B'].apply(lambda x: x+1)
    # val_matches['roundscore_B'] = val_matches['roundscore_B'].apply(lambda x: x+1)
    # test_matches['roundscore_B'] = test_matches['roundscore_B'].apply(lambda x: x+1)

    train_dataset = BadmintonDataset(train_matches, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # val_dataset = BadmintonDataset(val_matches, config)
    # val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    test_dataset = BadmintonDataset(test_matches, config)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return config, train_dataloader, test_dataloader, train_dataset, test_dataset, train_matches, test_matches