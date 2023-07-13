import torch
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
import numpy as np
from tqdm import tqdm
from utils import save
from loss import SupervisedContrastiveLoss

PAD = 0

def Gaussian2D_loss(V_pred, V_trgt):
    """
    Compute NLL on 2D loss. Refer to paper for more details
    """
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, 0] - V_pred[:, 0]
    normy = V_trgt[:, 1] - V_pred[:, 1]

    sx = torch.exp(V_pred[:, 2]) #sx
    sy = torch.exp(V_pred[:, 3]) #sy
    corr = torch.tanh(V_pred[:, 4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.sum(result)
    
    return result


def shotGen_validate_epoch(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        # encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        output_shot_logits = output_shot_logits[pad_mask]
        # print(target_shot)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] < 0:
                    lst[i] = 1
        
        replace_value(target_height)
        # print(target_height.shape)
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_validate_epoch_model3(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        output_shot_logits = output_shot_logits[pad_mask]
        # print(target_shot)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_validate_epoch_model_hybrid(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encoder_result = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, encoder_result, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        output_shot_logits = output_shot_logits[pad_mask]
        # print(target_shot)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_validate_epoch_model3_crl(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        output_shot_logits = output_shot_logits[pad_mask]
        # print(target_shot)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_validate_epoch_model3_small(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        # input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        # input_player_location_area = batch_input_player_location_area[:, :encode_length]
        # input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        # input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        # input_player_location_area = batch_input_player_location_area[:, encode_length:]
        # input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        # target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        # target_player_location_area = batch_target_player_location_area[:, encode_length:]
        # target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_aroundhead_logits, output_backhand_logits = decoder(input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand, encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        output_shot_logits = output_shot_logits[pad_mask]
        # print(target_shot)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        # output_height_logits = output_height_logits[pad_mask]
        # target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        # output_playerloc_logits = output_playerloc_logits[pad_mask]
        # target_player_location_area = target_player_location_area[pad_mask]
        # output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        # target_opponent_location_area = target_opponent_location_area[pad_mask]

        # _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        # def replace_value(lst):
        #     for i in range(len(lst)):
        #         if lst[i] == -9223372036854775808:
        #             lst[i] = 1
        #         elif lst[i] == -9223372036854775807:
        #             lst[i] = 2
        
        # replace_value(target_height)
        # loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        # loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        # loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_aroundhead + loss_backhand)
        # loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        # total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        # total_playerloc_loss += loss_playerloc.item()
        # total_opponentloc_loss += loss_opponentloc.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    # total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    # total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    # total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    # record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    # record_loss['playerloc'].append(total_playerloc_loss)
    # record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_validate_epoch_simple(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]


        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        output_shot_logits = output_shot_logits[pad_mask]
        # print(target_shot)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        loss = config['alpha'] * loss_area + (1 - config['alpha']) * loss_shot
        # loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()


    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)


    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    return total_loss
def shotGen_validate_epoch_Big(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        output_shot_logits = output_shot_logits[pad_mask]
        # print(target_shot)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_validate_epoch_Big2(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss, total_playerxy, total_opponentxy = 0, 0, 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)
        batch_input_player_location_x, batch_input_player_location_y = item[20].to(device), item[21].to(device)
        batch_input_opponent_location_x, batch_input_opponent_location_y = item[22].to(device), item[23].to(device)
        batch_target_player_location_x, batch_target_player_location_y = item[24].to(device), item[25].to(device)
        batch_target_opponent_location_x, batch_target_opponent_location_y = item[26].to(device), item[27].to(device)

        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]
        input_player_location_x = batch_input_player_location_x[:, :encode_length]
        input_player_location_y = batch_input_player_location_y[:, :encode_length]
        input_opponent_location_x = batch_input_opponent_location_x[:, :encode_length]
        input_opponent_location_y = batch_input_opponent_location_y[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                         input_player_location_area, input_opponent_location_area,
                                                                                                                                                                                                                                                                                                                                                                                                                                                         input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]
        input_player_location_x = batch_input_player_location_x[:, encode_length:]
        input_player_location_y = batch_input_player_location_y[:, encode_length:]
        input_opponent_location_x = batch_input_opponent_location_x[:, encode_length:]
        input_opponent_location_y = batch_input_opponent_location_y[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]
        target_player_location_x = batch_target_player_location_x[:, encode_length:]
        target_player_location_y = batch_target_player_location_y[:, encode_length:]
        target_opponent_location_x = batch_target_opponent_location_x[:, encode_length:]
        target_opponent_location_y = batch_target_opponent_location_y[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits, output_pl_xy, output_ol_xy = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y,
                                                                        encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol,
                                                                        encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_pl_B,
                                                                        encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        output_pl_xy = output_pl_xy[pad_mask]
        output_ol_xy = output_ol_xy[pad_mask]
        output_shot_logits = output_shot_logits[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]
        target_pl_x = target_player_location_x[pad_mask]
        target_pl_y = target_player_location_y[pad_mask]
        target_ol_x = target_opponent_location_x[pad_mask]
        target_ol_y = target_opponent_location_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        # _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        gold_pl_xy = torch.cat((target_pl_x.unsqueeze(-1), target_pl_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        loss_playerxy = Gaussian2D_loss(output_pl_xy, gold_pl_xy)

        gold_ol_xy = torch.cat((target_ol_x.unsqueeze(-1), target_ol_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        loss_opponentxy = Gaussian2D_loss(output_ol_xy, gold_ol_xy)

        total_instance += len(target_shot)
        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc + loss_playerxy + loss_opponentxy)
        # loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()
        total_playerxy += loss_playerxy.item()
        total_opponentxy += loss_opponentxy.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)
    total_playerxy = round(total_playerxy / total_instance, 4)
    total_opponentxy = round(total_opponentxy / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)
    record_loss['playerxy'].append(total_playerxy)
    record_loss['opponentxy'].append(total_opponentxy)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss


def shotGen_train_epoch(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, record_loss, encoder_scheduler=None, decoder_scheduler=None, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder

    encoder.train(), decoder.train()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)
        # print(batch_target_shot_type.shape)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        # encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        # print(output_shot_logits.shape)
        output_shot_logits = output_shot_logits[pad_mask]
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        # loss_shot = criterion['supcrl'](output_shot_logits, target_shot.long())
        # loss_shot = criterion['supcrl'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        # loss_height = criterion['supcrl'](output_height_logits, target_height)
        # loss_aroundhead = criterion['supcrl'](output_aroundhead_logits, target_aroundhead.long())
        # loss_backhand = criterion['supcrl'](output_backhand_logits, target_backhand.long())
        # loss_playerloc = criterion['supcrl'](output_playerloc_logits, target_player_location_area.long())
        # loss_opponentloc = criterion['supcrl'](output_opponentloc_logits, target_opponent_location_area.long())
        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss = loss_shot
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    if encoder_scheduler is not None:
        encoder_scheduler.step()
    if decoder_scheduler is not None:
        decoder_scheduler.step()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_train_epoch_model3(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, record_loss, encoder_scheduler=None, decoder_scheduler=None, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder

    encoder.train(), decoder.train()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)

        seq_len, seq_sets = item[18].to(device), item[19].to(device)
        # print(batch_target_shot_type.shape)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                                                                        encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        pad_mask = (input_shot!=PAD)
        # print(output_shot_logits.shape)
        output_shot_logits = output_shot_logits[pad_mask]
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        # loss_shot = criterion['supcrl'](output_shot_logits, target_shot.long())
        # loss_shot = criterion['supcrl'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        # loss_height = criterion['supcrl'](output_height_logits, target_height)
        # loss_aroundhead = criterion['supcrl'](output_aroundhead_logits, target_aroundhead.long())
        # loss_backhand = criterion['supcrl'](output_backhand_logits, target_backhand.long())
        # loss_playerloc = criterion['supcrl'](output_playerloc_logits, target_player_location_area.long())
        # loss_opponentloc = criterion['supcrl'](output_opponentloc_logits, target_opponent_location_area.long())
        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss = loss_shot
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    if encoder_scheduler is not None:
        encoder_scheduler.step()
    if decoder_scheduler is not None:
        decoder_scheduler.step()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss

def shotGen_train_epoch_model3_crl(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, record_loss, encoder_scheduler=None, decoder_scheduler=None, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder

    encoder.train(), decoder.train()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)
        # print(batch_target_shot_type.shape)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                                                                        encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        pad_mask = (input_shot!=PAD)
        # print(output_shot_logits.shape)
        output_shot_logits = output_shot_logits[pad_mask]
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] < 0:
                    lst[i] = 1

        
        replace_value(target_height)
        print(output_shot_logits.shape)
        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        # loss_shot = criterion['supcrl'](output_shot_logits, target_shot.long())
        # loss_shot = criterion['supcrl'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        # loss_height = criterion['supcrl'](output_height_logits, target_height)
        # loss_aroundhead = criterion['supcrl'](output_aroundhead_logits, target_aroundhead.long())
        # loss_backhand = criterion['supcrl'](output_backhand_logits, target_backhand.long())
        # loss_playerloc = criterion['supcrl'](output_playerloc_logits, target_player_location_area.long())
        # loss_opponentloc = criterion['supcrl'](output_opponentloc_logits, target_opponent_location_area.long())
        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss = loss_shot
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    if encoder_scheduler is not None:
        encoder_scheduler.step()
    if decoder_scheduler is not None:
        decoder_scheduler.step()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_train_epoch_simple(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, record_loss, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder

    encoder.train(), decoder.train()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)
        # print(batch_target_shot_type.shape)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]


        # print('input_player : ', input_player[0].shape)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]


        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]


        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_local_output_area, encode_local_output_shot, 
                                                                        encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        pad_mask = (input_shot!=PAD)
        # print(output_shot_logits.shape)
        output_shot_logits = output_shot_logits[pad_mask]
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_area = Gaussian2D_loss(output_xy, gold_xy)
        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
 
        loss = config['alpha'] * loss_area + (1 - config['alpha']) * loss_shot
        # loss = loss_shot
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()


    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    return total_loss
def shotGen_train_epoch_Big(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, record_loss, encoder_scheduler=None, decoder_scheduler=None, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder

    encoder.train(), decoder.train()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)
        # print(batch_target_shot_type.shape)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        # print(target_shot)
        # print(target_shot)
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, target_player)

        pad_mask = (input_shot!=PAD)
        # print(output_shot_logits.shape)
        output_shot_logits = output_shot_logits[pad_mask]
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        # loss_shot = criterion['supcrl'](output_shot_logits, target_shot.long())
        # loss_shot = criterion['supcrl'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        # loss_height = criterion['supcrl'](output_height_logits, target_height)
        # loss_aroundhead = criterion['supcrl'](output_aroundhead_logits, target_aroundhead.long())
        # loss_backhand = criterion['supcrl'](output_backhand_logits, target_backhand.long())
        # loss_playerloc = criterion['supcrl'](output_playerloc_logits, target_player_location_area.long())
        # loss_opponentloc = criterion['supcrl'](output_opponentloc_logits, target_opponent_location_area.long())
        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
        # loss = loss_shot
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()

    if encoder_scheduler is not None:
        encoder_scheduler.step()
    if decoder_scheduler is not None:
        decoder_scheduler.step()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_train_epoch_Big2(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, record_loss, encoder_scheduler=None, decoder_scheduler=None, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss, total_playerxy, total_opponentxy = 0, 0, 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)
        batch_input_player_location_x, batch_input_player_location_y = item[20].to(device), item[21].to(device)
        batch_input_opponent_location_x, batch_input_opponent_location_y = item[22].to(device), item[23].to(device)
        batch_target_player_location_x, batch_target_player_location_y = item[24].to(device), item[25].to(device)
        batch_target_opponent_location_x, batch_target_opponent_location_y = item[26].to(device), item[27].to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        input_player_location_area = batch_input_player_location_area[:, :encode_length]
        input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]
        input_player_location_x = batch_input_player_location_x[:, :encode_length]
        input_player_location_y = batch_input_player_location_y[:, :encode_length]
        input_opponent_location_x = batch_input_opponent_location_x[:, :encode_length]
        input_opponent_location_y = batch_input_opponent_location_y[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                         input_player_location_area, input_opponent_location_area,
                                                                                                                                                                                                                                                                                                                                                                                                                                                         input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        input_player_location_area = batch_input_player_location_area[:, encode_length:]
        input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]
        input_player_location_x = batch_input_player_location_x[:, encode_length:]
        input_player_location_y = batch_input_player_location_y[:, encode_length:]
        input_opponent_location_x = batch_input_opponent_location_x[:, encode_length:]
        input_opponent_location_y = batch_input_opponent_location_y[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        target_player_location_area = batch_target_player_location_area[:, encode_length:]
        target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]
        target_player_location_x = batch_target_player_location_x[:, encode_length:]
        target_player_location_y = batch_target_player_location_y[:, encode_length:]
        target_opponent_location_x = batch_target_opponent_location_x[:, encode_length:]
        target_opponent_location_y = batch_target_opponent_location_y[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits, output_pl_xy, output_ol_xy = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                        input_player_location_area, input_opponent_location_area, input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y,
                                                                        encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol,
                                                                        encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A,
                                                                        encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
      
        pad_mask = (input_shot!=PAD)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        output_pl_xy = output_pl_xy[pad_mask]
        output_ol_xy = output_ol_xy[pad_mask]
        output_shot_logits = output_shot_logits[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]
        target_pl_x = target_player_location_x[pad_mask]
        target_pl_y = target_player_location_y[pad_mask]
        target_ol_x = target_opponent_location_x[pad_mask]
        target_ol_y = target_opponent_location_y[pad_mask]

        output_height_logits = output_height_logits[pad_mask]
        target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        output_playerloc_logits = output_playerloc_logits[pad_mask]
        target_player_location_area = target_player_location_area[pad_mask]
        output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        target_opponent_location_area = target_opponent_location_area[pad_mask]

        # _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        gold_pl_xy = torch.cat((target_pl_x.unsqueeze(-1), target_pl_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        loss_playerxy = Gaussian2D_loss(output_pl_xy, gold_pl_xy)

        gold_ol_xy = torch.cat((target_ol_x.unsqueeze(-1), target_ol_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        loss_opponentxy = Gaussian2D_loss(output_ol_xy, gold_ol_xy)

        total_instance += len(target_shot)
        def replace_value(lst):
            for i in range(len(lst)):
                if lst[i] == -9223372036854775808:
                    lst[i] = 1
                elif lst[i] == -9223372036854775807:
                    lst[i] = 2
        
        replace_value(target_height)
        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc + loss_playerxy + loss_opponentxy)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        total_playerloc_loss += loss_playerloc.item()
        total_opponentloc_loss += loss_opponentloc.item()
        total_playerxy += loss_playerxy.item()
        total_opponentxy += loss_opponentxy.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)
    total_playerxy = round(total_playerxy / total_instance, 4)
    total_opponentxy = round(total_opponentxy / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    record_loss['playerloc'].append(total_playerloc_loss)
    record_loss['opponentloc'].append(total_opponentloc_loss)
    record_loss['playerxy'].append(total_playerxy)
    record_loss['opponentxy'].append(total_opponentxy)

    return total_loss
def shotGen_train_epoch_model3_small(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, record_loss, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    # print(encode_length)

    encoder.eval(), decoder.eval()
    total_loss, total_shot_loss, total_area_loss = 0, 0, 0
    total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
    total_instance = 0

    for loader_idx, item in enumerate(data_loader):
        batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
        batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
        batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
        batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
        batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
        batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
        seq_len, seq_sets = item[18].to(device), item[19].to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # print(batch_target_shot_type.shape)

        # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
        # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
        input_shot = batch_input_shot_type[:, :encode_length]
        input_x = batch_input_landing_x[:, :encode_length]
        input_y = batch_input_landing_y[:, :encode_length]
        input_player = batch_input_player[:, :encode_length]
        # input_height = batch_input_landing_height[:, :encode_length]
        input_aroundhead = batch_input_aroundhead[:, :encode_length]
        input_backhand = batch_input_backhand[:, :encode_length]
        # input_player_location_area = batch_input_player_location_area[:, :encode_length]
        # input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

        # print('input_player : ', input_player[0].shape)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand)
        # encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
        # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
        input_shot = batch_input_shot_type[:, encode_length:]
        input_x = batch_input_landing_x[:, encode_length:]
        input_y = batch_input_landing_y[:, encode_length:]
        input_player = batch_input_player[:, encode_length:]
        # input_height = batch_input_landing_height[:, encode_length:]
        input_aroundhead = batch_input_aroundhead[:, encode_length:]
        input_backhand = batch_input_backhand[:, encode_length:]
        # input_player_location_area = batch_input_player_location_area[:, encode_length:]
        # input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

        # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
        # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
        target_shot = batch_target_shot_type[:, encode_length:]
        target_x = batch_target_landing_x[:, encode_length:]
        target_y = batch_target_landing_y[:, encode_length:]
        target_player = batch_target_player[:, encode_length:]
        # target_height = batch_target_landing_height[:, encode_length:]
        target_aroundhead = batch_target_aroundhead[:, encode_length:]
        target_backhand = batch_target_backhand[:, encode_length:]
        # target_player_location_area = batch_target_player_location_area[:, encode_length:]
        # target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

        # print('target player : ', target_player[0].shape)
        output_xy, output_shot_logits, output_aroundhead_logits, output_backhand_logits = decoder(input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand, encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
        
        # output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
        #                                                                 input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
        
        pad_mask = (input_shot!=PAD)
        output_shot_logits = output_shot_logits[pad_mask]
        # print(target_shot)
        target_shot = target_shot[pad_mask]
        output_xy = output_xy[pad_mask]
        target_x = target_x[pad_mask]
        target_y = target_y[pad_mask]

        # output_height_logits = output_height_logits[pad_mask]
        # target_height = target_height[pad_mask]
        output_aroundhead_logits = output_aroundhead_logits[pad_mask]
        target_aroundhead = target_aroundhead[pad_mask]
        output_backhand_logits = output_backhand_logits[pad_mask]
        target_backhand = target_backhand[pad_mask]
        # output_playerloc_logits = output_playerloc_logits[pad_mask]
        # target_player_location_area = target_player_location_area[pad_mask]
        # output_opponentloc_logits = output_opponentloc_logits[pad_mask]
        # target_opponent_location_area = target_opponent_location_area[pad_mask]

        # _, output_height = torch.topk(output_height_logits, 1)
        # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
        # _, output_backhand = torch.topk(output_backhand_logits, 1)
        # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
        # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

        gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
        total_instance += len(target_shot)

        loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
        loss_area = Gaussian2D_loss(output_xy, gold_xy)

        # def replace_value(lst):
        #     for i in range(len(lst)):
        #         if lst[i] == -9223372036854775808:
        #             lst[i] = 1
        #         elif lst[i] == -9223372036854775807:
        #             lst[i] = 2
        
        # replace_value(target_height)
        # loss_height = criterion['entropy'](output_height_logits, target_height)
        loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
        loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
        # loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
        # loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

        loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_aroundhead + loss_backhand)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_shot_loss += loss_shot.item()
        total_area_loss += loss_area.item()
        # total_height_loss += loss_height.item()
        total_aroundhead_loss += loss_aroundhead.item()
        total_backhand_loss += loss_backhand.item()
        # total_playerloc_loss += loss_playerloc.item()
        # total_opponentloc_loss += loss_opponentloc.item()

    total_loss = round(total_loss / total_instance, 4)
    total_shot_loss = round(total_shot_loss / total_instance, 4)
    total_area_loss = round(total_area_loss / total_instance, 4)
    # total_height_loss = round(total_height_loss / total_instance, 4)
    total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
    total_backhand_loss = round(total_backhand_loss / total_instance, 4)
    # total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
    # total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

    record_loss['total'].append(total_loss)
    record_loss['shot'].append(total_shot_loss)
    record_loss['area'].append(total_area_loss)

    # record_loss['height'].append(total_height_loss)
    record_loss['aroundhead'].append(total_aroundhead_loss)
    record_loss['backhand'].append(total_backhand_loss)
    # record_loss['playerloc'].append(total_playerloc_loss)
    # record_loss['opponentloc'].append(total_opponentloc_loss)

    # config['total_loss'] = total_loss
    # config['total_shot_loss'] = total_shot_loss
    # config['total_area_loss'] = total_area_loss
    # config['total_height_loss'] = total_height_loss
    # config['total_aroundhead_loss'] = total_aroundhead_loss
    # config['total_backhand_loss'] = total_backhand_loss
    # config['total_playerloc_loss'] = total_playerloc_loss
    # config['total_opponentloc_loss'] = total_opponentloc_loss
    # save(encoder, decoder, config)

    return total_loss
def shotGen_trainer(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    record_loss = {
        'total': [],
        'shot': [],
        'area': [],
        'height': [],
        'aroundhead': [],
        'backhand': [],
        'playerloc': [],
        'opponentloc': []
    }

    for epoch in tqdm(range(config['epochs']), desc='Epoch: '):
        encoder.train(), decoder.train()
        total_loss, total_shot_loss, total_area_loss = 0, 0, 0
        total_height_loss, total_aroundhead_loss, total_backhand_loss, total_playerloc_loss, total_opponentloc_loss = 0, 0, 0, 0, 0
        total_instance = 0

        for loader_idx, item in enumerate(data_loader):
            batch_input_shot_type, batch_input_landing_x, batch_input_landing_y, batch_input_landing_height = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device)
            batch_input_aroundhead, batch_input_backhand, batch_input_player = item[4].to(device), item[5].to(device), item[6].to(device), 
            batch_input_player_location_area, batch_input_opponent_location_area = item[7].to(device), item[8].to(device)
            batch_target_shot_type, batch_target_landing_x, batch_target_landing_y, batch_target_landing_height = item[9].to(device), item[10].to(device), item[11].to(device), item[12].to(device)
            batch_target_aroundhead, batch_target_backhand, batch_target_player = item[13].to(device), item[14].to(device), item[15].to(device)
            batch_target_player_location_area, batch_target_opponent_location_area = item[16].to(device), item[17].to(device)
            seq_len, seq_sets = item[18].to(device), item[19].to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # input_roundscore_A = batch_input_roundscore_A[:, :encode_length]
            # input_roundscore_B = batch_input_roundscore_B[:, :encode_length]
            input_shot = batch_input_shot_type[:, :encode_length]
            input_x = batch_input_landing_x[:, :encode_length]
            input_y = batch_input_landing_y[:, :encode_length]
            input_player = batch_input_player[:, :encode_length]
            input_height = batch_input_landing_height[:, :encode_length]
            input_aroundhead = batch_input_aroundhead[:, :encode_length]
            input_backhand = batch_input_backhand[:, :encode_length]
            input_player_location_area = batch_input_player_location_area[:, :encode_length]
            input_opponent_location_area = batch_input_opponent_location_area[:, :encode_length]

            # print('input_player : ', input_player[0].shape)
            encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

            # input_roundscore_A = batch_input_roundscore_A[:, encode_length:]
            # input_roundscore_B = batch_input_roundscore_B[:, encode_length:]                                                                    
            input_shot = batch_input_shot_type[:, encode_length:]
            input_x = batch_input_landing_x[:, encode_length:]
            input_y = batch_input_landing_y[:, encode_length:]
            input_player = batch_input_player[:, encode_length:]
            input_height = batch_input_landing_height[:, encode_length:]
            input_aroundhead = batch_input_aroundhead[:, encode_length:]
            input_backhand = batch_input_backhand[:, encode_length:]
            input_player_location_area = batch_input_player_location_area[:, encode_length:]
            input_opponent_location_area = batch_input_opponent_location_area[:, encode_length:]

            # target_roundscore_A = batch_target_roundscore_A[:, encode_length:]
            # target_roundscore_B = batch_target_roundscore_B[:, encode_length:]
            target_shot = batch_target_shot_type[:, encode_length:]
            target_x = batch_target_landing_x[:, encode_length:]
            target_y = batch_target_landing_y[:, encode_length:]
            target_player = batch_target_player[:, encode_length:]
            target_height = batch_target_landing_height[:, encode_length:]
            target_aroundhead = batch_target_aroundhead[:, encode_length:]
            target_backhand = batch_target_backhand[:, encode_length:]
            target_player_location_area = batch_target_player_location_area[:, encode_length:]
            target_opponent_location_area = batch_target_opponent_location_area[:, encode_length:]

            # print('target player : ', target_player[0].shape)
            output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                            input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
            
            pad_mask = (input_shot!=PAD)
            output_shot_logits = output_shot_logits[pad_mask]
            target_shot = target_shot[pad_mask]
            output_xy = output_xy[pad_mask]
            target_x = target_x[pad_mask]
            target_y = target_y[pad_mask]

            output_height_logits = output_height_logits[pad_mask]
            target_height = target_height[pad_mask]
            output_aroundhead_logits = output_aroundhead_logits[pad_mask]
            target_aroundhead = target_aroundhead[pad_mask]
            output_backhand_logits = output_backhand_logits[pad_mask]
            target_backhand = target_backhand[pad_mask]
            output_playerloc_logits = output_playerloc_logits[pad_mask]
            target_player_location_area = target_player_location_area[pad_mask]
            output_opponentloc_logits = output_opponentloc_logits[pad_mask]
            target_opponent_location_area = target_opponent_location_area[pad_mask]

            _, output_height = torch.topk(output_height_logits, 1)
            # _, output_aroundhead = torch.topk(output_aroundhead_logits, 1)
            # _, output_backhand = torch.topk(output_backhand_logits, 1)
            # _, output_playerloc = torch.topk(output_playerloc_logits, 1)
            # _, output_opponentloc = torch.topk(output_opponentloc_logits, 1)

            gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)
            total_instance += len(target_shot)

            loss_shot = criterion['entropy'](output_shot_logits, target_shot.long()) # target_shot에서 target_shot.long()로 바꿈
            loss_area = Gaussian2D_loss(output_xy, gold_xy)

            def replace_value(lst):
                for i in range(len(lst)):
                    if lst[i] == -9223372036854775808:
                        lst[i] = 1
                    elif lst[i] == -9223372036854775807:
                        lst[i] = 2
            
            replace_value(target_height)
            loss_height = criterion['entropy'](output_height_logits, target_height)
            loss_aroundhead = criterion['entropy'](output_aroundhead_logits, target_aroundhead.long())
            loss_backhand = criterion['entropy'](output_backhand_logits, target_backhand.long())
            loss_playerloc = criterion['entropy'](output_playerloc_logits, target_player_location_area.long())
            loss_opponentloc = criterion['entropy'](output_opponentloc_logits, target_opponent_location_area.long())

            loss = config['alpha'] * (loss_shot + loss_area) + (1 - config['alpha']) * (loss_height + loss_aroundhead + loss_backhand + loss_playerloc + loss_opponentloc)
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            total_shot_loss += loss_shot.item()
            total_area_loss += loss_area.item()
            total_height_loss += loss_height.item()
            total_aroundhead_loss += loss_aroundhead.item()
            total_backhand_loss += loss_backhand.item()
            total_playerloc_loss += loss_playerloc.item()
            total_opponentloc_loss += loss_opponentloc.item()

        total_loss = round(total_loss / total_instance, 4)
        total_shot_loss = round(total_shot_loss / total_instance, 4)
        total_area_loss = round(total_area_loss / total_instance, 4)
        total_height_loss = round(total_height_loss / total_instance, 4)
        total_aroundhead_loss = round(total_aroundhead_loss / total_instance, 4)
        total_backhand_loss = round(total_backhand_loss / total_instance, 4)
        total_playerloc_loss = round(total_playerloc_loss / total_instance, 4)
        total_opponentloc_loss = round(total_opponentloc_loss / total_instance, 4)

        record_loss['total'].append(total_loss)
        record_loss['shot'].append(total_shot_loss)
        record_loss['area'].append(total_area_loss)

        record_loss['height'].append(total_height_loss)
        record_loss['aroundhead'].append(total_aroundhead_loss)
        record_loss['backhand'].append(total_backhand_loss)
        record_loss['playerloc'].append(total_playerloc_loss)
        record_loss['opponentloc'].append(total_opponentloc_loss)

    config['total_loss'] = total_loss
    config['total_shot_loss'] = total_shot_loss
    config['total_area_loss'] = total_area_loss
    config['total_height_loss'] = total_height_loss
    config['total_aroundhead_loss'] = total_aroundhead_loss
    config['total_backhand_loss'] = total_backhand_loss
    config['total_playerloc_loss'] = total_playerloc_loss
    config['total_opponentloc_loss'] = total_opponentloc_loss
    save(encoder, decoder, config)

    return record_loss

def shotgen_generator(given_seq, encoder, decoder, config, samples, device):
    encode_length = config['encode_length'] - 1
    encoder.eval(), decoder.eval()
    generated_shot_logits, generated_area_coordinates = [], []

    with torch.no_grad():
        # encoding stage
        input_shot = given_seq['given_shot'][:encode_length].unsqueeze(0)
        input_x = given_seq['given_x'][:encode_length].unsqueeze(0)
        input_y = given_seq['given_y'][:encode_length].unsqueeze(0)
        input_player = given_seq['given_player'][:encode_length].unsqueeze(0)

        input_height = given_seq['given_ball_height'][:encode_length].unsqueeze(0).to(torch.long)
        input_aroundhead = given_seq['given_ball_aroundhead'][:encode_length].unsqueeze(0).to(torch.long)
        input_backhand = given_seq['given_ball_backhand'][:encode_length].unsqueeze(0).to(torch.long)
        input_player_location_area = given_seq['given_ball_player_area'][:encode_length].unsqueeze(0).to(torch.long)
        input_opponent_location_area = given_seq['given_ball_opponent_area'][:encode_length].unsqueeze(0).to(torch.long)
        
        encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player,
                                                                            input_height,input_aroundhead, input_backhand,
                                                                            input_player_location_area, input_opponent_location_area)

        for sample_id in range(samples):
            # print('sample_id', sample_id)
            current_generated_shot, current_generated_area = [], []
            total_instance = len(given_seq['given_shot']) - len(given_seq['given_shot'][:encode_length])
            for seq_idx in range(encode_length, given_seq['rally_length']-1):
                # print('seq_idx', seq_idx)
                if seq_idx == encode_length:
                    input_shot = given_seq['given_shot'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_x = given_seq['given_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_y = given_seq['given_y'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_player = given_seq['given_player'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    # input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).to(torch.long)
                else:
                    # use its own predictions as the next input
                    input_shot = torch.cat((input_shot, prev_shot), dim=-1)
                    input_x = torch.cat((input_x, prev_x), dim=-1)
                    input_y = torch.cat((input_y, prev_y), dim=-1)
                    input_player = torch.cat((input_player, prev_player), dim=-1)
                    input_height = torch.cat((input_height, prev_height), dim=-1)
                    input_aroundhead = torch.cat((input_aroundhead, prev_aroundhead), dim=-1)
                    input_backhand = torch.cat((input_backhand, prev_backhand), dim=-1)
                    input_player_location_area = torch.cat((input_player_location_area, prev_player_location_area), dim=-1)
                    input_opponent_location_area = torch.cat((input_opponent_location_area, prev_opponent_location_area), dim=-1)
                    
                target_player = given_seq['target_player'][seq_idx-encode_length].unsqueeze(0).unsqueeze(0)

                output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                            input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player)
            
                
                # sample area coordinates
                sx = torch.exp(output_xy[:, -1, 2]) #sx
                sy = torch.exp(output_xy[:, -1, 3]) #sy
                corr = torch.tanh(output_xy[:, -1, 4]) #corr
                
                cov = torch.zeros(2, 2).cuda(output_xy.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = output_xy[:, -1, 0:2]
                
                mvnormal = torchdist.MultivariateNormal(mean, cov)
                output_xy = mvnormal.sample().unsqueeze(0)

                # sampling
                shot_prob = F.softmax(output_shot_logits, dim=-1)
                output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                height_prob = F.softmax(output_height_logits, dim=-1)
                output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                aroundhead_prob = F.softmax(output_aroundhead_logits, dim=-1)
                output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                backhand_prob = F.softmax(output_backhand_logits, dim=-1)
                output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                playerloc_prob = F.softmax(output_playerloc_logits, dim=-1)
                output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                opponentloc_prob = F.softmax(output_opponentloc_logits, dim=-1)
                output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                while output_shot[0, -1, 0] == 0:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_height[0, -1, 0] == 0:
                    output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_aroundhead[0, -1, 0] == 0:
                    output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_backhand[0, -1, 0] == 0:
                    output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_playerloc[0, -1, 0] == 0:
                    output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_opponentloc[0, -1, 0] == 0:
                    output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                prev_shot = output_shot[:, -1, :]
                prev_height = output_height[:, -1, :]
                prev_aroundhead = output_aroundhead[:, -1, :]
                prev_backhand = output_backhand[:, -1, :]
                prev_player_location_area = output_playerloc[:, -1, :]
                prev_opponent_location_area = output_opponentloc[:, -1, :]
                prev_x = output_xy[:, -1, 0].unsqueeze(1)
                prev_y = output_xy[:, -1, 1].unsqueeze(1)
                prev_player = target_player.clone()

                # transform to original format
                ori_shot = config['uniques_type'][prev_shot.item()-1]
                ori_x = prev_x.item()
                ori_y = prev_y.item()

                current_generated_shot.append(shot_prob[0][-1][1:].cpu().tolist()) # 0 is pad
                current_generated_area.append((ori_x, ori_y))

            generated_shot_logits.append(current_generated_shot), generated_area_coordinates.append(current_generated_area)

    return generated_shot_logits, generated_area_coordinates
def shotgen_generator_model3(given_seq, encoder, decoder, config, samples, device):
    encode_length = config['encode_length'] - 1
    encoder.eval(), decoder.eval()
    generated_shot_logits, generated_area_coordinates = [], []

    with torch.no_grad():
        # encoding stage
        input_shot = given_seq['given_shot'][:encode_length].unsqueeze(0)
        input_x = given_seq['given_x'][:encode_length].unsqueeze(0)
        input_y = given_seq['given_y'][:encode_length].unsqueeze(0)
        input_player = given_seq['given_player'][:encode_length].unsqueeze(0)

        input_height = given_seq['given_ball_height'][:encode_length].unsqueeze(0).to(torch.long)
        input_aroundhead = given_seq['given_ball_aroundhead'][:encode_length].unsqueeze(0).to(torch.long)
        input_backhand = given_seq['given_ball_backhand'][:encode_length].unsqueeze(0).to(torch.long)
        input_player_location_area = given_seq['given_ball_player_area'][:encode_length].unsqueeze(0).to(torch.long)
        input_opponent_location_area = given_seq['given_ball_opponent_area'][:encode_length].unsqueeze(0).to(torch.long)
        
        # print("input_shot", input_shot)
        # print("input_x", input_x)
        # print("input y", input_y)
        # print("input_player", input_player)
        # print("input_height", input_height)
        # print("input aroundhead", input_aroundhead)
        # print("input_backhand", input_backhand)
        # print("input_player_location_Area", input_player_location_area)
        # print("input_opponent_location_area", input_opponent_location_area)
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player,
                                                                            input_height,input_aroundhead, input_backhand,
                                                                            input_player_location_area, input_opponent_location_area)

        for sample_id in range(samples):
            # print('sample_id', sample_id)
            current_generated_shot, current_generated_area = [], []
            total_instance = len(given_seq['given_shot']) - len(given_seq['given_shot'][:encode_length])
            for seq_idx in range(encode_length, given_seq['rally_length']-1):
                # print('seq_idx', seq_idx)
                if seq_idx == encode_length:
                    input_shot = given_seq['given_shot'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_x = given_seq['given_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_y = given_seq['given_y'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_player = given_seq['given_player'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    # input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).to(torch.long)
                else:
                    # use its own predictions as the next input
                    input_shot = torch.cat((input_shot, prev_shot), dim=-1)
                    input_x = torch.cat((input_x, prev_x), dim=-1)
                    input_y = torch.cat((input_y, prev_y), dim=-1)
                    input_player = torch.cat((input_player, prev_player), dim=-1)
                    input_height = torch.cat((input_height, prev_height), dim=-1)
                    input_aroundhead = torch.cat((input_aroundhead, prev_aroundhead), dim=-1)
                    input_backhand = torch.cat((input_backhand, prev_backhand), dim=-1)
                    input_player_location_area = torch.cat((input_player_location_area, prev_player_location_area), dim=-1)
                    input_opponent_location_area = torch.cat((input_opponent_location_area, prev_opponent_location_area), dim=-1)
                
                target_player = given_seq['target_player'][seq_idx-encode_length].unsqueeze(0).unsqueeze(0)

                output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                            input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                                                                            encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
            
                
                # sample area coordinates
                sx = torch.exp(output_xy[:, -1, 2]) #sx
                sy = torch.exp(output_xy[:, -1, 3]) #sy
                corr = torch.tanh(output_xy[:, -1, 4]) #corr
                
                cov = torch.zeros(2, 2).cuda(output_xy.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = output_xy[:, -1, 0:2]
                
                mvnormal = torchdist.MultivariateNormal(mean, cov)
                output_xy = mvnormal.sample().unsqueeze(0)

                # sampling
                shot_prob = F.softmax(output_shot_logits, dim=-1)
                output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                height_prob = F.softmax(output_height_logits, dim=-1)
                output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                aroundhead_prob = F.softmax(output_aroundhead_logits, dim=-1)
                output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                backhand_prob = F.softmax(output_backhand_logits, dim=-1)
                output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                playerloc_prob = F.softmax(output_playerloc_logits, dim=-1)
                output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                opponentloc_prob = F.softmax(output_opponentloc_logits, dim=-1)
                output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                while output_shot[0, -1, 0] == 0:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_height[0, -1, 0] == 0:
                    output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_aroundhead[0, -1, 0] == 0:
                    output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_backhand[0, -1, 0] == 0:
                    output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_playerloc[0, -1, 0] == 0:
                    output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_opponentloc[0, -1, 0] == 0:
                    output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                prev_shot = output_shot[:, -1, :]
                prev_height = output_height[:, -1, :]
                prev_aroundhead = output_aroundhead[:, -1, :]
                prev_backhand = output_backhand[:, -1, :]
                prev_player_location_area = output_playerloc[:, -1, :]
                prev_opponent_location_area = output_opponentloc[:, -1, :]
                prev_x = output_xy[:, -1, 0].unsqueeze(1)
                prev_y = output_xy[:, -1, 1].unsqueeze(1)
                prev_player = target_player.clone()

                # transform to original format
                ori_shot = config['uniques_type'][prev_shot.item()-1]
                ori_x = prev_x.item()
                ori_y = prev_y.item()

                current_generated_shot.append(shot_prob[0][-1][1:].cpu().tolist()) # 0 is pad
                current_generated_area.append((ori_x, ori_y))

            generated_shot_logits.append(current_generated_shot), generated_area_coordinates.append(current_generated_area)

    return generated_shot_logits, generated_area_coordinates

def shotgen_generator_model3_small(given_seq, encoder, decoder, config, samples, device):
    encode_length = config['encode_length'] - 1
    encoder.eval(), decoder.eval()
    generated_shot_logits, generated_area_coordinates = [], []

    with torch.no_grad():
        # encoding stage
        input_shot = given_seq['given_shot'][:encode_length].unsqueeze(0)
        input_x = given_seq['given_x'][:encode_length].unsqueeze(0)
        input_y = given_seq['given_y'][:encode_length].unsqueeze(0)
        input_player = given_seq['given_player'][:encode_length].unsqueeze(0)

        # input_height = given_seq['given_ball_height'][:encode_length].unsqueeze(0).to(torch.long)
        input_aroundhead = given_seq['given_ball_aroundhead'][:encode_length].unsqueeze(0).to(torch.long)
        input_backhand = given_seq['given_ball_backhand'][:encode_length].unsqueeze(0).to(torch.long)
        # input_player_location_area = given_seq['given_ball_player_area'][:encode_length].unsqueeze(0).to(torch.long)
        # input_opponent_location_area = given_seq['given_ball_opponent_area'][:encode_length].unsqueeze(0).to(torch.long)
        
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand)

        for sample_id in range(samples):
            # print('sample_id', sample_id)
            current_generated_shot, current_generated_area = [], []
            total_instance = len(given_seq['given_shot']) - len(given_seq['given_shot'][:encode_length])
            for seq_idx in range(encode_length, given_seq['rally_length']-1):
                # print('seq_idx', seq_idx)
                if seq_idx == encode_length:
                    input_shot = given_seq['given_shot'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_x = given_seq['given_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_y = given_seq['given_y'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_player = given_seq['given_player'][seq_idx].unsqueeze(0).unsqueeze(0)
                    # input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    # input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    # input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    # input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).to(torch.long)
                else:
                    # use its own predictions as the next input
                    input_shot = torch.cat((input_shot, prev_shot), dim=-1)
                    input_x = torch.cat((input_x, prev_x), dim=-1)
                    input_y = torch.cat((input_y, prev_y), dim=-1)
                    input_player = torch.cat((input_player, prev_player), dim=-1)
                    # input_height = torch.cat((input_height, prev_height), dim=-1)
                    input_aroundhead = torch.cat((input_aroundhead, prev_aroundhead), dim=-1)
                    input_backhand = torch.cat((input_backhand, prev_backhand), dim=-1)
                    # input_player_location_area = torch.cat((input_player_location_area, prev_player_location_area), dim=-1)
                    # input_opponent_location_area = torch.cat((input_opponent_location_area, prev_opponent_location_area), dim=-1)
                    
                target_player = given_seq['target_player'][seq_idx-encode_length].unsqueeze(0).unsqueeze(0)

                output_xy, output_shot_logits, output_aroundhead_logits, output_backhand_logits = decoder(input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand,
                                                                            encode_local_output_area, encode_local_output_shot, 
                                                                            encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
            
                
                # sample area coordinates
                sx = torch.exp(output_xy[:, -1, 2]) #sx
                sy = torch.exp(output_xy[:, -1, 3]) #sy
                corr = torch.tanh(output_xy[:, -1, 4]) #corr
                
                cov = torch.zeros(2, 2).cuda(output_xy.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = output_xy[:, -1, 0:2]
                
                mvnormal = torchdist.MultivariateNormal(mean, cov)
                output_xy = mvnormal.sample().unsqueeze(0)

                # sampling
                shot_prob = F.softmax(output_shot_logits, dim=-1)
                output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                # height_prob = F.softmax(output_height_logits, dim=-1)
                # output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                aroundhead_prob = F.softmax(output_aroundhead_logits, dim=-1)
                output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                backhand_prob = F.softmax(output_backhand_logits, dim=-1)
                output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                # playerloc_prob = F.softmax(output_playerloc_logits, dim=-1)
                # output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                # opponentloc_prob = F.softmax(output_opponentloc_logits, dim=-1)
                # output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                while output_shot[0, -1, 0] == 0:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                # while output_height[0, -1, 0] == 0:
                #     output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_aroundhead[0, -1, 0] == 0:
                    output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_backhand[0, -1, 0] == 0:
                    output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                # while output_playerloc[0, -1, 0] == 0:
                #     output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                # while output_opponentloc[0, -1, 0] == 0:
                #     output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                prev_shot = output_shot[:, -1, :]
                # prev_height = output_height[:, -1, :]
                prev_aroundhead = output_aroundhead[:, -1, :]
                prev_backhand = output_backhand[:, -1, :]
                # prev_player_location_area = output_playerloc[:, -1, :]
                # prev_opponent_location_area = output_opponentloc[:, -1, :]
                prev_x = output_xy[:, -1, 0].unsqueeze(1)
                prev_y = output_xy[:, -1, 1].unsqueeze(1)
                prev_player = target_player.clone()

                # transform to original format
                ori_shot = config['uniques_type'][prev_shot.item()-1]
                ori_x = prev_x.item()
                ori_y = prev_y.item()

                current_generated_shot.append(shot_prob[0][-1][1:].cpu().tolist()) # 0 is pad
                current_generated_area.append((ori_x, ori_y))

            generated_shot_logits.append(current_generated_shot), generated_area_coordinates.append(current_generated_area)

    return generated_shot_logits, generated_area_coordinates
def shotgen_generator_simple(given_seq, encoder, decoder, config, samples, device):
    encode_length = config['encode_length'] - 1
    encoder.eval(), decoder.eval()
    generated_shot_logits, generated_area_coordinates = [], []

    with torch.no_grad():
        # encoding stage
        input_shot = given_seq['given_shot'][:encode_length].unsqueeze(0)
        input_x = given_seq['given_x'][:encode_length].unsqueeze(0)
        input_y = given_seq['given_y'][:encode_length].unsqueeze(0)
        input_player = given_seq['given_player'][:encode_length].unsqueeze(0)

        
        encode_local_output_area, encode_local_output_shot, encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot = encoder(input_shot, input_x, input_y, input_player)

        for sample_id in range(samples):
            # print('sample_id', sample_id)
            current_generated_shot, current_generated_area = [], []
            total_instance = len(given_seq['given_shot']) - len(given_seq['given_shot'][:encode_length])
            for seq_idx in range(encode_length, given_seq['rally_length']-1):
                # print('seq_idx', seq_idx)
                if seq_idx == encode_length:
                    input_shot = given_seq['given_shot'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_x = given_seq['given_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_y = given_seq['given_y'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_player = given_seq['given_player'][seq_idx].unsqueeze(0).unsqueeze(0)

                else:
                    # use its own predictions as the next input
                    input_shot = torch.cat((input_shot, prev_shot), dim=-1)
                    input_x = torch.cat((input_x, prev_x), dim=-1)
                    input_y = torch.cat((input_y, prev_y), dim=-1)
                    input_player = torch.cat((input_player, prev_player), dim=-1)
            
                target_player = given_seq['target_player'][seq_idx-encode_length].unsqueeze(0).unsqueeze(0)

                output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, encode_local_output_area, encode_local_output_shot, 
                                                                            encode_global_A_area, encode_global_A_shot, encode_global_B_area, encode_global_B_shot, target_player)
            
                
                # sample area coordinates
                sx = torch.exp(output_xy[:, -1, 2]) #sx
                sy = torch.exp(output_xy[:, -1, 3]) #sy
                corr = torch.tanh(output_xy[:, -1, 4]) #corr
                
                cov = torch.zeros(2, 2).cuda(output_xy.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = output_xy[:, -1, 0:2]
                
                mvnormal = torchdist.MultivariateNormal(mean, cov)
                output_xy = mvnormal.sample().unsqueeze(0)

                # sampling
                shot_prob = F.softmax(output_shot_logits, dim=-1)
                output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
  
                while output_shot[0, -1, 0] == 0:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
    
                prev_shot = output_shot[:, -1, :]
   
                prev_x = output_xy[:, -1, 0].unsqueeze(1)
                prev_y = output_xy[:, -1, 1].unsqueeze(1)
                prev_player = target_player.clone()

                # transform to original format
                ori_shot = config['uniques_type'][prev_shot.item()-1]
                ori_x = prev_x.item()
                ori_y = prev_y.item()

                current_generated_shot.append(shot_prob[0][-1][1:].cpu().tolist()) # 0 is pad
                current_generated_area.append((ori_x, ori_y))

            generated_shot_logits.append(current_generated_shot), generated_area_coordinates.append(current_generated_area)

    return generated_shot_logits, generated_area_coordinates
def shotgen_generator_Big(given_seq, encoder, decoder, config, samples, device):
    encode_length = config['encode_length'] - 1
    encoder.eval(), decoder.eval()
    generated_shot_logits, generated_area_coordinates = [], []

    with torch.no_grad():
        # encoding stage
        input_shot = given_seq['given_shot'][:encode_length].unsqueeze(0)
        input_x = given_seq['given_x'][:encode_length].unsqueeze(0)
        input_y = given_seq['given_y'][:encode_length].unsqueeze(0)
        input_player = given_seq['given_player'][:encode_length].unsqueeze(0)

        input_height = given_seq['given_ball_height'][:encode_length].unsqueeze(0).to(torch.long)
        input_aroundhead = given_seq['given_ball_aroundhead'][:encode_length].unsqueeze(0).to(torch.long)
        input_backhand = given_seq['given_ball_backhand'][:encode_length].unsqueeze(0).to(torch.long)
        input_player_location_area = given_seq['given_ball_player_area'][:encode_length].unsqueeze(0).to(torch.long)
        input_opponent_location_area = given_seq['given_ball_opponent_area'][:encode_length].unsqueeze(0).to(torch.long)
        
        encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, input_player_location_area, input_opponent_location_area)

        for sample_id in range(samples):
            # print('sample_id', sample_id)
            current_generated_shot, current_generated_area = [], []
            total_instance = len(given_seq['given_shot']) - len(given_seq['given_shot'][:encode_length])
            for seq_idx in range(encode_length, given_seq['rally_length']-1):
                # print('seq_idx', seq_idx)
                if seq_idx == encode_length:
                    input_shot = given_seq['given_shot'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_x = given_seq['given_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_y = given_seq['given_y'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_player = given_seq['given_player'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    # input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).to(torch.long)
                    # input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).to(torch.long)
                else:
                    # use its own predictions as the next input
                    input_shot = torch.cat((input_shot, prev_shot), dim=-1)
                    input_x = torch.cat((input_x, prev_x), dim=-1)
                    input_y = torch.cat((input_y, prev_y), dim=-1)
                    input_player = torch.cat((input_player, prev_player), dim=-1)
                    input_height = torch.cat((input_height, prev_height), dim=-1)
                    input_aroundhead = torch.cat((input_aroundhead, prev_aroundhead), dim=-1)
                    input_backhand = torch.cat((input_backhand, prev_backhand), dim=-1)
                    input_player_location_area = torch.cat((input_player_location_area, prev_player_location_area), dim=-1)
                    input_opponent_location_area = torch.cat((input_opponent_location_area, prev_opponent_location_area), dim=-1)
                    
                target_player = given_seq['target_player'][seq_idx-encode_length].unsqueeze(0).unsqueeze(0)

                output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, target_player)

                
                # sample area coordinates
                sx = torch.exp(output_xy[:, -1, 2]) #sx
                sy = torch.exp(output_xy[:, -1, 3]) #sy
                corr = torch.tanh(output_xy[:, -1, 4]) #corr
                
                cov = torch.zeros(2, 2).cuda(output_xy.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = output_xy[:, -1, 0:2]
                
                mvnormal = torchdist.MultivariateNormal(mean, cov)
                output_xy = mvnormal.sample().unsqueeze(0)

                # sampling
                shot_prob = F.softmax(output_shot_logits, dim=-1)
                output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                height_prob = F.softmax(output_height_logits, dim=-1)
                output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                aroundhead_prob = F.softmax(output_aroundhead_logits, dim=-1)
                output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                backhand_prob = F.softmax(output_backhand_logits, dim=-1)
                output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                playerloc_prob = F.softmax(output_playerloc_logits, dim=-1)
                output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                opponentloc_prob = F.softmax(output_opponentloc_logits, dim=-1)
                output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                while output_shot[0, -1, 0] == 0:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_height[0, -1, 0] == 0:
                    output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_aroundhead[0, -1, 0] == 0:
                    output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_backhand[0, -1, 0] == 0:
                    output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_playerloc[0, -1, 0] == 0:
                    output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_opponentloc[0, -1, 0] == 0:
                    output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                prev_shot = output_shot[:, -1, :]
                prev_height = output_height[:, -1, :]
                prev_aroundhead = output_aroundhead[:, -1, :]
                prev_backhand = output_backhand[:, -1, :]
                prev_player_location_area = output_playerloc[:, -1, :]
                prev_opponent_location_area = output_opponentloc[:, -1, :]
                prev_x = output_xy[:, -1, 0].unsqueeze(1)
                prev_y = output_xy[:, -1, 1].unsqueeze(1)
                prev_player = target_player.clone()

                # transform to original format
                ori_shot = config['uniques_type'][prev_shot.item()-1]
                ori_x = prev_x.item()
                ori_y = prev_y.item()

                current_generated_shot.append(shot_prob[0][-1][1:].cpu().tolist()) # 0 is pad
                current_generated_area.append((ori_x, ori_y))

            generated_shot_logits.append(current_generated_shot), generated_area_coordinates.append(current_generated_area)

    return generated_shot_logits, generated_area_coordinates
def shotgen_generator_Big2(given_seq, encoder, decoder, config, samples, device):
    encode_length = config['encode_length'] - 1
    encoder.eval(), decoder.eval()
    generated_shot_logits, generated_area_coordinates = [], []

    with torch.no_grad():
        # encoding stage
        input_shot = given_seq['given_shot'][:encode_length].unsqueeze(0)
        input_x = given_seq['given_x'][:encode_length].unsqueeze(0)
        input_y = given_seq['given_y'][:encode_length].unsqueeze(0)
        input_player_location_x = given_seq['given_player_location_x'][:encode_length].unsqueeze(0)
        input_player_location_y = given_seq['given_player_location_y'][:encode_length].unsqueeze(0)
        input_opponent_location_x = given_seq['given_opponent_location_x'][:encode_length].unsqueeze(0)
        input_opponent_location_y = given_seq['given_opponent_location_y'][:encode_length].unsqueeze(0)
        input_player = given_seq['given_player'][:encode_length].unsqueeze(0)

        input_height = given_seq['given_ball_height'][:encode_length].unsqueeze(0).to(torch.long)
        input_aroundhead = given_seq['given_ball_aroundhead'][:encode_length].unsqueeze(0).to(torch.long)
        input_backhand = given_seq['given_ball_backhand'][:encode_length].unsqueeze(0).to(torch.long)
        input_player_location_area = given_seq['given_ball_player_area'][:encode_length].unsqueeze(0).to(torch.long)
        input_opponent_location_area = given_seq['given_ball_opponent_area'][:encode_length].unsqueeze(0).to(torch.long)
        
        encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B = encoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                         input_player_location_area, input_opponent_location_area,
                                                                                                                                                                                                                                                                                                                                                                                                                                                         input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y)

        for sample_id in range(samples):
            # print('sample_id', sample_id)
            current_generated_shot, current_generated_area = [], []
            total_instance = len(given_seq['given_shot']) - len(given_seq['given_shot'][:encode_length])
            for seq_idx in range(encode_length, given_seq['rally_length']-1):
                # print('seq_idx', seq_idx)
                if seq_idx == encode_length:
                    input_shot = given_seq['given_shot'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_x = given_seq['given_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_y = given_seq['given_y'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_player = given_seq['given_player'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_height = given_seq['given_ball_height'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_aroundhead = given_seq['given_ball_aroundhead'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_backhand = given_seq['given_ball_backhand'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_player_location_area = given_seq['given_ball_player_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_opponent_location_area = given_seq['given_ball_opponent_area'][seq_idx].unsqueeze(0).unsqueeze(0).to(torch.long)
                    input_player_location_x = given_seq['given_player_location_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_player_location_y = given_seq['given_player_location_y'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_opponent_location_x = given_seq['given_opponent_location_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_opponent_location_y = given_seq['given_opponent_location_y'][seq_idx].unsqueeze(0).unsqueeze(0)

                else:
                    # use its own predictions as the next input
                    input_shot = torch.cat((input_shot, prev_shot), dim=-1)
                    input_x = torch.cat((input_x, prev_x), dim=-1)
                    input_y = torch.cat((input_y, prev_y), dim=-1)
                    input_player = torch.cat((input_player, prev_player), dim=-1)
                    input_height = torch.cat((input_height, prev_height), dim=-1)
                    input_aroundhead = torch.cat((input_aroundhead, prev_aroundhead), dim=-1)
                    input_backhand = torch.cat((input_backhand, prev_backhand), dim=-1)
                    input_player_location_area = torch.cat((input_player_location_area, prev_player_location_area), dim=-1)
                    input_opponent_location_area = torch.cat((input_opponent_location_area, prev_opponent_location_area), dim=-1)
                    input_player_location_x = torch.cat((input_player_location_x, prev_pl_x), dim=-1)
                    input_player_location_y = torch.cat((input_player_location_y, prev_pl_y), dim=-1)
                    input_opponent_location_x = torch.cat((input_opponent_location_x, prev_ol_x), dim=-1)
                    input_opponent_location_y = torch.cat((input_opponent_location_y, prev_ol_y), dim=-1)
                    
                target_player = given_seq['target_player'][seq_idx-encode_length].unsqueeze(0).unsqueeze(0)

                output_xy, output_shot_logits, output_height_logits, output_aroundhead_logits, output_backhand_logits, output_playerloc_logits, output_opponentloc_logits, output_pl_xy, output_ol_xy = decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                                                                    input_player_location_area, input_opponent_location_area, input_player_location_x, input_player_location_y,
                                                                    encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol,
                                                                    encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A,
                                                                    encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B, target_player)
    
    
                # sample area coordinates
                sx = torch.exp(output_xy[:, -1, 2]) #sx
                sy = torch.exp(output_xy[:, -1, 3]) #sy
                corr = torch.tanh(output_xy[:, -1, 4]) #corr

                sx_pl = torch.exp(output_pl_xy[:, -1, 2]) #sx
                sy_pl = torch.exp(output_pl_xy[:, -1, 3]) #sy
                corr_pl = torch.tanh(output_pl_xy[:, -1, 4]) #corr

                sx_ol = torch.exp(output_ol_xy[:, -1, 2]) #sx
                sy_ol = torch.exp(output_ol_xy[:, -1, 3]) #sy
                corr_ol = torch.tanh(output_ol_xy[:, -1, 4]) #corr
                
                cov = torch.zeros(2, 2).cuda(output_xy.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = output_xy[:, -1, 0:2]

                cov_pl = torch.zeros(2, 2).cuda(output_pl_xy.device)
                cov_pl[0, 0]= sx_pl * sx_pl
                cov_pl[0, 1]= corr_pl * sx_pl * sy_pl
                cov_pl[1, 0]= corr_pl * sx_pl * sy_pl
                cov_pl[1, 1]= sy_pl * sy_pl
                mean_pl = output_pl_xy[:, -1, 0:2]

                cov_ol = torch.zeros(2, 2).cuda(output_ol_xy.device)
                cov_ol[0, 0]= sx_ol * sx_ol
                cov_ol[0, 1]= corr_ol * sx_ol * sy_ol
                cov_ol[1, 0]= corr_ol * sx_ol * sy_ol
                cov_ol[1, 1]= sy_ol * sy_ol
                mean_ol = output_ol_xy[:, -1, 0:2]
                
                mvnormal = torchdist.MultivariateNormal(mean, cov)
                output_xy = mvnormal.sample().unsqueeze(0)

                mvnormal_pl = torchdist.MultivariateNormal(mean_pl, cov_pl)
                output_pl_xy = mvnormal_pl.sample().unsqueeze(0)

                mvnormal_ol = torchdist.MultivariateNormal(mean_ol, cov_ol)
                output_ol_xy = mvnormal_ol.sample().unsqueeze(0)

                # sampling
                shot_prob = F.softmax(output_shot_logits, dim=-1)
                output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                height_prob = F.softmax(output_height_logits, dim=-1)
                output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                aroundhead_prob = F.softmax(output_aroundhead_logits, dim=-1)
                output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                backhand_prob = F.softmax(output_backhand_logits, dim=-1)
                output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                playerloc_prob = F.softmax(output_playerloc_logits, dim=-1)
                output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                opponentloc_prob = F.softmax(output_opponentloc_logits, dim=-1)
                output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                while output_shot[0, -1, 0] == 0:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_height[0, -1, 0] == 0:
                    output_height = height_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_aroundhead[0, -1, 0] == 0:
                    output_aroundhead = aroundhead_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_backhand[0, -1, 0] == 0:
                    output_backhand = backhand_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_playerloc[0, -1, 0] == 0:
                    output_playerloc = playerloc_prob[0].multinomial(num_samples=1).unsqueeze(0)
                while output_opponentloc[0, -1, 0] == 0:
                    output_opponentloc = opponentloc_prob[0].multinomial(num_samples=1).unsqueeze(0)

                prev_shot = output_shot[:, -1, :]
                prev_height = output_height[:, -1, :]
                prev_aroundhead = output_aroundhead[:, -1, :]
                prev_backhand = output_backhand[:, -1, :]
                prev_player_location_area = output_playerloc[:, -1, :]
                prev_opponent_location_area = output_opponentloc[:, -1, :]
                prev_x = output_xy[:, -1, 0].unsqueeze(1)
                prev_y = output_xy[:, -1, 1].unsqueeze(1)
                prev_pl_x = output_pl_xy[:, -1, 0].unsqueeze(1)
                prev_pl_y = output_pl_xy[:, -1, 1].unsqueeze(1)
                prev_ol_x = output_ol_xy[:, -1, 0].unsqueeze(1)
                prev_ol_y = output_ol_xy[:, -1, 1].unsqueeze(1)
                prev_player = target_player.clone()

                # transform to original format
                ori_shot = config['uniques_type'][prev_shot.item()-1]
                ori_x = prev_x.item()
                ori_y = prev_y.item()

                current_generated_shot.append(shot_prob[0][-1][1:].cpu().tolist()) # 0 is pad
                current_generated_area.append((ori_x, ori_y))

            generated_shot_logits.append(current_generated_shot), generated_area_coordinates.append(current_generated_area)

    return generated_shot_logits, generated_area_coordinates


