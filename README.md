# IJCAI-CoachAI-Challenge-2023
This is a repository of IJCAI-CoachAI-Challenge-2023 Team Badminseok

The increasing use of artificial intelligence (AI) technology in turn-based sports, such as badminton, has sparked significant interest in evaluating strategies through the analysis of match video data. Predicting future shots based on past ones plays a vital role in coaching and strategic planning. In this study, we present a Multi-Layer Multi-Input Transformer Network (MuLMINet) that leverages professional badminton player match data to accurately predict future shot types and area coordinates. Our approach resulted in achieving the runner-up (2nd place) in the IJCAI CoachAI Badminton Challenge 2023, Track 2. To facilitate further research, we have made our code publicly accessible online, contributing to the broader research community's knowledge and advancements in the field of AI-assisted sports analysis.

In this challenge, the objective is to predict the future stroke shot type and area coordinates based on the data from the first 4 strokes. The evaluation metrics used for this task are defined as follows:

$Score=min(l_1, l_2, ..., l_6)$

$l_i=AVG(CE + MAE)$


The shot type is evaluated using the Cross-Entropy (CE) loss, which measures the discrepancy between the predicted and actual shot-type probabilities. The area coordinate is evaluated using the Mean Absolute Error (MAE), which quantifies the average difference between the predicted and actual area coordinates.
To calculate the final score, we take the average of six evaluation metrics and select the minimum value as the final score. This score serves as an overall measure of the predictive model's performance in predicting stroke shot type and landing coordination. The team with the lowest loss emerges as the winner of this challenge.


![BadminseokNet](https://github.com/stan5dard/IJCAI-CoachAI-Challenge-2023/assets/79134282/6ecf8c78-e0f1-41c7-8fdd-a521265ed26e)
