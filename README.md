## IJCAI-CoachAI-Challenge-2023
This is a repository of IJCAI-CoachAI-Challenge-2023 Team Badminseok [https://sites.google.com/view/coachai-challenge-2023/, https://github.com/wywyWang/CoachAI-Projects/tree/main]

The CoachAI Badminton Challenge 2023 is a competition that aims to apply intelligence technology to badminton analytics. The challenge is divided into two tasks:

Automatic Annotation of Technical Data for Badminton Match Videos (Track 1): This task involves designing solutions with computer vision techniques to automatically annotate shot-by-shot data from match videos.

Forecasting of Future Turn-Based Strokes in Badminton Rallies (Track 2): This task requires the design of predictive models that can forecast future strokes, including shot types and locations, based on past strokes.

The competition is being held in conjunction with IJCAI 2023 in Macao, S.A.R, from August 19th to 25th, 2023. The challenge is organized by a team from the National Yang Ming Chiao Tung University, Taiwan. For more details, participants are encouraged to visit the competition's repository to check out previous work.

The increasing use of artificial intelligence (AI) technology in turn-based sports, such as badminton, has sparked significant interest in evaluating strategies through the analysis of match video data. Predicting future shots based on past ones plays a vital role in coaching and strategic planning. In this study, we present a Multi-Layer Multi-Input Transformer Network (MuLMINet) that leverages professional badminton player match data to accurately predict future shot types and area coordinates. Our approach resulted in achieving the runner-up (2nd place) in the IJCAI CoachAI Badminton Challenge 2023, Track 2. To facilitate further research, we have made our code publicly accessible online, contributing to the broader research community's knowledge and advancements in the field of AI-assisted sports analysis.

In this challenge, the objective is to predict the future stroke shot type and area coordinates based on the data from the first 4 strokes. The evaluation metrics used for this task are defined as follows:

$$Score=min(l_1, l_2, ..., l_6)$$

$$l_i=AVG(CE + MAE)$$


The shot type is evaluated using the Cross-Entropy (CE) loss, which measures the discrepancy between the predicted and actual shot-type probabilities. The area coordinate is evaluated using the Mean Absolute Error (MAE), which quantifies the average difference between the predicted and actual area coordinates.
To calculate the final score, we take the average of six evaluation metrics and select the minimum value as the final score. This score serves as an overall measure of the predictive model's performance in predicting stroke shot type and landing coordination. The team with the lowest loss emerges as the winner of this challenge.

![MuLMINet](https://github.com/stan5dard/IJCAI-CoachAI-Challenge-2023/assets/79134282/4127b597-59c0-447f-b632-e96a7bbecdba)
