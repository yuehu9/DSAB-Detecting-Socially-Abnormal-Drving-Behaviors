

epoch=500
lr=5e-2 
train_file='simulation_processed'
batch_size=64


python3 train.py --train_file_name $train_file --experiment_name 'DSAB' --learning_rate $lr --epoch $epoch --batch_size $batch_size

for test_scenario_file in 'testing_comprehensive'
# 'testing_stalled_car'  'testing_comprehensive' 'testing_slow' 'testing_speeding'  'testing_tailgating' 
do
python3 test.py --train_file_name $train_file --experiment_name 'DSAB'  --test_scenario_file $test_scenario_file  
done
