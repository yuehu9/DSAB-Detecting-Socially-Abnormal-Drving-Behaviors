

train_file='simulation_processed'

for test_scenario_file in 'testing_comprehensive'
# 'testing_stalled_car'  'testing_comprehensive' 'testing_slow' 'testing_speeding'  'testing_tailgating' 
do
python3 test.py --train_file_name $train_file --experiment_name 'DSAB_trained'  --test_scenario_file $test_scenario_file
done
