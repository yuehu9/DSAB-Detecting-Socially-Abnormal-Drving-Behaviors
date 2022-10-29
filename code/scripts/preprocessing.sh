
### training ###
python3 preprocess_simulation.py  --save_dir_name "simulation_processed" --process_scenario 'training'

### testing ###
for process_scenario in 'comprehensive'
# 'speeding' 'slow' 'comprehensive' 'stalled_car' 'tailgating' 
do
    python3 preprocess_simulation.py --save_dir_name "simulation_processed" --process_scenario $process_scenario
done
