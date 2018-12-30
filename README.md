# Versions

To reproduce the results in [Automatic blood glucose prediction with confidence
using recurrent neural networks](http://ceur-ws.org/Vol-2148/paper10.pdf) revert to commit: [a5f0ebcf45f87b63d118dcad5e96eb505bb4269a](https://github.com/johnmartinsson/blood-glucose-prediction/commit/a5f0ebcf45f87b63d118dcad5e96eb505bb4269a) and follow the README.

# Prerequisites
The code is designed to be run on the OhioT1DM Dataset. So to use it the xml_path in e.g. the example experiment YAML configuration need to point to the path on disk where the XML data files are.

It would of cource be possible to write a new dataset module which loads the data into the required format and train the models on other data as well.

# Installation
    $> chmod +x setup.sh
    $> ./setup.sh

# Running an experiment
Note that this is designed to run on the Ohio Diabetes dataset. You need to
explicitly state the absolute file path to the XML file of the patient you want
to train the model for in the experiment configuration file (YAML file).

Except for that, everything should run out of the box.

    $> chmod +x run.py
    $> ./run.py --file experiments/example.yaml -m train

All results are collected and stored in the 'artifacts' directory. To visualize the training session you can run

    $> tensorboard --logdir artifacts/<artifacts-path>

and fire up tensorboard.

# Reproduce parameter search

    $> mkdir parameter_search
    $> python generate_search_configurations_over_lstm_states_and_past_steps.py -f <path-to-ohio-xml-file-dir> -o parameter_search
    $> ./train_all.sh parameter_search

# Reproduce final results

    $> mkdir final_experiments_nll
    $> python generate_final_experiments_nll.py -f <path-to-ohio-xml-file-dir> -o final_experiments_nll
    $> ./train_all.sh final_experiments_nll

and so on for mse and gmse.
