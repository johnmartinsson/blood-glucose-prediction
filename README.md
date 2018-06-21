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

# Reproduce parameter search

    $> mkdir parameter_search
    $> python generate_search_configurations_over_lstm_states_and_past_steps.py -f <path-to-ohio-xml-file-dir> -o final_experiments
    $> ./train_all.sh parameter_search

# Reproduce final results

    $> mkdir final_experiments
    $> python generate_final_experiments.py -f <path-to-ohio-xml-file-dir> -o final_experiments
    $> ./train_all.sh final_experiments
