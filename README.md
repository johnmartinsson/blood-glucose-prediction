# Versions

To reproduce the results in [Automatic blood glucose prediction with confidence
using recurrent neural networks](http://ceur-ws.org/Vol-2148/paper10.pdf) revert to commit: [a5f0ebcf45f87b63d118dcad5e96eb505bb4269a](https://github.com/johnmartinsson/blood-glucose-prediction/commit/a5f0ebcf45f87b63d118dcad5e96eb505bb4269a) and follow the README.

# Prerequisites
The code is designed to be run on the OhioT1DM Dataset. So to use it the xml_path in e.g. the example experiment YAML configuration need to point to the path on disk where the XML data files are. E.g., change "/home/ubuntu/ohio_data/OhioT1DM-training/" to "/path/on/disk/OhioT1DM-training/" where Ohiot1DM-training is the folder containing the XML files for the ohio dataset.

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

# Reproduce final results

    $> sh run_final_experiments.sh
    $> sh evaluate_final_experiments.sh
    $> python create_table.py

and so on for mse and gmse.
