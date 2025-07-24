This code repository includes code and data to run an event based spatio-temporal network transport model. To use it, one needs only python installed (all together should take only a few minutes).

First, create and activate a python virtual environment using "py -m venv venv" > "venv\Scripts\activate"
Download dependencies using "pip install -r "requirements.txt". The model was tested using python 3.10.7.

To run the model for the SZU network, first execute the script sim_schedule_v.py. Results of this one run are saved in the simdata folder. The expected runtime of this simulation is less than one minute.

Then, run the notebook spatial_map.ipynb. It will craft the spatial map as shown in Fig. 3, using the data from the simulated runs in step 1. The figure is saved to figures/stacked_heatmap.png.

This repo includes a version of DelayBufferNetwork, used to initialize a spatio-temporal network model developed for delay propagation. The script running the the model is static_delay_interaction.py. Here, we calculate the delay propagation as described in the paper. The framework presented here includes many more features, and possibilities for analysis of delay propagation. 