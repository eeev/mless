# Homework 2

Task description:
> 1. Extend the code in notebook 7_multi_model_plotting to read the forecast results from all trained models and plot all curves in one panel. Make sure that your code can be re-used if you make forecasts for other stations or time episodes.
> 2. Extend at least one of the models to input multivariate data. Download ozone data from TOAR (use the scripts in notebook 1). Copy the notebook with the model of your choice into a new notebook and extend the code so that temperazure and ozone data are used as inputs. The goal is to forecast ozone concentrations, you don't need to output temperature.
> 3. In reality, you will often have forecasts of weather variables (here temperature) available, so you can use future temperature values to forecast the ozone concentrations. Make another copy of your multivariate notebook and adjust the code accordingly. Use your multi-model plotter to compare the results from tasks 2 and 3.

## Solution:

### 1)

#### A) Preparation

First, working through notebooks 1-6, I solved some tasks and answered questions along the way, adding markdown units in-place. 

For the other notebooks, it was more easy to follow since PyTorch was used, and this is better supported by AMD graphics card using ROCm. I have had a lot of trouble installing tensorflow for the correct version of Python/ROCm, and even after installing everything properly, the kernel crashes for any invocation of a tensorflow function, even when just trying to display the version. This lead to a lot of time lost on an issue that can ultimately not be resolved. Thus, I resorted to running the notebooks via CPU.

I noticed the shape of the data is not the same as in the example notebooks (i.e., my csv had more data points..), but I was able to still properly normalize it and follow all notebooks until number five. The LSTM output was significantly different than that of the notebook uploaded to git, I suspect an error in data handling on my side.

Thus, I decided to download exactly the X_test, X_train, etc. files to match the notebook. Finally, the output of notebook 5 using the downloaded pkl's looked correct; 

I ran into issues trying to launch a local wandb server due to some error with the docker container, so I connected to the wandb web service to track runs.

#### B) Task Solving

Idea: Grab the code from previous notebooks to load trained models via their .pkl files, then use their prediction visualization code & generalize it to arbitrary inputs (i.e., stations codes, time episodes).

I started with the PatchTST model, which uses the raw data in the code to make predictions. However, this will lead to issues when plotting later, as the other models use a normalized (& sequenced) dataset.

For this reason I have re-trained PatchTST with the normalized data, which yields visually the same result as before, but now the same graph across all models can be used.

It turns out that there is some issue between PatchTST using the normalized data csv and the other models using the .pkl data (their ground truth context is different, thus predictions PatchTST <-> others are misaligned/off). For this reason I re-train PatchTST on the .pkl data.

After re-training, I plotted all 4 model predictions in one panel, it is adjusted such that the desired time episode can easily be adjusted in the viz code.

As a result, I compare model performance across different time episodes. At this time I assume my plotting code/notebook is adjustable to different stations; It should also be possible to run inference for different stations and use the trained models as-is.

> I noticed an issue with notebook 3, where the .pkl data is prepared. To start, the `create_sequences(...)` method creates sequences for all (3) stations correctly and concatenates results. However, then the train/test split is performed temporally across all stations. This is problematic, and we can easily check this: if querying the pkl for train/test datasets, we find that no test dataset for station "DEBW073" exists, or rather, said station is not part of the global test dataset. The last cells in notebook 3 elaborate on this.

In order to avoid confusion on the construction of train/test datasets, and which were used for what models, I implemented the following in notebook 3:
- X_train_corrected.pkl - contains sequenced priors for all 3 stations based on 80% of per-station data for training
- Y_train_corrected.pkl - contains sequenced posts for all 3 stations based on 20% of per-station data for training

accordingly, X_test_corrected.pkl and Y_test_corrected.pkl are used for testing in a similar fashion.

#### C) Model Comparison

Now that the data is standardized across models, here is how it is being used:
- PatchTST: Trained on station code "DENW094" using default train/test splits from .pkl's
- MLP, LSTM: Trained on all stations
- AR (SARIMAX): Fitted on the pkl's test sequences




### 2)

### 3)

