# A minimal worked example LUCAS segmentation model


## Setting up your python environment

Run the following code from the terminal to set up your repository and conda environment. 
Make sure that [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) 
and [python](https://wiki.python.org/moin/BeginnersGuide/Download) are installed prior. 

```bash
git clone git@github.com:pepaaran/python_proj_template.git
cd python_proj_template
conda env create -f environment.yml --name mlenv
conda activate mlenv
```

### Docker

An isolated, self-contained environment can be created using the provided Dockerfile.

```
# In the main project directory run
docker build -f Dockerfile -t mlenv .
```

Data will be mounted in the docker virtual machine at `/workspace` and is fully accessible (writing and reading of files on your local file system), using the following command.

```
docker run -it --rm -v "${PWD}":/workspace/ mlenv
```

## Loading the required data

Download the data in the main project directory in the `data/` directory. Using the `src/prepare_data.py` script divide the data in a train, validation and test datasets. This split will be written as a `data.json` file in the diretory containing images and mask files.

## Training the model

 Use the `train.py` script to train the model by passing as single argument, the data directory. It is assumed that the `data.json` file is generated.
