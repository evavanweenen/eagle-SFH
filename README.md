# EAGLE - SFH

This repository is used to infer stellar masses and star formation rates from spectra of galaxies of the [EAGLE](http://eagle.strw.leidenuniv.nl/) simulations using neural networks, as a research project of the MSc Astronomy and Data Science at Leiden University.

The supervisors of this project are prof. Joop Schaye, dr. Camila Correa and dr. James Trayford.

The thesis can be found [here](https://github.com/evavanweenen/eagle-SFH/Master_thesis_Eva_van_Weenen.pdf)

## Data
The data can be downloaded from the [EAGLE database](http://virgodb.dur.ac.uk:8080/Eagle/) with the following script:

    SELECT
        SH.TopLeafID as topleafid,
        SH.GalaxyID as galid,
        SH.Redshift as z,
        SH.SubGroupNumber as subgroup,
        AP.Mass_Star as m_star,
        DF.SDSS_u as dustyflux_sdss_u,
        DF.SDSS_g as dustyflux_sdss_g,
        DF.SDSS_r as dustyflux_sdss_r,
        DF.SDSS_i as dustyflux_sdss_i,
        DF.SDSS_z as dustyflux_sdss_z
    FROM
        RefL0100N1504_SubHalo as SH,
        RefL0100N1504_Aperture as AP,
        RefL0100N1504_DustyFluxes as DF
    WHERE
        -- Select aperture size to be 30 pkpc
        AP.ApertureSize = 30
        and SH.SnapNum = 27
        -- Join the objects in 3 tables
        and SH.GalaxyID = AP.GalaxyID
        and SH.GalaxyID = DF.GalaxyID
    ORDER BY
        SH.TopLeafID,
        SH.GalaxyID asc,
        SH.Redshift asc 

When reading the data, change the filename in `io.py` to your filename.

The Sloan Digital Sky Survey (SDSS) catalogue used is the catalogue of [Chang et al. 2015](https://iopscience.iop.org/article/10.1088/0067-0049/219/1/8). Pleasure ensure that the heading of the table corresponds with the headings used in the preamble of the code.

## Repository
The repository consists of the eagle module, which consists of code to read data from EAGLE simulations and SDSS, cosmological calculations, the framework for the neural network used and the plots in this project.

Additionally, the repository contains scripts that use the eagle module, to perform experiments in the research project. One basic script is the `mlp.py` script that can be used to train and test a multi-layer perceptron.

Note that some scripts used in the beginning of the project are still old and need to be updated for new versions of the eagle module (you will notice this if you get errors when running the script). This update is planned for August/September 2019.

### Overview of scripts
The code can be run as follows:

    mlp.py <input> <sampling>
where `<input>` is the input features to use (either *nocolors*, *subsetcolors* or *allcolors*) and `<sampling>` is the sampling procedure to use (either *None*, *random*, or *uniform*).

In the preamble of the code, the simulation, snapshot (and thus redshift) are defined, that are used when reading the data in `io.py`. These settings are also used in the filename and title of plots. Furthermore, the names of the data columns that are read are defined (*xcols* and *ycols*) and whether this table contains fluxes or magnitudes (*xtype*). The names that should be used in the plots are defined as *xnames* and *ynames*. For a detailed description of the preprocessing steps, please refer to the thesis.

Additionally, the settings of the *nocolors*, *subsetcolors* and *allcolors* architectures are defined, as optimized with `hyperas`. If you want to use a different architecture, you can define it here.

Hyperparameters are optimized with the python scripts `gridsearch.py` and `tpe.py`. The script `gridsearch.py` is used as an initial probe of the hyperparameter-space. The script `tpe.py` is used for the final optimization of the hyperparameters and uses Tree-Structured Parzen Estimators of the `hyperas` package.

After training the neural network, SHAP values are calculated using the `shap` package. 

The following experiments are performed:
* `mlp_cs.py` Central and satellite analysis 
* `mlp_earlystopping.py` [old] Implement early stopping (this is not used in the thesis)
* `mlp_featureimportance.py` [old] Try-out all combinations of inputs (this is not used in the thesis)
* `mlp_featureimportance_bottomup.py` [probably-old] Perform a bottom-up sequential search to find the best combination of inputs based on errors of the EAGLE data
* `mlp_featureimportance_bottomup_sdss.py` [probably-old] Perform a bottom-up sequential search to find the best combination of inputs based on errors of the SDSS dta
* `mlp_inverse.py` Train on SDSS galaxies and test on EAGLE galaxies
* `mlp_modifyinput.py` [probably-old] Add noise to the input features of the neural network
* `mlp_modifyoutput.py` Add noise to the output features of the neural network

To perform an experiment for all different samplings (*None*, *random*, *uniform*) and input features (*nocolors*, *subsetcolors*, *allcolors*), use the bash script `mlp` (and edit it to run the experiment of your choice). To combine all plots of the samplings and inputs into one pdf, run the bash script `summary`. The tables with error measures can be read with `load_tables.py`.

Currently the code is used to predict stellar mass at one specific redshift. This could be improved by predicting the stellar mass at several redshifts (e.g. use redshift as an input), and by predicting the star formation rate instead of stellar mass. 
Before publishing these results, don't forget to try out different random seeds to ensure that we are not optimizing over this specific random seed.

## Benchmark
The code is run with `python 3.6.8` and the following packages are used:
* `numpy 1.16.1`
* `scipy 1.2.0`
* `keras 2.2.4` (and `tensorflow 1.5.0`)
* `hyperas 0.4` (and `hyperopt 0.1.1`)
* `scikit-learn 0.20.2`
* `shap 0.28.3`
* `astropy 3.1.1`
* `h5py 2.9.0`
* `matplotlib 3.0.2`
Please note that I had to use an older version of tensorflow because I used an old computer, of which the CPU did not understand some instructions. Newer versions of these packages will probably work as well on your computer.

### Instructions for creating a virtual environment
If you want to create a virtual environment, please follow the following instructions:

1. Open a terminal and go to the directory you want to put your virtual environment in. Create a directory `env` for the environment.

        cd <dir>
        bash -l
        mkdir env
2. Download and unpack the correct python version (version 3.6.8 here)

        wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
        tar -zxvf Python-3.6.8.tgz
3. Install python in a directory `.python3.6`

        cd Python-3.6.8
        mkdir ../.python3.6
        ./configure --prefix=<dir>/.python3.6
        make
        make install
        cd ..
4. Install `virtualenv`

        wget https://files.pythonhosted.org/packages/59/38/55dd25a965990bd93f77eb765b189e72cf581ce1c2de651cb7b1dea74ed1/virtualenv-16.2.0.tar.gz
        tar -zxvf virtualenv-16.2.0.tar.gz
        cd virtualenv-16.2.0
        <dir>/.python3.6/bin/python3 setup.py install
        cd ..
5. Create virtual environment and activate it

        virtualenv-16.2.0/virtualenv.py env -p <dir>/.python3.6/bin/python3.6
        source env/bin/activate
        
Now you can run the code in your virtual environment. The lines to repeat before running the code are then:

    bash -l
    source env/bin/activate

There are likely mistakes in these instructions and they were specific for my machine as I did not have admin rights and had to install this specific version of python locally, but maybe you have and can follow easier steps. If you encounter any mistakes, google will help you ;)



