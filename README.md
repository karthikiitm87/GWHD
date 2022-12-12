# Setting up the environment
We recommend to create a virtual environment and use the following command inside the virtual environemnt. Executing the following command will install the required libraries. 

```
pip install -r requirements.txt
```


# GWHD
This repo has all the codes which we used for global wheat head detection 2021 challenge. Once all the dependencies are installed as mentioned in 'requirements.txt', the dataset can be downloaded by running the following script in the command prompt

```
./downloads.sh
```

After running the above script, ensure that the data is organised in the following folder structure

```
.
├── datasets
│   ├── gwhd_2021
    |     ├── images
    ├── Annots
    |     ├── official_train.csv
    |     ├── official_val.csv
    |     ├── official_test.csv
    
```

Once the above directory structure is in place, the following command can be executed to train the basline Faster-RCNN for wheat detection.

```
python baseline.py
```
