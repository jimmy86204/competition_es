# introduce
* Preprocess/: store code for preprocess and feature engineering
* Model/: store code for build model and infer
* requirements.txt: dependency package
* main.py: main process for training and inference
# usage

0. store the data: to store training / testing / submission data into ./Data/training_data/
1. install package

    ```
    pip install -r requirements.txt
    ```

2. preprocess and feature engineer
    ```
    python ./Preprocess/x.py
    python ./Preprocess/y.py
    ```
    will create train / valid / test data in ./Data

3. training and inference
    ```
    python main.py
    ```
    will create final result in ./Output. 
    
    Note: there should exist a submission data in path ./Data/training_data which is named: 31_範例繳交檔案.csv.