

This is the official implementation of WDC-2023 paper by Kishor Kumar Bhaumikand Simon S. Woo \
[Exploiting Inconsistencies in Object Representations for DeepFake Video Detection](https://dl.acm.org/doi/pdf/10.1145/3595353.3595885).


Setup: Run

  ```shell
  pip install -r requirements.txt
  ```

Step 1: Download IMD-20 Real Life Manipulated Images from [Link](http://staff.utia.cas.cz/novozada/db/).

step2: set the dataset path in  

for example, if you have downloaded and unzipped the IMD2020 dataset in the following directory: ``` /home/forgery/ ```  then put  ``` /home/forgery/ ```  as the base_dir  in the config file. (DO NOT put  ``` /home/forgery/IMD2020/ ``` in base_dir )

Step 3: To train the model run   
  ```shell
  python trainer.py
  ```
Step 4: To test the model run   
  ```shell
  python evaluate.py
  ```
