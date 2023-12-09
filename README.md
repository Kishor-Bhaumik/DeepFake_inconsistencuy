

This is the official implementation of WDC-2023 paper by Kishor Kumar Bhaumikand Simon S. Woo \
[Exploiting Inconsistencies in Object Representations for DeepFake Video Detection](https://dl.acm.org/doi/pdf/10.1145/3595353.3595885).


Setup: Run

  ```shell
  pip install -r requirements.txt
  ```


step1: in dataloader_cons.py  set the dataset path in  

  ``` /home/data/DeepFake/ ```

Step 3: To train the model run   
  ```shell
  python train_endtoend.py
  ```
Step 4: To test the model run   
  ```shell
  python evaluate.py
  ```
