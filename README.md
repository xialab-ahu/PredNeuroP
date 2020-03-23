# Prediction of Neuropeptides from Sequence Information Using Ensemble Classifier and Hybrid Features

Yannan Bin, Wei Zhang, Wending Tang, Ruyu Dai, Qizhi Zhu, Junfeng Xia*  

Institutes of Physical Science and Information Technology and School of Computer Science and Technology, Anhui University, Hefei, Anhui 230601, China

Email: [jfxia[at]ahu.edu.cn](mailto:jfxia@ahu.edu.cn)



## Installation

- Requirement
  
  OS：
  
  - `Windows` ：Windows7 or later
  
  - `Linux`：Ubuntu 16.04 LTS or later
  
  Python：
  
  - `Python` >= 3.6
  
- Download `PredNeuroP`to your computer

  ```bash
  git clone https://github.com/xialab-ahu/PredNeuroP.git
  ```

- open the dir and install `requirement.txt` with `pip`

  ```
  cd PredNeuroP
  pip install -r requirement.txt
  ```

## Predict 

```shell
python PredNeuroP.py -f ./sample/sample.txt -o ./sample/sample_result.csv
```

- `-f `: input the test file with fasta format
- `-o` : the result of probalitity 