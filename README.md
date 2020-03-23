# Prediction of Neuropeptides from Sequence Information Using Ensemble Classifier and Hybrid Features

Yannan Bin, Wei Zhang, Wending Tang, Ruyu Dai, Qizhi Zhu, Junfeng Xia*  

Institutes of Physical Science and Information Technology and School of Computer Science and Technology, Anhui University, Hefei, Anhui 230601, China

Email: [jfxia[at]ahu.edu.cn](mailto:jfxia@ahu.edu.cn)



## Installation

- Requirement
  
  OS：
  
  - `Windows` ：Windows7 or later
  
  - `Linux`：Ubuntu16.04 LTS or later
  
  Python：
  
  - `Python` >= 3.6
  
- Download `PredNeuroP `to your computer

  ```bash
  git clone https://e.coding.net/xtzhwei/PreNeuroP/PredNeuroP.git
  ```

- open the dir and install `requirement.txt` with `pip`

  ```
  cd PredNeuroP
  pip install -r requirement.txt
  ```

## Predict 

```
python PredNeuroP.py -f file.fasta -o result.txt
```

- `-f `: input the test file with fasta format
- `-o` : the result of probalitity 