# Transformer Implementation

## 0. Preparation
* Python 3.6+
* torch>=1.6.0+cu101
* sentencepiece>=0.1.94
* numpy>=1.18.5
* progressbar>=2.5
* lxml>=4.6.1

## 1. Data Preprocessing
* Process  
    ```
    web => (urllib) => tar 
        => (tarfile) => xml 
        => (lxml) => txt
        => (setencepiece) => embedding vector
    ```
