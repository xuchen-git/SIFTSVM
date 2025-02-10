# SIFTSVM
A Matlab code for Intuitionistic Fuzzy Twin Support Vector Machine for Semi-Supervised Learning(SIFTSVM)
Title：Intuitionistic Fuzzy Twin Support Vector Machine for Semi-Supervised Learning(Submited) 
Author: Chen X  , Shao Y H , Bai L ,Wang Z . 

flowchart TD
    A[输入数据<br/>(如原始特征、序列等)]
    A --> B[编码器<br/>(Encoder)]
    B --> C[隐空间表示/编码<br/>(Latent Representation)]
    C --> D[解码器<br/>(Decoder)]
    D --> E[输出解<br/>(生成的序列、选址方案)]
