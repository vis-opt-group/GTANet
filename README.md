## GTA-Net: Gradual Temporal Aggregation Network for Fast Video Deraining
This repo contains code accompaning the paper, [GTA-Net: Gradual Temporal Aggregation Network for Fast Video Deraining (Xue et al., ICASSP 2021)](https://ieeexplore.ieee.org/document/9413698).

### Abstract
Recently, the development of intelligent technology arouses the requirements of high-quality videos. Rain streak is a frequent and inevitable factor to degrade the video. Many researchers have put their energies into eliminating the adverse effects of rainy video. Unfortunately, how to fully utilize the temporal information from rainy video is still in suspense. In this work, to effectively exploit temporal information, we develop a simple but effective network, Gradual Temporal Aggregation Network (GTA-Net for short). To be specific, according to the temporal distance between rainy frames and the reference frame, we divide the rainy frames into different groups. A multi-stream coarse temporal aggregation module is first performed to aggregate different temporal information with equal status and importance. Then we design a singlestream fine temporal aggregation module to further fuse the integrated frames that maintain the different distances with the target frame. In this way of coarse-to-fine, we not only achieve superior performance, but also gain the surprising execution speed owing to abandon the time-consuming alignment operation. Plenty of experimental results demonstrate that our GTA-Net performs favorably compared to other stateof-the-art approaches. The meticulous ablation study further indicates the effectiveness of our designed GTA-Net.

### Dependencies

```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts```


###  Data Preparation

You can download the [RainSynLight25](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018), 
[RainSynComplex25](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018), and [NTURain](https://github.com/hotndy/SPAC-SupplementaryMaterials) dataset from the attached link.

### Usage

You can first modify `config.py`.

For train the model:
```
python multi_train.py
```

For evaluate the test dataset:
```
python multi_evaluate.py
```
 
### Citation

If you use GTANet for academic research, you are highly encouraged to cite the following paper:
- Xinwei Xue, Xiangyu Meng, Long Ma, Risheng Liu, Xin Fan. ["GTA-Net: Gradual Temporal Aggregation Network for Fast Video Deraining"](https://ieeexplore.ieee.org/document/9413698). ICASSP, 2021.

### License 

MIT License

Copyright (c) 2021 Vision Optimizaion Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
