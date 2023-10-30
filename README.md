<p align="center">

  <h1 align="center">Cycling Augmentation (CycAug) on Visual RL</h1>
  <h2 align="center"><a href="https://arxiv.org/abs/2305.16379">Learning Better with Less: Effective Augmentation for Sample-Efficient VRL</a></h2>
  <p align="center">
    <a><strong>Guozheng Ma</strong></a>
    ·
    <a><strong>Linrui Zhang</strong></a>
    ·
    <a><strong>Haoyu Wang</strong></a>
    ·
    <a><strong>Lu Li</strong></a>
    ·
    <a><strong>Zilin Wang</strong></a>
  </p>
  <p align="center">
    <a><strong>Zhen Wang</strong></a>
    ·
    <a><strong>Li Shen</strong></a>
    ·
    <a><strong>Xueqian Wang</strong></a>
    ·
    <a><strong>DaCheng Tao</strong></a>
  </p>

</p>

<div align="center">
  <img src="Figures/CycAug.gif" alt="main" width="90%">
</div>

## 🌟 Methods
To investigate this issue and further explore the potential of DA, we conduct comprehensive experiments to assess the impact of DA’s attributes on its efficacy and provides the following insights and improvements: 
- For individual DA operations, we reveal that both ample spatial diversity and slight hardness are indispensable. Building on this finding, we introduce Random PadResize (Rand PR), a new DA operation that offers abundant spatial diversity with minimal hardness. 
- For multi-type DA fusion schemes, the increased DA hardness and unstable data distribution result in the current fusion schemes being unable to achieve higher sample efficiency than their corresponding individual operations. **Taking the non-stationary nature of RL into account, we propose a RL-tailored multi-type DA fusion scheme called Cycling Augmentation (CycAug), which performs periodic cycles of different DA operations to increase type diversity while maintaining data distribution consistency**.

## 🤖 Implementation

The implementation of CycAug on DMC tasks and CARLA tasks can be found in the respective subfolders named `CycAug_on_DMC` and `CycAug_on_CARLA`.

## 📝 Citation
If this repository is useful to you, please consider citing our paper:
```
@article{ma2023learning,
  title={Learning Better with Less: Effective Augmentation for Sample-Efficient Visual Reinforcement Learning},
  author={Ma, Guozheng and Zhang, Linrui and Wang, Haoyu and Li, Lu and Wang, Zilin and Wang, Zhen and Shen, Li and Wang, Xueqian and Tao, Dacheng},
  journal={arXiv preprint arXiv:2305.16379},
  year={2023}
}
```
