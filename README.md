# OpenMoCap: Rethinking Optical Motion Capture under Real-world Occlusion

[![Paper](https://img.shields.io/badge/Paper-MM%2725-blue)](https://arxiv.org/abs/2508.12610)  [![Project](https://img.shields.io/badge/Project-OpenMoCap-green)](https://qianchen214.github.io/openmocap.github.io/) [![Data](https://img.shields.io/badge/Data-Occlu-red)](https://huggingface.co/datasets/chen-qian/CMU-Occlu/tree/main) [![HF](https://img.shields.io/badge/HF-OpenMoCap-yellow)](https://huggingface.co/papers/2508.12610)

Official implementation of **OpenMoCap** from *ACM Multimedia 2025*:  
**OpenMoCap: Rethinking Optical Motion Capture under Real-world Occlusion**  
Chen Qian, Danyang Li, Xinran Yu, Zheng Yang, Qiang Ma



## 📌 Introduction

Optical motion capture (MoCap) suffers severe performance degradation under real-world occlusions. Existing methods rely on unrealistic random occlusion augmentations and fail to model long-range dependencies between markers.  

We propose:  

- **CMU-Occlu Dataset**: a large-scale dataset simulating realistic occlusion via ray tracing.  
- **OpenMoCap Solver**: a novel motion-solving model designed specifically for robust motion capture in environments with significant occlusions.  



## 💡 Notes
- The pretrained weights have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1lt9LWSYykaD_lA_ubnddJs10HA0rHyjv?usp=sharing) or [Hugging Face](https://huggingface.co/chen-qian/OpenMoCap/tree/main). Please download them and place the files under `./marker_joint_6d/` and `./openmocap_position/`.
- These weights are trained on the **CMU-Occlu** dataset.  
- For **SFU** or other datasets, fine-tuning is recommended.  



## 🛠 Preparation

### Docker

#### Build

```bash
cd OpenMoCap
docker build -t openmocap:cu116 .
```

#### Enter Container

```bash
docker run -it --rm \
  --gpus all \
  -v /path/to/your/local/test_cases:/app/test_cases \
  -v /path/to/your/local/results:/app/results \
  --entrypoint /bin/bash \
  openmocap:cu116
```

Example
```bash
docker run -it --rm \
  --gpus all \
  -v $(pwd)/test_cases:/app/test_cases \
  -v $(pwd)/results:/app/results \
  --entrypoint /bin/bash \
  openmocap:cu116
```



## 🚀 Usage

### Inference + Rendering

Run the pretrained models on test sequences:

```bash
python solve_and_render.py \
    --data_dir /path/to/your/test_data \
    --out_dir /path/to/your/output \
    --model_pos /path/to/model_pos.pth \
    --model_rot /path/to/model_ang.pth \
    --render \
    --key pred_j_p \
    --random_one
```

Example

```bash
python solve_and_render.py \
    --data_dir ./test_cases/sfu \
    --out_dir ./results \
    --model_pos ./openmocap_position/model_pos.pth \
    --model_rot ./marker_joint_6d/model_ang.pth \
    --render \
    --key pred_j_p \
    --random_one
```

#### Key Arguments Explanation
- `--random_one`: 
If enabled, randomly select only one .npz file from data_dir for processing. Useful for quick testing or debugging without processing all files.

- `--render`: Enable 3D visualization and render results to an MP4 video. Detailed rendering parameters (e.g., view angles, frame rate, resolution) can be adjusted via command-line arguments for customized visualization.

- `--key`: Data key to read from the predicted NPZ for rendering.


We provide the visualization files for input markers, which are stored in `./visualization/input_marker_visualization`.



### Training

Position solver:

```bash
python train_position.py --dataset /path/to/your/dataset --out /path/to/your/output
```

Rotation solver:

```bash
python train_6d.py --dataset /path/to/your/dataset --out /path/to/your/output
```



## Acknowledgement
Our code is inspired by [MAE](https://github.com/facebookresearch/mae) and [MoCap-Solver](https://github.com/NetEase-GameAI/MoCap-Solver), and benefits from the [AMASS](https://amass.is.tue.mpg.de/) dataset.

We sincerely thank the authors for their excellent work.




## 📜 Citation

If you find our work inspiring or use our code or dataset in your research, please consider giving a star ⭐ and a citation.

```bibtex
@inproceedings{qian2025openmocap,
  title={OpenMoCap: Rethinking Optical Motion Capture under Real-world Occlusion},
  author={Qian, Chen and Li, Danyang and Yu, Xinran and Yang, Zheng and Ma, Qiang},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={7529--7537},
  year={2025}
}
```
