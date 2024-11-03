# *Feelsight* : A visuo-tactile robot manipulation dataset

<!-- Provide a quick summary of the dataset. -->

<div style="text-align: center;">
    <video width="80%" onmouseover="this.pause()" onmouseout="this.play()" autoplay="" loop="" muted="">
        <source src="https://suddhu.github.io/neural-feels/video/dataset_zoom.mp4" type="video/mp4">
    </video>
</div>

<br> 

The FeelSight dataset is a dataset of vision, touch, and proprioception data collected from in-hand rotation of objects via an RL policy. It consists of a total of 70 experiments, 30 in the real-world and 40 in simulation, each lasting 30 seconds. For training neural field models with FeelSight, refer to the [NeuralFeels](https://github.com/facebookresearch/neuralfeels) repository.

## Simulation data 

[Our simulated data](https://suddhu.github.io/neural-feels/video/feelsight_sim_rubber_duck.mp4") is collected in IsaacGym with TACTO touch simulation in the loop. 

## Real-world data

[Here's](https://suddhu.github.io/neural-feels/video/feelsight_real_bell_pepper.mp4") an example of real-world data from our three-camera setup and the DIGIT-Allegro hand.

## Robot setup 

The Allegro hand is mounted on the Franka Emika Panda robot. The hand is sensorized with DIGIT tactile sensors, and surrounded by three Intel RealSense cameras.

<img src="https://suddhu.github.io/neural-feels/img/robot_cell.jpg" width="90%">

## Dataset structure

For dataloaders, refer to the [NeuralFeels](https://github.com/facebookresearch/neuralfeels) repository.

```bash
feelsight/ # root directory, either feelsight or feelsight_real
├── object_1/ # e.g. 077_rubiks_cube
│   ├── 00/ # log directory
│   │   ├── allegro/ # tactile sensor data
│   │   │    ├── index/ # finger id
│   │   │    │    ├── depth # only in sim, ground-truth
|   |   |    |    |     └── ..jpg 
│   │   │    │    ├── image # RGB tactile images
|   |   |    |    |     └── ..jpg 
│   │   │    │    └── mask # only in sim, ground-truth
|   |   |    |          └── ..jpg  
│   │   │    └── ..
│   │   ├── realsense/ # RGB-D data
│   │   │    ├── front-left/ # camera id
│   │   │    │    ├── image # RGB images 
|   |   |    |    |     └── ..jpg
│   │   │    │    ├── seg # only in sim, ground-truth
|   |   |    |    |     └── ..jpg
│   │   │    |    └── depth.npz # depth images
│   │   ├── object_1.mp4 # video of sensor stream
│   │   └── data.pkl # proprioception data
│   └── ..
├── object_2/
│   └── ..
└── ..
```

## Citation

If you find NeuralFeels useful in your research, please consider citing our paper:

```bibtex
@article{suresh2024neuralfeels,
  title={{N}eural feels with neural fields: {V}isuo-tactile perception for in-hand manipulation},
  author={Suresh, Sudharshan and Qi, Haozhi and Wu, Tingfan and Fan, Taosha and Pineda, Luis and Lambeta, Mike and Malik, Jitendra and Kalakrishnan, Mrinal and Calandra, Roberto and Kaess, Michael and Ortiz, Joseph and Mukadam, Mustafa},
  journal={Science Robotics},
  pages={adl0628},
  year={2024},
  publisher={American Association for the Advancement of Science}
}
```
