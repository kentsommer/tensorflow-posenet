# TensorFlow-PoseNet
**This is an implementation for TensorFlow of the [PoseNet architecture](http://mi.eng.cam.ac.uk/projects/relocalisation/)**

As described in the ICCV 2015 paper **PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization** Alex Kendall, Matthew Grimes and Roberto Cipolla [http://mi.eng.cam.ac.uk/projects/relocalisation/]

## Getting Started

 * Download the Cambridge Landmarks King's College dataset [from here.](https://www.repository.cam.ac.uk/handle/1810/251342)

 * Download the starting and trained weights [from here.](https://drive.google.com/file/d/0B5DVPd_zGgc8ZmJ0VmNiTXBGUkU/view?usp=sharing)

 * The PoseNet model is defined in the posenet.py file

 * The starting and trained weights (posenet.npy and PoseNet.ckpt respectively) for training were obtained by converting caffemodel weights [from here](http://3dvision.princeton.edu/pvt/GoogLeNet/Places/) and then training.

 * To run:
   * Extract the King's College dataset to wherever you prefer
   * Extract the starting and trained weights to wherever you prefer
   * Update the paths on line 13 (train.py) as well as lines 15 and 17 (test.py)
   * If you want to retrain, simply run train.py (note this will take a long time)
   * If you just want to test, simply run test.py 
