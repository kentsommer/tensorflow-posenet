# TensorFlow-PoseNet
**This is an implementation for TensorFlow of the [PoseNet architecture](http://mi.eng.cam.ac.uk/projects/relocalisation/)**

As described in the ICCV 2015 paper **PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization** Alex Kendall, Matthew Grimes and Roberto Cipolla [http://mi.eng.cam.ac.uk/projects/relocalisation/]

## Getting Started

 * Download the Cambridge Landmarks King's College dataset [from here.](https://www.repository.cam.ac.uk/handle/1810/251342)

 * Download the starting weights [from here.](https://github.com/kentsommer/tensorflow-posenet/releases/download/1.0.0/posenet.npy)

 * The PoseNet model is defined in the posenet.py file

 * The starting weights (posenet.npy) for training were obtained by converting caffemodel weights [from here](http://3dvision.princeton.edu/pvt/GoogLeNet/Places/).

 * To run:
   * Extract the King's College dataset to wherever you prefer
   * Extract the starting weights
   * To train, simply run train.py after setting the [path to the King's College dataset](https://github.com/kentsommer/tensorflow-posenet/blob/master/train.py#L13-L14) (note this will take a long time)
   * After training, to test run test.py after updating the paths on [these lines](https://github.com/kentsommer/tensorflow-posenet/blob/master/test.py#L17-L18)
