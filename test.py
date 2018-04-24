# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from posenet import GoogLeNet as PoseNet
import cv2
from tqdm import tqdm
import math

batch_size = 75
max_iterations = 30000

# Set this path to your project directory
path = 'path_to_project/'
# Set this path to your dataset directory
directory = 'path_to_datasets/KingsCollege/'
dataset = 'dataset_test.txt'

class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

class vecsource(object):
	def __init__(self, vecs, poses):
		self.vecs = vecs
		self.poses = poses

def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = (new_height - output_side_length) / 2
	width_offset = (new_width - output_side_length) / 2
	cropped_img = img[height_offset:height_offset + output_side_length,
						width_offset:width_offset + output_side_length]
	return cropped_img

def preprocess(images):
	images_out = [] #final result
	#Resize and crop and compute mean!
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		X = cv2.resize(X, (455, 256))
		X = centeredCrop(X, 224)
		images_cropped.append(X)
	#compute images mean
	N = 0
	mean = np.zeros((1, 3, 224, 224))
	for X in tqdm(images_cropped):
                X = np.transpose(X,(2,0,1)) 
		mean[0][0] += X[:,:,0]
		mean[0][1] += X[:,:,1]
		mean[0][2] += X[:,:,2]
		N += 1
	mean[0] /= N
	#Subtract mean from all images
	for X in tqdm(images_cropped):
		X = np.transpose(X,(2,0,1))
		X = X - mean
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		Y = np.expand_dims(X, axis=0)
		images_out.append(Y)
	return images_out

def get_data():
	poses = []
	images = []

	with open(directory+dataset) as f:
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			p4 = float(p4)
			p5 = float(p5)
			p6 = float(p6)
			poses.append((p0,p1,p2,p3,p4,p5,p6))
			images.append(directory+fname)
	images = preprocess(images)
	return datasource(images, poses)

def gen_data(source):
	while True:
		indices = range(len(source.images))
		random.shuffle(indices)
		for i in indices:
			image = source.images[i]
			pose_x = source.poses[i][0:3]
			pose_q = source.poses[i][3:7]
			yield image, pose_x, pose_q

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)

def main():
	image = tf.placeholder(tf.float32, [1, 224, 224, 3])
	datasource = get_data()
	results = np.zeros((len(datasource.images),2))

	net = PoseNet({'data': image})

	p3_x = net.layers['cls3_fc_pose_xyz']
	p3_q = net.layers['cls3_fc_pose_wpqr']

	init = tf.initialize_all_variables()
	outputFile = "PoseNet.ckpt"

	saver = tf.train.Saver()

	with tf.Session() as sess:
		# Load the data
		sess.run(init)
		saver.restore(sess, path + 'PoseNet.ckpt')

		data_gen = gen_data_batch(datasource)
		for i in range(len(datasource.images)):
			np_image = datasource.images[i]
			feed = {image: np_image}


			pose_q= np.asarray(datasource.poses[i][3:7])
			pose_x= np.asarray(datasource.poses[i][0:3])
			predicted_x, predicted_q = sess.run([p3_x, p3_q], feed_dict=feed)

			pose_q = np.squeeze(pose_q)
			pose_x = np.squeeze(pose_x)
			predicted_q = np.squeeze(predicted_q)
			predicted_x = np.squeeze(predicted_x)

			#Compute Individual Sample Error
			q1 = pose_q / np.linalg.norm(pose_q)
			q2 = predicted_q / np.linalg.norm(predicted_q)
			d = abs(np.sum(np.multiply(q1,q2)))
			theta = 2 * np.arccos(d) * 180/math.pi
			error_x = np.linalg.norm(pose_x-predicted_x)
			results[i,:] = [error_x,theta]
			print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

	median_result = np.median(results,axis=0)
	print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'

if __name__ == '__main__':
	main()
