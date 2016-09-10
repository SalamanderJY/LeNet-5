
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "readubyte.h"
#include "kernel.cuh"

int main(int argc, char **argv)
{
	// 单通道灰度图
	size_t width, height, channels = 1;

	// 读取Mnist训练集及测试集数据
	printf("读取输入数据...\n");

	// 读取数据集大小
	size_t train_size = ReadUByteDataset(
		FLAGS_train_images.c_str(),
		FLAGS_train_labels.c_str(),
		nullptr,
		nullptr,
		width,
		height);
	size_t test_size = ReadUByteDataset(
		FLAGS_test_images.c_str(), 
		FLAGS_test_labels.c_str(), 
		nullptr, 
		nullptr, 
		width, 
		height);
	if (train_size == 0)
		return 1;

	std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
	std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

	// 读取数据集数据
	if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
		return 2;
	if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
		return 3;

	printf("Mnist训练集图片数量: %d, Mnist测试集图片数量: %d\n", (int)train_size, (int)test_size);
	printf("Batch size: %lld, iterations: %d\n", FLAGS_batch_size, FLAGS_iterations);

	// Choose GPU
	int num_gpus;
	//获取可用的gpu数量
	checkCudaErrors(cudaGetDeviceCount(&num_gpus));
	if (FLAGS_gpu < 0 || FLAGS_gpu >= num_gpus)
	{
		printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n",
			FLAGS_gpu, num_gpus);
		return 4;
	}

	//构造经典的LENET卷积神经网络
	//第一个卷积层 20个卷积核 卷积核是 5 * 5 
	ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
	//最大池化层的操作  定义了最大池化的大小和步长
	MaxPoolLayer pool1(2, 2);
	//第二个卷积层 50个卷积核 卷积核是 5 * 5 
	ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
	//最大池化层的操作  定义了最大池化的大小和步长
	MaxPoolLayer pool2(2, 2);
	//全连接层的定义 50 * 8 *8 / (2*2) = 800 个输入 , 500 个输出
	FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride),
		500);
	//全连接层的定义 500 个输入 , 10 个输出
	FullyConnectedLayer fc2(fc1.outputs, 10);

	// Initialize CUDNN/CUBLAS training context
	/*
	FLAGS_gpu ：       可用gpu的编号, 在这里是0号
	FLAGS_batch_size : 每次批量训练的数量
	*/
	TrainingContext context(FLAGS_gpu, FLAGS_batch_size, conv1, pool1, conv2, pool2, fc1, fc2);// Initialize CUDNN/CUBLAS training context
	
	// Determine initial network structure
	// 此处并没有利用cudnn样例里已经训练好的模型，采用随机化初始整个网络
	if (FLAGS_pretrained)
	{
		conv1.FromFile("conv1");
		conv2.FromFile("conv2");
		fc1.FromFile("ip1");
		fc2.FromFile("ip2");
	}
	else
	{
		// Create random network
		// 随机初始化整个网络
		std::random_device rd;
		std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed));

		// Xavier weight filling
		float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
		std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
		float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
		std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
		float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
		std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
		float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
		std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

		// Randomize network
		for (auto&& iter : conv1.pconv)
			iter = static_cast<float>(dconv1(gen));
		for (auto&& iter : conv1.pbias)
			iter = static_cast<float>(dconv1(gen));
		for (auto&& iter : conv2.pconv)
			iter = static_cast<float>(dconv2(gen));
		for (auto&& iter : conv2.pbias)
			iter = static_cast<float>(dconv2(gen));
		for (auto&& iter : fc1.pneurons)
			iter = static_cast<float>(dfc1(gen));
		for (auto&& iter : fc1.pbias)
			iter = static_cast<float>(dfc1(gen));
		for (auto&& iter : fc2.pneurons)
			iter = static_cast<float>(dfc2(gen));
		for (auto&& iter : fc2.pbias)
			iter = static_cast<float>(dfc2(gen));
	}

	/////////////////////////////////////////////////////////////////////////////
	//下面给所有需要用到的数据分配cuda的内存
	// Create GPU data structures    
	// d_data  训练的图像原始数据指针
	// d_label 训练的图像的label的数据
	// d_conv1 第一次卷积后的输出
	// d_pool1,d_conv2...... 同上
	// d_fc1relu 第一个全连接层的激活函数 使用的是relu（相关函数）
	// d_fc2smax 第二个全连接层的激活函数 softmax 
	// Forward propagation data
	float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
	//                         Buffer    | Element       | N                   | C                  | H                                 | W
	//-----------------------------------------------------------------------------------------------------------------------------------------
	checkCudaErrors(cudaMalloc(&d_data, sizeof(float)* context.m_batchSize * channels           * height                            * width));
	checkCudaErrors(cudaMalloc(&d_labels, sizeof(float)* context.m_batchSize * 1 * 1 * 1));
	checkCudaErrors(cudaMalloc(&d_conv1, sizeof(float)* context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
	checkCudaErrors(cudaMalloc(&d_pool1, sizeof(float)* context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
	checkCudaErrors(cudaMalloc(&d_conv2, sizeof(float)* context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
	checkCudaErrors(cudaMalloc(&d_pool2, sizeof(float)* context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
	checkCudaErrors(cudaMalloc(&d_fc1, sizeof(float)* context.m_batchSize * fc1.outputs));
	checkCudaErrors(cudaMalloc(&d_fc1relu, sizeof(float)* context.m_batchSize * fc1.outputs));
	checkCudaErrors(cudaMalloc(&d_fc2, sizeof(float)* context.m_batchSize * fc2.outputs));
	checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float)* context.m_batchSize * fc2.outputs));

	// Network parameters
	// d_pconv1     第一个卷积层的卷积核的数据
	// d_pconv1bias 第一个卷积层的偏置
	// d_pfc1       第一个全连接层的参数
	// d_pfc1bias   第一个全连接层的偏置
	float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
	float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;

	checkCudaErrors(cudaMalloc(&d_pconv1, sizeof(float)* conv1.pconv.size()));
	checkCudaErrors(cudaMalloc(&d_pconv1bias, sizeof(float)* conv1.pbias.size()));
	checkCudaErrors(cudaMalloc(&d_pconv2, sizeof(float)* conv2.pconv.size()));
	checkCudaErrors(cudaMalloc(&d_pconv2bias, sizeof(float)* conv2.pbias.size()));
	checkCudaErrors(cudaMalloc(&d_pfc1, sizeof(float)* fc1.pneurons.size()));
	checkCudaErrors(cudaMalloc(&d_pfc1bias, sizeof(float)* fc1.pbias.size()));
	checkCudaErrors(cudaMalloc(&d_pfc2, sizeof(float)* fc2.pneurons.size()));
	checkCudaErrors(cudaMalloc(&d_pfc2bias, sizeof(float)* fc2.pbias.size()));

	// Network parameter gradients
	// d_gconv1     第一层卷积层的卷积核的梯度
	// d_gconv1bias 第一层卷积层的偏置的梯度
	float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
	float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;

	checkCudaErrors(cudaMalloc(&d_gconv1, sizeof(float)* conv1.pconv.size()));
	checkCudaErrors(cudaMalloc(&d_gconv1bias, sizeof(float)* conv1.pbias.size()));
	checkCudaErrors(cudaMalloc(&d_gconv2, sizeof(float)* conv2.pconv.size()));
	checkCudaErrors(cudaMalloc(&d_gconv2bias, sizeof(float)* conv2.pbias.size()));
	checkCudaErrors(cudaMalloc(&d_gfc1, sizeof(float)* fc1.pneurons.size()));
	checkCudaErrors(cudaMalloc(&d_gfc1bias, sizeof(float)* fc1.pbias.size()));
	checkCudaErrors(cudaMalloc(&d_gfc2, sizeof(float)* fc2.pneurons.size()));
	checkCudaErrors(cudaMalloc(&d_gfc2bias, sizeof(float)* fc2.pbias.size()));

	// Differentials w.r.t. data
	float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
	//                         Buffer     | Element       | N                   | C                  | H                                 | W
	//-----------------------------------------------------------------------------------------------------------------------------------------
	checkCudaErrors(cudaMalloc(&d_dpool1, sizeof(float)* context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
	checkCudaErrors(cudaMalloc(&d_dpool2, sizeof(float)* context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
	checkCudaErrors(cudaMalloc(&d_dconv2, sizeof(float)* context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
	checkCudaErrors(cudaMalloc(&d_dfc1, sizeof(float)* context.m_batchSize * fc1.inputs));
	checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float)* context.m_batchSize * fc1.outputs));
	checkCudaErrors(cudaMalloc(&d_dfc2, sizeof(float)* context.m_batchSize * fc2.inputs));
	checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float)* context.m_batchSize * fc2.outputs));
	checkCudaErrors(cudaMalloc(&d_dlossdata, sizeof(float)* context.m_batchSize * fc2.outputs));

	// Temporary buffers and workspaces
	float *d_onevec;
	void *d_cudnn_workspace = nullptr;
	checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float)* context.m_batchSize));
	if (context.m_workspaceSize > 0)
		checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));

	// Copy initial network to device
	// 将初始化的网络加载到显卡显存上
	checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0], sizeof(float)* conv1.pconv.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float)* conv1.pbias.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pconv[0], sizeof(float)* conv2.pconv.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float)* conv2.pbias.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0], sizeof(float)* fc1.pneurons.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0], sizeof(float)* fc1.pbias.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0], sizeof(float)* fc2.pneurons.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0], sizeof(float)* fc2.pbias.size(), cudaMemcpyHostToDevice));

	// Fill one-vector with ones
	FillOnes << <RoundUp(context.m_batchSize, BW), BW >> >(d_onevec, context.m_batchSize);

	printf("准备将数据集加载到显存...\n");

	// Normalize training set to be in [0,1]
	// 载入训练数据
	std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
	for (size_t i = 0; i < train_size * channels * width * height; ++i)
		train_images_float[i] = (float)train_images[i] / 255.0f;

	for (size_t i = 0; i < train_size; ++i)
		train_labels_float[i] = (float)train_labels[i];

	printf("训练中...\n");

	// 使用随机梯度下降的方法训练网络
	checkCudaErrors(cudaDeviceSynchronize());
	auto t1 = std::chrono::high_resolution_clock::now();
	//训练的次数FLAGS_iterations
	for (int iter = 0; iter < FLAGS_iterations; ++iter)
	{
		// Train
		int imageid = iter % (train_size / context.m_batchSize);

		// Prepare current batch on device
		checkCudaErrors(cudaMemcpyAsync(d_data, &train_images_float[imageid * context.m_batchSize * width*height*channels],
			sizeof(float)* context.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * context.m_batchSize],
			sizeof(float)* context.m_batchSize, cudaMemcpyHostToDevice));

		// Forward propagation
		context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
			d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
			d_cudnn_workspace, d_onevec);

		// Backward propagation
		context.Backpropagation(conv1, pool1, conv2, pool2,
			d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
			d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
			d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias,
			d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);

		// Compute learning rate
		float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));

		// Update weights
		context.UpdateWeights(learningRate, conv1, conv2,
			d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
			d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = std::chrono::high_resolution_clock::now();

	printf("Every iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);

	float classification_error = 1.0f;

	int classifications = FLAGS_classify;
	if (classifications < 0)
		classifications = (int)test_size;

	// Test the resulting neural network's classification
	if (classifications > 0)
	{
		// Initialize a TrainingContext structure for testing (different batch size)
		TrainingContext test_context(FLAGS_gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);

		// Ensure correct workspaceSize is allocated for testing
		if (context.m_workspaceSize < test_context.m_workspaceSize)
		{
			checkCudaErrors(cudaFree(d_cudnn_workspace));
			checkCudaErrors(cudaMalloc(&d_cudnn_workspace, test_context.m_workspaceSize));
		}

		int num_errors = 0;
		for (int i = 0; i < classifications; ++i)
		{
			std::vector<float> data(width * height);
			// Normalize image to be in [0,1]
			for (int j = 0; j < width * height; ++j)
				data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;

			checkCudaErrors(cudaMemcpyAsync(d_data, &data[0], sizeof(float)* width * height, cudaMemcpyHostToDevice));

			// Forward propagate test image
			test_context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
				d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,
				d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);

			// Perform classification
			std::vector<float> class_vec(10);

			// Copy back result
			checkCudaErrors(cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float)* 10, cudaMemcpyDeviceToHost));

			// Determine classification according to maximal response
			int chosen = 0;
			for (int id = 1; id < 10; ++id)
			{
				if (class_vec[chosen] < class_vec[id]) chosen = id;
			}

			if (chosen != test_labels[i])
				++num_errors;
		}
		classification_error = (float)num_errors / (float)classifications;

		printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
		int finish;
		std::cin >> finish;
	}

	// Free data structures
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_conv1));
	checkCudaErrors(cudaFree(d_pool1));
	checkCudaErrors(cudaFree(d_conv2));
	checkCudaErrors(cudaFree(d_pool2));
	checkCudaErrors(cudaFree(d_fc1));
	checkCudaErrors(cudaFree(d_fc2));
	checkCudaErrors(cudaFree(d_pconv1));
	checkCudaErrors(cudaFree(d_pconv1bias));
	checkCudaErrors(cudaFree(d_pconv2));
	checkCudaErrors(cudaFree(d_pconv2bias));
	checkCudaErrors(cudaFree(d_pfc1));
	checkCudaErrors(cudaFree(d_pfc1bias));
	checkCudaErrors(cudaFree(d_pfc2));
	checkCudaErrors(cudaFree(d_pfc2bias));
	checkCudaErrors(cudaFree(d_gconv1));
	checkCudaErrors(cudaFree(d_gconv1bias));
	checkCudaErrors(cudaFree(d_gconv2));
	checkCudaErrors(cudaFree(d_gconv2bias));
	checkCudaErrors(cudaFree(d_gfc1));
	checkCudaErrors(cudaFree(d_gfc1bias));
	checkCudaErrors(cudaFree(d_dfc1));
	checkCudaErrors(cudaFree(d_gfc2));
	checkCudaErrors(cudaFree(d_gfc2bias));
	checkCudaErrors(cudaFree(d_dfc2));
	checkCudaErrors(cudaFree(d_dpool1));
	checkCudaErrors(cudaFree(d_dconv2));
	checkCudaErrors(cudaFree(d_dpool2));
	checkCudaErrors(cudaFree(d_labels));
	checkCudaErrors(cudaFree(d_dlossdata));
	checkCudaErrors(cudaFree(d_onevec));
	if (d_cudnn_workspace != nullptr)
		checkCudaErrors(cudaFree(d_cudnn_workspace));

	return 0;
}