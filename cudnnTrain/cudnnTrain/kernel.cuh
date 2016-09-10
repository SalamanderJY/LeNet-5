
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <random>
#include <sstream>

#include <cublas_v2.h>
#include <cudnn.h>

#include "readubyte.h"

// Definition and helper utilities

// Block width for CUDA kernels
#define BW 128

// Constant versions of gflags
#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))

/**
* Computes ceil(x / y) for integral nonnegative values.
*/
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}

/**
* 保存为无符号8bit的PGM灰度图
* Saves a PGM grayscale image out of unsigned 8-bit data
*/
void SavePGMFile(const unsigned char *data, size_t width, size_t height, const char *filename)
{
	FILE *fp = fopen(filename, "wb");
	if (fp)
	{
		fprintf(fp, "P5\n%lu %lu\n255\n", width, height);
		fwrite(data, sizeof(unsigned char), width * height, fp);
		fclose(fp);
	}
}

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
	std::stringstream _where, _message;                                \
	_where << __FILE__ << ':' << __LINE__;                             \
	_message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
	std::cerr << _message.str() << "\nAborting...\n";                  \
	cudaDeviceReset();                                                 \
	exit(1);                                                           \
} while (0)

#define checkCUDNN(status) do {                                        \
	std::stringstream _error;                                          \
	if (status != CUDNN_STATUS_SUCCESS) {                              \
		_error << "CUDNN failure: " << cudnnGetErrorString(status);    \
		FatalError(_error.str());                                      \
	}                                                                  \
} while (0)

#define checkCudaErrors(status) do {                                   \
	std::stringstream _error;                                          \
	if (status != 0) {                                                 \
		_error << "Cuda failure: " << status;                          \
		FatalError(_error.str());                                      \
	}                                                                  \
} while (0)

// Command-line flags

// Application parameters
DEFINE_int32(gpu, 0, "The GPU ID to use");
DEFINE_int32(iterations, 10000, "Number of iterations for training");
DEFINE_int32(random_seed, -1, "Override random seed (default uses std::random_device)");
DEFINE_int32(classify, -1, "Number of images to classify to compute error rate (default uses entire test set)");

// Batch parameters
DEFINE_uint64(batch_size, 64, "Batch size for training");

// Filenames
DEFINE_bool(pretrained, false, "Use the pretrained CUDNN model as input");
DEFINE_string(train_images, "train-images.idx3-ubyte", "Training images filename");
DEFINE_string(train_labels, "train-labels.idx1-ubyte", "Training labels filename");
DEFINE_string(test_images, "t10k-images.idx3-ubyte", "Test images filename");
DEFINE_string(test_labels, "t10k-labels.idx1-ubyte", "Test labels filename");

// Solver parameters
DEFINE_double(learning_rate, 0.01, "Base learning rate");
DEFINE_double(lr_gamma, 0.0001, "Learning rate policy gamma");
DEFINE_double(lr_power, 0.75, "Learning rate policy power");


// Layer representations
/**
* Represents a convolutional layer with bias.
*/
struct ConvBiasLayer
{
	int in_channels, out_channels, kernel_size;
	int in_width, in_height, out_width, out_height;

	std::vector<float> pconv, pbias;

	ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_,
		int in_w_, int in_h_) : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_),
		pbias(out_channels_)
	{
		in_channels = in_channels_;
		out_channels = out_channels_;
		kernel_size = kernel_size_;
		in_width = in_w_;
		in_height = in_h_;
		out_width = in_w_ - kernel_size_ + 1;
		out_height = in_h_ - kernel_size_ + 1;
	}

	void FromFile(const char *fileprefix)
	{
		std::stringstream ssf, ssbf;
		ssf << fileprefix << ".bin";
		ssbf << fileprefix << ".bias.bin";

		// Read weights file
		FILE *fp = fopen(ssf.str().c_str(), "rb");
		if (!fp)
		{
			printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
			exit(2);
		}
		fread(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
		fclose(fp);

		// Read bias file
		fp = fopen(ssbf.str().c_str(), "rb");
		if (!fp)
		{
			printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
			exit(2);
		}
		fread(&pbias[0], sizeof(float), out_channels, fp);
		fclose(fp);
	}
};

/**
* Represents a max-pooling layer.
*/
struct MaxPoolLayer
{
	int size, stride;
	MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

/**
* Represents a fully-connected neural network layer with bias.
*/
struct FullyConnectedLayer
{
	int inputs, outputs;
	std::vector<float> pneurons, pbias;

	FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
		pneurons(inputs_ * outputs_), pbias(outputs_) {}

	void FromFile(const char *fileprefix)
	{
		std::stringstream ssf, ssbf;
		ssf << fileprefix << ".bin";
		ssbf << fileprefix << ".bias.bin";

		// Read weights file
		FILE *fp = fopen(ssf.str().c_str(), "rb");
		if (!fp)
		{
			printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
			exit(2);
		}
		fread(&pneurons[0], sizeof(float), inputs * outputs, fp);
		fclose(fp);

		// Read bias file
		fp = fopen(ssbf.str().c_str(), "rb");
		if (!fp)
		{
			printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
			exit(2);
		}
		fread(&pbias[0], sizeof(float), outputs, fp);
		fclose(fp);
	}
};

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
* Fills a floating-point array with ones.
*
* @param vec The array to fill.
* @param size The number of elements in the array.
*/
__global__ void FillOnes(float *vec, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	vec[idx] = 1.0f;
}

/**
* Computes the backpropagation results of the Softmax loss for each result in a batch.
* Uses the softmax values obtained from forward propagation to compute the difference.
*
* @param label The training batch label values.
* @param num_labels The number of possible labels.
* @param batch_size The size of the trained batch.
* @param diff The resulting gradient.
*/
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;

	const int label_value = static_cast<int>(label[idx]);

	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context

struct TrainingContext
{
	cudnnHandle_t cudnnHandle;

	cublasHandle_t cublasHandle;

	cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor,
		conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
	cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
	cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
	cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
	cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
	cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
	cudnnPoolingDescriptor_t poolDesc;
	cudnnActivationDescriptor_t fc1Activation;

	int m_gpuid;
	int m_batchSize;
	size_t m_workspaceSize;

	FullyConnectedLayer& ref_fc1, &ref_fc2;

	// Disable copying
	TrainingContext& operator=(const TrainingContext&) = delete;
	TrainingContext(const TrainingContext&) = delete;

	TrainingContext(int gpuid, int batch_size,
		ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
		FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid)
	{
		m_batchSize = batch_size;

		// Create CUBLAS and CUDNN handles
		checkCudaErrors(cudaSetDevice(gpuid));
		checkCudaErrors(cublasCreate(&cublasHandle));
		checkCUDNN(cudnnCreate(&cudnnHandle));

		// Create tensor descriptors
		checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));

		checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));

		checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
		checkCUDNN(cudnnCreateFilterDescriptor(&conv2filterDesc));

		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));

		checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));


		// Set tensor descriptor sizes
		checkCUDNN(cudnnSetTensor4dDescriptor(conv1BiasTensor,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			1, conv1.out_channels,
			1, 1));
		checkCUDNN(cudnnSetTensor4dDescriptor(conv2BiasTensor,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			1, conv2.out_channels,
			1, 1));

		checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
			CUDNN_POOLING_MAX,
			CUDNN_PROPAGATE_NAN,
			pool1.size, pool1.size,
			0, 0,
			pool1.stride, pool1.stride));
		checkCUDNN(cudnnSetTensor4dDescriptor(pool2Tensor,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batch_size, conv2.out_channels,
			conv2.out_height / pool2.stride,
			conv2.out_width / pool2.stride));

		checkCUDNN(cudnnSetTensor4dDescriptor(fc1Tensor,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batch_size, fc1.outputs, 1, 1));

		checkCUDNN(cudnnSetTensor4dDescriptor(fc2Tensor,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batch_size, fc2.outputs, 1, 1));

		checkCUDNN(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
			CUDNN_PROPAGATE_NAN, 0.0));


		// Set convolution tensor sizes and compute workspace size
		size_t workspace = 0;
		workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
		workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

		workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
		workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

		// The workspace is allocated later (if necessary)
		m_workspaceSize = workspace;
	}

	~TrainingContext()
	{
		checkCudaErrors(cudaSetDevice(m_gpuid));

		checkCudaErrors(cublasDestroy(cublasHandle));
		checkCUDNN(cudnnDestroy(cudnnHandle));
		checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv1BiasTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(pool1Tensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv2Tensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(pool2Tensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
		checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
		checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
		checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
		checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
	}

	size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
		cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
		cudnnConvolutionFwdAlgo_t& algo)
	{
		size_t sizeInBytes = 0;

		int n = m_batchSize;
		int c = conv.in_channels;
		int h = conv.in_height;
		int w = conv.in_width;

		checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c,
			h, w));

		checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NCHW,
			conv.out_channels,
			conv.in_channels,
			conv.kernel_size,
			conv.kernel_size));

		checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
			0, 0,
			1, 1,
			1, 1,
			CUDNN_CROSS_CORRELATION));
		// Find dimension of convolution output
		checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
			srcTensorDesc,
			filterDesc,
			&n, &c, &h, &w));

		checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c,
			h, w));
		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
			srcTensorDesc,
			filterDesc,
			convDesc,
			dstTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			0,
			&algo));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
			srcTensorDesc,
			filterDesc,
			convDesc,
			dstTensorDesc,
			algo,
			&sizeInBytes));

		return sizeInBytes;
	}

	void ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
		float *fc2, float *result,
		float *pconv1, float *pconv1bias,
		float *pconv2, float *pconv2bias,
		float *pfc1, float *pfc1bias,
		float *pfc2, float *pfc2bias, void *workspace, float *onevec)
	{
		float alpha = 1.0f, beta = 0.0f;
		checkCudaErrors(cudaSetDevice(m_gpuid));

		// Conv1 layer
		checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
			data, conv1filterDesc, pconv1, conv1Desc,
			conv1algo, workspace, m_workspaceSize, &beta,
			conv1Tensor, conv1));
		checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,
			pconv1bias, &alpha, conv1Tensor, conv1));

		// Pool1 layer
		checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
			conv1, &beta, pool1Tensor, pool1));

		// Conv2 layer
		checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
			pool1, conv2filterDesc, pconv2, conv2Desc,
			conv2algo, workspace, m_workspaceSize, &beta,
			conv2Tensor, conv2));
		checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
			pconv2bias, &alpha, conv2Tensor, conv2));

		// Pool2 layer
		checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
			conv2, &beta, pool2Tensor, pool2));

		// FC1 layer
		// Forward propagate neurons using weights (fc1 = pfc1'*pool2)
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			ref_fc1.outputs, m_batchSize, ref_fc1.inputs,
			&alpha,
			pfc1, ref_fc1.inputs,
			pool2, ref_fc1.inputs,
			&beta,
			fc1, ref_fc1.outputs));
		// Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			ref_fc1.outputs, m_batchSize, 1,
			&alpha,
			pfc1bias, ref_fc1.outputs,
			onevec, 1,
			&alpha,
			fc1, ref_fc1.outputs));

		// ReLU activation
		checkCUDNN(cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
			fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));

		// FC2 layer
		// Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			ref_fc2.outputs, m_batchSize, ref_fc2.inputs,
			&alpha,
			pfc2, ref_fc2.inputs,
			fc1relu, ref_fc2.inputs,
			&beta,
			fc2, ref_fc2.outputs));
		// Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			ref_fc2.outputs, m_batchSize, 1,
			&alpha,
			pfc2bias, ref_fc2.outputs,
			onevec, 1,
			&alpha,
			fc2, ref_fc2.outputs));

		// Softmax loss
		checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
	}

	size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
		cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
		cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
	{
		size_t sizeInBytes = 0, tmpsize = 0;

		// If backprop filter algorithm was requested
		if (falgo)
		{
			checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
				cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
				CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo));

			checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
				cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
				*falgo, &tmpsize));

			sizeInBytes = std::max(sizeInBytes, tmpsize);
		}

		// If backprop data algorithm was requested
		if (dalgo)
		{
			checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
				cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo));

			checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
				cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
				*dalgo, &tmpsize));

			sizeInBytes = std::max(sizeInBytes, tmpsize);
		}

		return sizeInBytes;
	}

	void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
		float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
		float *fc2, float *fc2smax, float *dloss_data,
		float *pconv1, float *pconv1bias,
		float *pconv2, float *pconv2bias,
		float *pfc1, float *pfc1bias,
		float *pfc2, float *pfc2bias,
		float *gconv1, float *gconv1bias, float *dpool1,
		float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
		float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
		float *gfc2, float *gfc2bias, float *dfc2,
		void *workspace, float *onevec)
	{
		float alpha = 1.0f, beta = 0.0f;

		float scalVal = 1.0f / static_cast<float>(m_batchSize);

		checkCudaErrors(cudaSetDevice(m_gpuid));

		// Initialization (using the training error function)
		checkCudaErrors(cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float)* m_batchSize * ref_fc2.outputs, cudaMemcpyDeviceToDevice));

		// Softmax layer TODO
		SoftmaxLossBackprop << < RoundUp(m_batchSize, BW), BW >> >(labels, ref_fc2.outputs, m_batchSize, dloss_data);

		// Accounting for batch size in SGD
		checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));

		// FC2 layer
		// Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchSize,
			&alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs));
		// Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
		checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize,
			&alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
		// Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, m_batchSize, ref_fc2.outputs,
			&alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));

		// ReLU activation
		checkCUDNN(cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
			fc1Tensor, fc1relu, fc1Tensor, dfc2,
			fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));

		// FC1 layer
		// Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchSize,
			&alpha, pool2, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
		// Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
		checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize,
			&alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
		// Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, m_batchSize, ref_fc1.outputs,
			&alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));

		// Pool2 layer
		checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
			pool2Tensor, pool2, pool2Tensor, dfc1,
			conv2Tensor, conv2, &beta, conv2Tensor, dpool2));

		// Conv2 layer
		checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
			dpool2, &beta, conv2BiasTensor, gconv2bias));


		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,
			pool1, conv2Tensor, dpool2, conv2Desc,
			conv2bwfalgo, workspace, m_workspaceSize,
			&beta, conv2filterDesc, gconv2));

		checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,
			pconv2, conv2Tensor, dpool2, conv2Desc,
			conv2bwdalgo, workspace, m_workspaceSize,
			&beta, pool1Tensor, dconv2));

		// Pool1 layer
		checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
			pool1Tensor, pool1, pool1Tensor, dconv2,
			conv1Tensor, conv1, &beta, conv1Tensor, dpool1));

		// Conv1 layer
		checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
			dpool1, &beta, conv1BiasTensor, gconv1bias));

		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
			data, conv1Tensor, dpool1, conv1Desc,
			conv1bwfalgo, workspace, m_workspaceSize,
			&beta, conv1filterDesc, gconv1));

		// No need for convBackwardData because there are no more layers below
	}

	void UpdateWeights(float learning_rate,
		ConvBiasLayer& conv1, ConvBiasLayer& conv2,
		float *pconv1, float *pconv1bias,
		float *pconv2, float *pconv2bias,
		float *pfc1, float *pfc1bias,
		float *pfc2, float *pfc2bias,
		float *gconv1, float *gconv1bias,
		float *gconv2, float *gconv2bias,
		float *gfc1, float *gfc1bias,
		float *gfc2, float *gfc2bias)
	{
		float alpha = -learning_rate;

		checkCudaErrors(cudaSetDevice(m_gpuid));

		// Conv1
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
			&alpha, gconv1, 1, pconv1, 1));
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
			&alpha, gconv1bias, 1, pconv1bias, 1));

		// Conv2
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
			&alpha, gconv2, 1, pconv2, 1));
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
			&alpha, gconv2bias, 1, pconv2bias, 1));

		// Fully connected 1
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
			&alpha, gfc1, 1, pfc1, 1));
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
			&alpha, gfc1bias, 1, pfc1bias, 1));

		// Fully connected 2
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
			&alpha, gfc2, 1, pfc2, 1));
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
			&alpha, gfc2bias, 1, pfc2bias, 1));
	}
};