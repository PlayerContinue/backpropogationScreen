//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains the functions required for backpropagation and feedforward algorithms
// utilizing the gpu
//----------------------------------------------------------------------------------------
#pragma once


#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <iostream>
#include "structures_cuda.cuh"

//Prototypes
size_t getCurrentGPUMemory();
int getWeights(SNeuronLayer currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int neuronPosition);
int setWeights(SNeuronLayer currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int startPosition, int neuronPosition);

//Function to free memory from GPU
template<class T> void free(T &V) {
	V.clear();
	V.shrink_to_fit();
}

template void free<thrust::device_vector<int> >(thrust::device_vector<int>& V);
template void free<thrust::device_vector<double> >(
	thrust::device_vector<double>& V);


//Functors

//Perform the operation for the functor
struct OutputDelta_functor
{
	OutputDelta_functor(){};

	//Overload the function operator
	__host__ __device__
		double operator()(const double &x, const double &y) const{
		//Output * (1-output) * (target-output)
		return (x * (1 - x) * (y - x));
	}

	//Overload the function operator
	__host__ __device__
		float operator()(const float &x, const float &y) const{
		//Output * (1-output) * (1-output)
		return (x * (1 - x) * (y - x));
	}
};

template <typename T>
struct dot_product_special
{


	dot_product_special(){};
	//Perform a matrix multiplication
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple t){
		return (thrust::get<0>(t) * thrust::get<1>(t));
	}


};

template <typename T>
struct saxpy_functor
{
	const double alpha;
	saxpy_functor(double _a) : alpha(_a){};

	__host__ __device__
		T operator()(T &x, T &y){
		return (x * alpha) + y;
	}


};




//Find the delta of the output layer
inline void findOutputDelta(thrust::host_vector<double> output, thrust::host_vector<double> &target){
	thrust::device_vector<double> X = output;
	thrust::device_vector<double> Y = target;

	//Transform the target vector
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), OutputDelta_functor());

	thrust::copy(Y.begin(), Y.end(), target.begin());
}

//Input sum - the sums of all the delta * weight, output - The output of the current layer
//output - results
//Perform secondary part of operation to find the new delta (output * (1- output) * sum
inline thrust::host_vector<double> findNewHiddenDelta(thrust::device_vector<double> sum, thrust::device_vector<double> output){

	thrust::host_vector<double> results(sum.size());

	thrust::transform(output.begin(), output.end(), sum.begin(), sum.end(), OutputDelta_functor());

	//Return the product as a host_copy
	thrust::copy(sum.begin(), sum.end(), results.begin());

	return results;
}

//input weights, delta
//Retrieve the deltas for the current layer
//output return_sums
inline thrust::host_vector<double> findHiddenDelta(SNeuronLayer neurons_weights, SNeuronLayer previous_layer){
	thrust::host_vector<double> return_sums(previous_layer.number_per_layer);//Sums to be returned 
	int size = neurons_weights.neurons.size();

	if (size > 0){
		thrust::device_vector<double> weights = neurons_weights.getWeights(0);//Stores the weights
		thrust::device_vector<double> deltas = neurons_weights.getDelta();//Store the Delta in GPU Memory
		thrust::device_vector<double> gpu_sums;//Sums in GPU, same number as delta

		//Fill the sums with 0
		//thrust::fill(gpu_sums.begin(), gpu_sums.end(), 0);

		//setup arguments
		dot_product_special<double> unary_op;
		thrust::plus<double> binary_op;

		//Perform the first iteration
		return_sums[0] = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(weights.begin(), deltas.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(weights.end(), deltas.end())), unary_op, 0, binary_op);


		//TODO - Modify to take advantage of the total memory available in the gpu
		//Sum the (weights) * delta of the neurons
		for (int i = 1; i < neurons_weights.number_per_layer; i++){
			weights = neurons_weights.getWeights(i);
			return_sums[i] = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(weights.begin(), deltas.begin())),
				thrust::make_zip_iterator(thrust::make_tuple(weights.end(), deltas.end())), unary_op, 0, binary_op);

		}

		free(weights);

		return findNewHiddenDelta(return_sums, previous_layer.getOutput());
	}
	else{
		throw new exception("No Nuerons");
	}
}





//Apply the learning momentum to all the weights in each layer
//Input CurrentLayer - the current layer of the neuron
//
inline void applyMomentum(SNeuronLayer currentLayer, double alpha){
	size_t availableMemory = getCurrentGPUMemory();

	//Get number of neurons which can fit in GPU memory
	int available_slots = ((availableMemory - 100000) / sizeof(double)) / ((currentLayer.neurons[0].weights.size() + 1) * 2);
	int size_available = 0;
	int numberNeurons = currentLayer.neurons.size();
	//Set the size of the arrays
	if (available_slots > numberNeurons){
		size_available = numberNeurons * (currentLayer.neurons[0].weights.size() + 1);
		available_slots = numberNeurons;
	}
	else{
		size_available = available_slots * (currentLayer.neurons[0].weights.size() + 1);
	}



	thrust::device_vector<double> weights(size_available);
	thrust::device_vector<double> previousWeights(size_available);


	for (int i = 0; i < numberNeurons; i += available_slots){
		//Store Weights and Bias in Memory
		getWeights(currentLayer, weights, previousWeights, i);

		//Transform the weights using saxpy
		thrust::transform(weights.begin(), weights.end(), previousWeights.begin(), weights.begin(), saxpy_functor<double>(alpha));

		//Set the weights back from memory
		setWeights(currentLayer, weights, previousWeights, i, i+available_slots);
	}

}

//Fill the weight bars
inline int getWeights(SNeuronLayer currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int neuronPosition){

	//Retrieve number of weights in the current layer
	int weightsPerNeuron = currentLayer.neurons[0].weights.size() + 1;
	int weightSize = weights.size();
	int weightPosition = neuronPosition * weightsPerNeuron;
	//Retrieve the current list of weights
	while (neuronPosition < currentLayer.number_per_layer && weightPosition < weightSize){

		thrust::copy(currentLayer.neurons[neuronPosition].weights.begin(),
			currentLayer.neurons[neuronPosition].weights.end(),
			weights.begin() + (weightPosition));
		
		thrust::copy(currentLayer.neurons[neuronPosition].previousWeight.begin(),
			currentLayer.neurons[neuronPosition].previousWeight.end(),
			previousWeight.begin() + (weightPosition));

		//Get the bias as well
		weights[weightPosition + weightsPerNeuron - 1] = currentLayer.neurons[neuronPosition].bias;
		previousWeight[weightPosition + weightsPerNeuron - 1] = currentLayer.neurons[neuronPosition].previousBias;

		weightPosition += weightsPerNeuron;
		neuronPosition += 1;

	}



	return (weightPosition / weightsPerNeuron);
}

inline int setWeights(SNeuronLayer currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int startPosition, int neuronPosition){
	int size = currentLayer.neurons[0].weights.size() + 1;
	startPosition = startPosition * size;
	neuronPosition = neuronPosition*(size);
	//Transfer the neurons back if it doesn't work
	for (int i = startPosition; i < neuronPosition; i += size){
		thrust::copy(weights.begin() + (i), weights.begin() + (i + size - 2), currentLayer.neurons[i / size].weights.begin());
		currentLayer.neurons[i / size].bias = weights[i+size - 1];
	}
	return 0;
}

inline size_t getCurrentGPUMemory(){
	size_t mem_tot;
	size_t mem_free;
	cudaMemGetInfo(&mem_free, &mem_tot);
	return mem_free;
}





