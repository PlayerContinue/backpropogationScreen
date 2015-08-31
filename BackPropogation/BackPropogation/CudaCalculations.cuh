//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains the functions required for backpropagation and feedforward algorithms
// utilizing the gpu
//----------------------------------------------------------------------------------------
#pragma once


#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/complex.h>
#include <iostream>
#include "structures_cuda.cuh"

using namespace thrust::placeholders;

//Prototypes
size_t getCurrentGPUMemory();
int getWeights(SNeuronLayer currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int neuronPosition);
int getWeights(SNeuronLayer currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int neuronPosition, bool getPrevious, bool getBias);


int setWeights(SNeuronLayer &currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int startPosition, int neuronPosition);
int setWeights(SNeuronLayer &currentLayer, thrust::device_vector<double> &weights,
	thrust::device_vector<double> &previousWeight, int startPosition, int neuronPosition, bool include_previous);

namespace vector_free{
	//Function to free memory from GPU
	template<class T> void free(T &V) {
		V.clear();
		V.shrink_to_fit();
	}

	template void free<thrust::device_vector<int> >(thrust::device_vector<int>& V);
	template void free<thrust::device_vector<double> >(
		thrust::device_vector<double>& V);

}

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
struct dot_product_special : public unary_function < T, T >
{


	dot_product_special(){};


	//Overload the function operator
	__host__ __device__
		T operator()(const T &x, const T &y) const{
		//Output * (1-output) * (sum)
		return (x * (1 - x) * y);
	}

};



template <typename T>
struct apply_correction
{
	const T beta;
	apply_correction(double _b) : beta(_b){};

	template <typename Tuple>
	__host__ __device__
		void operator()(Tuple &t){
		thrust::get<1>(t) = beta * thrust::get<2>(t) * thrust::get<3>(t);
		thrust::get<0>(t) += thrust::get<1>(t);
	}
};


template <typename T>
struct saxpy_functor
{
	const T alpha;
	saxpy_functor(T _a) : alpha(_a){};

	__host__ __device__
		T operator()(const T &x, const  T &y)const{
		return (x * alpha) + y;
	}


};


template <typename T>
struct sigmoid_functor : public unary_function < T, T > {
	sigmoid_functor(){};

	__host__ __device__
		thrust::complex<T> operator()(const thrust::complex<T> &x) const{
		thrust::complex<T> z = ((thrust::complex<T>) - 1) * x;
		z = ((thrust::complex<T>)1 / ((thrust::complex<T>)1 + thrust::exp(z)));
		return z;
	}

};

template <typename T>
struct square_means_sum_functor : public unary_function < T, T > {
	square_means_sum_functor(){};
	
	template <typename Tuple>
	__host__ __device__
	T operator()(Tuple &x){
		return ((thrust::get<0>(x) - thrust::get<1>(x)) * (thrust::get<0>(x) - thrust::get<1>(x)));
	}

};


//Print Out Device Values

template <typename T>
void printValues(thrust::device_vector<T> out){
	cout.precision(20);
	thrust::copy(out.begin(), out.end(), std::ostream_iterator<T>(std::cout, "\n"));
}

//*******************************************************
//Output Delta
//*******************************************************

//Find the delta of the output layer
inline void findOutputDelta(thrust::host_vector<double> output, thrust::host_vector<double> &target){
	thrust::device_vector<double> X = output;
	thrust::device_vector<double> Y = target;

	//Transform the target vector
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), OutputDelta_functor());

	thrust::copy(Y.begin(), Y.end(), target.begin());

	//Free the GPU memory
	vector_free::free(X);
	vector_free::free(Y);
}

//*******************************************************
//Hidden Delta
//*******************************************************

//Input sum - the sums of all the delta * weight, output - The output of the current layer
//output - results
//Perform secondary part of operation to find the new delta (output * (1- output) * sum
inline thrust::host_vector<double> findNewHiddenDelta(thrust::device_vector<double> sum, thrust::device_vector<double> output){

	thrust::host_vector<double> results(sum.size());

	thrust::transform(output.begin(), output.end(), sum.begin(), sum.begin(), dot_product_special<double>());
#ifdef TRIAL2
	printValues(output);
	cout << "_____________________" << endl;
#endif
	//Return the product as a host_copy
	thrust::copy(sum.begin(), sum.end(), results.begin());

	return results;
}



//input current layer to be modified and the next layer
//Retrieve the deltas for the current layer

inline void findHiddenDelta(SNeuronLayer neurons_weights, SNeuronLayer &previous_layer){
	size_t availableMemory = getCurrentGPUMemory();
	int size_available = 0;
	int numberNeurons = neurons_weights.neurons.size();
	int number_weights_per_neuron = neurons_weights.neurons[0].weights.size();
	//Get number of neurons which can fit in GPU memory
	int available_slots = ((((availableMemory - 100000) / sizeof(double))
		- (neurons_weights.delta.size() +
		(2 * previous_layer.number_per_layer) + (2 * number_weights_per_neuron)))
		/
		((number_weights_per_neuron)));

	//Set the size of the arrays
	if (available_slots > numberNeurons){
		size_available = numberNeurons * (number_weights_per_neuron);
		available_slots = numberNeurons;
	}
	else{
		size_available = available_slots * (number_weights_per_neuron);
	}


	//thrust::host_vector<double> return_sums(previous_layer.number_per_layer);//Sums to be returned 
	//thrust::host_vector<double> temp(previous_layer.number_per_layer);//Temporary location for weight
	int size = neurons_weights.neurons.size();


	//temp = neurons_weights.getWeights(0);
	thrust::device_vector<double> weights(size_available);//Stores the weights
	thrust::device_vector<double> deltas = neurons_weights.delta;//Store the Delta in GPU Memory
	thrust::device_vector<double> gpu_sums(previous_layer.number_per_layer);//Sums in GPU, same number as delta
	thrust::device_vector<double> gpu_sums_temp(previous_layer.number_per_layer);//Sums in GPU, same number as delta
	thrust::device_vector<int> map(size_available);
	thrust::device_vector<int> map2(size_available);
	//Fill the sums with 0
	thrust::fill(gpu_sums.begin(), gpu_sums.end(), 0);

	//Create the map going 0,0,...,5,5,...,n,n
	thrust::copy(thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), _1 % (number_weights_per_neuron)), thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), _1 % (number_weights_per_neuron)) + weights.size(), map.begin());
	thrust::copy(thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), _1 / (number_weights_per_neuron)), thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), _1 / (number_weights_per_neuron)) + weights.size(), map2.begin());


	for (int i = 0; i < numberNeurons; i += available_slots){
		//Store Weights and Bias in Memory
		getWeights(neurons_weights, weights, thrust::device_vector<double>(0), i, false, false);
		//Transform the weights by multiplication
		thrust::transform(weights.begin(), weights.end(), thrust::make_permutation_iterator(deltas.begin(), map2.begin()),
			weights.begin(), thrust::multiplies<double>());
#ifdef TRIAL2
		printValues(map2);
		printValues(weights);
		cout << "_____________________" << endl;
#endif
		if (i == 0){
			thrust::sort_by_key(map.begin(), map.end(), weights.begin());
			thrust::reduce_by_key(map.begin(), map.end(), weights.begin(), thrust::make_discard_iterator(), gpu_sums.begin());

#ifdef TRIAL2
			printValues(gpu_sums);
			cout << "_____________________" << endl;
#endif
		}
		else{
			thrust::copy(thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), _1 % (number_weights_per_neuron)), thrust::make_transform_iterator(thrust::make_counting_iterator((int)0), _1 % (number_weights_per_neuron)) + weights.size(), map.begin());

			thrust::sort_by_key(map.begin(), map.end(),
				weights.begin());


			thrust::reduce_by_key(map.begin(), map.end(), weights.begin(), thrust::make_discard_iterator(), gpu_sums_temp.begin());

			//Sum the new sum
			thrust::transform(gpu_sums_temp.begin(), gpu_sums_temp.end(), gpu_sums.begin(), gpu_sums.begin(), thrust::plus<double>());
		}
	}

	vector_free::free(deltas);
	vector_free::free(weights);

	gpu_sums_temp = previous_layer.getOutput();

	previous_layer.delta = findNewHiddenDelta(gpu_sums, gpu_sums_temp);

	//Free the memory

	vector_free::free(gpu_sums);
	vector_free::free(gpu_sums_temp);
	vector_free::free(map);
	vector_free::free(map2);

}


//*******************************************************
//Apply Momentum
//*******************************************************


//Apply the learning momentum to all the weights in each layer
//Input CurrentLayer - the current layer of the neuron
//
inline void applyMomentum(SNeuronLayer &currentLayer, double alpha){
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
		thrust::transform(previousWeights.begin(), previousWeights.end(), weights.begin(), weights.begin(), saxpy_functor<double>(alpha));

		//Set the weights back from memory
		setWeights(currentLayer, weights, previousWeights, i, i + available_slots);
	}

	vector_free::free(weights);
	vector_free::free(previousWeights);

}
inline int getWeights(SNeuronLayer currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int neuronPosition){
	return getWeights(currentLayer, weights, previousWeight, neuronPosition, true, true);

}

//Fill the weight bars
inline int getWeights(SNeuronLayer currentLayer, thrust::device_vector<double> &weights, thrust::device_vector<double> &previousWeight, int neuronPosition, bool getPrevious, bool getBias){

	//Retrieve number of weights in the current layer

	int weightsPerNeuron = getBias ? currentLayer.neurons[0].weights.size() + 1 : currentLayer.neurons[0].weights.size();
	int weightSize = weights.size();
	int weightPosition = neuronPosition * weightsPerNeuron;
	//Retrieve the current list of weights
	while (neuronPosition < currentLayer.number_per_layer && weightPosition < weightSize){

		thrust::copy(currentLayer.neurons[neuronPosition].weights.begin(),
			currentLayer.neurons[neuronPosition].weights.end(),
			weights.begin() + (weightPosition));

		if (getPrevious){
			thrust::copy(currentLayer.neurons[neuronPosition].previousWeight.begin(),
				currentLayer.neurons[neuronPosition].previousWeight.end(),
				previousWeight.begin() + (weightPosition));
		}

		if (getBias){
			//Get the bias as well
			weights[weightPosition + weightsPerNeuron - 1] = currentLayer.neurons[neuronPosition].bias;

			if (getPrevious){
				previousWeight[weightPosition + weightsPerNeuron - 1] = currentLayer.neurons[neuronPosition].previousBias;
			}

		}
		weightPosition += weightsPerNeuron;
		neuronPosition += 1;
	}



	return (weightPosition / weightsPerNeuron);
}


inline int setWeights(SNeuronLayer &currentLayer, thrust::device_vector<double> &weights,
	thrust::device_vector<double> &previousWeight, int startPosition, int neuronPosition){
	setWeights(currentLayer, weights, previousWeight, startPosition, neuronPosition, false);
	return 0;
}

inline int setWeights(SNeuronLayer &currentLayer, thrust::device_vector<double> &weights,
	thrust::device_vector<double> &previousWeight, int startPosition, int neuronPosition, bool include_previous){
	int size = currentLayer.neurons[0].weights.size() + 1;
	int position;
	startPosition = startPosition * size;
	neuronPosition = neuronPosition * (size);
	//Transfer the neurons back if it doesn't work
	for (int i = startPosition; i < neuronPosition; i += size){
		position = i / size;
		thrust::copy(weights.begin() + (i), weights.begin() + (i + size - 1), currentLayer.neurons[position].weights.begin());
		currentLayer.neurons[position].bias = weights[i + size - 1];
		if (include_previous){//Also modify the previousWeight
			thrust::copy(previousWeight.begin() + (i), previousWeight.begin() + (i + size - 1), currentLayer.neurons[position].previousWeight.begin());
			currentLayer.neurons[position].previousBias = previousWeight[i + size - 1];
		}
	}
	return 0;
}



//*******************************************************
//Apply Correction
//*******************************************************

inline void applyCorrection(SNeuronLayer &currentLayer, thrust::host_vector<double> initial_output, double beta){
	size_t availableMemory = getCurrentGPUMemory();
	int position = 0;
	int number_weights_per_neuron = currentLayer.neurons[0].weights.size();
	//Get number of neurons which can fit in GPU memory
	int available_slots = ((availableMemory - 100000) / sizeof(double)) /
		((number_weights_per_neuron + 1) * 2);
	int size_available = 0;
	int numberNeurons = currentLayer.neurons.size();
	//Set the size of the arrays
	if (available_slots > numberNeurons){
		size_available = numberNeurons * (number_weights_per_neuron + 1);
		available_slots = numberNeurons;
	}
	else{
		size_available = available_slots * (number_weights_per_neuron + 1);
	}


	thrust::device_vector<double> weights(size_available);
	thrust::device_vector<double> previousWeights(size_available);
	thrust::device_vector<double> output = initial_output;
	thrust::device_vector<double> delta = currentLayer.getDelta();
	thrust::device_vector<int> map(size_available);
	thrust::device_vector<int> map2(size_available);
	for (int i = 0; i < size_available; i++){
		map[i] = i % (number_weights_per_neuron + 1);



		if (i % (number_weights_per_neuron + 1) == 0 && i != 0){
			position++;
		}

		map2[i] = position;
	}

	for (int i = 0; i < numberNeurons; i += available_slots){

		//Store Weights and Bias in Memory
		getWeights(currentLayer, weights, previousWeights, i);


		//Weights, Previous Weights, Output, Delta
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(weights.begin(),
			previousWeights.begin(),
			thrust::make_permutation_iterator(output.begin(), map.begin()),
			thrust::make_permutation_iterator(delta.begin(), map2.begin()))),
			thrust::make_zip_iterator(thrust::make_tuple(weights.end(), previousWeights.end(),
			thrust::make_permutation_iterator(output.end(), map.end()),
			thrust::make_permutation_iterator(delta.end(), map2.end()))),
			apply_correction<double>(beta));
		//Set the weights back from memory
		setWeights(currentLayer, weights, previousWeights, i, i + available_slots, true);
	}

	vector_free::free(weights);
	vector_free::free(previousWeights);
	vector_free::free(map);
	vector_free::free(map2);
	vector_free::free(output);
	vector_free::free(delta);
}

//*******************************************************
//FeedForward
//*******************************************************
//Place the output into the currentlayer from start onward
inline void setOutput(SNeuronLayer &currentLayer, thrust::device_vector<double> output, int start){
	currentLayer.setOutput(output);
	//thrust::copy(output.begin(), output.end(), currentLayer.output.begin() + start);
}

//Find the output of each neuron of the current layer
inline void feedForwardGPU(SNeuronLayer &currentLayer, SNeuronLayer previousLayer){
	size_t availableMemory = getCurrentGPUMemory();
	double temp;
	int number_weights_per_neuron = currentLayer.neurons[0].weights.size();
	//Get number of neurons which can fit in GPU memory
	//Based on needing memory for the weights of the current layer and the output of the previous layer
	int available_slots = (((availableMemory - 100000) / sizeof(double)) -
		previousLayer.number_per_layer) / ((number_weights_per_neuron + 2) * 2);
	int size_available = 0;
	int numberNeurons = currentLayer.neurons.size();
	//Set the size of the arrays
	if (available_slots > numberNeurons){
		size_available = numberNeurons * (number_weights_per_neuron + 1);
		available_slots = numberNeurons;
	}
	else{
		size_available = available_slots * (number_weights_per_neuron + 1);
	}

	thrust::device_vector<double> weights(size_available);//Store the weights of the neurons
	thrust::device_vector<double> output = previousLayer.getOutput(1, 1);//Store the output with an additional one at the end (used for the 
	thrust::device_vector<double> gpu_output(numberNeurons);
	//Fill the map with a set of data

	for (int i = 0; i < numberNeurons; i += available_slots){
		getWeights(currentLayer, weights, weights, i, false, true);//Get the weights with the bias

		//Multiply each of the weights by the output they recieve from the previous layer
		thrust::transform(weights.begin(),
			weights.end(),
			thrust::make_permutation_iterator(output.begin(),
			thrust::make_transform_iterator(thrust::make_counting_iterator((int)0),
			(_1 % (number_weights_per_neuron + 1)))),
			weights.begin(),
			thrust::multiplies<double>());

#ifdef TRIAL5
		printValues(weights);
#endif

		//Sum all of the rows
		thrust::reduce_by_key(
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0),
			_1 / (number_weights_per_neuron + 1)
			),
			thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0),
			_1 / (number_weights_per_neuron + 1)
			) + size_available,
			weights.begin(),
			thrust::make_discard_iterator(),
			gpu_output.begin());

#ifdef TRIAL5
		printValues(gpu_output);
#endif

		for (int j = 0; j < (int)gpu_output.size(); j++){
			temp = gpu_output[j];
			gpu_output[j] = std::exp((double)-temp);
		}

		//Run the sigmond function on the output
		thrust::transform(gpu_output.begin(), gpu_output.end(), gpu_output.begin(), ((double)1 / ((double)1 + _1)));



		//Place the results back into main memory
		setOutput(currentLayer, gpu_output, i);
	}

	vector_free::free(gpu_output);

	vector_free::free(weights);

	vector_free::free(output);


}




//*******************************************************
//Misc
//*******************************************************
inline size_t getCurrentGPUMemory(){
	size_t mem_tot;
	size_t mem_free;
	cudaMemGetInfo(&mem_free, &mem_tot);
	return mem_free;
}



template<typename T>
inline T square_means_sums(T* target, T* output, int size){
	thrust::device_vector<T> tgt(target, target + size);
	thrust::device_vector<T> out(output, output + size);
	double temp = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(tgt.begin(), out.begin())), thrust::make_zip_iterator(thrust::make_tuple(tgt.end(), out.end())), square_means_sum_functor<double>(), 0.0, thrust::plus<double>());
	
	vector_free::free(tgt);
	vector_free::free(out);
	return temp;
}




