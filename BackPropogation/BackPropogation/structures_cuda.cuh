//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains structures used throughout the application
//----------------------------------------------------------------------------------------
#pragma once
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "util.h"
using namespace std;

//Structure containing the neuron
struct SNeuron{
	//Bias for the neuron
	double bias = 0;

	//Previous Bias
	double previousBias = 0;

	//The change for the current layer
	double delta = 0;

	//List of weights for outgoing neurons
	thrust::host_vector<double> weights;

	//Store the previously stored weight
	thrust::host_vector<double> previousWeight;

	//The current output
	double output = 0;

	//Store how many times the neuron was activated
	int activated = 0;

	//Neuron is either removed (1) or active (0)
	//If 1, the neuron will be skipped during feedback
	short removed = 0;

	//Creates an empty Neuron
	SNeuron(){

	}
	//Create a neuron with a bias and weight
	SNeuron(double bias, thrust::host_vector<double> weights) : bias(bias), weights(weights){

	}

};

//Structure for the neuron layer
struct SNeuronLayer{
	//number of neurons in layer
	int number_per_layer = 0;

	//Store whether the current layer is an input/output layer
	// 0 is input, 1 is output, 2 is hidden layer, 3 is in training
	int input_output_layer = 0;

	//List of neurons
	vector<SNeuron> neurons;
	
#ifdef DEBUG
	int num_locked = 0;
#endif

	//Create empty Neuron Layer
	SNeuronLayer() : input_output_layer(0){}

	//Add new weights to the current layer
	void addNewWeights(int numberOfNeuronsAdded){
		for (int i = 0; i < this->number_per_layer; i++){
			for (int k = 0; k < numberOfNeuronsAdded; k++){
				this->neurons[i].weights.push_back(RandomClamped());
				this->neurons[i].previousWeight.push_back(0);
			}
		}
	}

};

