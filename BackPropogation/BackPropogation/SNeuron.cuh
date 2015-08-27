#pragma once
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include "util.h"
using namespace std;

//Structure containing the neuron
struct SNeuron{
	//Bias for the neuron
	double bias = 0;

	//Previous Bias
	double previousBias = 0;

	//List of weights for outgoing neurons
	thrust::host_vector<double> weights;

	//Store the previously stored weight
	thrust::host_vector<double> previousWeight;

	//Store how many times the neuron was activated
	int activated = 0;

	//Neuron is either removed (1) or active (0)
	//If 1, the neuron will be skipped during feedback
	short removed = 0;

	//Creates an empty Neuron
	SNeuron(){

	}

	//Create empty SNeuron with x weights
	SNeuron(int x){
		weights = thrust::host_vector<double>(x);
		previousWeight = thrust::host_vector<double>(x);
	}

	//Create a neuron with a bias and weight
	SNeuron(double bias, thrust::host_vector<double> weights) : bias(bias), weights(weights){

	}

	friend ostream& operator<<(ostream& os, const SNeuron& neuron){
		//Output the weights
		for (int i = 0; i < (int)neuron.weights.size(); i++){
			os << neuron.weights[i] << " ";
		}
		//Output a seperator
		os << "/ ";

		//Output the bias
		os << neuron.bias;
		return os;
	}


};
