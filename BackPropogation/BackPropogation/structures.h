//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains structures used throughout the application
//----------------------------------------------------------------------------------------

#pragma once
#include <vector>
using namespace std;

//Structure containing the neuron
struct SNeuron{
	//Bias for the neuron
	double bias;
	
	//Previous Bias
	double previousBias;

	//The change for the current layer
	double delta;

	//List of weights for outgoing neurons
	vector<double> weights;

	//Store the previously stored weight
	vector<double> previousWeight;

	//The current output
	double output;

	//Creates an empty Neuron
	SNeuron(){

	}
	//Create a neuron with a bias and weight
	SNeuron(double bias, vector<double> weights) : bias(bias), weights(weights){

	}

};

//Structure for the neuron layer
struct SNeuronLayer{
	//number of neurons in layer
	int number_per_layer;

	//Store whether the current layer is an input/output layer
	// 0 is input, 1 is output, 2 is hidden layer, 3 is in training
	int input_output_layer;

	//List of neurons
	vector<SNeuron> neurons;

	//Create empty Neuron Layer
	SNeuronLayer(){}

};

