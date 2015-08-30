//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains structures used throughout the application
//----------------------------------------------------------------------------------------
#pragma once
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include "SNeuron.cuh"
#include "util.h"
using namespace std;

//Structure for the neuron layer
struct SNeuronLayer{
	//number of neurons in layer
	int number_per_layer = 0;

	//Store whether the current layer is an input/output layer
	// 0 is input, 1 is output, 2 is hidden layer, 3 is in training
	int input_output_layer = 0;

	double average_delta = 0;

	//List of neurons
	vector<SNeuron> neurons;

	//Store the output of the neurons
	thrust::host_vector<double> output;

	//Holds the delta for the current row
	thrust::host_vector<double> delta;
#ifdef DEBUG
	int num_locked = 0;
#endif


	//***************************************
	//Constructors
	//***************************************

	//Create empty Neuron Layer
	SNeuronLayer() : input_output_layer(0){}


	//***************************************
	//Weight Modifiers
	//***************************************

	//Add new weights to the current layer
	void addNewWeights(int numberOfNeuronsAdded){
		for (int i = 0; i < this->number_per_layer; i++){
			for (int k = 0; k < numberOfNeuronsAdded; k++){
				this->neurons[i].weights.push_back(this->neurons[i].weights[k]/2);//Set the new weight to half the previous weight
				this->neurons[i].weights[k] = this->neurons[i].weights[k] / 2;//Set the divided weight to the same as the new weights
				//REASON: ((n/2)*k) + ((n/2)*l) = ((n) * k) approximately 
				this->neurons[i].previousWeight.push_back(0);
			}
		}
	}

	//Remove all except X weights from the neurons
	void keepXWeights(int X){
		for (int i = 0; i < this->number_per_layer; i++){
			this->neurons[i].weights.resize(X);
			this->neurons[i].previousWeight.resize(X);
		}
	}

	//Remove all weights which are connected to a neuron in the previous layer
	//at position y
	//Primarily used when removing a neuron for testing
	void removeWeightsAtY(int y){
		for (int i = 0; i < this->number_per_layer; i++){
			this->neurons[i].weights.erase(this->neurons[i].weights.begin() + y);//Remove the neuron
		}
	}
	

	//***************************************
	//Overload Operators
	//***************************************
	friend ostream& operator<<(ostream& os, const SNeuronLayer layer){
		//Output number of neurons
		os << layer.number_per_layer << endl;
		//Print number of weights per neuron
		os << layer.neurons[0].weights.size() << endl;
		//Print the values of each neuron
		for (int i = 0; i < layer.number_per_layer; i++){
			os << layer.neurons[i] << endl;
		}
		return os;
	}

	friend istream& operator>>(istream& is, SNeuronLayer& layer){
		int number_of_weights;
		char next; 
		//Retrieve number of neurons
		is >> layer.number_per_layer;

		//Retrieve the number of weights
		is >> number_of_weights;

		//Create the delta holder
		layer.delta = thrust::host_vector<double>(layer.number_per_layer);

		//Create the output holder
		layer.output = thrust::host_vector<double>(layer.number_per_layer);

		//Create the neurons
		layer.neurons = vector<SNeuron>(layer.number_per_layer);

		for (int i = 0; i < layer.number_per_layer; i++){
			layer.neurons[i] = SNeuron(number_of_weights);
			//Set the weights
			for (int j = 0; j < number_of_weights; j++){
				is >> layer.neurons[i].weights[j];
			}
			//Read in the next value to check for correct formatting of file
			//Should be a /
			is >> next;
			if (next != '/'){
				throw new exception("File not formatted correctly");
			}
			//Set the bias
			is >> layer.neurons[i].bias;
		}


		return is;
	}




	//***************************************
	//Get and set
	//***************************************

	//Retrieve a list of the current output in the form
	//|out_11, out_21, out_31, ... , out_n1|
	thrust::host_vector<double> getOutput(){
		return this->output;
	}

	//Retrieve a list of the current output in the form
	//|out_11, out_21, out_31, ... , out_n1|
	//Input expand - include extra values, value - value to add
	thrust::host_vector<double> getOutput(int expand, double value){
		thrust::host_vector<double> y(this->number_per_layer + expand);
		//for (int i = 0; i < this->number_per_layer; i++){
			//Store the weight in the vector
			//y[i] = this->neurons[i].output;
		//}
		//Copy the output into the expanded array
		thrust::copy(this->output.begin(), this->output.end(), y.begin());

		//Insert extra values
		for (int i = this->number_per_layer; i < this->number_per_layer + expand; i++){
			y[i] = value;
		}

		return y;
	}

	//Retrieve a list of the current weights in the form
	//|x_11, x_21, x_31, ... , x_n1|
	//input n (the position of the weights desired)
	//output y (the array of weights)
	thrust::host_vector<double> getWeights(int n){
		thrust::host_vector<double> y;
		for (int i = 0; i < this->number_per_layer; i++){
			//Store the weight in the vector
			y.push_back(this->neurons[i].weights[n]);
		}

		return y;
	}

	//Retrieve a list of the current weights in the form
	//|x_11, x_21, x_31, ... , x_N1
	//input n (the position of the weights desired)
	//output y (the array of weights)
	thrust::host_vector<double> getDelta(){

		return this->delta;
	}

	//Set the current output using a device vector
	//Throws error when output is wrong size
	void setOutput(thrust::device_vector<double> new_output){
		if (new_output.size() != this->number_per_layer){
			throw new exception("To many Outputs");
		} 

		//Error causing size change 
		if ((int)this->output.size() < this->number_per_layer){
			this->output.resize(this->number_per_layer);
		}


		thrust::copy(new_output.begin(), new_output.end(), this->output.begin());
	}

	void setOutput(thrust::host_vector<double> new_output){
		if (new_output.size() != this->number_per_layer){
			throw new exception("To many Outputs");
		}

		//Error causing size change 
		if ((int)this->output.size() < this->number_per_layer){
			this->output.resize(this->number_per_layer);
		}
		//Set the output
		this->output = new_output;
	}


};


