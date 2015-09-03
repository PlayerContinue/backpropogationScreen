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

	//Contains a list stating which nodes are locked
	//When marked as 1 and a node exists at position, the node is locked.
	//Locked nodes do not change weight or width
	thrust::host_vector<double> locked_nodes;

#ifdef DEBUG
	int num_locked = 0;
#endif


	//***************************************
	//Constructors
	//***************************************

	//Create empty Neuron Layer
	SNeuronLayer() : input_output_layer(0){}

	//Create a neuron layer containing n nodes
	SNeuronLayer(int number_nodes){
		this->output.resize(number_nodes);
		this->delta.resize(number_nodes);
		this->locked_nodes.resize(number_nodes);
	}

	//***************************************
	//Weight Modifiers
	//***************************************

	//Add new weights to the current layer
	void addNewWeights(int numberOfNeuronsAdded){
		for (int i = 0; i < this->number_per_layer; i++){
			for (int k = 0; k < numberOfNeuronsAdded; k++){
				this->neurons[i].weights.push_back(this->neurons[i].weights[k] / 2);//Set the new weight to half the previous weight
				this->neurons[i].weights[k] = this->neurons[i].weights[k] / 2;//Set the divided weight to the same as the new weights
				//REASON: ((n/2)*k) + ((n/2)*l) = ((n) * k) approximately 
				this->neurons[i].previousWeight.push_back(0);
			}
		}
	}

	//Remove all except X weights from the neurons
	void keepXWeights(int X){
		int i_previous_number_weights = this->neurons[0].weights.size();
		for (int i = 0; i < this->number_per_layer; i++){
			this->neurons[i].weights.resize(X);
			this->neurons[i].previousWeight.resize(X);
		}

		if (X > i_previous_number_weights){
			for (int i = 0; i < this->number_per_layer; i++){
				for (int j = i_previous_number_weights; j < X; j++){
					//Set a new weight
					this->neurons[i].weights[j] = RandomClamped();
				}
			}
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



//Structure containing information to create a checkpoint
struct SCheckpoint{

	//Count the total number of loops which have occured
	int i_number_of_loops_checkpoint = 0;
	
	//Count the number of current loops traveled through before reaching a reset
	int i_number_of_loops = 0;

	//Count the number of times left for the mean to be larger than the previous mean before trying to add new neurons
	int i_times_lowest_mean_square_error_to_large;

	// Store the mean square error
	double d_mean_square_error = (double)INT_MAX;

	//Store the previous round mean_square_error to test if the value changed between rounds
	double d_previous_mean_square_error = 0;

	//Store the lowest mean_square_error found
	double d_lowest_mean_square_error = (double)INT_MAX;

	//Store the most recent d_row_distance_threshold
	double d_row_distance_threshold;

	//store the most recent d_neuron_distance_threshold
	double d_neuron_distance_threshold;

	//store the most recently recorded network file
	string s_network_file_name;

	//***************************************
	//Overload Operators
	//***************************************
	//Save to file
	friend ostream& operator<<(ostream& os, const SCheckpoint checkpoint){
		
		os << "i_number_of_loops_checkpoint " << checkpoint.i_number_of_loops_checkpoint << endl;
		
		os << "i_times_lowest_mean_square_error_to_large " << checkpoint.i_times_lowest_mean_square_error_to_large << endl;

		os << "d_mean_square_error " << checkpoint.d_mean_square_error << endl;
		
		os << "d_previous_mean_square_error " << checkpoint.d_previous_mean_square_error << endl;
		
		os << "i_number_of_loops " << checkpoint.i_number_of_loops << endl;

		os << "d_lowest_mean_square_error " << checkpoint.d_lowest_mean_square_error << endl;

		os << "d_row_distance_threshold " << checkpoint.d_row_distance_threshold << endl;

		os << "d_neuron_distance_threshold " << checkpoint.d_neuron_distance_threshold << endl;

		os << "s_network_file_name " << checkpoint.s_network_file_name << endl;

		return os;
	}

	//Load from file
	friend istream& operator>>(istream& is, SCheckpoint& checkpoint){
		string next;
		is >> next;
		is >> checkpoint.i_number_of_loops_checkpoint;

		is >> next;
		is >> checkpoint.i_times_lowest_mean_square_error_to_large;

		is >> next;
		is >> checkpoint.d_mean_square_error;

		is >> next;
		is >> checkpoint.d_previous_mean_square_error;

		is >> next;
		is >> checkpoint.i_number_of_loops;

		is >> next;
		is >> checkpoint.d_lowest_mean_square_error;

		is >> next;
		is >> checkpoint.d_row_distance_threshold;

		is >> next;
		is >> checkpoint.d_neuron_distance_threshold;

		is >> next;
		is >> checkpoint.s_network_file_name;

		return is;
	}


};


