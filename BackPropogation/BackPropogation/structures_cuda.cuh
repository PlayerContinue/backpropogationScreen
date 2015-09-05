//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains structures used throughout the application
//----------------------------------------------------------------------------------------
#pragma once
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <thrust/execution_policy.h>
#include "CSettings.h"
#include "SNeuron.cuh"
#include "util.h"
using namespace std;
using namespace thrust::placeholders;
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
	thrust::host_vector<bool> locked_nodes;

	//Pointer to the settings object
	CSettings *settings;
#ifdef DEBUG
	int num_locked = 0;
#endif


	//***************************************
	//Constructors
	//***************************************

	//Create empty Neuron Layer
	SNeuronLayer() : input_output_layer(0){}

	//Used when the number/value of the weights is not random
	SNeuronLayer(int number_nodes){
		//Resize the network
		new_resizeNetwork(number_nodes);
	}

	//Create a neuron layer containing n nodes
	//The bias and weights are random
	SNeuronLayer(int number_nodes, int number_nodes_previous_layer){
		
		//Resize the network
		new_resizeNetwork(number_nodes);

		//Randomly create a bias for each of the neurons
		for (int j = 0; j < number_nodes; j++){//Travel through neurons
			this->neurons.push_back(SNeuron());


			//Add the bias (Random Number between 0 and 1)
			this->neurons[j].bias = RandomClamped();


			if (number_nodes_previous_layer > 0){//Only add weights to non-input layers
				//Add the weights
				for (int k = 0; k < number_nodes_previous_layer; k++){//Number of neurons in next layer used as number of outgoing outputs
					this->neurons[j].weights.push_back(RandomClamped());//Add a random weight between 0 and 1
					this->neurons[j].previousWeight.push_back(0);//Set previous weight to 0
				}
			}
			else{//The input layer
				this->neurons[j].weights.push_back(RandomClamped());//Add a random weight between 0 and 1
				this->neurons[j].previousWeight.push_back(0);//Set previous weight to 0
			}

			//Set the initial previousbias to 0
			this->neurons[j].previousBias = 0;
		}
	}

	SNeuronLayer(int number_nodes, int number_nodes_previous_layer, CSettings *settings):SNeuronLayer(number_nodes,number_nodes_previous_layer){
		this->settings = settings;
	}

	//***************************************
	//Resize Network
	//***************************************
private:
	//Initilize the size of the network when creating a new layer
	void new_resizeNetwork(int number_nodes){
		this->output = thrust::host_vector<double>(number_nodes);
		this->delta = thrust::host_vector<double>(number_nodes);
		this->locked_nodes = thrust::host_vector<bool>(number_nodes);
		this->number_per_layer = number_nodes;
	}
public:

	//Resizes the network to accomadate more/less nodes
	void resizeNetwork(int number_nodes){
		this->output.resize(number_nodes);
		this->delta.resize(number_nodes);
		this->locked_nodes.resize(number_nodes);
		this->neurons.resize(number_nodes);
		this->number_per_layer = number_nodes;
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
			os << layer.neurons[i] << " / " << layer.locked_nodes[i] << endl;
		}
		os << endl;
		return os;
	}

	friend istream& operator>>(istream& is, SNeuronLayer& layer){
		int number_of_weights;
		char next;
		//Retrieve number of neurons
		is >> layer.number_per_layer;

		//Retrieve the number of weights
		is >> number_of_weights;

		layer.resizeNetwork(layer.number_per_layer);

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

			//Read in the next value to check for correct formatting of file
			//Should be a /
			is >> next;
			if (next != '/'){
				throw new exception("File not formatted correctly");
			}
			is >> layer.locked_nodes[i];
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

	//Count the number of times which the current d_mean_square_error == d_previous_mean_square_error
	int i_equal_square_errors;

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

	//Store the threshold for whether a neuron or a layer is added
	double d_neuron_or_layer_threshold;

	//store the most recently recorded network file
	string s_network_file_name;



	//*********************************
	//Constructors
	//*********************************
	//Empty Constructor
	SCheckpoint(){};

	//Create a checkpoint containing a distance based on how far apart the output is meant to be
	//The numbers can be used to test whether a new row or a new neuron should be added
	//Closer to the distance means it should add a new layer, otherwise a neuron should be added
	//Adding a neuron increases the spread of the output
	SCheckpoint(double** objects, int row_size,int col_size){
		get_layer_or_row(objects, row_size, col_size);
	}

	//*********************************
	//Functions
	//*********************************
	void get_layer_or_row(double** objects, int row_size, int col_size){
		//Seed the random
		srand((unsigned)(time(NULL)));
		int size = (row_size < 100 ? row_size : row_size / 100);
		int randomPosition;
		this->d_neuron_or_layer_threshold = 0;
		thrust::device_vector<double> temp_output(col_size);
		thrust::device_vector<double> temp_results(col_size);
		for (int k = 0; k < size; k++){
			randomPosition = RandInt(0, row_size);
			for (int j = 0; j < col_size; j++){
				temp_output[j] = objects[randomPosition][j];
			}

			randomPosition = 0;
			//Get the total distance between all possible points
			for (int i = 0; i < col_size; i++){
				thrust::transform(
					temp_output.begin() + i, temp_output.end(),
					thrust::make_constant_iterator<double>(temp_output[i]),
					temp_results.begin(),
					(_1 - _2) * (_1 - _2));
				//Reduce and retrieve the answer
				randomPosition += thrust::reduce(temp_results.begin(), temp_results.end());
			}
			this->d_neuron_or_layer_threshold += sqrt(randomPosition / (double)(col_size * (col_size-1)) );
		}
		temp_output.clear();
		temp_output.shrink_to_fit();
		temp_results.clear();
		temp_results.shrink_to_fit();

		//Get the average
		this->d_neuron_distance_threshold /= size;
	}
		
	//***************************************
	//Overload Operators
	//***************************************
public:
	//Save to file
	friend ostream& operator<<(ostream& os, const SCheckpoint checkpoint){
		
		os << "i_number_of_loops_checkpoint " << checkpoint.i_number_of_loops_checkpoint << endl;
		
		os << "i_times_lowest_mean_square_error_to_large " << checkpoint.i_times_lowest_mean_square_error_to_large << endl;

		os << "i_equal_square_errors " << checkpoint.i_equal_square_errors << endl;

		os << "d_mean_square_error " << checkpoint.d_mean_square_error << endl;
		
		os << "d_previous_mean_square_error " << checkpoint.d_previous_mean_square_error << endl;
		
		os << "i_number_of_loops " << checkpoint.i_number_of_loops << endl;

		os << "d_lowest_mean_square_error " << checkpoint.d_lowest_mean_square_error << endl;

		os << "d_row_distance_threshold " << checkpoint.d_row_distance_threshold << endl;

		os << "d_neuron_distance_threshold " << checkpoint.d_neuron_distance_threshold << endl;

		os << "d_neuron_or_layer_threshold " << checkpoint.d_neuron_or_layer_threshold << endl;

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
		if (next.compare("i_equal_square_errors") == 0){
			is >> checkpoint.i_equal_square_errors;
		}
		else{
			checkpoint.i_equal_square_errors = 0;
		}

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
		if (next.compare("d_neuron_or_layer_threshold") == 0){
			is >> checkpoint.d_neuron_or_layer_threshold;
		}
		else{
			checkpoint.d_neuron_or_layer_threshold = 10;
		}

		is >> next;
		is >> checkpoint.s_network_file_name;

		return is;
	}


};


