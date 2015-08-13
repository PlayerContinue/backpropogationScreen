//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains a backpropagation algorithm for a neural network
//----------------------------------------------------------------------------------------

#pragma once
#include <vector>
#include <time.h>
#include "structures.h"
#include "util.h"

using namespace std;

class CNetwork
{

private:
	//Networks Variables
	//The number of layers
	int v_num_layers;

	//Vector containing the layers of neurons
	vector<SNeuronLayer> v_layers;

	//Learning rate (look up)
	double beta;

	//momentum (Look up)
	double alpha;

	//Activation Threshold
	const double neuron_activated = .1;

	//Success Rate
	int success = 0;
	
	//Failure Rate
	int failure = 0;

	//Number of Outputs
	int I_output;

	//Number of inputs
	int I_input;




public:
	//-----------------------------------------------------------------------------------------------------------
	//Constructors
	//-----------------------------------------------------------------------------------------------------------
	CNetwork();
	//Constructor 
	//sizes - The number of neurons per layer
	//i.e. 3 inputs, first hidden layer 2, second 4 would, output 1 is [3,2,4,1]
	CNetwork(vector<int> &sizes);
	//Constructor 
	//sizes - The number of neurons per layer
	//i.e. 3 inputs, first hidden layer 2, second 4 would, output 1 is [3,2,4,1]
	CNetwork(vector<int> &sizes, double beta, double alpha);



	/*Update the networks weights and baises by applying gradiant descent using
	backpropagation to a single mini batch. The mini_batch
	is a vector of vectors (x,y), and "eta" is the learning rate*/
	void update_mini_batch(vector<vector<double>> &mini_batch, double eta);

	//Feed Forward one set of inputs 
	void feedForward(double *in);

	/*Return a vector "(nabla_b, nabla_w)" representing the
		gradient for the cost function C_x.  "nabla_b" and
		"nabla_w" are layer - by - layer lists of numpy arrays, similar
		to "self.biases" and "self.weights".*/
	void backprop(double *in, double *tgt);

private:

	/*
	Return the vector of the partial derivatives \partial C_X /
	\partial a for the output activations
	*/
	double cost_derivative(double output_activation, double y);

	/*
	The sigmoid function

	*/
	double sigmoid(double z){
		double exp_value;
		double return_value;

		//Exponential Value calculation
		exp_value = exp((double)-z);
		//Final sigmoid value
		return_value = 1 / (1 + exp_value);
		return return_value;
	}

	//Derivative of the sigmoid function
	double sigmoid_prime(double z);

	//-----------------------------------------------------------------------------------------------------------

	//Retrive the average of the next layers weights
	double inline average_of_next_weights(int position,int nodePosition){
		int size = this->v_layers.at(position + 1).number_per_layer;//Get number of neurons in next layer
		double results = 0;
		//Sum all the weights of the nodes in the given position and layer
		for (int i = 0; i < size; i++){

			results = this->v_layers.at(position + 1).neurons.at(i).weights.at(nodePosition);

		}
		//Return the average value
		return (results / size);
	}

	//Retrive the average of the next layers weights
	double inline average_of_bias(int position){
		int size = this->v_layers.at(position + 1).number_per_layer;//Get number of neurons in next layer
		double results = 0;
		//Sum all the weights of the nodes in the given position and layer
		for (int i = 0; i < size; i++){

			results = this->v_layers.at(position + 1).neurons.at(i).bias;

		}
		//Return the average value
		return (results / size);
	}
	
	//-----------------------------------------------------------------------------------------------------------
public:
	//For testing purposes
	vector<double> getOutput(){
		vector<double> results = vector<double>();

		for (int i = 0; i < this->v_layers.back().number_per_layer; i++){
			results.push_back(this->v_layers.back().neurons.at(i).output);
		}

		return results;
	}


	//-----------------------------------------------------------------------------------------------------------
	//Add New Layers and Neurons
	//-----------------------------------------------------------------------------------------------------------
	//Add a new layer before the position passed in 
	void addLayer(int position, int numberPerLayer);

	//Add a new neuron to a particular layer
	void addNeuronToLayer(int layerPosition);


	//-----------------------------------------------------------------------------------------------------------
	//Remove Layers and Neurons
	//-----------------------------------------------------------------------------------------------------------
	void removeLayer(int position);

	void removeNeuron(int layerPosition, int neuronPosition);

private: 
	//Checks if the current neuron is designated as temporarily removed
	bool checkNeuronRemoved(SNeuron &neuron){

		if (neuron.removed == 0){
			return false;
		}
		else{
			return true;
		}

	}

	bool isNeuronActivated(SNeuron &neuron){
		return neuron.output > this->neuron_activated ? true : false;
	}

	//-----------------------------------------------------------------------------------------------------------
	//Check Success
	//-----------------------------------------------------------------------------------------------------------

	//Check if the target and results match
	inline void updateSuccess(double *target){

		for (int i = 0; i < this->I_output; i++){
			
			//Unless all results equal the target result, it is a failure
			if (this->v_layers.back().neurons.at(i).output - .01 < target[i] < this->v_layers.back().neurons.at(i).output + .01){
				this->failure += 1;
				return;
			}
		}

		//They all match so it was a success
		this->success += 1;



	}

};

