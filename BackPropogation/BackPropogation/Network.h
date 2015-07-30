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






public:
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

	//Add a new layer before the position passed in 
	void addLayer(int position,int numberPerLayer);

	//Add a new neuron to a particular layer
	void addNeuronToLayer(int layer);

public:
	//For testing purposes
	vector<double> getOutput(){
		vector<double> results = vector<double>();

		for (int i = 0; i < this->v_layers.back().number_per_layer; i++){
			results.push_back(this->v_layers.back().neurons.at(i).output);
		}

		return results;
	}

};

