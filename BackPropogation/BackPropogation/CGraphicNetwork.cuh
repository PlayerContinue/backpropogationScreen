//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains a backpropagation algorithm for a neural network
//----------------------------------------------------------------------------------------
#define LOCKED .000000000000000000001
#pragma once
#include <vector>
#include <time.h>
#include <math.h>
#include <cmath>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <fstream>
#include "util.h"
#include "structures_cuda.cuh"
#include "CudaCalculations.cuh"
#include "CSettings.h"
#if defined(TRIAL2) || defined(TRIAL1)|| defined(TRIAL3) || defined(TRIAL4) || defined(TRIAL5)
#include <iostream>
#endif

using namespace std;

class CGraphicsNetwork
{

private:
	//Networks Variables
	//The number of layers
	int v_num_layers;

	//Vector containing the layers of neurons
	vector<SNeuronLayer> v_layers;

	//Keep track of total number of nodes
	int total_num_nodes=0;

	//The learning threshold for the current network
	double threshold;
	
	//Learning rate (look up)
	double beta;

	//momentum (Look up)
	double alpha;

	//Activation Threshold
	const double neuron_activated = .1;

	//Success Rate
	int success = 0;
	int previousSuccess=0;

	//Failure Rate
	int failure = 0;
	int previousFailure=1;

	//Average distance 
	double total_distance;

	double previous_average_distance=0;

	double average_delta=0;





#ifdef FULL_SUCCESS
	//Temporary current, may remove
	//Keep track of all full successes
	int full_success = 0;
	int full_failure = 0;
	//Keep track of previous full success
	int prev_full_success = 0;
	int prev_full_failure = 1;
#endif
	//Number of Outputs
	int I_output;

	//Number of inputs
	int I_input;

public:
	//Store a settings object
	CSettings* settings;


public:
	//-----------------------------------------------------------------------------------------------------------
	//Constructors
	//-----------------------------------------------------------------------------------------------------------
	CGraphicsNetwork();

	//Create only with a link to settings
	CGraphicsNetwork(CSettings* settings);

	//Constructor 
	//sizes - The number of neurons per layer
	//i.e. 3 inputs, first hidden layer 2, second 4 would, output 1 is [3,2,4,1]
	CGraphicsNetwork(vector<int> &sizes);
	//Constructor 
	//sizes - The number of neurons per layer
	//i.e. 3 inputs, first hidden layer 2, second 4 would, output 1 is [3,2,4,1]
	CGraphicsNetwork(vector<int> &sizes, double beta, double alpha);

	//Develop a initial network from a settings object
	CGraphicsNetwork(vector<int> &sizes, CSettings* settings);

	//-----------------------------------------------------------------------------------------------------------
	//Overloaded Operators
	//-----------------------------------------------------------------------------------------------------------

	friend ostream& operator<<(ostream& os, const CGraphicsNetwork& network);
	friend istream& operator>>(istream& os, CGraphicsNetwork& network);

	CGraphicsNetwork& operator=(const CGraphicsNetwork& network){
		this->v_num_layers = network.v_num_layers;
		this->alpha = network.alpha;
		this->beta = network.beta;
		this->I_input = network.I_input;
		this->I_output = network.I_output;
		this->v_layers = network.v_layers;
		this->total_num_nodes = network.total_num_nodes;
		return *this;

	}
	//-----------------------------------------------------------------------------------------------------------
	//Main Learning And Teaching Functions
	//-----------------------------------------------------------------------------------------------------------
	

	//Feed Forward one set of inputs 
	void feedForward(double *in);

	/*Return a vector "(nabla_b, nabla_w)" representing the
	gradient for the cost function C_x.  "nabla_b" and
	"nabla_w" are layer - by - layer lists of numpy arrays, similar
	to "self.biases" and "self.weights".*/
	void backprop(double *in, double *tgt);

	//-----------------------------------------------------------------------------------------------------------
	//Output
	//-----------------------------------------------------------------------------------------------------------

private:


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


	//-----------------------------------------------------------------------------------------------------------

	//Retrive the average of the next layers weights
	double inline average_of_next_weights(int position, int nodePosition){
		int size = this->v_layers[position + 1].number_per_layer;//Get number of neurons in next layer
		double results = 0;
		//Sum all the weights of the nodes in the given position and layer
		for (int i = 0; i < size; i++){

			results = this->v_layers[position + 1].neurons[i].weights[nodePosition];

		}
		//Return the average value
		return (results / size);
	}

	//Retrive the average of the next layers weights
	double inline average_of_bias(int position){
		int size = this->v_layers[position + 1].number_per_layer;//Get number of neurons in next layer
		double results = 0;
		//Sum all the weights of the nodes in the given position and layer
		for (int i = 0; i < size; i++){

			results = this->v_layers[position + 1].neurons[i].bias;

		}
		//Return the average value
		return (results / size);
	}

	//-----------------------------------------------------------------------------------------------------------
public:
	

	//-----------------------------------------------------------------------------------------------------------
	//Add New Layers and Neurons
	//-----------------------------------------------------------------------------------------------------------
	//Add a new layer before the position passed in 
	void addLayer(int position, int numberPerLayer);

	//Add a new neuron to a particular layer
	void addNeuronToLayer(int layerPosition, int layerPositionEnd,int numToAdd);


	//-----------------------------------------------------------------------------------------------------------
	//Remove Layers and Neurons
	//-----------------------------------------------------------------------------------------------------------
	void removeLayer(int position);

	void removeNeuron(int layerPosition, int neuronPosition);

	void reloadNetwork();


private:
	//Checks if the current neuron is designated as temporarily removed
	bool checkNeuronRemoved(SNeuron &neuron){

		if (neuron.removed == 1){
			return true;
		}
		else{
			return false;
		}

	}

	bool checkNeuronLocked(SNeuron &neuron){
		return (neuron.removed == 2);
	}

	bool isNeuronActivated(SNeuron &neuron){
		//return neuron.output > this->neuron_activated ? true : false;
		return false;
	}

	//-----------------------------------------------------------------------------------------------------------
	//Check Success
	//-----------------------------------------------------------------------------------------------------------

	//Check if the target and results match
	inline void updateSuccess(double *target){
		bool fail = false;
		for (int i = 0; i < this->I_output; i++){

			//Unless all results equal the target result, it is a failure
			if (abs(target[i] - this->v_layers.back().output[i]) > .00000000000001){
				this->failure += 1;
				//Add the average distance
				total_distance += abs(target[i] - this->v_layers.back().output[i]);
				fail = true;
			}
			else{
				//They all match so it was a success
				this->success += 1;
			}
		}
#ifdef FULL_SUCCESS
		if (fail){//A failure of at least one of the numbers has occured, mark it
			this->full_failure++;
		}
		else{
			this->full_success++;
		}
#endif

		return;
	}

	//-----------------------------------------------------------------------------------------------------------
	//Reset Network Counts
	//-----------------------------------------------------------------------------------------------------------
public:
	void resetNetwork(){
		//Reset the activated count in the neurons
		for (int layerNum = 0; layerNum < this->v_num_layers; layerNum++){
			for (int neuronNum = 0; neuronNum < this->v_layers[layerNum].number_per_layer; neuronNum++){
				this->v_layers[layerNum].neurons[neuronNum].activated = 0;
			}
		}
		
		//Grab previous distance
		this->previous_average_distance = (this->total_distance / ((((double)this->success + (double)this->failure))));
		//Reset Previous Distance
		this->total_distance = 0;

		//Reset previous and current success
		this->previousSuccess = this->success;
		this->previousFailure = this->failure;
		this->success = 0;
		this->failure = 0;

#ifdef FULL_SUCCESS
		this->prev_full_success = this->full_success;
		this->full_success = 0;
		this->prev_full_failure = this->full_failure;
		this->full_failure = 0;
#endif
	}

	//-----------------------------------------------------------------------------------------------------------
	//Get And Set 
	//-----------------------------------------------------------------------------------------------------------

	//Return the Mean Square Error
	double getMeanSquareError(double **in, double **tgt, int size){
		double sum = 0; //Stores the sum
		double* output;


		for (int i = 0; i < size; i++){
			//Feed the input value in to get the output
			this->feedForward(in[i]);
			output = getOutputArray<double>();
			for (int j = 0; j < this->I_output;j++){
				//Add the sum of the (target - output)^2 

				sum += square_means_sums<double>(tgt[i],output,this->I_output);
			}
		}
		free(output);
		//Return (1/number of total ouputs) * sum
		return ((1.0/(size*this->I_output)) * sum);

	}

	int getI_input(){
		return this->I_input;
	}

	int getI_output(){
		return this->I_output;
	}

	//Return the Success percentage
	double getSuccessRate(){
		return ((double)this->success / (((double)this->success + (double)this->failure)));
	}

	//Return the previous Success Rate
	double getPreviousSuccessRate(){
		return ((double)this->previousSuccess / ((((double)this->previousSuccess + (double)this->previousFailure))));
	}

#ifdef FULL_SUCCESS
	//Percentage of times all of the targets were met

	//Get the full success rate
	double getFullSuccessRate(){
		return ((double)this->full_success / ((double)this->full_success + (double)this->full_failure));
	}

	//Return the previous Success Rate
	double getFullPreviousSuccessRate(){
		return ((double)this->prev_full_success / (((double)this->prev_full_success + (double)this->prev_full_failure)));
	}
#endif

	//Return the average change
	double getAverageDelta(){
		double numberNeurons = 0;
		double sum = 0;
		//Get number of neurons
		for (int i = 1; i < this->v_num_layers; i++){
			numberNeurons += this->v_layers[i].number_per_layer;
			for (int j = 0; j < (int) this->v_layers[i].delta.size(); j++){
				sum += abs(this->v_layers[i].delta[j]);
			}
		}
		return sum / numberNeurons;
	}

	//Return the average distance
	double getAverageDistance(){
		return (this->total_distance / (((double)this->success + (double)this->failure)));
		
	}

	int getTotalDistance(){
		return this->total_distance;
	}

	//Return the average distance
	double getPreviousAverageDistance(){
		return this->previous_average_distance;
	}

	//Retrieve the number of layers
	int getNumLayers(){
		return this->v_num_layers;
	}

	//Retrieve the number of neurons in a given layer
	int getNumNeuronsInLayer(int layerPosition){
		return this->v_layers[layerPosition].number_per_layer;
	}

	//Get the outputs of the current run
	vector<double> getOutput(){
		vector<double> results = vector<double>();

		for (int i = 0; i < this->v_layers.back().number_per_layer; i++){
			results.push_back(this->v_layers.back().output[i]);
		}

		return results;
	}
	template<typename T>
	T* getOutputArray(){
		T* results = new T[this->I_output];
		for (int i = 0; i < this->v_layers.back().number_per_layer; i++){
			results[i] = (T) this->v_layers.back().output[i];
		}
		return results;
	}

};

