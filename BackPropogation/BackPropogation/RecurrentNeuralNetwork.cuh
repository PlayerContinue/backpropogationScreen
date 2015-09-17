#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <vector>
#include <time.h>
#include "util.h"
#ifdef __X_H_INCLUDED__

#else
#define __X_H_INCLUDED__
#include "CSettings.h"
#include "NetworkBase.cuh"
#endif
using namespace thrust;
using namespace thrust::placeholders;
//Define a type so it can use either double or float, depending on what turns out to be better
#define weight_type thrust::complex<double>
//****************************************************************************************************
//
//Programmer: David Greenberg
//Purpose: Contains all the functions and methods to train and alter a ReccurentNeuralNetwork
//Initial Version utilizes Long Short Term Memory
//
//****************************************************************************************************


//Contains the methods
class RecurrentNeuralNetwork:public NetworkBase {
	//*********************
	//Class Variables
	//*********************
private:

	vector<long> positionOfLastWeightToNode;
	long numberOfNodes; //The number of nodes currently in the system which can be linked to
	long numberNonWeights; //Keeps track of the number of non-weights before an actual weight appears
	long input_weights;
	//Stores the weights between neurons
	host_vector<weight_type> weights;
	//Stores the weights in GPU Memory
	thrust::device_vector<weight_type> GPUWeights;

	//Stores the values of the neurons
	host_vector<weight_type> output_values;

	//Stores the values of the neuron in GPU Memory
	thrust::device_vector<weight_type> GPUOutput_values;
	thrust::device_vector<weight_type> GPUPreviousOutput_Values;

	//Stores the delta in GPU Memory
	host_vector<weight_type> host_deltas;
	thrust::device_vector<weight_type> device_deltas;
	
	//Stores the total error
	weight_type total_error;
	


	//Contains whuch weight is connected to which neuron
	host_vector<int> mapTo;
	host_vector<int> mapFrom;

	//Stores it in gpu_memory
	thrust::device_vector<int> GPUMapTo;
	thrust::device_vector<int> GPUMapFrom;


	CSettings settings;
public:
	//*********************
	//Constructors
	//*********************
	//Default Constructor
	//Creates a network with 1 input and 1 output
	RecurrentNeuralNetwork();

	//Constructor which asks for a settings object 
	//The settings object contains all the information required to perform a function
	RecurrentNeuralNetwork(CSettings& settings);
	//*********************
	//Initialization
	//*********************
private:
	//Initialize the network from the settings object if possible
	void initialize_network();

public:
	//*********************
	//Run The Network
	//*********************
	device_vector<weight_type> runNetwork(weight_type* in);

	//***************************
	//Train the Network
	//***************************

	//Initilialize the network for training
	virtual void InitializeTraining(){
		this->InitializeHessianFreeOptimizationTraining();
	}
	//Run a round of training
	virtual void StartTraining(weight_type* in, weight_type* out){
		this->HessianFreeOptimizationTraining(in, out);
	}
	//Apply the error to the network
	virtual void ApplyError(){
		this->HessianFreeOptimizationApplyError();
	}

private:
	void LongShortTermMemoryTraining(device_vector<weight_type> in, weight_type* out);

	//Set up for HessianFreeoptimization
	void InitializeHessianFreeOptimizationTraining();
	void HessianFreeOptimizationTraining(weight_type* in, weight_type* out);
	void HessianFreeOptimizationApplyError();

	//Set up for Real Time Training
	void InitializeRealTimeRecurrentTraining();
	void RealTimeRecurrentLearningTraining(weight_type* in, weight_type* out);
	//Modify the weights of a RealTimeRecurrentLearning Algorithm
	void RealTimeRecurrentLearningApplyError();
private:
	//Retrieve the error of a single step
	weight_type RealTimeRecurrentLearningTraining(weight_type* in, weight_type* out,weight_type total_error, thrust::device_vector<int> &GPUMapTo,
		thrust::device_vector<int> &GPUMapFrom, thrust::device_vector<weight_type> &GPUWeights, thrust::device_vector<weight_type> &GPUOutput_values, thrust::device_vector<weight_type> &GPUPreviousOutput_Values,
		thrust::device_vector<weight_type> &deltas);
public:
	
	//*********************
	//Visualization
	//*********************

	//Only used for dubug. Outputs a simple example of what the network looks like
	void VisualizeNetwork();
	ostream& OutputNetwork(ostream &os);
	//***************************
	//Modify Structure Of Neuron
	//***************************
	void addNeuron(int numberNeuronsToAdd);

	//Add a new weight between neurons
	void addWeight(int numberWeightsToAdd);
private:
	//Decide which node the new weight should be attached to 
	int decideNodeToAttachTo();
	//Decide which node the new weight should be attached from
	//Requires knowing which node it will be attaching to in order to avoid double connections
	int decideNodeToAttachFrom(int attachTo);

	//Get a new weight
	weight_type getNewWeight();

	//***************************
	//Perform Functionality
	//***************************
	//Finds the sum of the current values in the network
	//Updates the network values
	void sumNetworkValues(device_vector<weight_type> &GPUOutput_values,//Copy the output_nodes
		device_vector<weight_type> &GPUPreviousOutput_Values,
		device_vector<int> &GPUMapFrom,//Copy the map from
		device_vector<int> &GPUMapTo, //Copy the mapTo
		device_vector<weight_type> &GPUWeights, int number_of_rounds
		);

public:
	void ResetSequence();

	//Copies the information stored on the GPU into main memory
	void CopyToHost();

	//Copies the information stored in Main Memory into GPU Memory
	void CopyToDevice();

	//Copies information stored on the GPU memory into the Main Memory
	//Removes the GPU Memory copies
	void cleanNetwork();
};