#pragma once
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
//Purpose: Contains the functions and information required to run a LongTermShortTerm Network
//
//****************************************************************************************************
class LongTermShortTermNetwork : public NetworkBase{

	//*********************
	//Class Variables
	//*********************
private:

	vector<long> positionOfLastWeightToNode;
	long numberOfNodes; //The number of nodes currently in the system which can be linked to
	long numberNonWeights; //Keeps track of the number of non-weights before an actual weight appears
	long input_weights;
	long last_output_cell_pos;
	long last_memory_cell_pos;
	long last_input_cell_pos;
	//Stores the weights between neurons
	host_vector<weight_type> weights;
	//Stores the weights in GPU Memory
	thrust::device_vector<weight_type> GPUWeights;

	//Vectors for the inputs
	vector<weight_type> input_bias;
	vector<weight_type> input_weights;
	vector<int> input_mapTo;
	vector<int> input_mapFrom;

	//Vectors for the inputs
	vector<weight_type> output_bias;
	vector<weight_type> output_weights;
	vector<int> output_mapTo;
	vector<int> output_mapFrom;

	//Vectors for the inputs
	vector<weight_type> memory_cell_bias;
	vector<weight_type> memory_cell_weights;
	vector<int> memory_cell_mapTo;
	vector<int> memory_cell_mapFrom;

	//Stores the values of the neurons
	host_vector<weight_type> bias;

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

	 

	//Contains a list of iterators for the position of the last weight for a node
	vector<thrust::host_vector<weight_type>::iterator> weight_position;
	vector<thrust::host_vector<int>::iterator> mapTo_position;
	vector<thrust::host_vector<int>::iterator> mapFrom_position;

	//Vector Containing Layer Info
	vector<long> layerPosition;

	CSettings settings;
public:
	//*********************
	//Constructors
	//*********************
	//Default Constructor
	//Creates a network with 1 input and 1 output
	LongTermShortTermNetwork();

	//Constructor which asks for a settings object 
	//The settings object contains all the information required to perform a function
	LongTermShortTermNetwork(CSettings& settings);
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
		this->InitializeLongShortTermMemory();
	}
	//Run a round of training
	virtual void StartTraining(weight_type* in, weight_type* out){
		this->LongShortTermMemoryTraining(in, out);
	}
	//Apply the error to the network
	virtual void ApplyError(){
		this->ApplyLongTermShortTermMemoryError();
	}

private:
	void InitializeLongShortTermMemory();
	void LongShortTermMemoryTraining(weight_type* in, weight_type* out);
	void ApplyLongTermShortTermMemoryError();
public:

	//*********************
	//Visualization
	//*********************

	//Only used for dubug. Outputs a simple example of what the network looks like
	void VisualizeNetwork();

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
	
	//Creates a new memory block with connections to all inputs
	void InitialcreateMemoryBlock(int numberMemoryCells);
	void createMemoryBlock(int numberMemoryCells);

	void addNewNeuron(int store,int position, weight_type weight, int mapFrom, int mapTo);
	void addNewPositionInList();

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