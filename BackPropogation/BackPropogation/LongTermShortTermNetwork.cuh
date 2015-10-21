#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <thrust/sort.h>
#ifndef __TESTCODE_CUH_INCLUDED___
#include "testcode.cuh"
#define __TESTCODE_CUH_INCLUDED___
#endif

#ifndef __FUNCTORS__H__INCLUDED__
#include "Recurrent_Functors.cuh"
#define __FUNCTORS__H__INCLUDED__
#endif
#include <vector>
#ifndef __TIME_H_INCLUDED__
#include <time.h>
#define __TIME_H_INCLUDED__
#endif
#include "memory_block.cuh"
#include "util.h"

#ifdef __IOSTREAM_H_INCLUDED__

#else
#include <iostream>
#define  __IOSTREAM_H_INCLUDED__
#endif

#ifdef __X_H_INCLUDED__

#else
#define __X_H_INCLUDED__
#include "CSettings.h"
#include "NetworkBase.cuh"
#endif

//#define TEST_DEBUG

#ifdef TEST_DEBUG
#include "TestCode.cuh"
#endif

using namespace thrust;
using namespace thrust::placeholders;
//Define a type so it can use either double or float, depending on what turns out to be better
#ifndef weight_type
#define weight_type double
#endif

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
	
	int numberNonWeights; //Keeps track of the number of non-weights before an actual weight appears
	long last_output_cell_pos;
	long last_memory_cell_pos;
	long last_input_cell_pos;
	enum cell_type{MEMORY_CELL, POTENTIAL_MEMORY_CELL,INPUT_CELL,OUTPUT_CELL,FORGET_CELL,NONE_CELL};
	//Stores the weights between neurons
	host_vector<weight_type> weights;
	vector<unsigned int> numberOfWeightsInLayers;
	//Stores the weights in GPU Memory
	thrust::device_vector<weight_type> GPUWeights;
	thrust::device_vector<weight_type> GPUPreviousWeights;



	//Vectors for the inputs
	host_vector<weight_type> output_bias;

	//Stores the values of the neurons
	host_vector<weight_type> bias;

	//Stores the bias in the GPU
	thrust::device_vector<weight_type> GPUBias;
	thrust::device_vector<weight_type> GPUPreviousBias;
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

	//Stores the order of the values such that they can be added together by reduce by key
	thrust::device_vector<int> positionToSum;
	thrust::device_vector<int> count;

	//Container for output values
	thrust::device_vector<weight_type> RealOutput;

	//Vector Containing Layer Info
	vector<vector<Memory_Block>> mBlocksLayers;

	
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

	//Create a LongTermShortTermNetwork from a checkpoint
	LongTermShortTermNetwork(CSettings& settings,bool checkpoint);
	//*********************
	//Destructor
	//*********************
	~LongTermShortTermNetwork();
	//*********************
	//Initialization
	//*********************
private:
	//Initialize the network from the settings object if possible
	void initialize_network();
	void count_weights_in_layers();
public:
	//*********************
	//Run The Network
	//*********************
	virtual device_vector<weight_type> runNetwork(weight_type* in);
	device_vector<weight_type> runNetwork(weight_type* in,int number_extra_weights);
	virtual device_vector<weight_type> runNetwork(weight_type* in, run_type type);
	void InitializeLongShortTermMemoryForRun();
	void InitializeRun(){
		this->InitializeLongShortTermMemoryForRun();
	}
	//***************************
	//Train the Network
	//***************************

	//Initilialize the network for training
	virtual void InitializeTraining(){
		this->InitializeLongShortTermMemory();
	}

	virtual void StartTraining(weight_type* in, weight_type* out){
		//Does nothing for the moment
	}

	//Run a round of training
	virtual void StartTraining(weight_type** in, weight_type* out){
		//this->LongShortTermMemoryTraining(in, out);
	}

	virtual void StartTraining(weight_type** in, weight_type** out){
		this->LongShortTermMemoryTraining(in, out);
	}
	//Apply the error to the network
	virtual void ApplyError(){
		this->ApplyLongTermShortTermMemoryError();
	}

private:
	//Add the input into the GPU_Weight_objects
	void setInput(weight_type* in);
	void setInput(weight_type** in);
	//Set the training network such that the input is the sum of the results
	void averageWeights();
	//Inititalize the Network For training
	void InitializeLongShortTermMemory();
	//Unroll the network into a multilayer representation
	void UnrollNetwork(int numLayers);
	

	//Train the network using Backpropogation through time
	void LongShortTermMemoryTraining(weight_type** in, weight_type** out);
	//Find the delta values of the current output from the expected gradiant
	void FindBackPropDelta(weight_type** out, int current_layer);

	//Apply the error
	void ApplyLongTermShortTermMemoryError();

	//Apply the error to the bias
	void ApplyErrorToBias();

	//Combine these two function, they do the same thing
	template <typename T>
	void specialCopyToNodes(int start_output, int number_output, device_vector<T> &GPUWeightVector, device_vector<int> &toPosition, device_vector<int> &fromPosition, host_vector<T> &weights, host_vector<int> map);
	
	template <typename T>
	void copyNodesToDevice(device_vector<T> &GPU_Vector, device_vector<int> &fromPosition, host_vector<T> &local_host_Vector, host_vector<int> host_from_vector);
public:

	//*********************
	//Visualization
	//*********************

	//Only used for dubug. Outputs a simple example of what the network looks like
	void VisualizeNetwork();
	virtual ostream& OutputNetwork(ostream &os);
	virtual istream& LoadNetwork(istream& is);
	//***************************
	//Modify Structure Of Neuron
	//***************************

	//Creates a new memory block with connections to all inputs
	void InitialcreateMemoryBlock(int numberMemoryCells);
	void createMemoryBlock(int numberMemoryCells,int layer_num);

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

public:
	void ResetSequence();

	//Copies the information stored on the GPU into main memory
	void CopyToHost();

	//Copies the information stored in Main Memory into GPU Memory
	void CopyToDevice();

	//Copies information stored on the GPU memory into the Main Memory
	//Removes the GPU Memory copies
	void cleanNetwork();

	//Empty all data from memory
	void emptyGPUMemory();
private:
	//Load the bias into the system
	void moveBiasToGPU();
	void moveBiasToGPU(bool add_memory_cells);
	//Load a single layer from Host memory to device memory
	void loadLayerToDevice(unsigned int j);
	//Unroll a row into the network
	void loadUnrolledToDevice(int unrolled, unsigned int layer);
	//Get Number Memory Cells In A Layer
	unsigned int getNumberMemoryCells(unsigned int layer);
	//Number weights in a cell
	unsigned int getNumberWeightsInLayer(unsigned int layer);
	//number weights of a certain type in a layer
	unsigned int getNumberTypeWeightsInLayer(unsigned int layer, cell_type cell);
	//Get the permutation of the order for summing the layers
	void getSumPermutation();
	//***************************
	//Overload Functions
	//***************************

	friend ostream& operator<<(ostream &os, const LongTermShortTermNetwork &network){
		cout.precision(20);
		os << network.numberOfNodes << endl;
		os << network.numberNonWeights << endl;

		os << "number_weights_in_layers" << endl;
		os << network.numberOfWeightsInLayers.size() << endl;
		for (int i = 0; i < network.numberOfWeightsInLayers.size(); i++){
			os << network.numberOfWeightsInLayers[i] << endl;
		}
		os << endl;

		os << network.mBlocksLayers.size() << endl;//Get number of layers
		for (unsigned int j = 0; j < network.mBlocksLayers.size(); j++){
			os << network.mBlocksLayers[j].size() << endl;
			for (unsigned int i = 0; i < network.mBlocksLayers[j].size(); i++){
				os << network.mBlocksLayers[j][i] << endl;
			}
			os << "layer_" << j << endl;
		}
		os << endl;
		os << endl;
		os << "GPUWeights" << endl;
		os << network.GPUWeights.size() << endl;
		for (unsigned int i = 0; i < network.GPUWeights.size(); i++){
			os << (weight_type)network.GPUWeights[i] << endl;
		}
		os << endl;
		os << endl;

		os << "deltas" << endl;
		os << network.device_deltas.size() << endl;
		for (unsigned int i = 0; i < network.device_deltas.size(); i++){
			os << (weight_type)network.device_deltas[i] << endl;
		}

		os << endl;
		os << endl;

		os << "GPU_Values" << endl;
		os << network.GPUOutput_values.size() << endl;
		//Output the current output values
		for (unsigned int i = 0; i < network.GPUOutput_values.size(); i++){
			os << (weight_type)network.GPUOutput_values[i]  << endl;
		}

		os << endl;
		os << endl;

		os << "GPUPrevious_Values" << endl;
		os << network.GPUPreviousOutput_Values.size() << endl;
		//Output the current output values
		for (unsigned int i = 0; i < network.GPUPreviousOutput_Values.size(); i++){
			os <<  (weight_type)network.GPUPreviousOutput_Values[i] << endl;
		}

		os << endl;
		os << endl;


		os << "GPUPreviousWeights" << endl;
		os << network.GPUPreviousWeights.size() << endl;
		//Output the current output values
		for (unsigned int i = 0; i < network.GPUPreviousWeights.size(); i++){
			os << (weight_type)network.GPUPreviousWeights[i] << endl;
		}

		os << endl;
		os << endl;

		os << "(from,to)" << endl;
		os << network.GPUMapFrom.size() << endl;
		for (unsigned int i = 0; i < network.GPUMapFrom.size(); i++){
			os << network.GPUMapFrom[i] << " " << network.GPUMapTo[i] << endl;
		}
		os << endl;
		os << endl;

		os << "SumOrder" << endl;
		os << network.positionToSum.size() << endl;
		thrust::copy(network.positionToSum.begin(), network.positionToSum.end(), std::ostream_iterator<int>(os, " "));

		os << endl;

		os << "count_orders" << endl;
		os << network.count.size() << endl;
		thrust::copy(network.count.begin(), network.count.end(), std::ostream_iterator<int>(os, " "));


		os << endl;
		os << endl;


		os << endl;
		os << "GPU_BIAS" << endl;
		os << network.GPUBias.size() << endl;
		for (unsigned int i = 0; i < network.GPUBias.size(); i++){
			os << (weight_type)network.GPUBias[i] << " ";
		}
	


		os << endl;
		os << endl;

		os << endl;
		os << "Prev_GPU_BIAS" << endl;
		os << network.GPUPreviousBias.size() << endl;
		for (unsigned int i = 0; i < network.GPUPreviousBias.size(); i++){
			os << (weight_type)network.GPUPreviousBias[i] << " ";
		}



		os << endl;
		os << endl;


		return os;
	}

};