/*
Programmer: David Greenberg
Reason: The current header is an object designed to be overriden which contains the topology for the network. An Interface.

*/
#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "NSettings.h"
#include "Util.h"

#define WEIGHT_TYPE double
#define CLASS_DEFINED_TOPOLOGY_BASE //Define this class so it is only added once

struct TopologyLayerData{

	thrust::device_vector<WEIGHT_TYPE>::iterator weight_vector_begin;//Contains the list of weights
	thrust::device_vector<WEIGHT_TYPE>::iterator weight_vector_end;
	thrust::device_vector<unsigned int>::iterator from_vector_begin;//contains the list from
	thrust::device_vector<unsigned int>::iterator from_vector_end;
	thrust::device_vector<unsigned int>::iterator to_vector_begin;//Contains List to
	thrust::device_vector<unsigned int>::iterator to_vector_end;
	thrust::device_vector<WEIGHT_TYPE>::iterator device_output_vector_begin;//Contains list of output
	thrust::device_vector<WEIGHT_TYPE>::iterator device_output_vector_end;
	thrust::device_vector<WEIGHT_TYPE>::iterator device_bias_begin;//Contains list of the bias
	thrust::device_vector<WEIGHT_TYPE>::iterator device_bias_end;

	TopologyLayerData(){};

	TopologyLayerData(thrust::device_vector<WEIGHT_TYPE>::iterator _weight_vector_begin, thrust::device_vector<WEIGHT_TYPE>::iterator _weight_vector_end,
		thrust::device_vector<unsigned int>::iterator _from_vector_begin, thrust::device_vector<unsigned int>::iterator _from_vector_end,
		thrust::device_vector<unsigned int>::iterator _to_vector_begin, thrust::device_vector<unsigned int>::iterator _to_vector_end,
		thrust::device_vector<WEIGHT_TYPE>::iterator _device_output_vector_begin, thrust::device_vector<WEIGHT_TYPE>::iterator _device_output_vector_end,
		thrust::device_vector<WEIGHT_TYPE>::iterator _device_bias_begin, thrust::device_vector<WEIGHT_TYPE>::iterator _device_bias_end
		){
		weight_vector_begin = _weight_vector_begin;
		weight_vector_end = _weight_vector_end;
		from_vector_begin = _from_vector_begin;
		from_vector_end = _from_vector_end;
		to_vector_begin = _to_vector_begin;
		to_vector_end = _to_vector_end;
		device_output_vector_begin = _device_output_vector_begin;
		device_output_vector_end = _device_output_vector_end;
		device_bias_begin = _device_bias_begin;
		device_bias_end = _device_bias_end;
	};

};



class TopologyBase{
public:
	//Locations to retrieve data about the network from
	enum INFO_TYPE { INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, INPUT_WEIGHTS, HIDDEN_WEIGHTS, OUTPUT_WEIGHTS, END_INFO_TYPE };

protected:
	NSettings settings;//Contains the settings for the current topology

public:
	TopologyBase(){};
	//*********************************************
	//Network Construction
	//
	//*********************************************

	//Build the Topology of the Network based on the settings provided
	virtual bool buildTopology(NSettings settings) = 0;

	//*********************************************
	//Network Information
	//Retrieves Information about the network to provide 
	// to the training and running to allow them to function
	//*********************************************
	//Return the number of layers in the network
	virtual int numberLayers() = 0;


	//Get info about network
	virtual int InfoInLayer(int, INFO_TYPE) = 0;

	virtual NSettings& getSettings() = 0;
	//*********************************************
	//Device Vectors
	//Retrieves the networks layers and positions
	//*********************************************

	virtual TopologyLayerData getLayer(int layer) = 0;

	//*********************************************
	//Clean Up Topology
	//Remove the current values
	//*********************************************
	//Empty the topology entirely
	virtual bool cleanTopology() = 0;

	//Clean the network for a run
	virtual bool emptyTopology() = 0;

	//*********************************************
	//CheckPoint
	//Output Values as a checkpoint
	//*********************************************
	virtual std::ostream& createCheckpoint(std::ostream&) = 0;
};


//*********************
//Contains Functions to empty a vector
//*********************
namespace clean_device{
	//Function to free memory from GPU
	template<class T> void free(T &V) {
		V.clear();
		V.shrink_to_fit();
	}

	template void free<thrust::device_vector<int> >(thrust::device_vector<int>& V);
	template void free<thrust::device_vector<double> >(
		thrust::device_vector<double>& V);
}
