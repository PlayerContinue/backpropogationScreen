/*
Programmer: David Greenberg
Reason: The current header is an object designed to be overriden which contains the topology for the network. An Interface.

*/
#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <time.h>
#ifndef CLASS_DEFINED_NSETTINGS
#include "NSettings.h"
#endif
#ifndef CLASS_DEFINED_UTIL
#include "Util.h"
#endif
#define CLASS_DEFINED_TOPOLOGY_BASE //Define this class so it is only added once

struct TopologyLayerData{

	thrust::device_vector<double>::iterator weight_vector_begin;//Contains the list of weights
	thrust::device_vector<double>::iterator weight_vector_end;//Contains the list of weights
	thrust::device_vector<unsigned int>::iterator from_vector_begin;//Contains the list of weights
	thrust::device_vector<unsigned int>::iterator from_vector_end;//Contains the list of weights
	thrust::device_vector<unsigned int>::iterator to_vector_begin;//Contains the list of weights
	thrust::device_vector<unsigned int>::iterator to_vector_end;//Contains the list of weights

	TopologyLayerData(thrust::device_vector<double>::iterator _weight_vector_begin, thrust::device_vector<double>::iterator _weight_vector_end,
		thrust::device_vector<unsigned int>::iterator _from_vector_begin, thrust::device_vector<unsigned int>::iterator _from_vector_end,
		thrust::device_vector<unsigned int>::iterator _to_vector_begin, thrust::device_vector<unsigned int>::iterator _to_vector_end){
		weight_vector_begin = _weight_vector_begin;
		weight_vector_end = _weight_vector_end;
		from_vector_begin = _from_vector_begin;
		from_vector_end = _from_vector_end;
		to_vector_begin = _to_vector_begin;
		to_vector_end = _to_vector_end;
	};

};


class TopologyBase{
	
public:
	NSettings settings;//Contains the settings for the current topology
public:
	TopologyBase();
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
	virtual int numberLayers()=0;

	//Get number of nodes in a particular layer
	virtual int numberNodesInLayer(int layer) = 0;

	//*********************************************
	//Device Vectors
	//Retrieves the networks layers and positions
	//*********************************************


	virtual TopologyLayerData getLayer(int layer)=0;




};


