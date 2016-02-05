/*
Programmer: David Greenberg
Reason: Contains the topology for a single layer recurrent neural network. Access to the network values will be allowed through functions

*/
#pragma once
#include "TopologyBase.cuh"
#include "SpecialIterators.cuh"
#define CLASS_DEFINED_RRNTOPOLOGY
#define WEIGHT_TYPE double
class RNNTopology :TopologyBase{
public:
	enum HOST_DEVICE { HOST, DEVICE };

private:

	HOST_DEVICE host_or_device;//Says whether the Topology is for a host or for a network
	thrust::device_vector<WEIGHT_TYPE> device_weights;
	thrust::device_vector<unsigned int> device_from;
	thrust::device_vector<unsigned int> device_to;
	thrust::host_vector<WEIGHT_TYPE> host_weights;
	thrust::host_vector<unsigned int> host_from;
	thrust::host_vector<unsigned int> host_to;

public:
	//*********************************************
	//Constructor/Destructor
	//*********************************************
	RNNTopology();
	//Constructor Telling if the network is host or device
	RNNTopology(HOST_DEVICE);

	//*********************************************
	//Network Construction
	//Construct the network from the settings
	//*********************************************
	//Build the Topology of the Network based on the settings provided
	bool buildTopology(NSettings settings);

	//*********************************************
	//Network Information
	//Retrieves Information about the network to provide 
	// to the training and running to allow them to function
	//*********************************************
	//Return the number of layers in the network
	int numberLayers();

	//Get number of nodes in a particular layer
	int numberNodesInLayer(int layer);

	//*********************************************
	//Device Vectors
	//Retrieves the networks layers and positions
	//*********************************************

	
	TopologyLayerData getLayer(int layer);
	


};