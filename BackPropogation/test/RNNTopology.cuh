/*
Programmer: David Greenberg
Reason: Contains the topology for a single layer recurrent neural network. Access to the network values will be allowed through functions

*/
#pragma once
#include "TopologyBase.cuh"
#include "SpecialIterators.cuh"
#define CLASS_DEFINED_RRNTOPOLOGY
class RNNTopology :public TopologyBase{
public:
	enum HOST_DEVICE { HOST, DEVICE };
	
	
	
private:

	HOST_DEVICE host_or_device;//Says whether the Topology is for a host or for a network
	thrust::device_vector<WEIGHT_TYPE> device_weights;
	thrust::device_vector<WEIGHT_TYPE> device_output;
	thrust::device_vector<WEIGHT_TYPE> device_bias;
	thrust::device_vector<unsigned int> device_from;
	thrust::device_vector<unsigned int> device_to;
	
	thrust::host_vector<WEIGHT_TYPE> host_weights;
	thrust::host_vector<WEIGHT_TYPE> host_output;
	thrust::host_vector<WEIGHT_TYPE> host_bias;
	thrust::host_vector<unsigned int> host_from;
	thrust::host_vector<unsigned int> host_to;
	int network_info[END_INFO_TYPE];//Contains number of weights by type

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

	//Get info about network 
	int InfoInLayer(int layer, INFO_TYPE info_desired);
	
	//Retrieve the settings
	NSettings& getSettings(){
		return this->settings;
	}
	//*********************************************
	//Device Vectors
	//Retrieves the networks layers and positions
	//*********************************************

	TopologyLayerData getLayer(int layer);
	
	//*********************************************
	//Clean Up Topology
	//Remove the current values
	//*********************************************
	bool cleanTopology();
	bool emptyTopology();

	//*********************************************
	//CheckPoint
	//Output Values as a checkpoint
	//*********************************************
	std::ostream& createCheckpoint(std::ostream&);

};