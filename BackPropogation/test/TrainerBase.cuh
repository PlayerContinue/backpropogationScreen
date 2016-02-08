/*
Programmer: David Greenberg
Reason: Contains the base functions required to train a network. Is an abstract class meant to be extended

*/
#pragma once
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include "NSettings.h"
#include "TopologyBase.cuh"
#include "RNNTopology.cuh"

class TrainerBase{
	friend class RunnerBase;

protected:
	TopologyBase* _topology;//The topology of the network

public:
	//*********************************************
	//Contructor
	//
	//*********************************************
	TrainerBase(){};

	//*********************************************
	//Create Training Enviornment
	//
	//*********************************************
	//Create the training enviornment before training starts
	virtual bool createTrainingEnviornment(TopologyBase&,NSettings) = 0;

	
	//*********************************************
	//Run Training
	//Train the network on the given input, output pair
	//*********************************************
	virtual void train(thrust::host_vector<WEIGHT_TYPE> input, thrust::host_vector<WEIGHT_TYPE> output) = 0;
	virtual void train(thrust::device_vector<WEIGHT_TYPE> input, thrust::device_vector<WEIGHT_TYPE> output) = 0;

	//*********************************************
	//Clean Up Topology
	//Remove the current values
	//*********************************************
	//Empty the topology entirely
	virtual bool cleanTrainer() = 0;

	//Clean the network for a run
	virtual bool emptyTrainer() = 0;

	//*********************************************
	//Checkpoint
	//Create a Checkpoint of the Training Enviornment
	//*********************************************
	virtual std::ostream& createCheckpoint(std::ostream&) = 0;
};

