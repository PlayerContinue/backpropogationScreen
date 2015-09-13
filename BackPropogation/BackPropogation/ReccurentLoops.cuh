#pragma once
#include <string.h>
#include "CSettings.h"
#include "RecurrentNeuralNetwork.cuh"
#include "structures_cuda.cuh"
#define weight_type double
#define VISUALIZE this->mainNetwork.VisualizeNetwork();
//****************************************************************************************************
//
//Programmer: David Greenberg
//Purpose: Contains the function required to train and run a RecurrentNeuralNetwork
//
//****************************************************************************************************

class ReccurentLoops
{
	//*********************
	//Class Variables
	//*********************
private:
	//typedef float weight_type;
	RecurrentNeuralNetwork mainNetwork;
	CSettings settings;
	SCheckpoint checkpoint;
	weight_type** input;
	weight_type** output;

	//*********************
	//Constructors
	//*********************
public:
	//Constructor for an empty network
	//Creates a settings object from defaults
	ReccurentLoops();

	//Constructor for an network created from user settings
	ReccurentLoops(CSettings settings);

	//Constructor for an object with settings and a checkpoint
	ReccurentLoops(CSettings settings, SCheckpoint checkpoint);

private:
	//*********************
	//Initialize Internal Structure
	//*********************

	void InitializeNetwork();

	//*********************
	//Load Network From File
	//*********************
	bool loadNetworkFromFile();

	//*********************
	//Utilization
	//*********************
	bool runNetwork();

	//*********************
	//Training
	//*********************
public:
	void startTraining();

private:
	//Training data is passed in
	bool train_network();

	//Retrieve the training data from the file passed in by the settings
	bool load_training_data_from_file();

};

