#pragma once
#include <string.h>
#include <vector>
#include "CSettings.h"
//#include "NetworkBase.cuh"
#include "RecurrentNeuralNetwork.cuh"
#include "CRecurrentCheckpoint.h"
#include <thrust/complex.h>
#define weight_type thrust::complex<double>
#define RETURN_WEIGHT_TYPE  thrust::complex<double>
//****************************************************************************************************
//
//Programmer: David Greenberg
//Purpose: Contains the function required to train and run a RecurrentNeuralNetwork
//
//****************************************************************************************************

class ReccurentLoops
{
	//*********************
	//Constant Class Variables
	//*********************
	//Decide type of training to use
public:
	static const int RealTimeTraining=0;
	static const int HessianFreeOptimization = 1;
	//*********************
	//Class Variables
	//*********************
private:
	//typedef float weight_type;
	NetworkBase* mainNetwork;
	CSettings settings;
	CRecurrentCheckpoint checkpoint;
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
	ReccurentLoops(CSettings settings, CRecurrentCheckpoint checkpoint);

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
public:
	vector<RETURN_WEIGHT_TYPE> runNetwork(double* in);
	vector<RETURN_WEIGHT_TYPE> runNetwork(weight_type* in);

	template <typename T>
	weight_type* convert_array(T* in);

	//*********************
	//Training
	//*********************
public:
	void startTraining(int type);

private:
	//Training data is passed in
	bool train_network_RealTimeRecurrentTraining();
	bool train_network_HessianFreeOptimizationTraining();

	//Retrieve the training data from the file passed in by the settings
	bool load_training_data_from_file();


	//*********************
	//Testing Methods
	//*********************
	weight_type* createTestInputOutput(int numberOfInput, int input_output);
	void train_network_RealTimeRecurrentTraininguserControlOutput();

};

