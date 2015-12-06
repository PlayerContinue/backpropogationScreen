#pragma once
#include <string.h>
#include <vector>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thread>
#include <boost/interprocess/managed_shared_memory.hpp>


#ifdef __IOSTREAM_H_INCLUDED__

#else
#include <iostream>
#endif

#ifndef __NETWORKTIMER_H_INCLUDED__
#define __NETWORKTIMER_H_INCLUDED__
#include "NetworkTimer.h"
#endif

#ifndef __X_H_INCLUDED__
#define __X_H_INCLUDED__
#include "CSettings.h"
#include "NetworkBase.cuh"
#endif
#include "RecurrentNeuralNetwork.cuh"
#include "LongTermShortTermNetwork.cuh"




#ifndef __TESTCODE_CUH_INCLUDED___
#include "testcode.cuh"
#define __TESTCODE_CUH_INCLUDED___
#endif

#include "CRecurrentCheckpoint.h"
#include <thrust/complex.h>

#ifndef weight_type
#define weight_type double
#endif
#define RETURN_WEIGHT_TYPE  double

#define CHECKPOINT_TIMER "CHECKPOINT_TIMER"
#define TIMER_SHARED "timer_shared_memory"
#define TIMER_NEEDED "timer_needed"
#define TIMER_PRINT "timer_print"
#define TIMER_PRINT_VALUE "timer_print_value"
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
	static const int LongTermShortTerm = 2;
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
	//Set of input/output for testing to train
	//If no training set is given, the training set will be taken from a randomly selected 
	//set of the input/output
	weight_type** training_input;
	weight_type** training_output;
	int number_in_training_sequence;
	//Files connected to the input/output files containing the data
	std::fstream* inputfile;
	std::fstream* outputfile;
	host_vector<weight_type> mean_square_error_results_old;
	host_vector<weight_type> mean_square_error_results_new;
	enum data_type {OUTPUT,INPUT,TRAINING=2,TRAINING_1 = 2, TRAINING_2 = 3};
	int length_of_arrays[4];
	//Timer to keep track of length
	NetworkTimer timer;
	boost::interprocess::managed_shared_memory* timer_shared_memory;//Keeps track of which, if any, stops need to be made in the running
	vector<std::thread*> thread_list;
	enum thread_types {TIMER_THREAD,PIPE_THREAD,FINAL_THREAD_POS};
	std::istream::streampos length_of_file = (std::istream::streampos)0;

	//*********************
	//Constructors
	//*********************
public:
	//Constructor for an empty network
	//Creates a settings object from defaults
	ReccurentLoops();

	//Constructor for an network created from user settings
	ReccurentLoops(CSettings settings);

	ReccurentLoops(CSettings settings, int type);

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
	//Load from a file, returns the length of the sequence and the length of the returned list
	void loadFromFile(std::fstream &file, int length_of_results, double** storage, int* sequence_length, data_type type, int max_allowed, bool first_run);
	void loadFromFile(std::fstream &file, int length_of_results, double** storage, int sequence_length[2], int length, data_type type, int max_allowed, bool first_run);
	//Loads the training set from a file
	void LoadTrainingSet();
	//*********************
	//Utilization
	//*********************
public:
	vector<RETURN_WEIGHT_TYPE> runNetwork(int* in);
	
	vector<RETURN_WEIGHT_TYPE> runNetwork(weight_type* in);
	
	//Run the network until either the end_value is reached or the max_sequence_length is reached
	//end_value The value of the end
	//in the input
	//save_location where to place the output
	//max_sequence_length the max length of the sequence
	void runContinousNetwork(weight_type* in, std::string save_location, weight_type* end_value);
	void runContinousNetwork(weight_type* in, std::string save_location, weight_type* end_value,int max_sequence_length);

public:
	template <typename T>
	weight_type* convert_array(T* in);

	//*********************
	//Training
	//*********************
public:
	void startTraining(int type);

	void testTraining();


private:
	//Training data is passed in
	bool train_network_RealTimeRecurrentTraining();
	bool train_network_HessianFreeOptimizationTraining();
	device_vector<weight_type> runTrainingNetwork(weight_type* in);
	void getMeanSquareError();
	
	//function to run at the end of sequence, seperated for cleaner looking code
	void sequenceEnd(int &length_of_sequence, int &count_sequences, int &growth_check);
	//Resets the file and other information for a new loop
	void reset_file_for_loop();
	//*********************
	//Clean the Network
	//*********************
	void cleanLoops();
	//*********************
	//Testing Methods
	//*********************
	weight_type* createTestInputOutput(int numberOfInput, int input_output);
	void createCheckpoint();
	void createCheckpoint(string file_name);
	void loadCheckpoint();


	//*********************
	//Threaded Functions
	//*********************
	//Starts any threads required for training the network
	//Returns true when all threads created, false otherwise
	bool start_training_threads();

	//Stop any threads required for training the network
	//Returns true when sucessful, false otherwise
	bool stop_training_thread();

	//Initialize all objects required for use in threads
	void initialize_threads();

	//Clean Up All Threads
	void clean_threads();

	//*********************
	//Override Operators
	//*********************

	friend ostream& operator<<(ostream &os, const ReccurentLoops &loop){
		os.precision(30);
		os << loop.checkpoint;//Output the checkpoint
		loop.mainNetwork->OutputNetwork(os);
		return os;
	}
};

