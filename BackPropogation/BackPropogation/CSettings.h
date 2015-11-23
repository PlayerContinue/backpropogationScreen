//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Contains the settings for a particular network/group of networks
//----------------------------------------------------------------------------------------

#pragma once
#include <string>
#include <iostream>
using namespace std;
class CSettings
{
public:
	//Stores the name of the network
	string s_network_name = "network";

	//Contains the number of times to loop if loop is set as choice
	int i_loops = 0;

	//Allowable number of failures
	int i_number_allowed_failures = 0;

	//Number of inputs/outpus
	int i_input = 1;
	int i_output = 1;

	//Number of rounds before a check occurs
	int i_number_before_growth_potential = 0;

	//Threshold - Value which sets the system as trained
	//If Mean Square Error is higher than this
	double d_threshold;

	//Store the threshold for distance to add a new row
	double d_row_distance_threshold;

	//Store the threshold for distance to add a new neuron to a row
	double d_neuron_distance_threshold;

	//Store the threshold for distance to add a new row
	double d_row_success_threshold;

	//Store the threshold for distance to add a new neuron to a row
	double d_neuron_success_threshold;

	//Store the value which allows the change in Square Mean Error to flucuate slightly
	double d_fluctuate_square_mean;

	//Alpha and Beta
	double d_alpha;
	double d_beta;

	//Stores the name of the file containing the training set
	bool b_trainingFromFile = false;
	string s_trainingSet;
	//Stores an int stating what type the data is (i.e. 0 double/int, 1 char, 2 string) 
	int i_trainingSetType;

	string s_outputTrainingFile;

	//Stores an int stating what type the data is (i.e. 0 double/int, 1 char, 2 string) 
	int i_outputTrainingSetType;
	//Stores the name of the file containing the test set
	bool b_testingFromFile = false;
	string s_testSet;
	string s_outputTestSet;

	//Number of pieces from the testing set per round
	int i_number_of_training;

	//Stores the name of the file to load a checkpoint from
	bool b_loadFromCheckpoint;
	string s_checkpoint_file;


	//Number of allowed output matches to allow a layer in
	int i_number_allowed_same;

	//Load From File
	bool b_loadNetworkFromFile = false;
	string s_loadNetworkFile;

	//Allow the nodes to be locked
	//Node locking involves nodes being unable to change weights or bias after a certain point is reached
	//Current concept is working from the delta becoming low enough
	bool b_allow_node_locking;
	double d_lock_node_level;

	//Number Of Rounds before providing output from an input
	int i_recurrent_flip_flop = 3;

	//Number layers unrolled when performing backpropogation through time
	int i_backprop_unrolled = 3;

	//Number of inputs in a sequence
	int i_number_in_sequence = 1;

	//Number of nodes to start with
	int i_number_start_nodes;

	//Number extra connections to add
	int i_number_new_weights;
	
	int i_number_of_testing_items;

	bool b_allow_growth = false;

	//Number of times to go through the training file before ending
	int i_numberTimesThroughFile;

	CSettings();


	//Take a user defined type and return the int representing that type
	int getTypeOfInput(string input_type){
		if (input_type.compare("double") == 0){
			return 0;
		}
		else if (input_type.compare("char") == 0){
			return 1;
		}
		else if (input_type.compare("string") == 0){
			return 2;
		}
		else{//Unsupported Type
			return -1;
		}
	}


	//****************************
	//Overloaded Operators
	//****************************
	//Read in a new settings object
	friend std::istream& operator>>(std::istream& is, CSettings& settings){
		//Storage for the name of the operator
		string next;

		is >> next;
		is >> settings.s_network_name;

		is >> next;
		is >> settings.i_loops;

		is >> next;
		is >> settings.i_number_allowed_failures;

		is >> next;
		is >> settings.i_number_before_growth_potential;

		is >> next;
		is >> settings.i_number_allowed_same;

		is >> next;
		is >> settings.i_input;

		is >> next;
		is >> settings.i_output;

		is >> next;
		is >> settings.d_threshold;

		is >> next;
		is >> settings.d_row_distance_threshold;

		is >> next;
		is >> settings.d_neuron_distance_threshold;

		is >> next;
		is >> settings.d_row_success_threshold;

		is >> next;
		is >> settings.d_neuron_success_threshold;

		is >> next;
		is >> settings.d_fluctuate_square_mean;

		//Allow locked nodes
		is >> next;
		is >> settings.b_allow_node_locking;

		is >> next;
		is >> settings.d_lock_node_level;

		//Set Alpha and Beta
		is >> next;
		is >> settings.d_alpha;

		is >> next;
		is >> settings.d_beta;

		//Set the training set
		is >> next;
		is >> settings.b_trainingFromFile;


		is >> next;
		is >> settings.s_trainingSet;

		is >> next;
		is >> settings.s_outputTrainingFile;

		
		//Set the test set
		is >> next;
		is >> settings.b_testingFromFile;


		is >> next;
		is >> settings.s_testSet;

		is >> next;
		is >> settings.s_outputTestSet;

		//Load number in each training set
		is >> next;
		is >> settings.i_number_of_training;

		//Get training type
		is >> next;
		is >> next;

		settings.i_trainingSetType = settings.getTypeOfInput(next);

		is >> next;
		is >> next;

		settings.i_outputTrainingSetType = settings.getTypeOfInput(next);


		//Recurrent Network Info
		is >> next;
		is >> settings.i_recurrent_flip_flop;

		//Load from checkpoint 
		is >> next;
		is >> settings.b_loadFromCheckpoint;

		is >> next;
		is >> settings.s_checkpoint_file;

		//Load network file
		is >> next;
		is >> settings.b_loadNetworkFromFile;

		is >> next;
		is >> settings.s_loadNetworkFile;

		//Number of times to go through training file
		is >> next;
		is >> settings.i_numberTimesThroughFile;

		//LongTermShortTerm_items
		is >> next; //States data is for special kind of network
		is >> next;
		is >> settings.i_backprop_unrolled;

		is >> next;
		is >> settings.i_number_in_sequence;

		is >> next;
		is >> settings.i_number_start_nodes;

		is >> next;
		is >> settings.i_number_new_weights;

		is >> next;
		is >> settings.i_number_of_testing_items;


		is >> next;
		is >> settings.b_allow_growth;

		return is;
	}
};

