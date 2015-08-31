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

	//Stores the name of the file containing the test set
	bool b_testingFromFile = false;
	string s_testSet;

	//Number of pieces from the testing set per round
	int i_number_of_training;


	//Load From File
	bool b_loadNetworkFromFile = false;
	string s_loadNetworkFile;

	

	CSettings();

	friend istream& operator>>(istream& is, CSettings& settings){
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

		is >> next;
		is >> settings.d_alpha;

		is >> next;
		is >> settings.d_beta;

		is >> next;
		is >> settings.b_trainingFromFile;

		is >> next;
		is >> settings.s_trainingSet;

		is >> next;
		is >> settings.b_testingFromFile;

		is >> next;
		is >> settings.s_testSet;

		is >> next;
		is >> settings.i_number_of_training;
		
		is >> next;
		is >> settings.b_loadNetworkFromFile;

		is >> next;
		is >> settings.s_loadNetworkFile;

		return is;
	}
};

