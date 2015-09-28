#pragma once
#include <string.h>
#include <string>
#include <iostream>
#include "CSettings.h"
using namespace std;
//****************************************************************************************************
//
//Programmer: David Greenberg
//Purpose: Class for containing information about the current position of the learning process
//
//****************************************************************************************************
class CRecurrentCheckpoint
{
public:
	CRecurrentCheckpoint();
	CRecurrentCheckpoint(CSettings settings);

		//Count the total number of loops which have occured
		int i_number_of_loops_checkpoint = 0;

		//Count the number of current loops traveled through before reaching a reset
		int i_number_of_loops = 0;

		//Count the number of times left for the mean to be larger than the previous mean before trying to add new neurons
		int i_times_lowest_mean_square_error_to_large;

		//Count the number of times which the current d_mean_square_error == d_previous_mean_square_error
		int i_equal_square_errors;

		// Store the mean square error
		double d_mean_square_error = (double)INT_MAX;

		//Store the previous round mean_square_error to test if the value changed between rounds
		double d_previous_mean_square_error = 0;

		//Store the lowest mean_square_error found
		double d_lowest_mean_square_error = (double)INT_MAX;

		//Store the most recent d_row_distance_threshold
		double d_row_distance_threshold;

		//store the most recent d_neuron_distance_threshold
		double d_neuron_distance_threshold;

		//Store the threshold for whether a neuron or a layer is added
		double d_neuron_or_layer_threshold;

		//store the most recently recorded network file
		string s_network_file_name;

		//Input file current pos
		int i_current_position_in_input_file;

		//Output file current pos
		int i_current_position_in_output_file;

		

		//***************************************
		//Overload Operators
		//***************************************
		//Save to file
		friend ostream& operator<<(ostream& os, const CRecurrentCheckpoint checkpoint);

		//Load from file
		friend istream& operator>>(istream& is, CRecurrentCheckpoint& checkpoint);
};

