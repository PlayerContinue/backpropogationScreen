#include "CRecurrentCheckpoint.h"


CRecurrentCheckpoint::CRecurrentCheckpoint()
{

	
}

CRecurrentCheckpoint::CRecurrentCheckpoint(CSettings settings){
	this->i_number_of_loops_checkpoint = 0;
	this->i_number_of_loops = 0;
	this->i_times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
	this->d_mean_square_error = (double)INT_MAX;
	this->d_lowest_mean_square_error = (double)INT_MAX;
	this->d_previous_mean_square_error = 0;
	this->d_row_distance_threshold = settings.d_row_distance_threshold;
	this->d_neuron_distance_threshold = settings.d_neuron_distance_threshold;
	this->i_current_position_in_output_file = 0;
	this->i_current_position_in_output_file = 0;
	this->s_network_file_name = settings.s_network_name;
}


ostream& operator<<(ostream& os, const CRecurrentCheckpoint checkpoint){
	os << "b_still_running " << checkpoint.b_still_running << endl;

	os << "i_number_of_loops_checkpoint " << checkpoint.i_number_of_loops_checkpoint << endl;

	os << "i_times_lowest_mean_square_error_to_large " << checkpoint.i_times_lowest_mean_square_error_to_large << endl;

	os << "i_equal_square_errors " << checkpoint.i_equal_square_errors << endl;

	os << "d_mean_square_error " << checkpoint.d_mean_square_error << endl;

	os << "d_previous_mean_square_error " << checkpoint.d_previous_mean_square_error << endl;

	os << "i_number_of_loops " << checkpoint.i_number_of_loops << endl;

	os << "d_lowest_mean_square_error " << checkpoint.d_lowest_mean_square_error << endl;

	os << "d_row_distance_threshold " << checkpoint.d_row_distance_threshold << endl;

	os << "d_neuron_distance_threshold " << checkpoint.d_neuron_distance_threshold << endl;

	os << "d_neuron_or_layer_threshold " << checkpoint.d_neuron_or_layer_threshold << endl;

	//Input file current pos
	os << "i_current_position_in_input_file " << checkpoint.i_current_position_in_input_file << endl;

	//Output file current pos
	os << "i_current_position_in_output_file " << checkpoint.i_current_position_in_output_file << endl;

	os << "s_network_file_name " << checkpoint.s_network_file_name << endl;



	return os;
}

//Load from file
istream& operator>>(istream& is, CRecurrentCheckpoint& checkpoint){
	string next;

	is >> next;
	is >> checkpoint.b_still_running;

	is >> next;
	is >> checkpoint.i_number_of_loops_checkpoint;

	is >> next;
	is >> checkpoint.i_times_lowest_mean_square_error_to_large;

	is >> next;
	if (next.compare("i_equal_square_errors") == 0){
		is >> checkpoint.i_equal_square_errors;
	}
	else{
		checkpoint.i_equal_square_errors = 0;

	}

	is >> next;
	is >> checkpoint.d_mean_square_error;

	is >> next;
	is >> checkpoint.d_previous_mean_square_error;

	is >> next;
	is >> checkpoint.i_number_of_loops;


	is >> next;
	is >> checkpoint.d_lowest_mean_square_error;

	is >> next;
	is >> checkpoint.d_row_distance_threshold;

	is >> next;
	is >> checkpoint.d_neuron_distance_threshold;

	is >> next;
	if (next.compare("d_neuron_or_layer_threshold") == 0){
		is >> checkpoint.d_neuron_or_layer_threshold;
	}
	else{
		checkpoint.d_neuron_or_layer_threshold = 10;
	}

	//Input file current pos
	is >> next;
	is >> checkpoint.i_current_position_in_input_file;

	//Output file current pos
	is >> next;
	is >> checkpoint.i_current_position_in_output_file;

	is >> next;
	is >> checkpoint.s_network_file_name;



	return is;
}