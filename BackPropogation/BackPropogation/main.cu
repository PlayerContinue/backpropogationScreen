

//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Initizializing algorithm. Contains the main function and only the main function
//----------------------------------------------------------------------------------------

//#define PROBLEMS 5
//#define FIRST_TEST
#define PAUSE
#pragma once
#include <vector>
#include <stdlib.h>
#include <bitset>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <csignal>
#include "CSettings.h"
#include "ReccurentLoops.cuh"
#include "CGraphicNetwork.cuh"

#ifdef WINDOWS_COMPUTER
#include <wincon.h>
#endif

using namespace std;


static volatile bool pause;


//PROTOTYPES
int writeToFile(CGraphicsNetwork &network, CSettings settings);
void testOutput2(double** value, CGraphicsNetwork &test, int size);
bool addToNetwork(CGraphicsNetwork &test, CSettings settings, SCheckpoint& checkpoint, double** testSet, double mean_square_error);
int getDataFromFile(string fileName, int start, int end, int numberResults, double** input, int type_of_input);
void testOutput3(double* value, CGraphicsNetwork &test, int size);

void printVectorOutput(vector<double> vectorA){
	int size = vectorA.size();
	int value = 0;//Store a int with 32 0
	int pos = 1;
	for (int i = vectorA.size() - 1; i >= 0; i--){

		//Either add one at the current position or add zero
		value = value | (vectorA[i] > .5 ? pos : 0);

		pos = pos << 1;
	}
	std::cout << value << endl;
	if (value <= 255 & value > 0){
		std::cout << (char)value << endl;
	}

}

int printVectorOutputChar(vector<double> vectorA){
	int size = vectorA.size();
	int value = 0;//Store a int with 32 0
	int pos = 1;
	for (int i = vectorA.size() - 1; i >= 0; i--){

		//Either add one at the current position or add zero
		value = value | (vectorA[i] > .5 ? pos : 0);

		pos = pos << 1;
	}

	if (value <= 255 & value > 0){
		std::cout << (char)value;
	}
	else{
		std::cout << value;
	}

	return value;

}

void printVectorOutput2(vector<double> vectorA){
	int size = vectorA.size();
	int value = 0;//Store a int with 32 0
	int pos = 1;
	for (int i = vectorA.size() - 1; i >= 0; i--){

		//Either add one at the current position or add zero
		value = value | (vectorA[i] > .5 ? pos : 0);

		pos = pos << 1;
	}

	std::cout << value;
}

void printArray(double* arrayA, int size){
	for (int i = 0; i < size; i++){
		std::cout << (char)arrayA[i];
		std::cout << endl;
	}
}

void trainNetwork(double* value_1, double* value_2, double* value_3, double* value_4, CGraphicsNetwork &test, int rounds){
	//Train the network on simple test values
	for (int i = 0; i < rounds; i++){
		if (i % 2 == 0){
			test.backprop(value_1, value_2);
		}
		else{
			test.backprop(value_3, value_4);
		}
	}

}

void trainNetwork2(double* value[], double* results[], CGraphicsNetwork &test, int start, int end, int rounds){

	int count = 0;
	//Train the network on simple test values
	for (int i = start; i < rounds; i++){
		test.backprop(value[count], results[count]);
		count++;
		if (count >= end){
			count = start;
		}
	}

}

//Returns true if the current mean_square_error is less than the lowest_mean_square
template <typename T>
bool checkThreshold(T mean_square, T lowest_mean_square, T threshold){
	if (mean_square < threshold && threshold >= 0){
		return false;
	}
	else if (mean_square <= lowest_mean_square){
		return true;
	}
	else{
		return false;
	}
}

int numberFullSame(CGraphicsNetwork test, double** in, int size){
	vector<double> output;
	vector<double> output2;
	int count_success = 0;
	for (int i = 0; i < size; i++){
		test.feedForward(in[i]);
		if (i != 0){
			output2 = output;
		}
		output = test.getOutput();
		if (i != 0){
			for (unsigned int j = 0; j < output.size(); j++){
				if (output[j] != output2[j]){
					count_success++;
					break;
				}
			}
		}
	}

	return (size - count_success);
}

//******************************
//Save, Load, Create Checkpoints
//******************************

//Create a checkpoint containing all the current values of the system
void createCheckpoint(CGraphicsNetwork test, SCheckpoint& checkpoint, CSettings settings){
	int checkpoint_number = writeToFile(test, settings);
	checkpoint.s_network_file_name = "networks/" + settings.s_network_name + std::to_string(checkpoint_number) + ".txt";
	//Write the checkpoint to a file
	std::ofstream outputfile;
	outputfile.precision(30);
	outputfile.open("checkpoints/" + settings.s_network_name + "_checkpoint_" + std::to_string(checkpoint_number) + ".txt", ios::trunc);
	if (outputfile.is_open()){
		//Output the network
		outputfile << checkpoint << flush;
		outputfile.close();
	}
	else{
		std::cout << "Unable to write checkpoint to file." << endl;
		std::cout << "continue?";
	}

}

void LoadCheckpointFromFile(SCheckpoint& checkpoint, string s_file_name){
	std::ifstream inputfile;

	inputfile.open(s_file_name, ios_base::beg);
	if (inputfile.is_open()){
		inputfile >> checkpoint;
		inputfile.close();
	}
	else{
		std::cout << "Unable to read checkpoint from file." << endl;
		std::cout << "continue?";
		if (cin.get() == 'n'){

			exit(0);
		}
	}
}

//Create a new checkpoint object from the created settings
void createNewCheckpoint(SCheckpoint& checkpoint, CSettings settings, double** in){
	checkpoint.i_number_of_loops_checkpoint = 0;
	checkpoint.i_number_of_loops = 0;
	checkpoint.i_times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
	checkpoint.d_mean_square_error = (double)INT_MAX;
	checkpoint.d_lowest_mean_square_error = (double)INT_MAX;
	checkpoint.d_previous_mean_square_error = 0;
	checkpoint.d_row_distance_threshold = settings.d_row_distance_threshold;
	checkpoint.d_neuron_distance_threshold = settings.d_neuron_distance_threshold;
	checkpoint.i_current_position_in_output_file = 0;
	checkpoint.i_current_position_in_output_file = 0;
	std::cout << "Getting Layer Or Row Number " << endl;
	checkpoint.get_layer_or_row(in, settings.i_number_of_training, settings.i_output);
}

//*********************************
//Signal Handlers
//*********************************

void signal_handler(int signal)
{
	if (signal == SIGINT){
		pause = true;
	}
	std::signal(SIGINT, signal_handler);
}

//*********************************
//User Signal
//*********************************
#ifdef PAUSE
void userSignal(double** results, double** value, SCheckpoint checkpoint, double** testSetIn, double** testSetOut, CGraphicsNetwork test, int testLength, CSettings settings){




	char userin = 'l';
	double* tempDistanceHolderForPause;
	double tempDistanceHolderForPauseSingle;
	std::cout << "1) Print output " << endl;
	std::cout << "2) Print Output On Current Input " << endl;
	std::cout << "3) Run same check " << endl;
	std::cout << "4) Get mean square " << endl;
	std::cout << "5) Save Checkpoint " << endl;
	std::cout << "6) Check MSE on current input " << endl;
	std::cout << "7) Get distance check " << endl;
	std::cout << "8) Continue " << endl;
	std::cout << "0) Exit " << endl;

	std::cout << " loop " << checkpoint.i_number_of_loops_checkpoint << endl;
	std::cout.precision(30);

	do{

		if (userin != 'l'){
			std::cout << "Anything else? ";
		}
		std::cin.sync();
		userin = std::cin.get();
		switch (userin){
		case '1':
			//Test the output
			std::cout << "Getting Output " << endl;
			testOutput2(testSetIn, test, testLength);
			break;
		case '2':
			std::cout << "Getting Single Output " << endl;
			test.feedForward(value[0]);
			testOutput3(value[checkpoint.i_number_of_loops], test, test.getI_input());
			break;
		case '3':
			std::cout << "Getting Number Same " << endl;
			std::cout << numberFullSame(test, testSetIn, settings.i_number_of_training) << endl;
			break;
		case '4':
			std::cout << "Getting Mean Square " << endl;
			std::cout << test.getMeanSquareError(testSetIn, testSetOut, testLength) << endl;
			break;
		case '5':
			std::cout << "Creating Checkpoint " << endl;
			createCheckpoint(test, checkpoint, settings);
			std::cout << "Checkpoint created " << endl;
			break;
		case '6':
			std::cout << "Getting MSE " << endl;
			std::cout << test.getSingleMeanSquareError(value[checkpoint.i_number_of_loops - 1], results[checkpoint.i_number_of_loops - 1], testLength) << endl;
			break;
		case '7':
			std::cout << "Getting Distance Measure " << endl;
			tempDistanceHolderForPause = test.getRootMeanSquareErrorForAllLayer(value[checkpoint.i_number_of_loops]);
			tempDistanceHolderForPauseSingle = 0;
			for (int i = 0; i < test.getNumLayers() - 1; i++){
				std::cout << i << ") " << tempDistanceHolderForPause[i] << endl;
				tempDistanceHolderForPauseSingle += tempDistanceHolderForPause[i];
			}
			tempDistanceHolderForPauseSingle /= (test.getNumLayers() - 1);
			std::cout << tempDistanceHolderForPauseSingle << endl;
			break;
		case '0':
			std::wcout << "Would you like to create a checkpoint?";
			std::cin.sync();
			if (std::cin.get() == 'y'){
				std::cout << "Creating Checkpoint " << endl;
				createCheckpoint(test, checkpoint, settings);
				std::cout << "Checkpoint created " << endl;
			}
			exit(0);
			break;
		default:
			break;
		}
	} while (userin != '9' && userin != 'n');

	pause = false;
	std::cout << "finished" << endl;



}
#endif

//******************************
//Training the Network
//******************************

//Train the current network
//Check if the limit has been reached as the stopping point
//Use mean square error to check distance
void trainNetworkDelta(double* value[], double* results[], CGraphicsNetwork &test, int input_start, int output_start, double* testSetIn[], double* testSetOut[], int testLength, CSettings settings, SCheckpoint& checkpoint){

	int new_end = settings.i_number_of_training;

	do{

		//Pause the program and perform one of the following options
#ifdef PAUSE
		if (pause){
			userSignal(results, value, checkpoint, testSetIn, testSetOut, test, testLength, settings);
		}
#endif

		//Run the backpropogation algorithm to both get the current output and train the network
		test.backprop(value[checkpoint.i_number_of_loops], results[checkpoint.i_number_of_loops]);

		//Create a checkpoint when X number of iterations have occured, where X is defined by the user
		if (checkpoint.i_number_of_loops_checkpoint % settings.i_loops == 0){
			createCheckpoint(test, checkpoint, settings);
		}
		checkpoint.i_number_of_loops++;
		checkpoint.i_number_of_loops_checkpoint++;

		//Grab more training data when the current list of training data has been exhausted

		try{
			if (checkpoint.i_number_of_loops >= settings.i_number_of_training){
				checkpoint.i_number_of_loops = 0;

				if (new_end < settings.i_number_of_training){
					//TODO Add asking for a new file or to reuse file
					createCheckpoint(test, checkpoint, settings);
					exit(0);
				}

				if (settings.b_trainingFromFile){

					//Store training set
					checkpoint.i_current_position_in_input_file = getDataFromFile(settings.s_trainingSet, checkpoint.i_current_position_in_input_file, settings.i_number_of_training, settings.i_input, value, settings.i_trainingSetType);

				}

				if (settings.b_testingFromFile){
					checkpoint.i_current_position_in_output_file = getDataFromFile(settings.s_outputTrainingFile, checkpoint.i_current_position_in_output_file, settings.i_number_of_training, settings.i_output, results, settings.i_outputTrainingSetType);
				}

				new_end = (checkpoint.i_current_position_in_input_file < checkpoint.i_current_position_in_output_file ? checkpoint.i_current_position_in_input_file : checkpoint.i_current_position_in_output_file);

			}
		}
		catch (exception e){
			createCheckpoint(test, checkpoint, settings);
			exit(0);
		}
		//Get the mean_square_error when the number of loops reaches a user defined values
		if (checkpoint.i_number_of_loops_checkpoint%settings.i_number_before_growth_potential == 0){

			//Set the current mean_square_error as the previous error
			checkpoint.d_previous_mean_square_error = checkpoint.d_mean_square_error;

			//Retrieve the new mean_square_error
			checkpoint.d_mean_square_error = test.getMeanSquareError(testSetIn, testSetOut, testLength);

			//Set the current lowest
			if (checkpoint.d_mean_square_error < checkpoint.d_lowest_mean_square_error && checkpoint.d_mean_square_error != checkpoint.d_previous_mean_square_error){
				checkpoint.d_lowest_mean_square_error = checkpoint.d_mean_square_error;

				//Reset value since error was lowered
				checkpoint.i_times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
			}
			else{
				checkpoint.i_times_lowest_mean_square_error_to_large--;

				//Keep track of the number of times the mean square error has been equal
				if (checkpoint.d_previous_mean_square_error == checkpoint.d_mean_square_error){
					checkpoint.i_equal_square_errors++;
				}
				//If the below is true, something close to the limit has been reached, the network needs to change size 
				if (checkpoint.i_times_lowest_mean_square_error_to_large <= 0){
					//Add new nodes to the network
					if (addToNetwork(test, settings, checkpoint, testSetIn, checkpoint.d_mean_square_error)){
						//Reset the number allowed
						//Since new ones may have been
						checkpoint.i_times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures + 10;
					}
					else{
						//Reset the number of times before a growth is attempted
						checkpoint.i_times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
					}

					//Reset the number of times the mean_square_error has been equal
					checkpoint.i_equal_square_errors = 0;

				}
			}

		}

		//Loop until the error is smaller than the threshold
	} while (settings.d_threshold < checkpoint.d_mean_square_error);
	createCheckpoint(test, checkpoint, settings);
}

//******************************
//Modifying the Network
//******************************

//Returns true if a neuron was added
bool addToNetwork(CGraphicsNetwork &test, CSettings settings, SCheckpoint& checkpoint, double** testSet, double mean_square_error){


	double* differencePerLayer = test.getRootMeanSquareErrorForAllLayer(testSet[checkpoint.i_number_of_loops]);

	//Get the average difference per layer between the outputs in the layers
	double averageDifferencePerLayer = 0;

	for (int i = 0; i < test.getNumLayers() - 1; i++){
		averageDifferencePerLayer += differencePerLayer[i];
	}

	averageDifferencePerLayer /= (test.getNumLayers() - 1);


	//double success = test.getSuccessRate() - test.getPreviousSuccessRate();
	//double mean_square_error_dif = mean_square_error - settings.d_threshold;
#ifdef FULL_SUCCESS
	double full_success = test.getFullSuccessRate() - test.getFullPreviousSuccessRate();
#endif
	//Get the number of test sets returning the exact same values
	//int numberSame = numberFullSame(test, testSet, settings.i_number_of_training);

	//Add a new layer if the success is too low and the threshold has not been reached
	//A layer should be added if the mean square error remains constant as the current layer has been fully trained to give a particular output
	//Therefore a function should be added to deal with that particular output
	if (averageDifferencePerLayer > checkpoint.d_neuron_or_layer_threshold){ //&& numberSame < settings.i_number_allowed_same){
		test.addLayer(-1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) * 5);
		//Increment the size of the need mean distance to get a new layer
		//And decrease the size of the needed mean distance to get a new neuron
		/*if (checkpoint.d_neuron_distance_threshold > 0){
			checkpoint.d_row_distance_threshold += settings.d_row_distance_threshold * .1;
			checkpoint.d_neuron_distance_threshold -= settings.d_neuron_distance_threshold * .1;
			}*/
		test.resetNetwork();
		return true;
	}
	else if (averageDifferencePerLayer < checkpoint.d_neuron_or_layer_threshold){
		if (test.getNumLayers() == 2){
			//Since neurons cannot be added, if only the input/output layer exists, and new neuron is chosen, than a new layer is added instead
			test.addLayer(-1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) * 5);
		}
		else{
			//Double the number of nodes in every non input/output row 
			//Keeps size of each row equivalent
			//Good for initial growth
			for (int i = 1; i < test.getNumLayers() - 1; i++){
				if (differencePerLayer[i - 1] < checkpoint.d_neuron_or_layer_threshold){
					test.addNeuronToLayer(i, i, 10);
				}
			}
			//Increment the size of the need mean distance to get a new neuron
			//And decrease the size of the needed mean distance to get a new row

			/*if (checkpoint.d_row_distance_threshold > 0){
				checkpoint.d_row_distance_threshold -= settings.d_row_distance_threshold * .1;
				checkpoint.d_neuron_distance_threshold += settings.d_neuron_distance_threshold * .1;
				}*/
		}
		test.resetNetwork();
		return true;
	}
	//If both fail, but a large gap still exists between the threshold and the set distances, decrease them both.
	//This is to allow the system to grow further when needed.
	//However, only occur when the network is fully trained
	/*else if (checkpoint.i_equal_square_errors >= settings.i_number_allowed_failures){
		checkpoint.d_row_distance_threshold -= settings.d_row_distance_threshold * .1;
		checkpoint.d_neuron_distance_threshold -= settings.d_neuron_distance_threshold * .1;
		}*/
	return false;


}

//******************************
//Testing Output
//******************************

void testOutput(double* value_1, double* value_3, CGraphicsNetwork &test){
	vector<double> temp2;
	std::cout << "input";
	std::cout << endl;
	printArray(value_1, 2);
	test.feedForward(value_1);
	temp2 = test.getOutput();
	std::cout << "output";
	std::cout << endl;
	printVectorOutput(temp2);
	std::cout << endl;


	test.feedForward(value_3);
	temp2 = test.getOutput();

	std::cout << "input";
	std::cout << endl;
	printArray(value_3, 2);

	temp2 = test.getOutput();
	std::cout << "output";
	std::cout << endl;
	printVectorOutput(temp2);
	std::cout << endl;
}

void testOutput2(double** value, CGraphicsNetwork &test, int size){
	vector<double> temp2;

	for (int i = 0; i < size; i++){
		std::cout << "input";
		std::cout << endl;

		printArray(value[i], 6);

		test.feedForward(value[i]);

		temp2 = test.getOutput();

		std::cout << "output";
		std::cout << endl;
		printVectorOutput(temp2);
		std::cout << endl;
	}
}

void testOutput3(double* value, CGraphicsNetwork &test, int size){
	vector<double> temp2;


	std::cout << "input";
	std::cout << endl;

	printArray(value, size);

	test.feedForward(value);

	temp2 = test.getOutput();

	std::cout << "output";
	std::cout << endl;
	printVectorOutput(temp2);
	std::cout << endl;

}

//******************************
//Writing And Loading From Files
//******************************

//Output the network to a file
int writeToFile(CGraphicsNetwork &network, CSettings settings){
	static int file_number = 0;
	file_number++;
	std::ofstream outputfile;
	outputfile.open("networks/" + settings.s_network_name + std::to_string(file_number) + ".txt", ios::trunc);
	if (outputfile.is_open()){
		//Output the network
		outputfile << network << flush;
		outputfile.close();
	}
	else{
		std::cout << "Unable to write checkpoint to file." << endl;
		std::cout << "continue?";
	}

	return file_number;

}

bool loadFromFile(CGraphicsNetwork& network, string fileName){
	std::ifstream inputfile;

	inputfile.open(fileName, ios_base::beg);
	if (inputfile.is_open()){
		inputfile >> network;
		inputfile.close();
		return true;
	}
	else{
		std::cout << "Unable to read from file." << endl;
		std::cout << "continue?";
		if (cin.get() == 'n'){

			exit(0);
		}
		return false;
	}
}

CSettings loadSettings(string fileName){
	std::ifstream inputfile;
	inputfile.open(fileName, ios_base::beg);
	CSettings settings;
	if (inputfile.is_open()){

		inputfile >> settings;
		inputfile.close();
		return settings;
	}
	else{
		std::cout << "Unable to read from file." << endl;
		std::cout << "continue?";
		if (cin.get() == 'n'){

			exit(0);
		}

		return settings;
	}
}


#ifdef WINDOWS_COMPUTER
BOOL WINAPI ConsoleHandlerRoutine(DWORD dwCtrlType)
{
	if (dwCtrlType == CTRL_CLOSE_EVENT)
	{
		return TRUE;
	}

	return FALSE;
}


void initialize(){
	if (SetConsoleCtrlHandler(ConsoleHandlerRoutine, TRUE) == false){
		printf("Unable to attach Handler");
		std::cout << "continue? ";
		if (cin.get() == 'n'){
			exit(0);
		}
	}
}

#endif

//Retrieve the data from a file
//start: Where to start gathering characters from in the file
//numberOfRounds: The number of character sets which are needed to be retrieve
//numberResults: How many characters should be retrieved for a single round
//input : the storage container for the input
//type_of_input : The type of input/output i.e. is char, double, ect
int getDataFromFile(string fileName, int start, int numberOfRounds, int numberResults, double** input, int type_of_input){
	std::ifstream inputfile;
	char individual_delimiter = (char)29;
	char group_delimiter = (char)30;
	char current_char = '1';
	string current_string = "";
	inputfile.open(fileName);
	if (inputfile.is_open()){
		inputfile.seekg(start);
		int letterPosition = 0;
		for (int i = 0; i < (numberOfRounds); i++){
			//Reset Everything
			input[i] = new double[numberResults];
			current_char = '1';
			current_string = "";
			letterPosition = 0;
			//While not at the end of a group, retrieve the current dataset
			while (current_char != group_delimiter){
				if (inputfile.eof()){
					return i;
				}
				//Get the current char
				current_char = inputfile.get();

				if (current_char == individual_delimiter){//Reached the end of the current set


					if (type_of_input == 0){
						//The type is double, convert to double
						input[i][letterPosition] = stod(current_string);

					}
					else if (type_of_input == 1){
						//Is a char, get the first and only letter and make a double
						input[i][letterPosition] = (double)current_string.at(0);

					}
					else if (type_of_input == 2){
						//Is a string, work on conversion later
					}

					letterPosition++;
					current_string = "";
				}
				else{//String has not ended and is still correct
					current_string += current_char;
				}

			}

		}

		//Get current location in file
		int currentPosition = inputfile.tellg();
		inputfile.close();
		return currentPosition;
	}
	else{
		std::cout << "Unable to read from file." << endl;
		std::cout << "continue?";
		if (cin.get() == 'n'){
			exit(0);
		}
		return numberOfRounds;
	}
}

//*********************************
//Running The Network
//*********************************

void recursiveTestInput(CGraphicsNetwork network){
	char userchoice;
	string userstartentry;
	double* input = new double[network.getI_input()];
	//Ask User for input
	std::cout << "Please enter " << network.getI_input() << " inputs seperated by spaces " << endl;
	cin.sync();
	std::getline(std::cin, userstartentry);
	for (int i = 0, k = 0; i < network.getI_input(); i++, k++){
		if (userstartentry.at(k) != ' '){
			input[i] = (double)userstartentry.at(k);
		}
		else{
			k++;
			input[i] = (double)userstartentry.at(k);
		}
	}


	while (true){
		if (pause){
			std::cout << endl << "Would you like to quit? ";
			cin.sync();
			userchoice = cin.get();
			if (userchoice == 'y'){
				exit(0);
			}
			else{
				pause = false;
				//Attach the signal handler
				//std::signal(SIGINT, signal_handler);
				continue;
			}
		}
		network.feedForward(input);
		//Move all in the array to the left by one
		for (int i = 0; i < network.getI_input() - 1; i++){
			input[i] = input[i + 1];
		}

		input[network.getI_input() - 1] = printVectorOutputChar(network.getOutput());


	}
}



void initializeFeedForwardNetwork(int argc, char** argv, CSettings settings){
	CGraphicsNetwork test;
	std::cout << "Would you like to train?";
	cin.sync();
	char in = cin.get();
	bool loadCheckFromFile = false;
	int PROBLEMS = std::stoi(argv[2]);
	SCheckpoint checkpoint = SCheckpoint();
	if (settings.b_loadFromCheckpoint){
		std::cout << "loading checkpoint " << endl;
		//Load the checkpoint from a file
		LoadCheckpointFromFile(checkpoint, settings.s_checkpoint_file);
		//Load the information from the checkpoint
		std::cout << "loading network " << endl;
		test = CGraphicsNetwork(&settings);
		loadFromFile(test, checkpoint.s_network_file_name);
		test.setSettings(&settings);
		loadCheckFromFile = true;
	}
	else if (settings.b_loadNetworkFromFile){//Load only the network from file
		std::cout << "loading network " << endl;
		test = CGraphicsNetwork(&settings);
		loadFromFile(test, settings.s_loadNetworkFile);
		test.setSettings(&settings);
	}
	else{//Start with a brand new network
		std::cout << "creating new network " << endl;
		vector<int> temp = vector<int>();
		temp.push_back(settings.i_input);
		temp.push_back(settings.i_output);
		vector<double> temp2 = vector<double>(settings.i_output);

		test = CGraphicsNetwork(temp, &settings);
	}
	if (in == 'y'){
		double **value;
		double **results;
		double **testIn;
		double **testOut;
		int start_Input;
		int start_output;
		value = new double*[settings.i_number_of_training];
		results = new double*[settings.i_number_of_training];
		testIn = new double*[settings.i_number_of_training];
		testOut = new double*[settings.i_number_of_training];

		if (settings.b_trainingFromFile){
			//Store training set
			std::cout << "loading training set " << endl;
			start_Input = getDataFromFile(settings.s_trainingSet, checkpoint.i_current_position_in_input_file, settings.i_number_of_training, settings.i_input, value, settings.i_trainingSetType);
			start_output = getDataFromFile(settings.s_outputTrainingFile, checkpoint.i_current_position_in_output_file, settings.i_number_of_training, settings.i_output, results, settings.i_outputTrainingSetType);

			if (!loadCheckFromFile){//Create a new checkpoint
				createNewCheckpoint(checkpoint, settings, results);
			}

			if (settings.b_testingFromFile){
				std::cout << "loading testing data " << endl;
				//Store the data to test the neural network
				getDataFromFile(settings.s_testSet, 0, settings.i_number_of_training, settings.i_input, testIn, settings.i_trainingSetType);
				getDataFromFile(settings.s_outputTestSet, 0, settings.i_number_of_training, settings.i_output, testOut, settings.i_outputTrainingSetType);
			}
			else{
				std::cout << "loading testing data " << endl;
				//If no test file is given, use some from the training set
				//Store the testing set
				getDataFromFile(settings.s_trainingSet, 0, settings.i_number_of_training, settings.i_input, testIn, settings.i_trainingSetType);
				getDataFromFile(settings.s_outputTrainingFile, 0, settings.i_number_of_training, settings.i_output, testOut, settings.i_outputTrainingSetType);

			}
		}
		std::cout << "training start " << endl;
		trainNetworkDelta(value, results, test, start_Input, start_output, testIn, testOut, settings.i_number_of_training, settings, checkpoint);

		for (int i = 0; i < settings.i_number_of_training; i++){
			delete value[i];
			delete results[i];
		}

		//Clean up memory
		delete value;
		delete results;
	}
	else{
		std::cout << "Starting output Loops" << endl;
		recursiveTestInput(test);
	}
}

//
//

void initializeRecurrentNetwork(int argc, char** argv, CSettings settings){
	ReccurentLoops RLoops = ReccurentLoops(settings,ReccurentLoops::LongTermShortTerm);
	
	double* temp = new double[settings.i_input];
#ifdef _DEBUG
	for (int i = 0; i < settings.i_input; i++){
		temp[i] = i;
	}
	//RLoops.runNetwork(temp);
	RLoops.testTraining();

#else
	RLoops.startTraining();
#endif
	
}

void initialize_loops(int argc, char** argv){

	CSettings settings;
	if (argc > 1){
		std::cout << "loading settings " << endl;
		settings = loadSettings(argv[1]);
	}
	else{
		string settingsLocation;
		std::cout << "Where are the settings? " << endl;
		std::getline(std::cin, settingsLocation);
		settings = loadSettings(settingsLocation);
	}
	std::cout << "1) Recurrent Neural Network" << endl;
	std::cout << "2) Feedforward Neural Network" << endl;
	switch (cin.get()){
	case '1':
		initializeRecurrentNetwork(argc, argv, settings);
		exit(0);
		break;
	case '2':
		pause = false;
		//Attach the signal handler
		std::signal(SIGINT, signal_handler);
		initializeFeedForwardNetwork(argc, argv, settings);
		exit(0);
		break;


	}


}



//*********************************
//Main Function
//*********************************

int main(int argc, char** argv){
#ifdef WINDOWS_COMPUTER
	initialize();
#endif


	pause = false;

	std::cout << "Starting Program... " << endl;


	initialize_loops(argc, argv);




	return 0;
}

