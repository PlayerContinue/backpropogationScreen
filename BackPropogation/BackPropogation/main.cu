

//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Initizializing algorithm. Contains the main function and only the main function
//----------------------------------------------------------------------------------------

//#define PROBLEMS 5
#define FIRST_TEST
#pragma once
#include <vector>
#include <stdlib.h>
#include <bitset>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include "CSettings.h"
#include "CGraphicNetwork.cuh"
#ifdef WINDOWS_COMPUTER
#include <wincon.h>
#endif

using namespace std;

//PROTOTYPES
int writeToFile(CGraphicsNetwork &network, CSettings settings);
void testOutput2(double** value, CGraphicsNetwork &test, int size);
bool addToNetwork(CGraphicsNetwork &test, CSettings settings, SCheckpoint& checkpoint, double** testSet, double mean_square_error);
void getDataFromFile(string fileName, int start, int end, int numberResults, double** input);

void printVectorOutput(vector<double> vectorA){
	int size = vectorA.size();
	int value = 0;//Store a int with 32 0
	int pos = 1;
	for (int i = vectorA.size() - 1; i >= 0; i--){

		//Either add one at the current position or add zero
		value = value | (vectorA[i] > .5 ? pos : 0);

		pos = pos << 1;
	}
	cout << value << endl;
	if (value <= 255 & value > 0){
		cout << (char)value << endl;
	}
	
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
	cout << value;
}

void printArray(double* arrayA, int size){
	for (int i = 0; i < size; i++){
		cout << (char)arrayA[i];
		cout << endl;
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
			for (int j = 0; j < output.size(); j++){
				if (output[j] != output2[j]){
					count_success++;
					break;
				}
			}
		}
	}

	return (size-count_success);
}

//******************************
//Save, Load, Create Checkpoints
//******************************

//Create a checkpoint containing all the current values of the system
void createCheckpoint(CGraphicsNetwork test,SCheckpoint& checkpoint,CSettings settings){
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
		cout << "Unable to write checkpoint to file." << endl;
		cout << "continue?";
	}

}

void LoadCheckpointFromFile(SCheckpoint& checkpoint,string s_file_name){
	std::ifstream inputfile;

	inputfile.open(s_file_name, ios_base::beg);
	if (inputfile.is_open()){
		inputfile >> checkpoint;
		inputfile.close();
	}
	else{
		cout << "Unable to read checkpoint from file." << endl;
		cout << "continue?";
		if (cin.get() == 'n'){

			exit(0);
		}
	}
}

//Create a new checkpoint object from the created settings
void createNewCheckpoint(SCheckpoint& checkpoint, CSettings settings){
	checkpoint.i_number_of_loops_checkpoint = 0;
	checkpoint.i_number_of_loops = 0;
	checkpoint.i_times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
	checkpoint.d_mean_square_error = (double)INT_MAX;
	checkpoint.d_lowest_mean_square_error = (double)INT_MAX;
	checkpoint.d_previous_mean_square_error = 0;
	checkpoint.d_row_distance_threshold = settings.d_row_distance_threshold;
	checkpoint.d_neuron_distance_threshold = settings.d_neuron_distance_threshold;
}

//******************************
//Training the Network
//******************************

//Train the current network
//Check if the limit has been reached as the stopping point
//Use mean square error to check distance
void trainNetworkDelta(double* value[], double* results[], CGraphicsNetwork &test, int start, int end, double* testSetIn[], double* testSetOut[], int testLength, CSettings settings, SCheckpoint checkpoint){
	do{

		test.backprop(value[checkpoint.i_number_of_loops], results[checkpoint.i_number_of_loops]);


		if (checkpoint.i_number_of_loops_checkpoint % settings.i_loops == 0){
			createCheckpoint(test,checkpoint, settings);
			//Test the output
			testOutput2(testSetIn, test, testLength);
			cout << " loop " << checkpoint.i_number_of_loops_checkpoint << endl;
		}
		checkpoint.i_number_of_loops++;
		checkpoint.i_number_of_loops_checkpoint++;

		if (checkpoint.i_number_of_loops >= end){
			checkpoint.i_number_of_loops = start;

			if (settings.b_trainingFromFile){

				//Store training set
				getDataFromFile(settings.s_trainingSet, checkpoint.i_number_of_loops_checkpoint*settings.i_input, settings.i_number_of_training, settings.i_input, value);

			}

			if (settings.b_testingFromFile){
				getDataFromFile(settings.s_outputTrainingFile, checkpoint.i_number_of_loops_checkpoint*settings.i_output, settings.i_number_of_training, settings.i_output, results);
			}
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
				//If the below is true, something close to the limit has been reached, the network needs to change size 
				if (checkpoint.i_times_lowest_mean_square_error_to_large == 0){
					//Add new nodes to the network
					if (addToNetwork(test, settings,checkpoint, testSetIn, checkpoint.d_mean_square_error)){
						//Reset the number allowed
						//Since new ones may have been
						checkpoint.i_times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures + 10;
					}
					else{
						//Reset the number of times before a growth is attempted
						checkpoint.i_times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
					}
				}
			}
			
		}

		//Loop until the error is smaller than the threshold
	} while (settings.d_threshold < checkpoint.d_mean_square_error && checkpoint.i_number_of_loops == start || checkpoint.i_number_of_loops != start);
}

//Returns true if a neuron was added
bool addToNetwork(CGraphicsNetwork &test,CSettings settings,SCheckpoint& checkpoint, double** testSet, double mean_square_error){

	//Get delta in success
	double success = test.getSuccessRate() - test.getPreviousSuccessRate();
	double averagedistance = abs(test.getPreviousAverageDistance() - test.getAverageDistance());
	double delta = abs(test.getAverageDelta());
	double mean_square_error_dif = mean_square_error - settings.d_threshold;
#ifdef FULL_SUCCESS
	double full_success = test.getFullSuccessRate() - test.getFullPreviousSuccessRate();
#endif
	//Get the number of test sets returning the exact same values
	int numberSame = numberFullSame(test,testSet, settings.i_number_of_training);

	//Add a new layer if the success is too low and the threshold has not been reached
	if (success <= settings.d_row_success_threshold && mean_square_error_dif > 0 && mean_square_error_dif >= checkpoint.d_row_distance_threshold && numberSame < settings.i_number_allowed_same){
		test.addLayer(-1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) * 5);
		//Increment the size of the need mean distance to get a new layer
		//And decrease the size of the needed mean distance to get a new neuron
		if (checkpoint.d_neuron_distance_threshold > 0 ){
			checkpoint.d_row_distance_threshold += settings.d_row_distance_threshold * .1;
			checkpoint.d_neuron_distance_threshold -= settings.d_neuron_distance_threshold * .1;
		}
		test.resetNetwork();
		return true;
	}
	else if (success <= settings.d_neuron_success_threshold && mean_square_error_dif > 0 && mean_square_error_dif >= checkpoint.d_neuron_distance_threshold){
		if (test.getNumLayers() == 2){
			test.addLayer(-1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) * 5);
			//Increment the size of the need mean distance to get a new layer
			//And decrease the size of the needed mean distance to get a new neuron
			if (checkpoint.d_neuron_distance_threshold > 0){
				checkpoint.d_row_distance_threshold += settings.d_row_distance_threshold * .1;
				checkpoint.d_neuron_distance_threshold -= settings.d_neuron_distance_threshold * .1;
			}
		}
		else{
			test.addNeuronToLayer(1, test.getNumLayers() - 2, 8);
			//Increment the size of the need mean distance to get a new neuron
			//And decrease the size of the needed mean distance to get a new row
			if (checkpoint.d_row_distance_threshold > 0){
				checkpoint.d_row_distance_threshold -= settings.d_row_distance_threshold * .1;
				checkpoint.d_neuron_distance_threshold += settings.d_neuron_distance_threshold * .1;
			}
		}
		test.resetNetwork();
		return true;
	}
	
	return false;


}

void testOutput(double* value_1, double* value_3, CGraphicsNetwork &test){
	vector<double> temp2;
	cout << "input";
	cout << endl;
	printArray(value_1, 2);
	test.feedForward(value_1);
	temp2 = test.getOutput();
	cout << "output";
	cout << endl;
	printVectorOutput(temp2);
	cout << endl;


	test.feedForward(value_3);
	temp2 = test.getOutput();

	cout << "input";
	cout << endl;
	printArray(value_3, 2);

	temp2 = test.getOutput();
	cout << "output";
	cout << endl;
	printVectorOutput(temp2);
	cout << endl;
}

void testOutput2(double** value, CGraphicsNetwork &test, int size){
	vector<double> temp2;

	for (int i = 0; i < size; i++){
		cout << "input";
		cout << endl;
		
			printArray(value[i], 6);
		
		test.feedForward(value[i]);
		
			temp2 = test.getOutput();
		
		cout << "output";
		cout << endl;
		printVectorOutput(temp2);
		cout << endl;
	}
}

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
		cout << "Unable to write checkpoint to file." << endl;
		cout << "continue?";
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
		cout << "Unable to read from file." << endl;
		cout << "continue?";
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
		cout << "Unable to read from file." << endl;
		cout << "continue?";
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
		cout << "continue? ";
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
void getDataFromFile(string fileName, int start, int numberOfRounds, int numberResults, double** input){
	std::ifstream inputfile;

	inputfile.open(fileName);
	if (inputfile.is_open()){
		inputfile.seekg(start);
		int k = -1;
		int letterPosition = 0;
		for (int i = 0; i < (numberOfRounds)*numberResults; i++){
			if (i%numberResults == 0){
				k++;
				input[k] = new double[numberResults];
				letterPosition = 0;
			}

			input[k][letterPosition] = (int)inputfile.get();
#ifdef FIRST_TEST 
			if (input[k][letterPosition] == 48.0){
				input[k][letterPosition] = .1;
			}
			else if (input[k][letterPosition] == 49.0){
				input[k][letterPosition] = .9;
			}
#endif


			letterPosition++;



		}
		inputfile.close();
	}
	else{
		cout << "Unable to read from file." << endl;
		cout << "continue?";
		if (cin.get() == 'n'){

			exit(0);
		}
	}
}

void initialize_loops(int argc, char** argv){
	CGraphicsNetwork test;
	CSettings settings;
	if (argc > 1){
		settings = loadSettings(argv[1]);
	}

	int PROBLEMS = std::stoi(argv[2]);
	SCheckpoint checkpoint = SCheckpoint();
	if (settings.b_loadFromCheckpoint){
		//Load the checkpoint from a file
		LoadCheckpointFromFile(checkpoint,settings.s_checkpoint_file);
		//Load the information from the checkpoint
		test = CGraphicsNetwork();
		loadFromFile(test, checkpoint.s_network_file_name);
	}
	else if (settings.b_loadNetworkFromFile){//Load only the network from file
		test = CGraphicsNetwork();
		createNewCheckpoint(checkpoint,settings);
		loadFromFile(test, settings.s_loadNetworkFile);
	}
	else{//Start with a brand new network

		vector<int> temp = vector<int>();
		createNewCheckpoint(checkpoint,settings);
		temp.push_back(settings.i_input);
		temp.push_back(settings.i_output);
		vector<double> temp2 = vector<double>(settings.i_output);

		test = CGraphicsNetwork(temp, settings.d_beta, settings.d_alpha);
	}
	double **value;
	double **results;
	double **testIn;
	double **testOut;
	
	value = new double*[settings.i_number_of_training];
	results = new double*[settings.i_number_of_training];
	testIn = new double*[settings.i_number_of_training];
	testOut = new double*[settings.i_number_of_training];


	//Store training set
	getDataFromFile(settings.s_trainingSet, 0, settings.i_number_of_training, settings.i_input, value);
	getDataFromFile(settings.s_outputTrainingFile, 0, settings.i_number_of_training, settings.i_output, results);

	//Store training set
	getDataFromFile(settings.s_trainingSet, 0, settings.i_number_of_training, settings.i_input, testIn);
	getDataFromFile(settings.s_outputTrainingFile, 0, settings.i_number_of_training, settings.i_output, testOut);

	trainNetworkDelta(value, results, test, 0, settings.i_number_of_training, testIn, testOut, settings.i_number_of_training, settings, checkpoint);

	for (int i = 0; i < settings.i_number_of_training; i++){
		delete value[i];
		delete results[i];
	}
	
	
	/*if (!settings.b_trainingFromFile && !settings.b_trainingFromFile){
		int zero;
		int number2;
		value = new double*[PROBLEMS];
		results = new double*[PROBLEMS];
		for (int i = 0; i < PROBLEMS; i++){
			value[i] = new double[2];
			double number = (double)RandInt(0, PROBLEMS) + 1;
			value[i][0] = number;
			value[i][1] = number + 1;
			//number = (double)(1 / (number + number + 1));
			results[i] = new double[32];
			number2 = number + number + 1;
			zero = 1;
			for (int j = 31; j >= 0; j--){
				results[i][j] = (double)(((int)(number2 & zero)) != 0 ? .7 : .1);
				//Shift left by one
				zero = zero << 1;
			}
		}

		testIn = new double*[PROBLEMS];
		testOut = new double*[PROBLEMS];

		for (int i = 0; i < 100; i++){
			testIn[i] = new double[2];
			double number = (double)RandInt(0, PROBLEMS) + 1;
			testIn[i][0] = number;
			testIn[i][1] = number + 1;
			testOut[i] = new double[32];
			number2 = number + number + 1;
			zero = 1;
			for (int k = 31; k >= 0; k--){
				testOut[i][k] = (double)(((int)(number2 & zero)) != 0 ? .7 : .1);
				//Shift left by one
				zero = zero << 1;
			}
		}
		//trainNetwork2(value, results, test, 0, PROBLEMS, settings.i_loops);
		trainNetworkDelta(value, results, test, 0, PROBLEMS, testIn, testOut, 100, settings);
		for (int i = 0; i < PROBLEMS; i++){
			delete value[i];
			delete results[i];
		}
	}
	else{*/
		
	//}




	//Clean up memory
	delete value;
	delete results;
}

int main(int argc, char** argv){
#ifdef WINDOWS_COMPUTER
	initialize();
#endif
	initialize_loops(argc, argv);

	return 0;
}

