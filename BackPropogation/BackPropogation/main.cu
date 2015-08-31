

//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Initizializing algorithm. Contains the main function and only the main function
//----------------------------------------------------------------------------------------

//#define PROBLEMS 5
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
bool writeToFile(CGraphicsNetwork &network, CSettings settings);
void testOutput2(double** value, CGraphicsNetwork &test, int size);
bool addToNetwork(CGraphicsNetwork &test, CSettings settings, double mean_square_error);
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
	cout << (char)value;
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
		cout << arrayA[i];
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

//Train the current network
//Check if the limit has been reached as the stopping point
//Use mean square error to check distance
void trainNetworkDelta(double* value[], double* results[], CGraphicsNetwork &test, int start, int end, double* testSetIn[], double* testSetOut[], int testLength, CSettings settings){
	int training_position = start;
	double mean_square_error = 0;
	int count_loops = 0;
	int times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
	double lowest_mean_square_error = (double)INT_MAX;
	do{

		test.backprop(value[training_position], results[training_position]);


		if (count_loops % settings.i_loops == 0){
			writeToFile(test, settings);
			//Test the output
			testOutput2(testSetIn, test, testLength);
			cout << " loop " << count_loops << endl;
		}
		training_position++;
		count_loops++;

		if (training_position >= end){
			training_position = start;
		}

		if (count_loops%settings.i_number_before_growth_potential == 0){

			mean_square_error = test.getMeanSquareError(testSetIn, testSetOut, testLength);
			//Set the current lowest
			if (mean_square_error < lowest_mean_square_error){
				lowest_mean_square_error = mean_square_error;
				//Reset value since error was lowered
				times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
			}
			else{
				times_lowest_mean_square_error_to_large--;
				//If the below is true, something close to the limit has been reached, the network needs to change size 
				if (times_lowest_mean_square_error_to_large == 0){
					//Add new nodes to the network
					if (addToNetwork(test, settings, mean_square_error)){
						//Reset the number allowed
						//Since new ones may have been
						times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures + RandInt(10, 50);
					}
					else{
						//Reset the number of times before a growth is attempted
						times_lowest_mean_square_error_to_large = settings.i_number_allowed_failures;
					}
				}
			}
		}

		//Loop until the error is smaller than the threshold
	} while (settings.d_threshold < mean_square_error && training_position == start || training_position != start);
}

//Returns true if a neuron was added
bool addToNetwork(CGraphicsNetwork &test, CSettings settings, double mean_square_error){

	//Get delta in success
	double success = test.getSuccessRate() - test.getPreviousSuccessRate();
	double averagedistance = abs(test.getPreviousAverageDistance() - test.getAverageDistance());
	double delta = abs(test.getAverageDelta());
	double mean_square_error_dif = mean_square_error - settings.d_threshold;
#ifdef FULL_SUCCESS
	double full_success = test.getFullSuccessRate() - test.getFullPreviousSuccessRate();
#endif
	//Add a new layer if the success is too low and the threshold has not been reached
	if (success <= settings.d_neuron_success_threshold && mean_square_error_dif > 0 && mean_square_error_dif > settings.d_neuron_distance_threshold &&  mean_square_error_dif < settings.d_row_distance_threshold){
		if (test.getNumLayers() == 2){
			test.addLayer(test.getNumLayers() + 1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) / 5);
		}
		else{
			test.addNeuronToLayer(5, test.getNumLayers() - 2, 2);
		}
		test.resetNetwork();
		return true;
	}
	else if (success <= settings.d_row_success_threshold && mean_square_error_dif > 0 && mean_square_error_dif >= settings.d_row_distance_threshold){
		test.addLayer(test.getNumLayers() + 1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) / 5);
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
		printArray(value[i], 2);
		test.feedForward(value[i]);
		temp2 = test.getOutput();
		cout << "output";
		cout << endl;
		printVectorOutput(temp2);
		cout << endl;
	}
}

//Output the network to a file
bool writeToFile(CGraphicsNetwork &network, CSettings settings){
	static int file_number = 0;
	file_number++;
	std::ofstream outputfile;
	outputfile.open("networks/" + settings.s_network_name + std::to_string(file_number) + ".txt", ios::trunc);
	if (outputfile.is_open()){
		//Output the network
		outputfile << network << flush;
		outputfile.close();
		return true;
	}
	else{
		cout << "Unable to write checkpoint to file." << endl;
		cout << "continue?";
		return false;
	}


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

void getDataFromFile(string fileName, int start, int end, int numberResults, double** input){
	std::ifstream inputfile;

	inputfile.open(fileName);
	if (inputfile.is_open()){
		inputfile.seekg(start);
		int k = -1;
		int letterPosition = 0;
		for (int i = 0; i < end - start; i++){
			if (i%numberResults == 0){
				{
					k++;
					input[k] = new double[numberResults];
					letterPosition = 0;
				}

				input[k][letterPosition] = (int)inputfile.get();
				letterPosition++;
			}


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


int main(int argc, char** argv){
#ifdef WINDOWS_COMPUTER
	initialize();
#endif
	CGraphicsNetwork test;
	CSettings settings;
	if (argc > 1){
		settings = loadSettings(argv[1]);
	}

	int PROBLEMS = std::stoi(argv[2]);

	if (settings.b_loadNetworkFromFile){
		test = CGraphicsNetwork();
		loadFromFile(test, settings.s_loadNetworkFile);
	}
	else{

		vector<int> temp = vector<int>();
		temp.push_back(settings.i_input);
		temp.push_back(settings.i_output);
		vector<double> temp2 = vector<double>(settings.i_output);

		test = CGraphicsNetwork(temp, settings.d_beta, settings.d_alpha);
	}
	double **value;
	double **results;
	double **testIn;
	double **testOut;
	if (settings.b_trainingFromFile && settings.b_trainingFromFile){
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

		double **testIn = new double*[PROBLEMS];
		double **testOut = new double*[PROBLEMS];

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

	}
	else{
		value = new double*[settings.i_number_of_training];
		results = new double*[settings.i_number_of_training];
		getDataFromFile(settings.s_trainingSet, 0, settings.i_number_of_training, settings.i_input, value);
		//trainNetwork2(value, results, test, 0, PROBLEMS, settings.i_loops);
		trainNetworkDelta(value, results, test, 0, settings.i_number_of_training, testIn, testOut, 100, settings);
	}






	testOutput2(value, test, PROBLEMS);

	for (int i = 0; i < PROBLEMS; i++){
		delete value[i];
		delete results[i];
	}

	//Clean up memory
	delete value;
	delete results;

	return 0;
}

