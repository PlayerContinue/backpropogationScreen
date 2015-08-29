

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
using namespace std;


void printVectorOutput(vector<double> vectorA){
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
void trainNetworkDelta(double* value[], double* results[], CGraphicsNetwork &test, int start, int end, CSettings settings){
	int training_position = start;
	double mean_square_error = 0;
	double lowest_mean_square_error = (double) INT_MAX;
	do{
		
		test.backprop(value[training_position], results[training_position]);
		training_position++;

		if (training_position >= end){
			training_position = start;
			
			mean_square_error = test.getMeanSquareError(value, results, end - start);
			//Set the current lowest
			if (mean_square_error < lowest_mean_square_error){
				lowest_mean_square_error = mean_square_error;
			}
		}
		//Loop until the previously smallest mean is no longer the smallest
	} while (checkThreshold<double>(mean_square_error,lowest_mean_square_error + settings.d_fluctuate_square_mean,settings.d_threshold) && training_position==start || training_position != start);
}

void addToNetwork(CGraphicsNetwork &test,CSettings settings){

	//Get delta in success
	double success = test.getSuccessRate() - test.getPreviousSuccessRate();
	double averagedistance = test.getPreviousAverageDistance() - test.getAverageDistance();
	double delta = abs(test.getAverageDelta());
#ifdef FULL_SUCCESS
	double full_success = test.getFullSuccessRate() - test.getFullPreviousSuccessRate();
#endif
	//Add a new layer if the success is too low and the threshold has not been reached
	if (averagedistance<settings.d_row_distance_threshold && success <= settings.d_row_success_threshold){
		test.addLayer(test.getNumLayers() + 1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) / 5);
	}
	else if (averagedistance<settings.d_neuron_distance_threshold && success <= settings.d_neuron_success_threshold){
		if (test.getNumLayers() == 2){
			test.addLayer(test.getNumLayers() + 1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) / 5);
		}
		else{
			test.addNeuronToLayer(1, test.getNumLayers() - 2, 2);
		}
	}
	test.resetNetwork();

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
bool writeToFile(CGraphicsNetwork &network, int fileNumber){
	std::ofstream outputfile;
	char file_name[20];
	sprintf(file_name, "network%d.txt", fileNumber);
	outputfile.open(file_name, ios::trunc);
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
		return settings;
	}
}



int main(int argc, char** argv){
	
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
		temp.push_back(2);
		temp.push_back(32);
		vector<double> temp2 = vector<double>(32);

		test = CGraphicsNetwork(temp, settings.d_beta, settings.d_alpha);
	}
	int zero;
	int number2;
	double **value = new double*[PROBLEMS];
	double **results = new double*[PROBLEMS];
	for (int i = 0; i < PROBLEMS; i++){
		value[i] = new double[2];
		double number = (double)i + 1;
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

	for (int i = 0; i < 500; i++){
		try{
			if (i != 0){
				trainNetwork2(value, results, test, 0, PROBLEMS, settings.i_loops);
				trainNetworkDelta(value, results, test, 0, PROBLEMS,settings);
			}
			else{
				trainNetwork2(value, results, test, 0, PROBLEMS, settings.i_loops);
				trainNetworkDelta(value, results, test, 0, PROBLEMS, settings);
			}
		}
		catch (exception e){
			cout << i << endl;
		}

		//Test the output
		testOutput2(value, test, PROBLEMS);
		cout << " loop " << i << endl;
		writeToFile(test, i % 50);

		//Add new nodes to the network
		addToNetwork(test,settings);

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

