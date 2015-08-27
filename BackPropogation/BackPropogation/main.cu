

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

void addToNetwork(CGraphicsNetwork &test){

	//Get delta in success
	double success = test.getSuccessRate() - test.getPreviousSuccessRate();
	double averagedistance = test.getPreviousAverageDistance() - test.getAverageDistance();
	double delta = abs(test.getAverageDelta());
	double distanceMeasure = 1;
#ifdef FULL_SUCCESS
	double full_success = test.getFullSuccessRate() - test.getFullPreviousSuccessRate();
#endif
	//Add a new layer if the success is too low and either the there are no hidden layers or the previous layer has too many nodes
	if (
#ifdef FULL_SUCCESS
		(success!=0 && success < .2 || success == 0 && full_success < .3) &&

#else
		success < .1 &&
#endif

		((-1 * distanceMeasure * .002) > averagedistance
		|| 0 < averagedistance && averagedistance < distanceMeasure * .00000000000000000000001))
	{
		test.addLayer(test.getNumLayers() + 1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) / 5);
	}
	else if (

#ifdef FULL_SUCCESS
		(success != 0 && success < .2 || success == 0 && full_success < .5) &&
#else

		success < .2 &&
#endif

		((-1 * distanceMeasure * .0002) > averagedistance
		|| 0 < averagedistance && averagedistance < (distanceMeasure * .00000003))){
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
		return true;
	}
	else{
		cout << "Unable to read from file." << endl;
		cout << "continue?";
		return false;
	}
}

int main(int argc, char** argv){
	int PROBLEMS = std::stoi(argv[1]);
	int loop = std::stoi(argv[2]);
	CGraphicsNetwork test;
	if (argc > 3){
		test = CGraphicsNetwork();
		loadFromFile(test, argv[3]);
	}
	else{

		vector<int> temp = vector<int>();
		temp.push_back(2);
		temp.push_back(32);
		vector<double> temp2 = vector<double>(32);
	
	test = CGraphicsNetwork(temp, 1, 2);
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
				trainNetwork2(value, results, test, 0, PROBLEMS, loop);
			}
			else{
				trainNetwork2(value, results, test, 0, PROBLEMS, loop);
			}
		}
		catch (exception e){
			cout << i << endl;
		}

		//Test the output
		testOutput2(value, test, PROBLEMS);
		cout << " loop " << i << endl;;
		writeToFile(test, i);

		//Add new nodes to the network
		addToNetwork(test);

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

