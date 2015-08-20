

//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Initizializing algorithm. Contains the main function and only the main function
//----------------------------------------------------------------------------------------

#define PROBLEMS 1000
#pragma once
#include <vector>
#include <stdlib.h>
#include <bitset>
#include <math.h>
#include <iostream>
#include "util.h"
#include "CGraphicNetwork.cuh"
using namespace std;


void printVectorOutput(vector<double> vectorA){
	for (int i = 0; i < vectorA.size(); i++){
		cout << 1 / vectorA.at(i);
		cout << endl;
		cout << vectorA.at(i);
		cout << endl;
	}
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
	double averagedistance = abs(test.getAverageDistance());
		//Add a new layer if the success is too low and either the there are no hidden layers or the previous layer has too many nodes
	if (success < .1 && averagedistance > .002 && (test.getNumLayers() == 2 || test.getSuccessRate() < .3 && test.getPreviousSuccessRate() < .3 && RandBool())){
			test.addLayer(-1, 1);
		}

	if (success < .2 && averagedistance > .001 && test.getSuccessRate() < .8){
			test.addNeuronToLayer(test.getNumLayers() - 2);
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

int main(int argc, char* argv){
	vector<int> temp = vector<int>();
	temp.push_back(2);
	temp.push_back(sizeof(double));
	CGraphicsNetwork test = CGraphicsNetwork(temp, 1, 2);
	double zero =(double) ~0;
	double **value = new double*[PROBLEMS];
	double **results = new double*[PROBLEMS];
	for (int i = 0; i < PROBLEMS; i++){
		value[i] = new double[2];
		double number = (double)i + 1;
		value[i][0] = number;
		value[i][1] = number + 1;
		number = (double)(1 / (number + number + 1));
		results[i] = new double[sizeof(double)];
		for (int j = 0; j < sizeof(double); j++){
			results[i][j] = (double)((number & zero) ? 0 : 1)
		}
		
	}


	for (int i = 0; i < 500; i++){
		trainNetwork2(value, results, test, 0, PROBLEMS, 1000);

		if (i % 2 == 0){
			//Test the output
			testOutput2(value, test, PROBLEMS);
		}

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

