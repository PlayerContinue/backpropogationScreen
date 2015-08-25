

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
#include "util.h"
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
	double averagedistance = abs(test.getAverageDistance() - test.getPreviousAverageDistance());
	double delta = abs(test.getAverageDelta());
	double distanceMeasure = .01;
	//Add a new layer if the success is too low and either the there are no hidden layers or the previous layer has too many nodes
	if (success < .1 && averagedistance < distanceMeasure * .0000001 && (test.getNumLayers() == 2 || test.getSuccessRate() < .3 && test.getPreviousSuccessRate() < .3)){
		test.addLayer(-1, test.getNumNeuronsInLayer(test.getNumLayers()-1)/5);
	}else if (success < .2 && averagedistance < distanceMeasure * .0003 && test.getSuccessRate() < .8){
		if (test.getNumLayers() == 2){
			test.addLayer(-1, test.getNumNeuronsInLayer(test.getNumLayers() - 1) / 5);
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

int main(int argc, char** argv){
	int PROBLEMS = (((int)argv[1][0])-48);
	vector<int> temp = vector<int>();
	temp.push_back(2);
	temp.push_back(32);
	vector<double> temp2 = vector<double>(32);

	CGraphicsNetwork test = CGraphicsNetwork(temp, 1, 2);
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
		trainNetwork2(value, results, test, 0, PROBLEMS, 1000);

		if (i % 2 == 0 || true){
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

