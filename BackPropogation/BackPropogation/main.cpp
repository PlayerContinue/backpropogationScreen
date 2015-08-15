

//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Initizializing algorithm. Contains the main function and only the main function
//----------------------------------------------------------------------------------------

#define PROBLEMS 50
#pragma once
#include <vector>
#include "util.h"
#include "Network.h"
#include <stdlib.h>
#include <iostream>
using namespace std;


void printVectorOutput(vector<double> vectorA){
	for (int i = 0; i < vectorA.size(); i++){
		cout << 1 / vectorA.at(i);
		cout << endl;
		cout << vectorA.at(i);
		cout << endl;
	}
}

void printArray(double* arrayA,int size){
	for (int i = 0; i < size; i++){
		cout << arrayA[i];
		cout << endl;
	}
}

void trainNetwork(double* value_1, double* value_2, double* value_3, double* value_4, CNetwork &test, int rounds){
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

void trainNetwork2(double* value[], double* results[], CNetwork &test, int start, int end, int rounds){
	
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

void addToNetwork(CNetwork &test){

	//Add a new layer if the success is too low and either the there are no hidden layers or the previous layer has too many nodes
	if (test.getSuccessRate() < .8 && (test.getNumLayers() == 2 || test.getNumNeuronsInLayer(test.getNumLayers() - 2) > 100)){
		test.addLayer(-1, 1);
	}

	if (test.getSuccessRate() < .8){
		test.addNeuronToLayer(test.getNumLayers() - 2);
	}
	test.resetNetwork();

}

void testOutput(double* value_1, double* value_3, CNetwork &test){
	vector<double> temp2;
	cout << "input";
	cout << endl;
	printArray(value_1,2);
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
	printArray(value_3,2);

	temp2 = test.getOutput();
	cout << "output";
	cout << endl;
	printVectorOutput(temp2);
	cout << endl;
}

void testOutput2(double** value, CNetwork &test, int size){
	vector<double> temp2;

	for (int i = 0; i < size;i++){
		cout << "input";
		cout << endl;
		printArray(value[i],2);
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
	temp.push_back(1);
	CNetwork test = CNetwork(temp, 1, 2);
	double **value = new double*[PROBLEMS];
	double **results = new double*[PROBLEMS];
	for (int i = 0; i < PROBLEMS; i++){
		value[i] = new double[2];
		double number = (double)i;
		value[i][0] = number;
		value[i][1] = number + 1;
		number = (double)(1/(number + number + 1));
		results[i] = new double[1];
		results[i][0] = number;
	}


	for (int i = 0; i < 500; i++){
		trainNetwork2(value, results, test, 0,PROBLEMS, 1000);

		if (i % 100 == 0){
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

	int i = 0;
}

