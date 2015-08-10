

//---------------------------------------------------------------------------------------
//Author:David Greenberg
//Desc: Initizializing algorithm. Contains the main function and only the main function
//----------------------------------------------------------------------------------------
#pragma once
#include <vector>
#include "Network.h"
#include <stdlib.h>
#include <iostream>
using namespace std;


void printVectorOutput(vector<double> vectorA){
	for (int i = 0; i < vectorA.size(); i++){
		cout << vectorA.at(i);
		cout << endl;
	}
}

void printArray(double* arrayA){
	int test = sizeof(*arrayA);
	int size = sizeof(*arrayA) / sizeof(arrayA[0]);
	for (int i = 0; i < size; i++){
		cout << arrayA[i];
		cout << endl;
	}
}

int main(int argc, char* argv){
	vector<int> temp = vector<int>();
	temp.push_back(1);
	temp.push_back(10);
	temp.push_back(10);
	temp.push_back(10);
	temp.push_back(2);
	CNetwork test = CNetwork(temp, 1, 2);
	double value_1[1] = { 1.0 };
	double value_2[2] = { .2, .5 };
	double value_3[1] = { 2.0 };
	double value_4[2] = { .7, .1 };
	vector<double> temp2;

	//Train the network on simple test values
	for (int i = 0; i < 20000; i++){
		if (i % 2 == 0){
			test.backprop(value_1, value_2);
		}
		else{
			test.backprop(value_3, value_4);
		}
	}
	
	//Test the output

	
	cout << "input";
	cout << endl;
	printArray(value_1);
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
	printArray(value_3);

	temp2 = test.getOutput();
	cout << "output";
	cout << endl;
	printVectorOutput(temp2);
	cout << endl;
	

	cout << "Add New Layer";
	cout << endl;



	//Test adding a new layer
	test.addLayer(-1, 1);

	cout << "input";
	cout << endl;
	printArray(value_1);
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
	printArray(value_3);

	temp2 = test.getOutput();
	cout << "output";
	cout << endl;
	printVectorOutput(temp2);
	cout << endl;

	//Trained

	//Train the network on simple test values
	for (int i = 0; i < 1000; i++){
		if (i % 2 == 0){
			test.backprop(value_1, value_2);
		}
		else{
			test.backprop(value_3, value_4);
		}
	}

	cout << "Training";
	cout << endl;

	cout << "input";
	cout << endl;
	printArray(value_1);
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
	printArray(value_3);

	temp2 = test.getOutput();
	cout << "output";
	cout << endl;
	printVectorOutput(temp2);
	cout << endl;

	//Add a New Neuron
	test.addNeuronToLayer(3);

	int i = 0;
}

