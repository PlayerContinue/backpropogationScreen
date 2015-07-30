

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
	for (int i = 0; i < 20000; i++){
		if (i % 2 == 0){
			test.backprop(value_1, value_2);
		}
		else{
			test.backprop(value_3, value_4);
		}
	}
	
	test.feedForward(value_1);
	vector<double> temp2 = test.getOutput();

	temp2 = test.getOutput();
	
	test.feedForward(value_3);
	temp2 = test.getOutput();

	int i = 0;
}