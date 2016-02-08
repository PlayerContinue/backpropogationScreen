/*
Programmer: David Greenberg
Reason : Class designed primarily for testing the network

*/
#include <iostream>
#include <fstream>
#include "TestCode.cuh";
#include "NSettings.h"
#include "TopologyBase.cuh"
#include "RNNTopology.cuh"
#include "TrainerBase.cuh"
#include "RNNTrainer.cuh"





class StartClass{
public:	
	NSettings loadSettings(string fileName){
		std::ifstream inputfile;
		inputfile.open(fileName, ios_base::beg);
		NSettings settings;
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

	void initialize_loops(int argc, char** argv){

		NSettings settings;
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
		char start;
		if (argc > 2){
			start = argv[2][0];
		}
		else{
			std::cout << "1) Recurrent Neural Network" << endl;
			std::cout << "2) Feedforward Neural Network" << endl;
			start = cin.get();
		}
		TrainerBase *host;
		TopologyBase *top;
		TopologyLayerData temp;
		thrust::device_vector<WEIGHT_TYPE> input = thrust::device_vector<WEIGHT_TYPE>(thrust::make_constant_iterator((WEIGHT_TYPE)1), thrust::make_constant_iterator((WEIGHT_TYPE)1)+settings.i_input);
		thrust::device_vector<WEIGHT_TYPE> output = thrust::device_vector<WEIGHT_TYPE>(thrust::make_constant_iterator((WEIGHT_TYPE)1), thrust::make_constant_iterator((WEIGHT_TYPE)1) + settings.i_output);
		std::ofstream outputfile;
		switch (start){
		case '1':
			host = new RNNTrainer();
			top = new RNNTopology();
			host->createTrainingEnviornment(*top,settings);
			host->train(input,output);
			outputfile.precision(30);
			outputfile.open(settings.s_network_name, ios::trunc);
			host->createCheckpoint(outputfile);
			outputfile.close();
			exit(0);
			break;
		case '2':
			
			exit(0);
			break;


		}


	}
};



