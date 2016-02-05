/*
Programmer: David Greenberg
Reason : Class designed primarily for testing the network

*/
#include <iostream>
#include <fstream>
#include "TestCode.cuh";
#ifndef CLASS_DEFINED_NSETTINGS
#include "NSettings.h"
#endif
#include "TopologyBase.cuh"
#ifndef CLASS_DEFINED_RRNTOPOLOGY
#include "RNNTopology.cuh"
#endif




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
		RNNTopology host;
		TopologyLayerData *temp;
		switch (start){
		case '1':
			host = RNNTopology(RNNTopology::HOST_DEVICE::DEVICE);
			host.buildTopology(settings);
			temp = &(host.getLayer(0));
			testing::outputToFile<int>(temp->to_vector_begin, temp->to_vector_end, "test", "tests/to_vector.txt");
			testing::outputToFile<int>(temp->from_vector_begin, temp->from_vector_end, "test", "tests/from_vector.txt");
			testing::outputToFile<double>(temp->weight_vector_begin, temp->weight_vector_end, "test", "tests/weights_vector.txt");
			exit(0);
			break;
		case '2':
			
			exit(0);
			break;


		}


	}
};



