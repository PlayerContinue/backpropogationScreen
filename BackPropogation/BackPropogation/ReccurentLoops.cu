#include "ReccurentLoops.cuh"


ReccurentLoops::ReccurentLoops()
{

}

ReccurentLoops::ReccurentLoops(CSettings settings){
	this->settings = settings;
	this->mainNetwork = RecurrentNeuralNetwork(settings);
	this->InitializeNetwork();
	VISUALIZE
}

ReccurentLoops::ReccurentLoops(CSettings settings, SCheckpoint checkpoint):ReccurentLoops(settings){


}


void ReccurentLoops::InitializeNetwork(){
	this->input = new weight_type*[this->settings.i_number_of_training];
	this->output = new weight_type*[this->settings.i_number_of_training];
}

bool ReccurentLoops::loadNetworkFromFile(){
	return true;
}


bool ReccurentLoops::runNetwork(){
	return true;
}
//*****************************
//Get Data From the users file
//*****************************
bool ReccurentLoops::load_training_data_from_file(){
	return true;
}

//**********************
//Training
//**********************
void ReccurentLoops::startTraining(){
	do{

		this->load_training_data_from_file();
		VISUALIZE
		this->mainNetwork.addNeuron(1);
		VISUALIZE
		this->mainNetwork.addNeuron(1);
		VISUALIZE
		this->mainNetwork.addWeight(2);
		VISUALIZE
	} while (true);
}

bool ReccurentLoops::train_network(){
	return true;
}
