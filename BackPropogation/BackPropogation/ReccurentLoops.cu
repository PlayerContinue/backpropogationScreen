#include "ReccurentLoops.cuh"

//*****************************
//Constructor
//*****************************
ReccurentLoops::ReccurentLoops()
{

}

ReccurentLoops::ReccurentLoops(CSettings settings){
	this->settings = settings;
	this->InitializeNetwork();
	this->checkpoint = CRecurrentCheckpoint();
	this->mainNetwork = new RecurrentNeuralNetwork(settings);

}

ReccurentLoops::ReccurentLoops(CSettings settings, int type){
	this->settings = settings;

	switch (type){
	case ReccurentLoops::RealTimeTraining:
		//Train the network using real time recurrent
		//this->train_network_RealTimeRecurrentTraining();
		break;
	case ReccurentLoops::HessianFreeOptimization:
		this->mainNetwork = new RecurrentNeuralNetwork(settings);
		break;
	case ReccurentLoops::LongTermShortTerm:
		this->mainNetwork = new LongTermShortTermNetwork(settings);
		break;
	}

	
	this->InitializeNetwork();
	this->checkpoint = CRecurrentCheckpoint();
}

ReccurentLoops::ReccurentLoops(CSettings settings, CRecurrentCheckpoint checkpoint) :ReccurentLoops(settings){
	this->checkpoint = checkpoint;

}

//*****************************
//Intialzie the Internal Requirements For Training
//*****************************

void ReccurentLoops::InitializeNetwork(){
	this->input = new weight_type*[this->settings.i_number_of_training];
	this->output = new weight_type*[this->settings.i_number_of_training];
}


//*****************************
//Reload a Network From a File
//*****************************
bool ReccurentLoops::loadNetworkFromFile(){
	
	return true;
}

//*****************************
//Convert Input Types to Required Type
//*****************************
template <typename T>
weight_type* ReccurentLoops::convert_array(T* in){
	weight_type* temp = new weight_type[this->settings.i_input];
	for (int i = 0; i < settings.i_input; i++){
		temp[i] = (weight_type)in[i];
	}
	return temp;
}

vector<RETURN_WEIGHT_TYPE> ReccurentLoops::runNetwork(int* in){

	return this->runNetwork(this->convert_array<int>(in));

}

vector<RETURN_WEIGHT_TYPE> ReccurentLoops::runNetwork(weight_type* in){
	//this->mainNetwork->InitializeRun();
	device_vector<weight_type> temp_device = this->mainNetwork->runNetwork(in);
	vector<RETURN_WEIGHT_TYPE> to_return = vector <RETURN_WEIGHT_TYPE>(temp_device.size());

	for (unsigned int i = 0; i < temp_device.size(); i++){
		to_return[i] = temp_device[i];
	}
	clear_vector::free(temp_device);

	return to_return;
}
//*****************************
//Get Data From the users file
//*****************************
bool ReccurentLoops::load_training_data_from_file(){
	for (int i = 0; i < this->settings.i_number_of_training; i++){
		this->input[i] = this->createTestInputOutput(this->settings.i_input,0);
	}
	for (int i = 0; i < this->settings.i_number_of_training; i++){
		this->output[i] = this->createTestInputOutput(this->settings.i_output, 1);
	}
	
	return true;
}

//**********************
//Training
//**********************
void ReccurentLoops::startTraining(int type){
	//Load the data from a file
	if (!load_training_data_from_file()){
		throw exception("Unable to read from file.");
	}
	
	
	switch (type){
	case ReccurentLoops::RealTimeTraining:
		//Train the network using real time recurrent
		this->train_network_RealTimeRecurrentTraining();
		break;
	case ReccurentLoops::HessianFreeOptimization:
		this->train_network_HessianFreeOptimizationTraining();
		break;
	}
}

#ifdef _DEBUG 
void ReccurentLoops::testTraining(){
	try{
		this->load_training_data_from_file();
		this->mainNetwork->InitializeTraining();
		for (int i = 0; i < this->settings.i_loops; i++){
			this->mainNetwork->StartTraining(this->input[this->checkpoint.i_number_of_loops_checkpoint], this->output[this->checkpoint.i_number_of_loops_checkpoint]);
			
			if (i%this->settings.i_number_allowed_same == 0){
				this->createCheckpoint();
			}

			this->mainNetwork->ApplyError();
			if (i%this->settings.i_number_allowed_same == 0){
				this->createCheckpoint();
			}
			
			//Apply the error
			
		
			if (i%this->settings.i_number_in_sequence == 0 && i!=0){//Reset the sequence once the sequence has finished
				
				this->mainNetwork->ResetSequence();
			}
			this->checkpoint.i_number_of_loops_checkpoint += 1;
			
		}
		try{
			this->createCheckpoint();
			/*this->mainNetwork->ResetSequence();
			for (int i = 0; i < this->settings.i_number_in_sequence; i++){
				cout << i << ") " << endl;
				thrust::device_vector<weight_type> temp = this->mainNetwork->runNetwork(this->input[i]);
				testing::outputToFile<weight_type>(temp, "results", "tests/results.txt");
			}*/
			this->createCheckpoint("RunResultsInMemory");
			this->mainNetwork->cleanNetwork();
			this->mainNetwork->InitializeRun();
			this->mainNetwork->ResetSequence();
			this->createCheckpoint("RunStart");
			for (int i = 0; i < this->settings.i_number_in_sequence; i++){
				if (i == 0){
					testing::outputArrayToFile<weight_type>(this->input[i], this->settings.i_input, "tests/results2.txt");
					testing::outputArrayToFile<weight_type>(this->output[i], this->settings.i_output, "tests/results2.txt");
				}
				testing::outputArrayToFile<weight_type>(this->input[i], this->settings.i_input, "tests/results2.txt");
				testing::outputArrayToFile<weight_type>(this->output[i], this->settings.i_output, "tests/results2.txt");
				std::vector<weight_type> temp2 = this->runNetwork(this->input[i]);
				testing::outputVectorToFile<weight_type>(temp2, "results", "tests/results2.txt");
			}
			this->createCheckpoint("RunResultsFromHost");
			this->mainNetwork->emptyGPUMemory();
		}
		catch (exception e){
			cout << e.what();
			cin.sync();
			cin.get();
			//this->mainNetwork->emptyGPUMemory();
		}

		
	}
	catch (exception e){//Edit to write the problems to file later
		cout << e.what();
		cin.sync();
		cin.get();
	}
}
#endif

bool ReccurentLoops::train_network_HessianFreeOptimizationTraining(){
	this->mainNetwork->InitializeTraining();
	do{
		this->mainNetwork->StartTraining(this->input[this->checkpoint.i_number_of_loops_checkpoint], this->output[this->checkpoint.i_number_of_loops_checkpoint]);
		if (this->checkpoint.i_number_of_loops % this->settings.i_loops == 0){
			this->mainNetwork->VisualizeNetwork();
			this->mainNetwork->ApplyError();//Apply the error gained from the last steps
			this->mainNetwork->CopyToHost();
			this->mainNetwork->VisualizeNetwork();
			this->createCheckpoint();
			this->mainNetwork->ResetSequence();
		}

		this->checkpoint.i_number_of_loops_checkpoint++;
		this->checkpoint.i_number_of_loops++;
	} while (checkpoint.i_number_of_loops_checkpoint < this->settings.i_number_of_training);
	this->mainNetwork->cleanNetwork();
	return true;
}

bool ReccurentLoops::train_network_RealTimeRecurrentTraining(){
	return true;
}

//*********************
//DEBUG FUNCTIONS
//*********************

//Creates a test input/output
weight_type* ReccurentLoops::createTestInputOutput(int numberOfInput, int input_output){
	static int position = 0;
	weight_type* temp = new weight_type[numberOfInput];
	weight_type count = .1;
	for (int i = position; i < position + numberOfInput; i++){

		if (input_output == 0){
			temp[i - position] = (weight_type)(i%this->settings.i_number_in_sequence) + 1;
		}
		else{
			temp[i - position] = (weight_type)(.1*(i%this->settings.i_number_in_sequence)) + .1 + count;
		}
		count += .1;
	}
	position += numberOfInput;
	return  temp;
}

//Create a checkpoint with the network name
//Default function
void ReccurentLoops::createCheckpoint(){
	this->createCheckpoint(this->settings.s_network_name);
}

//Create a Checkpoint with any name
void ReccurentLoops::createCheckpoint(string file_name){
	static int count = 0 ;

		std::ofstream outputfile;
		outputfile.open("recurrent_networks/" + file_name + std::to_string(count) + ".txt", ios::trunc);
		if (outputfile.is_open()){
			outputfile << *this << flush;
			outputfile << endl;
			outputfile.close();
		}
		else{
			std::cout << "Unable to write checkpoint to file." << endl;
			std::cout << "continue?";
		}
		
		count++;
		

	

}