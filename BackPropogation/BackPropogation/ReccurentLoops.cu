#include "ReccurentLoops.cuh"
//#define TESTING
#ifndef SEQUENCE_DELIMITER
#define SEQUENCE_DELIMITER 4
#endif
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
	this->mainNetwork = new LongTermShortTermNetwork(settings, true);
	this->loadCheckpoint();
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
	this->checkpoint = CRecurrentCheckpoint(settings);
}

ReccurentLoops::ReccurentLoops(CSettings settings, CRecurrentCheckpoint checkpoint){
	this->settings = settings;
	this->checkpoint = checkpoint;
	this->mainNetwork = new LongTermShortTermNetwork(settings,true);
	this->InitializeNetwork();
}

//*****************************
//Intialzie the Internal Requirements For Training
//*****************************

void ReccurentLoops::InitializeNetwork(){
	this->input = new weight_type*[this->settings.i_number_of_training];
	this->output = new weight_type*[this->settings.i_number_of_training];
	this->inputfile = new std::fstream();
	this->outputfile = new std::fstream();
	this->inputfile->open(this->settings.s_trainingSet);
	this->outputfile->open(this->settings.s_outputTrainingFile);
	this->LoadTrainingSet();
}



//*****************************
//Reload a Network From a File
//*****************************
bool ReccurentLoops::loadNetworkFromFile(){
	
	return true;
}

void ReccurentLoops::loadCheckpoint(){
	std::ifstream is;
	is.open(this->settings.s_checkpoint_file);
	is >> this->checkpoint;
	
	this->mainNetwork->LoadNetwork(is);
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
	device_vector<weight_type> temp_device;
	if (this->checkpoint.b_still_running){
		temp_device = this->mainNetwork->runNetwork(in, NetworkBase::run_type::WITH_MEMORY_CELLS);
	}
	else{
		temp_device = this->mainNetwork->runNetwork(in,NetworkBase::run_type::WITHOUT_MEMORY_CELLS);
	}
	
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
#ifdef TESTING 
	for (int i = 0; i < this->settings.i_number_of_training; i++){
		this->input[i] = this->createTestInputOutput(this->settings.i_input,0);

		testing::outputArrayToFile(this->input[i], this->settings.i_input, "tests/inout.txt");

	}
#endif
#ifdef TESTING 
	for (int i = 0; i < this->settings.i_number_of_training; i++){
		this->output[i] = this->createTestInputOutput(this->settings.i_output, 1);

		testing::outputArrayToFile(this->output[i], this->settings.i_output, "tests/inout.txt");

	}
#endif
	return true;
}

//file - the file to read from
//length_of_results - maximum length of the array
//storage - array to contain the results
//sequence_length - [0] = length of the current sequence, [1] - if the sequence is longer than storage returns 0, else returns 1
//type - the type of data which should be retrieved
void ReccurentLoops::loadFromFile(std::fstream &file, int length_of_results, double** storage, int sequence_length[2], data_type type){

	//reset sequence length
	sequence_length[0] = 0;

	char individual_delimiter = (char)37;
	char group_delimiter = (char)30;
	char sequence_delimiter = (char)SEQUENCE_DELIMITER;
	char current_char = '1';
	string current_string = "";
	int start = file.tellg();
	if (file.is_open()){
		int letterPosition = 0;
		for (int i = 0; i < this->settings.i_number_of_training;i++){
			//Reset Everything
			storage[i] = new weight_type[length_of_results];
			current_char = '1';
			current_string = "";
			letterPosition = 0;

			//While not at the end of a group, retrieve the current dataset
			while (current_char != group_delimiter && !file.eof() && current_char!=sequence_delimiter){
				//Get the current char
				current_char = file.get();

				if (current_char == individual_delimiter){//Reached the end of the current set
					if (type == OUTPUT){
						storage[i][letterPosition] = stod(current_string);
					}
					else if(type==INPUT){
						storage[i][letterPosition] = (weight_type)current_string.at(0);
					}
					letterPosition++;
					current_string = "";
				}
				else{//String has not ended and is still correct
					current_string += current_char;
				}
			}
			//Increment Length Of String
			sequence_length[0]++;
			if (current_char == sequence_delimiter){
				i++;//Increment i and store a special sequence which is used to represent a new sequence
				storage[i] = new weight_type[length_of_results];
				for (int j = 0; j < length_of_results; j++){
					storage[i][j] = SEQUENCE_DELIMITER;
				}
			}
		}
		//Get current location in file
		int currentPosition = file.tellg();
		if (file.eof()){
			sequence_length[1] = -1;//File has ended
		}
		else if (current_char == sequence_delimiter){
			sequence_length[1] = 0;//The sequence has ended
		}
		else{
			sequence_length[1] = 1;//There is more to the sequence
		}
	}
	else{
		std::cout << "Unable to read from file." << endl;
		std::cout << "continue?";
		if (cin.get() == 'n'){
			exit(0);
		}
	}
}

void ReccurentLoops::LoadTrainingSet(){
	this->training_input = new weight_type*[this->settings.i_number_of_training];
	this->training_output = new weight_type*[this->settings.i_number_of_training];
	int training_length[2];
	std::fstream stream;
	if (this->settings.b_testingFromFile){//A training file has been included and should be read from for the training set
		stream.open(this->settings.s_testSet);
		this->loadFromFile(stream, this->settings.i_number_of_training, this->training_input, training_length,INPUT);
		stream.close();
		stream.open(this->settings.s_outputTestSet);
		this->loadFromFile(stream, this->settings.i_number_of_training, this->training_output, training_length, OUTPUT);
		stream.close();
		this->number_in_training_sequence = training_length[0];
	}
	else{//A training file has not been included, get a random set from the input file
		stream.open(this->settings.s_trainingSet);
		this->loadFromFile(stream, this->settings.i_number_of_training, this->training_input, training_length, INPUT);
		stream.close();
		stream.open(this->settings.s_outputTrainingFile);
		this->loadFromFile(stream, this->settings.i_number_of_training, this->training_output, training_length, OUTPUT);
		stream.close();
		this->number_in_training_sequence = training_length[0];
	}
}

//**********************
//Training
//**********************
void ReccurentLoops::startTraining(int type){
	//Load the data from a file
	//if (!load_training_data_from_file()){
		//throw exception("Unable to read from file.");
	//}
	this->inputfile->open(this->settings.s_outputTrainingFile);
	this->outputfile->open(this->settings.s_outputTestSet);
	if (this->inputfile->is_open() && this->outputfile->is_open()){
		switch (type){
		case ReccurentLoops::RealTimeTraining:
			//Train the network using real time recurrent
			this->train_network_RealTimeRecurrentTraining();
			break;
		case ReccurentLoops::HessianFreeOptimization:
			this->train_network_HessianFreeOptimizationTraining();
			break;
		case ReccurentLoops::LongTermShortTerm:
			this->testTraining();
			break;
		}
	}
	else{
		std::cout << "Unable to read from file." << endl;
		std::cout << "continue?";
		if (cin.get() == 'n'){
			exit(0);
		}
	}
}


void ReccurentLoops::testTraining(){
	weight_type** trainingInput = new weight_type*[this->settings.i_backprop_unrolled];
	weight_type** trainingOutput = new weight_type*[this->settings.i_backprop_unrolled];
	int length[2];
	bool sequence_end = false;//Tell if the sequence ends
	
	try{
		this->load_training_data_from_file();
		if (!this->checkpoint.b_still_running){
			this->mainNetwork->InitializeTraining();
		}
		this->checkpoint.b_still_running = true;
		this->createCheckpoint("Initial Checkpoint");
		while (length[1] != -1){

			this->loadFromFile(*(this->outputfile), this->settings.i_output, this->output, length, OUTPUT);
			this->loadFromFile(*(this->inputfile), this->settings.i_input, this->input, length, INPUT);
			//testing::outputArrayToFile(this->input, length[0], this->settings.i_input, "tests/input_output.txt");
			//testing::outputArrayToFile(this->output, length[0], this->settings.i_output, "tests/input_output.txt");
			for (int i = 0; i < this->settings.i_number_of_training; i += this->settings.i_backprop_unrolled){

				for (int j = i, k = 0; k < this->settings.i_backprop_unrolled; j++, k++){
					if (!sequence_end && (this->input[j][0] != SEQUENCE_DELIMITER || this->output[j][0] != SEQUENCE_DELIMITER)){//If both are a sequence_delimiter, then the sequence has ended
						trainingInput[k] = this->input[j];
						trainingOutput[k] = this->output[j];
					}
					else{
						sequence_end = true;
						if (k < this->settings.i_backprop_unrolled){
							trainingInput[k] = this->input[0];
							trainingOutput[k] = this->output[0];
						}
						//Reached the end of the sequence, load the next sequence
						//Normally load more from the file
						//break;
					}
				}
				this->mainNetwork->StartTraining(trainingInput, trainingOutput);

				if (this->checkpoint.i_number_of_loops_checkpoint%this->settings.i_number_allowed_same == 0){
					this->createCheckpoint();
				}
				//Apply the error
				this->mainNetwork->ApplyError();
				if (this->checkpoint.i_number_of_loops_checkpoint%this->settings.i_number_allowed_same == 0){
					this->createCheckpoint();
				}




				
				this->checkpoint.i_number_of_loops_checkpoint += 1;


				if (sequence_end){
					//The sequence has ended, so we need to reset the sequence
					this->mainNetwork->ResetSequence();
					sequence_end = false;
				}

			}
			if (length[1] == 0){//Reset the sequence once the sequence has finished

				this->mainNetwork->ResetSequence();
			}
			//Load more data from the file
			
			
		}
		//No longer running loops
		
		try{
			this->createCheckpoint();
			this->mainNetwork->ResetSequence();
			for (int i = 0; i < this->settings.i_number_of_training; i++){
				if (i == 0){
					testing::outputArrayToFile<weight_type>(this->training_input[i], this->settings.i_input, "tests/results.txt");
					testing::outputArrayToFile<weight_type>(this->training_output[i], this->settings.i_output, "tests/results.txt");
				}
				if (this->training_input[i][0] != SEQUENCE_DELIMITER || this->training_output[i][0] != SEQUENCE_DELIMITER){
					testing::outputArrayToFile<weight_type>(this->training_input[i], this->settings.i_input, "tests/results.txt");
					testing::outputArrayToFile<weight_type>(this->training_output[i], this->settings.i_output, "tests/results.txt");
					thrust::device_vector<weight_type> temp = this->mainNetwork->runNetwork(this->training_input[i]);
					testing::outputToFile<weight_type>(temp, "results", "tests/results.txt");
				}
				else{
					this->mainNetwork->ResetSequence();
				}
			}
			this->checkpoint.b_still_running = false;
			this->createCheckpoint("RunResultsInMemory");
			this->mainNetwork->cleanNetwork();
			this->mainNetwork->InitializeRun();
			this->mainNetwork->ResetSequence();
			this->createCheckpoint("RunStart");
			for (int i = 0; i < this->settings.i_number_of_training; i++){
				if (i == 0){
					testing::outputArrayToFile<weight_type>(this->training_input[i], this->settings.i_input, "tests/results2.txt");
					testing::outputArrayToFile<weight_type>(this->training_output[i], this->settings.i_output, "tests/results2.txt");
				}
				if (this->training_input[i][0] != SEQUENCE_DELIMITER || this->training_output[i][0] != SEQUENCE_DELIMITER){
					testing::outputArrayToFile<weight_type>(this->training_input[i], this->settings.i_input, "tests/results2.txt");
					testing::outputArrayToFile<weight_type>(this->training_output[i], this->settings.i_output, "tests/results2.txt");
					std::vector<weight_type> temp2 = this->runNetwork(this->training_input[i]);
					testing::outputVectorToFile<weight_type>(temp2, "results", "tests/results2.txt");
				}
				else{
					this->mainNetwork->ResetSequence();
				}
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
	static int previous = input_output;
	if (input_output != previous){
		position = 0;
		previous = input_output;
	}
	if (input_output == 0 && position >= this->settings.i_number_in_sequence){
		position = 0;
	}
	else if (position >= this->settings.i_number_in_sequence){
		position = 0;
	}
	weight_type* temp = new weight_type[numberOfInput];
	weight_type count = .1;
	for (int i = 0; i < numberOfInput; i++){

		if (input_output == 0){
			temp[i] = (weight_type)(position);
		}
		else{
			temp[i] = (weight_type)(.01*(position));
		}
		count += .1;
		position++;
	}
	
	return  temp;
}

void ReccurentLoops::cleanLoops(){
	for (int i = 0; i < this->settings.i_number_of_training; i++){
		std::free(this->input[i]);
		std::free(this->output[i]);
		std::free(this->training_input[i]);
		std::free(this->training_output[i]);
	}
	std::free(this->input);
	std::free(this->output);
	std::free(this->training_input);
	std::free(this->training_output);
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