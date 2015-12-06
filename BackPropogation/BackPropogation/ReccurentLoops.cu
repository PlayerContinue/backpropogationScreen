#include "ReccurentLoops.cuh"
//#define TESTING
#ifndef SEQUENCE_DELIMITER
#define SEQUENCE_DELIMITER 4
#endif
using namespace boost::interprocess;
//*****************************
//Constructor
//*****************************
ReccurentLoops::ReccurentLoops()
{

}

/*ReccurentLoops::~ReccurentLoops(){
	this->cleanLoops();
	delete this->input;
	delete this->output;
	delete this->training_input;
	delete this->training_output;
	delete this->mainNetwork;
	}*/

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
	this->mainNetwork = new LongTermShortTermNetwork(settings, true);
	this->InitializeNetwork();
}

//*****************************
//Intialzie the Internal Requirements For Training
//*****************************

void ReccurentLoops::InitializeNetwork(){
	this->timer = NetworkTimer(this->settings.i_number_minutes_to_checkpoint);
	this->input = new weight_type*[this->settings.i_number_of_training];
	this->length_of_arrays[OUTPUT] = 0;
	this->output = new weight_type*[this->settings.i_number_of_training];
	this->length_of_arrays[INPUT] = 0;
	this->mean_square_error_results_new = host_vector<weight_type>(this->settings.i_output + 1);
	this->mean_square_error_results_old = host_vector<weight_type>(this->settings.i_output + 1);
	this->inputfile = new std::fstream();
	this->outputfile = new std::fstream();
	this->inputfile->open(this->settings.s_trainingSet);
	this->outputfile->open(this->settings.s_outputTrainingFile);
	try{
		this->LoadTrainingSet();
	}
	catch (exception e){
		cout << "error" << endl;
		cout << e.what();
		cin.sync();
		cin.get();
		std::exit(0);
	}
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

	device_vector<weight_type> temp_device = this->runTrainingNetwork(in);
	vector<RETURN_WEIGHT_TYPE> to_return = vector<RETURN_WEIGHT_TYPE>(temp_device.size());


	for (unsigned int i = 0; i < temp_device.size(); i++){
		to_return[i] = temp_device[i];
	}
	clear_vector::free(temp_device);

	return to_return;
}

void ReccurentLoops::runContinousNetwork(weight_type* in, std::string save_location, weight_type* end_value){
	this->runContinousNetwork(in, save_location, end_value, 1000);
}
void ReccurentLoops::runContinousNetwork(weight_type* in, std::string save_location, weight_type* end_value, int max_sequence_length){
	//Open a File for writing the output
	ofstream results;
	results.open(save_location, ios::trunc);//Open the location
	if (!results.is_open()){
		//The file could not be opened. Throw an exception for usage
		throw new exception("Save Location Could Not Be Opened");
	}

	//Run with initially given input	
	device_vector<weight_type> input;
	device_vector<weight_type> output;

	if (this->checkpoint.b_still_running){
		input = this->mainNetwork->runNetwork(in, NetworkBase::run_type::WITH_MEMORY_CELLS);
	}
	else{
		input = this->mainNetwork->runNetwork(in, NetworkBase::run_type::WITHOUT_MEMORY_CELLS);
	}


	max_sequence_length -= 1;//Remove one from the sequence
	if (this->settings.i_input == this->settings.i_output){//Only allow recurrence runs with equal length input/output
		//Transfer the end_value to the gpu
		output = thrust::device_vector<weight_type>();
		for (int i = 0; i < this->settings.i_output; i++){
			output.push_back(end_value[i]);
		}
		//Run with gathered output
		for (int i = 0; i < max_sequence_length
			/*&& thrust::mismatch(output.begin(), output.end(), input.begin()) != thrust::pair<device_vector<weight_type>::iterator, device_vector<weight_type>::iterator>(output.end(), input.end())*/; i++){
			if (this->checkpoint.b_still_running){
				input = this->mainNetwork->runNetwork(input, NetworkBase::run_type::WITH_MEMORY_CELLS);
			}
			else{
				input = this->mainNetwork->runNetwork(input, NetworkBase::run_type::WITHOUT_MEMORY_CELLS);
			}
			thrust::copy(input.begin(), input.end(), std::ostream_iterator<weight_type>(results, ","));
			results << endl;
		}

	}



}

//Run the network during training
device_vector<weight_type> ReccurentLoops::runTrainingNetwork(weight_type* in){
	device_vector<weight_type> temp_device;
	if (this->checkpoint.b_still_running){
		temp_device = this->mainNetwork->runNetwork(in, NetworkBase::run_type::WITH_MEMORY_CELLS);
	}
	else{
		temp_device = this->mainNetwork->runNetwork(in, NetworkBase::run_type::WITHOUT_MEMORY_CELLS);
	}

	return temp_device;

}
//*****************************
//Get Data From the users file
//*****************************

//file - the file to read from
//length_of_results - maximum length of the array
//storage - array to contain the results
//sequence_length - [0] = length of the current sequence, [1] - if the sequence is longer than storage returns 0, else returns 1
//type - the type of data which should be retrieved
void ReccurentLoops::loadFromFile(std::fstream &file, int length_of_results, double** storage, int sequence_length[2], data_type type, int max_allowed, bool first_run){
	this->loadFromFile(file, length_of_results, storage, sequence_length, this->settings.i_number_of_training, type, max_allowed, first_run);
}

void ReccurentLoops::loadFromFile(std::fstream &file, int length_of_results, double** storage, int sequence_length[2], int length, data_type type, int max_allowed, bool first_run){

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
		for (int i = 0; i < length && !file.eof(); i++){
			//Reset Everything
			if (first_run){
				storage[i] = new weight_type[length_of_results];
			}
			current_char = '1';
			current_string = "";
			letterPosition = 0;

			//While not at the end of a group, retrieve the current dataset
			while (current_char != group_delimiter && !file.eof() && current_char != sequence_delimiter){
				//Get the current char
				current_char = file.get();
				if (letterPosition >= max_allowed && current_char != group_delimiter && current_char != sequence_delimiter){
					throw new exception("Too Many Values In A Group");
				}
				if (current_char == individual_delimiter){//Reached the end of the current set
					if (type == OUTPUT){
						storage[i][letterPosition] = stod(current_string);
					}
					else if (type == INPUT){
						storage[i][letterPosition] = (weight_type)stod(current_string);
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
			if (current_char == sequence_delimiter || file.eof()){
				for (int j = 0; j < length_of_results; j++){
					storage[i][j] = SEQUENCE_DELIMITER;
				}
			}


		}
		//Get current location in file
		int currentPosition = file.tellg();
		if (file.eof()){
			if (sequence_length[0] < length){
				if (first_run){
					//storage[sequence_length[0]] = new weight_type[length_of_results];
				}
				for (int j = 0; j < length_of_results; j++){
					//storage[sequence_length[0]][j] = SEQUENCE_DELIMITER;
				}
			}
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
	}
	else{//A training file has not been included, get a random set from the input file
		stream.open(this->settings.s_trainingSet);
	}

	if (stream.is_open()){
		this->loadFromFile(stream, this->settings.i_input, this->training_input, training_length, this->settings.i_number_of_testing_items, INPUT, this->settings.i_input, true);
		stream.close();
		this->length_of_arrays[TRAINING_1] = training_length[0];

		if (this->settings.b_testingFromFile){//A training file has been included and should be read from for the training set
			stream.open(this->settings.s_outputTestSet);
		}
		else{//A training file has not been included, get a random set from the input file
			stream.open(this->settings.s_outputTrainingFile);
		}

		if (stream.is_open()){
			this->loadFromFile(stream, this->settings.i_output, this->training_output, training_length, this->settings.i_number_of_testing_items, OUTPUT, this->settings.i_output, true);
			stream.close();
			this->number_in_training_sequence = training_length[0];
			this->length_of_arrays[TRAINING_2] = training_length[0];
		}
		else{
			throw new exception("Output File Not Found");
		}
	}
	else{
		throw new exception("Input File Not Found");
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

void ReccurentLoops::reset_file_for_loop(){
	if ((int)this->inputfile->tellg() == 0){
		this->inputfile->clear();
		this->inputfile->seekg(0, this->inputfile->end);
		this->length_of_file = this->inputfile->tellg();
		//Set the timer information about the file size
		this->timer.set_file_size(this->length_of_file * (std::istream::streampos) this->settings.i_numberTimesThroughFile);
	}

	this->inputfile->clear();
	this->inputfile->seekg(0, ios::beg);
	this->outputfile->clear();
	this->outputfile->seekg(0, ios::beg);

	int temp1 = this->outputfile->tellg();
	int temp2 = this->inputfile->tellg();

	this->mainNetwork->ResetSequence();
}

void ReccurentLoops::sequenceEnd(int &length_of_sequence, int &count_sequences, int &growth_check){
	if (length_of_sequence > 0){
		this->mainNetwork->seti_backprop_unrolled(length_of_sequence);
		//Apply the error at the end of the sequence
		this->mainNetwork->ApplyError();
	
		length_of_sequence = 0;
	}
	//The sequence has ended, so we need to reset the sequence
	this->mainNetwork->ResetSequence();
	if (count_sequences >= this->settings.i_loops){
		//Copy the previous set of error to the new set of errors
		std::copy(this->mean_square_error_results_new.begin(), this->mean_square_error_results_new.end(), this->mean_square_error_results_old.begin());

		//Get the mean Square error
		this->getMeanSquareError();

		//Check to see if the change is great enough
		growth_check = 0;
		for (int mean_pos = 0; mean_pos < this->mean_square_error_results_new.size(); mean_pos++){
			if (this->mean_square_error_results_old[mean_pos] < this->mean_square_error_results_new[mean_pos] - this->settings.d_fluctuate_square_mean){
				growth_check++;
			}
		}

		if (this->settings.b_allow_growth && this->mean_square_error_results_old[0] < this->mean_square_error_results_new[0] - this->settings.d_fluctuate_square_mean && growth_check >= (this->mean_square_error_results_new.size() / 2) + 1){

			this->mainNetwork->addNeuron(3);
			//Set a new old mean square error so it will attempt to learn before gaining a new node
			this->getMeanSquareError();
			std::copy(this->mean_square_error_results_new.begin(), this->mean_square_error_results_new.end(), this->mean_square_error_results_old.begin());
		}
		int temp[1];
		this->mainNetwork->getInfoAboutNetwork(temp);
		testing::outputToFile<weight_type>(this->mean_square_error_results_new, "new", "tests/meansquare.txt");
		testing::outputArrayToFile<int>(temp, 1, "tests/meansquare.txt");
		this->mainNetwork->ResetSequence();
		count_sequences = 0;
	}
	count_sequences++;
}

void ReccurentLoops::testTraining(){

	weight_type** trainingInput = new weight_type*[this->settings.i_backprop_unrolled];

	weight_type** trainingOutput = new weight_type*[this->settings.i_backprop_unrolled + 1];
	int length[2];
	bool sequence_end = false;//Tell if the sequence ends
	int count_sequences = 0;
	int k = 0;
	bool first_run = true;
	int length_of_sequence = 0;
	int growth_check = 0;
	int output_length;
	int output_stop;
	this->mean_square_error_results_new[0] = this->settings.d_threshold + 1;
	this->stop_training_thread();//Remove any threads which may have been missed and remove all open shared_memory_locations
	managed_shared_memory managed{ create_only, TIMER_SHARED, 1024 };
	try{
		if (!this->checkpoint.b_still_running){
			this->mainNetwork->InitializeTraining();
		}

		this->checkpoint.b_still_running = true;
		//this->createCheckpoint("Initial Checkpoint");
		if (this->settings.b_allow_growth && this->settings.b_allow_node_locking){
			//this->mainNetwork->addWeight(1);
			this->mainNetwork->removeWeight();
			this->createCheckpoint("Remove_Checkpoint_1");
			this->mainNetwork->cleanNetwork();
			exit(0);
		}

		cout << "Training Start" << endl;

		for (int loops = 0; loops < this->settings.i_numberTimesThroughFile; loops++){
			reset_file_for_loop();
			if (this->checkpoint.i_current_position_in_input_file > 0 && this->checkpoint.i_current_position_in_output_file > 0 && length[1] != -1){
				this->inputfile->seekg(this->checkpoint.i_current_position_in_input_file);
				this->outputfile->seekg(this->checkpoint.i_current_position_in_output_file);
			}
			this->getMeanSquareError();
			testing::outputToFile<weight_type>(this->mean_square_error_results_new, "new", "tests/meansquare.txt");

			this->mainNetwork->ResetSequence();
			length[1] = 0;
			this->timer.start();

			while (length[1] != -1 && this->mean_square_error_results_new[0] > this->settings.d_threshold){
				if (length[1] != 0){
					//Find the estimated time remaining from the length 
					//The length of file * loops is number of previously found values
					//this->timer.restart_timer();
					//cout << this->timer.estimated_time_remaining(this->inputfile->tellg() + (this->length_of_file*loops)) << endl;
					std::memset(managed.find<std::istream::streampos>(TIMER_PRINT_VALUE).first, this->inputfile->tellg() + (this->length_of_file*loops), 
						managed.find<std::istream::streampos>(TIMER_PRINT_VALUE).second);
					std::memset(managed.find<bool>(TIMER_PRINT).first, (bool)1, managed.find<bool>(TIMER_PRINT).second);
				}
				this->timer.clear_timer();


				this->loadFromFile(*(this->outputfile), this->settings.i_output, this->output, length, OUTPUT, this->settings.i_output, first_run);
				
				output_length = length[0];
				output_stop = length[1];
				if (this->length_of_arrays[OUTPUT] < length[0]){
					this->length_of_arrays[OUTPUT] = length[0];
				}
				this->loadFromFile(*(this->inputfile), this->settings.i_input, this->input, length, INPUT, this->settings.i_input, first_run);

				if (length[0] > output_length){//The length should be the value of the shortest one
					length[0] = output_length;
				}
				if (length[0] > this->length_of_arrays[INPUT]){
					this->length_of_arrays[INPUT] = length[0];
				}

				if (first_run){
					
					if (this->checkpoint.i_current_position_in_input_file > 0 && this->checkpoint.i_current_position_in_output_file > 0){
						this->timer.set_size_of_round(this->inputfile->tellg() - (std::ios_base::streampos)this->checkpoint.i_current_position_in_input_file);//Set it to the position in the file after a single round
					}
					else{
						this->timer.set_size_of_round(this->inputfile->tellg());//Set it to the position in the file after a single round
						
					}
					this->initialize_threads();

					
				}
				this->checkpoint.i_current_position_in_input_file = this->inputfile->tellg();//Done afterwards for the purpose of setting the difference
				this->checkpoint.i_current_position_in_output_file = this->outputfile->tellg();
				if (length[1] == -1 || output_stop == -1){//Sequence may break early
					break;
				}
				first_run = false;

				for (int i = 0; i < length[0] && this->mean_square_error_results_new[0] > this->settings.d_threshold;){

					for (; k < this->settings.i_backprop_unrolled; k++){
						if (i < length[0] && (this->input[i][0] == SEQUENCE_DELIMITER || this->output[i][0] == SEQUENCE_DELIMITER) && this->output[i][0] != this->input[i][0]){//Checks for sequence input and output mistmatch
							cout << "Sequence End Mismatch. The input or output do not both have the same end sequence" << endl;
							cout << "Countinue? ";
							cin.sync();
							if (cin.get() == 'n'){
								this->cleanLoops();
								exit(0);
							}
							else{
								cout << endl;
							}

						}
						if (!sequence_end && i < length[0] && (this->input[i][0] != SEQUENCE_DELIMITER || this->output[i][0] != SEQUENCE_DELIMITER)){//If both are a sequence_delimiter, then the sequence has ended
							trainingInput[k] = this->input[i];
							if (k == 0){
								trainingOutput[k] = this->output[i];
							}
							trainingOutput[k + 1] = this->output[i];
							i++;//Increment i here because the next sequence follows
						}
						else{
							if (k != 0){//If the sequence ended right away, don't do a backprop
								//Since the sequence ended, but we have not reached the end of the backdrop, we need to add an extra layer
								trainingInput[k] = this->input[i - 1];
								trainingOutput[k + 1] = this->output[i - 1];
								k++;
							}
							if (!sequence_end){
								i++;//Skip passed the end of the sequence
							}

							sequence_end = true;

							break;
						}
					}

					//Set the i_backpropunrolled of the mainNetworks settings so it only applys the information on the current sequence length
					//Allows for multilength sequences
					this->mainNetwork->seti_backprop_unrolled(k);

					if (k > 0){
						length_of_sequence += k;
						//Run the sequence to find the results

						this->mainNetwork->StartTraining(trainingInput, trainingOutput);

						//this->checkpoint.i_number_of_loops_checkpoint += 1;

						


					}


					if (sequence_end){
						//Perform the sequence end functions
						sequenceEnd(length_of_sequence, count_sequences, growth_check);

						sequence_end = false;
						bool* temp = managed.find<bool>(CHECKPOINT_TIMER).first;
						if (*(managed.find<bool>(CHECKPOINT_TIMER).first)==true){//Create a checkpoint if the set checkpoint values has been passed
							this->createCheckpoint();
							std::memset(managed.find<bool>(CHECKPOINT_TIMER).first, false, managed.find<bool>(CHECKPOINT_TIMER).second);
						}

						//A new sequence, start from the beginning
						k = 0;
					}
					else{
						//The input of the initial is the beginning of it.
						trainingInput[0] = trainingInput[k - 1];
						trainingOutput[0] = trainingOutput[k];
						//Continue the sequence, we need to keep the previous input
						k = 1;
					}
				}
				if (length[1] == 0){//Reset the sequence once the sequence has finished
					this->mainNetwork->ResetSequence();

				}



			}
		}
		//No longer running loops
		this->mainNetwork->ResetSequence();
		//this->mainNetwork->seti_backprop_unrolled(this->settings.i_backprop_unrolled - 2);
		//this->mainNetwork->StartTraining(this->input, this->output);
		this->createCheckpoint("Last_Train_Check");
		try{



			//Copy the previous set of error to the new set of errors
			std::copy(this->mean_square_error_results_new.begin(), this->mean_square_error_results_new.end(), this->mean_square_error_results_old.begin());
			//Get the mean Square error
			this->getMeanSquareError();
			testing::outputToFile<weight_type>(this->mean_square_error_results_new, "new", "tests/meansquare.txt");
			this->mainNetwork->seti_backprop_unrolled(0);
			this->mainNetwork->ResetSequence();
			this->createCheckpoint();
			this->mainNetwork->ResetSequence();
			for (int i = 0; i < this->number_in_training_sequence; i++){
				if (i == 0){
					testing::outputArrayToFile<weight_type>(this->training_input[i], this->settings.i_input, "tests/results.txt");
					testing::outputArrayToFile<weight_type>(this->training_output[i], this->settings.i_output, "tests/results.txt");
				}
				if (this->training_input[i][0] != SEQUENCE_DELIMITER || this->training_output[i][0] != SEQUENCE_DELIMITER){
					testing::outputArrayToFile<weight_type>(this->training_input[i], this->settings.i_input, "tests/results.txt");
					testing::outputArrayToFile<weight_type>(this->training_output[i], this->settings.i_output, "tests/results.txt");
					thrust::device_vector<weight_type> temp = this->mainNetwork->runNetwork(this->training_input[i], NetworkBase::run_type::WITH_MEMORY_CELLS);
					testing::outputToFile<weight_type>(temp, "results", "tests/results.txt");
				}
				else{
					this->mainNetwork->ResetSequence();
				}
			}
			this->checkpoint.b_still_running = false;
			this->createCheckpoint("RunResultsInMemory");
			this->mainNetwork->ResetSequence();
			this->mainNetwork->cleanNetwork();
			this->mainNetwork->InitializeRun();
			this->createCheckpoint("RunStart");
			for (int i = 0; i < this->number_in_training_sequence; i++){
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

			this->cleanLoops();
		}
		catch (exception e){
			cout << "error" << endl;
			cout << e.what();
			cin.sync();
			cin.get();
			//this->mainNetwork->emptyGPUMemory();
		}


	}

	catch (exception e){//Edit to write the problems to file later
		cout << "error" << endl;
		cout << e.what();
		cin.sync();
		cin.get();
	}
}

void ReccurentLoops::getMeanSquareError(){
	thrust::device_vector<weight_type> vec;
	thrust::device_vector<weight_type> real_output = thrust::device_vector<weight_type>(this->settings.i_output);
	for (int i = 0; i < this->number_in_training_sequence; i++){
		if (this->training_input[i][0] == SEQUENCE_DELIMITER && this->training_output[i][0] == SEQUENCE_DELIMITER){

			this->mainNetwork->ResetSequence();

		}
		else{
			vec = this->runTrainingNetwork(this->training_input[i]);
			for (int j = 0; j < this->settings.i_output; j++){
				real_output[j] = this->training_output[i][j];
			}

			if (i != 0){
				value_testing::getMeanSquareErrorSum(vec.begin(), vec.end(), real_output.begin(), real_output.end(), this->mean_square_error_results_new);
			}
			else{
				value_testing::getMeanSquareError(vec.begin(), vec.end(), real_output.begin(), real_output.end(), this->mean_square_error_results_new);
			}
		}
	}

	//Divide the summed value
	for (int i = 0; i < this->mean_square_error_results_new.size(); i++){
		this->mean_square_error_results_new[i] /= this->number_in_training_sequence;
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
	this->mainNetwork->cleanNetwork();
	this->stop_training_thread();
	this->checkpoint.i_current_position_in_input_file = 0;
	this->checkpoint.i_current_position_in_output_file = 0;
	this->checkpoint.b_still_running = false;
	this->inputfile->close();
	this->outputfile->close();
	for (int i = 0; i < this->length_of_arrays[INPUT]; i++){
		delete[] this->input[i];
	}
	for (int i = 0; i < this->length_of_arrays[OUTPUT]; i++){
		delete[] this->output[i];
	}

	for (int i = 0; i < this->length_of_arrays[TRAINING_1]; i++){
		delete[] this->training_input[i];
	}

	for (int i = 0; i < this->length_of_arrays[TRAINING_2]; i++){
		delete[] this->training_output[i];
	}

	this->length_of_arrays[INPUT] = 0;
	this->length_of_arrays[OUTPUT] = 0;
	this->length_of_arrays[TRAINING_1] = 0;
	this->length_of_arrays[TRAINING_2] = 0;

}

//Create a checkpoint with the network name
//Default function
void ReccurentLoops::createCheckpoint(){
	this->createCheckpoint(this->settings.s_network_name);
}

//Create a Checkpoint with any name
void ReccurentLoops::createCheckpoint(string file_name){
	static int count = 0;
	if (count > 20){
		count = 0;
	}
	std::ofstream outputfile;
	outputfile.open(file_name + std::to_string(count) + ".txt", ios::trunc);
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