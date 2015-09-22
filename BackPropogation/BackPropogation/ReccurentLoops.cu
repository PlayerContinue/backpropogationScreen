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

vector<RETURN_WEIGHT_TYPE> ReccurentLoops::runNetwork(double* in){

	return this->runNetwork(this->convert_array<double>(in));

}


vector<RETURN_WEIGHT_TYPE> ReccurentLoops::runNetwork(weight_type* in){
	this->mainNetwork->InitializeRun();
#ifdef _DEBUG
	this->createCheckpoint();
#endif
	device_vector<weight_type> temp_device = this->mainNetwork->runNetwork(in);
#ifdef _DEBUG
	this->createCheckpoint();
#endif
	vector<RETURN_WEIGHT_TYPE> to_return = vector <RETURN_WEIGHT_TYPE>(temp_device.size());

	for (unsigned int i = 0; i < temp_device.size(); i++){
		to_return[i] = temp_device[i];
	}
	this->mainNetwork->emptyGPUMemory();
	clear_vector::free(temp_device);

	return to_return;
}
//*****************************
//Get Data From the users file
//*****************************
bool ReccurentLoops::load_training_data_from_file(){
	for (int i = 0; i < this->settings.i_number_of_training; i++){
		this->input[i] = this->createTestInputOutput(this->settings.i_input,0);
		this->output[i] = this->createTestInputOutput(this->settings.i_output,1);
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
			//Apply the error
			//this->mainNetwork->ApplyError();
			if (i%this->settings.i_number_allowed_same == 0){
				this->createCheckpoint();
			}
			this->checkpoint.i_number_of_loops_checkpoint += 1;
			
			if (i%this->settings.i_number_in_sequence == 0){//Reset the sequence once the sequence has finished
				this->mainNetwork->ResetSequence();
			}
			
		}
		try{
			this->createCheckpoint();
			this->mainNetwork->cleanNetwork();
			//this->runNetwork(this->input[0]);

		}
		catch (exception e){
			this->mainNetwork->emptyGPUMemory();
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
	//this->mainNetwork->addNeuron(1);
	//this->mainNetwork->VisualizeNetwork();
	//this->mainNetwork->addNeuron(2);
	//this->mainNetwork->VisualizeNetwork();

	//this->mainNetwork->addWeight(5);
	//this->mainNetwork->VisualizeNetwork();
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
	for (int i = position; i < position + numberOfInput; i++){
		if (input_output == 0){
			temp[i - position] = (weight_type)i;
		}
		else{
			temp[i - position] = (weight_type).1;
		}
	}
	position += numberOfInput;
	return  temp;
}

void ReccurentLoops::createCheckpoint(){
	static int count = 0 ;

		std::ofstream outputfile;
		outputfile.open("networks/" + settings.s_network_name + std::to_string(count) + ".txt", ios::trunc);
		if (outputfile.is_open()){
			for (int i = 0; i < this->settings.i_input; i++){
				//cout << i << ") " << this->input[0][i] << endl;
			}


			for (int i = 0; i < this->settings.i_input; i++){
				//cout << i << ") " << this->output[0][i] << endl;
			}
			//Output the network
			outputfile << *this << flush;
			outputfile << endl;
			cout << *this << endl;
			//vector<weight_type> vect = this->runNetwork(this->input[0]);

			for (int i = 0; i < this->settings.i_input; i++){
				//cout << i << ") " << vect[i] << endl;
			}
			outputfile.close();
		}
		else{
			std::cout << "Unable to write checkpoint to file." << endl;
			std::cout << "continue?";
		}
		
		count++;
		

	

}