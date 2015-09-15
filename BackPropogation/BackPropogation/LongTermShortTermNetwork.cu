#include "LongTermShortTermNetwork.cuh"

LongTermShortTermNetwork::LongTermShortTermNetwork(){
	this->settings = CSettings();
	LongTermShortTermNetwork(this->settings);
}

LongTermShortTermNetwork::LongTermShortTermNetwork(CSettings& settings){
	this->settings = settings;
	this->initialize_network();
}

//***************************
//Initialize Network
//***************************

void LongTermShortTermNetwork::initialize_network(){
	this->weights = host_vector<weight_type>();
	this->mapTo = host_vector<int>();
	this->mapFrom = host_vector<int>();
	this->bias = host_vector<weight_type>();

	//Initialize the position
	this->weight_position = vector<thrust::host_vector<weight_type>::iterator>();
	this->mapTo_position = vector<thrust::host_vector<int>::iterator>();
	this->mapFrom_position = vector<thrust::host_vector<int>::iterator>();

	positionOfLastWeightToNode = vector<long>();

	for (int i = 0; i < this->settings.i_input; i++){
		this->bias.push_back(0);
	}

	

	//Initially only create on memory block connected to all input/output
	this->InitialcreateMemoryBlock(1);
	for (int i = 0; i < this->settings.i_output; i++){
		this->addNewPositionInList();
		this->addNewNeuron(this->bias.size() - this->settings.i_input, this->bias.size() - this->settings.i_input, 1, this->last_output_cell_pos, this->bias.size() - this->settings.i_input);
		this->bias.push_back(0);
	}

	this->createMemoryBlock(1);

}


//***************************
//Modify Structure Of Neuron
//***************************
int LongTermShortTermNetwork::decideNodeToAttachTo(){
	vector<int> notFullyConnected = vector<int>();
	//Find how many nodes are not fully connected
	for (int k = this->settings.i_output; k<this->numberOfNodes; k++){
		if (this->positionOfLastWeightToNode[k] - (k > this->settings.i_output ? this->positionOfLastWeightToNode[k - 1] : -1) < (this->numberOfNodes + this->settings.i_input - this->settings.i_output)){
			notFullyConnected.push_back(k);
		}
	}

	if (notFullyConnected.size() > 0){
		//Return a random number in the set of not fully connected nodes
		//It's the number contained in the positionOfLastWeightToNode which is this->settings.i_input less than its actual position
		return notFullyConnected[RandInt(0, notFullyConnected.size() - 1)];
	}
	else{
		//All nodes are fully connected and no new weights can be added
		return -1;
	}
}

int LongTermShortTermNetwork::decideNodeToAttachFrom(int attachTo){
	vector<int> notConnectedTo = vector<int>();
	bool containsValue = false;
	int start = (this->settings.i_output != attachTo ? this->positionOfLastWeightToNode[attachTo - 1] : 0);
	int end = (this->settings.i_output != attachTo ? this->positionOfLastWeightToNode[attachTo] : this->positionOfLastWeightToNode[attachTo]);
	for (int k = this->numberNonWeights; k < this->bias.size(); k++){
		for (int i = start; i <= end; i++){
			if (this->mapFrom[i] == k){
				//The value is already contained in the system
				containsValue = true;
				break;
			}
		}

		if (!containsValue){
			notConnectedTo.push_back(k);
		}
		containsValue = false;


	}



	if (notConnectedTo.size() != 0){
		//It's the number contained in the positionOfLastWeightToNode which is this->settings.i_input less than its actual position
		return notConnectedTo[RandInt(0, notConnectedTo.size() - 1)];
	}
	else{
		return -1;
	}

}





weight_type LongTermShortTermNetwork::getNewWeight(){
	return RandomClamped();
}



void LongTermShortTermNetwork::addWeight(int numberWeightsToAdd){
}

void LongTermShortTermNetwork::addNeuron(int numberNeuronsToAdd){
	
}

void LongTermShortTermNetwork::addNewNeuron(int store,int position, weight_type weight, int mapFrom, int mapTo){
	this->weight_position[store] = this->weights.insert(this->weight_position[position], weight) + 1;
	//Push back the node the values is coming from
	this->mapFrom_position[store] = this->mapFrom.insert(this->mapFrom_position[position], mapFrom) + 1;
	this->mapTo_position[store] = this->mapTo.insert(this->mapTo_position[position], mapTo) + 1;
}

void LongTermShortTermNetwork::addNewPositionInList(){
	this->mapTo_position.push_back(this->mapTo.begin() + this->settings.i_input - 1);
	this->mapFrom_position.push_back(this->mapFrom.begin() + this->settings.i_input - 1);
	this->weight_position.push_back(this->weights.begin() + this->settings.i_input - 1);
}

void LongTermShortTermNetwork::createMemoryBlock(int numberMemoryCells){
	//Create the input block
	//Create the input block
	//Create the input Lock

	this->addNewPositionInList();
	//Add the weight for the inputs
	for (int i = 0; i < this->settings.i_input; i++){
		this->addNewNeuron(this->last_input_cell_pos, this->last_input_cell_pos-1, this->getNewWeight(), i, this->last_input_cell_pos);
	}
	
	this->bias.push_back(this->getNewWeight());

	this->weight_position[this->last_input_cell_pos]++;
	//Push back the node the values is coming from
	this->mapFrom_position[this->last_input_cell_pos]++;
	this->mapTo_position[this->last_input_cell_pos]++;

	this->last_input_cell_pos++;


	//Create the memory cell
	for (int i = 0; i < numberMemoryCells; i++){
		this->addNewPositionInList();
		//Add connection to input
		this->addNewNeuron(this->last_memory_cell_pos, this->last_memory_cell_pos - 1, 1, this->last_input_cell_pos - 1, this->last_memory_cell_pos);
		//Add conection to itself
		this->addNewNeuron(this->last_memory_cell_pos,this->last_memory_cell_pos-1, 1, this->last_input_cell_pos - 1, this->last_memory_cell_pos);

		
		this->bias.push_back(this->getNewWeight());
		
		this->last_memory_cell_pos++;
	}

	this->addNewPositionInList();
	//Add connections from memory cells to node
	for (int j = 0; j < numberMemoryCells; j++){
		this->addNewNeuron(this->last_output_cell_pos, this->last_output_cell_pos - 1, 1, this->last_memory_cell_pos - j, this->last_memory_cell_pos);
	}

	//Add connections from input to the output nodes
	for (int i = this->settings.i_input - 1; i < this->settings.i_input + this->settings.i_output - 1; i++){
		this->addNewNeuron(this->last_output_cell_pos, this->last_output_cell_pos - 1, this->getNewWeight(), i, this->last_output_cell_pos - 1);
	}
	
	//Add conections to output
	for (int i = 0; i < this->settings.i_output; i++){
		this->addNewNeuron(this->bias.size() - 1 - i - this->settings.i_input, this->bias.size() - 1 - this->settings.i_input - i, 1, this->weight_position.size() + this->settings.i_input, this->bias.size() - this->settings.i_input - 1 - i);
	}

	this->last_output_cell_pos++;
	this->bias.push_back(this->getNewWeight());

	

	//Increment number of memory blocks
	this->numberOfNodes++;
}

void LongTermShortTermNetwork::InitialcreateMemoryBlock(int numberMemoryCells){
	


}



//*********************
//Run The Network
//*********************
//Multiply two values
template <typename T>
struct multiply : public thrust::unary_function < T, T > {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple &x) const{
		return (thrust::get<0>(x) * thrust::get<1>(x));
	}

};

//Perform a sigmoid function
template <typename T>
struct sigmoid_functor : public thrust::unary_function < T, T > {
	sigmoid_functor(){};

	__host__ __device__
		T operator()(const T &x) const{
		T z = thrust::exp(((T)-1) * x);
		return (T)1 / ((T)1 + z);
	}

};

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in){
	//Sum all the input values
	device_vector<weight_type> GPUOutput_values = this->bias;//Copy the output_nodes
	device_vector<weight_type> GPUPreviousOutput_Values = this->bias;
	device_vector<int> GPUMapFrom = this->mapFrom;//Copy the map from
	device_vector<int> GPUMapTo = this->mapTo; //Copy the mapTo
	device_vector<weight_type> GPUWeights = this->weights;

	//Copy the input into the GPU memory
	for (int i = 0; i < this->settings.i_input; i++){
		GPUOutput_values[i] = (weight_type)in[i];
	}
	this->sumNetworkValues(GPUOutput_values,//Copy the output_nodes
		GPUPreviousOutput_Values,
		GPUMapFrom,//Copy the map from
		GPUMapTo, //Copy the mapTo
		GPUWeights, 1);

	//Free the used memory
	clear_vector::free(GPUMapFrom);
	clear_vector::free(GPUMapTo);
	clear_vector::free(GPUWeights);
	//Return either of these two as the output
	if (this->settings.i_recurrent_flip_flop % 2 == 0){
		clear_vector::free(GPUOutput_values);
		return GPUPreviousOutput_Values;
	}
	else{
		clear_vector::free(GPUPreviousOutput_Values);
		return GPUOutput_values;
	}

}
//*********************
//Training the Network
//*********************
template <typename T>
struct find_error : public thrust::unary_function < T, T > {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple &x) const{
		return thrust::pow((thrust::get<0>(x) -thrust::get<1>(x)), (T)2);
	}

};


//*********************
//Hessian Free
//*********************

void LongTermShortTermNetwork::InitializeLongShortTermMemory(){

	//Store all the values in the device
	//Will later add option for too little memory
	this->host_deltas = host_vector<weight_type>(this->GPUOutput_values.size());
	//Copy the information to the device
	this->CopyToDevice();
	//Fill the intial previous output as 0
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
}
void LongTermShortTermNetwork::LongTermShortTermNetwork::LongShortTermMemoryTraining(weight_type* in, weight_type* out){

	
}
void LongTermShortTermNetwork::ApplyLongTermShortTermMemoryError(){
	
}

//*********************
//Perform Functionality
//*********************

void LongTermShortTermNetwork::sumNetworkValues(device_vector<weight_type> &GPUOutput_values,//Copy the output_nodes
	device_vector<weight_type> &GPUPreviousOutput_Values,
	device_vector<int> &GPUMapFrom,//Copy the map from
	device_vector<int> &GPUMapTo, //Copy the mapTo
	device_vector<weight_type> &GPUWeights, int number_of_rounds){

	for (int i = 0; i < number_of_rounds; i++){
		if (i % 2 != 0){//Store the results in the previous Output
			//Reduce the input into the sum for each neuron
			thrust::reduce_by_key(
				GPUMapTo.begin(),
				GPUMapTo.end(),
				//Transform by multiplying the weight by the previous output
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				GPUWeights.begin(),
				make_permutation_iterator(
				GPUOutput_values.begin(),
				GPUMapFrom.begin()
				))
				),
				multiply<weight_type>()),
				thrust::make_discard_iterator(),
				GPUPreviousOutput_Values.begin()
				);
			//Transform the output using the sigmoid function
			thrust::transform(GPUPreviousOutput_Values.begin(), GPUPreviousOutput_Values.end(), GPUPreviousOutput_Values.begin(), sigmoid_functor<weight_type>());
		}
		else{//Store in current output
			//Reduce the input into the sum for each neuron
			thrust::reduce_by_key(
				GPUMapTo.begin(),
				GPUMapTo.end(),
				//Transform by multiplying the weight by the previous output
				thrust::make_transform_iterator(
				thrust::make_zip_iterator(
				thrust::make_tuple(GPUPreviousOutput_Values.begin(),
				make_permutation_iterator(
				GPUOutput_values.begin(),
				GPUMapFrom.begin()
				))
				),
				multiply<weight_type>()),
				thrust::make_discard_iterator(),
				GPUOutput_values.begin()
				);

			//Transform the output using the sigmoid function
			thrust::transform(GPUOutput_values.begin(), GPUOutput_values.end(), GPUOutput_values.begin(), sigmoid_functor<weight_type>());
		}
	}

}

void LongTermShortTermNetwork::ResetSequence(){
	thrust::fill(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), 0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), 0);
}


void LongTermShortTermNetwork::CopyToHost(){
	//Copy the device memory to local
	thrust::copy(this->GPUMapFrom.begin(), this->GPUMapFrom.end(), this->mapFrom.begin());
	thrust::copy(this->GPUMapTo.begin(), this->GPUMapTo.end(), this->mapTo.begin());
	thrust::copy(this->GPUWeights.begin(), this->GPUWeights.end(), this->weights.begin());
	thrust::copy(this->device_deltas.begin(), this->device_deltas.end(), this->host_deltas.begin());
	thrust::copy(this->device_deltas.begin(), this->device_deltas.end(), this->host_deltas.begin());
	thrust::copy(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), this->bias.begin());
}

void LongTermShortTermNetwork::CopyToDevice(){
	this->device_deltas = this->host_deltas;
	this->GPUMapTo = this->mapTo;
	this->GPUMapFrom = this->mapFrom;
	this->GPUOutput_values = this->bias;
	this->GPUPreviousOutput_Values = this->bias;
	this->GPUWeights = this->weights;
}

void  LongTermShortTermNetwork::cleanNetwork(){
	this->CopyToHost();
	//Free the used memory
	clear_vector::free(this->GPUMapFrom);
	clear_vector::free(this->GPUMapTo);
	clear_vector::free(this->GPUWeights);
	clear_vector::free(this->device_deltas);
	clear_vector::free(this->GPUOutput_values);
	clear_vector::free(this->GPUPreviousOutput_Values);
}
//*********************
//Misc
//*********************
void LongTermShortTermNetwork::VisualizeNetwork(){
	cout.precision(20);
	std::cout << "Weight" << "\t" << "In" << "\t" << "Out" << endl;
	for (int i = 0; i < this->weights.size(); i++){
		std::cout << i << ") " << this->weights[i] << "\t" << this->mapFrom[i] << "\t" << this->mapTo[i] << endl;
	}
	std::cout << endl;
	cout << "Neuron Values" << endl;

	for (int i = this->settings.i_input; i < this->bias.size(); i++){
		std::cout << i << ") " << this->bias[i] << endl;
	}

	std::cout << endl;

	std::cout << endl;
	cout << "deltas" << endl;
	for (int i = 0; i < this->host_deltas.size(); i++){
		std::cout << i << ") " << this->host_deltas[i] << endl;
	}



}