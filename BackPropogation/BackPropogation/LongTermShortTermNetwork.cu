#include "LongTermShortTermNetwork.cuh"
#define NUMBER_NODES_IN_MEMORY_CELL 5

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

	//Initialize the layers
	this->mBlocksLayers = vector<vector<Memory_Block>>();


	positionOfLastWeightToNode = vector<long>();
	this->numberNonWeights = this->settings.i_input;

	this->createMemoryBlock(2);

}


//***************************
//Modify Structure Of Neuron
//***************************
int LongTermShortTermNetwork::decideNodeToAttachTo(){
	vector<int> notFullyConnected = vector<int>();
	int numberMBlocks = this->mBlocksLayers[0].size();
	//Find how many nodes are not fully connected
	for (int i = 0; i < numberMBlocks; i++){
		//A node is considered not fully connected if it is not connected to at least one other memory block
		if (this->mBlocksLayers[0][i].input_weights.size() < this->numberNonWeights + numberMBlocks){
			notFullyConnected.push_back(i);
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
	for (int k = this->numberNonWeights; k < this->bias.size() - this->settings.i_output; k++){
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
	srand(time(NULL));
	return RandomClamped();
}



void LongTermShortTermNetwork::addWeight(int numberWeightsToAdd){
	int decideTo = this->decideNodeToAttachTo();
	if (decideTo != -1){
		this->mBlocksLayers[0][decideTo].addNewConnection(this->numberNonWeights - 1, this->mBlocksLayers[0].size() + this->numberNonWeights);
	}
	else{

	}

}

void LongTermShortTermNetwork::addNeuron(int numberNeuronsToAdd){
	this->createMemoryBlock(numberNeuronsToAdd);
}


void LongTermShortTermNetwork::createMemoryBlock(int numberMemoryCells){
	if (this->mBlocksLayers.size() == 0){
		this->mBlocksLayers.push_back(vector<Memory_Block>());//Add one hidden layer
		this->mBlocksLayers.push_back(vector<Memory_Block>());//Add one output layer
		for (unsigned int i = 0; i < this->settings.i_output; i++){
			this->mBlocksLayers[1].push_back(Memory_Block(numberMemoryCells + this->numberNonWeights, numberMemoryCells));
		}
	}

	for (int i = 0; i < numberMemoryCells; i++){
		this->mBlocksLayers[0].push_back(Memory_Block(this->settings.i_input));
	}

}

void LongTermShortTermNetwork::InitialcreateMemoryBlock(int numberMemoryCells){
	if (this->mBlocksLayers.size() == 0){
		this->mBlocksLayers.push_back(vector<Memory_Block>());
	}
	this->mBlocksLayers[0].push_back(Memory_Block(this->settings.i_input));
}



//*********************
//Run The Network
//*********************
//Multiply two values
template <typename T>
struct multiply : public thrust::unary_function <T,T> {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
		T operator()(Tuple x) const{
		return (thrust::get<0>(x) * thrust::get<1>(x));
	}

};
template <typename T>
struct run_memory_block_functon : public::unary_function < T, T > {


	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple &x){//Received Tuple is in the form input, output, forget, potential memory cell, memory cell value
		thrust::get<3>(x) = thrust::get<0>(x)*thrust::get<3>(x);//Multiply the input by the potential_memory_value
		thrust::get<2>(x) = thrust::get<2>(x) * thrust::get<4>(x); //Multiply the forget * the old memory cell value
		thrust::get<4>(x) = thrust::get<2>(x) + thrust::get<3>(x) + thrust::get<4>(x); //Sum the forget,input, and old cell value to get the new vaue the new potential memory cell value
		thrust::get<1>(x) = thrust::get<4>(x) * thrust::get<1>(x); //Multiply the new memory_cell value by the new output value 
	}

};

//Perform Sigmoid Operation of a Tuple
template <typename T>
struct sigmoid_tuple_functor : public thrust::unary_function <T, T> {

	//Overload the function operator
	template <typename Tuple>
	__host__ __device__
	T operator()(Tuple x) const{
		T z = (T)(thrust::get<0>(x)*thrust::get<1>(x));
		z = thrust::exp(((T)-1) * z);
		return (T)1 / ((T)1 + z);
	}

};



//Perform a sigmoid function
template <typename T>
struct sigmoid_functor : public thrust::unary_function <T,T> {
	sigmoid_functor(){};

	__host__ __device__
		T operator()(const T &x) const{
		T z = thrust::exp(((T)-1) * x);
		return (T)1 / ((T)1 + z);
	}

};

void LongTermShortTermNetwork::setInput(weight_type* in){
	//Place the input into the GPU values matrix
	for (int i = 0; i < this->settings.i_input; i++){
		this->GPUOutput_values[i] = in[i];
		this->GPUPreviousOutput_Values[i] = in[i];
	}

}

thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in){

	this->setInput(in);
	//Stores the numberofmblocks in a layer
	unsigned int numberMBlocks;
	unsigned int previousnumberMBlocks = 0;
	//Perform the transformation on each layer
	for (unsigned int i = 0; i < this->mBlocksLayers.size(); i++){
		
		if (i != 0){
			previousnumberMBlocks = numberMBlocks;
		}
		numberMBlocks = this->mBlocksLayers[i].size();
		//Sum the values of the input/output/forget/potential_memory_cell_values nodes
		//The values in the GPU weights are in the order input, output, forget, memory cells
		//Subtracting this->mBlocksLayers[i].size() from the end will remove the memory cells from doing anything
		//Output to Previous
		thrust::reduce_by_key(this->GPUMapTo.begin(), this->GPUMapTo.end(), make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin() + this->numberNonWeights + previousnumberMBlocks, // We don't want to multiply the actual input/out values, so we skip them
			make_permutation_iterator( // Create an iterator which maps the values coming from to those going to
			this->GPUOutput_values.begin(),
			this->GPUMapFrom.begin())
			)
			),
			sigmoid_tuple_functor<weight_type>()), //Multiply the two values then run them through a sigmoid function
			thrust::make_discard_iterator(), // Discard the retrieved order, the order should be constant
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks//Store in the previous in order to not overwrite the saved values
			);

		//Create a input/output/forget/potential_memory_cell_values/memory_cell_value value
		//Essentially run the gate and get the output value
		thrust::for_each(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks, //input values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + numberMBlocks + previousnumberMBlocks,//output values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (2 * numberMBlocks) + previousnumberMBlocks,//forget values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (3 * numberMBlocks) + previousnumberMBlocks,//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (4 * numberMBlocks) + previousnumberMBlocks
			)),
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + numberMBlocks + previousnumberMBlocks, //input values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (2 * numberMBlocks) + previousnumberMBlocks,//output values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (3 * numberMBlocks) + previousnumberMBlocks,//forget values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (4 * numberMBlocks) + previousnumberMBlocks,//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (5 * numberMBlocks) + previousnumberMBlocks
			)),
			run_memory_block_functon<weight_type>());
	}



	return device_vector<weight_type>();
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


void LongTermShortTermNetwork::InitializeLongShortTermMemory(){
	//Store all the values in the device
	//Will later add option for too little memory
	//Copy the information to the device
	this->UnrollNetwork(3);

	//Form the delta objects
	this->host_deltas = host_vector<weight_type>(this->GPUOutput_values.size());
	this->device_deltas = device_vector<weight_type>(this->GPUOutput_values.size());
	this->VisualizeNetwork();
}
void LongTermShortTermNetwork::LongTermShortTermNetwork::LongShortTermMemoryTraining(weight_type* in, weight_type* out){
	
	//Set the input values
	this->setInput(in);
	//Special functionality is required for the first layer
	thrust::reduce_by_key
		(
		thrust::make_transo
		
		);

	for (int i = 1; i < this->settings.i_backprop_unrolled - 1; i++){

	}

	//Special functionality is required for the output layer

}


void LongTermShortTermNetwork::ApplyLongTermShortTermMemoryError(){

}

//*********************
//Perform Functionality
//*********************

void LongTermShortTermNetwork::UnrollNetwork(int numLayers){
	vector<vector<Memory_Block>> Unrolled_Layers = vector<vector<Memory_Block>>();//Storage of the memory blocks as new layers
	//Add room for the intial input values
	this->GPUOutput_values.resize(this->numberNonWeights);
	
	for (unsigned int i = 0; i < this->mBlocksLayers.size() - 1; i++){
		this->loadUnrolledToDevice(0, i);
	}

	//Unroll the output layer only once
	this->loadUnrolledToDevice(1, this->mBlocksLayers.size() - 1);
	//Create an empty array for the current values in the network
	this->ResetSequence();
}

void LongTermShortTermNetwork::ResetSequence(){
	thrust::fill(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), 0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), 0);
}

template <typename T>
void copyValuesToHost(int start, device_vector<T> &GPU_Vector, host_vector<T> &local_host_Vector){
	//Copy the values into the network
	thrust::copy(GPU_Vector.begin() + start, GPU_Vector.begin() + local_host_Vector.size() + start, local_host_Vector.begin());
}



void LongTermShortTermNetwork::CopyToHost(){
	//Copy the device memory to local
	this->output_bias.resize(this->GPUOutput_values.size());
	this->bias.resize(this->GPUPreviousOutput_Values.size());
	//Copy the output
	thrust::copy(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), this->output_bias.begin());
	//Copy the secondary output
	thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), this->bias.begin());
	int start = 0;
	for (int j = 0; j < this->mBlocksLayers.size(); j++){
		//Copy back the input
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].input_weights);
			start += this->mBlocksLayers[j][i].input_weights.size();
		}
		//Copy back to output
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].output_weights);
			start += this->mBlocksLayers[j][i].output_weights.size();
		}

		//Copy back to forget
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].forget_weights);
			start += this->mBlocksLayers[j][i].forget_weights.size();
		}

		//Copy back to potential memory cell
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].potential_memory_cell_value);
			start += this->mBlocksLayers[j][i].potential_memory_cell_value.size();
		}

		//Get the new memory_cell_weight
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			this->mBlocksLayers[j][i].memory_cell_weights = this->GPUOutput_values[this->GPUOutput_values.size() - i - this->mBlocksLayers[j].size()];
		}
	}
}

template <typename T>
void LongTermShortTermNetwork::copyNodesToDevice(device_vector<T> &GPU_Vector, device_vector<int> &fromPosition, host_vector<T> &local_host_Vector, host_vector<int> host_from_vector){
	int GPU_VecSize = GPU_Vector.size();
	int GPUFromPos = fromPosition.size();
	GPU_Vector.resize(GPU_VecSize + local_host_Vector.size());
	fromPosition.resize(GPU_VecSize + local_host_Vector.size());
	//Copy the values into the network
	thrust::copy(local_host_Vector.begin(), local_host_Vector.end(), GPU_Vector.begin() + GPU_VecSize);
	thrust::copy(host_from_vector.begin(), host_from_vector.end(), fromPosition.begin() + GPUFromPos);
}

//Copies only the input and not the device
template <typename T>
void LongTermShortTermNetwork::specialCopyToNodes(int start_output, int number_output, device_vector<T> &GPUWeightVector, device_vector<int> &toPosition, device_vector<int> &fromPosition, host_vector<T> &local_weights, host_vector<int> map){
	int GPU_VecSize = GPUWeightVector.size();
	int GPUFromPos = fromPosition.size();
	
	//We need to store the number a special copy of a map, such that it has input from both the input of the sequence and the output of the previous layer
	GPUWeightVector.resize(GPU_VecSize + local_weights.size());
	fromPosition.resize(GPU_VecSize + map.size());
	thrust::copy(map.begin(), map.end(),fromPosition.begin() + GPUFromPos);
	thrust::copy(local_weights.begin(), local_weights.end(), GPUWeightVector.begin() + GPU_VecSize);
	GPU_VecSize = GPU_VecSize + local_weights.size();
	
	for (unsigned int i = start_output; i <start_output + number_output; i++){
		fromPosition.push_back(i);
		GPUWeightVector.push_back(1);
	}

	toPosition.resize(fromPosition.size());
	thrust::fill(toPosition.end() - (local_weights.size() + number_output), toPosition.end(), this->GPUOutput_values.size());

}

void LongTermShortTermNetwork::loadUnrolledToDevice(int type_of_row,unsigned int j){
	//We need to keep track of the end of the number of inputs in order to add in a connection to the outputs for the next level
	unsigned int start_output_position = 0;
	unsigned int number_output_to_add = 0;
	
	if (type_of_row == 0){//Is not an output row
		number_output_to_add = this->mBlocksLayers[j].size();
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			start_output_position = this->mBlocksLayers[j][i].input_weights.size();
		}



		//Increment it by the input numbers
		start_output_position += this->settings.i_input - 1;
	}
	//Set all the input values
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].input_weights, this->mBlocksLayers[j][i].mapFrom);
		
		this->GPUOutput_values.push_back(0);
	}


	//Set all the outputs
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].output_weights, this->mBlocksLayers[j][i].mapFrom);
		this->GPUOutput_values.push_back(0);
	}

	//Set all the forget nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].forget_weights, this->mBlocksLayers[j][i].mapFrom);
		this->GPUOutput_values.push_back(0);
	}

	//Set all the potential_output nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].potential_memory_cell_value, this->mBlocksLayers[j][i].mapFrom);
		this->GPUOutput_values.push_back(0);
	}

	this->GPUPreviousOutput_Values.resize(this->GPUOutput_values.size());
	//Set the values of the Memory Cells
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		this->GPUOutput_values.push_back(this->mBlocksLayers[j][i].memory_cell_weights);
		this->GPUPreviousOutput_Values.push_back(this->mBlocksLayers[j][i].memory_cell_weights);
	}
}


void LongTermShortTermNetwork::loadLayerToDevice(unsigned int j){
	//Set all the input values
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].input_weights, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].input_weights.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].input_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		this->GPUOutput_values.push_back(0);
	}


	//Set all the outputs
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].output_weights, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].output_weights.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].output_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		this->GPUOutput_values.push_back(0);
	}

	//Set all the forget nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].forget_weights, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].forget_weights.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].forget_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		this->GPUOutput_values.push_back(0);
	}

	//Set all the potential_output nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].potential_memory_cell_value, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].potential_memory_cell_value.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].potential_memory_cell_value.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		this->GPUOutput_values.push_back(0);
	}

	this->GPUPreviousOutput_Values.resize(this->GPUOutput_values.size());
	//Set the values of the Memory Cells
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		this->GPUOutput_values.push_back(this->mBlocksLayers[j][i].memory_cell_weights);
		this->GPUPreviousOutput_Values.push_back(this->mBlocksLayers[j][i].memory_cell_weights);
	}
}

void LongTermShortTermNetwork::CopyToDevice(){
	this->device_deltas = device_vector<weight_type>();
	this->GPUMapTo = device_vector<int>();
	this->GPUMapFrom = device_vector<int>();
	this->GPUOutput_values = device_vector<weight_type>();
	this->GPUPreviousOutput_Values = device_vector<weight_type>();
	this->GPUWeights = device_vector<weight_type>();

	//Set the input values to 0
	this->GPUOutput_values.resize(this->numberNonWeights);
	for (unsigned int j = 0; j < this->mBlocksLayers.size(); j++){
		this->loadLayerToDevice(j);
	}

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

void LongTermShortTermNetwork::emptyGPUMemory(){
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
	cout << *this;
}

ostream& LongTermShortTermNetwork::OutputNetwork(ostream& os){
	os << *this;
	return os;
}