#include "LongTermShortTermNetwork.cuh"
#define NUMBER_NODES_IN_MEMORY_CELL 5
#define TEST_DEBUG

//#define _DEBUG_WEIGHTS
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
	srand(time(NULL));
	this->createMemoryBlock(5);

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
			this->mBlocksLayers[1].push_back(Memory_Block(numberMemoryCells + this->numberNonWeights, numberMemoryCells,Memory_Block::OUTPUT));
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


thrust::device_vector<weight_type> LongTermShortTermNetwork::runNetwork(weight_type* in){

	this->setInput(in);
	//Stores the numberofmblocks in a layer
	unsigned int numberMBlocks;
	unsigned int previousnumberMBlocks = 0;
	//Perform the transformation on each layer
	for (unsigned int i = 0; i < this->mBlocksLayers.size()-1; i++){

		if (i != 0){
			previousnumberMBlocks = numberMBlocks;
		}
		numberMBlocks = this->mBlocksLayers[i].size();
		//Sum the values of the input/output/forget/potential_memory_cell_values nodes
		//The values in the GPU weights are in the order input, output, forget, memory cells
		//Subtracting this->mBlocksLayers[i].size() from the end will remove the memory cells from doing anything
		//Output to Previous
		thrust::reduce_by_key(this->GPUMapTo.begin(), this->GPUMapTo.end(), thrust::make_transform_iterator(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUWeights.begin() + this->numberNonWeights + previousnumberMBlocks, // We don't want to multiply the actual input/out values, so we skip them
			make_permutation_iterator( // Create an iterator which maps the values coming from to those going to
			this->GPUOutput_values.begin(),
			this->GPUMapFrom.begin())
			)
			),
			functors::multiply<weight_type>()), //Multiply the two values then run them through a sigmoid function
			thrust::make_discard_iterator(), // Discard the retrieved order, the order should be constant
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks//Store in the previous in order to not overwrite the saved values
			);

		thrust::transform(this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks,
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks + (5 * numberMBlocks),
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks,
			functors::sigmoid_functor<weight_type>());

		//Create a input/output/forget/potential_memory_cell_values/memory_cell_value value
		//Essentially run the gate and get the output value
		thrust::for_each(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + previousnumberMBlocks + this->settings.i_input, //input values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + numberMBlocks + previousnumberMBlocks + this->settings.i_input,//output values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (2 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//forget values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (3 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (4 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input
			)),
			thrust::make_zip_iterator(
			thrust::make_tuple(
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + numberMBlocks + previousnumberMBlocks + this->settings.i_input, //input values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (2 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//output values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (3 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//forget values
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (4 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input,//potential_memory_cell_value
			this->GPUPreviousOutput_Values.begin() + this->numberNonWeights + (5 * numberMBlocks) + previousnumberMBlocks + this->settings.i_input
			)),
			functors::run_memory_block_functon<weight_type>());
		thrust::copy(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), this->GPUOutput_values.begin());
		
	}

	device_vector<weight_type> toReturn = device_vector<weight_type>(this->settings.i_output);
	//Transform the output of the current
	/*thrust::transform(this->GPUOutput_values.end() - 
		(5 * this->mBlocksLayers[this->mBlocksLayers.size() - 1].size()) + this->settings.i_output,
		this->GPUOutput_values.end() - 
		(5 * this->mBlocksLayers[this->mBlocksLayers.size() - 1].size())+ this->settings.i_output + this->settings.i_output,
		toReturn.begin(), _1 * (weight_type)1);*/

	int output_weight_size = ((this->mBlocksLayers[this->mBlocksLayers.size() - 2].size()));

	thrust::reduce_by_key(
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1 / output_weight_size // The number of output nodes in layer before the output layer
		), 
		thrust::make_transform_iterator(
		thrust::make_counting_iterator((int)0),
		_1 / output_weight_size // The number of output nodes in layer before the output layer
		) + output_weight_size * this->settings.i_output,
		thrust::make_transform_iterator(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		thrust::make_permutation_iterator(
		this->GPUOutput_values.begin(),
		this->GPUMapFrom.end() - (output_weight_size * this->settings.i_output)),

		thrust::make_permutation_iterator(
		this->GPUWeights.begin(),
		this->GPUMapTo.end() -( output_weight_size*this->settings.i_output)
		)
		)
		),
		functors::multiply<weight_type>()
		),
		thrust::make_discard_iterator(),
		toReturn.begin()
		);

	thrust::transform(toReturn.begin(), toReturn.end(), toReturn.begin(), functors::sigmoid_functor<weight_type>());

	return toReturn;
}


//*********************
//Perform Functionality
//*********************

//Add the input
void LongTermShortTermNetwork::setInput(weight_type* in){
	//Place the input into the GPU values matrix
	for (int i = 0; i < this->settings.i_input; i++){
		this->GPUOutput_values[i] = in[i];
	}

}

//Copies the bias on Main Memory to GPU Memory
void LongTermShortTermNetwork::moveBiasToGPU(){
	this->GPUBias = thrust::device_vector<weight_type>();

	for (int j = 0; j < this->mBlocksLayers.size(); j++){
		//Copy back the input
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[0]);
			}
		}

		//Copy back to output
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[1]);
			}
		}

		//Copy back to forget
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[2]);
			}
		}

		//Copy back to potential memory cell
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[3]);
		}

		
	}
	//Make the previous list of bias to the bias
	this->GPUPreviousBias = thrust::device_vector<weight_type>(this->GPUBias.size());
}

void LongTermShortTermNetwork::UnrollNetwork(int numLayers){
	vector<vector<Memory_Block>> Unrolled_Layers = vector<vector<Memory_Block>>();//Storage of the memory blocks as new layers
	this->numberOfNodes = 0;
	//Add room for the intial input values
	this->GPUOutput_values.resize(this->numberNonWeights);
	for (unsigned int i = 0; i < this->mBlocksLayers.size() - 1; i++){
		this->loadUnrolledToDevice(2, i);
	}

	//Unroll the output layer only once
	//The output layer will contain only n node (n is the number of output) and will merely sum all input passed into it
	//This makes performing analysis far easier than using a extra layer of memory cells
	this->loadUnrolledToDevice(2, this->mBlocksLayers.size()-1);
	//Expand the output container
	
	this->GPUPreviousOutput_Values.resize(this->GPUOutput_values.size() - this->numberNonWeights);

	int GPUOutput_values_size = this->GPUOutput_values.size();

	//Resize the network to contain locations for the other layer
	this->GPUOutput_values.resize(this->GPUOutput_values.size() + ((this->settings.i_backprop_unrolled - 1)*(this->GPUOutput_values.size() - this->numberNonWeights)));
	
	this->getSumPermutation();

	//Create a container for the previous weights (i.e. prev_delta * alpha)
	this->GPUPreviousWeights = thrust::device_vector<weight_type>(this->GPUWeights.size());
	
	//Copy the bias to GPU
	this->moveBiasToGPU();


	//Create an empty array for the current values in the network
	this->ResetSequence();
}

void LongTermShortTermNetwork::ResetSequence(){
	thrust::fill(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), 0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), 0);
	thrust::fill(this->GPUPreviousWeights.begin(), this->GPUPreviousWeights.end(), 0);
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
	int start = this->settings.i_input;
	for (int j = 0; j < this->mBlocksLayers.size(); j++){
		//Copy back the input
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].input_weights);
				start += this->mBlocksLayers[j][i].input_weights.size();
			}
		}

		//Copy back to output
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].output_weights);
				start += this->mBlocksLayers[j][i].output_weights.size();
			}
		}

		//Copy back to forget
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].forget_weights);
				start += this->mBlocksLayers[j][i].forget_weights.size();
			}
		}

		//Copy back to potential memory cell
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].potential_memory_cell_value);
			start += this->mBlocksLayers[j][i].potential_memory_cell_value.size();
		}

		//Get the new memory_cell_weight
		for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
					this->mBlocksLayers[j][i].memory_cell_weights[0] = 0;
			}
		}
	}
}

template <typename T>
void LongTermShortTermNetwork::copyNodesToDevice(device_vector<T> &GPU_Vector, device_vector<int> &fromPosition, host_vector<T> &local_host_Vector, host_vector<int> host_from_vector){
	int GPU_VecSize = GPU_Vector.size();
	int GPUFromPos = fromPosition.size();
	GPU_Vector.resize(GPU_VecSize + local_host_Vector.size());
	fromPosition.resize(GPUFromPos + local_host_Vector.size());
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
	fromPosition.resize(GPUFromPos + map.size());
	thrust::copy(map.begin(), map.end(), fromPosition.begin() + GPUFromPos);
	thrust::copy(local_weights.begin(), local_weights.end(), GPUWeightVector.begin() + GPU_VecSize);
	GPU_VecSize = GPU_VecSize + local_weights.size();

	for (unsigned int i = start_output; i < start_output + number_output; i++){
		fromPosition.push_back(i);
		GPUWeightVector.push_back(1);
	}

	toPosition.resize(fromPosition.size());
	thrust::fill(toPosition.end() - (local_weights.size() + number_output), toPosition.end(), this->GPUOutput_values.size());

}

void LongTermShortTermNetwork::loadUnrolledToDevice(int type_of_row, unsigned int j){
	//We need to keep track of the end of the number of inputs in order to add in a connection to the outputs for the next level
	unsigned int start_output_position = 0;
	unsigned int number_output_to_add = 0;
	unsigned int* input_nodes = new unsigned int[this->mBlocksLayers[j].size() * 3];
	host_vector<int> memory_cell_from = host_vector<int>(4);
	host_vector<weight_type> memory_cell_weights = host_vector<weight_type>(4);
	memory_cell_weights[0] = 1;//From the input
	memory_cell_weights[1] = 1;//From the potential
	memory_cell_weights[2] = 1;//From the forget
	memory_cell_weights[3] = 1;//From itself

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
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].input_weights, this->mBlocksLayers[j][i].mapFrom);
			input_nodes[i] = this->GPUOutput_values.size();//Get the position of an input node
			if (type_of_row == 2){
				this->numberOfNodes++;
			}
			this->GPUOutput_values.push_back(0);
		}
	}


	//Set all the outputs
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].output_weights, this->mBlocksLayers[j][i].mapFrom);
			//Add a connection to the memory cell
			this->weights.push_back(1);//Push back a 1 for the multiplication value
			input_nodes[i + this->mBlocksLayers[j].size()] = this->GPUOutput_values.size() + (this->mBlocksLayers[j].size() * 2) + this->mBlocksLayers[j].size();//Store the position of the memory cell for the forget node
			this->GPUMapFrom.push_back(input_nodes[i + this->mBlocksLayers[j].size()]);//Push back connection to the memory cell
			this->GPUMapTo.push_back(this->GPUOutput_values.size());
			this->GPUWeights.push_back((weight_type)1);
			if (type_of_row == 2){
				this->numberOfNodes++;
			}
			this->GPUOutput_values.push_back(0);
		}
	}

	//Set all the forget nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			if (type_of_row == 2){
				this->numberOfNodes++;
			}

			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].forget_weights, this->mBlocksLayers[j][i].mapFrom);
			this->weights.push_back(1);//Push back a 1 for the multiplication value
			this->GPUMapFrom.push_back(input_nodes[i + this->mBlocksLayers[j].size()]);//Push back connection to the memory cell
			this->GPUMapTo.push_back(this->GPUOutput_values.size());
			this->GPUWeights.push_back((weight_type)1);
			input_nodes[i + this->mBlocksLayers[j].size()] = this->GPUOutput_values.size();//Set position of the forget blocks
			this->GPUOutput_values.push_back(0);
		}
	}

	//Set all the potential_output nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		
		input_nodes[i + (this->mBlocksLayers[j].size() * 2)] = this->GPUOutput_values.size();//Store the position of the 
		
		if (type_of_row == 2){
			this->numberOfNodes++;
		}

		specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].potential_memory_cell_value, this->mBlocksLayers[j][i].mapFrom);
		
		this->GPUOutput_values.push_back(0);
	}

	//Set the values of the Memory Cells
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			memory_cell_from[0] = input_nodes[i];//Get input in
			memory_cell_from[1] = input_nodes[i + this->mBlocksLayers[j].size()];//The potential input
			memory_cell_from[2] = input_nodes[i + (this->mBlocksLayers[j].size() * 2)];
			memory_cell_from[3] = this->GPUOutput_values.size();//itself

			if (type_of_row == 2){
				this->numberOfNodes++;
			}

			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, memory_cell_weights, memory_cell_from);
			this->GPUOutput_values.push_back(this->mBlocksLayers[j][i].memory_cell_weights[0]);
			this->GPUPreviousOutput_Values.push_back(this->mBlocksLayers[j][i].memory_cell_weights[0]);
		}
	}

	free(input_nodes);
}


void LongTermShortTermNetwork::loadLayerToDevice(unsigned int j){

	//Add a place for the input
	if (j == 0){
		this->GPUOutput_values.push_back(0);
		this->GPUPreviousOutput_Values.push_back(0);
	}

	//Set all the input values
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].input_weights, this->mBlocksLayers[j][i].mapFrom);
			this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].input_weights.size());
			thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].input_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
			this->GPUOutput_values.push_back(0);
		}
	}


	//Set all the outputs
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].output_weights, this->mBlocksLayers[j][i].mapFrom);
			this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].output_weights.size());
			thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].output_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
			this->GPUOutput_values.push_back(0);
		}
	}

	//Set all the forget nodes
	for (int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].forget_weights, this->mBlocksLayers[j][i].mapFrom);
			this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].forget_weights.size());
			thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].forget_weights.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
			this->GPUOutput_values.push_back(0);
		}
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
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			this->GPUOutput_values.push_back(this->mBlocksLayers[j][i].memory_cell_weights[0]);
			this->GPUPreviousOutput_Values.push_back(this->mBlocksLayers[j][i].memory_cell_weights[0]);
		}
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

unsigned int LongTermShortTermNetwork::getNumberMemoryCells(unsigned int layer){
	if (layer >= this->mBlocksLayers.size()){
		throw new exception("Layer does not exist");
	}
	unsigned int memory_cell_count = 0;
	for (int i = 0; i < this->mBlocksLayers[layer].size(); i++){
		memory_cell_count += this->mBlocksLayers[layer][i].number_memory_cells;
	}
	return memory_cell_count;

}

unsigned int LongTermShortTermNetwork::getNumberWeightsInLayer(unsigned int layer){
	if (layer >= this->mBlocksLayers.size()){
		throw new exception("Layer does not exist");
	}

	unsigned int weights_count = 0;
	for (int i = 0; i < this->mBlocksLayers[layer].size(); i++){
		weights_count += this->mBlocksLayers[layer][i].input_weights.size();
		weights_count += this->mBlocksLayers[layer][i].forget_weights.size();
		weights_count += this->mBlocksLayers[layer][i].output_weights.size();
		weights_count += (this->mBlocksLayers[layer][i].number_memory_cells * 2);//2 is because there is a weight between the memory cell, the forget node, and the output node
		weights_count += (this->mBlocksLayers[layer][i]).number_memory_cells * 3; //The number of weights from the input,potential, and the forget node
	}
	return weights_count;
}

//Returns number of weights going to the type of node in the layer
unsigned int LongTermShortTermNetwork::getNumberTypeWeightsInLayer(unsigned int layer, cell_type cell){
	if (layer >= this->mBlocksLayers.size()){
		throw new exception("Layer does not exist");
	}
	unsigned int number_types = 0;

	for (int i = 0; i < this->mBlocksLayers[layer].size(); i++){
		switch (cell){
		case MEMORY_CELL:
			number_types += (this->mBlocksLayers[layer][i].number_memory_cells) * 3;
			break;
		case POTENTIAL_MEMORY_CELL:
			number_types += (this->mBlocksLayers[layer][i].potential_memory_cell_value.size());
			break;
		case FORGET_CELL:
			number_types += (this->mBlocksLayers[layer][i].forget_weights.size()) + this->mBlocksLayers[layer][i].number_memory_cells;
			break;
		case INPUT_CELL:
			number_types += this->mBlocksLayers[layer][i].input_weights.size();
			break;
		case OUTPUT_CELL:
			number_types += this->mBlocksLayers[layer][i].output_weights.size() + this->mBlocksLayers[layer][i].number_memory_cells;//+1 for the number of memory cell connections
			break;
		}

	}



	return number_types;

}
void LongTermShortTermNetwork::getSumPermutation(){
	//Create a permutation list containing a list of object
	this->positionToSum = thrust::device_vector<int>();
	this->count = thrust::device_vector<int>();
	int* list_of_found_values = new int[this->numberOfNodes];
#ifdef  _DEBUG
	vector<int> temp = vector<int>();
#endif
	for (unsigned int i = 0; i < this->numberOfNodes; i++){
			for (unsigned int j = 0; j < this->GPUMapTo.size(); j++){
				if (i == this->GPUMapFrom[j]){
					//This is a position of one of the matching nodes
					this->positionToSum.push_back(j);
					this->count.push_back(i);

#ifdef  _DEBUG
					temp.push_back(j);
#endif

				}
		}

	}
		
	

}

void  LongTermShortTermNetwork::cleanNetwork(){
	this->CopyToHost();
	//Free the used memory
	this->emptyGPUMemory();
}

void LongTermShortTermNetwork::emptyGPUMemory(){
	clear_vector::free(this->GPUMapFrom);
	clear_vector::free(this->GPUMapTo);
	clear_vector::free(this->GPUWeights);
	clear_vector::free(this->device_deltas);
	clear_vector::free(this->GPUOutput_values);
	clear_vector::free(this->GPUPreviousOutput_Values);
	clear_vector::free(this->positionToSum);
	clear_vector::free(this->count);
	clear_vector::free(this->GPUPreviousWeights);
	clear_vector::free(this->GPUBias);
	clear_vector::free(this->GPUPreviousBias);
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