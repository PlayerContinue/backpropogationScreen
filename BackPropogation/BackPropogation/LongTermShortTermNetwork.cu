#include "LongTermShortTermNetwork.cuh"
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

LongTermShortTermNetwork::LongTermShortTermNetwork(CSettings& settings, bool checkpoint){
	this->settings = settings;
	this->RealOutput = device_vector<weight_type>(this->settings.i_output);
}

LongTermShortTermNetwork::~LongTermShortTermNetwork(){
	this->emptyGPUMemory();//Empty the GPU Memory
}

//***************************
//Initialize Network
//***************************

void LongTermShortTermNetwork::initialize_network(){
	positionOfLastWeightToNode = vector<long>();
	this->numberNonWeights = this->settings.i_input;
	srand(time(NULL));
	this->numberOfWeightsInLayers = vector<unsigned int>();
	this->InitialcreateMemoryBlock(this->settings.i_number_start_nodes);
	this->weights = host_vector<weight_type>();
	this->mapTo = host_vector<int>();
	this->mapFrom = host_vector<int>();
	this->bias = host_vector<weight_type>();
	this->addWeight(this->settings.i_number_new_weights);
	//Initialize the weight of cell type count
	this->number_weights_by_type = std::vector<std::vector<int>>();
	this->number_nodes_by_type = std::vector<std::vector<int>>();
	this->number_nodes_in_layer = std::vector<int>();
	//Count the number of weights
	this->count_weights_in_layers();
	this->total_number_of_unrolled = this->settings.i_backprop_unrolled;
}





//*********************
//Perform Functionality
//*********************


void LongTermShortTermNetwork::moveBiasToGPU(bool add_memory_cells){
	this->GPUBias = thrust::device_vector<weight_type>();

	for (unsigned int j = 0; j < this->mBlocksLayers.size(); j++){
		//Copy back the input
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[0]);
			}
		}

		//Copy back to output
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[1]);
			}
		}

		//Copy back to forget
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[2]);
			}
		}

		//Copy back to potential memory cell
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[3]);
			}
			else if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::OUTPUT){
				this->GPUBias.push_back(this->mBlocksLayers[j][i].bias[0]);
			}
		}

		
			//Add the memory cells
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
				if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
					this->GPUBias.push_back((weight_type)0);
				}
			}
		
	}
	if (add_memory_cells){
		//Make the previous list of bias to the bias
		this->GPUPreviousBias = thrust::device_vector<weight_type>(this->GPUBias.size());
	}
}

//Copies the bias on Main Memory to GPU Memory
void LongTermShortTermNetwork::moveBiasToGPU(){
	this->moveBiasToGPU(true);
}

void LongTermShortTermNetwork::UnrollNetwork(int numLayers){
	vector<vector<Memory_Block>> Unrolled_Layers = vector<vector<Memory_Block>>();//Storage of the memory blocks as new layers
	this->numberOfNodes = 0;
	//Add room for the intial input values
	
	for (unsigned int i = 0; i < this->mBlocksLayers.size() - 1; i++){
		this->GPUOutput_values.resize(this->GPUOutput_values.size() + this->numberNonWeights);
		this->loadUnrolledToDevice(2, i);
	}

	//Unroll the output layer only once
	//The output layer will contain only n node (n is the number of output) and will merely sum all input passed into it
	//This makes performing analysis far easier than using a extra layer of memory cells
	this->loadUnrolledToDevice(2, this->mBlocksLayers.size()-1);
	
	//Transform the output pointers to get results from the next layer
	thrust::transform(
		this->GPUMapFrom.end() - this->number_weights_by_type[this->number_weights_by_type.size() - 1][POTENTIAL_MEMORY_CELL], 
		this->GPUMapFrom.end(),
		this->GPUMapFrom.end() - this->number_weights_by_type[this->number_weights_by_type.size() - 1][POTENTIAL_MEMORY_CELL],
		_1 + this->numberOfNodes + this->numberNonWeights);

	//Expand the output container
	
	this->GPUPreviousOutput_Values.resize(this->GPUOutput_values.size() - this->numberNonWeights);

	int GPUOutput_values_size = this->GPUOutput_values.size();

	//Resize the network to contain locations for the other layer
	this->GPUOutput_values.resize(this->GPUOutput_values.size() + ((this->settings.i_backprop_unrolled - 1)*(this->GPUOutput_values.size())));
	
	this->getSumPermutation();

	//Create a container for the previous weights (i.e. prev_delta * alpha)
	this->GPUPreviousWeights = thrust::device_vector<weight_type>(this->GPUWeights.size());
	
	//Copy the bias to GPU
	this->moveBiasToGPU();


	this->GPUPreviousTemp = thrust::device_vector<weight_type>(((this->GPUPreviousBias.size() > this->GPUPreviousWeights.size()) ? this->GPUPreviousBias.size() : this->GPUPreviousWeights.size()));

	//Create an empty array for the current values in the network
	this->ResetSequence();
}

void LongTermShortTermNetwork::ResetSequence(){
	thrust::fill(this->GPUOutput_values.begin(), this->GPUOutput_values.end(), (weight_type)0);
	thrust::fill(this->GPUPreviousOutput_Values.begin(), this->GPUPreviousOutput_Values.end(), (weight_type)0);
	thrust::fill(this->GPUPreviousWeights.begin(), this->GPUPreviousWeights.end(), (weight_type)0);
	thrust::fill(this->GPUPreviousBias.begin(), this->GPUPreviousBias.end(), (weight_type)0);
	this->newSequence = true;//Tell the network a new sequence is too be read. Only needed during training
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
	int start = 0;//Number counting position
	int start_of_layer = this->numberOfWeightsInLayers[0];
	int biasCount = 0;//Keeps track of the position in the bias list
	for (unsigned int j = 0; j < this->mBlocksLayers.size(); j++){
		if (j != 0){
			start = start_of_layer;
			start_of_layer += this->numberOfWeightsInLayers[j];
		}
		//Copy back the input
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].input_weights);
				this->mBlocksLayers[j][i].bias[0] = this->GPUBias[biasCount];
				biasCount++;
				start += this->mBlocksLayers[j][i].input_weights.size() + this->mBlocksLayers[j][i].number_memory_cells;
			}
		}

		//Copy back to output
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].output_weights);
				this->mBlocksLayers[j][i].bias[1] = this->GPUBias[biasCount];
				biasCount++;
				start += this->mBlocksLayers[j][i].output_weights.size() + this->mBlocksLayers[j][i].number_memory_cells;//+1 is for extra weight going to the memory node
			}
		}

		//Copy back to forget
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].forget_weights);
				this->mBlocksLayers[j][i].bias[2] = this->GPUBias[biasCount];
				biasCount++;
				start += this->mBlocksLayers[j][i].forget_weights.size() + this->mBlocksLayers[j][i].number_memory_cells;
			}
		}

		//Copy back to potential memory cell
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			copyValuesToHost<weight_type>(start, this->GPUWeights, this->mBlocksLayers[j][i].potential_memory_cell_value);
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
				this->mBlocksLayers[j][i].bias[3] = this->GPUBias[biasCount];
				start += this->mBlocksLayers[j][i].number_memory_cells;
			}
			else if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::OUTPUT){
				this->mBlocksLayers[j][i].bias[0] = this->GPUBias[biasCount];
			}
			biasCount++;
			start += this->mBlocksLayers[j][i].potential_memory_cell_value.size();
			
		}

		//Get the new memory_cell_weight
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			if (this->mBlocksLayers[j][0].getTypeOfMemoryBlock() == Memory_Block::LAYER){
					biasCount++;
					this->mBlocksLayers[j][i].memory_cell_weights[0] = (weight_type)0;
					start += NUMBER_WEIGHTS_TO_MEM;
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

	for (int i = start_output; i < start_output + number_output; i++){
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
	host_vector<int> memory_cell_from = host_vector<int>(3);
	host_vector<weight_type> memory_cell_weights = host_vector<weight_type>(3);
	memory_cell_weights[0] = 1;//From the input
	memory_cell_weights[1] = 1;//From the potential
	memory_cell_weights[2] = 1;//From the forget
	//memory_cell_weights[3] = 1;//From itself

	if (type_of_row == 0){//Is not an output row
		number_output_to_add = this->mBlocksLayers[j].size();
		for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
			start_output_position = this->mBlocksLayers[j][i].input_weights.size();
		}



		//Increment it by the input numbers
		start_output_position += this->settings.i_input - 1;
	}

	//Set all the input values
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].input_weights, this->mBlocksLayers[j][i].mapFrom);
			input_nodes[i] = this->GPUOutput_values.size();//Get the position of an input node
			this->GPUMapFrom.push_back(this->number_nodes_in_layer[j] - this->number_nodes_by_type[j][MEMORY_CELL] + this->numberNonWeights);//Push back connection to the memory cell
			this->GPUMapTo.push_back(this->GPUOutput_values.size());
			this->GPUWeights.push_back((weight_type)1);
			if (type_of_row == 2){
				this->numberOfNodes++;
			}
			this->GPUOutput_values.push_back((weight_type)0);
		}
	}


	//Set all the outputs
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
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
			this->GPUOutput_values.push_back((weight_type)0);
		}
	}

	//Set all the forget nodes
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
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
			this->GPUOutput_values.push_back((weight_type)0);
		}
	}

	//Set all the potential_output nodes
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
		
		input_nodes[i + (this->mBlocksLayers[j].size() * 2)] = this->GPUOutput_values.size();//Store the position of the potential output node for the memory cell
		
		if (type_of_row == 2){
			this->numberOfNodes++;
		}
		specialCopyToNodes<weight_type>(start_output_position, number_output_to_add, this->GPUWeights, this->GPUMapTo, this->GPUMapFrom, this->mBlocksLayers[j][i].potential_memory_cell_value, this->mBlocksLayers[j][i].mapFrom);
		
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){//Push back a memory cell
			this->GPUMapFrom.push_back(this->number_nodes_in_layer[j] - this->number_nodes_by_type[j][MEMORY_CELL] + this->numberNonWeights);//Push back connection to the memory cell
			this->GPUMapTo.push_back(this->GPUOutput_values.size());
			this->GPUWeights.push_back((weight_type)1);
		}
	
		
		this->GPUOutput_values.push_back(0);
	}

	//Set the values of the Memory Cells
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			memory_cell_from[0] = input_nodes[i];//Get input in
			memory_cell_from[1] = input_nodes[i + this->mBlocksLayers[j].size()];//The potential input
			memory_cell_from[2] = input_nodes[i + (this->mBlocksLayers[j].size() * 2)];
			//memory_cell_from[3] = this->GPUOutput_values.size();//itself
			
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

	

	//Set all the input values
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].input_weights, this->mBlocksLayers[j][i].mapFrom);
			this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].mapFrom.size());
			thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].mapFrom.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
			this->GPUOutput_values.push_back(0);
		}
	}


	//Set all the outputs
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].output_weights, this->mBlocksLayers[j][i].mapFrom);
			this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].mapFrom.size());
			thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].mapFrom.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
			this->GPUOutput_values.push_back(0);
		}
	}

	//Set all the forget nodes
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
		if (this->mBlocksLayers[j][i].getTypeOfMemoryBlock() == Memory_Block::LAYER){
			copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].forget_weights, this->mBlocksLayers[j][i].mapFrom);
			this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].mapFrom.size());
			thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].mapFrom.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
			this->GPUOutput_values.push_back(0);
		}
	}

	//Set all the potential_output nodes
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
		copyNodesToDevice<weight_type>(this->GPUWeights, this->GPUMapFrom, this->mBlocksLayers[j][i].potential_memory_cell_value, this->mBlocksLayers[j][i].mapFrom);
		this->GPUMapTo.resize(this->GPUMapTo.size() + this->mBlocksLayers[j][i].mapFrom.size());
		thrust::fill(this->GPUMapTo.end() - this->mBlocksLayers[j][i].mapFrom.size(), this->GPUMapTo.end(), this->GPUOutput_values.size());
		this->GPUOutput_values.push_back(0);
	}

	this->GPUPreviousOutput_Values.resize(this->GPUOutput_values.size());
	//Set the values of the Memory Cells
	for (unsigned int i = 0; i < this->mBlocksLayers[j].size(); i++){
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
	for (unsigned int i = 0; i < this->mBlocksLayers[layer].size(); i++){
		memory_cell_count += this->mBlocksLayers[layer][i].number_memory_cells;
	}
	return memory_cell_count;

}

unsigned int LongTermShortTermNetwork::getNumberWeightsInLayer(unsigned int layer){
	if (layer >= this->mBlocksLayers.size()){
		throw new exception("Layer does not exist");
	}

	unsigned int weights_count = 0;
	for (unsigned int i = 0; i < this->mBlocksLayers[layer].size(); i++){
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

	for (unsigned int i = 0; i < this->mBlocksLayers[layer].size(); i++){
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
	this->positionToSum = thrust::device_vector<int>(this->GPUMapFrom.size());
	this->count = thrust::device_vector<int>(
		
		thrust::make_transform_iterator(this->GPUMapFrom.begin(),
		functors::add_when_greater_than<int>(-(this->numberOfNodes + this->numberNonWeights), this->numberOfNodes + this->numberNonWeights)
		),
		thrust::make_transform_iterator(this->GPUMapFrom.end(),
		functors::add_when_greater_than<int>(-(this->numberOfNodes + this->numberNonWeights), this->numberOfNodes + this->numberNonWeights)
		)
		
		);
	thrust::sequence(this->positionToSum.begin(), this->positionToSum.end(), (int)(0));
	thrust::sort_by_key(this->count.begin(), this->count.end(), this->positionToSum.begin());
	thrust::device_vector<int>::iterator positionToSumStartRemove = thrust::remove_if(this->positionToSum.begin(), this->positionToSum.end(), this->count.begin(), _1 < this->numberNonWeights);
	thrust::device_vector<int>::iterator countStartRemove = thrust::remove_if(this->count.begin(), this->count.end(), _1 < this->numberNonWeights);
	this->count.erase(countStartRemove,this->count.end());
	this->positionToSum.erase(positionToSumStartRemove,this->positionToSum.end());

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
	clear_vector::free(this->GPUPreviousTemp);
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

istream& LongTermShortTermNetwork::LoadNetwork(istream& is){
	this->emptyGPUMemory();//Empty the memory
	string name;
	int count;
	int count2; 
	is >>  this->numberOfNodes;
	is >>  this->numberNonWeights;
	is >>  name;
	is >>  count;
	this->numberOfWeightsInLayers = vector<unsigned int>();
	for (int i = 0; i < count; i++){
		is >>  count2;
		this->numberOfWeightsInLayers.push_back(count2);
	}

	is >>  count;//Get the number of blocks
	
	this->mBlocksLayers = vector<vector<Memory_Block>>();
	for (int i = 0; i < count; i++){
		is >> count2;
		this->mBlocksLayers.push_back(vector<Memory_Block>());
		for (int j = 0; j < count2; j++){
			this->mBlocksLayers[i].push_back(Memory_Block());
			is >>  this->mBlocksLayers[i][j];
		}
		is >>  name;
	}
	
	weight_type value;
	is >>  name;
	is >>  count;
	this->GPUWeights = thrust::device_vector<weight_type>();
	for (int i = 0; i < count; i++){
		is >>  value;
		this->GPUWeights.push_back(value);
	}

	is >>  name;
	is >>  count;
	this->device_deltas = thrust::device_vector<weight_type>();
	for (int i = 0; i < count; i++){
		is >>  value;
		this->device_deltas.push_back(value);
	}

	is >>  name;
	is >>  count;
	this->GPUOutput_values = thrust::device_vector<weight_type>();
	for (int i = 0; i < count; i++){
		is >>  value;
		this->GPUOutput_values.push_back(value);
	}

	is >>  name;
	is >>  count;
	this->GPUPreviousOutput_Values = thrust::device_vector<weight_type>();
	for (int i = 0; i < count; i++){
		is >>  value;
		this->GPUPreviousOutput_Values.push_back(value);
	}

	is >>  name;
	is >>  count;
	this->GPUPreviousWeights = thrust::device_vector<weight_type>();
	for (int i = 0; i < count; i++){
		is >>  value;
		this->GPUPreviousWeights.push_back(value);
	}

	is >>  name;
	is >>  count;
	this->mapTo = thrust::device_vector<int>();
	this->mapFrom = thrust::device_vector<int>();
	for (int i = 0; i < count; i++){
		is >>  count2;
		this->GPUMapFrom.push_back(count2);
		is >>  count2;
		this->GPUMapTo.push_back(count2);
	}

	is >>  name;
	is >>  count;
	this->positionToSum = thrust::device_vector<int>();
	for (int i = 0; i < count; i++){
		is >>  count2;
		this->positionToSum.push_back(count2);
	}

	is >>  name;
	is >>  count;
	this->count = thrust::device_vector<int>();
	for (int i = 0; i < count; i++){
		is >>  count2;
		this->count.push_back(count2);
	}

	is >>  name;
	is >>  count;
	this->GPUBias = thrust::device_vector<weight_type>();
	for (int i = 0; i < count; i++){
		is >>  value;
		this->GPUBias.push_back(value);
	}

	is >>  name;
	is >>  count;
	this->GPUPreviousBias = thrust::device_vector<weight_type>();
	for (int i = 0; i < count; i++){
		is >>  value;
		this->GPUPreviousBias.push_back(value);
	}



	
	return is;

}