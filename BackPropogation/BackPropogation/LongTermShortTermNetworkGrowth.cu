#include "LongTermShortTermNetwork.cuh"

void LongTermShortTermNetwork::count_weights_in_layers(){
	this->count_weights_in_layers(false);
}

void LongTermShortTermNetwork::count_weights_in_layers(bool running){
	this->numberOfWeightsInLayers.clear();
	if (this->number_weights_by_type.size() < this->mBlocksLayers.size()){
		int length = this->number_weights_by_type.size();
		this->number_weights_by_type.resize(this->mBlocksLayers.size());
		this->number_nodes_by_type.resize(this->mBlocksLayers.size());
		this->number_nodes_in_layer.resize(this->mBlocksLayers.size() + 1);

		for (; length < this->number_weights_by_type.size(); length++){
			this->number_weights_by_type[length] = vector<int>(5);
			this->number_nodes_by_type[length] = vector<int>(5);
		}
	}
	else if (this->number_weights_by_type.size() > this->mBlocksLayers.size()){
		this->number_weights_by_type.resize(this->mBlocksLayers.size());
		this->number_nodes_by_type.resize(this->mBlocksLayers.size());
		this->number_nodes_in_layer.resize(this->mBlocksLayers.size() + 1);
	}

	std::fill(this->number_nodes_in_layer.begin(), this->number_nodes_in_layer.end(), (int)0);
	int i;
	//Count number of weights found in each layer
	for (i = 0; i < this->mBlocksLayers.size(); i++){
		this->numberOfWeightsInLayers.push_back(0);
		//Number Weights
		this->number_weights_by_type[i][MEMORY_CELL] = 0;
		this->number_weights_by_type[i][INPUT_CELL] = 0;
		this->number_weights_by_type[i][POTENTIAL_MEMORY_CELL] = 0;
		this->number_weights_by_type[i][OUTPUT_CELL] = 0;
		this->number_weights_by_type[i][FORGET_CELL] = 0;
		//Number Nodes
		this->number_nodes_by_type[i][MEMORY_CELL] = 0;
		this->number_nodes_by_type[i][INPUT_CELL] = 0;
		this->number_nodes_by_type[i][POTENTIAL_MEMORY_CELL] = 0;
		this->number_nodes_by_type[i][OUTPUT_CELL] = 0;
		this->number_nodes_by_type[i][FORGET_CELL] = 0;

		if (i != 0){
			this->number_nodes_in_layer[this->mBlocksLayers.size()] += this->number_nodes_in_layer[i - 1];
		}

		for (int j = 0; j < this->mBlocksLayers[i].size(); j++){
			this->numberOfWeightsInLayers[i] += this->mBlocksLayers[i][j].number_weights;

			//Add the different memory cell types

			this->number_weights_by_type[i][POTENTIAL_MEMORY_CELL] += this->mBlocksLayers[i][j].potential_memory_cell_value.size();
			this->number_nodes_in_layer[i] += 1;
			this->number_nodes_by_type[i][POTENTIAL_MEMORY_CELL] += 1;
			if (this->mBlocksLayers[i][j].getTypeOfMemoryBlock() == Memory_Block::memory_block_type::LAYER){
				this->number_weights_by_type[i][OUTPUT_CELL] += this->mBlocksLayers[i][j].output_weights.size();
				this->number_weights_by_type[i][FORGET_CELL] += this->mBlocksLayers[i][j].forget_weights.size();
				this->number_weights_by_type[i][INPUT_CELL] += this->mBlocksLayers[i][j].input_weights.size();

				this->number_nodes_by_type[i][OUTPUT_CELL] += 1;
				this->number_nodes_by_type[i][FORGET_CELL] += 1;
				this->number_nodes_by_type[i][MEMORY_CELL] += 1;
				this->number_nodes_by_type[i][INPUT_CELL] += 1;
				this->number_nodes_in_layer[i] += 4;
				if (running)
				{
					this->numberOfWeightsInLayers[i] += this->mBlocksLayers[i][j].number_memory_cells * NUMBER_MEM_CELL_WEIGHTS;
					this->number_weights_by_type[i][MEMORY_CELL] += NUMBER_WEIGHTS_TO_MEM;
					this->number_weights_by_type[i][OUTPUT_CELL] += 1;
					this->number_weights_by_type[i][FORGET_CELL] += 1;
					this->number_weights_by_type[i][INPUT_CELL] += 1;
					this->number_weights_by_type[i][POTENTIAL_MEMORY_CELL] += 1;
				}
			}
		}
	}
	if (i > 0){
		this->number_nodes_in_layer[this->mBlocksLayers.size()] += this->number_nodes_in_layer[i - 1];
	}


}


//***************************
//Modify Structure Of Neuron
//***************************

void LongTermShortTermNetwork::InitialcreateMemoryBlock(int numberMemoryCells){
	//Initialize the layers
	this->mBlocksLayers = vector<vector<Memory_Block>>();
	this->mBlocksLayers.push_back(vector<Memory_Block>());//Add one hidden layer
	this->mBlocksLayers.push_back(vector<Memory_Block>());//Add one output layer
	this->numberOfWeightsInLayers.resize(2);//set the size of the vector to the number of layer
	this->numberOfWeightsInLayers[0] = 0;//Set the number weights to zero
	this->numberOfWeightsInLayers[1] = 0;
	for (int i = 0; i < this->settings.i_output; i++){//Create the output layer
		this->mBlocksLayers[1].push_back(Memory_Block(numberMemoryCells + this->numberNonWeights, numberMemoryCells, this->settings.i_input, Memory_Block::OUTPUT));
		this->numberOfWeightsInLayers[1] += this->mBlocksLayers[1][i].number_weights;
	}



	//Create the rest of the nodes
	this->createMemoryBlock(numberMemoryCells, 0);
}

void LongTermShortTermNetwork::createMemoryBlock(int numberMemoryCells, int layer_num){
	for (int i = 0; i < numberMemoryCells; i++){
		this->mBlocksLayers[layer_num].push_back(Memory_Block(this->settings.i_input));
		this->numberOfWeightsInLayers[layer_num] += this->mBlocksLayers[layer_num][i].number_weights;
	}

}

void LongTermShortTermNetwork::transferCellToGPU(unsigned int &start_new,  unsigned int &start_of_weights_to_insert_on, unsigned int &start_of_nodes_to_insert_on, 
	unsigned int &number_new_added,
	unsigned int &number_new_added_total,
	unsigned int layer,
	thrust::device_vector<weight_type>::iterator &weight_iterator, 
	thrust::device_vector<int>::iterator &int_iterator,
	cell_type type,
	Memory_Block::cell_type memory_type
	){
	int size_to_add;
	for (unsigned int j = start_new; j < this->mBlocksLayers[layer].size(); j++){
		weight_iterator = this->GPUWeights.begin() + start_of_weights_to_insert_on;
		if (type != MEMORY_CELL){
			//Insert the new weights
			this->GPUWeights.insert(weight_iterator, this->mBlocksLayers[layer][j].weight_lists[memory_type].begin(), this->mBlocksLayers[layer][j].weight_lists[memory_type].end());
			weight_iterator += this->mBlocksLayers[layer][j].weight_lists[memory_type].size();
			//Insert weights to memory cell
			this->GPUWeights.insert(weight_iterator, thrust::make_constant_iterator((int)1), thrust::make_constant_iterator((int)1) + this->mBlocksLayers[layer][j].number_memory_cells);

			int_iterator = this->GPUMapFrom.begin() + start_of_weights_to_insert_on;
			//Insert where the value should come from
			this->GPUMapFrom.insert(int_iterator,
				this->mBlocksLayers[layer][j].mapFrom.begin(),
				this->mBlocksLayers[layer][j].mapFrom.end());

			int_iterator += this->mBlocksLayers[layer][j].number_memory_cells;


			//Temporary measure, will be altered to actual value later
			this->GPUMapFrom.insert(int_iterator,
				thrust::make_constant_iterator((int)-j),
				thrust::make_constant_iterator((int)-j) + this->mBlocksLayers[layer][j].number_memory_cells);

			//Insert where the values should go to
			int_iterator = this->GPUMapTo.begin() + start_of_weights_to_insert_on;
			this->GPUMapTo.insert(int_iterator,
				thrust::make_constant_iterator((int)this->GPUMapTo[start_of_weights_to_insert_on - 1] + 1),
				thrust::make_constant_iterator((int)this->GPUMapTo[start_of_weights_to_insert_on - 1] + 1)
				+ this->mBlocksLayers[layer][j].mapFrom.size() + this->mBlocksLayers[layer][j].number_memory_cells);

			weight_iterator = this->GPUBias.begin() + start_of_nodes_to_insert_on;
			this->GPUBias.insert(weight_iterator, this->mBlocksLayers[layer][j].getBias(memory_type));
		}
		else{

		}
		//Increase the stored values
		number_new_added += 1;//Note a new node has been added
		start_of_weights_to_insert_on += this->mBlocksLayers[layer][j].weight_lists[memory_type].size() + this->mBlocksLayers[layer][j].number_memory_cells;
		start_of_nodes_to_insert_on += 1;
		this->numberOfWeightsInLayers[layer] += this->mBlocksLayers[layer][j].weight_lists[memory_type].size() + this->mBlocksLayers[layer][j].number_memory_cells;
		this->number_weights_by_type[layer][type] += this->mBlocksLayers[layer][j].weight_lists[memory_type].size();
		this->number_nodes_by_type[layer][type] += 1;
		this->number_nodes_in_layer[layer] += 1;
		this->number_nodes_in_layer[this->number_nodes_in_layer.size() - 1] += 1;
		this->numberOfNodes += 1;
	}

	//Increment From to include the new values
	thrust::transform(this->GPUMapFrom.begin() + start_of_weights_to_insert_on, this->GPUMapFrom.end(), this->GPUMapFrom.begin() + start_of_weights_to_insert_on, functors::add_when_greater_than<int>(number_new_added, this->numberNonWeights));
	thrust::transform(this->GPUMapTo.begin() + start_of_weights_to_insert_on, this->GPUMapTo.end(), this->GPUMapTo.begin() + start_of_weights_to_insert_on, _1 + number_new_added);
	number_new_added_total += number_new_added;//Note n new nodes have been added
	number_new_added = 0;
}

void LongTermShortTermNetwork::addCellToGPU(unsigned int start_new, unsigned int layer){
	if (layer < this->mBlocksLayers.size()){
		thrust::device_vector<weight_type>::iterator weight_iterator;
		thrust::device_vector<int>::iterator int_iterator;
		unsigned int start_of_weights_to_insert_on = 0;
		unsigned int start_of_nodes_to_insert_on = 0;
		unsigned int number_new_added = 0;
		unsigned int number_new_added_total = 0;
		for (int i = 0; i < layer; i++){
			start_of_weights_to_insert_on += this->numberOfWeightsInLayers[i];
			start_of_nodes_to_insert_on += this->number_nodes_in_layer[i];
		}


		

		for (int type_cell = INPUT_CELL, int memory_type = Memory_Block::cell_type::INPUT_CELL; type_cell != MEMORY_CELL && memory_type != Memory_Block::cell_type::MEMORY_CELL; type_cell++, memory_type++){
			//Add New Input Nodes
			start_of_weights_to_insert_on += this->number_weights_by_type[layer][type_cell];
			start_of_nodes_to_insert_on += this->number_nodes_by_type[layer][type_cell];
			this->transferCellToGPU(start_new, start_of_weights_to_insert_on, start_of_nodes_to_insert_on,
				number_new_added, number_new_added_total, layer, weight_iterator, int_iterator, (cell_type)type_cell, (Memory_Block::cell_type)memory_type);
		
			
			
		}


		this->GPUPreviousBias.resize(this->GPUBias.size());
		this->GPUPreviousTemp.resize(this->GPUPreviousTemp.size() + number_new_added_total);
		this->GPUPreviousOutput_Values.resize(this->GPUPreviousOutput_Values.size() + number_new_added_total);
		this->GPUPreviousWeights.resize(this->GPUWeights.size());
		this->device_deltas.resize(this->device_deltas.size() + number_new_added_total);
		//Add the new outputs to each row
		this->GPUOutput_values.resize(this->GPUOutput_values.size() + (number_new_added_total*this->total_number_of_unrolled));
	}

}

void LongTermShortTermNetwork::addNeuron(int numberNeuronsToAdd){
	int start_value = this->mBlocksLayers[0].size();
	this->createMemoryBlock(numberNeuronsToAdd, 0);
	this->addCellToGPU(start_value, 0);
}



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
	for (unsigned int k = this->numberNonWeights; k < this->bias.size() - this->settings.i_output; k++){
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
	for (int i = 0; i < numberWeightsToAdd; i++){
		if (decideTo != -1){
			this->mBlocksLayers[0][decideTo].addNewConnection(this->settings.i_input + this->mBlocksLayers[0].size(), this->settings.i_input + (2 * this->mBlocksLayers[0].size()));
		}
		else{
			break;
		}
		decideTo = this->decideNodeToAttachTo();
	}

}






