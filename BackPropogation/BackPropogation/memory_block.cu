#include "memory_block.cuh"

Memory_Block::Memory_Block(){

}

Memory_Block::Memory_Block(unsigned int input) :Memory_Block(0, input){

}

Memory_Block::Memory_Block(unsigned int start, unsigned int numberInput):Memory_Block(start,numberInput,LAYER){
	
}

Memory_Block::Memory_Block(unsigned int start, unsigned int numberInput, memory_block_type type){
	//Initialize the values
	this->input_weights = host_vector<weight_type>();
	this->output_weights = host_vector<weight_type>();
	this->forget_weights = host_vector<weight_type>();
	this->potential_memory_cell_value = host_vector<weight_type>();
	this->memory_cell_weights = host_vector<weight_type>();
	this->bias = host_vector<weight_type>();
	this->number_weights = 0;
	this->number_inputs = numberInput;
	this->type = type;//Set the type of memory block this is
	if (type == LAYER){//Output layer does not require this, as it is only a set of input	
		this->memory_cell_weights.push_back(this->getNewWeight());
		this->number_memory_cells = 1;
	}
	else{
		this->number_memory_cells = 0;
	}
	this->mapFrom = host_vector<int>();
	
	setInitialWeights(start, numberInput, type);
	createStorage();
}

Memory_Block::Memory_Block(unsigned int start, unsigned int numberInput, unsigned int extra_at_start, memory_block_type type){

	//Initialize the values
	this->input_weights = host_vector<weight_type>();
	this->output_weights = host_vector<weight_type>();
	this->forget_weights = host_vector<weight_type>();
	this->potential_memory_cell_value = host_vector<weight_type>();
	this->memory_cell_weights = host_vector<weight_type>();
	this->bias = host_vector<weight_type>();
	this->number_weights = 0;
	this->number_inputs = numberInput;
	this->type = type;//Set the type of memory block this is
	
	if (type == LAYER){//Output layer does not require this, as it is only a set of input	
		this->memory_cell_weights.push_back(this->getNewWeight());
		this->number_memory_cells = 1;
	}
	else{
		this->number_memory_cells = 0;
	}

	this->mapFrom = host_vector<int>();
	//setInitialWeights(0, extra_at_start,type);
	setInitialWeights(start, numberInput, type);
	
	createStorage();
}

void Memory_Block::createStorage(){
	this->weight_lists = vector<thrust::host_vector<weight_type>>(5);
	this->weight_lists[cell_type::INPUT_CELL] = this->input_weights;
	this->weight_lists[cell_type::OUTPUT_CELL] = this->output_weights;
	this->weight_lists[cell_type::MEMORY_CELL] = this->memory_cell_weights;
	this->weight_lists[cell_type::FORGET_CELL] = this->forget_weights;
	this->weight_lists[cell_type::POTENTIAL_MEMORY_CELL] = this->potential_memory_cell_value;
}

void Memory_Block::setInitialWeights(int start, int numberInput, memory_block_type type){
	//Add weights which connect from the input nodes to the output nodes
	for (int i = 0; i < numberInput; i++){
		if (type == LAYER){//Only add these if the current node is in a layer which is not an output
			this->input_weights.push_back(this->getNewWeight());
			this->output_weights.push_back(this->getNewWeight());
			this->forget_weights.push_back(this->getNewWeight());

		}
		//If the layer is an output, it needs both a map from where the ouput is
		//and a cell for containing the output
		this->potential_memory_cell_value.push_back(this->getNewWeight());
		this->mapFrom.push_back(i + start);
		if (type == LAYER){
			number_weights += 4;//Increment the number of weights in the list
		}
		else if (type == OUTPUT){
			number_weights += 1;
		}
	}

	//Create Biases for each node
	//The biases are currently randomly chosen, but may change on future iterations
	//4 is the number of non-memory-cell nodes, memory cells have a bias of 0
	if (type == LAYER){
		for (int i = 0; i < 4; i++){
			this->bias.push_back(this->getNewWeight());
		}
	}
	else if (type == OUTPUT){
		this->bias.push_back(this->getNewWeight());
	}
}

weight_type Memory_Block::getBias(cell_type type){
	switch (type){
	case INPUT_CELL:
		return this->bias[0];
	case OUTPUT_CELL:
		return this->bias[1];
	case FORGET_CELL:
		return this->bias[2];
	case POTENTIAL_MEMORY_CELL:
		return this->bias[3];
	case MEMORY_CELL:
		return 0;
	default:
		return -1;
	}
}



void Memory_Block::addNewConnection(int pos){
	this->mapFrom.push_back(pos);
	
	this->potential_memory_cell_value.push_back(this->getNewWeight());
	this->weight_lists[POTENTIAL_MEMORY_CELL].push_back(this->potential_memory_cell_value[this->potential_memory_cell_value.size() - 1]);
	if (this->type != OUTPUT){
		this->input_weights.push_back(this->getNewWeight());
		this->output_weights.push_back(this->getNewWeight());
		this->forget_weights.push_back(this->getNewWeight());
		this->weight_lists[INPUT_CELL].push_back(this->input_weights[this->input_weights.size() - 1]);
		this->weight_lists[OUTPUT_CELL].push_back(this->output_weights[this->output_weights.size() - 1]);
		this->weight_lists[FORGET_CELL].push_back(this->forget_weights[this->forget_weights.size() - 1]);
		this->number_weights += 4;
	}
	else{
		this->number_weights += 1;
	}
	
}
void Memory_Block::addNewConnection(int min, int max){
	bool mappedFrom = false;
	for (int i = min; i < max; i++){
		mappedFrom = false;
		for (int j = 0; j < this->mapFrom.size(); j++){
			if (this->mapFrom[j] == i){
				mappedFrom = true;
			}
		}

		if (!mappedFrom){
			addNewConnection(i);
			break;
		}
	}
}

bool Memory_Block::removeConnection(int toRemove){
	

	if (toRemove < this->potential_memory_cell_value.size()){
		if (this->type == LAYER){
			for (unsigned int start = INPUT_CELL; start <= MEMORY_CELL; start++){
				if (start != POTENTIAL_MEMORY_CELL){
					this->weight_lists[start].erase(this->weight_lists[start].begin() + toRemove);
				}
			}
			this->input_weights.erase(this->input_weights.begin() + toRemove);
			this->output_weights.erase(this->output_weights.begin() + toRemove);
			this->forget_weights.erase(this->forget_weights.begin() + toRemove);
		}
		this->potential_memory_cell_value.erase(this->potential_memory_cell_value.begin() + toRemove);
		this->weight_lists[POTENTIAL_MEMORY_CELL].erase(this->weight_lists[POTENTIAL_MEMORY_CELL].begin() + toRemove);
		this->mapFrom.erase(this->mapFrom.begin() + toRemove);
		this->number_weights--;
		return true;
	}
	else{
		return false;
	}
}

void Memory_Block::incrementFromPosition(int add){
	this->incrementFromPosition(add, 0);
}
void Memory_Block::incrementFromPosition(int add, int add_from){
	thrust::transform(this->mapFrom.begin() + add_from, this->mapFrom.end(), this->mapFrom.begin() + add_from, _1 + add);
}

weight_type Memory_Block::getNewWeight(){
	return RandomClamped();
}

//Return the type of node it is
Memory_Block::memory_block_type Memory_Block::getTypeOfMemoryBlock(){
	return this->type;
}