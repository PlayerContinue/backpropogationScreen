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
	this->type = type;//Set the type of memory block this is
	if (type == LAYER){//Output layer does not require this, as it is only a set of input	
		this->memory_cell_weights.push_back(this->getNewWeight());
	}
	this->number_memory_cells = 1;
	this->mapFrom = host_vector<int>();
	//Add weights which connect from the input nodes to the output nodes
	for (int i = 0; i < numberInput; i++){

		
		if (type==LAYER){//Only add these if the current node is in a layer which is not an output
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



void Memory_Block::addNewConnection(int min, int max){
	
}

weight_type Memory_Block::getNewWeight(){
	return RandomClamped();
}

//Return the type of node it is
Memory_Block::memory_block_type Memory_Block::getTypeOfMemoryBlock(){
	return this->type;
}