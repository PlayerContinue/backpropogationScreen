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
	if (type == LAYER){//Output layer does not require this, as it is only a set of input
		
		this->memory_cell_weights.push_back(this->getNewWeight());
	}
	this->number_memory_cells = 1;
	this->mapFrom = host_vector<int>();
	//Make the input weights
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
	}
}



void Memory_Block::addNewConnection(int min, int max){
	
}

weight_type Memory_Block::getNewWeight(){
	return RandomClamped();
}


Memory_Block::memory_block_type Memory_Block::getTypeOfMemoryBlock(){
	if (this->memory_cell_weights.size() > 0){
		return LAYER;
	}
	else{
		return OUTPUT;
	}
}