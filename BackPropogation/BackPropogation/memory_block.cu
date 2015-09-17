#include "memory_block.cuh"

Memory_Block::Memory_Block(){

}

Memory_Block::Memory_Block(unsigned int start, unsigned int numberInput){
	//Initialize the values
	this->input_weights = host_vector<weight_type>();
	this->output_weights = host_vector<weight_type>();
	this->forget_weights = host_vector<weight_type>();
	this->potential_memory_cell_value = host_vector<weight_type>();
	this->memory_cell_weights = this->getNewWeight();
	this->mapFrom = host_vector<int>();

	//Make the input weights
	for (int i = 0; i < numberInput; i++){
		this->input_weights.push_back(this->getNewWeight());
		this->output_weights.push_back(this->getNewWeight());
		this->forget_weights.push_back(this->getNewWeight());
		this->potential_memory_cell_value.push_back(this->getNewWeight());
		this->mapFrom.push_back(i + start);
	}
}

Memory_Block::Memory_Block(unsigned int input) :Memory_Block(0, input){
	
}

void Memory_Block::addNewConnection(int min, int max){
	
}

weight_type Memory_Block::getNewWeight(){
	return RandomClamped();
}
