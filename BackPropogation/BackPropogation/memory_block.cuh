#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#ifndef __TIME_H_INCLUDED__
#include <time.h>
#define __TIME_H_INCLUDED__
#endif

#ifdef __IOSTREAM_H_INCLUDED__

#else
#include <iostream>
#define  __IOSTREAM_H_INCLUDED__
#endif

#include "util.h"
#ifndef weight_type
#define weight_type double
#endif
using namespace thrust;
class Memory_Block{
public:
	//Contains the weights and input output of the input weights
	host_vector<weight_type> input_weights;
	host_vector<int> mapFrom;

	//contains the weights and input output of the output
	host_vector<weight_type> output_weights;

	//Contains information about forget cells
	host_vector<weight_type> forget_weights;

	host_vector<weight_type> potential_memory_cell_value;

	//Contains the information about the memory cell
	host_vector<weight_type> memory_cell_weights;

	//Contains Bias Information For Each Node
	host_vector<weight_type> bias;

	//Number of cells in the memory_block
	unsigned int number_memory_cells;

	//Number of weight in block
	unsigned int number_weights;

	//Number Of Inputs
	unsigned int number_inputs;

	enum memory_block_type { OUTPUT, LAYER };

private:
	memory_block_type type;

public:
	Memory_Block();
	Memory_Block(unsigned int input);
	//Start - The position of the first output from the previous layer
	//numberInput - the number of inputs
	Memory_Block(unsigned int start, unsigned int numberInput);
	Memory_Block(unsigned int start, unsigned int numberInput, memory_block_type type);

	//**************************
	//Add a New Input Connection
	//**************************

public:
	void addNewConnection(int min, int max);

	memory_block_type getTypeOfMemoryBlock();

private:
	weight_type getNewWeight();

	//**************************
	//Override
	//**************************
	friend ostream& operator<<(ostream& os, const Memory_Block& block){
		os << block.type << endl;
		os << block.potential_memory_cell_value.size() << endl;
		if (block.type == Memory_Block::LAYER){
			for (unsigned int i = 0; i < block.input_weights.size(); i++){
				os << block.input_weights[i] << " ";
			}
			os << endl;
			for (unsigned int i = 0; i < block.output_weights.size(); i++){
				os << block.output_weights[i] << " ";
			}
			os << endl;

			for (unsigned int i = 0; i < block.forget_weights.size(); i++){
				os << block.forget_weights[i] << " ";
			}
			os << endl;
		} 
		
		for (unsigned int i = 0; i < block.potential_memory_cell_value.size(); i++){
			os << block.potential_memory_cell_value[i] << " ";
		}
		
		
		if (block.type == Memory_Block::LAYER){
			os << endl;
			os << block.memory_cell_weights.size() << endl;
			for (unsigned int i = 0; i < block.memory_cell_weights.size(); i++){
				os << block.memory_cell_weights[i] << " ";
			}
		}


		os << endl;
		os << block.mapFrom.size() << endl;
		for (unsigned int i = 0; i < block.mapFrom.size(); i++){
			os << block.mapFrom[i] << " ";
		}

		os << endl;

		os << block.bias.size()<<endl;
		std::copy(block.bias.begin(), block.bias.end(), std::ostream_iterator<weight_type>(os, " "));

		os << endl;

		return os;
	}
	friend istream& operator>>(istream& is, Memory_Block& block){
		int count;
		double value;
		
		block.input_weights = host_vector<weight_type>();
		block.output_weights = host_vector<weight_type>();
		block.forget_weights = host_vector<weight_type>();
		block.potential_memory_cell_value = host_vector<weight_type>();
		block.memory_cell_weights = host_vector<weight_type>();
		block.bias = host_vector<weight_type>();
		block.number_weights = 0;
		is >> std::skipws >> count;
		block.type = (Memory_Block::memory_block_type) count;
		is >> std::skipws >> count;
		block.number_inputs = count;
		//input weights
		if (block.type == Memory_Block::LAYER){
			for (unsigned int i = 0; i < count; i++){
				is >> std::skipws >> value;
				block.input_weights.push_back(value);
				block.number_weights++;
			}

			for (unsigned int i = 0; i < count; i++){
				is >> std::skipws >> value;
				block.output_weights.push_back(value);
				block.number_weights++;
			}


			for (unsigned int i = 0; i < count; i++){
				is >> std::skipws >> value;
				block.forget_weights.push_back(value);
				block.number_weights++;
			}
		}

		for (unsigned int i = 0; i < count; i++){
			is >> std::skipws >> value;
			block.potential_memory_cell_value.push_back(value);
			block.number_weights++;
		}
		if (block.type == Memory_Block::LAYER){
			is >> count;
			for (unsigned int i = 0; i < count; i++){
				is >> std::skipws >> value;
				block.memory_cell_weights.push_back(value);


			}
		}

		int map;
		is >> count;
		for (unsigned int i = 0; i < count; i++){
			is >> std::skipws >> map;
			block.mapFrom.push_back(map);
		}

		
		is >> count;
		for (unsigned int i = 0; i < count; i++){
			is >> std::skipws >> value;
			block.bias.push_back(value);
		}
		return is;
	}


};