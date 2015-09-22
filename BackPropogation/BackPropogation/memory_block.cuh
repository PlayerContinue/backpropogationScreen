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
#define weight_type thrust::complex<double>
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
	weight_type memory_cell_weights;

	//Number of cells in the memory_block
	unsigned int number_memory_cells;

public:
	Memory_Block();
	Memory_Block(unsigned int input);
	//Start - The position of the first output from the previous layer
	//numberInput - the number of inputs
	Memory_Block(unsigned int start, unsigned int numberInput);

	//**************************
	//Add a New Input Connection
	//**************************

public:
	void addNewConnection(int min, int max);

private:
	weight_type getNewWeight();

	//**************************
	//Override
	//**************************
	friend ostream& operator<<(ostream& os, const Memory_Block& block){
		for (unsigned int i = 0; i < block.input_weights.size(); i++){
			os << block.input_weights[i] << ",";
		}
		os << endl;
		for (unsigned int i = 0; i < block.output_weights.size(); i++){
			os << block.output_weights[i] << ",";
		}
		os << endl;

		for (unsigned int i = 0; i < block.mapFrom.size(); i++){
			os << block.forget_weights[i] << ",";
		}
		os << endl;

		for (unsigned int i = 0; i < block.potential_memory_cell_value.size(); i++){
			os << block.potential_memory_cell_value[i] << ", ";
		}
		os << endl;

		os << block.memory_cell_weights;

		os << endl;

		

		for (unsigned int i = 0; i < block.mapFrom.size(); i++){
			os << block.mapFrom[i] << ",";
		}
		os << endl;
		

		return os;
	}
	friend istream& operator>>(istream& is, Memory_Block& network){

		return is;
	}


};