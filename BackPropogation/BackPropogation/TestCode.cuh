#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#ifndef weight_type
#define weight_type double
#endif

using namespace std;
namespace testing{


	inline void testOutputToTarget(thrust::device_vector<weight_type> weights, thrust::device_vector<weight_type> values, thrust::device_vector<weight_type> output){

	}


	

	template <typename T>
	void outputToFile(thrust::device_vector<T> values, string title){
		outputToFile(values, title, "tests/test1.txt");
	}

	template <typename T>
	void outputToFile(thrust::device_vector<T> values, string title, string file_name){
		static bool opened_once = false;
		std::ofstream outputfile;
		outputfile.precision(30);
		if (opened_once){
			outputfile.open(file_name, std::ios::app | std::ios::ate);
		}
		else{
			outputfile.open(file_name, ios::trunc);
		}

		if (outputfile.is_open()){
			outputfile << title << endl;
			thrust::copy(values.begin(), values.end(), std::ostream_iterator<T>(outputfile, "\n"));
			outputfile << endl << endl;
		}
		outputfile.close();
		opened_once = true;

	}

	template <typename T, typename Iterator>
	void outputToFile(Iterator begin, int length, string title){
		outputToFile<T>(begin, length, title, "test/test2.txt");
	}

	template <typename T, typename Iterator>
	void outputToFile(Iterator begin, int length, string title, string file_name){
		static bool opened_once = false;
		std::ofstream outputfile;
		outputfile.precision(30);
		if (opened_once){
			outputfile.open(file_name, std::ios::app | std::ios::ate);
		}
		else{
			outputfile.open(file_name, ios::trunc);
		}

		if (outputfile.is_open()){
			outputfile << title << endl;
			thrust::copy(begin, begin + length, std::ostream_iterator<T>(outputfile, "\n"));
			outputfile << endl << endl;
		}
		outputfile.close();
		opened_once = true;

	}

	template <typename T>
	void outputArrayToFile(T* out, int size, string file_name){
		static bool opened_once = false;
		std::ofstream outputfile;
		outputfile.precision(30);
		if (opened_once){
			outputfile.open(file_name, std::ios::app | std::ios::ate);
		}
		else{
			outputfile.open(file_name, ios::trunc);
		}

		if (outputfile.is_open()){
			for (int i = 0; i < size; i++){
				outputfile << out[i] << ",";
			}

			outputfile << endl;

			outputfile.close();
			opened_once = true;
		}
	}

	template<typename T>
	void outputVectorToFile(std::vector<T> vec, string title, string file_name){
		static bool opened_once = false;
		std::ofstream outputfile;
		outputfile.precision(30);
		if (opened_once){
			outputfile.open(file_name, std::ios::app | std::ios::ate);
		}
		else{
			outputfile.open(file_name, ios::trunc);
		}

		if (outputfile.is_open()){
			outputfile << title << endl;
			std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(outputfile, "\n"));
			outputfile << endl << endl;
		}
		outputfile.close();
		opened_once = true;
	}
}