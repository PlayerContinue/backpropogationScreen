#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <iostream>
#include <fstream>
#define weight_type thrust::complex<double>


namespace testing{
	inline void testOutputToTarget(thrust::device_vector<weight_type> weights, thrust::device_vector<weight_type> values, thrust::device_vector<weight_type> output){

	}

	template <typename T>
	void outputToFile(thrust::device_vector<T> values){
		static bool opened_once = false;
		std::ofstream outputfile;
		outputfile.precision(30);
		if (opened_once){
			outputfile.open("tests/test1.txt", std::ios::app);
		}
		else{
			outputfile.open("tests/test1.txt", ios::trunc);
		}

		if (outputfile.is_open()){
			thrust::copy(values.begin(), values.end(), std::ostream_iterator<T>(outputfile, "\n"));
			outputfile << endl << endl;
		}
		outputfile.close();
		opened_once = true;

	}
}