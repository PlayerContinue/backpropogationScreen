#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#define weight_type thrust::complex<double>
inline void testOutputToTarget(thrust::device_vector<weight_type> weights, thrust::device_vector<weight_type> values, thrust::device_vector<weight_type> output){
	for (int i = 0; i < weights.size(); i++){

	}
}