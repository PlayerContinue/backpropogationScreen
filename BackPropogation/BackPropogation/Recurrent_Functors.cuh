#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <vector>
#ifndef weight_type
#define weight_type thrust::complex<double>
#endif

using namespace thrust;
using namespace thrust::placeholders;

namespace functors{
	//Multiply two values
	template <typename T>
	struct multiply : public thrust::unary_function < T, T > {

		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple x) const{
			return ((T)thrust::get<0>(x) * (T)thrust::get<1>(x));
		}

	};

	template <typename T>
	struct subtract_tuple : public thrust::unary_function < T, T > {

		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(const Tuple &x){
			return (thrust::get<0>(x) -thrust::get<1>(x));
		}
	};

	template <typename T>
	struct run_memory_block_functon : public thrust::unary_function < T, T > {


		template <typename Tuple>
		__host__ __device__
			void operator()(Tuple &x){//Received Tuple is in the form input, output, forget, potential memory cell, memory cell value
			weight_type memory_value = sigmoid_function(thrust::get<0>(x) * thrust::get<3>(x));//Multiply the input by the potential_memory_value

			weight_type forget = (weight_type)thrust::get<2>(x);

			thrust::get<2>(x) = sigmoid_function((weight_type)thrust::get<2>(x) * (weight_type)thrust::get<4>(x)); //Multiply the forget * the old memory cell value
			thrust::get<4>(x) = thrust::get<4>(x) +memory_value + forget; //Sum the forget,input, and old cell value to get the new value the new potential memory cell value
			thrust::get<1>(x) = sigmoid_function((weight_type)thrust::get<4>(x) * (weight_type)thrust::get<1>(x)); //Multiply the new memory_cell value by the new output value 

		}

		__host__ __device__
			weight_type sigmoid_function(weight_type value){
			return (weight_type)1 / ((weight_type)1 + thrust::exp(((weight_type)-1 * value)));
		}

	};

	//Perform Sigmoid Operation of a Tuple
	template <typename T>
	struct sigmoid_tuple_functor : public thrust::unary_function < T, T > {

		//Overload the function operator
		template <typename Tuple>

		__host__ __device__
			T operator()(Tuple x) const{
			T z = (T)((T)thrust::get<0>(x)*(T)thrust::get<1>(x));
			z = thrust::exp(((T)-1) * z);
			return (T)1 / ((T)1 + z);
		}

	};




	//Perform a sigmoid function
	template <typename T>
	struct sigmoid_functor : public thrust::unary_function < T, T > {
		sigmoid_functor(){};

		__host__ __device__
			T operator()(const T &x) const{
			return ((T)1 / ((T)1 + thrust::exp(((T)-1) * x)));
		}

	};



	//Increase amount when beyond a certain numbers
	template <typename T>
	struct extend_value : public thrust::unary_function < T, T > {
		const T length;
		extend_value(T _length):length(_length){};

		template <typename Tuple>
		__host__ __device__
			T operator()(const Tuple &x) const{
			return ((thrust::get<0>(x)/length)*length) + thrust::get<1>(x);
		}

	};



	template <typename T>
	struct find_error : public thrust::unary_function < T, T > {

		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple &x) const{
			return thrust::pow((thrust::get<0>(x) -thrust::get<1>(x)), (T)2);
		}

	};

	//sum a value
	template <typename T>
	struct add_constant_value : public thrust::unary_function < T, T > {
		const T c;
		const unsigned int input;
		add_constant_value() : c(0), input(0){};

		add_constant_value(T _c, unsigned int _input) :c(_c), input(_input){};


		__host__ __device__
			T operator()(const T &x)const{

			if (x >= input){//The value is not an input
				return ((T)x + (T)c);
			}
			else{//The value is an input
				return x;
			}
		}


	};

	struct compare_plus : public thrust::unary_function < int, int > {
		int input;
		int numberIncrease;//Number to increase the number by 
		compare_plus(int max_input, int numberIncrease){
			this->input = max_input;
			this->numberIncrease = numberIncrease;
		}


		__host__ __device__
			int operator()(int &x) const{
			if (x < this->input){//Returns this directly, as it is an input
				return x;
			}
			else{
				return (x + numberIncrease);
			}
		}
	};

	template <typename T>
	struct add_one_when_equal_to : public thrust::unary_function < T, T > {
		const T equal_to;
		const T divide_by;
		add_one_when_equal_to(T _divide_by, T _equal_to) :equal_to(_equal_to), divide_by(_divide_by){}
		__host__ __device__
			T operator()(const T &x){
			if (x >= equal_to){
				return  x;
			}
			return (x / divide_by);
		}

	};

	//Function is _add_to, _greater_than_this
	template <typename T>
	struct add_when_greater_than : public thrust::unary_function < T, T > {
		const T greater_than_this;
		const T add_to;
		add_when_greater_than(T _add_to, T _greater_than_this) :greater_than_this(_greater_than_this), add_to(_add_to){}
		__host__ __device__
			T operator()(const T &x){
			if (x >= _equal_to){
				return (x + add_to);
			}
			else{
				return x;
			}
		}

	};


	template <typename T>
	struct find_previous_weight : public thrust::unary_function < T, T > {
		const T beta;
		find_previous_weight( T _beta) :  beta(_beta){

		}

		template <typename Tuple>
		__host__ __device__ //Delta,output
			T operator()(Tuple &x){
			//Multiply beta * output * delta
			return beta * thrust::get<0>(x) * thrust::get<1>(x);
		}


	};

	//Apply the error from the delta and the weight
	template <typename T>
	struct apply_new_error : public thrust::binary_function < T, T, T > {
		const T alpha;
		const T divide;
		apply_new_error(T _alpha, T _divide) :alpha(_alpha), divide(_divide){

		}
		template <typename Tuple>
		__host__ __device__ //previousWeight, weight
			T operator()(Tuple &x,T &y){
			thrust::get<1>(x) *= alpha;
			if (y != (T)1){//Don't change any weights which are going to or from a memory node. These need to remain as 1
				T toReturn = thrust::get<1>(x) +y;
				thrust::get<1>(x) = ((T)thrust::get<0>(x) / (T)divide);
				return toReturn + thrust::get<1>(x);
			}
			else{
				return (T)y;
			}
			
		}

	};

	//Apply the error from the delta and the weight
	template <typename T>
	struct apply_error : public thrust::binary_function < T, T, T > {
		const T alpha;
		const T beta;
		const T divide;
		apply_error(T _alpha, T _beta, T _divide) : alpha(_alpha), beta(_beta), divide(_divide){

		}

		//w = weight, d = delta, beta * (d/(number summed) + (w + (w*alpha)
		__host__ __device__
			T operator()(const T &d, const T &w)const{
			return (beta * (d / divide)) + (w + (w*alpha));
		}

		template <typename Tuple>
		__host__ __device__ //Delta,Weight,PreviousWeight,output
			T operator()(Tuple &x){
			//Get the new weight (essentially apply the learning rate/momentum
			thrust::get<1>(x) += (T)alpha * (T)thrust::get<2>(x);

			//Find the previous weight
			//delta * beta * output
			thrust::get<2>(x) = (T)thrust::get<0>(x) * (T)beta * (T)thrust::get<3>(x);

			//Return the new weight
			//Found by adding the current weight to the previous weight
			return ((T)thrust::get<1>(x) +(T)thrust::get<3>(x));


		}
	};

	template <typename T>
	struct find_non_output_delta : public thrust::unary_function < T, T > {


		find_non_output_delta(){};

		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple &t){
			return (T)thrust::get<0>(t) * ((T)1 - (T)thrust::get<0>(t)) * (T)thrust::get<1>(t);

		}



	};

	template <typename T>
	struct find_output_delta : public thrust::unary_function < T, T > {
		find_output_delta(){};
		__host__ __device__
			T operator()(const T &target, const T &output)const{
			return output*((T)1 - output)*(target - output);
		}
	};
}