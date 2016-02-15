/*
Programmer: David Greenberg
Reason : Contains functions which can be called by thrust functions

*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/complex.h>
namespace functors{
	namespace transform_functors{

		using namespace thrust;
		using namespace thrust::placeholders;



		__host__ __device__
			inline double sigmoid_function(const double &value){
			thrust::complex<double> exped = thrust::exp(((thrust::complex<double>) ((thrust::complex<double>) - 1 * (thrust::complex<double>)value)));
			return (double)1 / ((double)1 + (double)exped.real());
		};


		__host__ __device__
			inline double logistic_function(const double &value, const double &max_value, const double &midpoint){
			return ((thrust::complex<double>)max_value /
				((thrust::complex<double>)1 + thrust::exp((thrust::complex<double>)(-1) * ((thrust::complex<double>)value - (thrust::complex<double>)midpoint)))).real();
		};


		//Multiply a two object tuple
		template <typename T>
		struct multiplies : public thrust::unary_function < T, T > {

			//Overload the function operator
			template <typename Tuple>
			__host__ __device__
				T operator()(const Tuple &x) const{
				return ((T)thrust::get<0>(x) * (T)thrust::get<1>(x));
			}

			//Overload the function operator
			template <typename Tuple>
			__host__ __device__
				T operator()(const T &x, const T &y) const{
				return ((T)x * (T)y);
			}

		};


		//Multiply a two object tuple
		template <typename T>
		struct subtract : public thrust::unary_function < T, T > {

			//Overload the function operator
			template <typename Tuple>
			__host__ __device__
				T operator()(const Tuple &x) const{
				return ((T)thrust::get<0>(x) - (T)thrust::get<1>(x));
			}

		};

		//Multiply a two object tuple
		template <typename T>
		struct sum : public thrust::unary_function < T, T > {

			//Overload the function operator
			template <typename Tuple>
			__host__ __device__
				T operator()(const Tuple &x) const{
				return ((T)thrust::get<0>(x)  + (T)thrust::get<1>(x));
			}

		};



		//Add two numbers and sigmoid the results
		template <typename T>
		struct bias_sigmoid_functor : public thrust::unary_function < T, T > {

			//Overload the function operator
			__host__ __device__
				T operator()(const T x, const T y) const{
				return transform_functors::sigmoid_function(x + y);
			}

		};

		//Find the error of the provided two values
		template <typename T>
		struct find_output_gradiant : public thrust::unary_function < T, T > {

			//Overload the function operator
			__host__ __device__
				T operator()(const T output, const T target) const{
				//thrust::complex<T> temp = thrust::pow((thrust::complex<T>)(x - y), (thrust::complex<T>)2);
				//return temp.real();
				return ((T)output)*((T)1 - (T)output)*((T)target - (T)output);

			}

			//Overload the function operator
			template <typename Tuple>
			__host__ __device__
				T operator()(const Tuple t) const{// 0= output, 1 = target
				return ((T)thrust::get<0>(t))*((T)1 - (T)thrust::get<0>(t))*((T)thrust::get<1>(t) -(T)thrust::get<0>(t));

			}

		};

		//Find the error of the hidden layers
		template <typename T>
		struct find_hidden_node_gradiant : public thrust::unary_function < T, T > {

			//Overload the function operator
			template <typename Tuple>
			__host__ __device__
				T operator()(const Tuple t) const{//0=error of next layer, 1=weight

				return  (T)thrust::get<0>(t) * ((T)1 - (T)thrust::get<0>(t)) * (T)thrust::get<1>(t);

			}

		};

		//Find the gradiant of the weights with respect to a given node
		template <typename T>
		struct find_weight_gradiant : public thrust::unary_function < T, T > {

			//Overload the function operator
			template <typename Tuple>
			__host__ __device__
				T operator()(const Tuple t) const{//0=weight_error 1=output of previous layer 2=from 3=to

				if (thrust::get<2>(t) == thrust::get<3>(t)){
					return thrust::get<0>(t) +thrust::get<1>(t);
				}
				else{
					return thrust::get<0>(t);
				}

			}

		};


	}
}