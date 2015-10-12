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
#define weight_type double
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
			T operator()(const Tuple &x) const{
			return ((T)thrust::get<0>(x) * (T)thrust::get<1>(x));
		}

	};

	//Multiply two values
	//0 <, 1 >, 2 <=, 3 >=, 4 ==
	template <unsigned int pos_in_tuple,unsigned int return_on_fail_pos, typename T>
	struct multiply_if : public thrust::unary_function < T, T > {
		const int type_of_if;
		const T compare_to;
			multiply_if() :type_of_if(100),compare_to((T)0){};
			multiply_if( int _type_of_if, T _compare_to) :type_of_if(_type_of_if),compare_to(_compare_to){};


		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple x) const{
			bool perform_op = false;
			switch (type_of_if){
			case 0:
				if (thrust::get<pos_in_tuple>(x) < compare_to){
					perform_op = true;
				}
				break;
			case 1:
				if (thrust::get<pos_in_tuple>(x) > compare_to){
					perform_op = true;
				}
				break;
			case 2:
				if (thrust::get<pos_in_tuple>(x) <= compare_to){
					perform_op = true;
				}
				break;
			case 3:
				if (thrust::get<pos_in_tuple>(x) >= compare_to){
					perform_op = true;
				}
				break;
			case 4:
				if (thrust::get<pos_in_tuple>(x) == compare_to){
					perform_op = true;
				}
				break;
			default:
				perform_op = true;
				break;
			}
			if (perform_op){
				return ((T)thrust::get<0>(x) * (T)thrust::get<1>(x));
			}
			else{
				return thrust::get<return_on_fail_pos>(x);
			}
		}

	};

	template <unsigned int pos_in_tuple,  typename T>
	struct compare_two : public thrust::unary_function < bool, T > {
		const int type_of_if;
		const T compare_to;
		compare_two() :type_of_if(100), compare_to((T)0){};
		compare_two(int _type_of_if, T _compare_to) :type_of_if(_type_of_if), compare_to(_compare_to){};

		//Overload the function operator
		__host__ __device__
			T operator()(const T &x) const{
			bool perform_op = false;
			switch (type_of_if){
			case 0:
				if (x < compare_to){
					perform_op = true;
				}
				break;
			case 1:
				if (x > compare_to){
					perform_op = true;
				}
				break;
			case 2:
				if (x <= compare_to){
					perform_op = true;
				}
				break;
			case 3:
				if (x >= compare_to){
					perform_op = true;
				}
				break;
			case 4:
				if (x == compare_to){
					perform_op = true;
				}
				break;
			case 5:
				if (x != compare_to){
					perform_op = true;
				}
				break;
			default:
				perform_op = true;
				break;
			}
			return perform_op;
		}

		//Overload the function operator
		__host__ __device__
			T operator()(const T &y,const T &x) const{
			bool perform_op = false;
			switch (type_of_if){
			case 0:
				if (x < compare_to){
					perform_op = true;
				}
				break;
			case 1:
				if (x > compare_to){
					perform_op = true;
				}
				break;
			case 2:
				if (x <= compare_to){
					perform_op = true;
				}
				break;
			case 3:
				if (x >= compare_to){
					perform_op = true;
				}
				break;
			case 4:
				if (x == compare_to){
					perform_op = true;
				}
				break;
			case 5:
				if (x != compare_to){
					perform_op = true;
				}
				break;
			default:
				perform_op = true;
				break;
			}
			return perform_op;
		}

		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple x) const{
			bool perform_op = false;
			switch (type_of_if){
			case 0:
				if (thrust::get<pos_in_tuple>(x) < compare_to){
					perform_op = true;
				}
				break;
			case 1:
				if (thrust::get<pos_in_tuple>(x) > compare_to){
					perform_op = true;
				}
				break;
			case 2:
				if (thrust::get<pos_in_tuple>(x) <= compare_to){
					perform_op = true;
				}
				break;
			case 3:
				if (thrust::get<pos_in_tuple>(x) >= compare_to){
					perform_op = true;
				}
				break;
			case 4:
				if (thrust::get<pos_in_tuple>(x) == compare_to){
					perform_op = true;
				}
				break;
			case 5:
				if (thrust::get<pos_in_tuple>(x) != compare_to){
					perform_op = true;
				}
				break;
			default:
				perform_op = true;
				break;
			}
			return perform_op;
		}

	};
	

	//Add Two Values if true
	//0 <, 1 >, 2 <=, 3 >=, 4 ==
	template <unsigned int pos_in_tuple, unsigned int return_on_fail_pos, typename T>
	struct add_if : public thrust::unary_function < T, T > {
		const int type_of_if;
		const T compare_to;
		add_if() :type_of_if(100), compare_to((T)0){};
		add_if(int _type_of_if, T _compare_to) :type_of_if(_type_of_if), compare_to(_compare_to){};


		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple x) const{
			bool perform_op = false;
			switch (type_of_if){
			case 0:
				if (thrust::get<pos_in_tuple>(x) < compare_to){
					perform_op = true;
				}
				break;
			case 1:
				if (thrust::get<pos_in_tuple>(x) > compare_to){
					perform_op = true;
				}
				break;
			case 2:
				if (thrust::get<pos_in_tuple>(x) <= compare_to){
					perform_op = true;
				}
				break;
			case 3:
				if (thrust::get<pos_in_tuple>(x) >= compare_to){
					perform_op = true;
				}
				break;
			case 4:
				if (thrust::get<pos_in_tuple>(x) == compare_to){
					perform_op = true;
				}
				break;
			default:
				perform_op = true;
				break;
			}
			if (perform_op){
				return ((T)thrust::get<0>(x) + (T)thrust::get<1>(x));
			}
			else{
				return thrust::get<return_on_fail_pos>(x);
			}
		}

	};

	template < typename T>
	struct add : public thrust::unary_function < T, T > {
		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple x) const{
			return ((T)thrust::get<0>(x) + (T)thrust::get<1>(x));
		}


	};

	template<typename T, typename M>
	struct bin_add:public thrust::binary_function < T, M, T > {

		template <typename Tuple>
		__host__ __device__
			T operator()(const T &x,const M &y) const{
			return ((T)x +(T)y);
		}


	};


	template < typename T>
	struct add_and_store : public thrust::unary_function < T, T > {
		const T divide;
		add_and_store(T _divide):divide(_divide){};
		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple x){
			thrust::get<1>(x) = thrust::get<1>(x) /divide; 
			return ((T)thrust::get<0>(x) + (T)thrust::get<1>(x));
		}

	};

	

	//Multiply the first two values in a tuple, then add the last one
	template < typename T>
	struct multiply_add : public thrust::unary_function < T, T > {
		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple x){
			return ((T)thrust::get<0>(x) * (T)thrust::get<1>(x)) + (T)thrust::get<2>(x);
		}

	};



	//Multiply a value by a constant
	template <typename T>
	struct multiply_by_constant : public thrust::unary_function < T, T > {
		const T constant;
		multiply_by_constant(T _constant) : constant(_constant){};
		

		//Overload the function operator
		__host__ __device__
			T operator()(const T &x) const{
			return constant * x;
		}

		

	};



	template <typename T>
	struct add_bias : public thrust::binary_function < T, T, T > {
		const bool current_layer;
		add_bias(bool _current_layer) : current_layer(_current_layer){};
		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(T &bias, Tuple &x){
			if (thrust::get<1>(x) != 0){
				return (bias * (T)thrust::get<0>(x)) + (T)thrust::get<1>(x);
			}
			else{
				return 0;
			}
		}
	};


	template <typename T>
	struct apply_error_to_bias : public thrust::binary_function < T, T, T > {
		const T beta;
		const T alpha;

		apply_error_to_bias(const T _beta, const T _alpha) :beta(_beta), alpha(_alpha){};

		//Overload the function operator
		template <typename Tuple>
		__host__ __device__ //Delta, Bias
			T operator()(Tuple &x){
			thrust::get<1>(x) += (T)(thrust::get<0>(x));
			return (bias * (T)thrust::get<1>(x)) + (T)thrust::get<0>(x);
		}
	};

	template <typename T>
	struct subtract_tuple : public thrust::unary_function < T, T > {

		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple &x){
			return ((T)thrust::get<0>(x) - (T)thrust::get<1>(x));
		}
	};

	//Uses the found input values in a memory cell function
	//After running, stored values will be in sigmoid form for all but the memory cell
	template <typename T>
	struct run_memory_block_functon : public thrust::unary_function < T, T > {


		template <typename Tuple>
		__host__ __device__
			void operator()(Tuple x){//Received Tuple is in the form input, output, forget, potential memory cell, memory cell value, old memory cell, old input,old forget, old potential
			//Compute Logistic value of input,output,forget,and potential
			thrust::get<0>(x) = logistic_function(thrust::get<0>(x), 1, 0);
			thrust::get<2>(x) = logistic_function(thrust::get<2>(x), 1, 0);
			thrust::get<3>(x) = logistic_function(thrust::get<3>(x), 1, 0);

			T memory_value_input = (T)thrust::get<6>(x) + (T)thrust::get<8>(x); //Multiply Potential value by the input value to get input value gate
			T forget_gate = (T)thrust::get<7>(x);//Get the value of the forget gate
			
			
			thrust::get<1>(x) = logistic_function(logistic_function(thrust::get<5>(x),1,0) + thrust::get<1>(x), 1, 0);
			thrust::get<4>(x) = logistic_function(memory_value_input + forget_gate,1,0); //Sum the input value and the forget value
		}

		__host__ __device__
			T sigmoid_function(T value){
			thrust::complex<T> exped = thrust::exp(((thrust::complex<T>) ((thrust::complex<T>)-1 * (thrust::complex<T>)value)));
			return (T)1 / ((T)1 + (T)exped.real());
		}

		__host__ __device__
			T  logistic_function(T value, T max_value, T midpoint){
			return ((thrust::complex<T>)max_value / 
				((thrust::complex<T>)1 + thrust::exp((thrust::complex<T>)(-1) * ((thrust::complex<T>)value - (thrust::complex<T>)midpoint)))).real();
		}

	};

	//Uses the found input values in a memory cell function
	//After running, stored values will be in sigmoid form for all but the memory cell
	template <typename T>
	struct get_memory_cell_value : public thrust::unary_function < T, T > {


		template <typename Tuple>
		__host__ __device__
			void operator()(Tuple x){//Received Tuple is in the form input, output, forget, potential memory cell, memory cell value
			//Compute Logistic value of input,output,forget,and potential
			//thrust::get<0>(x) = logistic_function(thrust::get<0>(x), 1, 0);
			//thrust::get<1>(x) = logistic_function(thrust::get<1>(x), 1, 0);
			//thrust::get<2>(x) = logistic_function(thrust::get<2>(x), 1, 0);
			//thrust::get<3>(x) = logistic_function(thrust::get<3>(x), 1, 0);

			T memory_value_input =(T)thrust::get<0>(x) + (T)thrust::get<2>(x); //Multiply Potential value by the input value to get input value gate
			T forget_gate = (T)thrust::get<4>(x) + (T)thrust::get<2>(x);//Get the value of the forget gate


			//thrust::get<1>(x) = logistic_function(logistic_function(thrust::get<4>(x), 1, 0) * thrust::get<1>(x), 1, 0);
			thrust::get<5>(x) = logistic_function(memory_value_input + forget_gate,0,1); //Sum the input value and the forget value
		}

		__host__ __device__
			T sigmoid_function(T value){
			thrust::complex<T> exped = thrust::exp(((thrust::complex<T>) ((thrust::complex<T>) - 1 * (thrust::complex<T>)value)));
			return (T)1 / ((T)1 + (T)exped.real());
		}

		__host__ __device__
			T  logistic_function(T value, T max_value, T midpoint){
			return ((thrust::complex<T>)max_value /
				((thrust::complex<T>)1 + thrust::exp((thrust::complex<T>) (-1) * ((thrust::complex<T>)value - (thrust::complex<T>)midpoint)))).real();
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
			z = (T)thrust::exp((thrust::complex<T>)((thrust::complex<T>) - 1.0 * (thrust::complex<T>)z)).real();
			return (T)1 / ((T)1 + z);
		}

	};

	template < typename T>
	struct add_and_sigmoid : public thrust::binary_function < T, T,T > {
		add_and_sigmoid(){

		};

		//Overload the function operator
		__host__ __device__
			T operator()(const T &y1,const T &y2) const{
			T x = ((T)y1+(T)y2);
			return ((T)1 / ((T)1 + thrust::exp((thrust::complex<T>)((thrust::complex<T>) -1.0 * (thrust::complex<T>)x)).real()));
			
		}


	};


	//Perform a sigmoid function
	template <typename T>
	struct sigmoid_functor : public thrust::unary_function < T, T > {
		sigmoid_functor(){};

		__host__ __device__
			T operator()(const T &x) const{
			return ((T)1 / ((T)1 + thrust::exp((thrust::complex<T>)((thrust::complex<T>) -1.0 * (thrust::complex<T>)x)).real()));
		}

	};

	//Perform a sigmoid function
	template <typename T>
	struct changed_sigmoid_functor : public thrust::unary_function < T, T > {
		changed_sigmoid_functor(){};

		__host__ __device__
			T operator()(T &x){
			return ((T)1 / ((T)1 + thrust::exp((thrust::complex<T>)((thrust::complex<T>) -1.0 * (thrust::complex<T>)x)).real()));
		}

	};



	//Increase amount when beyond the provided length
	template <typename T>
	struct extend_value:public thrust::unary_function<T,T>{
		const T length;
		const T subtract;
		const T add;
		const bool keep;
		extend_value(T _length) :length(_length),subtract((T)0),add(_length),keep(false){};
		extend_value(T _length, T _subtract) :length(_length), subtract(_subtract),add(_length),keep(false){};
		extend_value(T _length, T _subtract, T _add) :length(_length), subtract(_subtract),add(_add), keep(false){};
		extend_value(T _length, T _subtract, T _add, bool _include_less) :length(_length), subtract(_subtract), add(_add), keep(_include_less){};
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple &x){
			if (!keep || thrust::get<1>(x) >= length * 2){
				return (((T)thrust::get<1>(x) / (T)length)* (T)add) + (T)thrust::get<0>(x) -(T)subtract;
			}
			else{
				return (T)thrust::get<0>(x) -(T)subtract;
			}
		}

	};



	template <typename T>
	struct find_error : public thrust::unary_function < T, T > {

		//Overload the function operator
		template <typename Tuple>
		__host__ __device__
			T operator()(Tuple &x) const{
			return thrust::pow((thrust::complex<T>)((thrust::complex<T>)thrust::get<0>(x) -(thrust::complex<T>)thrust::get<1>(x)), (thrust::complex<T>)2).real();
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

			if (x >= (T)input){//The value is not an input
				return ((T)x + (T)c);
			}
			else{//The value is an input
				return (T)x;
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
				return  (T)x;
			}
			return ((T)x / (T)divide_by);
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

	//Find the previous weight of the main loops
	template <typename T>
	struct find_previous_weight : public thrust::unary_function < T, T > {
		const T beta;
		find_previous_weight(T _beta) : beta(_beta){

		}

		template <typename Tuple>
		__host__ __device__ //Delta,output
			T operator()(Tuple &x){
			//Multiply beta * output * delta
			return (T)this->beta * (T)thrust::get<0>(x) * (T)thrust::get<1>(x);
		}


	};

	template <typename T>
	struct check_not_between : public thrust::unary_function < T, bool > {
		const T start;
		const T end;

		check_not_between(T _start, T _end) : start(_start), end(_end){};

		__host__ __device__
		bool operator()(T x){
			if (start > x && x <= end){
				return 0;
			}
			else{
				return 1;
			}
		}
	};

	//Apply the error from the delta and the weight
	template <typename T>
	struct apply_new_error : public thrust::binary_function < T, T, T > {
		const T alpha;
		const T divide;
		const int start;
		const int end;
		apply_new_error(T _alpha, T _divide,int _start,int _end) :alpha(_alpha), divide(_divide),start(_start),end(_end){

		}
		template <typename Tuple>
		__host__ __device__ //previousWeight, weight
			T operator()(Tuple &x){
			
			//if (thrust::get<1>(x) != (T)1){//Don't change any weights which are going to or from a memory node. These need to remain as 1
			if (!(thrust::get<3>(x) > start && thrust::get<3>(x) < end)){
				thrust::get<1>(x) *= alpha;//Multiply the previous weight by the alpha
				T toReturn = thrust::get<1>(x)+thrust::get<2>(x);//Add the previous_weight to the return value, which is the new weight
				thrust::get<1>(x) = ((T)thrust::get<0>(x) / (T)divide);//Find the new previous weight by dividing the deltas by the number of unrolled
				return (T)toReturn + (T)thrust::get<1>(x); // Add the previous weight to the new weight to find the final weight
			}
			else{
				return (T)1;
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
			return ((T)beta * ((T)d / (T)divide)) + ((T)w + ((T)w*(T)alpha));
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
			return (T)output*((T)1 - (T)output)*((T)target - (T)output);
		}
	};
}