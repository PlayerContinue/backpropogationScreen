#pragma once

/*
Author: David Greenberg

Contain several Iterators which are not contained in the main system

*/

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/function.h>
#include <thrust/functional.h>
#include <thrust/detail/raw_reference_cast.h>
namespace Special_Iterator{
	// derive repeat_iterator from iterator_adaptor
	template<typename Iterator>
	class repeat_iterator
		: public thrust::iterator_adaptor <
		repeat_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
		Iterator                   // the second template parameter is the name of the iterator we're adapting
		// we can use the default for the additional template parameters
		>
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor <
			repeat_iterator<Iterator>,
			Iterator
		> super_t;
		__host__ __device__
			repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// repeat each element of the adapted range n times
		unsigned int n;
		// used to keep track of where we began
		const Iterator begin;
		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			return *(begin + (this->base() - begin) / n);
		}
	};


	template<typename Iterator>
	inline __host__ __device__
		Special_Iterator::repeat_iterator<Iterator> make_repeat_iterator(Iterator it, int n){
		return Special_Iterator::repeat_iterator<Iterator>(it, n);
	};

	//Repeats several numbers in a list and then restarts
	template<typename Iterator>
	class repeat_list_iterator
		: public thrust::iterator_adaptor <
		repeat_list_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
		Iterator                   // the second template parameter is the name of the iterator we're adapting
		// we can use the default for the additional template parameters
		>
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor <
			repeat_list_iterator<Iterator>,
			Iterator
		> super_t;
		__host__ __device__
			repeat_list_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// repeat each element of the adapted range n times
		unsigned int n;
		// used to keep track of where we began
		const Iterator begin;
		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			return *(begin + ((this->base() - begin) % n));
		}
	};


	template<typename Iterator>
	inline __host__ __device__
		Special_Iterator::repeat_list_iterator<Iterator> make_repeat_list_iterator(Iterator it, int n){
		return Special_Iterator::repeat_list_iterator<Iterator>(it, n);
	};



	//Special Iterator for skipping a set number of places on each iteration
	template<typename Iterator>
	class skip_iterator : public thrust::iterator_adaptor < skip_iterator<Iterator>, Iterator>
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor <
			skip_iterator<Iterator>,
			Iterator
		> super_t;
		__host__ __device__
			skip_iterator(const Iterator &x, int n, int skip) : super_t(x), begin(x), n(n), skip(skip) {}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// skip skip elements every n elements
		const unsigned int n;
		const unsigned int skip;
		// used to keep track of where we began
		const Iterator begin;
		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			int pos = this->base() - begin;
			return *(begin + (pos + (((int)pos / n)*skip)));
		}
	};

	template<typename Iterator>
	Special_Iterator::skip_iterator<Iterator> make_skip_iterator(Iterator it, int n, int skip){
		return Special_Iterator::skip_iterator<Iterator>(it, n, skip);
	};

	//*************************************
	//return_zero_iterator
	//Special form iterator for dealing with iterators beyond the size of the array
	//n - the length of the array
	//x - the iterator
	//*************************************

	//Special Iterator for skipping a set number of places on each iteration
	template<typename Iterator, typename Iterator2>
	class return_zero_iterator : public thrust::iterator_adaptor < return_zero_iterator<Iterator, Iterator2>, Iterator >
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from

		typedef thrust::iterator_adaptor <
			return_zero_iterator<Iterator, Iterator2>,
			Iterator
		> super_t;

		__host__ __device__
			return_zero_iterator(const Iterator &begin, const Iterator &end, const Iterator2 &_return_type) : super_t(begin), begin(begin), end(end), return_value(_return_type) {}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:




		// used to keep track of where we began
		const Iterator begin;
		// used to keep track of where the value ends
		const Iterator end;
		//Value to return when beyond the scope
		const Iterator2 return_value;



		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			if (end - this->base() < 0){
				return *(return_value);
			}
			else{
				return *(this->base_reference());
			}
		}
	};

	template<typename Iterator, typename Iterator2>
	inline __host__ __device__
		Special_Iterator::return_zero_iterator<Iterator, Iterator2> make_return_zero_iterator(Iterator it, Iterator end, Iterator2 return_value){
		return Special_Iterator::return_zero_iterator<Iterator, Iterator2>(it, end, return_value);
	};

	//*************************************
	//return_value_iterator
	//return zero when value is between certain positions
	//n - the length of the array
	//x - the iterator
	//*************************************

	//Special Iterator for skipping a set number of places on each iteration
	template<typename Iterator>
	class return_iterator : public thrust::iterator_adaptor <  return_iterator<Iterator>, Iterator>
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor <
			return_iterator<Iterator>,
			Iterator
		> super_t;
		__host__ __device__
			return_iterator(const Iterator &x, const Iterator &_replace, int n, int skip, int _num_extra) : super_t(x), begin(x), replace(_replace), n(n), skip(skip), num_extra(_num_extra){}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// skip skip elements every n elements
		const unsigned int n;
		const unsigned int skip;
		const unsigned int num_extra;
		// used to keep track of where we began
		const Iterator begin;
		const Iterator replace;
		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			int pos = (this->base() - begin) - n;
			if (pos < 0 || (((pos + n) % skip)) < num_extra){
				return *(replace);
			}
			else{
				pos -= ((int)((pos + n) / skip))*num_extra;
				return *(begin + (pos));
			}

		}
	};

	template<typename Iterator>
	Special_Iterator::return_iterator<Iterator> make_return_iterator(Iterator it,
		Iterator begin, int skip_start, int count_to_skip, int extra_skip){
		return Special_Iterator::return_iterator<Iterator>(it, begin, skip_start, count_to_skip, extra_skip);
	};

	//*************************************
	//Make a transpose_iterator
	//*************************************

	//Special Iterator for skipping a set number of places on each iteration
	template<typename Iterator>
	class transpose_iterator : public thrust::iterator_adaptor <  transpose_iterator<Iterator>, Iterator>
	{
	public:
		// shorthand for the name of the iterator_adaptor we're deriving from
		typedef thrust::iterator_adaptor <
			transpose_iterator<Iterator>,
			Iterator
		> super_t;
		__host__ __device__
			transpose_iterator(const Iterator &x, const Iterator &y, int _row_length, int _col_length) : super_t(x), begin(x), end(y),row_length(_row_length),col_length(_col_length){}
		// befriend thrust::iterator_core_access to allow it access to the private interface below
		friend class thrust::iterator_core_access;
	private:
		// used to keep track of where we began
		const Iterator begin;
		const Iterator end;
		const int row_length;
		const int col_length;
		// it is private because only thrust::iterator_core_access needs access to it
		__host__ __device__
			typename super_t::reference dereference() const
		{
			int pos = this->base() - begin;//Number passed
			int row = pos / col_length;
			int col = pos%col_length;
			return *(begin + ((col*row_length) + row));


		}
	};

	template<typename Iterator>
	Special_Iterator::transpose_iterator<Iterator> make_transpose_iterator(Iterator begin, Iterator end, int row_length, int col_length){
		return Special_Iterator::transpose_iterator<Iterator>(begin, end, row_length, col_length);
	};

}

