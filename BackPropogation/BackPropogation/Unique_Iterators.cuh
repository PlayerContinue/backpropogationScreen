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
namespace Unique_Iterator{
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
		Unique_Iterator::repeat_iterator<Iterator> make_repeat_iterator(Iterator it, int n){
		return Unique_Iterator::repeat_iterator<Iterator>(it, n);
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
		Unique_Iterator::repeat_list_iterator<Iterator> make_repeat_list_iterator(Iterator it, int n){
		return Unique_Iterator::repeat_list_iterator<Iterator>(it, n);
	};


	//Special Iterator for skipping a set number of places on each iteration
	/*template<typename Iterator>
	class skip_iterator : public thrust::iterator_adaptor < skip_iterator<Iterator>, Iterator>
	{
	public:
	// shorthand for the name of the iterator_adaptor we're deriving from
	typedef thrust::iterator_adaptor <
	skip_iterator<Iterator>,
	Iterator,
	> super_t;
	__host__ __device__
	skip_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
	// befriend thrust::iterator_core_access to allow it access to the private interface below
	friend class thrust::iterator_core_access;
	private:
	// skip n elements
	unsigned int n;
	// used to keep track of where we began
	const Iterator begin;
	// it is private because only thrust::iterator_core_access needs access to it
	__host__ __device__
	typename super_t::reference dereference() const
	{

	return *(begin + ((this->base() - begin)*n));
	}
	};

	template<typename Iterator>
	Unique_Iterator::skip_iterator<Iterator> make_skip_iterator(Iterator it, int n){
	return Unique_Iterator::skip_iterator<Iterator>(it, n);
	};*/

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
		Unique_Iterator::return_zero_iterator<Iterator, Iterator2> make_return_zero_iterator(Iterator it, Iterator end, Iterator2 return_value){
		return Unique_Iterator::return_zero_iterator<Iterator, Iterator2>(it, end, return_value);
	};


}