
/*
Programmer: David Greenberg

Class function: Contain, analyze, and manipulate information about changing values of a network

T - The type of the data to be analyzed, must be a numeric value
  
*/

#include <string.h>
#include <vector>
#include <thread>
#include <fstream>

template <class T>

class DataPoints
{
	static_assert(std::is_arithmetic<T>::value, "T must be of type number");//Check the templated type is an arithmatic type
private:
	/*_______________Variables__________*/
	std::vector<std::vector<T>> dataPointContainer;//Contains the data points from the network
	std::vector<std::vector<T>> difference_sizes;//Size of change between the two points when they begin to jump
	int* limit_start = NULL;//The start point in the vector of the limit
	int* limit_end = NULL;//The end point in the vector of the limit 
	int number_containers;//The number of datalists which are being examined
	int current_position;//Position in the current data set
	T variance_limit;
	T* average = NULL;//The sum of the x
	T* variance = NULL;// The sum of the y
	T* sum_of_x_squared = NULL;//The sum of the x squared
	T* sum_of_xy_product = NULL;//The sum of the xy product
public:

	//Create an empty container of the object type
	DataPoints();
	//Create container with n number of available vectors
	DataPoints(int n);
	//Create container with n number of available vectors and a variance limit
	DataPoints(int n,T variance_limit);
	//~DataPoints();

	//---------------Initialize--------------------//

private:
	void initialize(int number_lists);

public:

	//---------------Manipulate List--------------------//
	//Remove all current information from the object
	bool clean_list();
	//Removes the previously found limit and leaves everything else the same
	bool remove_limit();
	//Remove the previous limit information and searches for new limit from limit_end
	bool reset_limit();
	//Add new information to the list
	bool add(std::vector<T>);

	//---------------Examine The Values--------------------//
	//Returns true when if a limit has been found
	bool is_limit_found();

private:
	void find_limits();//Find limits in the network if they exist
};
//*********************************************//
//---------------Definitions--------------------//
//*********************************************//

template <typename T>
DataPoints<T>::DataPoints(){};

template <typename T>
DataPoints<T>::DataPoints(int n) : DataPoints<T>::DataPoints(n, .2){}

template <typename T>
DataPoints<T>::DataPoints(int n, T variance_limit){
	this->variance_limit = variance_limit;
	initialize(n);
}

/*template <typename T>
DataPoints<T>::~DataPoints(){
	if (this->limit_start != NULL){
		delete[] this->limit_start;
		delete[] this->limit_end;
		delete[] average;//The sum of the x
		delete[] variance;// The sum of the y
		delete[] sum_of_x_squared;//The sum of the x squared
		delete[] sum_of_xy_product;//The sum of the xy product
	}
}*/

//---------------Initialize--------------------//
template <typename T>
void DataPoints<T>::initialize(int number_lists){
	this->dataPointContainer = std::vector<std::vector<T>>(number_lists);
	this->difference_sizes = std::vector<std::vector<T>>(number_lists);//Size of change between the two points when they begin to jump
	this->average = new T[number_lists];
	this->variance = new T[number_lists];
	this->sum_of_x_squared = new T[number_lists];
	this->sum_of_xy_product = new  T[number_lists];
	this->current_position = 0;
	this->number_containers = number_lists;
	for (int i = 0; i < number_lists; i++){
		this->difference_sizes[i] = vector<T>();
		this->dataPointContainer[i] = vector<T>();
		this->average[i] = 0;
		this->variance[i] = -1;
		this->sum_of_x_squared[i] = 0;
		this->sum_of_xy_product[i] = 0;
	}

	this->limit_start = new int[number_lists];
	this->limit_end = new int[number_lists];
	

}

//---------------Manipulate List--------------------//

template <typename T>
bool DataPoints<T>::clean_list(){
	for (int i = 0; i < this->number_containers; i++){
		this->difference_sizes[i].empty();
		this->dataPointContainer[i].empty();
		this->average[i] = 0;
		this->variance[i] = -1;
		this->sum_of_x_squared[i] = 0;
		this->sum_of_xy_product[i] = 0;
	}
	this->current_position = 0;
	return true;
}


template <typename T>
bool DataPoints<T>::remove_limit(){
	return true;
}

template <typename T>
bool DataPoints<T>::reset_limit(){
	for (int i = 0; i < this->number_containers; i++){
		this->average[i] = 0;
		this->variance[i] = -1;
		this->sum_of_x_squared[i] = 0;
		this->sum_of_xy_product[i] = 0;
	}
	return true;
}

template <typename T>
bool DataPoints<T>::add(std::vector<T> to_add){
	if (to_add.size() == this->number_containers){
		for (int i = 0; i < this->number_containers; i++){
			this->dataPointContainer[i].push_back(to_add[i]);
		}
		this->find_limits();
	}
	else{
		throw new std::exception("To Many/To Few values passed in");
	}
	return true;
}

//---------------Examine The Values--------------------//
template <typename T>
bool DataPoints<T>::is_limit_found(){
	for (int i = 0; i<this->number_containers; i++){
		if (this->current_position < 2 || this->variance[i] > this->variance_limit || this->variance[i]==-1){
			return false;
		}
	}
	return true;
}

template <typename T>
void DataPoints<T>::find_limits(){

	for (int i = 0; i < this->number_containers; i++){
		if (this->dataPointContainer[i].size() > this->current_position + 1){
			this->difference_sizes[i].push_back(this->dataPointContainer[i][this->current_position + 1] - this->dataPointContainer[i][this->current_position]);
			//Find the variance of the points
			this->average[i] += this->dataPointContainer[i][this->current_position + 1];
			this->variance[i] = std::pow(this->dataPointContainer[i][this->current_position + 1] - (this->average[i] / this->current_position + 1), 2);	
		}
	}
	if (this->dataPointContainer[0].size() > this->current_position + 1){
		this->current_position++;
	}
}

