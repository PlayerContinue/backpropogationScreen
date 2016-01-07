
/*
Programmer: David Greenberg

Class function: Contain, analyze, and manipulate information about changing values of a network

T - The type of the data to be analyzed, must be a numeric value

*/

#include <vector>
#define INTERNAL_TEST

#ifdef INTERNAL_TEST
#include <string>
#include <iostream>
#endif

#include <fstream>

template <class T>

class DataPoints
{
	static_assert(std::is_arithmetic<T>::value, "T must be of type number");//Check the templated type is an arithmatic type
private:
	/*_______________Variables__________*/
	std::vector<std::vector<T>> dataPointContainer;//Contains the data points from the network
	std::vector<std::vector<double>> difference_sizes;//Size of change between the two points when they begin to jump
	std::vector<std::vector<double>> x_points;//The points on the x-axis
	int* limit_start = NULL;//The start point in the vector of the limit
	int* limit_end = NULL;//The end point in the vector of the limit 
	int number_containers;//The number of datalists which are being examined
	int current_position;//Position in the current data set
	int slope_distance;//Distance between positions to find the slope
	int window_size; //Size of the window to find the standard deviation over, -1 if no ending 
	int* number_standard_deviation_y_above; // Number of times the current value is above the lowest standard deviation
	double variance_limit;
	double* average = NULL;//average
	double* variance = NULL;// variance
	double* variance_x = NULL; //The rolling variance of x
	double* variance_y = NULL; //The rolling variance of y
	double* standard_deviation_y = NULL;//The rolling SD of y
	double* sum_of_x = NULL;//The sum of the x
	double* sum_of_y = NULL;// The sum of the y
	double* sum_of_y_squared = NULL;//The sum of the squared ys
	double* sum_of_x_squared = NULL;//The sum of the x squared
	double* sum_of_xy_product = NULL;//The sum of the xy product
	double* x_average = NULL;//The average of the x values
	double* y_average = NULL; //The average of the y values
	double* linear_regression_slopes = NULL;//List of the linear regression slopes
	double* linear_regression_slopes_average = NULL;//List of the average slope for each line
	double* slope = NULL;//The slope

	double* lowest_standard_deviation_point_y = NULL;// The 

public:

	//Create an empty container of the object type
	DataPoints();
	//Create container with n number of available vectors
	DataPoints(int n);
	//Create container with n number of available vectors and a variance limit
	DataPoints(int n, T variance_limit);
	//Allows modification of the number searched in the variable
	DataPoints(int n, T variance_limit, int between_spots);

	//---------------Initialize--------------------//

private:
	void initialize(int number_lists, int slope_distance);

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
private:
	//Resets the internal values which are reset in multiple places
	//pos: position in the arrays to reset
	inline void reset_internal_values(int pos);



public:
	//---------------Examine The Values--------------------//
	//Returns true when if a limit has been found
	bool is_limit_found();

private:
	void find_limits();//Find limits in the network if they exist
	void find_slope();//Finds the slope of an approximation of a best fit line over the last x, where x is defined by the user
	inline void find_datapoints();//Finds all the datapoints required to find information about the network
	inline void find_datapoints(int pos);//Seperated for Cleanliness
	void find_linear_regression_slope();//Find the slope of the linear regression
	void find_rolling_variance(int pos, int window_size,
		double* sum, double* sum_squared, 
		double* variance, double* standard_deviation,
		double* average);//Find the standard deviation for x and y
	 

};




//*********************************************//
//---------------Definitions--------------------//
//*********************************************//

template <typename T>
DataPoints<T>::DataPoints(){};

template <typename T>
DataPoints<T>::DataPoints(int n) : DataPoints<T>::DataPoints(n, .2){}

template <typename T>
DataPoints<T>::DataPoints(int n, T variance_limit) : DataPoints<T>::DataPoints(n, variance_limit, 10){}

template <typename T>
DataPoints<T>::DataPoints(int n, T variance_limit, int slope_distance){
	this->variance_limit = variance_limit;
	this->slope_distance = slope_distance;
	this->window_size = slope_distance;
	initialize(n, slope_distance);
}


//---------------Initialize--------------------//
template <typename T>
void DataPoints<T>::initialize(int number_lists, int slope_distance){
	this->dataPointContainer = std::vector<std::vector<T>>(number_lists);
	this->difference_sizes = std::vector<std::vector<double>>(number_lists);//Size of change between the two points when they begin to jump
	this->x_points = std::vector<std::vector<double>>(number_lists);
	this->average = new double[number_lists];
	this->variance = new double[number_lists];
	this->variance_x = new double[number_lists];
	this->variance_y = new double[number_lists];
	this->standard_deviation_y = new double[number_lists];
	this->lowest_standard_deviation_point_y = new double[number_lists];
	this->sum_of_x = new double[number_lists];
	this->sum_of_y = new double[number_lists];
	this->sum_of_y_squared = new double[number_lists];
	this->sum_of_x_squared = new double[number_lists];
	this->sum_of_xy_product = new double[number_lists];
	this->x_average = new double[number_lists];
	this->y_average = new double[number_lists];
	this->linear_regression_slopes = new double[number_lists];
	this->linear_regression_slopes_average = new double[number_lists];
	this->slope = new double[number_lists];
	this->limit_start = new int[number_lists];
	this->limit_end = new int[number_lists];
	this->current_position = 0;
	this->number_standard_deviation_y_above = new int[number_lists];
	this->number_containers = number_lists;

	for (int i = 0; i < number_lists; i++){
		this->difference_sizes[i] = vector<double>();
		this->dataPointContainer[i] = vector<T>();
		this->x_points[i] = vector<double>();
		reset_internal_values(i);
	}




}

//---------------Manipulate List--------------------//

template <typename T>
bool DataPoints<T>::clean_list(){
	for (int i = 0; i < this->number_containers; i++){
		this->difference_sizes[i].clear();
		this->difference_sizes[i].resize(0);
		this->dataPointContainer[i].clear();
		this->dataPointContainer[i].resize(0);
		this->x_points[i].clear();
		this->x_points[i].resize(0);
		reset_internal_values(i);
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
		reset_internal_values(i);
	}

	return true;
}

template <typename T>
bool DataPoints<T>::add(std::vector<T> to_add){
	if (to_add.size() == this->number_containers){
		for (int i = 0; i < this->number_containers; i++){
			this->dataPointContainer[i].push_back(to_add[i]);
		}
		this->find_datapoints();
	}
	else{
		throw new std::exception("To Many/To Few values passed in");
	}
	return true;
}

template <typename T>
inline void DataPoints<T>::reset_internal_values(int pos){
	this->average[pos] = 0;
	this->variance[pos] = -1;
	this->variance_x[pos] = 0;
	this->variance_y[pos] = 0;
	this->y_average[pos] = 0;
	this->x_average[pos] = 0;
	this->standard_deviation_y[pos] = 0;
	this->lowest_standard_deviation_point_y[pos] = -1;
	this->sum_of_x[pos] = 0;
	this->sum_of_y[pos] = 0;
	this->sum_of_y_squared[pos] = 0;
	this->sum_of_x_squared[pos] = 0;
	this->sum_of_xy_product[pos] = 0;
	this->linear_regression_slopes[pos] = 0;
	this->linear_regression_slopes_average[pos] = 0;
	this->slope[pos] = 0;
	this->limit_start[pos] = 0;
	this->limit_end[pos] = slope_distance;
	this->number_standard_deviation_y_above[pos] = 1;
}

//---------------Examine The Values--------------------//
template <typename T>
bool DataPoints<T>::is_limit_found(){
	if (this->current_position >= this->slope_distance){
		int number_times_above_zero = 0;
		bool failure = false;
		for (int i = 0; i < this->number_containers; i++){
			/*if (
				this->current_position < this->window_size ||
				std::abs(this->linear_regression_slopes_average[i] / (this->current_position - 1)) > this->variance_limit){
				failure = false;
			}*/

			if (this->number_standard_deviation_y_above[i] > (this->window_size/2)){//If the slope is positive, then the value is going up, which is bad
				number_times_above_zero++;
			}

		}

		if ((number_times_above_zero > (this->number_containers / 2) && this->number_standard_deviation_y_above[0] >= (this->window_size / 2))){
			return true;//At least half of the slopes are positive or all of them are going down at a rate small enough not to matter
		}
		else{
			return false;
		}
	}
	else{
		return false;
	}

	
}

template <typename T>
void DataPoints<T>::find_limits(){
	for (int i = 0; i < this->number_containers; i++){
		if (this->dataPointContainer[i].size() > this->current_position + 1){
			//Find the variance of the points
			this->average[i] += this->dataPointContainer[i][this->current_position + 1];
			this->variance[i] = std::pow(this->dataPointContainer[i][this->current_position + 1] - (this->average[i] / (this->current_position + 1)), 2);
		}
	}
	if (this->dataPointContainer[0].size() > this->current_position + 1){
		this->current_position++;
	}
}

template <typename T>
void DataPoints<T>::find_slope(){


	for (int i = 0; i < this->number_containers; i++){
		if (this->limit_end[i] > this->limit_start[i] && limit_end[i] <= this->difference_sizes[i].size()){
			this->slope[i] = (this->difference_sizes[i][this->limit_end[i] - 1] - this->difference_sizes[i][this->limit_start[i]]) / (this->limit_end[i] - this->limit_start[i] - 1);
			this->limit_end[i]++;
			this->limit_start[i]++;
		}
	}

}

#ifdef INTERNAL_TEST
namespace temp{
	template <typename T>
	void outputArrayToFile(T* out, int size, std::string file_name){
		static int opened_once = 0;
		opened_once++;
		std::ofstream outputfile;
		outputfile.precision(30);
		if (opened_once < 3){
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
}

#endif


template <typename T>
inline void DataPoints<T>::find_datapoints(){
	double temp_point;
	

	if (this->dataPointContainer[0].size() > this->current_position + 1){
		for (int i = 0; i < this->number_containers; i++){
			temp_point = this->dataPointContainer[i][this->current_position]; //(this->dataPointContainer[i][this->current_position + 1] - this->dataPointContainer[i][this->current_position]); 
			this->x_points[i].push_back((this->current_position%this->window_size)+1);

			this->difference_sizes[i].push_back(temp_point);
			
			this->sum_of_x[i] += this->x_points[i].back();
			this->sum_of_y[i] += temp_point;
			this->sum_of_y_squared[i] += std::pow(temp_point, 2);
			this->sum_of_xy_product[i] += (temp_point* this->x_points[i].back());
			this->sum_of_x_squared[i] += std::pow(this->x_points[i].back(), 2);
			this->find_datapoints(i);//Seperated for easier reading
		}

		//if (this->window_size > 0 && this->current_position+1 > this->window_size){
			this->find_linear_regression_slope();
		//}
		
		//this->find_slope();
		this->current_position++;

#ifdef INTERNAL_TEST
		temp::outputArrayToFile<double>(this->standard_deviation_y, this->number_containers, "tests/standard_deviation.txt");
#endif

	}
}


template <typename T>
inline void DataPoints<T>::find_datapoints(int pos){
	
	if (this->window_size > 0 && this->difference_sizes[pos].size() > this->window_size){
		double temp_window = this->difference_sizes[pos][0];

		//Subtract the old values from the sum of the values to find the rolling window value
		this->sum_of_y[pos] -= temp_window;
		this->sum_of_y_squared[pos] -= std::pow(temp_window, 2);
		this->sum_of_x[pos] -= (this->x_points[pos][0]);//Remove the sum of the y value
		this->sum_of_xy_product[pos] -= (temp_window * this->x_points[pos][0]);
		this->sum_of_x_squared[pos] -= std::pow(this->x_points[pos][0], 2);
		
		//Remove the first one in the list
		this->difference_sizes[pos].erase(this->difference_sizes[pos].begin());
		this->x_points[pos].erase(this->x_points[pos].begin());
	}

	find_rolling_variance(pos, this->window_size,
		this->sum_of_y, this->sum_of_y_squared,
		this->variance_y, this->standard_deviation_y, this->y_average);//Find the variance of y
	if (this->current_position >= this->window_size){
		if (this->lowest_standard_deviation_point_y[pos] == -1 || this->standard_deviation_y[pos] < this->lowest_standard_deviation_point_y[pos]){
			this->lowest_standard_deviation_point_y[pos] = this->standard_deviation_y[pos];
			if (this->number_standard_deviation_y_above[pos] > 0){
				this->number_standard_deviation_y_above[pos] -= 1;
			}
		}
		else{
			this->number_standard_deviation_y_above[pos] += 1;
		}
	}
		
}


template <typename T>
void DataPoints<T>::find_rolling_variance(int pos, int window_size, double* sum, double* sum_squared, double* variance, double* standard_deviation, double* average){
	int length = (this->current_position <= window_size) ? (this->current_position+1) : this->window_size;

	average[pos] = sum[pos] / length;//Find the average
	//Find the variance
	variance[pos] = sum_squared[pos] - (2*(average[pos] * sum[pos])) + (length*std::pow(average[pos], 2));
	variance[pos] /= length;
	//Find the standard deviation
	standard_deviation[pos] = std::sqrt(variance[pos]);

	

}





template <typename T>
void DataPoints<T>::find_linear_regression_slope(){
	int difference_size;
	for (int i = 0; i < this->number_containers; i++){
		difference_size = (this->window_size <= this->current_position) ? this->window_size : this->difference_sizes[i].size();
		this->linear_regression_slopes[i] = ((difference_size) * this->sum_of_xy_product[i]) - (this->sum_of_x[i] * this->sum_of_y[i]);
		this->linear_regression_slopes[i] /= ((difference_size * this->sum_of_x_squared[i]) - std::pow(this->sum_of_x[i], 2));
		if (this->current_position > 1){
			this->linear_regression_slopes_average[i] += this->linear_regression_slopes[i];
		}
	}
#ifdef INTERNAL_TEST
	if (this->current_position > 1){
		for (int i = 0; i < this->number_containers; i++){
			this->linear_regression_slopes_average[i] /= this->current_position;
		}
		temp::outputArrayToFile<double>(this->linear_regression_slopes, this->number_containers, "tests/slopes.txt");
		temp::outputArrayToFile<double>(this->linear_regression_slopes_average, this->number_containers, "tests/slopes_average.txt");
		for (int i = 0; i < this->number_containers; i++){
			this->linear_regression_slopes_average[i] *= this->current_position;
		}
	}
	else{
		temp::outputArrayToFile<double>(this->linear_regression_slopes, this->number_containers, "tests/slopes.txt");
		temp::outputArrayToFile<double>(this->linear_regression_slopes_average, this->number_containers, "tests/slopes_average.txt");
	}
#endif
}

