#pragma once
#include <ctime>
#include <istream>
#include <string>
class NetworkTimer
{
private:
	std::clock_t timer_start;
	std::clock_t current_time;
	int number_rounds;
	double checkpoint_time;
	double checkpoint_timer;
	std::istream::streampos file_size;
	std::istream::streampos size_of_round;
public:
	

	NetworkTimer();

	NetworkTimer(double seconds);

	void start();

	double restart_timer();

	double clear_timer();

	double get_current_time(){
		return this->current_time;
	}
	
	double get_timer_past(){
		return (double)std::clock() - this->timer_start;
	}

	bool passed_checkpoint(){
		if (std::clock() - this->checkpoint_timer >= this->checkpoint_time){
			this->checkpoint_timer = std::clock();//Reset the clock
			return true;
		}
		return false;
	}
	void set_size_of_round(std::istream::streampos size){
		this->size_of_round = size;
	}
	void set_file_size(std::istream::streampos file_size);

	//Get the estimated time remaining to run from the current most recent value
	std::string estimated_time_remaining(std::istream::streampos file_pos);



};

