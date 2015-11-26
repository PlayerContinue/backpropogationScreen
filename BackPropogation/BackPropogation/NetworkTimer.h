#pragma once
#include <ctime>
class NetworkTimer
{
private:
	std::clock_t timer_start;
	double current_time;
	int number_rounds;
public:
	

	NetworkTimer();

	void start();

	double restart_timer();

	double clear_timer();

	double get_current_time(){
		return this->current_time;
	}
	
	double get_timer_past(){
		return (double)std::clock() - this->timer_start;
	}



};

