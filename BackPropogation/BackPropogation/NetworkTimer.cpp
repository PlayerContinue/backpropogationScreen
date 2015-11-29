#include "NetworkTimer.h"


NetworkTimer::NetworkTimer()
{
	this->timer_start = std::clock();
	this->current_time = 0;
	this->checkpoint_time = CLOCKS_PER_SEC * 18000;
}

NetworkTimer::NetworkTimer(double seconds){
	this->timer_start = std::clock();
	this->current_time = 0;
	this->checkpoint_time = CLOCKS_PER_SEC * seconds;
	this->checkpoint_timer = std::clock();
}



void NetworkTimer::start(){
	this->timer_start = std::clock();
}

void NetworkTimer::set_file_size(std::istream::streampos file_size){
	this->file_size = file_size;
}

std::string NetworkTimer::estimated_time_remaining(std::istream::streampos file_pos){
	std::istream::streamoff file_diff =(long double) ((this->file_size - file_pos) / this->size_of_round);//Amount of file remaining
	long double estimated_milliseconds = (this->current_time/CLOCKS_PER_SEC);//Time passed in one round
	estimated_milliseconds *= file_diff; //Multiply amount of file remaining by the time remaining
	long estimated_days = estimated_milliseconds / 3600 / 24;
	estimated_milliseconds -= estimated_days * 3600 * 24;
	long estimated_hours = (estimated_milliseconds / 3600);
	estimated_milliseconds -= (estimated_hours * 3600);
	long estimated_minutes = estimated_milliseconds / 60;
	estimated_milliseconds -= estimated_minutes * 60;
	return (std::to_string(estimated_days) + " days " +
		std::to_string(estimated_hours) + " hours " +
		std::to_string(estimated_minutes) + " minutes " +
		std::to_string(estimated_milliseconds) + " seconds ");
}

double NetworkTimer::restart_timer(){
	double end = std::clock();
	this->current_time += end - this->timer_start;
	this->number_rounds++;
	this->timer_start = end;
	return this->current_time/(double)this->number_rounds;
}

double NetworkTimer::clear_timer(){
	double current = this->restart_timer();
	this->timer_start = std::clock();
	this->current_time = 0;
	this->number_rounds = 0;
	return current;
}
