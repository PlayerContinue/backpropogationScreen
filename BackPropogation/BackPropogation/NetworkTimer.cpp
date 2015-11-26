#include "NetworkTimer.h"


NetworkTimer::NetworkTimer()
{
	this->timer_start = std::clock();
	this->current_time = 0;
}

void NetworkTimer::start(){
	this->timer_start = std::clock();
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
	this->timer_start = 0;
	this->current_time = 0;
	this->number_rounds = 0;
	return current;
}
