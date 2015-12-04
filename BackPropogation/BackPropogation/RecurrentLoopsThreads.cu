#include "ReccurentLoops.cuh"

using namespace boost::interprocess;


inline boost::interprocess::managed_shared_memory open_shared_memory_object(string shared_memory_name){
	managed_shared_memory shared_memory(open_only, shared_memory_name.c_str());
	return shared_memory;
}

bool ReccurentLoops::start_training_threads(){
	this->initialize_threads();

	return true;
}


bool ReccurentLoops::stop_training_thread(){
	return true;
}


void timer_thread(NetworkTimer old_timer,string shared){
	managed_shared_memory shared_timer = open_shared_memory_object(shared);
	NetworkTimer timer = NetworkTimer(old_timer);
	std::pair<bool*, std::size_t> bool_values;
	timer.start();
	while (static_cast<bool>(shared_timer.find<bool>("TIMER_NEEDED").first)){
		if (timer.passed_checkpoint()){//A checkpoint should be created
			bool_values = shared_timer.find<bool>(CHECKPOINT_TIMER);
			std::memcpy(bool_values.first,false,bool_values.second);//Set the memory to true
			cout << true;
		}

		if (static_cast<bool>(shared_timer.find<bool>("PRINT_TIMER").first)){
			timer.restart_timer();
			//cout << timer.estimated_time_remaining(reinterpret_cast<std::istream::streampos>(file_remaining.get_address()));
			timer.clear_timer();
		}

	}

}


void pipe_thread(){

}


void ReccurentLoops::initialize_threads(){
	this->thread_list = std::vector<thread*>(FINAL_THREAD_POS);
	string shared_name = "timer";
	try{
		shared_memory_object::remove(shared_name.c_str());
	}
	catch(exception e){

	}
	try{
		this->timer_shared_memory = new managed_shared_memory(create_only, shared_name.c_str(), sizeof(std::istream::streampos) + (sizeof(bool) * 3));
	}
	catch (exception e){
	}
	timer_shared_memory->construct<bool>("TIMER_NEEDED")(true);
	timer_shared_memory->construct<bool>("PRINT_TIMER")(false);
	timer_shared_memory->construct<bool>(CHECKPOINT_TIMER)(false);
	this->thread_list[TIMER_THREAD] = new thread(timer_thread,this->timer,shared_name);
	
}

